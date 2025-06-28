"""
Train and evaluate the enhanced deepfake detection model
"""

import os
import argparse
import logging
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, confusion_matrix
)

# Import custom modules
from enhanced_detection import EnhancedDeepfakeDetector
from model_trainer import DeepfakeModelTrainer
from advanced_augmentation import DeepfakeAugmenter
from backend.config import MODEL_DIR, DATASET_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Train and evaluate enhanced deepfake detection model")
    
    # General options
    parser.add_argument("--train-dir", type=str, required=True,
                       help="Directory containing training data")
    parser.add_argument("--val-dir", type=str, required=True,
                       help="Directory containing validation data")
    parser.add_argument("--test-dir", type=str, default=None,
                       help="Directory containing test data (if None, uses validation data)")
    parser.add_argument("--output-dir", type=str, default=os.path.join(MODEL_DIR, "enhanced"),
                       help="Directory to save models and results")
    
    # Training options
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of epochs to train (default: 20)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate for training (default: 1e-4)")
    parser.add_argument("--no-augmentation", action="store_true",
                       help="Disable advanced augmentation")
    
    # Model options
    parser.add_argument("--no-ensemble", action="store_true",
                       help="Disable ensemble learning")
    parser.add_argument("--no-temporal", action="store_true",
                       help="Disable temporal analysis")
    parser.add_argument("--no-cross-modal", action="store_true",
                       help="Disable cross-modal verification")
    parser.add_argument("--models", type=str, nargs="+", 
                       default=["efficientnet", "xception", "vit"],
                       help="Models to include in the ensemble")
    
    # Evaluation options
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training and only evaluate")
    parser.add_argument("--eval-videos", action="store_true",
                       help="Evaluate on videos instead of images")
    
    return parser.parse_args()

def train_model(args):
    """Train the enhanced deepfake detection model"""
    logger.info("Initializing enhanced deepfake detector for training")
    
    # Initialize detector
    detector = EnhancedDeepfakeDetector(
        use_ensemble=not args.no_ensemble,
        use_temporal=not args.no_temporal,
        use_cross_modal=not args.no_cross_modal,
        use_advanced_augmentation=not args.no_augmentation,
        model_names=args.models
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Train model
    logger.info(f"Training model for {args.epochs} epochs with batch size {args.batch_size}")
    training_results = detector.train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_augmentation=not args.no_augmentation,
        save_dir=args.output_dir
    )
    
    # Save training results
    with open(os.path.join(args.output_dir, "training_results.json"), "w") as f:
        # Convert numpy values to Python types
        results_dict = {}
        for k, v in training_results.items():
            if isinstance(v, dict):
                results_dict[k] = {k2: float(v2) if isinstance(v2, np.number) else v2 
                                  for k2, v2 in v.items()}
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.number):
                results_dict[k] = [float(x) for x in v]
            elif isinstance(v, np.number):
                results_dict[k] = float(v)
            else:
                results_dict[k] = v
        
        json.dump(results_dict, f, indent=2)
    
    # Plot training history
    if "history" in training_results:
        history = training_results["history"]
        
        # Plot accuracy
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.get("accuracy", []))
        plt.plot(history.get("val_accuracy", []))
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="lower right")
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.get("loss", []))
        plt.plot(history.get("val_loss", []))
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper right")
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "training_history.png"))
    
    logger.info(f"Training completed. Results saved to {args.output_dir}")
    
    return detector

def evaluate_model(detector, args):
    """Evaluate the enhanced deepfake detection model"""
    logger.info("Evaluating enhanced deepfake detector")
    
    # Use test directory if provided, otherwise use validation directory
    test_dir = args.test_dir or args.val_dir
    
    # Get list of test files
    test_files = []
    labels = []
    
    # Real images/videos
    real_dir = os.path.join(test_dir, "real")
    if os.path.exists(real_dir):
        for file in os.listdir(real_dir):
            if args.eval_videos:
                if file.endswith((".mp4", ".avi", ".mov")):
                    test_files.append(os.path.join(real_dir, file))
                    labels.append(0)  # Real
            else:
                if file.endswith((".jpg", ".jpeg", ".png")):
                    test_files.append(os.path.join(real_dir, file))
                    labels.append(0)  # Real
    
    # Fake images/videos
    fake_dir = os.path.join(test_dir, "fake")
    if os.path.exists(fake_dir):
        for file in os.listdir(fake_dir):
            if args.eval_videos:
                if file.endswith((".mp4", ".avi", ".mov")):
                    test_files.append(os.path.join(fake_dir, file))
                    labels.append(1)  # Fake
            else:
                if file.endswith((".jpg", ".jpeg", ".png")):
                    test_files.append(os.path.join(fake_dir, file))
                    labels.append(1)  # Fake
    
    if not test_files:
        logger.error(f"No test files found in {test_dir}")
        return
    
    logger.info(f"Found {len(test_files)} test files ({sum(labels)} fake, {len(labels) - sum(labels)} real)")
    
    # Evaluate each file
    predictions = []
    probabilities = []
    confidences = []
    processing_times = []
    
    for i, file in enumerate(test_files):
        logger.info(f"Processing file {i+1}/{len(test_files)}: {file}")
        
        try:
            # Detect deepfake
            if args.eval_videos:
                result = detector.detect_video(file)
            else:
                result = detector.detect_image(file)
            
            # Extract results
            predictions.append(1 if result.get("is_deepfake", False) else 0)
            probabilities.append(result.get("probability", 0.5))
            confidences.append(result.get("confidence", 0.5))
            processing_times.append(result.get("processing_time", 0.0))
            
        except Exception as e:
            logger.error(f"Error processing file {file}: {str(e)}")
            # Add default values for failed files
            predictions.append(0)
            probabilities.append(0.5)
            confidences.append(0.5)
            processing_times.append(0.0)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    # Calculate precision-recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(labels, probabilities)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Calculate average processing time
    avg_processing_time = np.mean(processing_times)
    
    # Prepare evaluation results
    eval_results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
        "avg_processing_time": float(avg_processing_time),
        "test_file_count": len(test_files),
        "real_count": len(labels) - sum(labels),
        "fake_count": sum(labels)
    }
    
    # Save evaluation results
    eval_output_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(eval_output_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    
    # Plot ROC curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Plot precision-recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "evaluation_curves.png"))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['Real', 'Fake'])
    plt.yticks(tick_marks, ['Real', 'Fake'])
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
    
    # Print results
    logger.info(f"Evaluation completed. Results saved to {eval_output_path}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"PR AUC: {pr_auc:.4f}")
    logger.info(f"Average Processing Time: {avg_processing_time:.4f} seconds")
    
    return eval_results

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    if not args.skip_training:
        detector = train_model(args)
    else:
        logger.info("Skipping training, initializing detector for evaluation")
        detector = EnhancedDeepfakeDetector(
            use_ensemble=not args.no_ensemble,
            use_temporal=not args.no_temporal,
            use_cross_modal=not args.no_cross_modal,
            use_advanced_augmentation=not args.no_augmentation,
            model_names=args.models
        )
    
    # Evaluate model
    eval_results = evaluate_model(detector, args)
    
    # Check if we achieved >95% accuracy
    if eval_results and eval_results["accuracy"] > 0.95:
        logger.info("🎉 Success! Achieved >95% accuracy in deepfake detection.")
    elif eval_results:
        logger.info(f"Current accuracy: {eval_results['accuracy']:.4f}. "
                   f"Consider further tuning to reach >95% accuracy.")

if __name__ == "__main__":
    main()