"""
Indian Face Testing Script for DeepDefend
This script evaluates the current system on Indian faces to identify areas for improvement
"""

import os
import sys
import logging
import argparse
import json
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from face_detector import FaceDetector
from indian_face_utils import IndianFacePreprocessor
from skin_tone_analyzer import SkinToneAnalyzer
from inference import predict_image, load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IndianFaceTester:
    """
    Tests deepfake detection performance on Indian faces
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the tester
        
        Args:
            model_path: Path to the model to test (if None, uses default model)
        """
        self.model_path = model_path
        
        # Initialize face detector
        self.face_detector = FaceDetector(confidence_threshold=0.5, detector_type="mtcnn")
        
        # Initialize face preprocessor
        self.face_preprocessor = IndianFacePreprocessor(self.face_detector)
        
        # Initialize skin tone analyzer
        self.skin_tone_analyzer = SkinToneAnalyzer()
        
        # Load model
        self.model = load_model(model_path)
        
        logger.info(f"Initialized Indian face tester with model: {model_path or 'default'}")
    
    def test_dataset(self, dataset_dir):
        """
        Test the model on a dataset
        
        Args:
            dataset_dir: Directory containing test data with 'real' and 'fake' subdirectories
            
        Returns:
            Dictionary with test results
        """
        # Validate dataset directory
        real_dir = os.path.join(dataset_dir, "real")
        fake_dir = os.path.join(dataset_dir, "fake")
        
        if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
            raise ValueError(f"Dataset directory must contain 'real' and 'fake' subdirectories: {dataset_dir}")
        
        # Get all image files
        real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        logger.info(f"Testing on {len(real_files)} real and {len(fake_files)} fake images")
        
        # Initialize results
        y_true = []
        y_pred = []
        y_scores = []
        
        # Track performance by skin tone
        skin_tone_results = {}
        
        # Process real images
        for image_file in tqdm(real_files, desc="Testing real images"):
            try:
                # Read image
                image = cv2.imread(image_file)
                if image is None:
                    logger.warning(f"Failed to read image: {image_file}")
                    continue
                
                # Get prediction
                result = predict_image(image, self.model, self.face_detector, self.face_preprocessor)
                
                if result and "probability" in result:
                    y_true.append(0)  # 0 = real
                    y_pred.append(1 if result["probability"] > 0.5 else 0)
                    y_scores.append(result["probability"])
                    
                    # Track performance by skin tone
                    if "regions" in result and result["regions"]:
                        for region in result["regions"]:
                            if "skin_tone" in region and region["skin_tone"].get("success", False):
                                if "indian_tone" in region["skin_tone"] and region["skin_tone"]["indian_tone"]:
                                    tone_type = region["skin_tone"]["indian_tone"].get("type", "unknown")
                                    
                                    if tone_type not in skin_tone_results:
                                        skin_tone_results[tone_type] = {
                                            "real": {"correct": 0, "total": 0},
                                            "fake": {"correct": 0, "total": 0}
                                        }
                                    
                                    skin_tone_results[tone_type]["real"]["total"] += 1
                                    if result["probability"] <= 0.5:  # Correctly classified as real
                                        skin_tone_results[tone_type]["real"]["correct"] += 1
                
            except Exception as e:
                logger.error(f"Error processing {image_file}: {str(e)}")
        
        # Process fake images
        for image_file in tqdm(fake_files, desc="Testing fake images"):
            try:
                # Read image
                image = cv2.imread(image_file)
                if image is None:
                    logger.warning(f"Failed to read image: {image_file}")
                    continue
                
                # Get prediction
                result = predict_image(image, self.model, self.face_detector, self.face_preprocessor)
                
                if result and "probability" in result:
                    y_true.append(1)  # 1 = fake
                    y_pred.append(1 if result["probability"] > 0.5 else 0)
                    y_scores.append(result["probability"])
                    
                    # Track performance by skin tone
                    if "regions" in result and result["regions"]:
                        for region in result["regions"]:
                            if "skin_tone" in region and region["skin_tone"].get("success", False):
                                if "indian_tone" in region["skin_tone"] and region["skin_tone"]["indian_tone"]:
                                    tone_type = region["skin_tone"]["indian_tone"].get("type", "unknown")
                                    
                                    if tone_type not in skin_tone_results:
                                        skin_tone_results[tone_type] = {
                                            "real": {"correct": 0, "total": 0},
                                            "fake": {"correct": 0, "total": 0}
                                        }
                                    
                                    skin_tone_results[tone_type]["fake"]["total"] += 1
                                    if result["probability"] > 0.5:  # Correctly classified as fake
                                        skin_tone_results[tone_type]["fake"]["correct"] += 1
                
            except Exception as e:
                logger.error(f"Error processing {image_file}: {str(e)}")
        
        # Calculate metrics
        if len(y_true) == 0:
            logger.error("No valid predictions were made")
            return {
                "success": False,
                "error": "No valid predictions were made"
            }
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Calculate classification report
        report = classification_report(y_true, y_pred, target_names=["real", "fake"], output_dict=True)
        
        # Calculate skin tone performance
        for tone_type in skin_tone_results:
            real_data = skin_tone_results[tone_type]["real"]
            fake_data = skin_tone_results[tone_type]["fake"]
            
            real_accuracy = real_data["correct"] / real_data["total"] if real_data["total"] > 0 else 0
            fake_accuracy = fake_data["correct"] / fake_data["total"] if fake_data["total"] > 0 else 0
            
            skin_tone_results[tone_type]["real"]["accuracy"] = real_accuracy
            skin_tone_results[tone_type]["fake"]["accuracy"] = fake_accuracy
            
            total_correct = real_data["correct"] + fake_data["correct"]
            total_samples = real_data["total"] + fake_data["total"]
            
            skin_tone_results[tone_type]["overall_accuracy"] = total_correct / total_samples if total_samples > 0 else 0
        
        # Prepare results
        results = {
            "success": True,
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "roc_auc": float(roc_auc)
            },
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "skin_tone_performance": skin_tone_results,
            "test_size": len(y_true),
            "real_count": y_true.count(0),
            "fake_count": y_true.count(1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"Test results: accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, auc={roc_auc:.4f}")
        
        return results
    
    def plot_results(self, results, output_dir):
        """
        Plot test results
        
        Args:
            results: Test results dictionary
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = np.array(results["confusion_matrix"])
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        classes = ["Real", "Fake"]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
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
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()
        
        # Plot skin tone performance
        skin_tone_data = results["skin_tone_performance"]
        if skin_tone_data:
            plt.figure(figsize=(12, 8))
            
            tones = list(skin_tone_data.keys())
            real_acc = [skin_tone_data[tone]["real"]["accuracy"] for tone in tones]
            fake_acc = [skin_tone_data[tone]["fake"]["accuracy"] for tone in tones]
            overall_acc = [skin_tone_data[tone]["overall_accuracy"] for tone in tones]
            
            x = np.arange(len(tones))
            width = 0.25
            
            plt.bar(x - width, real_acc, width, label='Real Accuracy')
            plt.bar(x, fake_acc, width, label='Fake Accuracy')
            plt.bar(x + width, overall_acc, width, label='Overall Accuracy')
            
            plt.xlabel('Skin Tone')
            plt.ylabel('Accuracy')
            plt.title('Performance by Skin Tone')
            plt.xticks(x, tones)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "skin_tone_performance.png"))
            plt.close()
        
        # Plot ROC curve
        if "metrics" in results and "roc_auc" in results["metrics"]:
            # We need to recalculate the ROC curve
            # This is just a placeholder - in a real implementation, you would save the FPR and TPR values
            plt.figure(figsize=(8, 6))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve (AUC = {results["metrics"]["roc_auc"]:.4f})')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "roc_curve.png"))
            plt.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test deepfake detection on Indian faces")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory containing test data")
    parser.add_argument("--model-path", type=str, help="Path to the model to test")
    parser.add_argument("--output-dir", type=str, default="test_results", help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tester
    tester = IndianFaceTester(model_path=args.model_path)
    
    # Test dataset
    results = tester.test_dataset(args.dataset_dir)
    
    # Save results
    results_path = os.path.join(args.output_dir, "test_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved test results to {results_path}")
    
    # Plot results
    if results["success"]:
        tester.plot_results(results, args.output_dir)
        logger.info(f"Saved result plots to {args.output_dir}")

if __name__ == "__main__":
    main()