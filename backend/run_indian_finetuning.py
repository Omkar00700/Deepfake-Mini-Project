"""
Indian Face Fine-tuning Runner for DeepDefend
This script runs the fine-tuning process on Indian faces
"""

import os
import sys
import logging
import argparse
import json
import time
from finetune_indian_faces import DatasetPreparation, ModelFinetuner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_finetuning(dataset_dir, model_type="efficientnet", epochs=10, fine_tune_epochs=5, 
                  batch_size=32, model_name="indian_deepfake_detector"):
    """
    Run the fine-tuning process
    
    Args:
        dataset_dir: Directory containing the dataset
        model_type: Type of model to fine-tune
        epochs: Number of epochs for initial training
        fine_tune_epochs: Number of epochs for fine-tuning
        batch_size: Batch size for training
        model_name: Name for the saved model
        
    Returns:
        Dictionary with fine-tuning results
    """
    try:
        # Initialize components
        dataset_prep = DatasetPreparation(dataset_dir=dataset_dir)
        model_finetuner = ModelFinetuner(model_dir="fine_tuned_models")
        
        # Create data generators
        logger.info("Creating data generators...")
        train_generator, val_generator, test_generator = dataset_prep.create_data_generators(
            batch_size=batch_size
        )
        
        # Create model
        logger.info(f"Creating {model_type} model...")
        model = model_finetuner.create_model(model_type=model_type)
        
        # Fine-tune model
        logger.info(f"Fine-tuning model for {epochs} initial epochs and {fine_tune_epochs} fine-tuning epochs...")
        model, history = model_finetuner.fine_tune(
            model,
            train_generator,
            val_generator,
            epochs=epochs,
            fine_tune_epochs=fine_tune_epochs,
            model_name=model_name
        )
        
        # Evaluate model
        logger.info("Evaluating model on test set...")
        evaluation = model_finetuner.evaluate_model(model, test_generator)
        
        # Plot training history
        logger.info("Plotting training history...")
        model_finetuner.plot_training_history(history, model_name)
        
        # Save evaluation results
        eval_path = os.path.join(model_finetuner.model_dir, f"{model_name}_evaluation.json")
        with open(eval_path, 'w') as f:
            json.dump(evaluation, f, indent=2)
        
        logger.info(f"Fine-tuning completed successfully. Model saved to {os.path.join(model_finetuner.model_dir, f'{model_name}.h5')}")
        
        return {
            "success": True,
            "model_path": os.path.join(model_finetuner.model_dir, f"{model_name}.h5"),
            "evaluation": evaluation,
            "history_plot": os.path.join(model_finetuner.model_dir, f"{model_name}_training_history.png"),
            "evaluation_path": eval_path
        }
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run fine-tuning on Indian faces")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--model-type", type=str, default="efficientnet", choices=["efficientnet", "resnet"], help="Model type")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for initial training")
    parser.add_argument("--fine-tune-epochs", type=int, default=5, help="Number of epochs for fine-tuning")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--model-name", type=str, default="indian_deepfake_detector", help="Model name")
    
    args = parser.parse_args()
    
    # Run fine-tuning
    results = run_finetuning(
        dataset_dir=args.dataset_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        model_name=args.model_name
    )
    
    # Save results
    results_path = os.path.join("fine_tuned_models", f"{args.model_name}_finetuning_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved fine-tuning results to {results_path}")
    
    if results["success"]:
        logger.info("Fine-tuning completed successfully!")
        logger.info(f"Model saved to: {results['model_path']}")
        logger.info(f"Evaluation results saved to: {results['evaluation_path']}")
        logger.info(f"Training history plot saved to: {results['history_plot']}")
    else:
        logger.error(f"Fine-tuning failed: {results['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()