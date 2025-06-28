"""
Improvement Validation Script for DeepDefend
This script validates the improvements after fine-tuning on Indian faces
"""

import os
import sys
import logging
import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from test_indian_faces import IndianFaceTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_improvements(original_model_path, fine_tuned_model_path, test_dataset_dir, output_dir):
    """
    Validate improvements by comparing original and fine-tuned models
    
    Args:
        original_model_path: Path to the original model
        fine_tuned_model_path: Path to the fine-tuned model
        test_dataset_dir: Directory containing test data
        output_dir: Directory to save validation results
        
    Returns:
        Dictionary with validation results
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Test original model
        logger.info(f"Testing original model: {original_model_path}")
        original_tester = IndianFaceTester(model_path=original_model_path)
        original_results = original_tester.test_dataset(test_dataset_dir)
        
        # Test fine-tuned model
        logger.info(f"Testing fine-tuned model: {fine_tuned_model_path}")
        fine_tuned_tester = IndianFaceTester(model_path=fine_tuned_model_path)
        fine_tuned_results = fine_tuned_tester.test_dataset(test_dataset_dir)
        
        # Save individual results
        with open(os.path.join(output_dir, "original_model_results.json"), 'w') as f:
            json.dump(original_results, f, indent=2)
        
        with open(os.path.join(output_dir, "fine_tuned_model_results.json"), 'w') as f:
            json.dump(fine_tuned_results, f, indent=2)
        
        # Calculate improvements
        improvements = {}
        
        if original_results["success"] and fine_tuned_results["success"]:
            # Calculate metric improvements
            original_metrics = original_results["metrics"]
            fine_tuned_metrics = fine_tuned_results["metrics"]
            
            metric_improvements = {}
            for metric in original_metrics:
                original_value = original_metrics[metric]
                fine_tuned_value = fine_tuned_metrics[metric]
                absolute_change = fine_tuned_value - original_value
                relative_change = (absolute_change / original_value) * 100 if original_value > 0 else float('inf')
                
                metric_improvements[metric] = {
                    "original": original_value,
                    "fine_tuned": fine_tuned_value,
                    "absolute_change": absolute_change,
                    "relative_change": relative_change
                }
            
            improvements["metrics"] = metric_improvements
            
            # Calculate skin tone improvements
            original_skin_tone = original_results["skin_tone_performance"]
            fine_tuned_skin_tone = fine_tuned_results["skin_tone_performance"]
            
            skin_tone_improvements = {}
            all_tones = set(list(original_skin_tone.keys()) + list(fine_tuned_skin_tone.keys()))
            
            for tone in all_tones:
                skin_tone_improvements[tone] = {
                    "real": {
                        "original": original_skin_tone.get(tone, {}).get("real", {}).get("accuracy", 0),
                        "fine_tuned": fine_tuned_skin_tone.get(tone, {}).get("real", {}).get("accuracy", 0)
                    },
                    "fake": {
                        "original": original_skin_tone.get(tone, {}).get("fake", {}).get("accuracy", 0),
                        "fine_tuned": fine_tuned_skin_tone.get(tone, {}).get("fake", {}).get("accuracy", 0)
                    },
                    "overall": {
                        "original": original_skin_tone.get(tone, {}).get("overall_accuracy", 0),
                        "fine_tuned": fine_tuned_skin_tone.get(tone, {}).get("overall_accuracy", 0)
                    }
                }
                
                # Calculate changes
                for category in ["real", "fake", "overall"]:
                    original_value = skin_tone_improvements[tone][category]["original"]
                    fine_tuned_value = skin_tone_improvements[tone][category]["fine_tuned"]
                    absolute_change = fine_tuned_value - original_value
                    relative_change = (absolute_change / original_value) * 100 if original_value > 0 else float('inf')
                    
                    skin_tone_improvements[tone][category]["absolute_change"] = absolute_change
                    skin_tone_improvements[tone][category]["relative_change"] = relative_change
            
            improvements["skin_tone"] = skin_tone_improvements
            
            # Plot comparison charts
            plot_metric_comparison(metric_improvements, output_dir)
            plot_skin_tone_comparison(skin_tone_improvements, output_dir)
            
            # Save improvements
            with open(os.path.join(output_dir, "improvements.json"), 'w') as f:
                json.dump(improvements, f, indent=2)
            
            logger.info("Validation completed successfully")
            
            return {
                "success": True,
                "improvements": improvements,
                "original_results": original_results,
                "fine_tuned_results": fine_tuned_results
            }
        else:
            error_message = "Failed to test one or both models"
            if not original_results["success"]:
                error_message += f": Original model - {original_results.get('error', 'Unknown error')}"
            if not fine_tuned_results["success"]:
                error_message += f": Fine-tuned model - {fine_tuned_results.get('error', 'Unknown error')}"
            
            logger.error(error_message)
            
            return {
                "success": False,
                "error": error_message
            }
        
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

def plot_metric_comparison(metric_improvements, output_dir):
    """
    Plot metric comparison between original and fine-tuned models
    
    Args:
        metric_improvements: Dictionary with metric improvements
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(12, 8))
    
    metrics = list(metric_improvements.keys())
    original_values = [metric_improvements[m]["original"] for m in metrics]
    fine_tuned_values = [metric_improvements[m]["fine_tuned"] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, original_values, width, label='Original Model')
    plt.bar(x + width/2, fine_tuned_values, width, label='Fine-tuned Model')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Comparison: Original vs. Fine-tuned Model')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(original_values):
        plt.text(i - width/2, v + 0.01, f"{v:.3f}", ha='center')
    
    for i, v in enumerate(fine_tuned_values):
        plt.text(i + width/2, v + 0.01, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metric_comparison.png"))
    plt.close()

def plot_skin_tone_comparison(skin_tone_improvements, output_dir):
    """
    Plot skin tone comparison between original and fine-tuned models
    
    Args:
        skin_tone_improvements: Dictionary with skin tone improvements
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(15, 10))
    
    tones = list(skin_tone_improvements.keys())
    
    # Plot overall accuracy by skin tone
    plt.subplot(2, 1, 1)
    original_overall = [skin_tone_improvements[tone]["overall"]["original"] for tone in tones]
    fine_tuned_overall = [skin_tone_improvements[tone]["overall"]["fine_tuned"] for tone in tones]
    
    x = np.arange(len(tones))
    width = 0.35
    
    plt.bar(x - width/2, original_overall, width, label='Original Model')
    plt.bar(x + width/2, fine_tuned_overall, width, label='Fine-tuned Model')
    
    plt.xlabel('Skin Tone')
    plt.ylabel('Overall Accuracy')
    plt.title('Overall Accuracy by Skin Tone: Original vs. Fine-tuned Model')
    plt.xticks(x, tones)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot real vs fake accuracy by skin tone
    plt.subplot(2, 1, 2)
    
    # Prepare data
    categories = []
    original_values = []
    fine_tuned_values = []
    
    for tone in tones:
        for category in ["real", "fake"]:
            categories.append(f"{tone}-{category}")
            original_values.append(skin_tone_improvements[tone][category]["original"])
            fine_tuned_values.append(skin_tone_improvements[tone][category]["fine_tuned"])
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, original_values, width, label='Original Model')
    plt.bar(x + width/2, fine_tuned_values, width, label='Fine-tuned Model')
    
    plt.xlabel('Skin Tone - Category')
    plt.ylabel('Accuracy')
    plt.title('Real vs. Fake Accuracy by Skin Tone: Original vs. Fine-tuned Model')
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "skin_tone_comparison.png"))
    plt.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Validate improvements after fine-tuning")
    parser.add_argument("--original-model", type=str, required=True, help="Path to the original model")
    parser.add_argument("--fine-tuned-model", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--test-dataset", type=str, required=True, help="Directory containing test data")
    parser.add_argument("--output-dir", type=str, default="validation_results", help="Directory to save validation results")
    
    args = parser.parse_args()
    
    # Validate improvements
    results = validate_improvements(
        original_model_path=args.original_model,
        fine_tuned_model_path=args.fine_tuned_model,
        test_dataset_dir=args.test_dataset,
        output_dir=args.output_dir
    )
    
    # Save results
    results_path = os.path.join(args.output_dir, "validation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved validation results to {results_path}")
    
    if results["success"]:
        # Print summary of improvements
        improvements = results["improvements"]["metrics"]
        logger.info("=== Performance Improvements ===")
        for metric, data in improvements.items():
            logger.info(f"{metric}: {data['original']:.4f} → {data['fine_tuned']:.4f} ({data['relative_change']:+.2f}%)")
        
        # Print skin tone improvements
        skin_tone_improvements = results["improvements"]["skin_tone"]
        logger.info("=== Skin Tone Improvements ===")
        for tone, data in skin_tone_improvements.items():
            logger.info(f"{tone} (overall): {data['overall']['original']:.4f} → {data['overall']['fine_tuned']:.4f} ({data['overall']['relative_change']:+.2f}%)")
    else:
        logger.error(f"Validation failed: {results['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()