"""
Test Framework for DeepDefend
Provides tools for testing and validating the deepfake detection system
with diverse Indian faces
"""

import os
import sys
import logging
import json
import csv
import time
import random
import shutil
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# Import project modules
from skin_tone_analyzer import SkinToneAnalyzer
from gan_detector import GANDetector
from enhanced_detection_handler import process_image_enhanced
from indian_face_utils import IndianFacePreprocessor
from face_detector import FaceDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestDatasetManager:
    """
    Manages test datasets for validation
    """
    
    def __init__(self, base_dir: str = "test_data"):
        """
        Initialize the test dataset manager
        
        Args:
            base_dir: Base directory for test datasets
        """
        self.base_dir = base_dir
        self.real_dir = os.path.join(base_dir, "real")
        self.fake_dir = os.path.join(base_dir, "fake")
        self.results_dir = os.path.join(base_dir, "results")
        
        # Create directories if they don't exist
        for directory in [self.base_dir, self.real_dir, self.fake_dir, self.results_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initialize skin tone analyzer for categorization
        self.skin_tone_analyzer = SkinToneAnalyzer()
        self.face_detector = FaceDetector()
        
        logger.info(f"Initialized test dataset manager with base directory: {base_dir}")
    
    def organize_dataset(self, source_dir: str, is_real: bool = True):
        """
        Organize a dataset of images into the test structure
        
        Args:
            source_dir: Source directory containing images
            is_real: Whether the images are real (True) or fake (False)
        """
        target_dir = self.real_dir if is_real else self.fake_dir
        
        # Create skin tone subdirectories
        skin_tones = ["fair", "wheatish", "medium", "dusky", "dark", "unknown"]
        for tone in skin_tones:
            tone_dir = os.path.join(target_dir, tone)
            if not os.path.exists(tone_dir):
                os.makedirs(tone_dir)
        
        # Process each image in the source directory
        image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        logger.info(f"Processing {len(image_files)} images from {source_dir}")
        
        for image_file in tqdm(image_files, desc="Organizing images"):
            try:
                # Read image
                image_path = os.path.join(source_dir, image_file)
                image = cv2.imread(image_path)
                
                if image is None:
                    logger.warning(f"Failed to read image: {image_path}")
                    continue
                
                # Detect faces
                faces = self.face_detector.detect_faces(image)
                
                if not faces:
                    logger.warning(f"No faces detected in: {image_path}")
                    # Copy to unknown category
                    shutil.copy(image_path, os.path.join(target_dir, "unknown", image_file))
                    continue
                
                # Get the largest face
                largest_face = max(faces, key=lambda face: face[2] * face[3])
                x, y, w, h = largest_face
                
                # Extract face region
                face_img = image[y:y+h, x:x+w]
                
                # Analyze skin tone
                skin_tone_result = self.skin_tone_analyzer.analyze_skin_tone(face_img)
                
                if not skin_tone_result.get("success", False):
                    logger.warning(f"Failed to analyze skin tone: {image_path}")
                    # Copy to unknown category
                    shutil.copy(image_path, os.path.join(target_dir, "unknown", image_file))
                    continue
                
                # Get Indian skin tone
                indian_tone = skin_tone_result.get("indian_tone", {})
                tone_type = indian_tone.get("type", "unknown")
                
                # Copy to appropriate directory
                shutil.copy(image_path, os.path.join(target_dir, tone_type, image_file))
                
            except Exception as e:
                logger.error(f"Error processing {image_file}: {str(e)}")
                # Copy to unknown category
                try:
                    shutil.copy(image_path, os.path.join(target_dir, "unknown", image_file))
                except Exception:
                    pass
        
        logger.info(f"Finished organizing dataset from {source_dir}")
    
    def count_images_by_category(self) -> Dict[str, Dict[str, int]]:
        """
        Count the number of images in each category
        
        Returns:
            Dictionary with counts by category
        """
        result = {
            "real": {},
            "fake": {}
        }
        
        # Count real images
        for tone in os.listdir(self.real_dir):
            tone_dir = os.path.join(self.real_dir, tone)
            if os.path.isdir(tone_dir):
                count = len([f for f in os.listdir(tone_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                result["real"][tone] = count
        
        # Count fake images
        for tone in os.listdir(self.fake_dir):
            tone_dir = os.path.join(self.fake_dir, tone)
            if os.path.isdir(tone_dir):
                count = len([f for f in os.listdir(tone_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                result["fake"][tone] = count
        
        return result
    
    def create_balanced_test_set(self, output_dir: str, samples_per_category: int = 20):
        """
        Create a balanced test set with equal representation of each category
        
        Args:
            output_dir: Output directory for the balanced test set
            samples_per_category: Number of samples to include per category
        """
        # Create output directories
        real_output = os.path.join(output_dir, "real")
        fake_output = os.path.join(output_dir, "fake")
        
        for directory in [output_dir, real_output, fake_output]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Process real images
        for tone in os.listdir(self.real_dir):
            tone_dir = os.path.join(self.real_dir, tone)
            if not os.path.isdir(tone_dir):
                continue
                
            # Get all images in this category
            images = [f for f in os.listdir(tone_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Select random samples
            selected = random.sample(images, min(samples_per_category, len(images)))
            
            # Copy selected images
            for image in selected:
                src = os.path.join(tone_dir, image)
                dst = os.path.join(real_output, f"{tone}_{image}")
                shutil.copy(src, dst)
        
        # Process fake images
        for tone in os.listdir(self.fake_dir):
            tone_dir = os.path.join(self.fake_dir, tone)
            if not os.path.isdir(tone_dir):
                continue
                
            # Get all images in this category
            images = [f for f in os.listdir(tone_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Select random samples
            selected = random.sample(images, min(samples_per_category, len(images)))
            
            # Copy selected images
            for image in selected:
                src = os.path.join(tone_dir, image)
                dst = os.path.join(fake_output, f"{tone}_{image}")
                shutil.copy(src, dst)
        
        logger.info(f"Created balanced test set in {output_dir}")


class TestEvaluator:
    """
    Evaluates the performance of the deepfake detection system
    """
    
    def __init__(self, results_dir: str = "test_results"):
        """
        Initialize the test evaluator
        
        Args:
            results_dir: Directory to store test results
        """
        self.results_dir = results_dir
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Initialize components
        self.gan_detector = GANDetector()
        self.skin_tone_analyzer = SkinToneAnalyzer()
        
        # Initialize metrics
        self.metrics = {
            "total": 0,
            "correct": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "by_skin_tone": {},
            "processing_times": []
        }
        
        logger.info(f"Initialized test evaluator with results directory: {results_dir}")
    
    def evaluate_image(self, image_path: str, is_real: bool) -> Dict[str, Any]:
        """
        Evaluate a single image
        
        Args:
            image_path: Path to the image
            is_real: Whether the image is real (True) or fake (False)
            
        Returns:
            Evaluation results
        """
        try:
            start_time = time.time()
            
            # Process image with enhanced detection
            probability, confidence, regions = process_image_enhanced(image_path)
            
            # Read image for additional analysis
            image = cv2.imread(image_path)
            
            # Analyze skin tone
            skin_tone_result = self.skin_tone_analyzer.analyze_skin_tone(image)
            
            # Detect GAN artifacts
            gan_result = self.gan_detector.detect(image)
            
            processing_time = time.time() - start_time
            
            # Determine prediction (deepfake or real)
            prediction = probability >= 0.5  # True if deepfake, False if real
            
            # Calculate metrics
            correct = prediction != is_real  # XOR operation (prediction is opposite of is_real)
            
            # Get skin tone
            indian_tone = "unknown"
            if skin_tone_result.get("success", False) and skin_tone_result.get("indian_tone"):
                indian_tone = skin_tone_result["indian_tone"].get("type", "unknown")
            
            # Create result
            result = {
                "image_path": image_path,
                "is_real": is_real,
                "prediction": "deepfake" if prediction else "real",
                "correct": correct,
                "probability": float(probability),
                "confidence": float(confidence),
                "processing_time": processing_time,
                "skin_tone": indian_tone,
                "gan_probability": gan_result.get("gan_probability", 0.0),
                "regions": regions
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating {image_path}: {str(e)}")
            return {
                "image_path": image_path,
                "is_real": is_real,
                "error": str(e),
                "correct": False
            }
    
    def evaluate_test_set(self, test_dir: str) -> Dict[str, Any]:
        """
        Evaluate a test set
        
        Args:
            test_dir: Directory containing the test set
            
        Returns:
            Evaluation results
        """
        # Reset metrics
        self.metrics = {
            "total": 0,
            "correct": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "by_skin_tone": {},
            "processing_times": []
        }
        
        # Get real and fake directories
        real_dir = os.path.join(test_dir, "real")
        fake_dir = os.path.join(test_dir, "fake")
        
        if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
            logger.error(f"Test directory structure invalid: {test_dir}")
            return self.metrics
        
        # Process real images
        real_images = [f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        logger.info(f"Processing {len(real_images)} real images")
        
        real_results = []
        for image in tqdm(real_images, desc="Evaluating real images"):
            image_path = os.path.join(real_dir, image)
            result = self.evaluate_image(image_path, True)
            real_results.append(result)
            
            # Update metrics
            self.metrics["total"] += 1
            if result.get("correct", False):
                self.metrics["correct"] += 1
            else:
                self.metrics["false_positives"] += 1
            
            # Update skin tone metrics
            skin_tone = result.get("skin_tone", "unknown")
            if skin_tone not in self.metrics["by_skin_tone"]:
                self.metrics["by_skin_tone"][skin_tone] = {
                    "total": 0, "correct": 0, "accuracy": 0.0
                }
            
            self.metrics["by_skin_tone"][skin_tone]["total"] += 1
            if result.get("correct", False):
                self.metrics["by_skin_tone"][skin_tone]["correct"] += 1
            
            # Update processing times
            if "processing_time" in result:
                self.metrics["processing_times"].append(result["processing_time"])
        
        # Process fake images
        fake_images = [f for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        logger.info(f"Processing {len(fake_images)} fake images")
        
        fake_results = []
        for image in tqdm(fake_images, desc="Evaluating fake images"):
            image_path = os.path.join(fake_dir, image)
            result = self.evaluate_image(image_path, False)
            fake_results.append(result)
            
            # Update metrics
            self.metrics["total"] += 1
            if result.get("correct", False):
                self.metrics["correct"] += 1
            else:
                self.metrics["false_negatives"] += 1
            
            # Update skin tone metrics
            skin_tone = result.get("skin_tone", "unknown")
            if skin_tone not in self.metrics["by_skin_tone"]:
                self.metrics["by_skin_tone"][skin_tone] = {
                    "total": 0, "correct": 0, "accuracy": 0.0
                }
            
            self.metrics["by_skin_tone"][skin_tone]["total"] += 1
            if result.get("correct", False):
                self.metrics["by_skin_tone"][skin_tone]["correct"] += 1
            
            # Update processing times
            if "processing_time" in result:
                self.metrics["processing_times"].append(result["processing_time"])
        
        # Calculate final metrics
        if self.metrics["total"] > 0:
            self.metrics["accuracy"] = self.metrics["correct"] / self.metrics["total"]
        else:
            self.metrics["accuracy"] = 0.0
        
        # Calculate skin tone accuracies
        for tone, data in self.metrics["by_skin_tone"].items():
            if data["total"] > 0:
                data["accuracy"] = data["correct"] / data["total"]
        
        # Calculate average processing time
        if self.metrics["processing_times"]:
            self.metrics["avg_processing_time"] = sum(self.metrics["processing_times"]) / len(self.metrics["processing_times"])
        else:
            self.metrics["avg_processing_time"] = 0.0
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f"evaluation_results_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump({
                "metrics": self.metrics,
                "real_results": real_results,
                "fake_results": fake_results
            }, f, indent=2)
        
        logger.info(f"Evaluation complete. Results saved to {results_file}")
        logger.info(f"Overall accuracy: {self.metrics['accuracy']:.4f}")
        
        return self.metrics
    
    def generate_report(self, metrics: Dict[str, Any] = None):
        """
        Generate a visual report of the evaluation results
        
        Args:
            metrics: Metrics to use for the report (uses self.metrics if None)
        """
        if metrics is None:
            metrics = self.metrics
        
        if not metrics:
            logger.error("No metrics available for report generation")
            return
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Overall accuracy
        plt.subplot(2, 2, 1)
        plt.bar(['Accuracy'], [metrics["accuracy"]], color='blue')
        plt.ylim(0, 1)
        plt.title('Overall Accuracy')
        plt.ylabel('Accuracy')
        
        # Plot 2: Confusion matrix
        plt.subplot(2, 2, 2)
        labels = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
        true_negatives = metrics["correct"] - metrics["false_negatives"]
        true_positives = metrics["correct"] - metrics["false_positives"]
        values = [true_negatives, metrics["false_positives"], 
                 metrics["false_negatives"], true_positives]
        plt.bar(labels, values, color=['green', 'red', 'red', 'green'])
        plt.title('Confusion Matrix')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        
        # Plot 3: Accuracy by skin tone
        plt.subplot(2, 2, 3)
        tones = list(metrics["by_skin_tone"].keys())
        accuracies = [data["accuracy"] for data in metrics["by_skin_tone"].values()]
        plt.bar(tones, accuracies, color='purple')
        plt.ylim(0, 1)
        plt.title('Accuracy by Skin Tone')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45, ha='right')
        
        # Plot 4: Processing time
        plt.subplot(2, 2, 4)
        plt.hist(metrics["processing_times"], bins=20, color='orange')
        plt.axvline(metrics["avg_processing_time"], color='red', linestyle='dashed', linewidth=2)
        plt.title('Processing Time Distribution')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency')
        
        # Adjust layout and save
        plt.tight_layout()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.results_dir, f"evaluation_report_{timestamp}.png")
        plt.savefig(report_file)
        
        logger.info(f"Report generated and saved to {report_file}")
        
        # Generate CSV report for skin tone analysis
        csv_file = os.path.join(self.results_dir, f"skin_tone_report_{timestamp}.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Skin Tone', 'Total Images', 'Correct Predictions', 'Accuracy'])
            
            for tone, data in metrics["by_skin_tone"].items():
                writer.writerow([tone, data["total"], data["correct"], data["accuracy"]])
        
        logger.info(f"Skin tone report saved to {csv_file}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test framework for DeepDefend")
    parser.add_argument("--organize", action="store_true", help="Organize a dataset")
    parser.add_argument("--source", type=str, help="Source directory for dataset organization")
    parser.add_argument("--real", action="store_true", help="Source contains real images (for --organize)")
    parser.add_argument("--create-test-set", action="store_true", help="Create a balanced test set")
    parser.add_argument("--samples", type=int, default=20, help="Samples per category for test set")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate a test set")
    parser.add_argument("--test-dir", type=str, help="Test set directory for evaluation")
    
    args = parser.parse_args()
    
    # Initialize managers
    dataset_manager = TestDatasetManager()
    evaluator = TestEvaluator()
    
    if args.organize and args.source:
        dataset_manager.organize_dataset(args.source, args.real)
        counts = dataset_manager.count_images_by_category()
        print("Dataset organization complete. Image counts:")
        print(json.dumps(counts, indent=2))
    
    elif args.create_test_set:
        output_dir = "balanced_test_set"
        dataset_manager.create_balanced_test_set(output_dir, args.samples)
        print(f"Balanced test set created in {output_dir}")
    
    elif args.evaluate and args.test_dir:
        metrics = evaluator.evaluate_test_set(args.test_dir)
        evaluator.generate_report(metrics)
        print("Evaluation complete. Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"False positives: {metrics['false_positives']}")
        print(f"False negatives: {metrics['false_negatives']}")
        print(f"Average processing time: {metrics['avg_processing_time']:.2f} seconds")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()