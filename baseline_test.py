"""
Baseline performance testing script for DeepDefend
"""

import os
import sys
import time
import json
import numpy as np
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import detection functions
from backend.detection_handler import process_image, process_video
from backend.inference import get_model_info

def test_on_directory(directory, is_deepfake=False, limit=None):
    """
    Test the model on all images and videos in a directory
    
    Args:
        directory: Path to directory containing images/videos
        is_deepfake: Whether the directory contains deepfake (True) or real (False) media
        limit: Maximum number of files to process
        
    Returns:
        Dictionary with test results
    """
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return {
            "error": f"Directory not found: {directory}",
            "results": []
        }
    
    # Get all image and video files
    image_extensions = ['.jpg', '.jpeg', '.png']
    video_extensions = ['.mp4', '.avi', '.mov']
    
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions or ext in video_extensions:
                files.append(os.path.join(root, filename))
    
    # Limit the number of files if specified
    if limit and len(files) > limit:
        files = files[:limit]
    
    # Process each file
    results = []
    correct_count = 0
    
    print(f"Processing {len(files)} files from {directory}...")
    
    for file_path in tqdm(files):
        try:
            # Determine if it's an image or video
            ext = os.path.splitext(file_path)[1].lower()
            is_video = ext in video_extensions
            
            # Process the file
            start_time = time.time()
            
            if is_video:
                probability, confidence, frame_count, _ = process_video(file_path)
                file_type = "video"
            else:
                probability, confidence, _ = process_image(file_path)
                file_type = "image"
                frame_count = 1
            
            processing_time = time.time() - start_time
            
            # Determine if prediction is correct
            predicted_deepfake = probability > 0.5
            is_correct = predicted_deepfake == is_deepfake
            
            if is_correct:
                correct_count += 1
            
            # Add to results
            results.append({
                "file": file_path,
                "type": file_type,
                "is_deepfake": is_deepfake,
                "predicted_deepfake": predicted_deepfake,
                "probability": float(probability),
                "confidence": float(confidence),
                "is_correct": is_correct,
                "processing_time": processing_time,
                "frame_count": frame_count if is_video else 1
            })
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            results.append({
                "file": file_path,
                "error": str(e)
            })
    
    # Calculate accuracy
    accuracy = correct_count / len(files) if files else 0
    
    return {
        "directory": directory,
        "is_deepfake": is_deepfake,
        "file_count": len(files),
        "correct_count": correct_count,
        "accuracy": accuracy,
        "results": results
    }

def main():
    """
    Run baseline tests on sample datasets
    """
    # Get model info
    model_info = get_model_info()
    print(f"Current model: {model_info}")
    
    # Define test directories
    # You should update these paths to point to your real and fake media directories
    real_dir = "data/test/real"
    fake_dir = "data/test/fake"
    
    # Create test directories if they don't exist
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    # Check if directories have files
    real_files = [f for f in os.listdir(real_dir) if os.path.isfile(os.path.join(real_dir, f))]
    fake_files = [f for f in os.listdir(fake_dir) if os.path.isfile(os.path.join(fake_dir, f))]
    
    if not real_files:
        print(f"Warning: No files found in {real_dir}")
    
    if not fake_files:
        print(f"Warning: No files found in {fake_dir}")
    
    # Run tests
    real_results = test_on_directory(real_dir, is_deepfake=False, limit=50)
    fake_results = test_on_directory(fake_dir, is_deepfake=True, limit=50)
    
    # Calculate overall metrics
    total_files = real_results["file_count"] + fake_results["file_count"]
    total_correct = real_results["correct_count"] + fake_results["correct_count"]
    overall_accuracy = total_correct / total_files if total_files > 0 else 0
    
    # Calculate precision, recall, and F1 score
    true_positives = fake_results["correct_count"]
    false_positives = real_results["file_count"] - real_results["correct_count"]
    false_negatives = fake_results["file_count"] - fake_results["correct_count"]
    true_negatives = real_results["correct_count"]
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Compile results
    overall_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": model_info,
        "overall_accuracy": overall_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "total_files": total_files,
        "total_correct": total_correct,
        "real_accuracy": real_results["accuracy"],
        "fake_accuracy": fake_results["accuracy"],
        "real_file_count": real_results["file_count"],
        "fake_file_count": fake_results["file_count"]
    }
    
    # Print results
    print("\n===== BASELINE PERFORMANCE =====")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Real Media Accuracy: {real_results['accuracy']:.4f}")
    print(f"Fake Media Accuracy: {fake_results['accuracy']:.4f}")
    print("================================\n")
    
    # Save results to file
    os.makedirs("analysis/day1", exist_ok=True)
    with open("analysis/day1/baseline_results.json", "w") as f:
        json.dump(overall_results, f, indent=2)
    
    print(f"Results saved to analysis/day1/baseline_results.json")

if __name__ == "__main__":
    main()