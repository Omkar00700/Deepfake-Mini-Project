"""
Run enhanced deepfake detection on images or videos
"""

import os
import argparse
import logging
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from enhanced_detection import EnhancedDeepfakeDetector

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Run enhanced deepfake detection")
    
    # Input options
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input image or video file")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save results")
    
    # Detection options
    parser.add_argument("--no-ensemble", action="store_true",
                       help="Disable ensemble learning")
    parser.add_argument("--no-temporal", action="store_true",
                       help="Disable temporal analysis")
    parser.add_argument("--no-cross-modal", action="store_true",
                       help="Disable cross-modal verification")
    parser.add_argument("--models", type=str, nargs="+", 
                       default=["efficientnet", "xception", "vit"],
                       help="Models to include in the ensemble")
    
    # Visualization options
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualization of results")
    
    return parser.parse_args()

def visualize_image_result(image_path, result, output_path):
    """Visualize image detection result"""
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Display image
    plt.imshow(image)
    
    # Add bounding boxes for faces
    if "faces" in result:
        for face in result["faces"]:
            x, y, w, h = face["x"], face["y"], face["width"], face["height"]
            prob = face["probability"]
            conf = face["confidence"]
            
            # Determine color based on probability (red for fake, green for real)
            color = (1, 0, 0) if prob > 0.5 else (0, 1, 0)
            
            # Draw rectangle
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            plt.gca().add_patch(rect)
            
            # Add label
            label = f"Fake: {prob:.2f}, Conf: {conf:.2f}"
            plt.text(x, y-10, label, color=color, fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.7))
    
    # Add overall result as title
    prob = result.get("probability", 0)
    conf = result.get("confidence", 0)
    is_fake = "FAKE" if prob > 0.5 else "REAL"
    title = f"Result: {is_fake} (Probability: {prob:.4f}, Confidence: {conf:.4f})"
    plt.title(title)
    
    # Remove axes
    plt.axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_video_result(video_path, result, output_path):
    """Visualize video detection result"""
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot frame-by-frame probabilities if available
    if "frame_results" in result:
        frame_indices = [f["frame_idx"] for f in result["frame_results"]]
        probabilities = [f["probability"] for f in result["frame_results"]]
        confidences = [f["confidence"] for f in result["frame_results"]]
        
        plt.plot(frame_indices, probabilities, 'r-', label='Deepfake Probability')
        plt.plot(frame_indices, confidences, 'b--', label='Confidence')
        plt.axhline(y=0.5, color='g', linestyle='-', label='Decision Threshold')
        
        plt.xlabel('Frame Index')
        plt.ylabel('Probability / Confidence')
        plt.legend()
        
        # Add temporal consistency if available
        if "temporal_analysis" in result and "consistency_score" in result["temporal_analysis"]:
            consistency = result["temporal_analysis"]["consistency_score"]
            plt.text(0.02, 0.1, f"Temporal Consistency: {consistency:.4f}", 
                    transform=plt.gca().transAxes, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7))
    
    # Add overall result as title
    prob = result.get("probability", 0)
    conf = result.get("confidence", 0)
    is_fake = "FAKE" if prob > 0.5 else "REAL"
    title = f"Result: {is_fake} (Probability: {prob:.4f}, Confidence: {conf:.4f})"
    plt.title(title)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize detector
    logger.info("Initializing enhanced deepfake detector")
    detector = EnhancedDeepfakeDetector(
        use_ensemble=not args.no_ensemble,
        use_temporal=not args.no_temporal,
        use_cross_modal=not args.no_cross_modal,
        model_names=args.models
    )
    
    # Check if input is image or video
    is_video = args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    # Run detection
    logger.info(f"Running detection on {args.input}")
    if is_video:
        result = detector.detect_video(
            args.input, 
            include_details=True
        )
    else:
        result = detector.detect_image(
            args.input, 
            include_details=True
        )
    
    # Save result as JSON
    output_json = os.path.join(args.output_dir, "detection_result.json")
    with open(output_json, "w") as f:
        # Convert numpy values to Python types
        def convert_numpy(obj):
            if isinstance(obj, np.number):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_numpy(result), f, indent=2)
    
    # Visualize result if requested
    if args.visualize:
        output_viz = os.path.join(args.output_dir, "detection_visualization.png")
        if is_video:
            visualize_video_result(args.input, result, output_viz)
        else:
            visualize_image_result(args.input, result, output_viz)
    
    # Print result
    prob = result.get("probability", 0)
    conf = result.get("confidence", 0)
    is_fake = "FAKE" if prob > 0.5 else "REAL"
    logger.info(f"Detection result: {is_fake}")
    logger.info(f"Probability: {prob:.4f}")
    logger.info(f"Confidence: {conf:.4f}")
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()