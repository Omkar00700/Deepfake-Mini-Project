"""
Run enhanced deepfake detection with improved Indian face detection
"""

import os
import sys
import argparse
import time
import json
import logging
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import detection functions
from backend.enhanced_detection_pipeline import process_image, process_video

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

def process_file(file_path, output_dir=None, use_ensemble=True, use_indian_enhancement=True):
    """
    Process a single file (image or video)
    
    Args:
        file_path: Path to the file
        output_dir: Directory to save results
        use_ensemble: Whether to use ensemble of models
        use_indian_enhancement: Whether to use Indian-specific enhancements
        
    Returns:
        Detection results
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        # Determine file type
        file_ext = os.path.splitext(file_path)[1].lower()
        is_video = file_ext in ['.mp4', '.avi', '.mov', '.wmv']
        
        # Process file
        start_time = time.time()
        
        if is_video:
            probability, confidence, frame_count, regions = process_video(
                file_path, 
                use_ensemble=use_ensemble,
                use_indian_enhancement=use_indian_enhancement
            )
            file_type = "video"
        else:
            probability, confidence, regions = process_image(
                file_path,
                use_ensemble=use_ensemble,
                use_indian_enhancement=use_indian_enhancement
            )
            file_type = "image"
            frame_count = 1
        
        processing_time = time.time() - start_time
        
        # Create result
        result = {
            "file": file_path,
            "type": file_type,
            "probability": float(probability),
            "confidence": float(confidence),
            "is_deepfake": probability > 0.5,
            "processing_time": processing_time,
            "frame_count": frame_count if is_video else 1,
            "regions": regions,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "settings": {
                "use_ensemble": use_ensemble,
                "use_indian_enhancement": use_indian_enhancement
            }
        }
        
        # Save result if output directory is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create output filename
            base_name = os.path.basename(file_path)
            output_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_result.json")
            
            # Save to JSON file
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Result saved to {output_path}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None

def process_directory(directory, output_dir=None, use_ensemble=True, use_indian_enhancement=True, recursive=False):
    """
    Process all files in a directory
    
    Args:
        directory: Directory containing files to process
        output_dir: Directory to save results
        use_ensemble: Whether to use ensemble of models
        use_indian_enhancement: Whether to use Indian-specific enhancements
        recursive: Whether to process subdirectories
        
    Returns:
        List of detection results
    """
    try:
        # Check if directory exists
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return []
        
        # Get all files
        files = []
        
        if recursive:
            for root, _, filenames in os.walk(directory):
                for filename in filenames:
                    files.append(os.path.join(root, filename))
        else:
            files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        # Filter for images and videos
        image_extensions = ['.jpg', '.jpeg', '.png']
        video_extensions = ['.mp4', '.avi', '.mov', '.wmv']
        
        files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions + video_extensions]
        
        if not files:
            logger.warning(f"No image or video files found in {directory}")
            return []
        
        # Process each file
        results = []
        
        for file_path in tqdm(files, desc="Processing files"):
            result = process_file(
                file_path,
                output_dir=output_dir,
                use_ensemble=use_ensemble,
                use_indian_enhancement=use_indian_enhancement
            )
            
            if result:
                results.append(result)
        
        # Save summary if output directory is specified
        if output_dir and results:
            summary_path = os.path.join(output_dir, "summary.json")
            
            # Calculate statistics
            total_files = len(results)
            deepfake_count = sum(1 for r in results if r["is_deepfake"])
            real_count = total_files - deepfake_count
            
            avg_confidence = sum(r["confidence"] for r in results) / total_files if total_files > 0 else 0
            avg_processing_time = sum(r["processing_time"] for r in results) / total_files if total_files > 0 else 0
            
            # Create summary
            summary = {
                "total_files": total_files,
                "deepfake_count": deepfake_count,
                "real_count": real_count,
                "deepfake_percentage": deepfake_count / total_files * 100 if total_files > 0 else 0,
                "avg_confidence": avg_confidence,
                "avg_processing_time": avg_processing_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "settings": {
                    "use_ensemble": use_ensemble,
                    "use_indian_enhancement": use_indian_enhancement
                }
            }
            
            # Save to JSON file
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Summary saved to {summary_path}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error processing directory {directory}: {str(e)}")
        return []

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Run enhanced deepfake detection")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", help="Path to file to process")
    input_group.add_argument("--dir", help="Path to directory to process")
    
    # Output options
    parser.add_argument("--output", help="Directory to save results")
    
    # Processing options
    parser.add_argument("--no-ensemble", action="store_true", help="Disable ensemble of models")
    parser.add_argument("--no-indian", action="store_true", help="Disable Indian-specific enhancements")
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories recursively")
    
    args = parser.parse_args()
    
    # Process input
    if args.file:
        result = process_file(
            args.file,
            output_dir=args.output,
            use_ensemble=not args.no_ensemble,
            use_indian_enhancement=not args.no_indian
        )
        
        if result:
            print("\n===== DETECTION RESULT =====")
            print(f"File: {result['file']}")
            print(f"Type: {result['type']}")
            print(f"Deepfake Probability: {result['probability']:.4f}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Verdict: {'DEEPFAKE' if result['is_deepfake'] else 'REAL'}")
            print(f"Processing Time: {result['processing_time']:.2f} seconds")
            print(f"Regions Detected: {len(result['regions'])}")
            print("============================\n")
    
    elif args.dir:
        results = process_directory(
            args.dir,
            output_dir=args.output,
            use_ensemble=not args.no_ensemble,
            use_indian_enhancement=not args.no_indian,
            recursive=args.recursive
        )
        
        if results:
            # Calculate statistics
            total_files = len(results)
            deepfake_count = sum(1 for r in results if r["is_deepfake"])
            real_count = total_files - deepfake_count
            
            avg_confidence = sum(r["confidence"] for r in results) / total_files
            avg_processing_time = sum(r["processing_time"] for r in results) / total_files
            
            print("\n===== DETECTION SUMMARY =====")
            print(f"Total Files: {total_files}")
            print(f"Deepfakes: {deepfake_count} ({deepfake_count/total_files*100:.1f}%)")
            print(f"Real: {real_count} ({real_count/total_files*100:.1f}%)")
            print(f"Average Confidence: {avg_confidence:.4f}")
            print(f"Average Processing Time: {avg_processing_time:.2f} seconds")
            print("=============================\n")

if __name__ == "__main__":
    main()