"""
Indian Face Collection Script for DeepDefend
This script helps collect and organize Indian face data for training and testing
"""

import os
import sys
import logging
import argparse
import shutil
import random
import cv2
import numpy as np
from tqdm import tqdm
from face_detector import FaceDetector
from indian_face_utils import IndianFacePreprocessor
from skin_tone_analyzer import SkinToneAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IndianFaceCollector:
    """
    Collects and organizes Indian face data for training and testing
    """
    
    def __init__(self, output_dir="data/indian_faces"):
        """
        Initialize the face collector
        
        Args:
            output_dir: Directory to save collected faces
        """
        self.output_dir = output_dir
        
        # Create output directories
        self.real_dir = os.path.join(output_dir, "real")
        self.fake_dir = os.path.join(output_dir, "fake")
        
        for directory in [self.output_dir, self.real_dir, self.fake_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize face detector
        self.face_detector = FaceDetector(confidence_threshold=0.5, detector_type="mtcnn")
        
        # Initialize face preprocessor
        self.face_preprocessor = IndianFacePreprocessor(self.face_detector)
        
        # Initialize skin tone analyzer
        self.skin_tone_analyzer = SkinToneAnalyzer()
        
        logger.info(f"Initialized Indian face collector with output directory: {output_dir}")
    
    def collect_from_directory(self, source_dir, target_class="real", min_faces=1, max_faces=10):
        """
        Collect faces from a directory of images
        
        Args:
            source_dir: Source directory containing images
            target_class: Target class ("real" or "fake")
            min_faces: Minimum number of faces to extract from each image
            max_faces: Maximum number of faces to extract from each image
        
        Returns:
            Number of faces collected
        """
        if target_class not in ["real", "fake"]:
            raise ValueError("Target class must be 'real' or 'fake'")
        
        target_dir = self.real_dir if target_class == "real" else self.fake_dir
        
        # Get all image files
        image_files = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(image_files)} images in {source_dir}")
        
        # Process each image
        face_count = 0
        for image_file in tqdm(image_files, desc=f"Processing {target_class} images"):
            try:
                # Read image
                image = cv2.imread(image_file)
                if image is None:
                    logger.warning(f"Failed to read image: {image_file}")
                    continue
                
                # Detect faces
                faces = self.face_detector.detect_faces(image)
                
                # Skip if no faces detected
                if not faces:
                    logger.debug(f"No faces detected in {image_file}")
                    continue
                
                # Limit number of faces per image
                faces = faces[:max_faces]
                
                # Process each face
                for i, face in enumerate(faces):
                    # Preprocess face
                    processed_face = self.face_preprocessor.preprocess_face(
                        image, face, target_size=(224, 224), margin_percent=0.2
                    )
                    
                    if processed_face is None:
                        logger.debug(f"Failed to preprocess face {i} in {image_file}")
                        continue
                    
                    # Analyze skin tone
                    skin_tone = processed_face.get("skin_tone", {})
                    
                    # Skip if not an Indian skin tone
                    if not skin_tone.get("success", False) or not skin_tone.get("indian_tone"):
                        logger.debug(f"Face {i} in {image_file} does not have a recognized Indian skin tone")
                        continue
                    
                    # Save face image
                    face_filename = f"{os.path.splitext(os.path.basename(image_file))[0]}_face{i}.jpg"
                    face_path = os.path.join(target_dir, face_filename)
                    cv2.imwrite(face_path, processed_face["image"])
                    
                    # Save metadata
                    metadata = {
                        "source_image": image_file,
                        "face_index": i,
                        "bbox": processed_face["bbox"],
                        "skin_tone": skin_tone
                    }
                    
                    metadata_path = os.path.join(target_dir, f"{os.path.splitext(face_filename)[0]}.json")
                    with open(metadata_path, 'w') as f:
                        import json
                        json.dump(metadata, f, indent=2)
                    
                    face_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing {image_file}: {str(e)}")
        
        logger.info(f"Collected {face_count} {target_class} faces")
        return face_count
    
    def organize_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Organize collected faces into train/val/test splits
        
        Args:
            train_ratio: Ratio of images to use for training
            val_ratio: Ratio of images to use for validation
            test_ratio: Ratio of images to use for testing
        """
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("Ratios must sum to 1.0")
        
        # Create split directories
        splits = ["train", "val", "test"]
        split_dirs = {}
        
        for split in splits:
            for class_name in ["real", "fake"]:
                split_dir = os.path.join(self.output_dir, split, class_name)
                os.makedirs(split_dir, exist_ok=True)
                split_dirs[(split, class_name)] = split_dir
        
        # Process each class
        for class_name in ["real", "fake"]:
            source_dir = self.real_dir if class_name == "real" else self.fake_dir
            
            # Get all face images
            face_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.jpg')]
            
            # Shuffle files
            random.shuffle(face_files)
            
            # Calculate split indices
            n_train = int(len(face_files) * train_ratio)
            n_val = int(len(face_files) * val_ratio)
            
            # Split files
            train_files = face_files[:n_train]
            val_files = face_files[n_train:n_train+n_val]
            test_files = face_files[n_train+n_val:]
            
            # Copy files to respective directories
            for split, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
                for file in files:
                    # Copy image file
                    source_path = os.path.join(source_dir, file)
                    target_path = os.path.join(split_dirs[(split, class_name)], file)
                    shutil.copy(source_path, target_path)
                    
                    # Copy metadata file if it exists
                    metadata_file = f"{os.path.splitext(file)[0]}.json"
                    metadata_source = os.path.join(source_dir, metadata_file)
                    if os.path.exists(metadata_source):
                        metadata_target = os.path.join(split_dirs[(split, class_name)], metadata_file)
                        shutil.copy(metadata_source, metadata_target)
            
            logger.info(f"Organized {class_name} faces: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    def analyze_dataset(self):
        """
        Analyze the collected dataset and generate statistics
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "total_faces": 0,
            "real_faces": 0,
            "fake_faces": 0,
            "skin_tone_distribution": {
                "real": {},
                "fake": {}
            }
        }
        
        # Process each class
        for class_name in ["real", "fake"]:
            source_dir = self.real_dir if class_name == "real" else self.fake_dir
            
            # Get all face images
            face_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.jpg')]
            
            # Update counts
            if class_name == "real":
                stats["real_faces"] = len(face_files)
            else:
                stats["fake_faces"] = len(face_files)
            
            stats["total_faces"] += len(face_files)
            
            # Analyze skin tones
            for file in face_files:
                metadata_file = f"{os.path.splitext(file)[0]}.json"
                metadata_path = os.path.join(source_dir, metadata_file)
                
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            import json
                            metadata = json.load(f)
                        
                        if "skin_tone" in metadata and metadata["skin_tone"].get("success", False):
                            if "indian_tone" in metadata["skin_tone"] and metadata["skin_tone"]["indian_tone"]:
                                tone_type = metadata["skin_tone"]["indian_tone"].get("type", "unknown")
                                
                                if tone_type not in stats["skin_tone_distribution"][class_name]:
                                    stats["skin_tone_distribution"][class_name][tone_type] = 0
                                
                                stats["skin_tone_distribution"][class_name][tone_type] += 1
                    except Exception as e:
                        logger.error(f"Error reading metadata {metadata_path}: {str(e)}")
        
        logger.info(f"Dataset statistics: {stats}")
        return stats

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Collect and organize Indian face data")
    parser.add_argument("--real-dir", type=str, help="Directory containing real face images")
    parser.add_argument("--fake-dir", type=str, help="Directory containing fake face images")
    parser.add_argument("--output-dir", type=str, default="data/indian_faces", help="Output directory")
    parser.add_argument("--organize", action="store_true", help="Organize dataset into train/val/test splits")
    parser.add_argument("--analyze", action="store_true", help="Analyze dataset and generate statistics")
    
    args = parser.parse_args()
    
    # Initialize face collector
    collector = IndianFaceCollector(output_dir=args.output_dir)
    
    # Collect faces if directories are provided
    if args.real_dir:
        collector.collect_from_directory(args.real_dir, target_class="real")
    
    if args.fake_dir:
        collector.collect_from_directory(args.fake_dir, target_class="fake")
    
    # Organize dataset if requested
    if args.organize:
        collector.organize_dataset()
    
    # Analyze dataset if requested
    if args.analyze:
        stats = collector.analyze_dataset()
        
        # Save statistics
        stats_path = os.path.join(args.output_dir, "dataset_stats.json")
        with open(stats_path, 'w') as f:
            import json
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved dataset statistics to {stats_path}")
    
    # If no actions were specified, print help
    if not (args.real_dir or args.fake_dir or args.organize or args.analyze):
        parser.print_help()

if __name__ == "__main__":
    main()