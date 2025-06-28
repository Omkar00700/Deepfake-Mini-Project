"""
Data preparation and augmentation script for DeepDefend
"""

import os
import sys
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm
import tensorflow as tf
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize face detector
detector = MTCNN()

def detect_and_extract_face(image_path, output_size=(224, 224), margin=0.2):
    """
    Detect and extract face from an image with margin
    
    Args:
        image_path: Path to the image
        output_size: Size of the output face image
        margin: Margin around the face as a fraction of face size
        
    Returns:
        Extracted face image or None if no face detected
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return None
        
        # Convert to RGB (MTCNN expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detector.detect_faces(image_rgb)
        
        if not faces:
            print(f"No face detected in {image_path}")
            return None
        
        # Get the largest face
        face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
        x, y, w, h = face['box']
        
        # Add margin
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        # Calculate coordinates with margin
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        
        # Extract face
        face_img = image[y1:y2, x1:x2]
        
        # Resize to output size
        face_img = cv2.resize(face_img, output_size)
        
        return face_img
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def augment_image(image):
    """
    Apply data augmentation to an image
    
    Args:
        image: Input image
        
    Returns:
        Augmented image
    """
    # Convert to tensor
    img_tensor = tf.convert_to_tensor(image)
    
    # Random brightness
    img_tensor = tf.image.random_brightness(img_tensor, max_delta=0.2)
    
    # Random contrast
    img_tensor = tf.image.random_contrast(img_tensor, lower=0.8, upper=1.2)
    
    # Random flip
    img_tensor = tf.image.random_flip_left_right(img_tensor)
    
    # Convert back to numpy
    augmented = img_tensor.numpy()
    
    return augmented

def augment_indian_faces(image):
    """
    Apply specialized augmentation for Indian faces
    
    Args:
        image: Input image
        
    Returns:
        Augmented image
    """
    # Convert to tensor
    img_tensor = tf.convert_to_tensor(image)
    
    # Adjust brightness slightly (Indian faces often have different lighting needs)
    img_tensor = tf.image.adjust_brightness(img_tensor, delta=random.uniform(-0.1, 0.1))
    
    # Adjust contrast to preserve skin tone details
    img_tensor = tf.image.adjust_contrast(img_tensor, contrast_factor=random.uniform(0.9, 1.1))
    
    # Adjust saturation to enhance skin tone
    img_tensor = tf.image.adjust_saturation(img_tensor, saturation_factor=random.uniform(0.9, 1.1))
    
    # Random flip (50% chance)
    if random.random() > 0.5:
        img_tensor = tf.image.flip_left_right(img_tensor)
    
    # Convert back to numpy
    augmented = img_tensor.numpy()
    
    return augmented

def prepare_dataset(input_dir, output_dir, is_indian=False, augment_count=5):
    """
    Prepare dataset by extracting faces and applying augmentation
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
        is_indian: Whether the dataset contains Indian faces
        augment_count: Number of augmented versions to create per image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    
    for root, _, filenames in os.walk(input_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                image_files.append(os.path.join(root, filename))
    
    print(f"Processing {len(image_files)} images from {input_dir}...")
    
    # Process each image
    for image_path in tqdm(image_files):
        try:
            # Extract face
            face_img = detect_and_extract_face(image_path)
            
            if face_img is None:
                continue
            
            # Save original face
            filename = os.path.basename(image_path)
            base_name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{base_name}_face{ext}")
            cv2.imwrite(output_path, face_img)
            
            # Create augmented versions
            for i in range(augment_count):
                # Apply appropriate augmentation
                if is_indian:
                    augmented = augment_indian_faces(face_img)
                else:
                    augmented = augment_image(face_img)
                
                # Save augmented image
                aug_path = os.path.join(output_dir, f"{base_name}_aug{i}{ext}")
                cv2.imwrite(aug_path, augmented)
        
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        data_dir: Directory containing the dataset
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
    """
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
    
    # Create output directories
    train_dir = os.path.join(os.path.dirname(data_dir), "train")
    val_dir = os.path.join(os.path.dirname(data_dir), "validation")
    test_dir = os.path.join(os.path.dirname(data_dir), "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    
    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                image_files.append(os.path.join(root, filename))
    
    # Split the dataset
    train_files, temp_files = train_test_split(image_files, test_size=(val_ratio + test_ratio))
    val_files, test_files = train_test_split(temp_files, test_size=test_ratio/(val_ratio + test_ratio))
    
    print(f"Splitting dataset: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")
    
    # Copy files to respective directories
    for src_file in tqdm(train_files, desc="Copying train files"):
        dst_file = os.path.join(train_dir, os.path.basename(src_file))
        shutil.copy2(src_file, dst_file)
    
    for src_file in tqdm(val_files, desc="Copying validation files"):
        dst_file = os.path.join(val_dir, os.path.basename(src_file))
        shutil.copy2(src_file, dst_file)
    
    for src_file in tqdm(test_files, desc="Copying test files"):
        dst_file = os.path.join(test_dir, os.path.basename(src_file))
        shutil.copy2(src_file, dst_file)

def download_indian_faces_dataset():
    """
    Download a dataset of Indian faces
    
    Note: This is a placeholder. In a real implementation, you would:
    1. Use a library like gdown to download from Google Drive
    2. Or use requests to download from a public dataset repository
    3. Or provide instructions for manual download
    """
    print("To download Indian faces dataset:")
    print("1. Visit https://www.kaggle.com/datasets/ashwingupta3012/human-faces")
    print("2. Download the dataset")
    print("3. Extract the files to data/indian_faces/real")
    
    # For demonstration, let's create a small sample dataset
    os.makedirs("data/indian_faces/real", exist_ok=True)
    os.makedirs("data/indian_faces/fake", exist_ok=True)
    
    print("\nPlease download the dataset manually and place it in the appropriate directory.")
    print("After downloading, run this script again with the --prepare flag.")

def main():
    """
    Main function to prepare datasets
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Data preparation for DeepDefend")
    parser.add_argument("--download", action="store_true", help="Download datasets")
    parser.add_argument("--prepare", action="store_true", help="Prepare and augment datasets")
    parser.add_argument("--split", action="store_true", help="Split dataset into train/val/test")
    
    args = parser.parse_args()
    
    if args.download:
        download_indian_faces_dataset()
    
    if args.prepare:
        # Prepare real Indian faces
        prepare_dataset(
            input_dir="data/indian_faces/real",
            output_dir="data/processed/indian_faces/real",
            is_indian=True,
            augment_count=5
        )
        
        # Prepare fake Indian faces
        prepare_dataset(
            input_dir="data/indian_faces/fake",
            output_dir="data/processed/indian_faces/fake",
            is_indian=True,
            augment_count=5
        )
    
    if args.split:
        # Split real dataset
        split_dataset("data/processed/indian_faces/real")
        
        # Split fake dataset
        split_dataset("data/processed/indian_faces/fake")

if __name__ == "__main__":
    main()