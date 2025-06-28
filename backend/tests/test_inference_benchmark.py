"""
Benchmark tests for inference performance
"""

import os
import sys
import pytest
import numpy as np
import cv2
import time
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import process_face_with_retry
from face_detector import FaceDetector
from model_manager import ModelManager

# Initialize components
face_detector = FaceDetector()
model_manager = ModelManager()

# Create a sample image for testing
def create_test_image(size=(224, 224)):
    """Create a random test image"""
    return np.random.randint(0, 255, (*size, 3), dtype=np.uint8)

@pytest.fixture
def sample_face_image():
    """Create a sample face image for testing"""
    # Try to load a real face image if available
    test_dir = Path(__file__).parent
    sample_path = test_dir / "sample_face.jpg"
    
    if sample_path.exists():
        image = cv2.imread(str(sample_path))
        if image is not None:
            return image
    
    # Fall back to random image
    return create_test_image()

def test_model_loading_time():
    """Test the time it takes to load a model"""
    start_time = time.time()
    model = model_manager.get_model()
    load_time = time.time() - start_time
    
    assert model is not None
    assert load_time < 5.0, f"Model loading took too long: {load_time:.2f} seconds"
    print(f"Model loading time: {load_time:.4f} seconds")

def test_face_detection_time(sample_face_image, benchmark):
    """Benchmark face detection time"""
    def detect():
        return face_detector.detect_faces(sample_face_image)
    
    # Run the benchmark
    result = benchmark(detect)
    
    # Verify that faces were detected
    assert result is not None

def test_inference_time(sample_face_image, benchmark):
    """Benchmark inference time for a single face"""
    # Ensure model is loaded
    model = model_manager.get_model()
    assert model is not None
    
    # Detect faces
    faces = face_detector.detect_faces(sample_face_image)
    
    # If no faces detected in the sample, create a fake face region
    if not faces:
        h, w = sample_face_image.shape[:2]
        faces = [(0, 0, w, h)]
    
    face = faces[0]
    
    def run_inference():
        face_region, prediction = process_face_with_retry(sample_face_image, face)
        return prediction
    
    # Run the benchmark
    result = benchmark(run_inference)
    
    # Verify that we got a prediction
    assert result is not None
    assert hasattr(result, 'probability')
    assert 0 <= result.probability <= 1

def test_batch_inference_latency():
    """Test inference latency for a batch of images"""
    # Create a batch of test images
    batch_size = 10
    images = [create_test_image() for _ in range(batch_size)]
    
    # Ensure model is loaded
    model = model_manager.get_model()
    assert model is not None
    
    # Process each image and measure time
    total_time = 0
    for image in images:
        # Detect faces
        faces = face_detector.detect_faces(image)
        
        # If no faces detected, create a fake face region
        if not faces:
            h, w = image.shape[:2]
            faces = [(0, 0, w, h)]
        
        face = faces[0]
        
        # Measure inference time
        start_time = time.time()
        face_region, prediction = process_face_with_retry(image, face)
        inference_time = time.time() - start_time
        
        total_time += inference_time
    
    avg_time = total_time / batch_size
    print(f"Average inference time per image: {avg_time:.4f} seconds")
    
    # Assert that average inference time is reasonable
    assert avg_time < 1.0, f"Average inference time too high: {avg_time:.4f} seconds"

if __name__ == "__main__":
    # Run tests manually
    test_model_loading_time()
    sample_image = create_test_image()
    
    # Simple timing for face detection
    start_time = time.time()
    faces = face_detector.detect_faces(sample_image)
    detection_time = time.time() - start_time
    print(f"Face detection time: {detection_time:.4f} seconds")
    
    # Simple timing for inference
    if not faces:
        h, w = sample_image.shape[:2]
        faces = [(0, 0, w, h)]
    
    start_time = time.time()
    face_region, prediction = process_face_with_retry(sample_image, faces[0])
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.4f} seconds")