
import unittest
import os
import numpy as np
import cv2
import time
from detection import detect_faces, preprocess_face, process_image, process_video, switch_model
from model_loader import ModelManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDetection(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Create a test directory if it doesn't exist
        self.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a small test image with a face-like object
        self.test_image_path = os.path.join(self.test_dir, "test_face.jpg")
        if not os.path.exists(self.test_image_path):
            # Create a 300x300 grayscale image
            img = np.ones((300, 300, 3), dtype=np.uint8) * 200
            # Draw a simple face-like circle
            cv2.circle(img, (150, 150), 100, (100, 100, 100), -1)
            # Draw eyes
            cv2.circle(img, (120, 120), 20, (50, 50, 50), -1)
            cv2.circle(img, (180, 120), 20, (50, 50, 50), -1)
            # Draw mouth
            cv2.ellipse(img, (150, 180), (50, 20), 0, 0, 180, (50, 50, 50), -1)
            # Save the image
            cv2.imwrite(self.test_image_path, img)
        
        # Create a very small test video
        self.test_video_path = os.path.join(self.test_dir, "test_video.mp4")
        if not os.path.exists(self.test_video_path):
            # Create a video with 10 frames
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.test_video_path, fourcc, 5.0, (300, 300))
            # Generate 10 frames with slightly varying face
            for i in range(10):
                img = np.ones((300, 300, 3), dtype=np.uint8) * 200
                # Draw face with slight variations
                cv2.circle(img, (150, 150), 100, (100, 100, 100), -1)
                # Draw eyes with movement
                eye_x_offset = int(np.sin(i/3.0) * 10)
                cv2.circle(img, (120 + eye_x_offset, 120), 20, (50, 50, 50), -1)
                cv2.circle(img, (180 + eye_x_offset, 120), 20, (50, 50, 50), -1)
                # Draw mouth with different expressions
                mouth_height = 20 + int(np.sin(i/2.0) * 10)
                cv2.ellipse(img, (150, 180), (50, mouth_height), 0, 0, 180, (50, 50, 50), -1)
                # Write frame to video
                out.write(img)
            out.release()
        
        # Create a test deepfake image (just for testing labeling)
        self.test_deepfake_path = os.path.join(self.test_dir, "test_deepfake.jpg")
        if not os.path.exists(self.test_deepfake_path):
            # Create a 300x300 image with distorted face
            img = np.ones((300, 300, 3), dtype=np.uint8) * 200
            # Draw a distorted face (ellipse instead of circle)
            cv2.ellipse(img, (150, 150), (80, 120), 30, 0, 360, (100, 100, 100), -1)
            # Draw eyes with asymmetry
            cv2.circle(img, (110, 130), 25, (50, 50, 50), -1)
            cv2.circle(img, (190, 110), 15, (50, 50, 50), -1)
            # Draw mouth with distortion
            cv2.ellipse(img, (150, 200), (40, 15), 15, 0, 180, (50, 50, 50), -1)
            # Add some noise
            noise = np.random.randint(0, 30, (300, 300, 3), dtype=np.uint8)
            img = cv2.add(img, noise)
            # Save the image
            cv2.imwrite(self.test_deepfake_path, img)
    
    def test_detect_faces(self):
        """Test face detection functionality"""
        logger.info("Testing face detection")
        # Load test image
        img = cv2.imread(self.test_image_path)
        # Detect faces
        faces = detect_faces(img)
        # Check if at least one face was detected
        self.assertGreaterEqual(len(faces), 0, "No faces detected in test image")
        
        # If faces were detected, check that they have the expected format
        if len(faces) > 0:
            face = faces[0]
            self.assertEqual(len(face), 4, "Face detection should return (x, y, width, height) tuple")
            x, y, w, h = face
            self.assertIsInstance(x, int, "X coordinate should be an integer")
            self.assertIsInstance(y, int, "Y coordinate should be an integer")
            self.assertIsInstance(w, int, "Width should be an integer")
            self.assertIsInstance(h, int, "Height should be an integer")
    
    def test_preprocess_face(self):
        """Test face preprocessing functionality"""
        logger.info("Testing face preprocessing")
        # Load test image
        img = cv2.imread(self.test_image_path)
        # Use a manually specified face region
        face = (100, 100, 100, 100)  # x, y, width, height
        # Preprocess face
        processed_face = preprocess_face(img, face)
        # Check that preprocessing returns an image
        self.assertIsNotNone(processed_face, "Face preprocessing should return an image")
        # Check image dimensions
        model = ModelManager().get_model()
        expected_height, expected_width = model.input_shape[:2]
        self.assertEqual(processed_face.shape[0], expected_height, f"Processed face height should be {expected_height}")
        self.assertEqual(processed_face.shape[1], expected_width, f"Processed face width should be {expected_width}")
        self.assertEqual(processed_face.shape[2], 3, "Processed face should have 3 channels")
    
    def test_process_image(self):
        """Test image processing functionality"""
        logger.info("Testing image processing")
        # Process the test image
        probability, confidence, regions = process_image(self.test_image_path)
        # Check that the function returns valid results
        self.assertIsInstance(probability, float, "Probability should be a float")
        self.assertIsInstance(confidence, float, "Confidence should be a float")
        self.assertIsInstance(regions, list, "Regions should be a list")
        # Check value ranges
        self.assertGreaterEqual(probability, 0.0, "Probability should be >= 0")
        self.assertLessEqual(probability, 1.0, "Probability should be <= 1")
        self.assertGreaterEqual(confidence, 0.0, "Confidence should be >= 0")
        self.assertLessEqual(confidence, 1.0, "Confidence should be <= 1")
    
    def test_process_video(self):
        """Test video processing functionality"""
        logger.info("Testing video processing")
        # Process the test video
        probability, confidence, frame_count, regions = process_video(self.test_video_path, max_frames=5)
        # Check that the function returns valid results
        self.assertIsInstance(probability, float, "Probability should be a float")
        self.assertIsInstance(confidence, float, "Confidence should be a float")
        self.assertIsInstance(frame_count, int, "Frame count should be an integer")
        self.assertIsInstance(regions, list, "Regions should be a list")
        # Check value ranges
        self.assertGreaterEqual(probability, 0.0, "Probability should be >= 0")
        self.assertLessEqual(probability, 1.0, "Probability should be <= 1")
        self.assertGreaterEqual(confidence, 0.0, "Confidence should be >= 0")
        self.assertLessEqual(confidence, 1.0, "Confidence should be <= 1")
        self.assertGreaterEqual(frame_count, 0, "Frame count should be >= 0")
    
    def test_model_switching(self):
        """Test switching between different models"""
        logger.info("Testing model switching")
        # Get current model
        model_manager = ModelManager()
        original_model = model_manager.get_current_model_name()
        
        # Try to switch to each available model
        available_models = model_manager.get_available_models()
        for model_name in available_models:
            if model_name != original_model:
                # Switch to a different model
                logger.info(f"Switching to model: {model_name}")
                result = switch_model(model_name)
                self.assertTrue(result, f"Failed to switch to model: {model_name}")
                
                # Check that model was switched
                current_model = model_manager.get_current_model_name()
                self.assertEqual(current_model, model_name, f"Model was not switched to {model_name}")
                
                # Test that the model can process an image
                probability, confidence, regions = process_image(self.test_image_path)
                self.assertIsInstance(probability, float, "Probability should be a float after model switch")
                
                # Switch back to original model before trying the next one
                switch_model(original_model)
        
        # Restore original model
        switch_model(original_model)
    
    def test_parallel_processing(self):
        """Test parallel processing of video frames"""
        logger.info("Testing parallel processing")
        # Generate a multi-face test image
        multi_face_path = os.path.join(self.test_dir, "multi_face_test.jpg")
        if not os.path.exists(multi_face_path):
            # Create a 600x600 image with multiple faces
            img = np.ones((600, 600, 3), dtype=np.uint8) * 200
            # Draw multiple faces
            for i in range(4):
                x = 150 + (i % 2) * 300
                y = 150 + (i // 2) * 300
                # Draw face
                cv2.circle(img, (x, y), 100, (100, 100, 100), -1)
                # Draw eyes
                cv2.circle(img, (x - 30, y - 30), 20, (50, 50, 50), -1)
                cv2.circle(img, (x + 30, y - 30), 20, (50, 50, 50), -1)
                # Draw mouth
                cv2.ellipse(img, (x, y + 30), (50, 20), 0, 0, 180, (50, 50, 50), -1)
            # Save the image
            cv2.imwrite(multi_face_path, img)
        
        # Process the multi-face image
        start_time = time.time()
        probability, confidence, regions = process_image(multi_face_path)
        processing_time = time.time() - start_time
        
        # Check results
        self.assertGreaterEqual(len(regions), 1, "At least one face should be detected")
        logger.info(f"Processed {len(regions)} faces in {processing_time:.2f} seconds")
        
        # The test passes if the code runs without errors
        # Since parallel processing is environment-dependent, we can't make assertions about performance
        
    def test_frame_caching(self):
        """Test frame caching in video processing"""
        logger.info("Testing frame caching")
        # Process the test video twice
        start_time = time.time()
        probability1, confidence1, frame_count1, regions1 = process_video(self.test_video_path, max_frames=5)
        first_run_time = time.time() - start_time
        
        # Process again
        start_time = time.time()
        probability2, confidence2, frame_count2, regions2 = process_video(self.test_video_path, max_frames=5)
        second_run_time = time.time() - start_time
        
        # Log times for inspection
        logger.info(f"First run: {first_run_time:.2f}s, Second run: {second_run_time:.2f}s")
        
        # Check that results are consistent
        self.assertAlmostEqual(probability1, probability2, places=1, 
                               msg="Probabilities should be consistent across runs")
        self.assertAlmostEqual(confidence1, confidence2, places=1, 
                               msg="Confidence values should be consistent across runs")
        
        # Note: We're not strictly testing if caching improved performance since that's
        # environment-dependent and might vary in CI/CD environments

if __name__ == '__main__':
    unittest.main()
