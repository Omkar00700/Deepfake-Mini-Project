
import unittest
import os
import cv2
import numpy as np
import time
from preprocessing import detect_faces, preprocess_face, get_video_metadata, calculate_frame_positions
from inference import process_face_with_retry, get_model_info
from postprocessing import calculate_image_confidence, calculate_video_confidence
from detection_handler import process_image, process_video
from metrics import PerformanceMetrics
from backend.config import TEST_DATA_DIR

class TestModules(unittest.TestCase):
    """
    Unit tests for the refactored deepfake detection modules
    """
    
    def setUp(self):
        """
        Set up test data
        """
        # Create test directory if it doesn't exist
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        
        # Create a test image
        self.test_image_path = os.path.join(TEST_DATA_DIR, "test_image.jpg")
        if not os.path.exists(self.test_image_path):
            # Create a blank image with a simple pattern that might look like a face
            img = np.zeros((500, 500, 3), np.uint8)
            # Draw a circle for a face
            cv2.circle(img, (250, 250), 100, (200, 200, 200), -1)
            # Draw eyes
            cv2.circle(img, (200, 220), 20, (255, 255, 255), -1)
            cv2.circle(img, (300, 220), 20, (255, 255, 255), -1)
            # Draw mouth
            cv2.ellipse(img, (250, 300), (60, 30), 0, 0, 180, (255, 255, 255), -1)
            # Save the image
            cv2.imwrite(self.test_image_path, img)
        
        # Create a test video
        self.test_video_path = os.path.join(TEST_DATA_DIR, "test_video.mp4")
        if not os.path.exists(self.test_video_path):
            # Create a blank video with a simple pattern
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.test_video_path, fourcc, 30.0, (500, 500))
            
            # Create 30 frames
            for i in range(30):
                # Create a blank image with a simple pattern that might look like a face
                img = np.zeros((500, 500, 3), np.uint8)
                # Draw a circle for a face
                cv2.circle(img, (250, 250), 100, (200, 200, 200), -1)
                # Draw eyes
                cv2.circle(img, (200, 220), 20, (255, 255, 255), -1)
                cv2.circle(img, (300, 220), 20, (255, 255, 255), -1)
                # Draw mouth
                cv2.ellipse(img, (250, 300), (60, 30), 0, 0, 180, (255, 255, 255), -1)
                # Add some movement
                cv2.putText(img, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # Write the frame to the video
                out.write(img)
            
            # Release the video writer
            out.release()
    
    def test_preprocessing_face_detection(self):
        """
        Test face detection in preprocessing module
        """
        # Test on the test image
        img = cv2.imread(self.test_image_path)
        faces = detect_faces(img)
        
        # May or may not detect faces in our simple test image, so just check that it runs
        self.assertIsInstance(faces, list)
    
    def test_preprocessing_video_metadata(self):
        """
        Test video metadata extraction in preprocessing module
        """
        # Test on the test video
        metadata = get_video_metadata(self.test_video_path)
        
        # Check if metadata was extracted correctly
        self.assertIsInstance(metadata, dict)
        self.assertIn("total_frames", metadata)
        self.assertIn("fps", metadata)
        self.assertIn("duration", metadata)
        self.assertIn("width", metadata)
        self.assertIn("height", metadata)
        
        # Basic validation
        self.assertGreater(metadata["total_frames"], 0)
        self.assertGreater(metadata["fps"], 0)
        self.assertGreater(metadata["duration"], 0)
        self.assertEqual(metadata["width"], 500)
        self.assertEqual(metadata["height"], 500)
    
    def test_preprocessing_frame_calculation(self):
        """
        Test frame position calculation in preprocessing module
        """
        # Create a simple metadata dictionary
        metadata = {
            "total_frames": 300,
            "fps": 30,
            "duration": 10.0
        }
        
        # Test with default max frames
        positions = calculate_frame_positions(metadata, 10)
        
        # Check that we got the expected number of frames
        self.assertEqual(len(positions), 10)
        
        # Check that frame positions are in ascending order
        self.assertEqual(positions, sorted(positions))
        
        # Check that they're spaced correctly
        if len(positions) >= 2:
            self.assertAlmostEqual(positions[1] - positions[0], 30, delta=5)
    
    def test_inference_model_info(self):
        """
        Test model info retrieval in inference module
        """
        # Get model info
        info = get_model_info()
        
        # Check if the info was retrieved correctly
        self.assertIsInstance(info, dict)
        self.assertIn("face_detector", info)
        self.assertIn("models", info)
        
        # Check models info
        models_info = info["models"]
        self.assertIn("available_models", models_info)
        self.assertIn("current_model", models_info)
    
    def test_image_processing(self):
        """
        Test end-to-end image processing
        """
        # Process the test image
        probability, confidence, regions = process_image(self.test_image_path)
        
        # Check the results
        self.assertIsInstance(probability, float)
        self.assertIsInstance(confidence, float)
        self.assertIsInstance(regions, list)
        
        # Basic validation
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_video_processing(self):
        """
        Test end-to-end video processing
        """
        # Process the test video
        probability, confidence, frame_count, regions = process_video(self.test_video_path, max_frames=5)
        
        # Check the results
        self.assertIsInstance(probability, float)
        self.assertIsInstance(confidence, float)
        self.assertIsInstance(frame_count, int)
        self.assertIsInstance(regions, list)
        
        # Basic validation
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertGreaterEqual(frame_count, 0)
    
    def test_performance_metrics(self):
        """
        Test performance metrics module
        """
        # Get the performance metrics instance
        metrics = PerformanceMetrics()
        
        # Record some sample metrics
        metrics.record_image_metrics(0.5, 2, 0.8, 0.3, "efficientnet")
        metrics.record_image_metrics(0.6, 1, 0.7, 0.6, "efficientnet")
        metrics.record_video_metrics(1.2, 5, 3, 0.6, 0.4, "xception")
        
        # Record an error
        metrics.record_error("test_error")
        
        # Get the metrics
        all_metrics = metrics.get_all_metrics()
        
        # Check that the metrics were recorded correctly
        self.assertIsInstance(all_metrics, dict)
        self.assertIn("images", all_metrics)
        self.assertIn("videos", all_metrics)
        self.assertIn("model_usage", all_metrics)
        self.assertIn("errors", all_metrics)
        
        # Check image metrics
        image_metrics = all_metrics["images"]
        self.assertEqual(image_metrics["count"], 2)
        
        # Check video metrics
        video_metrics = all_metrics["videos"]
        self.assertEqual(video_metrics["count"], 1)
        
        # Check model usage
        model_usage = all_metrics["model_usage"]
        self.assertEqual(model_usage.get("efficientnet", 0), 2)
        self.assertEqual(model_usage.get("xception", 0), 1)
        
        # Check errors
        errors = all_metrics["errors"]
        self.assertEqual(errors.get("test_error", 0), 1)
        
        # Reset metrics
        metrics.reset_metrics()
        
        # Check that metrics were reset
        reset_metrics = metrics.get_all_metrics()
        self.assertEqual(reset_metrics["images"]["count"], 0)
        self.assertEqual(reset_metrics["videos"]["count"], 0)
        self.assertEqual(len(reset_metrics["model_usage"]), 0)
        self.assertEqual(len(reset_metrics["errors"]), 0)

if __name__ == "__main__":
    unittest.main()
