
import unittest
import os
import numpy as np
import cv2
from explainability import explainer
from model_manager import ModelManager

class TestExplanation(unittest.TestCase):
    """
    Test the explainability features
    """
    
    def setUp(self):
        # Set up test data
        self.test_image_path = "test_data/test_image.jpg"
        
        # Create test directory and image if they don't exist
        os.makedirs("test_data", exist_ok=True)
        
        # If test image doesn't exist, create a dummy one
        if not os.path.exists(self.test_image_path):
            # Create a simple test image (200x200 with a face-like shape)
            img = np.zeros((200, 200, 3), np.uint8)
            # Draw a circle for the face
            cv2.circle(img, (100, 100), 80, (200, 200, 200), -1)
            # Draw eyes
            cv2.circle(img, (70, 80), 10, (255, 255, 255), -1)
            cv2.circle(img, (130, 80), 10, (255, 255, 255), -1)
            # Draw mouth
            cv2.ellipse(img, (100, 120), (30, 20), 0, 0, 180, (255, 255, 255), -1)
            # Save the image
            cv2.imwrite(self.test_image_path, img)
        
        # Load the test image
        self.image = cv2.imread(self.test_image_path)
        
        # Create a face coordinate (assume the whole image is a face for simplicity)
        self.face_coords = (0, 0, 200, 200)
    
    def test_grad_cam_explanation(self):
        """Test Grad-CAM explanation generation"""
        # Generate explanation
        explanation = explainer.explain_prediction(self.image, self.face_coords, method="grad_cam")
        
        # Check if explanation was generated successfully
        self.assertTrue(explanation["success"], "Explanation generation failed")
        self.assertEqual(explanation["method"], "grad_cam", "Wrong explanation method")
        
        # Check if visualization is present
        self.assertIn("visualization", explanation, "Visualization missing in explanation")
        self.assertIn("heatmap_base64", explanation["visualization"], "Heatmap missing in visualization")
        
        # Check if explanation data is present
        self.assertIn("explanation_data", explanation, "Explanation data missing")
        self.assertIn("top_activation_regions", explanation["explanation_data"], "Top activation regions missing")
        self.assertIn("regions_of_interest", explanation["explanation_data"], "Regions of interest missing")
    
    def test_integrated_gradients_explanation(self):
        """Test Integrated Gradients explanation generation"""
        # Generate explanation
        explanation = explainer.explain_prediction(
            self.image, self.face_coords, method="integrated_gradients"
        )
        
        # Check if explanation was generated successfully
        self.assertTrue(explanation["success"], "Explanation generation failed")
        self.assertEqual(explanation["method"], "integrated_gradients", "Wrong explanation method")
        
        # Check if visualization is present
        self.assertIn("visualization", explanation, "Visualization missing in explanation")
        self.assertIn("heatmap_base64", explanation["visualization"], "Heatmap missing in visualization")
        
        # Check if explanation data is present
        self.assertIn("explanation_data", explanation, "Explanation data missing")
        self.assertIn("top_attribution_regions", explanation["explanation_data"], "Top attribution regions missing")
        self.assertIn("regions_of_interest", explanation["explanation_data"], "Regions of interest missing")
    
    def test_invalid_explanation_method(self):
        """Test with invalid explanation method"""
        # Generate explanation with invalid method
        explanation = explainer.explain_prediction(
            self.image, self.face_coords, method="invalid_method"
        )
        
        # Check if explanation failed
        self.assertFalse(explanation["success"], "Invalid method should fail")
        self.assertEqual(explanation["method"], "invalid_method", "Wrong explanation method")
        self.assertIn("message", explanation, "Error message missing")
    
    def test_explanation_performance(self):
        """Test explanation performance"""
        import time
        
        # Time Grad-CAM explanation
        start_time = time.time()
        explainer.explain_prediction(self.image, self.face_coords, method="grad_cam")
        grad_cam_time = time.time() - start_time
        
        # Time Integrated Gradients explanation
        start_time = time.time()
        explainer.explain_prediction(self.image, self.face_coords, method="integrated_gradients")
        integrated_gradients_time = time.time() - start_time
        
        # Log performance
        print(f"Grad-CAM explanation time: {grad_cam_time:.3f}s")
        print(f"Integrated Gradients explanation time: {integrated_gradients_time:.3f}s")
        
        # Check if explanation time is reasonable (adjust thresholds as needed)
        self.assertLess(grad_cam_time, 10.0, "Grad-CAM explanation took too long")
        self.assertLess(integrated_gradients_time, 10.0, "Integrated Gradients explanation took too long")

if __name__ == '__main__':
    unittest.main()
