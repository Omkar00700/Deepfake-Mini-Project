"""
Test script for Indian Face Detector
"""

import cv2
import numpy as np
import os
import logging
from indian_face_detector import IndianFaceDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_indian_face_detector():
    """
    Test the Indian Face Detector
    """
    try:
        # Create detector
        detector = IndianFaceDetector()
        
        # Create test directory if it doesn't exist
        test_dir = os.path.join(os.path.dirname(__file__), "test_results")
        os.makedirs(test_dir, exist_ok=True)
        
        # Create a test image with different skin tones
        # We'll create a grid of faces with different skin tones
        grid_size = 3
        face_size = 150
        img_size = face_size * grid_size
        
        # Create a blank image
        test_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        
        # Define skin tones for Indian faces (BGR format)
        skin_tones = [
            (80, 120, 160),  # Fair
            (60, 100, 150),  # Wheatish
            (50, 80, 120),   # Medium
            (40, 60, 100),   # Dusky
            (30, 40, 70)     # Dark
        ]
        
        # Create faces with different skin tones
        for i in range(grid_size):
            for j in range(grid_size):
                # Skip if we run out of skin tones
                tone_idx = i * grid_size + j
                if tone_idx >= len(skin_tones):
                    continue
                
                # Calculate face position
                x1 = j * face_size
                y1 = i * face_size
                x2 = x1 + face_size
                y2 = y1 + face_size
                
                # Create face with current skin tone
                test_img[y1:y2, x1:x2] = skin_tones[tone_idx]
                
                # Add facial features
                # Eyes
                eye_size = face_size // 10
                eye_y = y1 + face_size // 3
                left_eye_x = x1 + face_size // 3
                right_eye_x = x2 - face_size // 3
                
                cv2.circle(test_img, (left_eye_x, eye_y), eye_size, (255, 255, 255), -1)
                cv2.circle(test_img, (right_eye_x, eye_y), eye_size, (255, 255, 255), -1)
                cv2.circle(test_img, (left_eye_x, eye_y), eye_size // 2, (0, 0, 0), -1)
                cv2.circle(test_img, (right_eye_x, eye_y), eye_size // 2, (0, 0, 0), -1)
                
                # Mouth
                mouth_y = y1 + face_size * 2 // 3
                mouth_width = face_size // 2
                mouth_height = face_size // 8
                mouth_x = x1 + face_size // 2
                
                cv2.ellipse(test_img, (mouth_x, mouth_y), (mouth_width // 2, mouth_height), 
                            0, 0, 180, (0, 0, 0), 2)
        
        # Save the test image
        test_img_path = os.path.join(test_dir, "test_skin_tones.jpg")
        cv2.imwrite(test_img_path, test_img)
        logger.info(f"Created test image with different skin tones: {test_img_path}")
        
        # Detect faces
        faces = detector.detect_faces(test_img)
        
        # Print results
        print(f"Detected {len(faces)} faces")
        for i, face in enumerate(faces):
            print(f"Face {i+1}:")
            print(f"  Box: {face['box']}")
            print(f"  Confidence: {face['confidence']:.4f}")
            
            if face['skin_tone']['success']:
                tone = face['skin_tone']['indian_tone']
                print(f"  Skin tone: {tone['name']} (score: {tone['score']:.4f})")
            
            # Draw face on image
            x, y, w, h = face['box']
            cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add skin tone text
            if face['skin_tone']['success']:
                tone = face['skin_tone']['indian_tone']['name']
                cv2.putText(test_img, tone, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save result
        output_path = os.path.join(test_dir, "detected_skin_tones.jpg")
        cv2.imwrite(output_path, test_img)
        logger.info(f"Result saved to {output_path}")
        
        # Test with real test face
        test_face_path = "test_face.jpg"
        if not os.path.exists(test_face_path):
            test_face_path = os.path.join("test_faces", "test_face.jpg")
        
        if os.path.exists(test_face_path):
            # Load image
            real_face = cv2.imread(test_face_path)
            
            # Detect faces
            faces = detector.detect_faces(real_face)
            
            # Print results
            print(f"\nDetected {len(faces)} faces in test face")
            for i, face in enumerate(faces):
                print(f"Face {i+1}:")
                print(f"  Box: {face['box']}")
                print(f"  Confidence: {face['confidence']:.4f}")
                
                if face['skin_tone']['success']:
                    tone = face['skin_tone']['indian_tone']
                    print(f"  Skin tone: {tone['name']} (score: {tone['score']:.4f})")
                
                # Draw face on image
                x, y, w, h = face['box']
                cv2.rectangle(real_face, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Add skin tone text
                if face['skin_tone']['success']:
                    tone = face['skin_tone']['indian_tone']['name']
                    cv2.putText(real_face, tone, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save result
            output_path = os.path.join(test_dir, "detected_test_face.jpg")
            cv2.imwrite(output_path, real_face)
            logger.info(f"Result saved to {output_path}")
        
        print("\nTest completed successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error testing Indian face detector: {str(e)}")
        return False

if __name__ == "__main__":
    test_indian_face_detector()
