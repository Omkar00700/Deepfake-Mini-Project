"""
Test script for Deepfake Detection Pipeline
"""

import os
import logging
import json
from deepfake_detector import DeepfakeDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_deepfake_detector():
    """
    Test the Deepfake Detector
    """
    try:
        # Create detector
        detector = DeepfakeDetector(use_indian_enhancement=True)
        
        # Create test directory if it doesn't exist
        test_dir = os.path.join(os.path.dirname(__file__), "test_results")
        os.makedirs(test_dir, exist_ok=True)
        
        # Test with image
        test_image_path = "test_image.jpg"
        if os.path.exists(test_image_path):
            # Detect deepfakes
            result = detector.detect_image(test_image_path)
            
            # Print results
            if result["success"]:
                print(f"Image Detection ID: {result['detection_id']}")
                print(f"Probability: {result['result']['probability']:.4f}")
                print(f"Confidence: {result['result']['confidence']:.4f}")
                print(f"Processing Time: {result['result']['processingTime']:.4f} seconds")
                print(f"Regions: {len(result['result']['regions'])}")
                
                # Save result to test directory
                result_path = os.path.join(test_dir, "image_detection_result.json")
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"Saved image detection result to {result_path}")
            else:
                print(f"Error: {result['error']}")
        else:
            print(f"Test image not found: {test_image_path}")
        
        # Test with face image
        test_face_path = "test_face.jpg"
        if not os.path.exists(test_face_path):
            test_face_path = os.path.join("test_faces", "test_face.jpg")
        
        if os.path.exists(test_face_path):
            # Detect deepfakes
            result = detector.detect_image(test_face_path)
            
            # Print results
            if result["success"]:
                print(f"\nFace Detection ID: {result['detection_id']}")
                print(f"Probability: {result['result']['probability']:.4f}")
                print(f"Confidence: {result['result']['confidence']:.4f}")
                print(f"Processing Time: {result['result']['processingTime']:.4f} seconds")
                print(f"Regions: {len(result['result']['regions'])}")
                
                # Print skin tone information
                for i, region in enumerate(result['result']['regions']):
                    if region['skin_tone']['success']:
                        tone = region['skin_tone']['indian_tone']
                        print(f"Region {i+1} Skin Tone: {tone['name']} (score: {tone['score']:.4f})")
                
                # Save result to test directory
                result_path = os.path.join(test_dir, "face_detection_result.json")
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"Saved face detection result to {result_path}")
            else:
                print(f"Error: {result['error']}")
        else:
            print(f"Test face image not found: {test_face_path}")
        
        # Test with different models
        models = ["efficientnet", "xception", "indian_specialized"]
        
        for model in models:
            print(f"\nTesting with model: {model}")
            
            # Create detector with current model
            model_detector = DeepfakeDetector(model_name=model, use_indian_enhancement=True)
            
            # Test with face image
            if os.path.exists(test_face_path):
                # Detect deepfakes
                result = model_detector.detect_image(test_face_path)
                
                # Print results
                if result["success"]:
                    print(f"  Probability: {result['result']['probability']:.4f}")
                    print(f"  Confidence: {result['result']['confidence']:.4f}")
                    print(f"  Processing Time: {result['result']['processingTime']:.4f} seconds")
                else:
                    print(f"  Error: {result['error']}")
        
        print("\nTest completed successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error testing deepfake detector: {str(e)}")
        return False

if __name__ == "__main__":
    test_deepfake_detector()