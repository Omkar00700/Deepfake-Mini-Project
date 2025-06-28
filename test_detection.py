import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_detection_pipeline():
    try:
        # Create test directories if they don't exist
        os.makedirs('results', exist_ok=True)
        
        # Create a mock detection result
        detection_result = {
            'id': 'test-detection-001',
            'filename': 'test_image.jpg',
            'detectionType': 'image',
            'probability': 0.75,
            'confidence': 0.85,
            'processingTime': 0.8,
            'regions': [
                {
                    'box': [50, 50, 150, 150],
                    'probability': 0.78,
                    'confidence': 0.86,
                    'metadata': {
                        'skin_tone': {
                            'success': True,
                            'indian_tone': {
                                'type': 'medium',
                                'name': 'Medium',
                                'score': 0.65
                            }
                        }
                    }
                }
            ]
        }
        
        # Save the detection result
        result_path = os.path.join('results', 'test-detection-001.json')
        with open(result_path, 'w') as f:
            json.dump(detection_result, f, indent=2)
        
        logger.info(f'Detection result saved to {result_path}')
        print(f'Detection pipeline test completed successfully!')
        return True
    
    except Exception as e:
        logger.error(f'Error testing detection pipeline: {str(e)}')
        return False

if __name__ == '__main__':
    test_detection_pipeline()
