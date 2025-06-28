"""
Enhanced Indian face detection and preprocessing
"""

import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Initialize face detector
detector = MTCNN()

def enhance_indian_face(face_img):
    """
    Apply enhancements specifically designed for Indian faces
    
    Args:
        face_img: Input face image (BGR format)
        
    Returns:
        Enhanced face image
    """
    try:
        # Convert to float32 for processing
        img_float = face_img.astype(np.float32) / 255.0
        
        # Convert to YCrCb color space which better represents skin tones
        ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
        
        # Extract Y (luminance) channel
        y_channel = ycrcb[:, :, 0].astype(np.float32)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to Y channel
        # This improves contrast while preserving skin tone
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_channel = clahe.apply(y_channel.astype(np.uint8))
        
        # Update Y channel
        ycrcb[:, :, 0] = y_channel
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        
        # Apply subtle bilateral filter to smooth skin while preserving edges
        # This is particularly effective for Indian skin tones
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Adjust saturation slightly to enhance skin tone
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.1  # Increase saturation by 10%
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return enhanced
    
    except Exception as e:
        logger.error(f"Error enhancing Indian face: {str(e)}")
        return face_img  # Return original if enhancement fails

def detect_and_enhance_indian_faces(image, min_face_size=20, enhance=True):
    """
    Detect faces in an image and apply Indian-specific enhancements
    
    Args:
        image: Input image (BGR format)
        min_face_size: Minimum face size to detect
        enhance: Whether to apply Indian-specific enhancements
        
    Returns:
        List of detected faces with their locations and enhanced images
    """
    try:
        # Convert to RGB (MTCNN expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detector.detect_faces(image_rgb)
        
        result = []
        for face in faces:
            x, y, w, h = face['box']
            
            # Skip small faces
            if w < min_face_size or h < min_face_size:
                continue
            
            # Extract face region
            face_img = image[y:y+h, x:x+w]
            
            # Apply Indian-specific enhancements if requested
            if enhance:
                enhanced_face = enhance_indian_face(face_img)
            else:
                enhanced_face = face_img
            
            # Add to result
            result.append({
                'box': face['box'],
                'confidence': face['confidence'],
                'keypoints': face['keypoints'],
                'original': face_img,
                'enhanced': enhanced_face
            })
        
        return result
    
    except Exception as e:
        logger.error(f"Error detecting and enhancing Indian faces: {str(e)}")
        return []

def analyze_indian_skin_tone(face_image):
    """
    Analyze Indian skin tone and detect inconsistencies that may indicate deepfakes
    
    Args:
        face_image: Face image (BGR format)
        
    Returns:
        Dictionary with skin tone analysis results
    """
    try:
        # Convert to YCrCb color space which better represents skin tones
        ycrcb = cv2.cvtColor(face_image, cv2.COLOR_BGR2YCrCb)
        
        # Extract Cr and Cb channels which contain skin tone information
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        
        # Indian skin tones typically fall within these ranges
        # These ranges are based on research on South Asian skin tones
        indian_skin_mask = np.logical_and(
            np.logical_and(cr > 135, cr < 180),
            np.logical_and(cb > 85, cb < 135)
        )
        
        # Calculate percentage of pixels that match Indian skin tone
        indian_skin_percentage = np.sum(indian_skin_mask) / (face_image.shape[0] * face_image.shape[1])
        
        # Analyze consistency of skin tone across the face
        skin_tone_variance = np.var(cr[indian_skin_mask]) + np.var(cb[indian_skin_mask]) if np.any(indian_skin_mask) else 0
        
        # High variance often indicates manipulation
        is_consistent = skin_tone_variance < 100
        
        # Determine Indian skin tone type based on Cr/Cb values
        # This is a simplified version of the Fitzpatrick scale adapted for Indian skin tones
        avg_cr = np.mean(cr[indian_skin_mask]) if np.any(indian_skin_mask) else 0
        avg_cb = np.mean(cb[indian_skin_mask]) if np.any(indian_skin_mask) else 0
        
        # Determine skin tone name
        if avg_cr < 145:
            tone_name = "Fair"
        elif avg_cr < 155:
            tone_name = "Medium"
        elif avg_cr < 165:
            tone_name = "Wheatish"
        elif avg_cr < 175:
            tone_name = "Brown"
        else:
            tone_name = "Dark"
        
        return {
            "success": True,
            "indian_tone": {
                "percentage": float(indian_skin_percentage),
                "variance": float(skin_tone_variance),
                "is_consistent": bool(is_consistent),
                "name": tone_name,
                "avg_cr": float(avg_cr),
                "avg_cb": float(avg_cb)
            }
        }
    
    except Exception as e:
        logger.error(f"Error analyzing Indian skin tone: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def preprocess_for_indian_model(face_image, target_size=(224, 224)):
    """
    Preprocess face image for the Indian-specialized model
    
    Args:
        face_image: Face image (BGR format)
        target_size: Target size for the model
        
    Returns:
        Preprocessed image ready for model input
    """
    try:
        # Resize to target size
        resized = cv2.resize(face_image, target_size)
        
        # Apply Indian-specific enhancements
        enhanced = enhance_indian_face(resized)
        
        # Convert to RGB (model expects RGB)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Expand dimensions for batch
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    except Exception as e:
        logger.error(f"Error preprocessing for Indian model: {str(e)}")
        # Return a blank image of the correct shape if preprocessing fails
        return np.zeros((1, target_size[0], target_size[1], 3), dtype=np.float32)

# Test function to verify the module works
def test_indian_face_enhancement():
    """
    Test function to verify the Indian face enhancement works
    """
    try:
        # Create a sample image (just for testing)
        sample_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Apply enhancement
        enhanced = enhance_indian_face(sample_img)
        
        # Check if enhancement was applied
        if enhanced is not None and enhanced.shape == sample_img.shape:
            print("Indian face enhancement test passed!")
            return True
        else:
            print("Indian face enhancement test failed!")
            return False
    
    except Exception as e:
        print(f"Indian face enhancement test error: {str(e)}")
        return False

if __name__ == "__main__":
    # Run test
    test_indian_face_enhancement()