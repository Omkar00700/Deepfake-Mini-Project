
"""
Indian Face Utilities for DeepDefend
This module provides specialized utilities for detecting and preprocessing Indian faces
with greater accuracy based on region-specific characteristics
"""

import cv2
import numpy as np
import logging
import os
from typing import List, Tuple, Dict, Any, Optional
from face_detector import FaceDetector
from skin_tone_analyzer import SkinToneAnalyzer

# Configure logging
logger = logging.getLogger(__name__)

class IndianFacePreprocessor:
    """
    Enhanced face preprocessor with specializations for Indian facial features
    and common imaging conditions in Indian contexts
    """
    
    def __init__(self, face_detector: FaceDetector):
        """
        Initialize the Indian face preprocessor
        
        Args:
            face_detector: Face detector instance to use
        """
        self.face_detector = face_detector
        
        # Initialize skin tone analyzer
        self.skin_tone_analyzer = SkinToneAnalyzer()
        
        # Parameters specific to Indian face detection
        self.contrast_adjustment = 1.2  # Slightly boost contrast for better feature detection
        self.brightness_adjustment = 10  # Slight brightness increase for darker skin tones
        self.noise_reduction_strength = 5  # Stronger noise reduction for low-light conditions
        
        # Parameters for different skin tones
        self.skin_tone_params = {
            "fair": {
                "contrast": 1.1,
                "brightness": 5,
                "noise_reduction": 3
            },
            "wheatish": {
                "contrast": 1.2,
                "brightness": 8,
                "noise_reduction": 4
            },
            "medium": {
                "contrast": 1.25,
                "brightness": 10,
                "noise_reduction": 5
            },
            "dusky": {
                "contrast": 1.3,
                "brightness": 15,
                "noise_reduction": 6
            },
            "dark": {
                "contrast": 1.35,
                "brightness": 20,
                "noise_reduction": 7
            }
        }
        
        logger.info("Initialized Indian face preprocessor with specialized parameters")
    
    def enhance_image(self, image: np.ndarray, skin_tone: str = None) -> np.ndarray:
        """
        Enhance image quality for better face detection in varied lighting conditions
        common in Indian contexts
        
        Args:
            image: Input image
            skin_tone: Optional skin tone to optimize enhancement parameters
            
        Returns:
            Enhanced image
        """
        # Get parameters based on skin tone
        if skin_tone and skin_tone in self.skin_tone_params:
            params = self.skin_tone_params[skin_tone]
            contrast = params["contrast"]
            brightness = params["brightness"]
            noise_reduction = params["noise_reduction"]
        else:
            contrast = self.contrast_adjustment
            brightness = self.brightness_adjustment
            noise_reduction = self.noise_reduction_strength
        
        # Convert to LAB color space for better brightness/contrast adjustments
        if len(image.shape) == 3 and image.shape[2] == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # More effective for varied skin tones
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Merge channels back
            limg = cv2.merge((cl, a, b))
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        else:
            # For grayscale or other formats
            enhanced = image.copy()
            
        # Apply additional noise reduction for low-quality images
        enhanced = cv2.fastNlMeansDenoisingColored(
            enhanced, 
            None, 
            noise_reduction, 
            noise_reduction, 
            7, 
            21
        )
        
        return enhanced
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces with enhancements for Indian facial features
        
        Args:
            image: Input image
            
        Returns:
            List of (x, y, width, height) face bounding boxes
        """
        # Enhance image for better face detection
        enhanced_image = self.enhance_image(image)
        
        # Detect faces using the enhanced image
        faces = self.face_detector.detect_faces(enhanced_image)
        
        # If no faces detected, try with original image
        if not faces:
            logger.debug("No faces detected with enhanced image, trying original")
            faces = self.face_detector.detect_faces(image)
        
        return faces
    
    def preprocess_face(self, 
                        image: np.ndarray, 
                        face: Tuple[int, int, int, int], 
                        target_size: Tuple[int, int],
                        margin_percent: float = 0.2) -> Optional[Dict[str, Any]]:
        """
        Extract and preprocess a face for deepfake detection with specialized
        adjustments for Indian facial features
        
        Args:
            image: Source image
            face: Face coordinates (x, y, width, height)
            target_size: Target size for the processed face
            margin_percent: Percentage of margin to add around the face
            
        Returns:
            Dictionary with preprocessed face image and metadata, or None if preprocessing failed
        """
        try:
            # Extract face with margin
            x, y, w, h = face
            
            # Calculate margin
            margin_x = int(w * margin_percent)
            margin_y = int(h * margin_percent)
            
            # Calculate coordinates with margin
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(image.shape[1], x + w + margin_x)
            y2 = min(image.shape[0], y + h + margin_y)
            
            # Extract face region
            face_img = image[y1:y2, x1:x2]
            
            # Skip if face extraction failed
            if face_img.size == 0:
                logger.warning("Face extraction failed: empty region")
                return None
            
            # Analyze skin tone
            skin_tone_result = self.skin_tone_analyzer.analyze_skin_tone(face_img)
            
            # Get skin tone type for enhancement
            skin_tone_type = None
            if skin_tone_result["success"] and skin_tone_result["indian_tone"]:
                skin_tone_type = skin_tone_result["indian_tone"]["type"]
            
            # Apply contrast and brightness adjustments based on skin tone
            if skin_tone_type and skin_tone_type in self.skin_tone_params:
                params = self.skin_tone_params[skin_tone_type]
                face_img = cv2.convertScaleAbs(
                    face_img, 
                    alpha=params["contrast"], 
                    beta=params["brightness"]
                )
            else:
                face_img = cv2.convertScaleAbs(
                    face_img, 
                    alpha=self.contrast_adjustment, 
                    beta=self.brightness_adjustment
                )
            
            # Resize to target size
            face_img = cv2.resize(face_img, target_size)
            
            # Check for skin anomalies
            skin_anomalies = self.skin_tone_analyzer.detect_skin_anomalies(face_img)
            
            # Return face image and metadata
            return {
                "image": face_img,
                "bbox": (x, y, w, h),
                "bbox_with_margin": (x1, y1, x2-x1, y2-y1),
                "skin_tone": skin_tone_result,
                "skin_anomalies": skin_anomalies
            }
            
        except Exception as e:
            logger.error(f"Error in face preprocessing: {str(e)}")
            return None
    
    def batch_process(self, 
                     image: np.ndarray, 
                     target_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Detect and preprocess all faces in an image
        
        Args:
            image: Source image
            target_size: Target size for the processed faces
            
        Returns:
            List of dictionaries with preprocessed face images and metadata
        """
        # Detect faces
        faces = self.detect_faces(image)
        
        # Preprocess each face
        processed_faces = []
        for face in faces:
            processed_face = self.preprocess_face(image, face, target_size)
            if processed_face is not None:
                processed_faces.append(processed_face)
        
        return processed_faces
    
    def analyze_face_authenticity(self, face_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze face authenticity based on skin tone and anomalies
        
        Args:
            face_data: Dictionary with face image and metadata
            
        Returns:
            Dictionary with authenticity analysis results
        """
        # Initialize result
        result = {
            "authenticity_score": 1.0,  # 1.0 = authentic, 0.0 = fake
            "confidence": 0.8,
            "factors": []
        }
        
        # Check if we have skin anomalies
        if "skin_anomalies" in face_data and face_data["skin_anomalies"]["success"]:
            anomalies = face_data["skin_anomalies"]["anomalies"]
            anomaly_score = face_data["skin_anomalies"]["anomaly_score"]
            
            # Adjust authenticity score based on anomalies
            if anomaly_score > 0:
                # More anomalies = lower authenticity score
                authenticity_reduction = min(0.5, anomaly_score * 0.7)
                result["authenticity_score"] -= authenticity_reduction
                
                # Add factors
                for anomaly in anomalies:
                    result["factors"].append({
                        "type": "skin_anomaly",
                        "description": anomaly["description"],
                        "impact": -anomaly["severity"] * 0.7
                    })
        
        # Check skin tone distribution
        if "skin_tone" in face_data and face_data["skin_tone"]["success"]:
            # Get skin tone evenness
            evenness_score = face_data["skin_tone"].get("evenness_score", 1.0)
            
            # Uneven skin tone can indicate deepfakes
            if evenness_score < 0.7:
                authenticity_reduction = (1.0 - evenness_score) * 0.3
                result["authenticity_score"] -= authenticity_reduction
                
                result["factors"].append({
                    "type": "uneven_skin_tone",
                    "description": "Uneven skin tone distribution",
                    "impact": -authenticity_reduction
                })
        
        # Ensure score is in valid range
        result["authenticity_score"] = max(0.0, min(1.0, result["authenticity_score"]))
        
        # Adjust confidence based on available data
        if not face_data.get("skin_tone", {}).get("success", False):
            result["confidence"] -= 0.2
        
        if not face_data.get("skin_anomalies", {}).get("success", False):
            result["confidence"] -= 0.2
        
        # Ensure confidence is in valid range
        result["confidence"] = max(0.5, min(0.95, result["confidence"]))
        
        return result
