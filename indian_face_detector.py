"""
Indian Face Detector
Specialized detector for Indian faces with different skin tones
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndianFaceDetector:
    def __init__(self, confidence_threshold=0.5, enhance_detection=True):
        """
        Initialize the Indian Face Detector
        
        Args:
            confidence_threshold: Minimum confidence for face detection
            enhance_detection: Whether to use enhanced detection for Indian faces
        """
        self.confidence_threshold = confidence_threshold
        self.enhance_detection = enhance_detection
        
        # Load face detector model
        # We're using OpenCV's DNN face detector which works well for diverse faces
        model_path = os.path.join(os.path.dirname(__file__), "models", "face_detector")
        os.makedirs(model_path, exist_ok=True)
        
        # Check if model files exist, otherwise use OpenCV's default face detector
        prototxt_path = os.path.join(model_path, "deploy.prototxt")
        model_weights = os.path.join(model_path, "res10_300x300_ssd_iter_140000.caffemodel")
        
        if os.path.exists(prototxt_path) and os.path.exists(model_weights):
            self.face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_weights)
            logger.info("Loaded face detection model from disk")
        else:
            # Use Haar cascade as fallback
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            logger.info("Using OpenCV's built-in Haar cascade for face detection")
            self.use_dnn = False
        
        logger.info("Indian Face Detector initialized")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of detected faces with bounding boxes and confidence scores
        """
        try:
            # Make a copy of the image
            img_copy = image.copy()
            
            # Get image dimensions
            h, w = img_copy.shape[:2]
            
            # Preprocess image for better detection of Indian faces
            if self.enhance_detection:
                img_copy = self._preprocess_for_indian_faces(img_copy)
            
            # Detect faces
            faces = []
            
            # Use DNN-based detector if available
            if hasattr(self, 'face_net'):
                # Create a blob from the image
                blob = cv2.dnn.blobFromImage(img_copy, 1.0, (300, 300), [104, 117, 123], False, False)
                
                # Set the blob as input to the network
                self.face_net.setInput(blob)
                
                # Get detections
                detections = self.face_net.forward()
                
                # Process detections
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    
                    if confidence > self.confidence_threshold:
                        # Get bounding box coordinates
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Ensure box is within image boundaries
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        # Calculate width and height
                        width, height = x2 - x1, y2 - y1
                        
                        # Skip invalid boxes
                        if width <= 0 or height <= 0:
                            continue
                        
                        # Add face to list
                        faces.append({
                            'box': [x1, y1, width, height],
                            'confidence': float(confidence),
                            'landmarks': None  # We'll add landmarks later
                        })
            else:
                # Use Haar cascade
                gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
                face_rects = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                for (x, y, w, h) in face_rects:
                    faces.append({
                        'box': [x, y, w, h],
                        'confidence': 0.9,  # Haar cascade doesn't provide confidence, so we use a default value
                        'landmarks': None
                    })
            
            # Sort faces by confidence
            faces = sorted(faces, key=lambda x: x['confidence'], reverse=True)
            
            # Add skin tone analysis for each face
            for face in faces:
                x, y, w, h = face['box']
                face_img = image[y:y+h, x:x+w]
                face['skin_tone'] = self._analyze_skin_tone(face_img)
            
            return faces
        
        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            return []
    
    def _preprocess_for_indian_faces(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better detection of Indian faces
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to float32 for processing
            img_float = image.astype(np.float32) / 255.0
            
            # Apply contrast enhancement
            # This helps with darker skin tones
            alpha = 1.2  # Contrast control (1.0 means no change)
            beta = 10.0  # Brightness control (0 means no change)
            
            # Apply contrast and brightness adjustment
            enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
            # Apply bilateral filter to reduce noise while preserving edges
            # This is particularly helpful for Indian skin textures
            enhanced = cv2.bilateralFilter(enhanced, 5, 75, 75)
            
            return enhanced
        
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return image
    
    def _analyze_skin_tone(self, face_image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the skin tone of a face
        
        Args:
            face_image: Face image in BGR format
            
        Returns:
            Dictionary with skin tone analysis results
        """
        try:
            # Check if face image is valid
            if face_image.size == 0:
                return {
                    'success': False,
                    'error': 'Invalid face image'
                }
            
            # Convert to different color spaces for analysis
            img_hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
            img_ycrcb = cv2.cvtColor(face_image, cv2.COLOR_BGR2YCrCb)
            
            # Create skin mask
            # HSV skin color range for diverse skin tones
            # Wider range to accommodate Indian skin tones
            lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
            upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
            
            # YCrCb skin color range
            lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
            upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
            
            # Create masks
            mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
            mask_ycrcb = cv2.inRange(img_ycrcb, lower_ycrcb, upper_ycrcb)
            
            # Combine masks for better results
            skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
            
            # Apply morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
            skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
            skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
            
            # Extract skin pixels
            skin = cv2.bitwise_and(face_image, face_image, mask=skin_mask)
            
            # Calculate skin percentage
            skin_percentage = np.count_nonzero(skin_mask) / (face_image.shape[0] * face_image.shape[1])
            
            # Calculate average skin color
            skin_pixels = skin[np.where((skin_mask > 0))]
            
            if skin_pixels.size == 0:
                return {
                    'success': False,
                    'error': 'No skin pixels detected'
                }
            
            # Calculate average skin color
            avg_skin_color = np.mean(skin_pixels, axis=0)
            
            # Convert to RGB for better interpretation
            avg_skin_color_rgb = cv2.cvtColor(np.uint8([[avg_skin_color]]), cv2.COLOR_BGR2RGB)[0][0]
            
            # Calculate skin tone darkness score (0 to 1, where 1 is darkest)
            # Convert RGB to L*a*b* color space and use L* channel (lightness)
            avg_skin_color_lab = cv2.cvtColor(np.uint8([[avg_skin_color_rgb]]), cv2.COLOR_RGB2Lab)[0][0]
            lightness = avg_skin_color_lab[0] / 255.0  # L* channel normalized to [0,1]
            darkness_score = 1.0 - lightness
            
            # Determine Indian skin tone
            indian_tone = None
            
            if darkness_score < 0.3:
                indian_tone = {
                    'type': 'fair',
                    'name': 'Fair',
                    'score': float(darkness_score)
                }
            elif darkness_score < 0.45:
                indian_tone = {
                    'type': 'wheatish',
                    'name': 'Wheatish',
                    'score': float(darkness_score)
                }
            elif darkness_score < 0.6:
                indian_tone = {
                    'type': 'medium',
                    'name': 'Medium',
                    'score': float(darkness_score)
                }
            elif darkness_score < 0.75:
                indian_tone = {
                    'type': 'dusky',
                    'name': 'Dusky',
                    'score': float(darkness_score)
                }
            else:
                indian_tone = {
                    'type': 'dark',
                    'name': 'Dark',
                    'score': float(darkness_score)
                }
            
            # Calculate skin tone evenness (standard deviation of skin pixels)
            skin_std = np.std(skin_pixels, axis=0).mean()
            evenness_score = 1.0 - min(1.0, skin_std / 50.0)  # Normalize to [0,1]
            
            return {
                'success': True,
                'skin_percentage': float(skin_percentage),
                'avg_skin_color': {
                    'bgr': avg_skin_color.tolist(),
                    'rgb': avg_skin_color_rgb.tolist(),
                    'hex': '#{:02x}{:02x}{:02x}'.format(*avg_skin_color_rgb)
                },
                'darkness_score': float(darkness_score),
                'evenness_score': float(evenness_score),
                'indian_tone': indian_tone
            }
        
        except Exception as e:
            logger.error(f"Error analyzing skin tone: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def enhance_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Enhance a face image for better deepfake detection
        
        Args:
            face_image: Face image in BGR format
            
        Returns:
            Enhanced face image
        """
        try:
            # Apply contrast enhancement
            alpha = 1.2  # Contrast control (1.0 means no change)
            beta = 10.0  # Brightness control (0 means no change)
            
            # Apply contrast and brightness adjustment
            enhanced = cv2.convertScaleAbs(face_image, alpha=alpha, beta=beta)
            
            # Apply bilateral filter to reduce noise while preserving edges
            enhanced = cv2.bilateralFilter(enhanced, 5, 75, 75)
            
            # Apply adaptive histogram equalization for better feature visibility
            # Create a CLAHE object with specified clip limit
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            
            # Convert to LAB color space
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            
            # Apply CLAHE to L channel
            lab_planes = cv2.split(lab)
            lab_planes[0] = clahe.apply(lab_planes[0])
            
            # Merge channels
            lab = cv2.merge(lab_planes)
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Apply subtle sharpening
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            return enhanced
        
        except Exception as e:
            logger.error(f"Error enhancing face: {str(e)}")
            return face_image

# Test the detector
if __name__ == "__main__":
    # Create detector
    detector = IndianFaceDetector()
    
    # Load test image
    test_image_path = "test_face.jpg"
    if not os.path.exists(test_image_path):
        test_image_path = os.path.join("test_faces", "test_face.jpg")
    
    if os.path.exists(test_image_path):
        # Load image
        image = cv2.imread(test_image_path)
        
        # Detect faces
        faces = detector.detect_faces(image)
        
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
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add skin tone text
            if face['skin_tone']['success']:
                tone = face['skin_tone']['indian_tone']['name']
                cv2.putText(image, tone, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save result
        output_path = "detected_faces.jpg"
        cv2.imwrite(output_path, image)
        print(f"Result saved to {output_path}")
    else:
        print(f"Test image not found: {test_image_path}")
