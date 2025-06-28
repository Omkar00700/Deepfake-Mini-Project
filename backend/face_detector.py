
"""
Face Detector module for DeepDefend
Provides face detection capabilities using DNN-based models and MTCNN
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
import tensorflow as tf
from mtcnn import MTCNN
# Removed dlib import as it's difficult to install on Windows

# Configure logging
logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Enhanced face detector using multiple detection methods:
    1. OpenCV's DNN-based face detector (default)
    2. MTCNN for more accurate face detection and alignment
    """
    
    def __init__(self, confidence_threshold=0.5, model_path=None, detector_type="auto"):
        """
        Initialize the face detector
        
        Args:
            confidence_threshold: Minimum confidence for face detection
            model_path: Path to a custom face detection model
            detector_type: Type of detector to use ('opencv', 'mtcnn', or 'auto')
        """
        self.confidence_threshold = confidence_threshold
        self.detector_type = detector_type
        self.detectors = {}
        
        # Initialize model directories
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize OpenCV DNN detector
        self._init_opencv_detector(model_path, model_dir)
        
        # Initialize MTCNN detector if requested
        if detector_type in ["mtcnn", "auto"]:
            try:
                logger.info("Initializing MTCNN face detector")
                self.detectors["mtcnn"] = MTCNN()
                logger.info("MTCNN face detector initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize MTCNN: {str(e)}")
                self.detectors["mtcnn"] = None
        
        # Removed dlib initialization as it's difficult to install on Windows
        
        logger.info(f"Face detector initialized with type: {detector_type}")
    
    def _init_opencv_detector(self, model_path, model_dir):
        """Initialize the OpenCV DNN face detector"""
        # Set default model paths
        default_prototxt = os.path.join(model_dir, "deploy.prototxt")
        default_model = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        
        # Check if we need to download the models
        if not os.path.exists(default_prototxt) or not os.path.exists(default_model):
            self._download_default_models(default_prototxt, default_model)
        
        # Load the face detection model
        if model_path:
            # Use custom model
            logger.info(f"Loading custom face detection model from {model_path}")
            self.detectors["opencv"] = cv2.dnn.readNetFromCaffe(model_path + ".prototxt", model_path + ".caffemodel")
        else:
            # Use default model
            logger.info("Loading default face detection model")
            self.detectors["opencv"] = cv2.dnn.readNetFromCaffe(default_prototxt, default_model)
        
        # Set preferred backend and target to improve performance
        self.detectors["opencv"].setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.detectors["opencv"].setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    def _download_default_models(self, prototxt_path, model_path):
        """
        Download the default face detection models if they don't exist
        
        Args:
            prototxt_path: Path to save the prototxt file
            model_path: Path to save the caffemodel file
        """
        try:
            import urllib.request
            
            logger.info("Downloading face detection models...")
            
            # Download prototxt
            prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            urllib.request.urlretrieve(prototxt_url, prototxt_path)
            
            # Download model
            model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            urllib.request.urlretrieve(model_url, model_path)
            
            logger.info("Face detection models downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download face detection models: {str(e)}")
            raise
    
    # Removed _download_dlib_predictor method as we're not using dlib
    
    def detect_faces(self, image) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image using the selected detector
        
        Args:
            image: Input image
            
        Returns:
            List of face bounding boxes as (x, y, width, height) tuples
        """
        try:
            # Get image dimensions
            (h, w) = image.shape[:2]
            
            # Initialize list of faces
            faces = []
            
            # Try MTCNN first if available
            if self.detector_type in ["mtcnn", "auto"] and self.detectors.get("mtcnn") is not None:
                try:
                    # Convert to RGB for MTCNN if needed
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        if image.dtype != np.uint8:
                            image_rgb = (image * 255).astype(np.uint8)
                        else:
                            image_rgb = image.copy()
                        
                        if image.shape[2] == 3:  # Check if it's a 3-channel image
                            # Convert BGR to RGB if needed
                            if image_rgb[0, 0, 0] != image_rgb[0, 0, 2]:  # Simple check for BGR
                                image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces with MTCNN
                    mtcnn_results = self.detectors["mtcnn"].detect_faces(image_rgb)
                    
                    if mtcnn_results:
                        logger.debug(f"MTCNN detected {len(mtcnn_results)} faces")
                        
                        for detection in mtcnn_results:
                            if detection['confidence'] >= self.confidence_threshold:
                                x, y, width, height = detection['box']
                                
                                # Ensure the bounding box falls within the image
                                x = max(0, x)
                                y = max(0, y)
                                width = min(w - x, width)
                                height = min(h - y, height)
                                
                                # Skip invalid dimensions
                                if width <= 0 or height <= 0:
                                    continue
                                
                                # Add face to list with additional metadata
                                face_data = (x, y, width, height)
                                faces.append(face_data)
                        
                        # If MTCNN found faces, return them
                        if faces:
                            return faces
                except Exception as e:
                    logger.warning(f"MTCNN detection failed: {str(e)}, falling back to OpenCV")
            
            # Removed dlib detection code as it's difficult to install on Windows
            
            # Fall back to OpenCV DNN detector if others failed or weren't used
            if not faces:
                # Create a blob from the image
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(image, (300, 300)), 1.0, (300, 300),
                    (104.0, 177.0, 123.0), swapRB=False, crop=False
                )
                
                # Set the blob as input to the network
                self.detectors["opencv"].setInput(blob)
                
                # Get detections
                detections = self.detectors["opencv"].forward()
                
                # Loop over the detections
                for i in range(detections.shape[2]):
                    # Extract the confidence
                    confidence = detections[0, 0, i, 2]
                    
                    # Filter out weak detections
                    if confidence > self.confidence_threshold:
                        # Compute the bounding box coordinates
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        
                        # Ensure the bounding box falls within the image
                        startX = max(0, startX)
                        startY = max(0, startY)
                        endX = min(w, endX)
                        endY = min(h, endY)
                        
                        # Calculate width and height
                        width = endX - startX
                        height = endY - startY
                        
                        # Skip invalid dimensions
                        if width <= 0 or height <= 0:
                            continue
                        
                        # Add face to list
                        faces.append((startX, startY, width, height))
            
            logger.debug(f"Detected {len(faces)} faces")
            return faces
            
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return []
    
    def extract_face(self, 
                    image, 
                    face: Tuple[int, int, int, int], 
                    target_size: Tuple[int, int],
                    margin_percent: float = 0.2,
                    align_face: bool = True) -> Optional[np.ndarray]:
        """
        Extract a face from an image with a margin and optional alignment
        
        Args:
            image: Source image
            face: Face coordinates (x, y, width, height)
            target_size: Target size for the processed face
            margin_percent: Percentage of margin to add around the face
            align_face: Whether to align the face using facial landmarks
            
        Returns:
            Extracted face image or None if extraction failed
        """
        try:
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
            
            # Face alignment using dlib has been removed
            # If MTCNN is available, we can use its keypoints for alignment
            if align_face and self.detector_type in ["mtcnn", "auto"] and self.detectors.get("mtcnn") is not None:
                try:
                    # Convert to RGB for MTCNN
                    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                        if face_img.dtype != np.uint8:
                            face_rgb = (face_img * 255).astype(np.uint8)
                        else:
                            face_rgb = face_img.copy()
                        
                        if face_img.shape[2] == 3:  # Check if it's a 3-channel image
                            # Convert BGR to RGB if needed
                            if face_rgb[0, 0, 0] != face_rgb[0, 0, 2]:  # Simple check for BGR
                                face_rgb = cv2.cvtColor(face_rgb, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces with MTCNN to get keypoints
                    mtcnn_results = self.detectors["mtcnn"].detect_faces(face_rgb)
                    
                    if mtcnn_results and len(mtcnn_results) > 0:
                        # Get keypoints for the first face
                        keypoints = mtcnn_results[0]['keypoints']
                        
                        # Get eye coordinates
                        left_eye = np.array(keypoints['left_eye'])
                        right_eye = np.array(keypoints['right_eye'])
                        
                        # Calculate angle for alignment
                        dY = right_eye[1] - left_eye[1]
                        dX = right_eye[0] - left_eye[0]
                        angle = np.degrees(np.arctan2(dY, dX))
                        
                        # Get center of face for rotation
                        center = (face_img.shape[1] // 2, face_img.shape[0] // 2)
                        
                        # Get rotation matrix
                        M = cv2.getRotationMatrix2D(center, angle, 1)
                        
                        # Apply rotation
                        aligned_face = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]), 
                                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                        
                        # Use aligned face
                        face_img = aligned_face
                        logger.debug("Face aligned using MTCNN keypoints")
                except Exception as e:
                    logger.warning(f"Face alignment with MTCNN failed: {str(e)}")
            
            # Resize to target size
            face_img = cv2.resize(face_img, target_size)
            
            return face_img
            
        except Exception as e:
            logger.error(f"Error in face extraction: {str(e)}")
            return None
