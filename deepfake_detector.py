"""
Deepfake Detection Pipeline
Main pipeline for detecting deepfakes in images and videos
"""

import cv2
import numpy as np
import os
import logging
import time
import json
import uuid
from typing import Dict, List, Any, Tuple, Optional
from indian_face_detector import IndianFaceDetector
from detection_model import DeepfakeDetectionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepfakeDetector:
    def __init__(self, model_name="efficientnet", use_ensemble=True, use_indian_enhancement=True):
        """
        Initialize the Deepfake Detector
        
        Args:
            model_name: Name of the model to use
            use_ensemble: Whether to use ensemble of models
            use_indian_enhancement: Whether to use Indian face enhancement
        """
        self.model_name = model_name
        self.use_ensemble = use_ensemble
        self.use_indian_enhancement = use_indian_enhancement
        
        # Initialize face detector
        self.face_detector = IndianFaceDetector()
        
        # Load models
        self.models = self._load_models()
        
        logger.info(f"Deepfake Detector initialized with model: {model_name}")
        logger.info(f"Ensemble: {use_ensemble}, Indian Enhancement: {use_indian_enhancement}")
    
    def _load_models(self) -> Dict[str, Any]:
        """
        Load deepfake detection models
        
        Returns:
            Dictionary of loaded models
        """
        models = {}
        
        # Load primary model
        models[self.model_name] = {
            "name": self.model_name,
            "model": DeepfakeDetectionModel(model_name=self.model_name, use_optimized=True),
            "loaded": True
        }
        
        # Load ensemble models if enabled - Enhanced with more models for better accuracy
        if self.use_ensemble:
            # Enhanced model list with more advanced models for better accuracy
            ensemble_models = [
                "efficientnet", 
                "xception", 
                "indian_specialized",
                "vit",               # Vision Transformer model
                "efficientnetv2",    # Newer EfficientNet version
                "convnext",          # ConvNeXt model
                "swin_transformer"   # Swin Transformer model
            ]
            
            # Assign weights to models (higher weight = more influence in ensemble)
            model_weights = {
                "efficientnet": 0.15,
                "xception": 0.15,
                "indian_specialized": 0.25,  # Higher weight for Indian specialized model
                "vit": 0.15,
                "efficientnetv2": 0.15,
                "convnext": 0.1,
                "swin_transformer": 0.05
            }
            
            for model_name in ensemble_models:
                if model_name != self.model_name:  # Skip if already loaded
                    try:
                        models[model_name] = {
                            "name": model_name,
                            "model": DeepfakeDetectionModel(model_name=model_name, use_optimized=True),
                            "loaded": True,
                            "weight": model_weights.get(model_name, 0.1)  # Default weight if not specified
                        }
                        logger.info(f"Loaded ensemble model: {model_name} with weight {model_weights.get(model_name, 0.1)}")
                    except Exception as e:
                        logger.error(f"Error loading ensemble model {model_name}: {str(e)}")
                        models[model_name] = {
                            "name": model_name,
                            "model": None,
                            "loaded": False,
                            "weight": 0.0  # Zero weight for failed models
                        }
        
        return models
    
    def detect_image(self, image_path: str) -> Dict[str, Any]:
        """
        Detect deepfakes in an image
        
        Args:
            image_path: Path to the image
            
        Returns:
            Detection result
        """
        try:
            # Generate unique ID for this detection
            detection_id = str(uuid.uuid4())
            
            # Start timer
            start_time = time.time()
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {
                    "success": False,
                    "error": f"Failed to load image: {image_path}"
                }
            
            # Detect faces
            faces = self.face_detector.detect_faces(image)
            if not faces:
                return {
                    "success": False,
                    "error": "No faces detected in the image"
                }
            
            # Process each face
            regions = []
            probabilities = []
            confidences = []
            
            for face in faces:
                # Extract face region
                x, y, w, h = face["box"]
                face_img = image[y:y+h, x:x+w]
                
                # Enhance face if enabled
                if self.use_indian_enhancement:
                    face_img = self.face_detector.enhance_face(face_img)
                
                # Resize face to model input size
                face_img = cv2.resize(face_img, (224, 224))
                
                # Detect deepfake (placeholder for actual detection)
                # In a real implementation, this would run the model on the face
                result = self._detect_face(face_img)
                
                # Add to results
                probabilities.append(result["probability"])
                confidences.append(result["confidence"])
                
                # Add region to results
                regions.append({
                    "box": face["box"],
                    "probability": result["probability"],
                    "confidence": result["confidence"],
                    "skin_tone": face["skin_tone"]
                })
            
            # Calculate overall probability and confidence
            if self.use_ensemble:
                # Weight by face size and confidence
                weights = [r["box"][2] * r["box"][3] * r["confidence"] for r in regions]
                total_weight = sum(weights)
                
                if total_weight > 0:
                    overall_probability = sum([p * w / total_weight for p, w in zip(probabilities, weights)])
                    overall_confidence = sum([c * w / total_weight for c, w in zip(confidences, weights)])
                else:
                    overall_probability = np.mean(probabilities)
                    overall_confidence = np.mean(confidences)
            else:
                overall_probability = np.mean(probabilities)
                overall_confidence = np.mean(confidences)
            
            # End timer
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Create result
            result = {
                "id": detection_id,
                "filename": os.path.basename(image_path),
                "detectionType": "image",
                "probability": float(overall_probability),
                "confidence": float(overall_confidence),
                "processingTime": float(processing_time),
                "regions": regions,
                "model": self.model_name,
                "ensemble": self.use_ensemble,
                "indianEnhancement": self.use_indian_enhancement
            }
            
            # Save result
            self._save_result(detection_id, result)
            
            return {
                "success": True,
                "detection_id": detection_id,
                "result": result
            }
        
        except Exception as e:
            logger.error(f"Error detecting deepfakes in image: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def detect_video(self, video_path: str, max_frames=30, frame_interval=1.0) -> Dict[str, Any]:
        """
        Detect deepfakes in a video
        
        Args:
            video_path: Path to the video
            max_frames: Maximum number of frames to process
            frame_interval: Interval between frames in seconds
            
        Returns:
            Detection result
        """
        try:
            # Generate unique ID for this detection
            detection_id = str(uuid.uuid4())
            
            # Start timer
            start_time = time.time()
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {
                    "success": False,
                    "error": f"Failed to open video: {video_path}"
                }
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            # Calculate frame interval in frames
            frame_interval_frames = int(frame_interval * fps)
            
            # Initialize results
            frame_results = []
            frame_idx = 0
            processed_frames = 0
            
            # Process frames
            while cap.isOpened() and processed_frames < max_frames:
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect faces
                faces = self.face_detector.detect_faces(frame)
                
                # Process each face
                frame_regions = []
                
                for face in faces:
                    # Extract face region
                    x, y, w, h = face["box"]
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Enhance face if enabled
                    if self.use_indian_enhancement:
                        face_img = self.face_detector.enhance_face(face_img)
                    
                    # Resize face to model input size
                    face_img = cv2.resize(face_img, (224, 224))
                    
                    # Detect deepfake (placeholder for actual detection)
                    # In a real implementation, this would run the model on the face
                    result = self._detect_face(face_img)
                    
                    # Add region to results
                    frame_regions.append({
                        "box": face["box"],
                        "probability": result["probability"],
                        "confidence": result["confidence"],
                        "frame": processed_frames,
                        "timestamp": frame_idx / fps,
                        "skin_tone": face["skin_tone"]
                    })
                
                # Add frame result
                if frame_regions:
                    frame_results.append({
                        "frame": processed_frames,
                        "timestamp": frame_idx / fps,
                        "regions": frame_regions
                    })
                
                # Update counters
                frame_idx += frame_interval_frames
                processed_frames += 1
            
            # Close video
            cap.release()
            
            # Calculate overall results
            all_probabilities = []
            all_confidences = []
            all_regions = []
            
            for frame_result in frame_results:
                for region in frame_result["regions"]:
                    all_probabilities.append(region["probability"])
                    all_confidences.append(region["confidence"])
                    all_regions.append(region)
            
            # Calculate overall probability and confidence
            if all_probabilities:
                overall_probability = np.mean(all_probabilities)
                overall_confidence = np.mean(all_confidences)
            else:
                overall_probability = 0.0
                overall_confidence = 0.0
            
            # End timer
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Create result
            result = {
                "id": detection_id,
                "filename": os.path.basename(video_path),
                "detectionType": "video",
                "probability": float(overall_probability),
                "confidence": float(overall_confidence),
                "processingTime": float(processing_time),
                "frameCount": processed_frames,
                "duration": float(duration),
                "fps": float(fps),
                "regions": all_regions,
                "frameResults": frame_results,
                "model": self.model_name,
                "ensemble": self.use_ensemble,
                "indianEnhancement": self.use_indian_enhancement
            }
            
            # Save result
            self._save_result(detection_id, result)
            
            return {
                "success": True,
                "detection_id": detection_id,
                "result": result
            }
        
        except Exception as e:
            logger.error(f"Error detecting deepfakes in video: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _detect_face(self, face_img: np.ndarray) -> Dict[str, Any]:
        """
        Detect if a face is deepfake
        
        Args:
            face_img: Face image in BGR format
            
        Returns:
            Detection result
        """
        try:
            # Use primary model for detection
            primary_model = self.models[self.model_name]["model"]
            primary_result = primary_model.detect_with_artifacts(face_img)
            
            # Initialize results
            results = [primary_result]
            weights = [1.0]  # Primary model has weight 1.0
            
            # Use ensemble if enabled - Enhanced with more models and better weighting
            if self.use_ensemble:
                # Use all available models in the ensemble
                for model_name, model_info in self.models.items():
                    if model_name != self.model_name and model_info["loaded"]:
                        try:
                            # Get model
                            model = model_info["model"]
                            
                            # Get result
                            result = model.detect_with_artifacts(face_img)
                            
                            # Add to results
                            results.append(result)
                            
                            # Use model weight from initialization (more sophisticated weighting)
                            model_weight = model_info.get("weight", 0.1)
                            
                            # Adjust weight based on confidence and model performance
                            adjusted_weight = model_weight * (0.5 + 0.5 * result["confidence"])
                            weights.append(adjusted_weight)
                            
                            logger.debug(f"Model {model_name} prediction: {result['probability']:.3f} with weight {adjusted_weight:.3f}")
                        except Exception as e:
                            logger.error(f"Error using ensemble model {model_name}: {str(e)}")
            
            # Calculate weighted average with improved weighting strategy
            total_weight = sum(weights)
            
            if total_weight > 0:
                # Apply softmax to probabilities for better ensemble fusion
                probabilities = [r["probability"] for r in results]
                confidences = [r["confidence"] for r in results]
                
                # Apply temperature scaling for sharper predictions (t=0.5 makes predictions more confident)
                temperature = 0.5
                scaled_weights = [w ** (1/temperature) for w in weights]
                scaled_total = sum(scaled_weights)
                
                if scaled_total > 0:
                    normalized_weights = [w / scaled_total for w in scaled_weights]
                    
                    # Calculate weighted average with normalized weights
                    probability = sum([p * w for p, w in zip(probabilities, normalized_weights)])
                    confidence = sum([c * w for c, w in zip(confidences, normalized_weights)])
                    
                    # Apply sigmoid calibration for better probability estimates
                    probability = 1 / (1 + np.exp(-5 * (probability - 0.5)))
                else:
                    probability = primary_result["probability"]
                    confidence = primary_result["confidence"]
            else:
                probability = primary_result["probability"]
                confidence = primary_result["confidence"]
            
            # Add artifacts from primary model
            artifacts = primary_result.get("artifacts", {})
            
            return {
                "probability": float(probability),
                "confidence": float(confidence),
                "artifacts": artifacts,
                "ensemble_results": results if self.use_ensemble else [primary_result]
            }
        
        except Exception as e:
            logger.error(f"Error detecting face: {str(e)}")
            
            # Return fallback result
            return {
                "probability": 0.5,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _save_result(self, detection_id: str, result: Dict[str, Any]) -> bool:
        """
        Save detection result
        
        Args:
            detection_id: ID of the detection
            result: Detection result
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create results directory if it doesn't exist
            results_dir = os.path.join(os.path.dirname(__file__), "results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Create result file path
            result_path = os.path.join(results_dir, f"{detection_id}.json")
            
            # Write result to file
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Saved detection result {detection_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error saving detection result: {str(e)}")
            return False

# Test the detector
if __name__ == "__main__":
    # Create detector
    detector = DeepfakeDetector(use_indian_enhancement=True)
    
    # Test with image
    test_image_path = "test_image.jpg"
    if os.path.exists(test_image_path):
        # Detect deepfakes
        result = detector.detect_image(test_image_path)
        
        # Print results
        if result["success"]:
            print(f"Detection ID: {result['detection_id']}")
            print(f"Probability: {result['result']['probability']:.4f}")
            print(f"Confidence: {result['result']['confidence']:.4f}")
            print(f"Processing Time: {result['result']['processingTime']:.4f} seconds")
            print(f"Regions: {len(result['result']['regions'])}")
        else:
            print(f"Error: {result['error']}")
    else:
        print(f"Test image not found: {test_image_path}")