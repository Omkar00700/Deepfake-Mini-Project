"""
Ensemble Detector for DeepDefend
This module implements an ensemble approach to deepfake detection,
combining multiple models for improved accuracy
"""

import numpy as np
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import cv2
from backend.model_loader import ModelManager
from backend.indian_face_utils import IndianFacePreprocessor
from backend.skin_tone_analyzer import SkinToneAnalyzer
from backend.config import ENSEMBLE_WEIGHTS, INDIAN_ENHANCEMENT_ENABLED

# Configure logging
logger = logging.getLogger(__name__)

class EnsembleDeepfakeDetector:
    """
    Ensemble-based deepfake detector that combines multiple models
    with specialized Indian face processing
    """
    
    def __init__(self, use_indian_enhancement: bool = True):
        """
        Initialize the ensemble deepfake detector
        
        Args:
            use_indian_enhancement: Whether to use Indian-specific enhancements
        """
        self.model_manager = ModelManager()
        self.use_indian_enhancement = use_indian_enhancement and INDIAN_ENHANCEMENT_ENABLED
        
        # Initialize face preprocessor
        from face_detector import FaceDetector
        self.face_detector = FaceDetector()
        self.face_preprocessor = IndianFacePreprocessor(self.face_detector)
        
        # Initialize skin tone analyzer
        self.skin_tone_analyzer = SkinToneAnalyzer()
        
        # Load default model weights
        self.ensemble_weights = ENSEMBLE_WEIGHTS.copy()
        
        # Adaptive weights based on performance
        self.adaptive_weights = {}
        
        # Performance tracking
        self.total_inference_time = 0
        self.detection_count = 0
        self.successful_detections = 0
        
        logger.info(f"Initialized EnsembleDeepfakeDetector (Indian enhancement: {use_indian_enhancement})")
    
    def detect_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect deepfakes in an image using ensemble approach
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Detection result with probability, confidence, and metadata
        """
        start_time = time.time()
        self.detection_count += 1
        
        try:
            # Detect faces
            faces = self.face_detector.detect_faces(image)
            
            if not faces:
                logger.warning("No faces detected in the image")
                return {
                    "success": False,
                    "error": "No faces detected in the image",
                    "probability": 0.5,
                    "confidence": 0.1,
                    "processingTime": time.time() - start_time,
                    "regions": []
                }
            
            # Process each face
            regions = []
            overall_probability = 0.0
            overall_confidence = 0.0
            
            for face_idx, face_box in enumerate(faces):
                # Process face
                face_result = self._process_face(image, face_box)
                
                if face_result["success"]:
                    regions.append(face_result)
                    
                    # Weight by confidence
                    weight = face_result["confidence"]
                    overall_probability += face_result["probability"] * weight
                    overall_confidence += face_result["confidence"] * weight
            
            # Calculate overall results
            if regions:
                # Normalize by total confidence
                total_confidence = sum(region["confidence"] for region in regions)
                if total_confidence > 0:
                    overall_probability /= total_confidence
                    overall_confidence /= len(regions)
                else:
                    overall_probability = 0.5
                    overall_confidence = 0.1
            else:
                overall_probability = 0.5
                overall_confidence = 0.1
            
            # Record successful detection
            self.successful_detections += 1
            processing_time = time.time() - start_time
            self.total_inference_time += processing_time
            
            return {
                "success": True,
                "probability": float(overall_probability),
                "confidence": float(overall_confidence),
                "processingTime": processing_time,
                "regions": regions,
                "faceCount": len(faces)
            }
            
        except Exception as e:
            logger.error(f"Error detecting deepfakes in image: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "probability": 0.5,
                "confidence": 0.1,
                "processingTime": time.time() - start_time,
                "regions": []
            }
    
    def detect_video(self, video_path: str, max_frames: int = 30) -> Dict[str, Any]:
        """
        Detect deepfakes in a video using ensemble approach with temporal analysis
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to analyze
            
        Returns:
            Detection result with probability, confidence, and metadata
        """
        start_time = time.time()
        self.detection_count += 1
        
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Failed to open video file: {video_path}")
                return {
                    "success": False,
                    "error": f"Failed to open video file: {video_path}",
                    "probability": 0.5,
                    "confidence": 0.1,
                    "processingTime": time.time() - start_time,
                    "regions": []
                }
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Calculate frame sampling
            if frame_count <= max_frames:
                # Process all frames
                frame_indices = list(range(frame_count))
            else:
                # Sample frames evenly
                frame_indices = [int(i * frame_count / max_frames) for i in range(max_frames)]
            
            # Process frames
            frame_results = []
            for frame_idx in frame_indices:
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame {frame_idx}")
                    continue
                
                # Process frame
                frame_result = self.detect_image(frame)
                if frame_result["success"]:
                    # Add frame index
                    frame_result["frameIndex"] = frame_idx
                    frame_results.append(frame_result)
            
            # Release video
            cap.release()
            
            # Check if we have any results
            if not frame_results:
                logger.warning("No valid frames processed")
                return {
                    "success": False,
                    "error": "No valid frames processed",
                    "probability": 0.5,
                    "confidence": 0.1,
                    "processingTime": time.time() - start_time,
                    "regions": []
                }
            
            # Perform temporal analysis
            temporal_result = self._analyze_temporal_consistency(frame_results)
            
            # Calculate overall results
            overall_probability = temporal_result["probability"]
            overall_confidence = temporal_result["confidence"]
            
            # Record successful detection
            self.successful_detections += 1
            processing_time = time.time() - start_time
            self.total_inference_time += processing_time
            
            return {
                "success": True,
                "probability": float(overall_probability),
                "confidence": float(overall_confidence),
                "processingTime": processing_time,
                "frameCount": len(frame_results),
                "totalFrames": frame_count,
                "duration": duration,
                "fps": fps,
                "temporalConsistency": temporal_result["consistency"],
                "frames": frame_results
            }
            
        except Exception as e:
            logger.error(f"Error detecting deepfakes in video: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "probability": 0.5,
                "confidence": 0.1,
                "processingTime": time.time() - start_time,
                "regions": []
            }
    
    def _process_face(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Process a single face with ensemble detection
        
        Args:
            image: Input image
            face_box: Face bounding box (x, y, width, height)
            
        Returns:
            Face detection result
        """
        try:
            # Preprocess face
            face_data = self.face_preprocessor.preprocess_face(
                image, 
                face_box, 
                target_size=(224, 224)
            )
            
            if face_data is None:
                logger.warning("Face preprocessing failed")
                return {
                    "success": False,
                    "error": "Face preprocessing failed",
                    "box": face_box,
                    "probability": 0.5,
                    "confidence": 0.1
                }
            
            # Get face image
            face_img = face_data["image"]
            
            # Get available models
            models = self.model_manager.get_available_models()
            
            # Initialize results
            model_results = {}
            
            # Run detection with each model
            for model_name in models:
                # Get model
                model = self.model_manager.get_model(model_name)
                
                # Run prediction
                prediction = model.predict(face_img)
                
                # Store result
                model_results[model_name] = prediction
            
            # Combine results using ensemble weights
            ensemble_result = self._combine_ensemble_results(model_results)
            
            # Add Indian face analysis if enabled
            if self.use_indian_enhancement:
                # Analyze skin tone
                skin_tone_result = face_data.get("skin_tone", {})
                
                # Analyze face authenticity
                authenticity_result = self.face_preprocessor.analyze_face_authenticity(face_data)
                
                # Adjust probability based on authenticity score
                if authenticity_result["authenticity_score"] < 0.5:
                    # Lower authenticity score increases deepfake probability
                    adjustment = (0.5 - authenticity_result["authenticity_score"]) * 0.2
                    ensemble_result["probability"] = min(0.95, ensemble_result["probability"] + adjustment)
                
                # Add results to metadata
                ensemble_result["indian_analysis"] = {
                    "authenticity_score": authenticity_result["authenticity_score"],
                    "authenticity_confidence": authenticity_result["confidence"],
                    "factors": authenticity_result["factors"]
                }
            
            # Add face data
            return {
                "success": True,
                "box": face_box,
                "probability": ensemble_result["probability"],
                "confidence": ensemble_result["confidence"],
                "model_predictions": {name: result["probability"] for name, result in model_results.items()},
                "metadata": {
                    "skin_tone": face_data.get("skin_tone", {}),
                    "skin_anomalies": face_data.get("skin_anomalies", {}),
                    "processing_metrics": {
                        "ensemble_weights": self.ensemble_weights,
                        "uncertainty": ensemble_result["uncertainty"]
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing face: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "box": face_box,
                "probability": 0.5,
                "confidence": 0.1
            }
    
    def _combine_ensemble_results(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine results from multiple models using weighted ensemble
        
        Args:
            model_results: Dictionary of model results
            
        Returns:
            Combined result
        """
        # Get adaptive weights from model manager
        adaptive_weights = self.model_manager.get_adaptive_weights()
        
        # Initialize weighted sum
        weighted_prob_sum = 0.0
        weighted_uncertainty_sum = 0.0
        total_weight = 0.0
        
        # Calculate weighted average
        for model_name, result in model_results.items():
            # Skip failed predictions
            if "error" in result:
                continue
            
            # Get probability and uncertainty
            probability = result["probability"]
            uncertainty = result.get("uncertainty", 0.1)
            
            # Get weight (combine fixed and adaptive weights)
            fixed_weight = self.ensemble_weights.get(model_name, 1.0)
            adaptive_weight = adaptive_weights.get(model_name, 1.0)
            
            # Calculate confidence-adjusted weight
            # Lower uncertainty = higher weight
            confidence = 1.0 - uncertainty
            confidence_factor = 0.5 + 0.5 * confidence  # Scale to [0.5, 1.0]
            
            # Final weight
            weight = fixed_weight * adaptive_weight * confidence_factor
            
            # Add to weighted sums
            weighted_prob_sum += probability * weight
            weighted_uncertainty_sum += uncertainty * weight
            total_weight += weight
        
        # Calculate final probability and uncertainty
        if total_weight > 0:
            final_probability = weighted_prob_sum / total_weight
            final_uncertainty = weighted_uncertainty_sum / total_weight
        else:
            final_probability = 0.5
            final_uncertainty = 1.0
        
        # Calculate confidence (inverse of uncertainty)
        confidence = 1.0 - final_uncertainty
        
        # Ensure values are in valid range
        final_probability = max(0.05, min(0.95, final_probability))
        confidence = max(0.1, min(0.95, confidence))
        
        return {
            "probability": float(final_probability),
            "confidence": float(confidence),
            "uncertainty": float(final_uncertainty)
        }
    
    def _analyze_temporal_consistency(self, frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze temporal consistency across video frames
        
        Args:
            frame_results: List of frame detection results
            
        Returns:
            Temporal analysis result
        """
        # Extract probabilities and confidences
        probabilities = []
        confidences = []
        
        for frame_result in frame_results:
            probabilities.append(frame_result["probability"])
            confidences.append(frame_result["confidence"])
        
        # Convert to numpy arrays
        probabilities = np.array(probabilities)
        confidences = np.array(confidences)
        
        # Calculate statistics
        mean_prob = np.mean(probabilities)
        std_prob = np.std(probabilities)
        
        # Calculate temporal consistency
        # Lower standard deviation = higher consistency
        consistency = 1.0 - min(1.0, std_prob * 5.0)
        
        # Calculate confidence-weighted probability
        if np.sum(confidences) > 0:
            weighted_prob = np.sum(probabilities * confidences) / np.sum(confidences)
        else:
            weighted_prob = mean_prob
        
        # Calculate overall confidence
        # Base confidence from average
        base_confidence = np.mean(confidences)
        
        # Adjust based on consistency and sample size
        consistency_factor = 0.5 + 0.5 * consistency  # Scale to [0.5, 1.0]
        sample_factor = min(1.0, len(frame_results) / 10.0)  # More frames = higher confidence
        
        overall_confidence = base_confidence * consistency_factor * sample_factor
        
        # Ensure values are in valid range
        weighted_prob = max(0.05, min(0.95, weighted_prob))
        overall_confidence = max(0.1, min(0.95, overall_confidence))
        
        return {
            "probability": float(weighted_prob),
            "confidence": float(overall_confidence),
            "consistency": float(consistency),
            "mean_probability": float(mean_prob),
            "std_probability": float(std_prob),
            "frame_count": len(frame_results)
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the detector
        
        Returns:
            Dictionary with performance metrics
        """
        # Calculate average inference time
        avg_inference_time = 0
        if self.successful_detections > 0:
            avg_inference_time = self.total_inference_time / self.successful_detections
        
        return {
            "detection_count": self.detection_count,
            "successful_detections": self.successful_detections,
            "success_rate": self.successful_detections / max(1, self.detection_count),
            "avg_inference_time": avg_inference_time,
            "total_inference_time": self.total_inference_time,
            "indian_enhancement_enabled": self.use_indian_enhancement
        }
