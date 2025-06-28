"""
Enhanced Deepfake Detection System
Integrates multiple advanced techniques to achieve >95% accuracy
"""

import os
import numpy as np
import tensorflow as tf
import logging
import time
from typing import List, Dict, Tuple, Optional, Union, Any
import cv2
from pathlib import Path
import json
from collections import deque

# Import custom modules
from enhanced_ensemble import EnhancedEnsemble
from advanced_augmentation import DeepfakeAugmenter
from cross_modal_verification import CrossModalVerifier
from temporal_analysis import TemporalConsistencyAnalyzer
from model_loader import DeepfakeDetectionModel, ModelManager
from inference_core import PredictionResult, InferenceResult, FaceRegion, evaluate_input_quality
from face_detector import FaceDetector
from backend.config import MODEL_DIR, DEEPFAKE_THRESHOLD

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedDeepfakeDetector:
    """
    Enhanced deepfake detection system that integrates multiple advanced techniques
    to achieve >95% accuracy
    """
    
    def __init__(self, 
                 use_ensemble: bool = True,
                 use_temporal: bool = True,
                 use_cross_modal: bool = True,
                 use_advanced_augmentation: bool = True,
                 model_names: Optional[List[str]] = None,
                 face_detector_type: str = "retinaface"):
        """
        Initialize the enhanced deepfake detector
        
        Args:
            use_ensemble: Whether to use ensemble learning
            use_temporal: Whether to use temporal analysis
            use_cross_modal: Whether to use cross-modal verification
            use_advanced_augmentation: Whether to use advanced augmentation
            model_names: List of model names to include in the ensemble
            face_detector_type: Type of face detector to use
        """
        self.use_ensemble = use_ensemble
        self.use_temporal = use_temporal
        self.use_cross_modal = use_cross_modal
        self.use_advanced_augmentation = use_advanced_augmentation
        
        # Initialize model manager
        self.model_manager = ModelManager()
        
        # Initialize face detector
        self.face_detector = FaceDetector(detector_type=face_detector_type)
        
        # Initialize ensemble if enabled
        self.ensemble = None
        if use_ensemble:
            self.ensemble = EnhancedEnsemble(
                models=model_names,
                dynamic_weighting=True,
                calibration_enabled=True,
                temporal_analysis=use_temporal
            )
        
        # Initialize temporal analyzer if enabled
        self.temporal_analyzer = None
        if use_temporal:
            self.temporal_analyzer = TemporalConsistencyAnalyzer(
                buffer_size=30,
                use_optical_flow=True,
                use_face_tracking=True,
                use_feature_consistency=True
            )
        
        # Initialize cross-modal verifier if enabled
        self.cross_modal_verifier = None
        if use_cross_modal:
            # Get visual model from ensemble or model manager
            visual_model = None
            if self.ensemble:
                # Use the first model from the ensemble
                model_name = self.ensemble.models[0]
                visual_model = self.model_manager.get_model(model_name).model
            else:
                # Use the default model
                visual_model = self.model_manager.get_model().model
            
            self.cross_modal_verifier = CrossModalVerifier(
                visual_model=visual_model,
                audio_model_path=os.path.join(MODEL_DIR, "audio_deepfake_model.h5"),
                use_lip_sync=True,
                use_audio_analysis=True
            )
        
        # Initialize augmenter if enabled
        self.augmenter = None
        if use_advanced_augmentation:
            self.augmenter = DeepfakeAugmenter(
                cache_dir=os.path.join(MODEL_DIR, "augmentation_cache"),
                use_gan=True,
                use_mixup=True,
                use_cutmix=True,
                use_domain_randomization=True
            )
        
        # Initialize state
        self.is_processing_video = False
        self.current_video_path = None
        self.current_video_results = {}
        
        logger.info(f"Initialized EnhancedDeepfakeDetector with ensemble={use_ensemble}, "
                   f"temporal={use_temporal}, cross_modal={use_cross_modal}")
    
    def detect_image(self, 
                    image_path: str,
                    return_faces: bool = True,
                    include_details: bool = False) -> Dict[str, Any]:
        """
        Detect deepfakes in an image
        
        Args:
            image_path: Path to the image file
            return_faces: Whether to return detected faces
            include_details: Whether to include detailed information
            
        Returns:
            Dictionary of detection results
        """
        start_time = time.time()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {
                "error": f"Failed to load image: {image_path}",
                "is_deepfake": False,
                "confidence": 0.0,
                "processing_time": time.time() - start_time
            }
        
        # Detect faces
        faces = self.face_detector.detect_faces(image)
        
        if not faces:
            return {
                "error": "No faces detected",
                "is_deepfake": False,
                "confidence": 0.0,
                "processing_time": time.time() - start_time
            }
        
        # Process each face
        face_results = []
        overall_prediction = 0.0
        overall_confidence = 0.0
        
        for face in faces:
            # Extract face region
            x, y, w, h = face
            face_img = image[y:y+h, x:x+w]
            
            # Preprocess face
            face_img = cv2.resize(face_img, (224, 224))
            
            # Apply augmentation if enabled
            if self.augmenter and np.random.random() < 0.3:
                face_img = self.augmenter.augment_image(face_img)
            
            # Get prediction
            if self.ensemble:
                # Use ensemble prediction
                pred_info = self.ensemble.predict(face_img, include_details=True)
                probability = pred_info["probability"]
                confidence = pred_info["confidence"]
                uncertainty = pred_info.get("uncertainty", 0.1)
            else:
                # Use single model prediction
                model = self.model_manager.get_model()
                pred_info = model.predict(face_img, include_uncertainty=True)
                
                if isinstance(pred_info, dict):
                    probability = pred_info.get("probability", 0.5)
                    confidence = pred_info.get("confidence", 0.8)
                    uncertainty = pred_info.get("uncertainty", 0.1)
                else:
                    probability = pred_info
                    confidence = 0.8
                    uncertainty = 0.1
            
            # Create face result
            face_result = {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "probability": float(probability),
                "confidence": float(confidence),
                "uncertainty": float(uncertainty),
                "is_deepfake": probability > DEEPFAKE_THRESHOLD
            }
            
            # Add detailed information if requested
            if include_details and isinstance(pred_info, dict):
                face_result["details"] = pred_info
            
            face_results.append(face_result)
            
            # Update overall prediction (weighted by face size and confidence)
            weight = (w * h) * confidence
            overall_prediction += probability * weight
            overall_confidence += confidence * weight
        
        # Normalize overall prediction and confidence
        total_weight = sum((f["width"] * f["height"]) * f["confidence"] for f in face_results)
        if total_weight > 0:
            overall_prediction /= total_weight
            overall_confidence /= total_weight
        else:
            overall_prediction = np.mean([f["probability"] for f in face_results])
            overall_confidence = np.mean([f["confidence"] for f in face_results])
        
        # Prepare result
        result = {
            "is_deepfake": overall_prediction > DEEPFAKE_THRESHOLD,
            "probability": float(overall_prediction),
            "confidence": float(overall_confidence),
            "face_count": len(face_results),
            "processing_time": time.time() - start_time
        }
        
        if return_faces:
            result["faces"] = face_results
        
        return result
    
    def detect_video(self, 
                    video_path: str,
                    max_frames: int = 30,
                    return_frames: bool = False,
                    include_details: bool = False) -> Dict[str, Any]:
        """
        Detect deepfakes in a video
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to process
            return_frames: Whether to return processed frames
            include_details: Whether to include detailed information
            
        Returns:
            Dictionary of detection results
        """
        start_time = time.time()
        
        # Reset state
        self.is_processing_video = True
        self.current_video_path = video_path
        
        if self.temporal_analyzer:
            self.temporal_analyzer.reset()
        
        if self.ensemble:
            self.ensemble.reset_temporal_buffer()
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "error": f"Failed to open video: {video_path}",
                "is_deepfake": False,
                "confidence": 0.0,
                "processing_time": time.time() - start_time
            }
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval to extract max_frames evenly
        if frame_count <= max_frames:
            frame_interval = 1
        else:
            frame_interval = frame_count // max_frames
        
        # Process frames
        frame_results = []
        processed_frames = []
        face_regions = []
        
        frame_idx = 0
        while cap.isOpened() and len(frame_results) < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # Process this frame
                frame_result = self._process_video_frame(frame, frame_idx)
                
                if frame_result:
                    frame_results.append(frame_result)
                    
                    if return_frames:
                        processed_frames.append(frame)
                    
                    if "faces" in frame_result:
                        face_regions.extend(frame_result["faces"])
            
            frame_idx += 1
        
        # Release video capture
        cap.release()
        
        # Perform temporal analysis if enabled
        temporal_results = {}
        if self.temporal_analyzer and len(frame_results) >= 2:
            temporal_results = self.temporal_analyzer.analyze()
        
        # Perform cross-modal verification if enabled
        cross_modal_results = {}
        if self.cross_modal_verifier:
            # Get overall prediction from frame results
            if frame_results:
                overall_prediction = np.mean([f["probability"] for f in frame_results])
            else:
                overall_prediction = 0.5
            
            cross_modal_results = self.cross_modal_verifier.verify(
                video_path, visual_prediction=overall_prediction
            )
        
        # Calculate final prediction and confidence
        final_prediction, final_confidence = self._calculate_final_video_result(
            frame_results, temporal_results, cross_modal_results
        )
        
        # Prepare result
        result = {
            "is_deepfake": final_prediction > DEEPFAKE_THRESHOLD,
            "probability": float(final_prediction),
            "confidence": float(final_confidence),
            "frame_count": len(frame_results),
            "processing_time": time.time() - start_time
        }
        
        # Add detailed information if requested
        if include_details:
            result["frame_results"] = frame_results
            
            if temporal_results:
                result["temporal_analysis"] = temporal_results
            
            if cross_modal_results:
                result["cross_modal_verification"] = cross_modal_results
        
        # Add frames if requested
        if return_frames:
            result["frames"] = processed_frames
        
        # Store current video results
        self.current_video_results = result
        self.is_processing_video = False
        
        return result
    
    def _process_video_frame(self, 
                            frame: np.ndarray, 
                            frame_idx: int) -> Dict[str, Any]:
        """
        Process a single video frame
        
        Args:
            frame: Video frame
            frame_idx: Frame index
            
        Returns:
            Dictionary of frame processing results
        """
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        
        if not faces:
            return None
        
        # Process each face
        face_results = []
        frame_prediction = 0.0
        frame_confidence = 0.0
        total_weight = 0.0
        
        for face in faces:
            # Extract face region
            x, y, w, h = face
            face_img = frame[y:y+h, x:x+w]
            
            # Skip if face is too small
            if w < 30 or h < 30:
                continue
            
            # Preprocess face
            face_img = cv2.resize(face_img, (224, 224))
            
            # Get prediction
            if self.ensemble:
                # Use ensemble prediction
                pred_info = self.ensemble.predict(face_img, include_details=True)
                probability = pred_info["probability"]
                confidence = pred_info["confidence"]
                uncertainty = pred_info.get("uncertainty", 0.1)
                features = pred_info.get("features", None)
            else:
                # Use single model prediction
                model = self.model_manager.get_model()
                pred_info = model.predict(face_img, include_uncertainty=True)
                
                if isinstance(pred_info, dict):
                    probability = pred_info.get("probability", 0.5)
                    confidence = pred_info.get("confidence", 0.8)
                    uncertainty = pred_info.get("uncertainty", 0.1)
                    features = pred_info.get("features", None)
                else:
                    probability = pred_info
                    confidence = 0.8
                    uncertainty = 0.1
                    features = None
            
            # Create face result
            face_result = {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "probability": float(probability),
                "confidence": float(confidence),
                "uncertainty": float(uncertainty),
                "is_deepfake": probability > DEEPFAKE_THRESHOLD
            }
            
            face_results.append(face_result)
            
            # Update frame prediction (weighted by face size and confidence)
            weight = (w * h) * confidence
            frame_prediction += probability * weight
            frame_confidence += confidence * weight
            total_weight += weight
            
            # Add to temporal analyzer if enabled
            if self.temporal_analyzer:
                self.temporal_analyzer.add_frame(
                    frame, probability, confidence, face_result, features
                )
        
        # Normalize frame prediction and confidence
        if total_weight > 0:
            frame_prediction /= total_weight
            frame_confidence /= total_weight
        else:
            frame_prediction = np.mean([f["probability"] for f in face_results])
            frame_confidence = np.mean([f["confidence"] for f in face_results])
        
        # Prepare frame result
        frame_result = {
            "frame_idx": frame_idx,
            "probability": float(frame_prediction),
            "confidence": float(frame_confidence),
            "face_count": len(face_results),
            "faces": face_results
        }
        
        return frame_result
    
    def _calculate_final_video_result(self,
                                     frame_results: List[Dict[str, Any]],
                                     temporal_results: Dict[str, Any],
                                     cross_modal_results: Dict[str, Any]) -> Tuple[float, float]:
        """
        Calculate final prediction and confidence for a video
        
        Args:
            frame_results: List of frame processing results
            temporal_results: Temporal analysis results
            cross_modal_results: Cross-modal verification results
            
        Returns:
            Tuple of (final_prediction, final_confidence)
        """
        # Initialize with average of frame results
        if frame_results:
            # Weight by confidence
            weights = [f["confidence"] for f in frame_results]
            total_weight = sum(weights)
            
            if total_weight > 0:
                initial_prediction = sum(f["probability"] * f["confidence"] for f in frame_results) / total_weight
                initial_confidence = sum(f["confidence"] * f["confidence"] for f in frame_results) / total_weight
            else:
                initial_prediction = np.mean([f["probability"] for f in frame_results])
                initial_confidence = np.mean([f["confidence"] for f in frame_results])
        else:
            initial_prediction = 0.5
            initial_confidence = 0.5
        
        # Apply temporal analysis if available
        if temporal_results and "final_prediction" in temporal_results:
            temporal_prediction = temporal_results["final_prediction"]
            temporal_confidence = temporal_results["final_confidence"]
            temporal_weight = 0.4  # 40% weight for temporal analysis
        else:
            temporal_prediction = initial_prediction
            temporal_confidence = initial_confidence
            temporal_weight = 0.0
        
        # Apply cross-modal verification if available
        if cross_modal_results and "cross_modal_score" in cross_modal_results:
            cross_modal_prediction = cross_modal_results.get("audio_prediction", initial_prediction)
            cross_modal_confidence = cross_modal_results.get("confidence", initial_confidence)
            cross_modal_weight = 0.3  # 30% weight for cross-modal verification
        else:
            cross_modal_prediction = initial_prediction
            cross_modal_confidence = initial_confidence
            cross_modal_weight = 0.0
        
        # Calculate frame weight
        frame_weight = 1.0 - temporal_weight - cross_modal_weight
        
        # Calculate final prediction
        final_prediction = (
            frame_weight * initial_prediction +
            temporal_weight * temporal_prediction +
            cross_modal_weight * cross_modal_prediction
        )
        
        # Calculate final confidence
        final_confidence = (
            frame_weight * initial_confidence +
            temporal_weight * temporal_confidence +
            cross_modal_weight * cross_modal_confidence
        )
        
        # Apply consistency adjustment if available
        if temporal_results and "final_consistency_score" in temporal_results:
            consistency = temporal_results["final_consistency_score"]
            
            # If highly inconsistent, increase deepfake probability
            if consistency < 0.5 and final_prediction < 0.8:
                # Adjust prediction based on inconsistency
                inconsistency_factor = (0.5 - consistency) * 2  # Scale to [0, 1]
                adjustment = inconsistency_factor * 0.2  # Max adjustment of 0.2
                final_prediction = min(0.95, final_prediction + adjustment)
            
            # Adjust confidence based on consistency
            if consistency < 0.7:
                # Reduce confidence based on inconsistency
                reduction_factor = (0.7 - consistency) / 0.7  # Scale to [0, 1]
                confidence_reduction = reduction_factor * 0.3  # Max reduction of 0.3
                final_confidence = max(0.5, final_confidence - confidence_reduction)
        
        # Apply lip sync adjustment if available
        if cross_modal_results and "lip_sync_score" in cross_modal_results:
            lip_sync_inconsistency = cross_modal_results["lip_sync_score"]
            
            # If lip sync is inconsistent, increase deepfake probability
            if lip_sync_inconsistency > 0.6 and final_prediction < 0.8:
                # Adjust prediction based on lip sync inconsistency
                inconsistency_factor = (lip_sync_inconsistency - 0.6) * 2.5  # Scale to [0, 1]
                adjustment = inconsistency_factor * 0.15  # Max adjustment of 0.15
                final_prediction = min(0.95, final_prediction + adjustment)
        
        # Ensure valid ranges
        final_prediction = min(0.98, max(0.02, final_prediction))
        final_confidence = min(0.95, max(0.5, final_confidence))
        
        return final_prediction, final_confidence
    
    def train(self, 
             train_dir: str,
             val_dir: str,
             epochs: int = 10,
             batch_size: int = 32,
             learning_rate: float = 1e-4,
             use_augmentation: bool = True,
             save_dir: str = None) -> Dict[str, Any]:
        """
        Train or fine-tune the detector on custom data
        
        Args:
            train_dir: Directory containing training data
            val_dir: Directory containing validation data
            epochs: Number of epochs to train
            batch_size: Batch size for training
            learning_rate: Learning rate for training
            use_augmentation: Whether to use advanced augmentation
            save_dir: Directory to save trained models
            
        Returns:
            Dictionary of training results
        """
        from model_trainer import DeepfakeModelTrainer
        
        # Create model trainer
        trainer = DeepfakeModelTrainer(
            model_name="efficientnet_b3",
            input_shape=(224, 224, 3),
            use_temporal=self.use_temporal,
            use_ensemble=self.use_ensemble,
            use_hyperparameter_tuning=True
        )
        
        # Set up augmentation
        if use_augmentation and self.augmenter:
            augmentation_fn = self.augmenter.create_augmentation_pipeline(is_training=True)
            val_augmentation_fn = self.augmenter.create_augmentation_pipeline(is_training=False)
        else:
            augmentation_fn = None
            val_augmentation_fn = None
        
        # Train model
        training_results = trainer.train(
            train_dir=train_dir,
            validation_dir=val_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            augmentation_fn=augmentation_fn,
            val_augmentation_fn=val_augmentation_fn,
            save_dir=save_dir or os.path.join(MODEL_DIR, "enhanced")
        )
        
        # Update ensemble if enabled
        if self.use_ensemble and self.ensemble:
            # Update model weights based on validation performance
            for model_name in self.ensemble.models:
                if model_name in training_results.get("model_metrics", {}):
                    metrics = training_results["model_metrics"][model_name]
                    self.ensemble.update_model_performance(model_name, metrics)
        
        return training_results