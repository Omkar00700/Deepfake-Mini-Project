import numpy as np
import logging
import time
import json
import os
from typing import Tuple, List, Dict, Any, Optional
import concurrent.futures
from model_manager import ModelManager
from preprocessing import preprocess_face, detect_faces
from backend.config import (
    MAX_RETRY_ATTEMPTS, ENABLE_PARALLEL_PROCESSING, MAX_WORKERS,
    CONFIDENCE_MIN_VALUE, CONFIDENCE_MAX_VALUE, MODEL_SAVE_PATH,
    DETECTION_BIAS_CORRECTION, CONFIDENCE_THRESHOLD_FOR_ENSEMBLE
)
from config_manager import config_manager
from inference_core import (
    PredictionResult, FaceRegion, InferenceResult,
    calibrate_confidence, get_prediction_consistency, combine_results
)
from debug_utils import log_inference_data, save_debug_image

# Configure logging
logger = logging.getLogger(__name__)

# Initialize model manager (singleton)
model_manager = ModelManager()

# Initialize ensemble weights with adaptive values based on model performance
# These weights will be dynamically updated based on performance metrics
ADAPTIVE_ENSEMBLE_WEIGHTS = {
    "efficientnet": 0.45,  # Higher weight for EfficientNet (more accurate)
    "xception": 0.35,      # Medium weight for Xception
    "mesonet": 0.20        # Lower weight for MesoNet (less accurate but fast)
}

# Path to store ensemble weights
ENSEMBLE_WEIGHTS_PATH = os.path.join(MODEL_SAVE_PATH, "ensemble_weights.json")

# Load ensemble weights if they exist
def load_ensemble_weights():
    """Load ensemble weights from file"""
    try:
        if os.path.exists(ENSEMBLE_WEIGHTS_PATH):
            with open(ENSEMBLE_WEIGHTS_PATH, "r") as f:
                weights = json.load(f)
                logger.info(f"Loaded ensemble weights: {weights}")
                return weights
        else:
            return ADAPTIVE_ENSEMBLE_WEIGHTS
    except Exception as e:
        logger.error(f"Error loading ensemble weights: {str(e)}")
        return ADAPTIVE_ENSEMBLE_WEIGHTS

# Initialize ensemble weights
ENSEMBLE_WEIGHTS = load_ensemble_weights()

def save_ensemble_weights(weights):
    """Save ensemble weights to file"""
    try:
        os.makedirs(os.path.dirname(ENSEMBLE_WEIGHTS_PATH), exist_ok=True)
        with open(ENSEMBLE_WEIGHTS_PATH, "w") as f:
            json.dump(weights, f, indent=2)
        logger.info(f"Saved ensemble weights: {weights}")
    except Exception as e:
        logger.error(f"Error saving ensemble weights: {str(e)}")

def get_ensemble_weights():
    """Get current ensemble weights"""
    return ENSEMBLE_WEIGHTS

def update_ensemble_weights(weights):
    """Update ensemble weights"""
    global ENSEMBLE_WEIGHTS
    ENSEMBLE_WEIGHTS = weights
    save_ensemble_weights(weights)
    return True

def process_face_with_retry(image, face, max_retries=MAX_RETRY_ATTEMPTS) -> Tuple[FaceRegion, PredictionResult]:
    """
    Process a face with retry logic for robustness
    
    Args:
        image: Source image
        face: Face coordinates (x, y, width, height)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (FaceRegion, PredictionResult)
    """
    x, y, w, h = face
    
    # Skip very small faces
    if w < 30 or h < 30:
        logger.warning(f"Face too small, skipping: {w}x{h}")
        
        face_region = FaceRegion(
            x=int(x),
            y=int(y),
            width=int(w),
            height=int(h),
            probability=0.5,
            status="skipped:too_small"
        )
        
        prediction = PredictionResult(
            probability=0.5,
            confidence=0.1,
            metadata={"status": "skipped:too_small"}
        )
        
        return face_region, prediction
    
    for attempt in range(max_retries):
        try:
            # Get current model
            model = model_manager.get_model()
            
            # Preprocess face
            processed_face = preprocess_face(image, face, model.input_shape[:2])
            
            # Skip if face preprocessing failed
            if processed_face is None:
                logger.warning(f"Failed to preprocess face (attempt {attempt+1}/{max_retries})")
                if attempt == max_retries - 1:
                    face_region = FaceRegion(
                        x=int(x),
                        y=int(y),
                        width=int(w),
                        height=int(h),
                        probability=0.5,
                        status="failed:preprocessing"
                    )
                    
                    prediction = PredictionResult(
                        probability=0.5,
                        confidence=0.1,
                        metadata={"status": "failed:preprocessing"}
                    )
                    
                    return face_region, prediction
                continue
            
            # Save debug image for the preprocessed face
            debug_path = save_debug_image(processed_face, f"preprocessed_face_{x}_{y}")
            logger.debug(f"Saved preprocessed face to {debug_path}")
            
            # Start timing
            start_time = time.time()
            
            # Get prediction from model
            probability = model.predict(processed_face)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log detailed prediction info
            logger.debug(f"Model prediction for face at ({x},{y}): probability={probability:.4f}, model={model.model_name}")
            
            # Apply bias correction for better calibration
            # This helps address potential biases in the model's predictions
            corrected_probability = apply_bias_correction(probability, model.model_name)
            
            # Calculate confidence based on prediction extremity 
            # (how far from 0.5/uncertain it is)
            extremity = abs(corrected_probability - 0.5) * 2  # 0 to 1 range
            base_confidence = 0.5 + extremity * 0.4
            
            # Create prediction result
            prediction = PredictionResult(
                probability=corrected_probability,
                confidence=base_confidence,
                model_name=model.model_name,
                processing_time=processing_time,
                metadata={
                    "original_probability": float(probability),
                    "corrected_probability": float(corrected_probability),
                    "face_quality": assess_face_quality(processed_face)
                }
            )
            
            # Create face region
            face_region = FaceRegion(
                x=int(x),
                y=int(y),
                width=int(w),
                height=int(h),
                prediction=prediction,
                status="success"
            )
            
            # Log inference data for debugging
            log_inference_data(model.model_name, {
                "face_location": {"x": x, "y": y, "width": w, "height": h},
                "original_probability": float(probability),
                "corrected_probability": float(corrected_probability),
                "confidence": float(base_confidence),
                "processing_time_ms": float(processing_time * 1000)
            })
            
            return face_region, prediction
            
        except Exception as e:
            logger.error(f"Error processing face (attempt {attempt+1}/{max_retries}): {str(e)}", exc_info=True)
            if attempt == max_retries - 1:
                face_region = FaceRegion(
                    x=int(x),
                    y=int(y),
                    width=int(w),
                    height=int(h),
                    probability=0.5,
                    status=f"failed:{str(e)}"
                )
                
                prediction = PredictionResult(
                    probability=0.5,
                    confidence=0.1,
                    metadata={"status": f"failed:{str(e)}"}
                )
                
                return face_region, prediction
    
    # Should never reach here, but just in case
    face_region = FaceRegion(
        x=int(x),
        y=int(y),
        width=int(w),
        height=int(h),
        probability=0.5,
        status="failed:unknown"
    )
    
    prediction = PredictionResult(
        probability=0.5,
        confidence=0.1,
        metadata={"status": "failed:unknown"}
    )
    
    return face_region, prediction

def assess_face_quality(face_image) -> float:
    """
    Assess the quality of a face image for more reliable predictions
    Returns a score between 0 (poor quality) and 1 (excellent quality)
    """
    try:
        # Convert to grayscale for analysis
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
            
        # Check face image size (at least 128x128 pixels)
        if gray.shape[0] < 128 or gray.shape[1] < 128:
            return 0.5  # Reduce quality score for small faces
        
        # Calculate blur using variance of Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = np.var(laplacian)
        
        # Normalize blur score (higher is better)
        # Typically, scores below 100 indicate blurry images
        blur_quality = min(blur_score / 500, 1.0)
        
        # Calculate contrast as range of pixel values
        min_val, max_val = np.min(gray), np.max(gray)
        contrast = (max_val - min_val) / 255.0
        
        # Combine scores (weighted average)
        quality_score = 0.6 * blur_quality + 0.4 * contrast
        
        return float(quality_score)
    except Exception as e:
        logger.warning(f"Error assessing face quality: {str(e)}")
        return 0.7  # Default medium-high quality

def apply_bias_correction(probability, model_name):
    """
    Apply model-specific bias correction to raw probability
    Enhanced with better calibration across different models and datasets
    """
    # Apply a small correction factor to adjust for model bias
    # Default correction slightly reduces extreme predictions
    correction = DETECTION_BIAS_CORRECTION
    
    # Model-specific adjustments based on empirical observations
    if model_name == "efficientnet":
        # EfficientNet tends to be slightly biased towards real
        if probability < 0.3:
            # Increase probability slightly for likely-real predictions
            return probability * (1.0 - correction) + 0.05
        elif probability > 0.7:
            # Slight decrease for high deepfake probabilities
            return min(probability + correction * 0.5, 1.0)
    elif model_name == "xception":
        # Xception tends to be more conservative
        if probability > 0.6:
            # Boost high deepfake probabilities slightly
            return min(probability + correction, 1.0)
    
    # Apply general correction that pulls extreme values slightly toward center
    if probability > 0.5:
        return min(probability + correction * (probability - 0.5), 1.0)
    else:
        return max(probability - correction * (0.5 - probability), 0.0)

def process_face_with_ensemble(image, face) -> Tuple[FaceRegion, PredictionResult]:
    """
    Process a face using an ensemble of models for more accurate detection
    Enhanced with dynamic weighting and extensive logging for debugging
    
    Args:
        image: Source image
        face: Face coordinates (x, y, width, height)
        
    Returns:
        Tuple of (FaceRegion, PredictionResult)
    """
    x, y, w, h = face
    
    # Skip very small faces
    if w < 30 or h < 30:
        logger.warning(f"Face too small for ensemble, skipping: {w}x{h}")
        
        face_region = FaceRegion(
            x=int(x),
            y=int(y),
            width=int(w),
            height=int(h),
            probability=0.5,
            confidence=0.1,
            status="skipped:too_small"
        )
        
        prediction = PredictionResult(
            probability=0.5,
            confidence=0.1,
            is_ensemble=False,
            metadata={"status": "skipped:too_small"}
        )
        
        return face_region, prediction
    
    try:
        # Start timing
        start_time = time.time()
        
        # Get all available models
        model_names = model_manager.get_models_info()["available_models"]
        predictions = []
        model_results = {}
        raw_predictions = {}
        
        # Save preprocessed face for debugging
        debug_path = save_debug_image(image[y:y+h, x:x+w], f"ensemble_face_{x}_{y}")
        logger.debug(f"Saved face for ensemble processing to {debug_path}")
        
        # Process with each model
        for model_name in model_names:
            # Temporarily switch to this model
            current_model = model_manager.get_model_by_name(model_name)
            if current_model:
                logger.debug(f"Processing face with model: {model_name}")
                
                # Preprocess face for this specific model
                processed_face = preprocess_face(image, face, current_model.input_shape[:2])
                
                if processed_face is not None:
                    # Get prediction
                    raw_probability = current_model.predict(processed_face)
                    
                    # Apply bias correction
                    corrected_probability = apply_bias_correction(raw_probability, model_name)
                    
                    # Store both raw and corrected probabilities
                    raw_predictions[model_name] = float(raw_probability)
                    model_results[model_name] = float(corrected_probability)
                    predictions.append(corrected_probability)
                    
                    logger.debug(f"Model {model_name} prediction: raw={raw_probability:.4f}, corrected={corrected_probability:.4f}")
                else:
                    logger.warning(f"Failed to preprocess face for model {model_name}")
        
        # End timing
        processing_time = time.time() - start_time
        
        # If we have predictions, aggregate them
        if predictions:
            # Calculate ensemble probability using current weights
            weights = []
            for model_name in model_results.keys():
                # Get weight for this model
                weight = ENSEMBLE_WEIGHTS.get(model_name, 0.2)  # Default weight if not found
                weights.append(weight)
            
            # Normalize weights
            if sum(weights) > 0:
                weights = [w/sum(weights) for w in weights]
            else:
                weights = [1.0/len(predictions)] * len(predictions)
            
            # Calculate weighted average
            model_names_list = list(model_results.keys())
            ensemble_probability = sum(model_results[name] * weight 
                                      for name, weight in zip(model_names_list, weights))
            
            # Calculate confidence based on prediction consistency
            # Higher consistency = higher confidence
            if len(predictions) > 1:
                prediction_consistency = get_prediction_consistency(predictions)
                logger.info(f"Ensemble prediction consistency: {prediction_consistency:.4f}")
                
                # Higher consistency threshold for more reliable predictions
                if prediction_consistency < 0.7:
                    logger.warning(f"Low prediction consistency: {prediction_consistency:.4f}")
                
                confidence = 0.5 + prediction_consistency * 0.4
                
                # Adjust confidence based on extremity of the prediction
                extremity = abs(ensemble_probability - 0.5) * 2
                confidence = confidence * 0.7 + (0.5 + extremity * 0.4) * 0.3
                
                # If models strongly disagree, reduce confidence
                if prediction_consistency < 0.5:
                    confidence *= prediction_consistency * 1.2  # Penalize inconsistency
            else:
                # For single model, base confidence on prediction extremity
                p = predictions[0]
                extremity = abs(p - 0.5) * 2  # How far from 0.5 (uncertain)
                confidence = 0.5 + extremity * 0.4
            
            # Ensure confidence stays within bounds
            confidence = max(CONFIDENCE_MIN_VALUE, min(CONFIDENCE_MAX_VALUE, confidence))
            
            # If prediction is near threshold but not confident, adjust slightly
            if (0.45 < ensemble_probability < 0.55) and confidence < 0.7:
                # Increase real bias for borderline cases (conservative approach)
                ensemble_probability = ensemble_probability * 0.9
                logger.info(f"Adjusting borderline prediction: {ensemble_probability:.4f} (low confidence)")
            
            # Create prediction result
            prediction = PredictionResult(
                probability=ensemble_probability,
                confidence=confidence,
                model_name="ensemble",
                is_ensemble=True,
                model_results=model_results,
                processing_time=processing_time,
                metadata={
                    "prediction_consistency": prediction_consistency if 'prediction_consistency' in locals() else None,
                    "ensemble_weights": {name: weight for name, weight in zip(model_names_list, weights)},
                    "raw_predictions": raw_predictions
                }
            )
            
            # Log detailed ensemble results
            log_inference_data("ensemble", {
                "face_location": {"x": x, "y": y, "width": w, "height": h},
                "individual_predictions": model_results,
                "raw_predictions": raw_predictions,
                "weights": {name: weight for name, weight in zip(model_names_list, weights)},
                "ensemble_probability": float(ensemble_probability),
                "confidence": float(confidence),
                "consistency": prediction_consistency if 'prediction_consistency' in locals() else None,
                "processing_time_ms": float(processing_time * 1000)
            })
            
            # Create face region
            face_region = FaceRegion(
                x=int(x),
                y=int(y),
                width=int(w),
                height=int(h),
                prediction=prediction,
                status="success"
            )
            
            return face_region, prediction
        else:
            # Fall back to single model if ensemble failed
            logger.warning("Ensemble failed, falling back to single model")
            return process_face_with_retry(image, face)
        
    except Exception as e:
        logger.error(f"Error in ensemble processing: {str(e)}", exc_info=True)
        # Fall back to single model
        return process_face_with_retry(image, face)

def process_image_faces(image, faces, use_ensemble=True):
    """
    Process all faces in an image and return probabilities, confidences, and regions
    
    Args:
        image: Source image
        faces: List of face coordinates (x, y, width, height)
        use_ensemble: Whether to use ensemble processing
        
    Returns:
        Tuple of (face_probabilities, face_confidences, regions, prediction_consistency)
    """
    logger.info(f"Processing {len(faces)} faces in image{'with ensemble' if use_ensemble else ''}")
    
    face_probabilities = []
    face_confidences = []
    regions = []
    
    # Process each face
    for face in faces:
        try:
            # Use ensemble or single model based on parameter
            if use_ensemble:
                face_region, prediction = process_face_with_ensemble(image, face)
            else:
                face_region, prediction = process_face_with_retry(image, face)
            
            # Extract probability and confidence
            probability = prediction.probability
            confidence = prediction.confidence
            
            # Add to results if processing was successful
            if face_region.status == "success":
                face_probabilities.append(probability)
                face_confidences.append(confidence)
            
            # Add region regardless of status
            regions.append(face_region.to_dict())
            
        except Exception as e:
            logger.error(f"Error processing face: {str(e)}", exc_info=True)
            # Skip this face
    
    # Calculate prediction consistency
    prediction_consistency = None
    if len(face_probabilities) > 1:
        prediction_consistency = get_prediction_consistency(face_probabilities)
    
    # Log results
    logger.info(f"Processed {len(face_probabilities)}/{len(faces)} faces successfully")
    logger.info(f"Face probabilities: {face_probabilities}")
    logger.info(f"Face confidences: {face_confidences}")
    
    return face_probabilities, face_confidences, regions, prediction_consistency

def process_frame(frame, frame_number, use_ensemble=True):
    """
    Process a video frame and return detection results
    
    Args:
        frame: Video frame image
        frame_number: Frame number in the video
        use_ensemble: Whether to use ensemble processing
        
    Returns:
        Dictionary with frame processing results
    """
    logger.info(f"Processing frame {frame_number}")
    
    try:
        # Save original frame for debugging
        debug_path = save_debug_image(frame, f"frame_{frame_number}")
        logger.debug(f"Saved original frame {frame_number} to {debug_path}")
        
        # Detect faces
        faces = detect_faces(frame)
        
        # If no faces detected
        if not faces:
            logger.warning(f"No faces detected in frame {frame_number}")
            return {
                "frame_number": frame_number,
                "status": "no_faces",
                "regions": [],
                "face_count": 0
            }
        
        # Process faces
        face_probabilities, face_confidences, regions, prediction_consistency = process_image_faces(
            frame, faces, use_ensemble
        )
        
        # Calculate frame metrics
        if face_probabilities:
            # Weight by face size (larger faces more important)
            if len(face_probabilities) > 1:
                weights = [r["width"] * r["height"] for r in regions if "width" in r and "height" in r]
                if sum(weights) > 0:
                    frame_probability = np.average(face_probabilities, weights=weights)
                    frame_confidence = np.average(face_confidences, weights=weights)
                else:
                    frame_probability = np.mean(face_probabilities)
                    frame_confidence = np.mean(face_confidences)
            else:
                frame_probability = face_probabilities[0]
                frame_confidence = face_confidences[0]
            
            # Save frame with detection regions for debugging
            debug_path = save_debug_image(frame, f"frame_{frame_number}_detected", regions)
            logger.debug(f"Saved annotated frame {frame_number} to {debug_path}")
            
            # Log detailed frame results
            logger.info(f"Frame {frame_number}: probability={frame_probability:.4f}, confidence={frame_confidence:.4f}")
            
            return {
                "frame_number": frame_number,
                "status": "processed",
                "probability": float(frame_probability),
                "confidence": float(frame_confidence),
                "regions": regions,
                "face_count": len(faces),
                "prediction_consistency": prediction_consistency
            }
        else:
            logger.warning(f"No faces successfully processed in frame {frame_number}")
            return {
                "frame_number": frame_number,
                "status": "failed",
                "regions": regions,
                "face_count": len(faces)
            }
            
    except Exception as e:
        logger.error(f"Error processing frame {frame_number}: {str(e)}", exc_info=True)
        return {
            "frame_number": frame_number,
            "status": "error",
            "error": str(e)
        }

def get_model_info():
    """Get information about available models"""
    try:
        model_info = model_manager.get_models_info()
        
        # Add face detector info
        face_detector_info = {
            "name": "MTCNN",
            "confidence_threshold": face_detector.confidence_threshold if hasattr(face_detector, "confidence_threshold") else "N/A"
        }
        
        return {
            "models": model_info,
            "face_detector": face_detector_info,
            "ensemble_weights": get_ensemble_weights(),
            "feature_flags": {
                "ensemble_enabled": ENABLE_PARALLEL_PROCESSING,
                "parallel_processing": ENABLE_PARALLEL_PROCESSING,
                "max_workers": MAX_WORKERS
            }
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return {
            "models": {
                "current_model": "unknown",
                "available_models": []
            },
            "face_detector": "unknown",
            "error": str(e)
        }

def switch_model(model_name):
    """Switch the current model"""
    try:
        success = model_manager.switch_model(model_name)
        if success:
            logger.info(f"Switched to model: {model_name}")
        else:
            logger.error(f"Failed to switch to model: {model_name}")
        return success
    except Exception as e:
        logger.error(f"Error switching model: {str(e)}")
        return False
