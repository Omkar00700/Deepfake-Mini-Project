
"""
Core inference engine for DeepDefend
This module contains the core inference logic for DeepDefend,
separate from the face processing and model management functionality.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import math
import cv2

# Configure logging
logger = logging.getLogger(__name__)

# Import configuration
from backend.config import (
    CONFIDENCE_MIN_VALUE,
    CONFIDENCE_MAX_VALUE,
    DEEPFAKE_THRESHOLD
)

from config_manager import config_manager

class PredictionResult:
    """Class representing a model prediction result"""
    
    def __init__(self, 
                 probability: float, 
                 confidence: float = None,
                 model_name: str = None,
                 processing_time: float = None,
                 is_ensemble: bool = False,
                 model_results: Dict[str, float] = None,
                 uncertainty: float = None,
                 raw_predictions: List[float] = None,
                 attention_map: Any = None,
                 metadata: Dict[str, Any] = None):
        """
        Initialize a prediction result
        
        Args:
            probability: Probability of being a deepfake (0-1)
            confidence: Confidence in the prediction (0-1)
            model_name: Name of the model used
            processing_time: Time taken to process in seconds
            is_ensemble: Whether this is an ensemble prediction
            model_results: Individual model results (for ensemble)
            uncertainty: Uncertainty measure from Monte Carlo dropout
            raw_predictions: List of raw predictions from Monte Carlo samples
            attention_map: Attention map highlighting important regions
            metadata: Additional metadata
        """
        self.probability = probability
        
        # Store uncertainty if provided
        self.uncertainty = uncertainty
        self.raw_predictions = raw_predictions or []
        self.attention_map = attention_map
        
        # Calculate confidence if not provided
        if confidence is None:
            confidence = self._calculate_confidence()
            
        self.confidence = min(CONFIDENCE_MAX_VALUE, max(CONFIDENCE_MIN_VALUE, confidence))
        self.model_name = model_name
        self.processing_time = processing_time
        self.is_ensemble = is_ensemble
        self.model_results = model_results or {}
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def _calculate_confidence(self):
        """Calculate confidence based on probability and uncertainty"""
        # Base confidence calculation
        extremity = abs(self.probability - 0.5) * 2
        confidence = 0.5 + extremity * 0.4
        
        # Adjust by uncertainty if available
        if self.uncertainty is not None and self.uncertainty > 0:
            # Higher uncertainty reduces confidence
            uncertainty_factor = max(0, 1 - (self.uncertainty * 5))  # Scale uncertainty effect
            confidence = confidence * uncertainty_factor
        
        # If we have raw predictions, use their distribution to adjust confidence
        if self.raw_predictions and len(self.raw_predictions) > 1:
            # Calculate interquartile range as a measure of consistency
            sorted_preds = sorted(self.raw_predictions)
            q1_idx = int(len(sorted_preds) * 0.25)
            q3_idx = int(len(sorted_preds) * 0.75)
            iqr = sorted_preds[q3_idx] - sorted_preds[q1_idx]
            
            # Large IQR indicates inconsistent predictions, reducing confidence
            iqr_factor = max(0, 1 - (iqr * 2))
            confidence = confidence * (0.7 + 0.3 * iqr_factor)
        
        return confidence
    
    @property
    def is_deepfake(self) -> bool:
        """Whether the prediction classifies the input as a deepfake"""
        threshold = config_manager.get("DEEPFAKE_THRESHOLD", DEEPFAKE_THRESHOLD)
        return self.probability > threshold
    
    @property
    def category(self) -> str:
        """Get the category of the prediction based on probability and confidence"""
        # Consider both probability and confidence in determining category
        if self.probability > 0.8 and self.confidence > 0.7:
            return "deepfake_high"
        elif self.probability > 0.6:
            return "deepfake_medium"
        elif self.probability < 0.2 and self.confidence > 0.7:
            return "real_high"
        elif self.probability < 0.4:
            return "real_medium"
        else:
            return "uncertain"
    
    @property
    def is_uncertain(self) -> bool:
        """Check if the prediction is uncertain based on confidence threshold"""
        return self.confidence < 0.5 or (self.uncertainty is not None and self.uncertainty > 0.2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "probability": float(self.probability),
            "confidence": float(self.confidence),
            "is_deepfake": self.is_deepfake,
            "model_name": self.model_name,
            "processing_time": self.processing_time,
            "is_ensemble": self.is_ensemble,
            "model_results": self.model_results,
            "category": self.category,
            "timestamp": self.timestamp,
        }
        
        # Include uncertainty information if available
        if self.uncertainty is not None:
            result["uncertainty"] = float(self.uncertainty)
            result["is_uncertain"] = self.is_uncertain
            
        # Include attention map if available (as base64 string for compactness)
        if self.attention_map is not None:
            if isinstance(self.attention_map, np.ndarray):
                # Convert numpy array to a more compact representation
                result["has_attention_map"] = True
            else:
                result["attention_map"] = self.attention_map
                result["has_attention_map"] = True
        
        # Add metadata
        if self.metadata:
            result.update(self.metadata)
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionResult':
        """Create from dictionary representation"""
        # Extract known fields
        probability = data.pop("probability", 0.5)
        confidence = data.pop("confidence", None)
        model_name = data.pop("model_name", None)
        processing_time = data.pop("processing_time", None)
        is_ensemble = data.pop("is_ensemble", False)
        model_results = data.pop("model_results", {})
        uncertainty = data.pop("uncertainty", None)
        attention_map = data.pop("attention_map", None)
        
        # Remove other known fields that shouldn't be in metadata
        data.pop("is_deepfake", None)
        data.pop("category", None)
        data.pop("timestamp", None)
        data.pop("has_attention_map", None)
        data.pop("is_uncertain", None)
        
        # Create instance with remaining data as metadata
        return cls(
            probability=probability,
            confidence=confidence,
            model_name=model_name,
            processing_time=processing_time,
            is_ensemble=is_ensemble,
            model_results=model_results,
            uncertainty=uncertainty,
            attention_map=attention_map,
            metadata=data
        )


class FaceRegion:
    """Class representing a detected face region with prediction results"""
    
    def __init__(self,
                 x: int,
                 y: int,
                 width: int,
                 height: int,
                 probability: float = 0.5,
                 confidence: float = None,
                 prediction: PredictionResult = None,
                 frame: Optional[int] = None,
                 status: str = "success",
                 quality_score: float = None):
        """
        Initialize a face region
        
        Args:
            x: X coordinate
            y: Y coordinate
            width: Width of the region
            height: Height of the region
            probability: Probability of being a deepfake
            confidence: Confidence in the prediction
            prediction: Full prediction result (if available)
            frame: Frame number (for videos)
            status: Processing status
            quality_score: Score indicating face image quality (0-1)
        """
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)
        
        if prediction is not None:
            self.prediction = prediction
        else:
            self.prediction = PredictionResult(
                probability=probability,
                confidence=confidence
            )
            
        self.frame = frame
        self.status = status
        self.quality_score = quality_score
    
    @property
    def area(self) -> int:
        """Get the area of the region in pixels"""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get the center coordinates of the region"""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def quality_weight(self) -> float:
        """Get weight based on face quality"""
        if self.quality_score is not None:
            # Use explicit quality score
            return max(0.5, self.quality_score)
        else:
            # Approximate quality based on size (larger faces typically have better quality)
            # Calculate relative to a reference size (e.g., 150x150)
            reference_area = 150 * 150
            relative_area = min(2.0, self.area / reference_area)
            return max(0.5, min(1.0, relative_area))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "probability": float(self.prediction.probability),
            "confidence": float(self.prediction.confidence),
            "status": self.status
        }
        
        if self.frame is not None:
            result["frame"] = self.frame
        
        if self.quality_score is not None:
            result["quality_score"] = float(self.quality_score)
            
        # Add uncertainty if available
        if hasattr(self.prediction, 'uncertainty') and self.prediction.uncertainty is not None:
            result["uncertainty"] = float(self.prediction.uncertainty)
            
        # Add metadata from the prediction if available
        if hasattr(self.prediction, 'metadata') and self.prediction.metadata:
            if "metadata" not in result:
                result["metadata"] = {}
            result["metadata"].update(self.prediction.metadata)
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FaceRegion':
        """Create from dictionary representation"""
        # Extract prediction data
        probability = data.get("probability", 0.5)
        confidence = data.get("confidence", None)
        uncertainty = data.get("uncertainty", None)
        
        # Create prediction
        prediction = PredictionResult(
            probability=probability,
            confidence=confidence,
            uncertainty=uncertainty
        )
        
        # Extract metadata if available
        if "metadata" in data:
            prediction.metadata = data.get("metadata", {})
        
        return cls(
            x=data.get("x", 0),
            y=data.get("y", 0),
            width=data.get("width", 0),
            height=data.get("height", 0),
            prediction=prediction,
            frame=data.get("frame"),
            status=data.get("status", "success"),
            quality_score=data.get("quality_score")
        )


class InferenceResult:
    """Class representing the overall result of an inference operation"""
    
    def __init__(self,
                 face_regions: List[FaceRegion],
                 overall_prediction: PredictionResult = None,
                 frame_number: Optional[int] = None,
                 face_count: int = None,
                 processing_time: float = None,
                 status: str = "success",
                 quality_scores: Dict[str, float] = None,
                 metadata: Dict[str, Any] = None):
        """
        Initialize an inference result
        
        Args:
            face_regions: List of detected face regions
            overall_prediction: Overall prediction for the image/frame
            frame_number: Frame number (for videos)
            face_count: Number of faces detected
            processing_time: Time taken to process
            status: Processing status
            quality_scores: Dictionary of quality metrics for this result
            metadata: Additional metadata
        """
        self.face_regions = face_regions
        self.face_count = face_count if face_count is not None else len(face_regions)
        self.frame_number = frame_number
        self.processing_time = processing_time
        self.status = status
        self.quality_scores = quality_scores or {}
        self.metadata = metadata or {}
        
        # Calculate overall prediction if not provided
        if overall_prediction is None and face_regions:
            self.overall_prediction = self._calculate_overall_prediction()
        else:
            self.overall_prediction = overall_prediction or PredictionResult(probability=0.5, confidence=0.1)
    
    def _calculate_overall_prediction(self) -> PredictionResult:
        """Calculate overall prediction from face regions with advanced weighting"""
        if not self.face_regions:
            return PredictionResult(probability=0.5, confidence=0.1)
            
        # Get probabilities, confidences, and uncertainties
        probabilities = [r.prediction.probability for r in self.face_regions]
        confidences = [r.prediction.confidence for r in self.face_regions]
        uncertainties = []
        
        for r in self.face_regions:
            if hasattr(r.prediction, 'uncertainty') and r.prediction.uncertainty is not None:
                uncertainties.append(r.prediction.uncertainty)
            else:
                # Approximate uncertainty from confidence
                uncertainties.append(1.0 - r.prediction.confidence)
        
        # Use multiple factors for weighting:
        # 1. Face size (larger faces are more important)
        # 2. Face quality (higher quality faces are more reliable)
        # 3. Prediction confidence (higher confidence predictions are more reliable)
        size_weights = [r.area for r in self.face_regions]
        quality_weights = [r.quality_weight for r in self.face_regions]
        confidence_weights = [(c + 0.5) for c in confidences]  # Ensure even low confidence has some weight
        
        # Combine weights
        combined_weights = []
        for sw, qw, cw in zip(size_weights, quality_weights, confidence_weights):
            # Size has highest importance, then quality, then confidence
            combined_weights.append(sw * 0.5 + qw * 0.3 + cw * 0.2)
        
        # Ensure weights are positive
        combined_weights = [max(0.1, w) for w in combined_weights]
        
        total_weight = sum(combined_weights)
        
        if total_weight == 0:
            # Equal weighting if no valid weights
            combined_weights = [1] * len(self.face_regions)
            total_weight = len(self.face_regions)
        
        # Calculate weighted average probability
        weighted_probability = sum(p * w for p, w in zip(probabilities, combined_weights)) / total_weight
        
        # Calculate prediction consistency
        if len(probabilities) > 1:
            # Higher value = more consistent predictions across faces
            prediction_consistency = 1.0 - np.std(probabilities)
        else:
            prediction_consistency = 1.0
        
        # Calculate weighted confidence and uncertainty
        weighted_confidence = sum(c * w for c, w in zip(confidences, combined_weights)) / total_weight
        weighted_uncertainty = sum(u * w for u, w in zip(uncertainties, combined_weights)) / total_weight
        
        # Adjust confidence based on prediction consistency and uncertainty
        adjusted_confidence = weighted_confidence * (0.5 + 0.5 * prediction_consistency)
        # Further adjust by uncertainty
        adjusted_confidence = adjusted_confidence * (1.0 - min(0.5, weighted_uncertainty))
        
        # Get model names and results
        model_names = set()
        is_ensemble = False
        model_results = {}
        
        for region in self.face_regions:
            if hasattr(region.prediction, "model_name") and region.prediction.model_name:
                model_names.add(region.prediction.model_name)
                
            if hasattr(region.prediction, "is_ensemble") and region.prediction.is_ensemble:
                is_ensemble = True
                
            if hasattr(region.prediction, "model_results") and region.prediction.model_results:
                for model, result in region.prediction.model_results.items():
                    if model in model_results:
                        model_results[model] = (model_results[model] + result) / 2
                    else:
                        model_results[model] = result
        
        # Create detailed metadata for analysis and debugging
        prediction_metadata = {
            "prediction_consistency": prediction_consistency,
            "face_count": self.face_count,
            "weighted_uncertainty": weighted_uncertainty,
            "region_probabilities": probabilities,
            "region_confidences": confidences,
            "region_weights": combined_weights,
            "aggregation_method": "advanced_weighted_average"
        }
        
        # Create final prediction result
        return PredictionResult(
            probability=weighted_probability,
            confidence=adjusted_confidence,
            uncertainty=weighted_uncertainty,
            model_name=", ".join(model_names) if model_names else None,
            is_ensemble=is_ensemble,
            model_results=model_results if model_results else None,
            metadata=prediction_metadata
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "face_count": self.face_count,
            "probability": float(self.overall_prediction.probability),
            "confidence": float(self.overall_prediction.confidence),
            "is_deepfake": self.overall_prediction.is_deepfake,
            "status": self.status,
            "regions": [r.to_dict() for r in self.face_regions]
        }
        
        # Include uncertainty if available
        if hasattr(self.overall_prediction, 'uncertainty') and self.overall_prediction.uncertainty is not None:
            result["uncertainty"] = float(self.overall_prediction.uncertainty)
            
        if self.frame_number is not None:
            result["frame_number"] = self.frame_number
            
        if self.processing_time is not None:
            result["processing_time"] = self.processing_time
            
        # Include quality scores if available
        if self.quality_scores:
            result["quality_scores"] = {k: float(v) for k, v in self.quality_scores.items()}
            
        # Add additional metadata
        result.update(self.metadata)
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InferenceResult':
        """Create from dictionary representation"""
        # Extract and remove known fields
        face_count = data.pop("face_count", 0)
        probability = data.pop("probability", 0.5)
        confidence = data.pop("confidence", None)
        uncertainty = data.pop("uncertainty", None)
        status = data.pop("status", "success")
        regions_data = data.pop("regions", [])
        frame_number = data.pop("frame_number", None)
        processing_time = data.pop("processing_time", None)
        quality_scores = data.pop("quality_scores", {})
        
        # Create face regions
        face_regions = [FaceRegion.from_dict(r) for r in regions_data]
        
        # Create overall prediction
        overall_prediction = PredictionResult(
            probability=probability,
            confidence=confidence,
            uncertainty=uncertainty
        )
        
        # Create instance with remaining data as metadata
        return cls(
            face_regions=face_regions,
            overall_prediction=overall_prediction,
            frame_number=frame_number,
            face_count=face_count,
            processing_time=processing_time,
            status=status,
            quality_scores=quality_scores,
            metadata=data
        )


def calibrate_confidence(raw_confidence: float, face_count: int, 
                      prediction_consistency: float, coverage: float = 1.0,
                      uncertainty: float = None) -> float:
    """
    Calibrates confidence scores to be more reliable and interpretable
    
    Args:
        raw_confidence: Initial confidence score
        face_count: Number of faces detected
        prediction_consistency: Consistency of predictions across faces
        coverage: For videos, the proportion of frames successfully processed
        uncertainty: Uncertainty estimate from Monte Carlo dropout or other methods
        
    Returns:
        Calibrated confidence score
    """
    # Adjust confidence based on multiple factors
    
    # 1. Face count factor: More faces = higher potential confidence
    face_factor = min(1.0, face_count / 3)  # Saturate at 3+ faces
    
    # 2. Consistency factor: More consistent predictions = higher confidence
    consistency_factor = prediction_consistency
    
    # 3. Coverage factor: Higher frame coverage = higher confidence
    coverage_factor = coverage
    
    # 4. Uncertainty factor: Lower uncertainty = higher confidence
    uncertainty_factor = 1.0
    if uncertainty is not None:
        uncertainty_factor = max(0.5, 1.0 - uncertainty)
    
    # Calculate weighted calibration with improved weighting
    calibrated = (
        raw_confidence * 0.45 +       # Base confidence has high weight
        face_factor * 0.15 +          # Face count has moderate weight
        consistency_factor * 0.2 +    # Consistency has significant weight
        coverage_factor * 0.1 +       # Coverage has lower weight
        uncertainty_factor * 0.1      # Uncertainty has meaningful impact
    )
    
    # Apply sigmoid-like function to emphasize differences in the middle range
    # and de-emphasize extreme values
    calibrated = 1.0 / (1.0 + math.exp(-8 * (calibrated - 0.5)))
    
    # Ensure we stay within bounds
    return max(CONFIDENCE_MIN_VALUE, min(CONFIDENCE_MAX_VALUE, calibrated))


def get_prediction_consistency(probabilities: List[float]) -> float:
    """
    Calculate prediction consistency across multiple probability values
    
    Args:
        probabilities: List of prediction probabilities
        
    Returns:
        Consistency score (0-1, higher is more consistent)
    """
    if not probabilities or len(probabilities) < 2:
        return 1.0
    
    # New improved consistency calculation
    # Consider both standard deviation and the distribution shape
    
    # 1. Standard deviation component
    std_dev = np.std(probabilities)
    std_consistency = 1.0 - min(1.0, std_dev * 2)
    
    # 2. Distribution shape component
    # Check if predictions are clustered or spread out
    sorted_probs = sorted(probabilities)
    gaps = [sorted_probs[i+1] - sorted_probs[i] for i in range(len(sorted_probs)-1)]
    max_gap = max(gaps) if gaps else 0
    gap_consistency = 1.0 - min(1.0, max_gap * 2)
    
    # 3. Range component
    prob_range = max(probabilities) - min(probabilities)
    range_consistency = 1.0 - min(1.0, prob_range)
    
    # Combine components with weights
    combined_consistency = (std_consistency * 0.5 + 
                           gap_consistency * 0.3 + 
                           range_consistency * 0.2)
    
    return combined_consistency


def combine_results(results: List[PredictionResult], 
                   weights: List[float] = None,
                   quality_scores: List[float] = None) -> PredictionResult:
    """
    Combine multiple prediction results with enhanced weighting
    
    Args:
        results: List of prediction results to combine
        weights: Optional weights for each result
        quality_scores: Optional quality scores for each result
        
    Returns:
        Combined prediction result
    """
    if not results:
        return PredictionResult(probability=0.5, confidence=0.1)
        
    if len(results) == 1:
        return results[0]
    
    # Initialize weights if not provided
    if weights is None:
        weights = [1.0] * len(results)
    
    # Adjust weights by quality scores if available
    if quality_scores is not None:
        weights = [w * max(0.5, q) for w, q in zip(weights, quality_scores)]
    
    # Further adjust weights by confidence scores
    confidence_weights = [r.confidence for r in results]
    weights = [w * max(0.5, c) for w, c in zip(weights, confidence_weights)]
    
    # Also consider uncertainty if available
    for i, result in enumerate(results):
        if hasattr(result, 'uncertainty') and result.uncertainty is not None:
            # Lower weight for higher uncertainty
            uncertainty_factor = max(0.5, 1.0 - result.uncertainty)
            weights[i] = weights[i] * uncertainty_factor
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        weights = [1.0] * len(results)
        total_weight = len(results)
        
    normalized_weights = [w / total_weight for w in weights]
    
    # Get probabilities and confidences
    probabilities = [r.probability for r in results]
    confidences = [r.confidence for r in results]
    
    # Collect uncertainties if available
    uncertainties = []
    for r in results:
        if hasattr(r, 'uncertainty') and r.uncertainty is not None:
            uncertainties.append(r.uncertainty)
        else:
            # Approximate uncertainty from confidence
            uncertainties.append(1.0 - r.confidence)
    
    # Calculate weighted average probability
    weighted_probability = sum(p * w for p, w in zip(probabilities, normalized_weights))
    
    # Calculate prediction consistency
    prediction_consistency = get_prediction_consistency(probabilities)
    
    # Calculate weighted confidence
    weighted_confidence = sum(c * w for c, w in zip(confidences, normalized_weights))
    
    # Calculate weighted uncertainty
    weighted_uncertainty = sum(u * w for u, w in zip(uncertainties, normalized_weights))
    
    # Adjust confidence based on prediction consistency
    adjusted_confidence = weighted_confidence * (0.5 + 0.5 * prediction_consistency)
    # Further adjust by uncertainty
    adjusted_confidence = adjusted_confidence * (1.0 - min(0.5, weighted_uncertainty))
    
    # Collect model information
    model_names = []
    is_ensemble = any(r.is_ensemble for r in results)
    model_results = {}
    
    for result in results:
        if result.model_name:
            model_names.append(result.model_name)
            
        if result.model_results:
            for model, prob in result.model_results.items():
                if model in model_results:
                    model_results[model] = (model_results[model] + prob) / 2
                else:
                    model_results[model] = prob
    
    # Collect raw predictions for ensemble
    all_raw_predictions = []
    for result in results:
        if hasattr(result, 'raw_predictions') and result.raw_predictions:
            all_raw_predictions.extend(result.raw_predictions)
    
    # Create final combined result
    return PredictionResult(
        probability=weighted_probability,
        confidence=adjusted_confidence,
        uncertainty=weighted_uncertainty,
        model_name=", ".join(set(model_names)) if model_names else None,
        is_ensemble=True,
        model_results=model_results if model_results else None,
        raw_predictions=all_raw_predictions if all_raw_predictions else None,
        metadata={
            "prediction_consistency": prediction_consistency,
            "combined_from": len(results),
            "weights": normalized_weights
        }
    )


def evaluate_input_quality(image):
    """
    Evaluate the quality of an input image for detection reliability
    
    Args:
        image: Input image to evaluate
        
    Returns:
        Dictionary of quality scores
    """
    # Initialize quality metrics
    quality_scores = {}
    
    # 1. Assess blur
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Calculate image blur using Laplacian variance
    blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(1.0, max(0.0, blur_var / 500))  # Normalize
    quality_scores['blur'] = blur_score
    
    # 2. Assess contrast
    min_val, max_val = np.min(gray), np.max(gray)
    contrast = (max_val - min_val) / max(1, max_val + min_val)
    quality_scores['contrast'] = contrast
    
    # 3. Assess brightness
    brightness = np.mean(gray) / 255
    brightness_score = 1.0 - abs(brightness - 0.5) * 2  # 0.5 is ideal brightness
    quality_scores['brightness'] = brightness_score
    
    # 4. Assess noise
    # Apply median filter and compare with original
    median_filtered = cv2.medianBlur(gray, 3)
    noise_level = np.mean(np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32))) / 255
    noise_score = 1.0 - min(1.0, noise_level * 5)
    quality_scores['noise'] = noise_score
    
    # 5. Assess resolution adequacy
    resolution_score = min(1.0, (gray.shape[0] * gray.shape[1]) / (250 * 250))
    quality_scores['resolution'] = resolution_score
    
    # Calculate overall quality score
    overall_score = (
        blur_score * 0.3 + 
        contrast * 0.2 + 
        brightness_score * 0.15 + 
        noise_score * 0.25 + 
        resolution_score * 0.1
    )
    quality_scores['overall'] = overall_score
    
    return quality_scores


def get_uncertainty_threshold(probability: float) -> float:
    """
    Get appropriate uncertainty threshold based on prediction probability
    
    Args:
        probability: Prediction probability
        
    Returns:
        Threshold for uncertainty
    """
    # For probabilities near the decision boundary, we want lower uncertainty thresholds
    # For very clear cases (near 0 or 1), we can tolerate more uncertainty
    if 0.4 <= probability <= 0.6:
        # Near decision boundary - require high certainty
        return 0.1
    elif 0.3 <= probability <= 0.7:
        # Somewhat ambiguous - require moderate certainty
        return 0.15
    else:
        # Clear case - can tolerate more uncertainty
        return 0.2

