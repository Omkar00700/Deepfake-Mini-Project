
"""
Postprocessing module for DeepDefend
Handles aggregation of detection results and confidence calculation
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
import math
from backend.config import (
    CONFIDENCE_MIN_VALUE, 
    CONFIDENCE_MAX_VALUE,
    TEMPORAL_CONSISTENCY_WEIGHT,
    MIN_FRAMES_FOR_VALID_DETECTION,
    DETECTION_BIAS_CORRECTION
)
from debug_utils import log_inference_data

# Configure logging
logger = logging.getLogger(__name__)

def calculate_image_confidence(probability: float, face_count: int = 1, face_consistency: float = 1.0) -> float:
    """
    Calculate confidence score for an image detection result
    
    Args:
        probability: Deepfake probability (0-1)
        face_count: Number of faces detected
        face_consistency: Consistency between face predictions (0-1)
        
    Returns:
        Confidence score (0-1)
    """
    # Base confidence calculation
    # More extreme probabilities (close to 0 or 1) get higher confidence
    extremity = abs(probability - 0.5) * 2  # 0 to 1 range
    
    # Basic confidence formula: 0.5 (base) + up to 0.4 for extremity
    base_confidence = 0.5 + extremity * 0.4
    
    # Adjust for multiple faces
    # More faces with consistent results = higher confidence
    if face_count > 1:
        # face_consistency is already in 0-1 range
        face_factor = face_consistency * min(face_count / 3, 1.0)  # Cap at 3 faces
        
        # Add up to 0.1 for multiple consistent faces
        base_confidence = min(base_confidence + face_factor * 0.1, 1.0)
    
    # Apply confidence adjustments for different probability ranges
    if probability < 0.3:  # Likely real
        # Slightly reduce confidence for real predictions to avoid overconfidence
        return min(base_confidence * 0.95, 0.95)
    elif probability > 0.7:  # Likely fake
        # Keep high confidence for obvious fakes
        return min(base_confidence, 0.95)
    else:  # Uncertain region
        # Significantly reduce confidence in the uncertain region
        return base_confidence * 0.7

def analyze_temporal_consistency(frame_results: List[Dict[str, Any]]) -> float:
    """
    Analyze temporal consistency across video frames
    Enhanced with better metrics for more accurate assessment
    
    Args:
        frame_results: List of frame processing results
        
    Returns:
        Temporal consistency score (0-1)
        Higher values indicate more consistent predictions across frames
    """
    # Extract probability and confidence values from frame results
    probabilities = []
    confidences = []
    frame_numbers = []
    
    for result in frame_results:
        # Skip frames without valid processing results
        if result.get('status') != 'processed':
            continue
            
        # Get probability for this frame
        frame_prob = result.get('probability')
        if frame_prob is None:
            continue
            
        # Add to lists
        probabilities.append(frame_prob)
        
        # Get confidence if available
        frame_conf = result.get('confidence')
        if frame_conf is not None:
            confidences.append(frame_conf)
            
        # Get frame number if available
        frame_number = result.get('frame_number')
        if frame_number is not None:
            frame_numbers.append(frame_number)
    
    # If we don't have enough data, return a default value
    if len(probabilities) < MIN_FRAMES_FOR_VALID_DETECTION:
        logger.warning(f"Not enough valid frames for temporal analysis: {len(probabilities)} < {MIN_FRAMES_FOR_VALID_DETECTION}")
        return 0.7  # Default value indicating we couldn't properly analyze
    
    # Calculate temporal consistency metrics
    
    # 1. Standard deviation of probabilities across frames
    # Lower std_dev means more consistent predictions
    prob_std = np.std(probabilities)
    
    # 2. Sequential consistency (how much predictions jump between frames)
    sequential_diffs = []
    if frame_numbers and len(frame_numbers) == len(probabilities):
        # Sort probabilities by frame number for proper sequential analysis
        sorted_indices = np.argsort(frame_numbers)
        sorted_probs = [probabilities[i] for i in sorted_indices]
        
        for i in range(1, len(sorted_probs)):
            sequential_diffs.append(abs(sorted_probs[i] - sorted_probs[i-1]))
    else:
        # No frame numbers, use the order they were provided
        for i in range(1, len(probabilities)):
            sequential_diffs.append(abs(probabilities[i] - probabilities[i-1]))
    
    avg_sequential_diff = sum(sequential_diffs) / len(sequential_diffs) if sequential_diffs else 0
    
    # 3. Temporal pattern detection (looking for alternating patterns which could indicate issues)
    temporal_pattern = 1.0  # Default, perfect consistency
    if len(probabilities) >= 5:
        # Apply autocorrelation to find patterns
        # High autocorrelation at specific lags might indicate unusual patterns
        # Simplification: look for alternating patterns (up-down-up-down)
        alternating_count = 0
        for i in range(2, len(probabilities)):
            if ((probabilities[i] > probabilities[i-1] and probabilities[i-1] < probabilities[i-2]) or
                (probabilities[i] < probabilities[i-1] and probabilities[i-1] > probabilities[i-2])):
                alternating_count += 1
        
        alternating_ratio = alternating_count / (len(probabilities) - 2)
        
        # If too many alternating patterns, reduce consistency score
        # Natural videos should show more smooth transitions
        if alternating_ratio > 0.7:
            temporal_pattern = 0.7
    
    # 4. Check for outliers (individual frames with very different predictions)
    # This can indicate problematic frames or detection issues
    outlier_count = 0
    if len(probabilities) >= 4:
        mean_prob = np.mean(probabilities)
        for p in probabilities:
            # Consider a prediction an outlier if it's more than 0.3 away from the mean
            if abs(p - mean_prob) > 0.3:
                outlier_count += 1
                
        # Calculate outlier ratio - if too high, reduce consistency score
        outlier_ratio = outlier_count / len(probabilities)
        outlier_factor = 1.0 - min(outlier_ratio * 2, 0.5)  # Reduce score by up to 50% for outliers
    else:
        outlier_factor = 1.0
    
    # Calculate overall temporal consistency score
    # - Lower std_dev is better (max value for std_dev is about 0.5 for binary classifications)
    # - Lower avg_sequential_diff is better
    # - Higher temporal_pattern is better
    # - Higher outlier_factor is better (fewer outliers)
    
    # Convert std_dev to a 0-1 score (0.5 std_dev -> 0, 0 std_dev -> 1)
    std_score = max(0, 1 - (prob_std * 2))
    
    # Convert avg_sequential_diff to a 0-1 score (0.5 diff -> 0, 0 diff -> 1)
    seq_score = max(0, 1 - (avg_sequential_diff * 2))
    
    # Weighted combination of all factors
    temporal_consistency = (
        (std_score * 0.4) + 
        (seq_score * 0.3) + 
        (temporal_pattern * 0.2) + 
        (outlier_factor * 0.1)
    )
    
    # Log detailed temporal consistency metrics
    logger.info(f"Temporal consistency metrics:")
    logger.info(f"  Standard deviation: {prob_std:.4f} -> Score: {std_score:.4f}")
    logger.info(f"  Sequential difference: {avg_sequential_diff:.4f} -> Score: {seq_score:.4f}")
    logger.info(f"  Temporal pattern: {temporal_pattern:.4f}")
    logger.info(f"  Outlier factor: {outlier_factor:.4f} (from {outlier_count} outliers)")
    logger.info(f"  Overall consistency: {temporal_consistency:.4f}")
    
    # Log data for further analysis
    log_inference_data("temporal_consistency", {
        "probabilities": [float(p) for p in probabilities],
        "std_dev": float(prob_std),
        "std_score": float(std_score),
        "avg_sequential_diff": float(avg_sequential_diff),
        "seq_score": float(seq_score),
        "temporal_pattern": float(temporal_pattern),
        "outlier_factor": float(outlier_factor),
        "outlier_count": outlier_count,
        "overall_consistency": float(temporal_consistency)
    })
    
    return temporal_consistency

def calculate_video_confidence(
    probability: float, 
    processed_frames: int, 
    total_frames: int,
    temporal_consistency: Optional[float] = None
) -> float:
    """
    Calculate confidence score for a video detection result
    Enhanced with better calibration for video-specific issues
    
    Args:
        probability: Deepfake probability (0-1)
        processed_frames: Number of successfully processed frames
        total_frames: Total number of frames analyzed
        temporal_consistency: Consistency of predictions across frames (0-1)
        
    Returns:
        Confidence score (0-1)
    """
    # Base confidence calculation (similar to image confidence)
    extremity = abs(probability - 0.5) * 2  # 0 to 1 range
    base_confidence = 0.5 + extremity * 0.4
    
    # Adjust for frame coverage
    # Higher coverage = higher confidence
    frame_coverage = processed_frames / max(1, total_frames)
    coverage_factor = min(frame_coverage * 1.2, 1.0)  # Boost low coverage slightly
    
    # Adjust confidence based on frame coverage
    confidence = base_confidence * coverage_factor
    
    # Factor in temporal consistency if available
    if temporal_consistency is not None:
        # Weight temporal consistency heavily for videos
        temporal_weight = TEMPORAL_CONSISTENCY_WEIGHT  # From config, typically 0.2-0.3
        confidence = confidence * (1.0 - temporal_weight) + temporal_consistency * temporal_weight
    
    # Apply additional calibrations based on probability range
    if probability < 0.3:  # Likely real
        # Be slightly more conservative for real videos
        confidence = min(confidence * 0.95, 0.95)
    elif probability > 0.7:  # Likely fake
        # Be slightly more confident for fake videos with high probabilities
        confidence = min(confidence * 1.05, 0.95)
    else:  # Uncertain region
        # Be more cautious in the uncertain region
        confidence = confidence * 0.8
    
    # Ensure confidence stays within bounds
    confidence = max(CONFIDENCE_MIN_VALUE, min(CONFIDENCE_MAX_VALUE, confidence))
    
    # Log confidence calculation details
    logger.info(f"Video confidence calculation:")
    logger.info(f"  Base confidence (from probability {probability:.4f}): {base_confidence:.4f}")
    logger.info(f"  Frame coverage factor ({processed_frames}/{total_frames}): {coverage_factor:.4f}")
    logger.info(f"  Temporal consistency factor: {temporal_consistency if temporal_consistency is not None else 'N/A'}")
    logger.info(f"  Final confidence: {confidence:.4f}")
    
    return confidence

def aggregate_video_results(
    frame_results: List[Dict[str, Any]], 
    processed_frame_count: int, 
    total_frames: int,
    temporal_consistency: Optional[float] = None
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Aggregate frame-level results into a video-level result
    Enhanced with better weighting and calibration for more accurate predictions
    
    Args:
        frame_results: List of frame processing results
        processed_frame_count: Number of successfully processed frames
        total_frames: Total number of frames analyzed
        temporal_consistency: Consistency of predictions across frames (0-1)
        
    Returns:
        Tuple of (probability, confidence, regions)
    """
    # Extract probabilities and confidences from frame results
    probabilities = []
    confidences = []
    frame_numbers = []
    all_regions = []
    
    # Collect data from all processed frames
    for result in frame_results:
        # Skip frames without valid processing results
        if result.get('status') != 'processed':
            continue
            
        # Get probability and confidence
        probability = result.get('probability')
        confidence = result.get('confidence')
        
        if probability is not None:
            probabilities.append(probability)
            frame_numbers.append(result.get('frame_number'))
            
            if confidence is not None:
                confidences.append(confidence)
        
        # Collect all regions
        regions = result.get('regions', [])
        for region in regions:
            # Add frame number to region
            region['frame_number'] = result.get('frame_number')
            all_regions.append(region)
    
    # If we have no valid probabilities, return default values
    if not probabilities:
        logger.warning("No valid frame probabilities to aggregate")
        return 0.1, 0.5, all_regions
    
    # Get the number of valid processed frames
    valid_frame_count = len(probabilities)
    
    # Weight frames by confidence if available
    if confidences and len(confidences) == len(probabilities):
        # Calculate quality-weighted average
        # Frames with higher confidence contribute more to the final result
        weights = np.array(confidences)
        # Normalize weights
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
        
        # Calculate weighted average of probabilities
        overall_probability = np.sum(np.array(probabilities) * weights)
        
        logger.info(f"Calculated confidence-weighted probability: {overall_probability:.4f}")
    else:
        # Simple average if confidences not available
        overall_probability = np.mean(probabilities)
        logger.info(f"Calculated average probability: {overall_probability:.4f}")
    
    # Calculate video confidence
    overall_confidence = calculate_video_confidence(
        overall_probability,
        valid_frame_count,
        total_frames,
        temporal_consistency
    )
    
    # Special case: If temporal consistency is very low but probability is extreme
    # This suggests manipulation or inconsistent detection
    if (temporal_consistency is not None and temporal_consistency < 0.4 and 
        (overall_probability > 0.8 or overall_probability < 0.2)):
        logger.warning(f"Extreme probability with poor temporal consistency: adjusting result")
        
        # Pull probability slightly toward the center
        adjustment = (0.5 - overall_probability) * 0.2  # 20% move toward center
        adjusted_probability = overall_probability + adjustment
        
        # Reduce confidence significantly
        adjusted_confidence = overall_confidence * 0.7
        
        logger.info(f"Adjusted probability: {overall_probability:.4f} -> {adjusted_probability:.4f}")
        logger.info(f"Adjusted confidence: {overall_confidence:.4f} -> {adjusted_confidence:.4f}")
        
        overall_probability = adjusted_probability
        overall_confidence = adjusted_confidence
    
    # Add probabilities histogram to the first region's metadata
    if all_regions:
        # Ensure metadata exists
        if 'metadata' not in all_regions[0]:
            all_regions[0]['metadata'] = {}
            
        # Add probabilities histogram
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(probabilities, bins=bins)
        hist_dict = {f"{bins[i]:.1f}-{bins[i+1]:.1f}": int(hist[i]) for i in range(len(hist))}
        
        all_regions[0]['metadata']['probabilities_histogram'] = hist_dict
        all_regions[0]['metadata']['frame_count'] = valid_frame_count
        all_regions[0]['metadata']['temporal_consistency'] = temporal_consistency
    
    # Log detailed aggregation results
    log_inference_data("video_aggregation", {
        "frame_probabilities": [float(p) for p in probabilities],
        "frame_confidences": [float(c) for c in confidences] if confidences else [],
        "overall_probability": float(overall_probability),
        "overall_confidence": float(overall_confidence),
        "temporal_consistency": float(temporal_consistency) if temporal_consistency is not None else None,
        "valid_frames": valid_frame_count,
        "total_frames": total_frames
    })
    
    return overall_probability, overall_confidence, all_regions
