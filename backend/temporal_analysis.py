"""
Temporal Analysis for Deepfake Detection
Implements advanced temporal consistency checks for video analysis
"""

import os
import numpy as np
import tensorflow as tf
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import cv2
import time
from pathlib import Path
import json
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
from collections import deque

# Configure logging
logger = logging.getLogger(__name__)

class TemporalAnalyzer:
    """
    Temporal analysis for deepfake detection in videos
    """
    
    def __init__(self, 
                 buffer_size: int = 30,
                 smoothing_window: int = 5,
                 consistency_threshold: float = 0.2,
                 min_sequence_length: int = 10):
        """
        Initialize the temporal analyzer
        
        Args:
            buffer_size: Maximum number of frames to buffer
            smoothing_window: Window size for temporal smoothing
            consistency_threshold: Threshold for temporal consistency
            min_sequence_length: Minimum sequence length for analysis
        """
        self.buffer_size = buffer_size
        self.smoothing_window = smoothing_window
        self.consistency_threshold = consistency_threshold
        self.min_sequence_length = min_sequence_length
        
        # Initialize buffers
        self.prediction_buffer = deque(maxlen=buffer_size)
        self.confidence_buffer = deque(maxlen=buffer_size)
        self.feature_buffer = deque(maxlen=buffer_size)
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # Initialize state
        self.frame_count = 0
        self.last_prediction = 0.5
        self.last_confidence = 0.5
        
        logger.info(f"Initialized temporal analyzer with buffer_size={buffer_size}, "
                   f"smoothing_window={smoothing_window}")
    
    def reset(self):
        """Reset the analyzer state"""
        self.prediction_buffer.clear()
        self.confidence_buffer.clear()
        self.feature_buffer.clear()
        self.frame_buffer.clear()
        self.frame_count = 0
        self.last_prediction = 0.5
        self.last_confidence = 0.5
    
    def add_frame(self, 
                 frame: np.ndarray,
                 prediction: float,
                 confidence: float,
                 features: Optional[np.ndarray] = None) -> None:
        """
        Add a frame to the analysis buffer
        
        Args:
            frame: Video frame
            prediction: Deepfake prediction for this frame
            confidence: Confidence in the prediction
            features: Optional feature vector for this frame
        """
        # Add to buffers
        self.prediction_buffer.append(prediction)
        self.confidence_buffer.append(confidence)
        self.frame_buffer.append(frame)
        
        if features is not None:
            self.feature_buffer.append(features)
        
        # Update state
        self.frame_count += 1
        self.last_prediction = prediction
        self.last_confidence = confidence
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the buffered frames for temporal consistency
        
        Returns:
            Dictionary of analysis results
        """
        # Check if we have enough frames
        if len(self.prediction_buffer) < self.min_sequence_length:
            return {
                "smoothed_prediction": self.last_prediction,
                "temporal_confidence": self.last_confidence,
                "is_temporally_consistent": True,
                "consistency_score": 1.0,
                "frame_count": self.frame_count,
                "sufficient_frames": False
            }
        
        # Convert buffers to numpy arrays
        predictions = np.array(self.prediction_buffer)
        confidences = np.array(self.confidence_buffer)
        
        # Apply temporal smoothing
        smoothed_predictions = self._smooth_predictions(predictions, confidences)
        
        # Calculate temporal consistency
        consistency_score = self._calculate_consistency(predictions)
        
        # Calculate temporal confidence
        temporal_confidence = self._calculate_temporal_confidence(
            predictions, confidences, consistency_score
        )
        
        # Determine if temporally consistent
        is_consistent = consistency_score >= (1.0 - self.consistency_threshold)
        
        # Get final smoothed prediction
        final_prediction = smoothed_predictions[-1]
        
        return {
            "smoothed_prediction": final_prediction,
            "temporal_confidence": temporal_confidence,
            "is_temporally_consistent": is_consistent,
            "consistency_score": consistency_score,
            "frame_count": self.frame_count,
            "sufficient_frames": True
        }
    
    def _smooth_predictions(self, 
                           predictions: np.ndarray,
                           confidences: np.ndarray) -> np.ndarray:
        """
        Apply temporal smoothing to predictions
        
        Args:
            predictions: Array of frame predictions
            confidences: Array of prediction confidences
            
        Returns:
            Smoothed predictions
        """
        # Apply confidence-weighted smoothing
        if len(predictions) >= self.smoothing_window:
            # Use median filter for robustness to outliers
            median_filtered = medfilt(predictions, self.smoothing_window)
            
            # Apply Gaussian smoothing
            gaussian_filtered = gaussian_filter1d(median_filtered, sigma=1.0)
            
            # Apply confidence weighting
            normalized_confidences = confidences / np.sum(confidences)
            for i in range(len(predictions)):
                # Calculate weighted average in a window around i
                start = max(0, i - self.smoothing_window // 2)
                end = min(len(predictions), i + self.smoothing_window // 2 + 1)
                
                if start < end:
                    window_predictions = predictions[start:end]
                    window_confidences = normalized_confidences[start:end]
                    
                    # Renormalize window confidences
                    window_confidences = window_confidences / np.sum(window_confidences)
                    
                    # Calculate weighted average
                    gaussian_filtered[i] = np.sum(window_predictions * window_confidences)
            
            return gaussian_filtered
        else:
            # Not enough frames for smoothing
            return predictions
    
    def _calculate_consistency(self, predictions: np.ndarray) -> float:
        """
        Calculate temporal consistency score
        
        Args:
            predictions: Array of frame predictions
            
        Returns:
            Consistency score (0-1, higher is more consistent)
        """
        if len(predictions) < 2:
            return 1.0
        
        # Calculate frame-to-frame differences
        diffs = np.abs(np.diff(predictions))
        
        # Calculate consistency score (inverse of average difference)
        avg_diff = np.mean(diffs)
        consistency = 1.0 - min(1.0, avg_diff * 5)  # Scale to [0, 1]
        
        # Calculate variance-based consistency
        variance = np.var(predictions)
        var_consistency = 1.0 - min(1.0, variance * 10)  # Scale to [0, 1]
        
        # Combine both metrics
        combined_consistency = 0.7 * consistency + 0.3 * var_consistency
        
        return combined_consistency
    
    def _calculate_temporal_confidence(self, 
                                      predictions: np.ndarray,
                                      confidences: np.ndarray,
                                      consistency_score: float) -> float:
        """
        Calculate confidence based on temporal consistency
        
        Args:
            predictions: Array of frame predictions
            confidences: Array of prediction confidences
            consistency_score: Temporal consistency score
            
        Returns:
            Temporal confidence score (0-1)
        """
        # Base confidence is the average of frame confidences
        base_confidence = np.mean(confidences)
        
        # Adjust based on consistency
        adjusted_confidence = base_confidence * (0.5 + 0.5 * consistency_score)
        
        # Adjust based on number of frames (more frames = higher confidence)
        frame_factor = min(1.0, len(predictions) / self.min_sequence_length)
        
        # Calculate final confidence
        final_confidence = adjusted_confidence * (0.8 + 0.2 * frame_factor)
        
        # Ensure valid range
        final_confidence = min(0.95, max(0.5, final_confidence))
        
        return final_confidence


class TemporalInconsistencyDetector:
    """
    Detect temporal inconsistencies in deepfake videos
    """
    
    def __init__(self, 
                 optical_flow_threshold: float = 0.5,
                 face_motion_threshold: float = 0.3,
                 feature_consistency_threshold: float = 0.7):
        """
        Initialize the temporal inconsistency detector
        
        Args:
            optical_flow_threshold: Threshold for optical flow inconsistency
            face_motion_threshold: Threshold for face motion inconsistency
            feature_consistency_threshold: Threshold for feature consistency
        """
        self.optical_flow_threshold = optical_flow_threshold
        self.face_motion_threshold = face_motion_threshold
        self.feature_consistency_threshold = feature_consistency_threshold
        
        # Initialize optical flow
        self.prev_frame = None
        self.prev_gray = None
        self.flow_buffer = deque(maxlen=10)
        
        logger.info("Initialized temporal inconsistency detector")
    
    def reset(self):
        """Reset the detector state"""
        self.prev_frame = None
        self.prev_gray = None
        self.flow_buffer.clear()
    
    def detect_inconsistencies(self, 
                              frames: List[np.ndarray],
                              face_regions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect temporal inconsistencies in a sequence of frames
        
        Args:
            frames: List of video frames
            face_regions: List of face region dictionaries
            
        Returns:
            Dictionary of inconsistency detection results
        """
        if len(frames) < 2:
            return {
                "inconsistency_score": 0.0,
                "is_inconsistent": False,
                "optical_flow_score": 0.0,
                "face_motion_score": 0.0,
                "sufficient_frames": False
            }
        
        # Calculate optical flow inconsistency
        optical_flow_score = self._calculate_optical_flow_inconsistency(frames)
        
        # Calculate face motion inconsistency
        face_motion_score = 0.0
        if face_regions:
            face_motion_score = self._calculate_face_motion_inconsistency(face_regions)
        
        # Calculate overall inconsistency score
        if face_regions:
            # Use both optical flow and face motion
            inconsistency_score = 0.6 * optical_flow_score + 0.4 * face_motion_score
        else:
            # Use only optical flow
            inconsistency_score = optical_flow_score
        
        # Determine if inconsistent
        is_inconsistent = inconsistency_score > 0.5
        
        return {
            "inconsistency_score": inconsistency_score,
            "is_inconsistent": is_inconsistent,
            "optical_flow_score": optical_flow_score,
            "face_motion_score": face_motion_score,
            "sufficient_frames": True
        }
    
    def _calculate_optical_flow_inconsistency(self, frames: List[np.ndarray]) -> float:
        """
        Calculate optical flow inconsistency score
        
        Args:
            frames: List of video frames
            
        Returns:
            Optical flow inconsistency score (0-1)
        """
        flow_magnitudes = []
        flow_inconsistencies = []
        
        for i in range(1, len(frames)):
            # Convert frames to grayscale
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Calculate flow magnitude
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_magnitude = np.mean(magnitude)
            flow_magnitudes.append(mean_magnitude)
            
            # Calculate flow consistency
            if i > 1:
                # Compare with previous flow
                prev_flow = cv2.calcOpticalFlowFarneback(
                    cv2.cvtColor(frames[i-2], cv2.COLOR_BGR2GRAY),
                    prev_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                # Calculate flow difference
                flow_diff = np.mean(np.abs(flow - prev_flow))
                normalized_diff = min(1.0, flow_diff / (mean_magnitude + 1e-5))
                flow_inconsistencies.append(normalized_diff)
        
        # Calculate inconsistency score
        if flow_inconsistencies:
            # Higher score means more inconsistent
            inconsistency_score = np.mean(flow_inconsistencies)
            
            # Scale to [0, 1]
            inconsistency_score = min(1.0, inconsistency_score * 2)
        else:
            inconsistency_score = 0.0
        
        return inconsistency_score
    
    def _calculate_face_motion_inconsistency(self, face_regions: List[Dict[str, Any]]) -> float:
        """
        Calculate face motion inconsistency score
        
        Args:
            face_regions: List of face region dictionaries
            
        Returns:
            Face motion inconsistency score (0-1)
        """
        if len(face_regions) < 2:
            return 0.0
        
        # Extract face centers
        centers = []
        for region in face_regions:
            x = region.get("x", 0)
            y = region.get("y", 0)
            width = region.get("width", 0)
            height = region.get("height", 0)
            
            center_x = x + width // 2
            center_y = y + height // 2
            centers.append((center_x, center_y))
        
        # Calculate frame-to-frame motion
        motions = []
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i-1][0]
            dy = centers[i][1] - centers[i-1][1]
            motion = np.sqrt(dx**2 + dy**2)
            motions.append(motion)
        
        # Calculate motion consistency
        motion_diffs = []
        for i in range(1, len(motions)):
            motion_diff = abs(motions[i] - motions[i-1])
            normalized_diff = min(1.0, motion_diff / (motions[i] + 1e-5))
            motion_diffs.append(normalized_diff)
        
        # Calculate inconsistency score
        if motion_diffs:
            # Higher score means more inconsistent
            inconsistency_score = np.mean(motion_diffs)
            
            # Scale to [0, 1]
            inconsistency_score = min(1.0, inconsistency_score * 2)
        else:
            inconsistency_score = 0.0
        
        return inconsistency_score


class TemporalConsistencyAnalyzer:
    """
    Comprehensive temporal consistency analyzer for deepfake detection
    """
    
    def __init__(self, 
                 buffer_size: int = 30,
                 use_optical_flow: bool = True,
                 use_face_tracking: bool = True,
                 use_feature_consistency: bool = True):
        """
        Initialize the temporal consistency analyzer
        
        Args:
            buffer_size: Maximum number of frames to buffer
            use_optical_flow: Whether to use optical flow analysis
            use_face_tracking: Whether to use face tracking analysis
            use_feature_consistency: Whether to use feature consistency analysis
        """
        self.buffer_size = buffer_size
        self.use_optical_flow = use_optical_flow
        self.use_face_tracking = use_face_tracking
        self.use_feature_consistency = use_feature_consistency
        
        # Initialize components
        self.temporal_analyzer = TemporalAnalyzer(buffer_size=buffer_size)
        self.inconsistency_detector = TemporalInconsistencyDetector()
        
        # Initialize buffers
        self.frame_buffer = deque(maxlen=buffer_size)
        self.face_region_buffer = deque(maxlen=buffer_size)
        self.prediction_buffer = deque(maxlen=buffer_size)
        self.confidence_buffer = deque(maxlen=buffer_size)
        self.feature_buffer = deque(maxlen=buffer_size)
        
        logger.info(f"Initialized temporal consistency analyzer with buffer_size={buffer_size}")
    
    def reset(self):
        """Reset the analyzer state"""
        self.temporal_analyzer.reset()
        self.inconsistency_detector.reset()
        self.frame_buffer.clear()
        self.face_region_buffer.clear()
        self.prediction_buffer.clear()
        self.confidence_buffer.clear()
        self.feature_buffer.clear()
    
    def add_frame(self, 
                 frame: np.ndarray,
                 prediction: float,
                 confidence: float,
                 face_region: Optional[Dict[str, Any]] = None,
                 features: Optional[np.ndarray] = None) -> None:
        """
        Add a frame to the analysis buffer
        
        Args:
            frame: Video frame
            prediction: Deepfake prediction for this frame
            confidence: Confidence in the prediction
            face_region: Optional face region dictionary
            features: Optional feature vector for this frame
        """
        # Add to temporal analyzer
        self.temporal_analyzer.add_frame(frame, prediction, confidence, features)
        
        # Add to local buffers
        self.frame_buffer.append(frame)
        self.prediction_buffer.append(prediction)
        self.confidence_buffer.append(confidence)
        
        if face_region is not None:
            self.face_region_buffer.append(face_region)
        
        if features is not None:
            self.feature_buffer.append(features)
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the buffered frames for temporal consistency
        
        Returns:
            Dictionary of analysis results
        """
        # Get temporal analysis results
        temporal_results = self.temporal_analyzer.analyze()
        
        # Get inconsistency detection results
        inconsistency_results = {}
        if len(self.frame_buffer) >= 2 and (self.use_optical_flow or self.use_face_tracking):
            face_regions = list(self.face_region_buffer) if self.use_face_tracking else None
            inconsistency_results = self.inconsistency_detector.detect_inconsistencies(
                list(self.frame_buffer), face_regions
            )
        
        # Calculate feature consistency if enabled
        feature_consistency = 1.0
        if self.use_feature_consistency and len(self.feature_buffer) >= 2:
            feature_consistency = self._calculate_feature_consistency()
        
        # Combine results
        combined_results = {
            **temporal_results,
            "feature_consistency": feature_consistency
        }
        
        if inconsistency_results:
            combined_results.update(inconsistency_results)
        
        # Calculate final consistency score
        if "inconsistency_score" in combined_results:
            # Invert inconsistency to get consistency
            flow_consistency = 1.0 - combined_results["inconsistency_score"]
            
            # Combine with temporal consistency
            combined_results["final_consistency_score"] = (
                0.5 * combined_results["consistency_score"] +
                0.3 * flow_consistency +
                0.2 * feature_consistency
            )
        else:
            # Use only temporal and feature consistency
            combined_results["final_consistency_score"] = (
                0.7 * combined_results["consistency_score"] +
                0.3 * feature_consistency
            )
        
        # Calculate final prediction
        combined_results["final_prediction"] = self._calculate_final_prediction(combined_results)
        
        # Calculate final confidence
        combined_results["final_confidence"] = self._calculate_final_confidence(combined_results)
        
        return combined_results
    
    def _calculate_feature_consistency(self) -> float:
        """
        Calculate consistency of feature vectors across frames
        
        Returns:
            Feature consistency score (0-1)
        """
        features = list(self.feature_buffer)
        
        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(1, len(features)):
            # Flatten features if needed
            feat1 = features[i-1].flatten()
            feat2 = features[i].flatten()
            
            # Calculate cosine similarity
            similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-7)
            similarities.append(similarity)
        
        # Calculate mean similarity
        if similarities:
            mean_similarity = np.mean(similarities)
            
            # Scale to [0, 1]
            consistency = (mean_similarity + 1) / 2
        else:
            consistency = 1.0
        
        return consistency
    
    def _calculate_final_prediction(self, results: Dict[str, Any]) -> float:
        """
        Calculate final prediction considering temporal consistency
        
        Args:
            results: Dictionary of analysis results
            
        Returns:
            Final prediction
        """
        # Use smoothed prediction as base
        final_prediction = results["smoothed_prediction"]
        
        # Adjust based on consistency
        if "final_consistency_score" in results:
            consistency = results["final_consistency_score"]
            
            # If highly inconsistent, increase deepfake probability
            if consistency < 0.5 and final_prediction < 0.8:
                # Adjust prediction based on inconsistency
                inconsistency_factor = (0.5 - consistency) * 2  # Scale to [0, 1]
                adjustment = inconsistency_factor * 0.2  # Max adjustment of 0.2
                final_prediction = min(0.95, final_prediction + adjustment)
        
        return final_prediction
    
    def _calculate_final_confidence(self, results: Dict[str, Any]) -> float:
        """
        Calculate final confidence considering temporal consistency
        
        Args:
            results: Dictionary of analysis results
            
        Returns:
            Final confidence
        """
        # Use temporal confidence as base
        base_confidence = results["temporal_confidence"]
        
        # Adjust based on consistency
        if "final_consistency_score" in results:
            consistency = results["final_consistency_score"]
            
            # Lower confidence if inconsistent
            if consistency < 0.7:
                # Reduce confidence based on inconsistency
                reduction_factor = (0.7 - consistency) / 0.7  # Scale to [0, 1]
                confidence_reduction = reduction_factor * 0.3  # Max reduction of 0.3
                adjusted_confidence = base_confidence - confidence_reduction
            else:
                # Slightly increase confidence if consistent
                confidence_boost = (consistency - 0.7) / 0.3 * 0.1  # Max boost of 0.1
                adjusted_confidence = base_confidence + confidence_boost
            
            # Ensure valid range
            final_confidence = min(0.95, max(0.5, adjusted_confidence))
        else:
            final_confidence = base_confidence
        
        return final_confidence