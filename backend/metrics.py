
import time
import logging
import os
import json
import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque
import threading
from backend.config import EVAL_CACHE_TIME

# Configure logging
logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """
    Class to track and provide performance metrics for the deepfake detection system
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PerformanceMetrics, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        
        # Image processing metrics
        self.image_processing_times = deque(maxlen=100)
        self.image_face_counts = deque(maxlen=100)
        self.image_confidence_values = deque(maxlen=100)
        self.image_probability_values = deque(maxlen=100)
        
        # Video processing metrics
        self.video_processing_times = deque(maxlen=100)
        self.video_frame_counts = deque(maxlen=100)
        self.video_face_counts = deque(maxlen=100)
        self.video_confidence_values = deque(maxlen=100)
        self.video_probability_values = deque(maxlen=100)
        
        # Model metrics
        self.model_usage = {}
        
        # Error metrics
        self.error_counts = {}
        
        # Last reset time
        self.last_reset_time = time.time()
        
        logger.info("Performance metrics initialized")
    
    def reset_metrics(self):
        """
        Reset all metrics
        """
        with self._lock:
            # Image processing metrics
            self.image_processing_times.clear()
            self.image_face_counts.clear()
            self.image_confidence_values.clear()
            self.image_probability_values.clear()
            
            # Video processing metrics
            self.video_processing_times.clear()
            self.video_frame_counts.clear()
            self.video_face_counts.clear()
            self.video_confidence_values.clear()
            self.video_probability_values.clear()
            
            # Model metrics
            self.model_usage = {}
            
            # Error metrics
            self.error_counts = {}
            
            # Update reset time
            self.last_reset_time = time.time()
            
            logger.info("Performance metrics reset")
    
    def record_image_metrics(self, processing_time, face_count, confidence, probability, model_name):
        """
        Record metrics for image processing
        """
        with self._lock:
            self.image_processing_times.append(processing_time)
            self.image_face_counts.append(face_count)
            self.image_confidence_values.append(confidence)
            self.image_probability_values.append(probability)
            
            # Update model usage
            self.model_usage[model_name] = self.model_usage.get(model_name, 0) + 1
    
    def record_video_metrics(self, processing_time, frame_count, face_count, confidence, probability, model_name):
        """
        Record metrics for video processing
        """
        with self._lock:
            self.video_processing_times.append(processing_time)
            self.video_frame_counts.append(frame_count)
            self.video_face_counts.append(face_count)
            self.video_confidence_values.append(confidence)
            self.video_probability_values.append(probability)
            
            # Update model usage
            self.model_usage[model_name] = self.model_usage.get(model_name, 0) + 1
    
    def record_error(self, error_type):
        """
        Record an error
        """
        with self._lock:
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def get_image_metrics(self):
        """
        Get metrics for image processing
        """
        with self._lock:
            return {
                "count": len(self.image_processing_times),
                "processing_time": {
                    "avg": np.mean(self.image_processing_times) if self.image_processing_times else 0,
                    "min": np.min(self.image_processing_times) if self.image_processing_times else 0,
                    "max": np.max(self.image_processing_times) if self.image_processing_times else 0
                },
                "face_count": {
                    "avg": np.mean(self.image_face_counts) if self.image_face_counts else 0,
                    "min": np.min(self.image_face_counts) if self.image_face_counts else 0,
                    "max": np.max(self.image_face_counts) if self.image_face_counts else 0
                },
                "confidence": {
                    "avg": np.mean(self.image_confidence_values) if self.image_confidence_values else 0,
                    "min": np.min(self.image_confidence_values) if self.image_confidence_values else 0,
                    "max": np.max(self.image_confidence_values) if self.image_confidence_values else 0
                },
                "probability": {
                    "avg": np.mean(self.image_probability_values) if self.image_probability_values else 0,
                    "min": np.min(self.image_probability_values) if self.image_probability_values else 0,
                    "max": np.max(self.image_probability_values) if self.image_probability_values else 0
                }
            }
    
    def get_video_metrics(self):
        """
        Get metrics for video processing
        """
        with self._lock:
            return {
                "count": len(self.video_processing_times),
                "processing_time": {
                    "avg": np.mean(self.video_processing_times) if self.video_processing_times else 0,
                    "min": np.min(self.video_processing_times) if self.video_processing_times else 0,
                    "max": np.max(self.video_processing_times) if self.video_processing_times else 0
                },
                "frame_count": {
                    "avg": np.mean(self.video_frame_counts) if self.video_frame_counts else 0,
                    "min": np.min(self.video_frame_counts) if self.video_frame_counts else 0,
                    "max": np.max(self.video_frame_counts) if self.video_frame_counts else 0
                },
                "face_count": {
                    "avg": np.mean(self.video_face_counts) if self.video_face_counts else 0,
                    "min": np.min(self.video_face_counts) if self.video_face_counts else 0,
                    "max": np.max(self.video_face_counts) if self.video_face_counts else 0
                },
                "confidence": {
                    "avg": np.mean(self.video_confidence_values) if self.video_confidence_values else 0,
                    "min": np.min(self.video_confidence_values) if self.video_confidence_values else 0,
                    "max": np.max(self.video_confidence_values) if self.video_confidence_values else 0
                },
                "probability": {
                    "avg": np.mean(self.video_probability_values) if self.video_probability_values else 0,
                    "min": np.min(self.video_probability_values) if self.video_probability_values else 0,
                    "max": np.max(self.video_probability_values) if self.video_probability_values else 0
                }
            }
    
    def get_all_metrics(self):
        """
        Get all metrics
        """
        with self._lock:
            return {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "uptime": time.time() - self.last_reset_time,
                "images": self.get_image_metrics(),
                "videos": self.get_video_metrics(),
                "model_usage": self.model_usage,
                "errors": self.error_counts,
                "total_processed": len(self.image_processing_times) + len(self.video_processing_times),
                "error_rate": sum(self.error_counts.values()) / max(1, len(self.image_processing_times) + len(self.video_processing_times))
            }

# Initialize performance metrics singleton
performance_metrics = PerformanceMetrics()
