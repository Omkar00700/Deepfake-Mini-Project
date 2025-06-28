
"""
Metrics Logger for DeepDefend
Tracks and logs model performance metrics for continuous evaluation
"""

import os
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional
from collections import deque
import numpy as np
from backend.config import (
    METRICS_STORAGE_PATH,
    METRICS_RETENTION_DAYS,
    ERROR_LOG_MAX_ENTRIES
)

# Configure logging
logger = logging.getLogger(__name__)

class MetricsLogger:
    """Singleton class for logging and tracking performance metrics"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.detection_metrics = []
        self.error_logs = deque(maxlen=ERROR_LOG_MAX_ENTRIES)
        self.metrics_lock = threading.Lock()
        self.last_save_time = time.time()
        
        # Create storage directory if it doesn't exist
        os.makedirs(METRICS_STORAGE_PATH, exist_ok=True)
        
        # Load previous metrics if available
        self._load_metrics()
        
        # Start periodic saving thread
        self._start_periodic_save()
        
        logger.info("Metrics logger initialized")
    
    def _load_metrics(self):
        """Load metrics from storage"""
        try:
            today = time.strftime("%Y-%m-%d")
            metrics_file = os.path.join(METRICS_STORAGE_PATH, f"metrics_{today}.json")
            
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    data = json.load(f)
                    self.detection_metrics = data.get("detection_metrics", [])
                    
                    # Convert deque for error logs
                    error_logs = data.get("error_logs", [])
                    self.error_logs = deque(error_logs, maxlen=ERROR_LOG_MAX_ENTRIES)
                
                logger.info(f"Loaded {len(self.detection_metrics)} metrics and {len(self.error_logs)} error logs")
        except Exception as e:
            logger.error(f"Error loading metrics: {str(e)}")
    
    def _save_metrics(self):
        """Save metrics to storage"""
        try:
            with self.metrics_lock:
                today = time.strftime("%Y-%m-%d")
                metrics_file = os.path.join(METRICS_STORAGE_PATH, f"metrics_{today}.json")
                
                data = {
                    "detection_metrics": self.detection_metrics,
                    "error_logs": list(self.error_logs),
                    "last_updated": time.time()
                }
                
                with open(metrics_file, "w") as f:
                    json.dump(data, f)
                
                self.last_save_time = time.time()
                self._cleanup_old_metrics()
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics files beyond retention period"""
        try:
            now = time.time()
            retention_seconds = METRICS_RETENTION_DAYS * 86400
            
            for filename in os.listdir(METRICS_STORAGE_PATH):
                if filename.startswith("metrics_") and filename.endswith(".json"):
                    file_path = os.path.join(METRICS_STORAGE_PATH, filename)
                    file_age = now - os.path.getmtime(file_path)
                    
                    if file_age > retention_seconds:
                        os.remove(file_path)
                        logger.info(f"Removed old metrics file: {filename}")
        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {str(e)}")
    
    def _start_periodic_save(self):
        """Start a background thread for periodic saving of metrics"""
        def save_worker():
            while True:
                try:
                    # Sleep for some time
                    time.sleep(300)  # Save every 5 minutes
                    
                    # Save metrics if needed
                    if time.time() - self.last_save_time > 300:
                        self._save_metrics()
                except Exception as e:
                    logger.error(f"Error in metrics save worker: {str(e)}")
        
        # Start the save thread
        thread = threading.Thread(target=save_worker, daemon=True)
        thread.start()
    
    def log_detection(self, metrics: Dict[str, Any]):
        """
        Log detection metrics
        
        Args:
            metrics: Dictionary with detection metrics
        """
        with self.metrics_lock:
            # Add timestamp
            metrics["timestamp"] = time.time()
            metrics["timestamp_human"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Add to metrics list
            self.detection_metrics.append(metrics)
            
            # Save periodically (every 50 entries)
            if len(self.detection_metrics) % 50 == 0:
                self._save_metrics()
    
    def log_error(self, error_type: str, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Log an error
        
        Args:
            error_type: Type of error
            message: Error message
            context: Additional context information
        """
        with self.metrics_lock:
            error_log = {
                "type": error_type,
                "message": message,
                "context": context or {},
                "timestamp": time.time(),
                "timestamp_human": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.error_logs.append(error_log)
            
            # Save after logging errors
            self._save_metrics()
    
    def get_recent_metrics(self, limit: int = 100, metric_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent detection metrics
        
        Args:
            limit: Maximum number of metrics to return
            metric_type: Optional filter by metric type (e.g., "image", "video")
            
        Returns:
            List of recent metrics
        """
        with self.metrics_lock:
            if metric_type:
                filtered_metrics = [m for m in self.detection_metrics if m.get("type") == metric_type]
                return filtered_metrics[-limit:]
            else:
                return self.detection_metrics[-limit:]
    
    def get_recent_errors(self, limit: int = 50, error_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent error logs
        
        Args:
            limit: Maximum number of errors to return
            error_type: Optional filter by error type
            
        Returns:
            List of recent errors
        """
        with self.metrics_lock:
            if error_type:
                filtered_errors = [e for e in self.error_logs if e.get("type") == error_type]
                return filtered_errors[-limit:]
            else:
                return list(self.error_logs)[-limit:]
    
    def calculate_performance_metrics(self, days: int = 1) -> Dict[str, Any]:
        """
        Calculate aggregate performance metrics
        
        Args:
            days: Number of days to include in calculation
            
        Returns:
            Dictionary with performance metrics
        """
        with self.metrics_lock:
            now = time.time()
            cutoff_time = now - (days * 86400)
            
            # Filter metrics by time
            recent_metrics = [m for m in self.detection_metrics if m.get("timestamp", 0) >= cutoff_time]
            
            if not recent_metrics:
                return {
                    "count": 0,
                    "period_days": days,
                    "empty": True
                }
            
            # Split by type
            image_metrics = [m for m in recent_metrics if m.get("type") == "image"]
            video_metrics = [m for m in recent_metrics if m.get("type") == "video"]
            
            # Calculate statistics
            stats = {
                "count": len(recent_metrics),
                "period_days": days,
                "image_count": len(image_metrics),
                "video_count": len(video_metrics),
                "average_processing_time": np.mean([m.get("processing_time", 0) for m in recent_metrics]),
                "image_processing_time": np.mean([m.get("processing_time", 0) for m in image_metrics]) if image_metrics else 0,
                "video_processing_time": np.mean([m.get("processing_time", 0) for m in video_metrics]) if video_metrics else 0,
                "average_confidence": np.mean([m.get("confidence", 0) for m in recent_metrics]),
                "errors": len([e for e in self.error_logs if e.get("timestamp", 0) >= cutoff_time]),
                "error_rate": len([e for e in self.error_logs if e.get("timestamp", 0) >= cutoff_time]) / max(1, len(recent_metrics))
            }
            
            # Add video-specific metrics
            if video_metrics:
                stats["video_frames_processed"] = np.mean([m.get("frames_processed", 0) for m in video_metrics])
                stats["video_temporal_consistency"] = np.mean([m.get("temporal_consistency", 0) for m in video_metrics 
                                                             if m.get("temporal_consistency") not in (None, "not_analyzed")])
            
            return stats

# Create singleton instance
_metrics_logger = MetricsLogger()

def log_detection_metrics(metrics: Dict[str, Any]):
    """
    Log detection metrics
    
    Args:
        metrics: Dictionary with detection metrics
    """
    _metrics_logger.log_detection(metrics)

def log_error(error_type: str, message: str, context: Optional[Dict[str, Any]] = None):
    """
    Log an error
    
    Args:
        error_type: Type of error
        message: Error message
        context: Additional context information
    """
    _metrics_logger.log_error(error_type, message, context)

def get_recent_metrics(limit: int = 100, metric_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get recent detection metrics
    
    Args:
        limit: Maximum number of metrics to return
        metric_type: Optional filter by metric type (e.g., "image", "video")
        
    Returns:
        List of recent metrics
    """
    return _metrics_logger.get_recent_metrics(limit, metric_type)

def get_recent_errors(limit: int = 50, error_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get recent error logs
    
    Args:
        limit: Maximum number of errors to return
        error_type: Optional filter by error type
        
    Returns:
        List of recent errors
    """
    return _metrics_logger.get_recent_errors(limit, error_type)

def calculate_performance_metrics(days: int = 1) -> Dict[str, Any]:
    """
    Calculate aggregate performance metrics
    
    Args:
        days: Number of days to include in calculation
        
    Returns:
        Dictionary with performance metrics
    """
    return _metrics_logger.calculate_performance_metrics(days)
