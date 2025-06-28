"""
Metrics Logger for DeepDefend
Logs metrics for model performance and system usage
"""

import os
import json
import time
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

def log_detection_metrics(detection_id, metrics):
    """
    Log metrics for a detection
    
    Args:
        detection_id: ID of the detection
        metrics: Dictionary of metrics to log
    """
    try:
        # Create metrics directory if it doesn't exist
        metrics_dir = os.path.join(os.path.dirname(__file__), "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Create metrics file path
        metrics_path = os.path.join(metrics_dir, f"detection_{detection_id}.json")
        
        # Add timestamp
        metrics["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Write metrics to file
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Logged metrics for detection {detection_id}")
        
        # Update aggregate metrics
        update_aggregate_metrics(metrics)
        
        return True
    
    except Exception as e:
        logger.error(f"Error logging metrics: {str(e)}")
        return False

def update_aggregate_metrics(metrics):
    """
    Update aggregate metrics with new detection metrics
    
    Args:
        metrics: Dictionary of metrics to add
    """
    try:
        # Create metrics directory if it doesn't exist
        metrics_dir = os.path.join(os.path.dirname(__file__), "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Create aggregate metrics file path
        aggregate_path = os.path.join(metrics_dir, "aggregate.json")
        
        # Load existing aggregate metrics if available
        if os.path.exists(aggregate_path):
            with open(aggregate_path, 'r') as f:
                aggregate = json.load(f)
        else:
            # Initialize aggregate metrics
            aggregate = {
                "total_detections": 0,
                "image_detections": 0,
                "video_detections": 0,
                "deepfake_detections": 0,
                "real_detections": 0,
                "average_confidence": 0,
                "average_processing_time": 0,
                "detection_history": []
            }
        
        # Update aggregate metrics
        aggregate["total_detections"] += 1
        
        # Update detection type counts
        if metrics.get("detection_type") == "image":
            aggregate["image_detections"] += 1
        elif metrics.get("detection_type") == "video":
            aggregate["video_detections"] += 1
        
        # Update deepfake/real counts
        if metrics.get("probability", 0) > 0.5:
            aggregate["deepfake_detections"] += 1
        else:
            aggregate["real_detections"] += 1
        
        # Update averages
        aggregate["average_confidence"] = (
            (aggregate["average_confidence"] * (aggregate["total_detections"] - 1) + metrics.get("confidence", 0)) / 
            aggregate["total_detections"]
        )
        
        aggregate["average_processing_time"] = (
            (aggregate["average_processing_time"] * (aggregate["total_detections"] - 1) + metrics.get("processing_time", 0)) / 
            aggregate["total_detections"]
        )
        
        # Add to detection history (keep last 100)
        aggregate["detection_history"].append({
            "timestamp": metrics.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
            "detection_type": metrics.get("detection_type", "unknown"),
            "probability": metrics.get("probability", 0),
            "confidence": metrics.get("confidence", 0),
            "processing_time": metrics.get("processing_time", 0)
        })
        
        # Limit history to last 100 entries
        if len(aggregate["detection_history"]) > 100:
            aggregate["detection_history"] = aggregate["detection_history"][-100:]
        
        # Add last updated timestamp
        aggregate["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Write aggregate metrics to file
        with open(aggregate_path, 'w') as f:
            json.dump(aggregate, f, indent=2)
        
        logger.debug("Updated aggregate metrics")
        
        return True
    
    except Exception as e:
        logger.error(f"Error updating aggregate metrics: {str(e)}")
        return False

def get_daily_metrics():
    """
    Get metrics for the current day
    
    Returns:
        Dictionary of daily metrics
    """
    try:
        # Create metrics directory if it doesn't exist
        metrics_dir = os.path.join(os.path.dirname(__file__), "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Get current date
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Create daily metrics file path
        daily_path = os.path.join(metrics_dir, f"daily_{today}.json")
        
        # Load existing daily metrics if available
        if os.path.exists(daily_path):
            with open(daily_path, 'r') as f:
                daily = json.load(f)
        else:
            # Initialize daily metrics
            daily = {
                "date": today,
                "total_detections": 0,
                "image_detections": 0,
                "video_detections": 0,
                "deepfake_detections": 0,
                "real_detections": 0,
                "average_confidence": 0,
                "average_processing_time": 0,
                "hourly_breakdown": {str(h): 0 for h in range(24)}
            }
        
        return daily
    
    except Exception as e:
        logger.error(f"Error getting daily metrics: {str(e)}")
        return None

def update_daily_metrics(metrics):
    """
    Update daily metrics with new detection metrics
    
    Args:
        metrics: Dictionary of metrics to add
    """
    try:
        # Get current daily metrics
        daily = get_daily_metrics()
        
        if not daily:
            logger.error("Failed to get daily metrics")
            return False
        
        # Update daily metrics
        daily["total_detections"] += 1
        
        # Update detection type counts
        if metrics.get("detection_type") == "image":
            daily["image_detections"] += 1
        elif metrics.get("detection_type") == "video":
            daily["video_detections"] += 1
        
        # Update deepfake/real counts
        if metrics.get("probability", 0) > 0.5:
            daily["deepfake_detections"] += 1
        else:
            daily["real_detections"] += 1
        
        # Update averages
        daily["average_confidence"] = (
            (daily["average_confidence"] * (daily["total_detections"] - 1) + metrics.get("confidence", 0)) / 
            daily["total_detections"]
        )
        
        daily["average_processing_time"] = (
            (daily["average_processing_time"] * (daily["total_detections"] - 1) + metrics.get("processing_time", 0)) / 
            daily["total_detections"]
        )
        
        # Update hourly breakdown
        hour = datetime.now().hour
        daily["hourly_breakdown"][str(hour)] += 1
        
        # Add last updated timestamp
        daily["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Write daily metrics to file
        metrics_dir = os.path.join(os.path.dirname(__file__), "metrics")
        daily_path = os.path.join(metrics_dir, f"daily_{daily['date']}.json")
        
        with open(daily_path, 'w') as f:
            json.dump(daily, f, indent=2)
        
        logger.debug("Updated daily metrics")
        
        return True
    
    except Exception as e:
        logger.error(f"Error updating daily metrics: {str(e)}")
        return False

def get_metrics_summary():
    """
    Get a summary of all metrics
    
    Returns:
        Dictionary with metrics summary
    """
    try:
        # Create metrics directory if it doesn't exist
        metrics_dir = os.path.join(os.path.dirname(__file__), "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Get aggregate metrics
        aggregate_path = os.path.join(metrics_dir, "aggregate.json")
        
        if os.path.exists(aggregate_path):
            with open(aggregate_path, 'r') as f:
                aggregate = json.load(f)
        else:
            aggregate = {
                "total_detections": 0,
                "last_updated": "Never"
            }
        
        # Get daily metrics
        today = datetime.now().strftime("%Y-%m-%d")
        daily_path = os.path.join(metrics_dir, f"daily_{today}.json")
        
        if os.path.exists(daily_path):
            with open(daily_path, 'r') as f:
                daily = json.load(f)
        else:
            daily = {
                "total_detections": 0,
                "last_updated": "Never"
            }
        
        # Create summary
        summary = {
            "all_time": {
                "total_detections": aggregate.get("total_detections", 0),
                "image_detections": aggregate.get("image_detections", 0),
                "video_detections": aggregate.get("video_detections", 0),
                "deepfake_detections": aggregate.get("deepfake_detections", 0),
                "real_detections": aggregate.get("real_detections", 0),
                "average_confidence": aggregate.get("average_confidence", 0),
                "average_processing_time": aggregate.get("average_processing_time", 0),
                "last_updated": aggregate.get("last_updated", "Never")
            },
            "today": {
                "total_detections": daily.get("total_detections", 0),
                "image_detections": daily.get("image_detections", 0),
                "video_detections": daily.get("video_detections", 0),
                "deepfake_detections": daily.get("deepfake_detections", 0),
                "real_detections": daily.get("real_detections", 0),
                "average_confidence": daily.get("average_confidence", 0),
                "average_processing_time": daily.get("average_processing_time", 0),
                "hourly_breakdown": daily.get("hourly_breakdown", {}),
                "last_updated": daily.get("last_updated", "Never")
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return summary
    
    except Exception as e:
        logger.error(f"Error getting metrics summary: {str(e)}")
        return None