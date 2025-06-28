"""
Debug utilities for DeepDefend
Provides functions for debugging and diagnostics
"""

import os
import json
import time
import logging
import platform
import psutil
import numpy as np
import cv2
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

def create_debug_report(detection_id, detection_result):
    """
    Create a debug report for a detection result
    
    Args:
        detection_id: ID of the detection
        detection_result: Detection result data
        
    Returns:
        Path to the debug report
    """
    try:
        # Create debug directory if it doesn't exist
        debug_dir = os.path.join(os.path.dirname(__file__), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Create report directory
        report_dir = os.path.join(debug_dir, f"report_{detection_id}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Create report data
        report = {
            "detection_id": detection_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "detection_result": detection_result,
            "system_info": get_system_info(),
            "model_info": get_model_info(),
            "processing_details": extract_processing_details(detection_result)
        }
        
        # Save report to file
        report_path = os.path.join(report_dir, "report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save visualization if available
        if "regions" in detection_result and detection_result["regions"]:
            try:
                # This is just a placeholder - in a real implementation,
                # you would generate visualizations based on the detection result
                create_debug_visualizations(detection_result, report_dir)
            except Exception as e:
                logger.error(f"Error creating visualizations: {str(e)}")
        
        logger.info(f"Created debug report for detection {detection_id} at {report_path}")
        
        return report_path
    
    except Exception as e:
        logger.error(f"Error creating debug report: {str(e)}")
        return None

def get_system_info():
    """
    Get system information
    
    Returns:
        Dictionary with system information
    """
    try:
        # Get system information
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "memory": {
                "total": psutil.virtual_memory().total / (1024 ** 3),  # GB
                "available": psutil.virtual_memory().available / (1024 ** 3),  # GB
                "percent_used": psutil.virtual_memory().percent
            },
            "disk": {
                "total": psutil.disk_usage('/').total / (1024 ** 3),  # GB
                "free": psutil.disk_usage('/').free / (1024 ** 3),  # GB
                "percent_used": psutil.disk_usage('/').percent
            },
            "cpu": {
                "cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "percent_used": psutil.cpu_percent(interval=1)
            }
        }
        
        # Try to get GPU information if available
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            system_info["gpu"] = {
                "available": len(gpus) > 0,
                "count": len(gpus),
                "devices": [gpu.name for gpu in gpus] if gpus else []
            }
        except:
            system_info["gpu"] = {
                "available": False,
                "error": "Could not get GPU information"
            }
        
        return system_info
    
    except Exception as e:
        logger.error(f"Error getting system information: {str(e)}")
        return {
            "error": str(e)
        }

def get_model_info():
    """
    Get model information
    
    Returns:
        Dictionary with model information
    """
    try:
        # This is a placeholder - in a real implementation,
        # you would get actual model information
        model_info = {
            "name": "DeepDefend AI",
            "version": "1.0.2",
            "type": "Ensemble",
            "components": [
                {
                    "name": "EfficientNet",
                    "weight": 0.4
                },
                {
                    "name": "Xception",
                    "weight": 0.3
                },
                {
                    "name": "Indian Specialized",
                    "weight": 0.3
                }
            ],
            "input_shape": [224, 224, 3],
            "last_updated": "2023-04-15"
        }
        
        return model_info
    
    except Exception as e:
        logger.error(f"Error getting model information: {str(e)}")
        return {
            "error": str(e)
        }

def extract_processing_details(detection_result):
    """
    Extract processing details from a detection result
    
    Args:
        detection_result: Detection result data
        
    Returns:
        Dictionary with processing details
    """
    try:
        # Extract processing details
        details = {
            "detection_type": detection_result.get("detectionType", "unknown"),
            "probability": detection_result.get("probability", 0),
            "confidence": detection_result.get("confidence", 0),
            "processing_time": detection_result.get("processingTime", 0),
            "frame_count": detection_result.get("frameCount", 1) if detection_result.get("detectionType") == "video" else 1,
            "region_count": len(detection_result.get("regions", [])),
            "metadata": {}
        }
        
        # Extract metadata from regions
        if "regions" in detection_result and detection_result["regions"]:
            first_region = detection_result["regions"][0]
            if "metadata" in first_region:
                details["metadata"] = first_region["metadata"]
            
            # Extract face information
            details["faces"] = []
            for region in detection_result["regions"]:
                face_info = {
                    "box": region.get("box", [0, 0, 0, 0]),
                    "probability": region.get("probability", 0),
                    "confidence": region.get("confidence", 0)
                }
                
                # Add skin tone information if available
                if "metadata" in region and "skin_tone" in region["metadata"] and region["metadata"]["skin_tone"].get("success", False):
                    if "indian_tone" in region["metadata"]["skin_tone"] and region["metadata"]["skin_tone"]["indian_tone"]:
                        face_info["skin_tone"] = region["metadata"]["skin_tone"]["indian_tone"].get("name", "Unknown")
                
                details["faces"].append(face_info)
        
        return details
    
    except Exception as e:
        logger.error(f"Error extracting processing details: {str(e)}")
        return {
            "error": str(e)
        }

def create_debug_visualizations(detection_result, output_dir):
    """
    Create debug visualizations for a detection result
    
    Args:
        detection_result: Detection result data
        output_dir: Directory to save visualizations
        
    Returns:
        List of paths to visualizations
    """
    try:
        # This is a placeholder - in a real implementation,
        # you would create actual visualizations based on the detection result
        
        # Create a simple visualization
        vis_path = os.path.join(output_dir, "visualization.jpg")
        
        # Create a blank image
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Add detection information
        cv2.putText(img, f"Detection ID: {detection_result.get('id', 'unknown')}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, f"Type: {detection_result.get('detectionType', 'unknown')}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, f"Probability: {detection_result.get('probability', 0):.4f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, f"Confidence: {detection_result.get('confidence', 0):.4f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add verdict
        verdict = "DEEPFAKE" if detection_result.get("probability", 0) > 0.5 else "REAL"
        color = (0, 0, 255) if verdict == "DEEPFAKE" else (0, 255, 0)
        cv2.putText(img, f"Verdict: {verdict}", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Add timestamp
        cv2.putText(img, f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add regions information
        region_count = len(detection_result.get("regions", []))
        cv2.putText(img, f"Regions: {region_count}", (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add processing time
        cv2.putText(img, f"Processing Time: {detection_result.get('processingTime', 0):.2f} seconds", (20, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add frame count for videos
        if detection_result.get("detectionType") == "video":
            cv2.putText(img, f"Frames: {detection_result.get('frameCount', 0)}", (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Save image
        cv2.imwrite(vis_path, img)
        
        logger.info(f"Created debug visualization at {vis_path}")
        
        return [vis_path]
    
    except Exception as e:
        logger.error(f"Error creating debug visualizations: {str(e)}")
        return []

def generate_model_diagnostic_report():
    """
    Generate a diagnostic report for the model
    
    Returns:
        Path to the diagnostic report
    """
    try:
        # Create debug directory if it doesn't exist
        debug_dir = os.path.join(os.path.dirname(__file__), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Create report directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(debug_dir, f"model_diagnostic_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Create report data
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": get_system_info(),
            "model_info": get_model_info(),
            "performance_metrics": get_performance_metrics()
        }
        
        # Save report to file
        report_path = os.path.join(report_dir, "model_diagnostic.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated model diagnostic report at {report_path}")
        
        return report_path
    
    except Exception as e:
        logger.error(f"Error generating model diagnostic report: {str(e)}")
        return None

def get_performance_metrics():
    """
    Get performance metrics for the model
    
    Returns:
        Dictionary with performance metrics
    """
    try:
        # This is a placeholder - in a real implementation,
        # you would get actual performance metrics
        metrics = {
            "accuracy": 0.92,
            "precision": 0.93,
            "recall": 0.91,
            "f1_score": 0.92,
            "auc": 0.95,
            "average_processing_time": {
                "image": 0.5,  # seconds
                "video": 2.5   # seconds
            },
            "memory_usage": {
                "peak": 1200,  # MB
                "average": 800  # MB
            },
            "throughput": {
                "images_per_second": 2.0,
                "frames_per_second": 10.0
            }
        }
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        return {
            "error": str(e)
        }