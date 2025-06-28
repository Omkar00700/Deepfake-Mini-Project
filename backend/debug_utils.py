
"""
Debug utilities for DeepDefend
Comprehensive logging, visualization, and diagnostic tools
"""

import os
import json
import logging
import time
import traceback
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile
import shutil

# Configure logging
logger = logging.getLogger(__name__)

# Create debug directory
DEBUG_DIR = os.environ.get('DEEPDEFEND_DEBUG_DIR', 'debug_output')
if not os.path.exists(DEBUG_DIR):
    try:
        os.makedirs(DEBUG_DIR)
    except Exception as e:
        logger.warning(f"Failed to create debug directory: {e}")
        DEBUG_DIR = tempfile.gettempdir()

# Enable/disable debug features
ENABLE_SAVE_IMAGES = os.environ.get('DEEPDEFEND_SAVE_IMAGES', 'true').lower() == 'true'
ENABLE_INFERENCE_LOGGING = os.environ.get('DEEPDEFEND_LOG_INFERENCE', 'true').lower() == 'true'
ENABLE_VISUALIZATION = os.environ.get('DEEPDEFEND_VISUALIZE', 'true').lower() == 'true'

# Generate a unique session ID
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

def get_debug_dir(detection_id: Optional[str] = None) -> str:
    """
    Get debug directory for a specific detection or the global debug directory
    """
    if detection_id:
        # Create a subdirectory for this detection
        detection_dir = os.path.join(DEBUG_DIR, f"detection_{detection_id}")
        if not os.path.exists(detection_dir):
            try:
                os.makedirs(detection_dir)
            except Exception as e:
                logger.warning(f"Failed to create detection debug directory: {e}")
                return DEBUG_DIR
        return detection_dir
    
    return DEBUG_DIR

def save_debug_image(image, prefix: str, regions: List[Dict[str, Any]] = None, detection_id: Optional[str] = None) -> str:
    """
    Save an image for debugging purposes
    Optionally draw regions on the image for visualization
    
    Args:
        image: The image to save
        prefix: Prefix for the filename
        regions: Optional list of regions to draw on the image
        detection_id: Optional detection ID for organization
        
    Returns:
        Path to the saved image
    """
    if not ENABLE_SAVE_IMAGES:
        return ""
    
    try:
        # Create debug directory
        debug_dir = get_debug_dir(detection_id)
        
        # Create a copy of the image
        img_copy = image.copy()
        
        # Draw regions if provided
        if regions:
            for region in regions:
                x = region.get("x", 0)
                y = region.get("y", 0)
                width = region.get("width", 0)
                height = region.get("height", 0)
                
                # Draw rectangle around the region
                cv2.rectangle(img_copy, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                # If probability is provided, add it as text
                probability = region.get("probability", None)
                if probability is not None:
                    text = f"{probability:.2f}"
                    cv2.putText(img_copy, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = os.path.join(debug_dir, filename)
        
        # Save the image
        cv2.imwrite(filepath, img_copy)
        
        return filepath
    
    except Exception as e:
        logger.warning(f"Failed to save debug image: {e}")
        return ""

def log_inference_data(stage: str, data: Dict[str, Any], detection_id: Optional[str] = None) -> None:
    """
    Log detailed inference data to JSON file for analysis
    
    Args:
        stage: The inference stage (e.g., "preprocessing", "model", "ensemble")
        data: Data to log
        detection_id: Optional detection ID for organization
    """
    if not ENABLE_INFERENCE_LOGGING:
        return
    
    try:
        # Create debug directory
        debug_dir = get_debug_dir(detection_id)
        
        # Add timestamp and stage
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "session_id": SESSION_ID,
            "detection_id": detection_id,
            "data": data
        }
        
        # Process numpy arrays for serialization
        def process_for_json(obj):
            if isinstance(obj, np.ndarray):
                if obj.size > 100:  # For large arrays, just log shape and stats
                    return {
                        "type": "ndarray",
                        "shape": obj.shape,
                        "dtype": str(obj.dtype),
                        "stats": {
                            "min": float(np.min(obj)),
                            "max": float(np.max(obj)),
                            "mean": float(np.mean(obj)),
                            "std": float(np.std(obj))
                        }
                    }
                else:
                    return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: process_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [process_for_json(item) for item in obj]
            return obj
        
        processed_data = process_for_json(log_data)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%H%M%S_%f")
        filename = f"inference_{stage}_{timestamp}.json"
        filepath = os.path.join(debug_dir, filename)
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
    except Exception as e:
        logger.warning(f"Failed to log inference data: {e}")

def visualize_prediction(
    image, 
    probability: float, 
    confidence: float,
    faces: List[Dict[str, Any]],
    heatmap: Optional[np.ndarray] = None,
    detection_id: Optional[str] = None
) -> str:
    """
    Create a visualization of the prediction results
    
    Args:
        image: The original image
        probability: Deepfake probability (0-1)
        confidence: Confidence score (0-1)
        faces: List of face regions with probabilities
        heatmap: Optional attention heatmap
        detection_id: Optional detection ID for organization
        
    Returns:
        Path to the visualization image
    """
    if not ENABLE_VISUALIZATION:
        return ""
    
    try:
        # Create figure with subplots
        fig, ax = plt.subplots(1, 2 if heatmap is not None else 1, figsize=(12, 6))
        
        # Convert single axis to array for consistent indexing
        if heatmap is None:
            ax = [ax]
        
        # Plot original image with face boxes
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[0].set_title(f"Probability: {probability:.4f}, Confidence: {confidence:.4f}")
        
        # Draw face boxes
        for face in faces:
            x, y, w, h = face.get("x", 0), face.get("y", 0), face.get("width", 0), face.get("height", 0)
            face_prob = face.get("probability", 0)
            
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax[0].add_patch(rect)
            
            # Add probability text
            ax[0].text(x, y - 10, f"{face_prob:.2f}", color='white', 
                      backgroundcolor='red', fontsize=8)
        
        ax[0].axis('off')
        
        # Plot heatmap if provided
        if heatmap is not None:
            ax[1].imshow(heatmap, cmap='jet')
            ax[1].set_title("Attention Heatmap")
            ax[1].axis('off')
        
        # Create debug directory
        debug_dir = get_debug_dir(detection_id)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%H%M%S_%f")
        filename = f"visualization_{timestamp}.jpg"
        filepath = os.path.join(debug_dir, filename)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close(fig)
        
        return filepath
    
    except Exception as e:
        logger.warning(f"Failed to create visualization: {e}")
        return ""

def generate_performance_graph(
    metrics: List[Dict[str, Any]],
    output_dir: Optional[str] = None
) -> str:
    """
    Generate a performance graph from collected metrics
    
    Args:
        metrics: List of performance metrics
        output_dir: Optional output directory
        
    Returns:
        Path to the generated graph image
    """
    try:
        if not metrics:
            return ""
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Extract data
        timestamps = [m.get("timestamp", 0) for m in metrics]
        timestamps = [(t - timestamps[0]) / 1000 for t in timestamps]  # Convert to seconds from start
        
        probabilities = [m.get("probability", 0) for m in metrics]
        confidences = [m.get("confidence", 0) for m in metrics]
        processing_times = [m.get("processing_time", 0) for m in metrics]
        
        # Plot probabilities and confidences
        ax1.plot(timestamps, probabilities, 'b-', label='Probability')
        ax1.plot(timestamps, confidences, 'r-', label='Confidence')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Score')
        ax1.set_title('Detection Probabilities and Confidences')
        ax1.legend()
        ax1.grid(True)
        
        # Plot processing times
        ax2.plot(timestamps, processing_times, 'g-')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Processing Time (ms)')
        ax2.set_title('Processing Times')
        ax2.grid(True)
        
        # Determine output directory
        if output_dir is None:
            output_dir = DEBUG_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_graph_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close(fig)
        
        return filepath
    
    except Exception as e:
        logger.warning(f"Failed to generate performance graph: {e}")
        return ""

def create_diagnostic_package(
    detection_id: str,
    output_dir: Optional[str] = None,
    include_images: bool = True
) -> str:
    """
    Create a comprehensive diagnostic package for a detection
    
    Args:
        detection_id: The detection ID
        output_dir: Optional output directory for the package
        include_images: Whether to include images in the package
        
    Returns:
        Path to the created diagnostic package
    """
    try:
        # Get debug directory for this detection
        detection_dir = get_debug_dir(detection_id)
        
        # Create output directory
        if output_dir is None:
            output_dir = DEBUG_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate package filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_name = f"diagnostic_package_{detection_id}_{timestamp}"
        package_dir = os.path.join(output_dir, package_name)
        
        # Create package directory
        os.makedirs(package_dir)
        
        # Collect inference logs
        inference_logs = []
        log_files = [f for f in os.listdir(detection_dir) if f.startswith('inference_') and f.endswith('.json')]
        
        for log_file in log_files:
            try:
                with open(os.path.join(detection_dir, log_file), 'r') as f:
                    log_data = json.load(f)
                    inference_logs.append(log_data)
            except Exception as e:
                logger.warning(f"Failed to read log file {log_file}: {e}")
        
        # Write combined logs
        with open(os.path.join(package_dir, 'inference_logs.json'), 'w') as f:
            json.dump(inference_logs, f, indent=2)
        
        # Include images if requested
        if include_images:
            image_files = [f for f in os.listdir(detection_dir) 
                         if f.endswith(('.jpg', '.png')) and not f.startswith('visualization_')]
            
            if image_files:
                images_dir = os.path.join(package_dir, 'images')
                os.makedirs(images_dir)
                
                for image_file in image_files:
                    shutil.copy(
                        os.path.join(detection_dir, image_file),
                        os.path.join(images_dir, image_file)
                    )
        
        # Include visualizations
        vis_files = [f for f in os.listdir(detection_dir) if f.startswith('visualization_')]
        
        if vis_files:
            vis_dir = os.path.join(package_dir, 'visualizations')
            os.makedirs(vis_dir)
            
            for vis_file in vis_files:
                shutil.copy(
                    os.path.join(detection_dir, vis_file),
                    os.path.join(vis_dir, vis_file)
                )
        
        # Create summary file
        summary = {
            "detection_id": detection_id,
            "timestamp": datetime.now().isoformat(),
            "session_id": SESSION_ID,
            "log_count": len(inference_logs),
            "image_count": len(image_files) if include_images else 0,
            "visualization_count": len(vis_files)
        }
        
        with open(os.path.join(package_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create zip file
        zip_path = shutil.make_archive(
            base_name=package_dir,
            format='zip',
            root_dir=os.path.dirname(package_dir),
            base_dir=os.path.basename(package_dir)
        )
        
        # Clean up temporary directory
        shutil.rmtree(package_dir)
        
        return zip_path
    
    except Exception as e:
        logger.error(f"Failed to create diagnostic package: {e}")
        traceback.print_exc()
        return ""
