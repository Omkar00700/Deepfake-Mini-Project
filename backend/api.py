"""
Main API for DeepDefend
Provides endpoints for deepfake detection
"""

from flask import Blueprint, jsonify, request, send_file
import os
import json
import logging
import time
import uuid
import numpy as np
import cv2
from werkzeug.utils import secure_filename
from backend.ensemble_detector import EnsembleDeepfakeDetector
from backend.config import (
    ENABLE_ENSEMBLE_DETECTION,
    INDIAN_FACE_DETECTION_ENABLED,
    UPLOAD_FOLDER,
    DEBUG_MODE_ENABLED
)

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__)

# Initialize detector
detector = EnsembleDeepfakeDetector(use_indian_enhancement=INDIAN_FACE_DETECTION_ENABLED)

@api_bp.route('/status', methods=['GET'])
def status():
    """Status endpoint for API"""
    return jsonify({
        "status": "online",
        "module": "api",
        "version": "1.0.0",
        "ensemble_enabled": ENABLE_ENSEMBLE_DETECTION,
        "indian_enhancement": INDIAN_FACE_DETECTION_ENABLED,
        "debug_mode": DEBUG_MODE_ENABLED
    })

@api_bp.route('/detect', methods=['POST'])
def detect():
    """Detect deepfakes in uploaded media"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded"
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400

        # Get parameters
        use_ensemble = request.form.get('use_ensemble', 'true').lower() == 'true'
        use_indian_enhancement = request.form.get('use_indian_enhancement', 'true').lower() == 'true'

        # Generate unique ID for this detection
        detection_id = str(uuid.uuid4())

        # Create upload directory if it doesn't exist
        upload_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), UPLOAD_FOLDER)
        os.makedirs(upload_dir, exist_ok=True)

        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_dir, f"{detection_id}_{filename}")
        file.save(file_path)

        # Determine file type
        file_ext = os.path.splitext(filename)[1].lower()
        is_video = file_ext in ['.mp4', '.avi', '.mov', '.wmv']

        # Process file with enhanced detection
        start_time = time.time()

        if is_video:
            # Process video
            detection_result = detector.detect_video(file_path)
            detection_type = "video"
        else:
            # Process image
            image = cv2.imread(file_path)
            if image is None:
                return jsonify({
                    "success": False,
                    "error": "Failed to read image file"
                }), 400

            detection_result = detector.detect_image(image)
            detection_type = "image"

        # Calculate processing time
        processing_time = time.time() - start_time

        # Create result object
        result = {
            "id": detection_id,
            "filename": filename,
            "detectionType": detection_type,
            "probability": detection_result.get("probability", 0.5),
            "confidence": detection_result.get("confidence", 0.5),
            "processingTime": processing_time,
            "regions": detection_result.get("regions", [])
        }

        # Add video-specific fields
        if is_video:
            result["frameCount"] = detection_result.get("frameCount", 0)
            result["totalFrames"] = detection_result.get("totalFrames", 0)
            result["temporalConsistency"] = detection_result.get("temporalConsistency", 0.0)

        # Save result to database
        save_detection_result(detection_id, result)

        # Log successful detection
        logger.info(f"Successful detection: ID={detection_id}, Type={detection_type}, " +
                   f"Probability={result['probability']:.4f}, Confidence={result['confidence']:.4f}, " +
                   f"Time={processing_time:.2f}s")

        return jsonify({
            "success": True,
            "detection_id": detection_id,
            "result": result
        })

    except Exception as e:
        logger.error(f"Error detecting deepfakes: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@api_bp.route('/result/<detection_id>', methods=['GET'])
def get_result(detection_id):
    """Get detection result by ID"""
    try:
        # Get result from database (placeholder)
        # In a real implementation, this would get from a database
        result = get_detection_result(detection_id)

        if not result:
            return jsonify({
                "success": False,
                "error": f"Detection result with ID {detection_id} not found"
            }), 404

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as e:
        logger.error(f"Error getting detection result: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@api_bp.route('/visualization/<detection_id>', methods=['GET'])
def get_visualization(detection_id):
    """Get visualization for a detection result"""
    try:
        # Get result from database (placeholder)
        # In a real implementation, this would get from a database
        result = get_detection_result(detection_id)

        if not result:
            return jsonify({
                "success": False,
                "error": f"Detection result with ID {detection_id} not found"
            }), 404

        # Create visualization directory if it doesn't exist
        vis_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # Create visualization file path
        vis_path = os.path.join(vis_dir, f"{detection_id}.jpg")

        # Check if visualization already exists
        if not os.path.exists(vis_path):
            # Create visualization (placeholder)
            # In a real implementation, this would create an actual visualization
            create_visualization(result, vis_path)

        # Return visualization
        return send_file(vis_path, mimetype='image/jpeg')

    except Exception as e:
        logger.error(f"Error getting visualization: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@api_bp.route('/models', methods=['GET'])
def get_models():
    """Get available models"""
    try:
        # Get available models (placeholder)
        # In a real implementation, this would get actual models
        models = [
            {
                "id": "efficientnet",
                "name": "EfficientNet",
                "version": "1.0.0",
                "description": "EfficientNet-based deepfake detection model",
                "accuracy": 0.92,
                "input_size": [224, 224, 3]
            },
            {
                "id": "xception",
                "name": "Xception",
                "version": "1.0.0",
                "description": "Xception-based deepfake detection model",
                "accuracy": 0.93,
                "input_size": [299, 299, 3]
            },
            {
                "id": "indian_specialized",
                "name": "Indian Specialized",
                "version": "1.0.0",
                "description": "Specialized model for Indian faces",
                "accuracy": 0.95,
                "input_size": [224, 224, 3]
            }
        ]

        return jsonify({
            "success": True,
            "models": models
        })

    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@api_bp.route('/model/<model_id>', methods=['GET'])
def get_model(model_id):
    """Get model by ID"""
    try:
        # Get model (placeholder)
        # In a real implementation, this would get the actual model
        if model_id == "efficientnet":
            model = {
                "id": "efficientnet",
                "name": "EfficientNet",
                "version": "1.0.0",
                "description": "EfficientNet-based deepfake detection model",
                "accuracy": 0.92,
                "input_size": [224, 224, 3],
                "parameters": 5.3e6,
                "last_updated": "2023-04-15"
            }
        elif model_id == "xception":
            model = {
                "id": "xception",
                "name": "Xception",
                "version": "1.0.0",
                "description": "Xception-based deepfake detection model",
                "accuracy": 0.93,
                "input_size": [299, 299, 3],
                "parameters": 22.9e6,
                "last_updated": "2023-04-15"
            }
        elif model_id == "indian_specialized":
            model = {
                "id": "indian_specialized",
                "name": "Indian Specialized",
                "version": "1.0.0",
                "description": "Specialized model for Indian faces",
                "accuracy": 0.95,
                "input_size": [224, 224, 3],
                "parameters": 6.5e6,
                "last_updated": "2023-04-15"
            }
        else:
            return jsonify({
                "success": False,
                "error": f"Model with ID {model_id} not found"
            }), 404

        return jsonify({
            "success": True,
            "model": model
        })

    except Exception as e:
        logger.error(f"Error getting model: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Helper functions (placeholders)

def save_detection_result(detection_id, result):
    """
    Save detection result to database

    Args:
        detection_id: ID of the detection
        result: Detection result data
    """
    try:
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)

        # Create result file path
        result_path = os.path.join(results_dir, f"{detection_id}.json")

        # Write result to file
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"Saved detection result {detection_id}")

        return True

    except Exception as e:
        logger.error(f"Error saving detection result: {str(e)}")
        return False

def get_detection_result(detection_id):
    """
    Get detection result from database

    Args:
        detection_id: ID of the detection

    Returns:
        Detection result data or None if not found
    """
    try:
        # Create result file path
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        result_path = os.path.join(results_dir, f"{detection_id}.json")

        # Check if file exists
        if not os.path.exists(result_path):
            logger.warning(f"Detection result file not found: {result_path}")
            return None

        # Read result from file
        with open(result_path, 'r') as f:
            result = json.load(f)

        return result

    except Exception as e:
        logger.error(f"Error getting detection result: {str(e)}")
        return None

def create_visualization(result, output_path):
    """
    Create visualization for a detection result

    Args:
        result: Detection result data
        output_path: Path to save the visualization
    """
    try:
        # Create a blank image
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255

        # Add detection information
        cv2.putText(img, f"Detection ID: {result.get('id', 'unknown')}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, f"Type: {result.get('detectionType', 'unknown')}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, f"Probability: {result.get('probability', 0):.4f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, f"Confidence: {result.get('confidence', 0):.4f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Add verdict
        verdict = "DEEPFAKE" if result.get("probability", 0) > 0.5 else "REAL"
        color = (0, 0, 255) if verdict == "DEEPFAKE" else (0, 255, 0)
        cv2.putText(img, f"Verdict: {verdict}", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Add timestamp
        cv2.putText(img, f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Add regions information
        region_count = len(result.get("regions", []))
        cv2.putText(img, f"Regions: {region_count}", (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Add processing time
        cv2.putText(img, f"Processing Time: {result.get('processingTime', 0):.2f} seconds", (20, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Add frame count for videos
        if result.get("detectionType") == "video":
            cv2.putText(img, f"Frames: {result.get('frameCount', 0)}", (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Save image
        cv2.imwrite(output_path, img)

        logger.info(f"Created visualization at {output_path}")

        return True

    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        return False