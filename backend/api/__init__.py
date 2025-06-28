"""
API Blueprint for DeepDefend
"""

from flask import Blueprint, jsonify, request, current_app
import logging
import os
import time
import uuid
from werkzeug.utils import secure_filename

# Create API blueprint
api_bp = Blueprint('api', __name__)

# Configure logging
logger = logging.getLogger(__name__)

@api_bp.route('/detect', methods=['POST'])
def detect_deepfake():
    """
    Detect deepfakes in images or videos
    """
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file provided"
            }), 400
            
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400
            
        # Generate a unique ID for this detection
        detection_id = str(uuid.uuid4())
        
        # Get file extension
        _, file_ext = os.path.splitext(file.filename)
        file_ext = file_ext.lower()
        
        # Determine detection type based on file extension
        if file_ext in ['.jpg', '.jpeg', '.png']:
            detection_type = 'image'
        elif file_ext in ['.mp4', '.avi', '.mov', '.wmv']:
            detection_type = 'video'
        else:
            return jsonify({
                "success": False,
                "error": f"Unsupported file type: {file_ext}"
            }), 400
            
        # Create a secure filename
        filename = f"{detection_id}{file_ext}"
        
        # Save the file to the upload folder
        from backend.config import UPLOAD_FOLDER
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Process the file with enhanced Indian face detection
        try:
            # Try to use enhanced detection handler first
            from enhanced_detection_handler import process_image_enhanced, process_video_enhanced
            
            if detection_type == 'image':
                probability, confidence, regions = process_image_enhanced(filepath)
                result = {
                    "probability": float(probability),
                    "confidence": float(confidence),
                    "regions": regions,
                    "imageName": os.path.basename(file.filename),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "detectionType": detection_type,
                    "model": "enhanced_indian_detection"
                }
            else:  # video
                probability, confidence, frame_count, regions = process_video_enhanced(filepath)
                result = {
                    "probability": float(probability),
                    "confidence": float(confidence),
                    "frameCount": frame_count,
                    "regions": regions,
                    "imageName": os.path.basename(file.filename),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "detectionType": detection_type,
                    "model": "enhanced_indian_detection"
                }
            
            # Save result to database
            from database import save_detection_result
            result_id = save_detection_result(result)
            
            # Add ID to result
            result["id"] = result_id
            
            logger.info(f"Enhanced detection completed successfully: {detection_id}")
        except Exception as e:
            logger.warning(f"Enhanced detection failed, falling back to standard detection: {str(e)}")
            
            # Fall back to standard detection
            from detection_handler import process_media
            result = process_media(filepath, detection_type, detection_id)
        
        # Return the result
        return jsonify({
            "success": True,
            "detection_id": detection_id,
            "detection_type": detection_type,
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Error in detect_deepfake: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Detection failed: {str(e)}"
        }), 500

@api_bp.route('/status', methods=['GET'])
def api_status():
    """
    Get API status
    """
    try:
        # Get model information
        from inference import get_model_info
        model_info = get_model_info()
        
        # Get system metrics
        from metrics import performance_metrics
        metrics = performance_metrics.get_recent_metrics()
        
        return jsonify({
            "status": "online",
            "model_info": model_info,
            "metrics": metrics
        })
        
    except Exception as e:
        logger.error(f"Error in api_status: {str(e)}", exc_info=True)
        return jsonify({
            "status": "degraded",
            "error": str(e)
        }), 500

@api_bp.route('/models', methods=['GET'])
def get_models():
    """
    Get available models
    """
    try:
        # Get model information
        from inference import get_available_models
        models = get_available_models()
        
        return jsonify({
            "success": True,
            "models": models
        })
        
    except Exception as e:
        logger.error(f"Error in get_models: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@api_bp.route('/detection/<detection_id>', methods=['GET'])
def get_detection(detection_id):
    """
    Get detection result by ID
    """
    try:
        # Get detection result from database
        from database import get_detection_result
        result = get_detection_result(detection_id)
        
        if not result:
            return jsonify({
                "success": False,
                "error": f"Detection result with ID {detection_id} not found"
            }), 404
            
        return jsonify({
            "success": True,
            "detection_id": detection_id,
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Error in get_detection: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500