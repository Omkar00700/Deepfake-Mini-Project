"""
API endpoints for explainability features
"""

import os
import logging
import json
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from explainability import DeepfakeExplainer
from backend.auth import token_required
from backend.config import UPLOAD_FOLDER, VISUALIZATION_OUTPUT_DIR

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
explain_api = Blueprint('explain_api', __name__)

# Create explainer
explainer = DeepfakeExplainer()

@explain_api.route('/explain/image', methods=['POST'])
@token_required
def explain_image():
    """
    Generate explanation for an image
    
    Expected request format:
    {
        "image_id": "12345",  # Optional, if already analyzed
        "method": "grad_cam"  # Optional, defaults to grad_cam
    }
    
    With file upload
    """
    try:
        # Check if image file is provided
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "message": "No file provided"
            }), 400
        
        # Get file
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({
                "success": False,
                "message": "Empty file provided"
            }), 400
        
        # Get method
        method = request.form.get('method', 'grad_cam')
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)
        
        # Read image
        image = cv2.imread(temp_path)
        
        # Check if image was loaded successfully
        if image is None:
            return jsonify({
                "success": False,
                "message": "Failed to load image"
            }), 400
        
        # Detect faces
        from detection_handler import detect_faces
        faces = detect_faces(image)
        
        # Check if faces were detected
        if not faces:
            return jsonify({
                "success": False,
                "message": "No faces detected in the image"
            }), 400
        
        # Use the first face for explanation
        face = faces[0]
        
        # Generate explanation
        explanation = explainer.explain_prediction(image, face, method=method)
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify(explanation)
        
    except Exception as e:
        logger.error(f"Error in explain_image: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Error generating explanation: {str(e)}"
        }), 500

@explain_api.route('/explain/frame/<frame_id>', methods=['GET'])
@token_required
def explain_frame(frame_id):
    """
    Generate explanation for a specific video frame
    
    Query parameters:
    - method: Explanation method (grad_cam, integrated_gradients, shap)
    """
    try:
        # Get method
        method = request.args.get('method', 'grad_cam')
        
        # Get frame from database or storage
        from database import get_frame_by_id
        frame_data = get_frame_by_id(frame_id)
        
        if not frame_data:
            return jsonify({
                "success": False,
                "message": f"Frame with ID {frame_id} not found"
            }), 404
        
        # Load image
        image_path = frame_data.get('image_path')
        if not image_path or not os.path.exists(image_path):
            return jsonify({
                "success": False,
                "message": "Frame image not found"
            }), 404
        
        # Read image
        image = cv2.imread(image_path)
        
        # Get face coordinates
        face_coords = frame_data.get('face_coords')
        if not face_coords:
            # Detect faces
            from detection_handler import detect_faces
            faces = detect_faces(image)
            
            # Check if faces were detected
            if not faces:
                return jsonify({
                    "success": False,
                    "message": "No faces detected in the frame"
                }), 400
            
            face_coords = faces[0]
        
        # Generate explanation
        explanation = explainer.explain_prediction(image, face_coords, method=method)
        
        return jsonify(explanation)
        
    except Exception as e:
        logger.error(f"Error in explain_frame: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Error generating explanation: {str(e)}"
        }), 500

@explain_api.route('/explain/methods', methods=['GET'])
def get_explanation_methods():
    """Get available explanation methods"""
    methods = [
        {
            "id": "grad_cam",
            "name": "Grad-CAM",
            "description": "Gradient-weighted Class Activation Mapping highlights regions that influenced the prediction"
        },
        {
            "id": "integrated_gradients",
            "name": "Integrated Gradients",
            "description": "Attributes prediction to input features by integrating gradients along a path"
        },
        {
            "id": "shap",
            "name": "SHAP",
            "description": "SHapley Additive exPlanations provides feature importance based on game theory"
        }
    ]
    
    return jsonify({
        "success": True,
        "methods": methods
    })

@explain_api.route('/explain/visualization/<visualization_id>', methods=['GET'])
@token_required
def get_visualization(visualization_id):
    """
    Get a visualization image by ID
    """
    try:
        # Secure the filename
        visualization_id = secure_filename(visualization_id)
        
        # Check file extension
        if not visualization_id.endswith('.png'):
            visualization_id += '.png'
        
        # Build path
        visualization_path = os.path.join(VISUALIZATION_OUTPUT_DIR, visualization_id)
        
        # Check if file exists
        if not os.path.exists(visualization_path):
            return jsonify({
                "success": False,
                "message": f"Visualization with ID {visualization_id} not found"
            }), 404
        
        # Return file
        return send_file(visualization_path, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Error in get_visualization: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Error retrieving visualization: {str(e)}"
        }), 500