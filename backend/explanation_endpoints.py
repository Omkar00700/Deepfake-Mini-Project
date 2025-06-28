
from flask import Blueprint, request, jsonify
import os
import logging
import json
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from explainability import explainer
from preprocessing import detect_faces
from backend.auth import token_required, role_required
from rate_limiter import rate_limit
from backend.config import UPLOAD_FOLDER
from monitoring import monitoring

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
explanation_bp = Blueprint('explanation', __name__, url_prefix='/api')

@explanation_bp.route('/explain', methods=['POST'])
@rate_limit
@token_required
def explain_detection():
    """
    Endpoint to get explainability data for deepfake detection
    """
    try:
        # Start timing for metrics
        import time
        start_time = time.time()

        # Check if image file is provided
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No image provided'
            }), 400

        # Get image file
        image_file = request.files['image']
        
        # Validate filename
        if image_file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No image selected'
            }), 400
            
        # Get explanation method
        method = request.form.get('method', 'grad_cam')
        allowed_methods = ['grad_cam', 'integrated_gradients']
        if method not in allowed_methods:
            return jsonify({
                'success': False,
                'message': f'Invalid explanation method. Allowed methods: {", ".join(allowed_methods)}'
            }), 400
        
        # Save file temporarily
        filename = secure_filename(image_file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(filepath)
        
        # Read image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({
                'success': False,
                'message': 'Failed to read image'
            }), 400
            
        # Detect faces
        faces = detect_faces(image)
        if len(faces) == 0:
            return jsonify({
                'success': False,
                'message': 'No faces detected in the image'
            }), 400
            
        # Process each face and generate explanations
        explanation_results = []
        for face_idx, face in enumerate(faces):
            explanation = explainer.explain_prediction(image, face, method=method)
            explanation['face_index'] = face_idx
            explanation_results.append(explanation)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Record metrics
        monitoring.record_request(
            endpoint='/api/explain',
            method='POST',
            status=200,
            request_size=request.content_length or 0
        )
        
        # Build response
        response = {
            'success': True,
            'message': f'Explanation generated using {method}',
            'explanation_results': explanation_results,
            'processing_time': processing_time,
            'faces_detected': len(faces),
            'explanation_method': method
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in explain_detection: {str(e)}", exc_info=True)
        
        # Record error
        monitoring.record_error(
            error_type="explanation_error",
            severity="error",
            traceback_info=str(e)
        )
        
        return jsonify({
            'success': False,
            'message': f'Error generating explanation: {str(e)}'
        }), 500

@explanation_bp.route('/explain-methods', methods=['GET'])
def get_explanation_methods():
    """
    Endpoint to get available explanation methods
    """
    try:
        # Return available methods with descriptions
        methods = {
            'grad_cam': {
                'name': 'Gradient-weighted Class Activation Mapping',
                'description': 'Visualizes regions of the image that are important for the model\'s prediction by using the gradients flowing into the final convolutional layer.',
                'resources': 'https://arxiv.org/abs/1610.02391'
            },
            'integrated_gradients': {
                'name': 'Integrated Gradients',
                'description': 'Attributes the prediction to input features by integrating the gradients along a path from a baseline (usually zero) to the input.',
                'resources': 'https://arxiv.org/abs/1703.01365'
            }
        }
        
        return jsonify({
            'success': True,
            'methods': methods
        }), 200
        
    except Exception as e:
        logger.error(f"Error in get_explanation_methods: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500
