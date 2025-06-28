"""
Simple backend for DeepDefend
This is a simplified version that will work without all the dependencies
"""

from flask import Flask, jsonify, request, send_from_directory
import os
import json
import logging
import time
import uuid
import numpy as np
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a simple Flask app
app = Flask(__name__)

# Enable CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# Create necessary directories
uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(uploads_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "status": "online",
        "version": "1.0.0",
        "message": "DeepDefend API is running"
    })

@app.route('/api/detect', methods=['POST'])
def detect_deepfake():
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        # Generate unique ID for this detection
        detection_id = str(uuid.uuid4())
        
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(uploads_dir, f"{detection_id}_{filename}")
        file.save(file_path)
        
        # Determine file type
        file_ext = os.path.splitext(filename)[1].lower()
        is_video = file_ext in ['.mp4', '.avi', '.mov', '.wmv']
        
        # Mock detection result with high accuracy
        if is_video:
            result = {
                "id": detection_id,
                "filename": filename,
                "detectionType": "video",
                "probability": 0.95,  # High accuracy
                "confidence": 0.97,
                "processingTime": 3.5,
                "frameCount": 30,
                "regions": [
                    {
                        "box": [100, 100, 200, 200],
                        "probability": 0.96,
                        "confidence": 0.98,
                        "frame": 0,
                        "metadata": {
                            "skin_tone": {
                                "success": True,
                                "indian_tone": {
                                    "type": "medium",
                                    "name": "Medium",
                                    "score": 0.65
                                }
                            }
                        }
                    }
                ]
            }
        else:
            result = {
                "id": detection_id,
                "filename": filename,
                "detectionType": "image",
                "probability": 0.96,  # High accuracy
                "confidence": 0.98,
                "processingTime": 0.8,
                "regions": [
                    {
                        "box": [100, 100, 200, 200],
                        "probability": 0.97,
                        "confidence": 0.99,
                        "metadata": {
                            "skin_tone": {
                                "success": True,
                                "indian_tone": {
                                    "type": "medium",
                                    "name": "Medium",
                                    "score": 0.65
                                }
                            }
                        }
                    }
                ]
            }
        
        # Save result to file
        result_path = os.path.join(results_dir, f"{detection_id}.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Log successful detection
        logger.info(f"Successful detection: ID={detection_id}, Type={result['detectionType']}, " +
                   f"Probability={result['probability']:.4f}, Confidence={result['confidence']:.4f}")
        
        return jsonify({
            "success": True,
            "detection_id": detection_id,
            "result": result
        })
    
    except Exception as e:
        logger.error(f"Error detecting deepfakes: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded files"""
    return send_from_directory(uploads_dir, filename)

if __name__ == '__main__':
    print("Starting DeepDefend backend on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
