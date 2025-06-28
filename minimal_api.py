"""
Minimal API for Deepfake Detector
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure results folder
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    try:
        # Define available models
        models = [
            {
                "id": "efficientnet",
                "name": "EfficientNet",
                "description": "General-purpose deepfake detection model"
            },
            {
                "id": "xception",
                "name": "Xception",
                "description": "High-accuracy deepfake detection model"
            },
            {
                "id": "indian_specialized",
                "name": "Indian Specialized",
                "description": "Model specialized for Indian faces"
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

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    """Get dashboard data"""
    try:
        # Create mock dashboard data
        dashboard_data = {
            "total_detections": 42,
            "verdicts": {
                "deepfakes": 18,
                "suspicious": 12,
                "authentic": 12
            },
            "models": {
                "efficientnet": 20,
                "xception": 15,
                "indian_specialized": 7
            },
            "avg_confidence": 0.78,
            "avg_processing_time": 1250,
            "recent_detections": [
                {
                    "id": "mock-1",
                    "filename": "sample1.jpg",
                    "probability": 0.85,
                    "confidence": 0.92,
                    "model": "efficientnet"
                },
                {
                    "id": "mock-2",
                    "filename": "sample2.jpg",
                    "probability": 0.32,
                    "confidence": 0.88,
                    "model": "xception"
                },
                {
                    "id": "mock-3",
                    "filename": "sample3.jpg",
                    "probability": 0.67,
                    "confidence": 0.75,
                    "model": "indian_specialized"
                }
            ]
        }
        
        return jsonify({
            "success": True,
            "dashboard": dashboard_data
        })
    
    except Exception as e:
        logger.error(f"Error getting dashboard data: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)