"""
API Endpoints for Deepfake Detector
Provides API endpoints for the React frontend
"""

import os
import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np

from deepfake_detector import DeepfakeDetector
from report_generator import ReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Add error handler
@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all exceptions"""
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({
        "success": False,
        "error": str(e)
    }), 500

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure results folder
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Configure reports folder
REPORTS_FOLDER = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# Configure visualizations folder
VISUALIZATIONS_FOLDER = os.path.join(os.path.dirname(__file__), "visualizations")
os.makedirs(VISUALIZATIONS_FOLDER, exist_ok=True)

# Set maximum file size (10 MB)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Create detector
detector = DeepfakeDetector(use_ensemble=True, use_indian_enhancement=True)

# Create report generator
report_generator = ReportGenerator()

@app.route('/api/detect', methods=['POST'])
def detect():
    """
    Detect deepfakes in an uploaded file
    """
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
        model_name = request.form.get('model', 'efficientnet')
        use_ensemble = request.form.get('ensemble', 'true').lower() == 'true'
        use_indian_enhancement = request.form.get('indianEnhancement', 'true').lower() == 'true'
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Determine file type
        file_extension = os.path.splitext(filename)[1].lower()
        
        # Create detector with specified parameters
        detector = DeepfakeDetector(
            model_name=model_name,
            use_ensemble=use_ensemble,
            use_indian_enhancement=use_indian_enhancement
        )
        
        # Detect deepfakes
        if file_extension in ['.mp4', '.avi', '.mov', '.wmv']:
            # Video file
            result = detector.detect_video(file_path)
        else:
            # Image file
            result = detector.detect_image(file_path)
        
        if not result["success"]:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 400
        
        # Create visualization
        create_visualization(result["result"])
        
        return jsonify({
            "success": True,
            "detection_id": result["detection_id"],
            "result": result["result"]
        })
    
    except Exception as e:
        logger.error(f"Error detecting deepfakes: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/results/<detection_id>', methods=['GET'])
def get_result(detection_id):
    """
    Get detection result by ID
    """
    try:
        # Get result file path
        result_path = os.path.join(RESULTS_FOLDER, f"{detection_id}.json")
        
        if not os.path.exists(result_path):
            return jsonify({
                "success": False,
                "error": f"Result not found: {detection_id}"
            }), 404
        
        # Load result
        with open(result_path, 'r') as f:
            result = json.load(f)
        
        return jsonify({
            "success": True,
            "result": result
        })
    
    except Exception as e:
        logger.error(f"Error getting result: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/results', methods=['GET'])
def get_results():
    """
    Get all detection results
    """
    try:
        # Get all result files
        result_files = [f for f in os.listdir(RESULTS_FOLDER) if f.endswith(".json")]
        
        # Load results
        results = []
        
        for file in result_files:
            try:
                with open(os.path.join(RESULTS_FOLDER, file), 'r') as f:
                    result = json.load(f)
                
                # Add to results
                results.append({
                    "id": result["id"],
                    "filename": result["filename"],
                    "detectionType": result["detectionType"],
                    "probability": result["probability"],
                    "confidence": result["confidence"],
                    "processingTime": result["processingTime"],
                    "timestamp": os.path.getmtime(os.path.join(RESULTS_FOLDER, file))
                })
            except Exception as e:
                logger.error(f"Error loading result {file}: {str(e)}")
        
        # Sort by timestamp (newest first)
        results = sorted(results, key=lambda x: x["timestamp"], reverse=True)
        
        return jsonify({
            "success": True,
            "results": results
        })
    
    except Exception as e:
        logger.error(f"Error getting results: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/visualization/<detection_id>', methods=['GET'])
def get_visualization(detection_id):
    """
    Get visualization for a detection
    """
    try:
        # Get visualization file path
        vis_path = os.path.join(VISUALIZATIONS_FOLDER, f"{detection_id}.jpg")
        
        if not os.path.exists(vis_path):
            # Get result
            result_path = os.path.join(RESULTS_FOLDER, f"{detection_id}.json")
            
            if not os.path.exists(result_path):
                return jsonify({
                    "success": False,
                    "error": f"Result not found: {detection_id}"
                }), 404
            
            # Load result
            with open(result_path, 'r') as f:
                result = json.load(f)
            
            # Create visualization
            create_visualization(result)
        
        # Return visualization
        return send_file(vis_path, mimetype='image/jpeg')
    
    except Exception as e:
        logger.error(f"Error getting visualization: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/report/<detection_id>', methods=['GET'])
def get_report(detection_id):
    """
    Get report for a detection
    """
    try:
        # Get report format
        report_format = request.args.get('format', 'pdf')
        
        if report_format == 'pdf':
            # Generate PDF report
            result = report_generator.generate_report(detection_id)
        elif report_format == 'html':
            # Generate HTML report
            result = report_generator.generate_html_report(detection_id)
        else:
            return jsonify({
                "success": False,
                "error": f"Unsupported report format: {report_format}"
            }), 400
        
        if not result["success"]:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 404
        
        # Return report
        if report_format == 'pdf':
            return send_file(result["report_path"], mimetype='application/pdf')
        else:
            return send_file(result["report_path"], mimetype='text/html')
    
    except Exception as e:
        logger.error(f"Error getting report: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """
    Get available models
    """
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

def create_visualization(result: Dict[str, Any]):
    """
    Create visualization for a detection result
    
    Args:
        result: Detection result
    """
    try:
        # Create visualization file path
        vis_path = os.path.join(VISUALIZATIONS_FOLDER, f"{result['id']}.jpg")
        
        # Check if original image exists
        upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
        image_path = os.path.join(upload_dir, result["filename"])
        
        if os.path.exists(image_path) and result["detectionType"] == "image":
            # Load image
            image = cv2.imread(image_path)
            
            # Draw faces
            for region in result["regions"]:
                # Get face box
                x, y, w, h = region["box"]
                
                # Determine color based on probability
                if region["probability"] > 0.5:
                    color = (0, 0, 255)  # Red for deepfake
                else:
                    color = (0, 255, 0)  # Green for real
                
                # Draw rectangle
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                
                # Add probability text
                prob_text = f"{region['probability']:.2f}"
                cv2.putText(image, prob_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Add skin tone text if available
                if "skin_tone" in region and region["skin_tone"]["success"]:
                    tone = region["skin_tone"]["indian_tone"]["name"]
                    cv2.putText(image, tone, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Add overall result
            if result["probability"] > 0.5:
                verdict = "DEEPFAKE"
                color = (0, 0, 255)  # Red
            else:
                verdict = "REAL"
                color = (0, 255, 0)  # Green
            
            cv2.putText(image, verdict, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(image, f"Prob: {result['probability']:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(image, f"Conf: {result['confidence']:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Save visualization
            cv2.imwrite(vis_path, image)
            logger.info(f"Created visualization at {vis_path}")
        else:
            # Create a blank image
            image = np.ones((400, 600, 3), dtype=np.uint8) * 255
            
            # Add detection information
            cv2.putText(image, f"Detection ID: {result['id']}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(image, f"Type: {result['detectionType']}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(image, f"Probability: {result['probability']:.4f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(image, f"Confidence: {result['confidence']:.4f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Add verdict
            if result["probability"] > 0.5:
                verdict = "DEEPFAKE"
                color = (0, 0, 255)  # Red
            else:
                verdict = "REAL"
                color = (0, 255, 0)  # Green
            
            cv2.putText(image, f"Verdict: {verdict}", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Add timestamp
            cv2.putText(image, f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Add regions information
            region_count = len(result.get("regions", []))
            cv2.putText(image, f"Regions: {region_count}", (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Add processing time
            cv2.putText(image, f"Processing Time: {result.get('processingTime', 0):.2f} seconds", (20, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Save visualization
            cv2.imwrite(vis_path, image)
            logger.info(f"Created placeholder visualization at {vis_path}")
    
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)