"""
Simple API Endpoints for Deepfake Detector
"""

import os
import json
import logging
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

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
        
        # For testing, return a mock result
        import uuid
        detection_id = str(uuid.uuid4())
        
        # Create a mock result
        result = {
            "id": detection_id,
            "filename": filename,
            "detectionType": "image",
            "probability": 0.75,
            "confidence": 0.85,
            "processingTime": 1.23,
            "model": model_name,
            "ensemble": use_ensemble,
            "indianEnhancement": use_indian_enhancement,
            "regions": [
                {
                    "box": [100, 100, 200, 200],
                    "probability": 0.75,
                    "confidence": 0.85,
                    "skin_tone": {
                        "success": True,
                        "indian_tone": {
                            "name": "Medium",
                            "value": 0.5
                        }
                    }
                }
            ]
        }
        
        # Save result to file
        result_path = os.path.join(RESULTS_FOLDER, f"{detection_id}.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        return jsonify({
            "success": True,
            "detection_id": detection_id,
            "result": result
        })
    
    except Exception as e:
        logger.error(f"Error detecting deepfakes: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Detection failed: {str(e)}"
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

@app.route('/api/report/<detection_id>', methods=['GET'])
def get_report(detection_id):
    """
    Get report for a detection
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
        
        # Get report format
        format_param = request.args.get('format', 'json')
        
        # Handle different formats
        if format_param == 'pdf':
            # Import PDF report generator
            from pdf_report import PDFReportGenerator
            
            # Get the original image path
            image_filename = result.get('filename')
            image_path = os.path.join(UPLOAD_FOLDER, image_filename)
            
            if not os.path.exists(image_path):
                return jsonify({
                    "success": False,
                    "error": f"Original image not found: {image_filename}"
                }), 404
            
            # Generate PDF report
            pdf_generator = PDFReportGenerator()
            pdf_path = pdf_generator.generate_report(image_path, result, detection_id)
            
            if pdf_path and os.path.exists(pdf_path):
                return send_file(pdf_path, mimetype='application/pdf', as_attachment=True, 
                               download_name=f"deepfake_report_{detection_id}.pdf")
            else:
                return jsonify({
                    "success": False,
                    "error": "Failed to generate PDF report"
                }), 500
        
        elif format_param == 'html':
            # For HTML format, we'll return a simple HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Deepfake Detection Report - {detection_id}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    .result {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                    .high {{ background-color: #ffebee; }}
                    .medium {{ background-color: #fff8e1; }}
                    .low {{ background-color: #e8f5e9; }}
                    .label {{ font-weight: bold; }}
                    pre {{ background-color: #f5f5f5; padding: 10px; overflow: auto; }}
                </style>
            </head>
            <body>
                <h1>Deepfake Detection Report</h1>
                <div class="result {'high' if result.get('probability', 0) > 0.7 else 'medium' if result.get('probability', 0) > 0.4 else 'low'}">
                    <h2>Detection Result: {
                        'DEEPFAKE DETECTED' if result.get('probability', 0) > 0.7 
                        else 'SUSPICIOUS CONTENT' if result.get('probability', 0) > 0.4 
                        else 'LIKELY AUTHENTIC'
                    }</h2>
                    <p><span class="label">Probability:</span> {result.get('probability', 0):.2f}</p>
                    <p><span class="label">Confidence:</span> {result.get('confidence', 0):.2f}</p>
                    <p><span class="label">Model:</span> {result.get('model', 'Unknown')}</p>
                    <p><span class="label">Processing Time:</span> {result.get('processingTime', 0):.2f}ms</p>
                </div>
                
                <h2>Full Result Data</h2>
                <pre>{json.dumps(result, indent=2)}</pre>
            </body>
            </html>
            """
            
            return html_content, 200, {'Content-Type': 'text/html'}
        
        else:  # Default to JSON
            return jsonify({
                "success": True,
                "report": result
            })
    
    except Exception as e:
        logger.error(f"Error getting report: {str(e)}")
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
        
        # Get the original image path
        image_filename = result.get('filename')
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        
        if not os.path.exists(image_path):
            return jsonify({
                "success": False,
                "error": f"Original image not found: {image_filename}"
            }), 404
        
        # Get visualization type
        vis_type = request.args.get('type', 'standard')
        
        # Import visualization module
        from visualization import DetectionVisualizer
        
        # Create visualizer
        visualizer = DetectionVisualizer()
        
        # Generate visualization based on type
        if vis_type == 'heatmap':
            vis_path = os.path.join(visualizer.output_dir, f"{detection_id}_heatmap.jpg")
            if not os.path.exists(vis_path):
                visualizer.create_heatmap(image_path, result, detection_id)
        elif vis_type == 'comparison':
            vis_path = os.path.join(visualizer.output_dir, f"{detection_id}_comparison.jpg")
            if not os.path.exists(vis_path):
                visualizer.create_comparison_visualization(image_path, result, detection_id)
        else:  # standard visualization
            vis_path = os.path.join(visualizer.output_dir, f"{detection_id}_visualization.jpg")
            if not os.path.exists(vis_path):
                visualizer.visualize_detection(image_path, result, detection_id)
        
        # Check if visualization was created
        if not os.path.exists(vis_path):
            return jsonify({
                "success": False,
                "error": f"Failed to create visualization"
            }), 500
        
        # Return the visualization image
        return send_file(vis_path, mimetype='image/jpeg')
    
    except Exception as e:
        logger.error(f"Error getting visualization: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    """
    Get dashboard data for analytics
    """
    try:
        # Get all results
        results = []
        for filename in os.listdir(RESULTS_FOLDER):
            if filename.endswith('.json'):
                with open(os.path.join(RESULTS_FOLDER, filename), 'r') as f:
                    result = json.load(f)
                    results.append(result)
        
        # Calculate statistics
        total_detections = len(results)
        
        # Count by verdict
        deepfakes = sum(1 for r in results if r.get('probability', 0) > 0.7)
        suspicious = sum(1 for r in results if 0.4 < r.get('probability', 0) <= 0.7)
        authentic = sum(1 for r in results if r.get('probability', 0) <= 0.4)
        
        # Count by model
        models = {}
        for r in results:
            model = r.get('model', 'unknown')
            models[model] = models.get(model, 0) + 1
        
        # Average confidence
        avg_confidence = sum(r.get('confidence', 0) for r in results) / total_detections if total_detections > 0 else 0
        
        # Average processing time
        avg_processing_time = sum(r.get('processingTime', 0) for r in results) / total_detections if total_detections > 0 else 0
        
        # Return dashboard data
        return jsonify({
            "success": True,
            "dashboard": {
                "total_detections": total_detections,
                "verdicts": {
                    "deepfakes": deepfakes,
                    "suspicious": suspicious,
                    "authentic": authentic
                },
                "models": models,
                "avg_confidence": avg_confidence,
                "avg_processing_time": avg_processing_time,
                "recent_detections": results[-5:] if len(results) > 5 else results
            }
        })
    
    except Exception as e:
        logger.error(f"Error getting dashboard data: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)