"""
Enhanced API Endpoints for Deepfake Detector

This module implements enhanced API endpoints including:
- Advanced detection with multiple models
- Multi-modal analysis of media
- Batch processing of multiple files
- Detection settings management
- Model performance metrics
"""

import os
import json
import logging
import uuid
import time
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from advanced_detection import AdvancedDetector
from multimodal_analyzer import MultiModalAnalyzer
from pdf_report import PDFReportGenerator
from visualization import DetectionVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "results")
REPORTS_FOLDER = os.path.join(os.path.dirname(__file__), "reports")
VISUALIZATIONS_FOLDER = os.path.join(os.path.dirname(__file__), "visualizations")
BATCH_FOLDER = os.path.join(os.path.dirname(__file__), "batch_results")

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATIONS_FOLDER, exist_ok=True)
os.makedirs(BATCH_FOLDER, exist_ok=True)

# Configure allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'mp3', 'wav'}

# Initialize detector and analyzer
detector = AdvancedDetector()
analyzer = MultiModalAnalyzer()

# Add error handler
@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all exceptions"""
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({
        "success": False,
        "error": str(e)
    }), 500

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    try:
        # Get model performance metrics
        model_performance = detector.get_model_performance()
        
        # Format models for API response
        models = []
        for model_id, metrics in model_performance.items():
            models.append({
                "id": model_id,
                "name": metrics["name"],
                "type": metrics["type"],
                "accuracy": metrics["accuracy"],
                "avg_inference_time": metrics["avg_inference_time"],
                "description": f"{metrics['name']} model for {metrics['type']} analysis"
            })
        
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

@app.route('/api/advanced-detect', methods=['POST'])
def advanced_detect():
    """Perform advanced detection with multiple models"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file provided"
            }), 400
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Generate unique ID for this detection
        detection_id = str(uuid.uuid4())
        
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, f"{detection_id}_{filename}")
        file.save(file_path)
        
        logger.info(f"File saved to {file_path}")
        
        # Get selected model from request
        model_id = request.form.get('model', 'ensemble')
        
        # Update detector settings if provided
        if 'settings' in request.form:
            try:
                settings = json.loads(request.form['settings'])
                detector.update_settings(settings)
            except json.JSONDecodeError:
                logger.warning("Invalid settings JSON, using default settings")
        
        # Perform detection
        result = detector.detect(file_path, detection_id)
        
        # Add file info to result
        result["filename"] = filename
        result["detection_id"] = detection_id
        
        return jsonify({
            "success": True,
            "result": result
        })
    
    except Exception as e:
        logger.error(f"Error in advanced detection: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/multimodal-analyze', methods=['POST'])
def multimodal_analyze():
    """Perform multi-modal analysis of media"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file provided"
            }), 400
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Generate unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, f"{analysis_id}_{filename}")
        file.save(file_path)
        
        logger.info(f"File saved to {file_path}")
        
        # Update analyzer settings if provided
        if 'settings' in request.form:
            try:
                settings = json.loads(request.form['settings'])
                analyzer.update_settings(settings)
            except json.JSONDecodeError:
                logger.warning("Invalid settings JSON, using default settings")
        
        # Perform analysis
        result = analyzer.analyze(file_path, analysis_id)
        
        # Add file info to result
        result["filename"] = filename
        result["analysis_id"] = analysis_id
        
        return jsonify({
            "success": True,
            "result": result
        })
    
    except Exception as e:
        logger.error(f"Error in multi-modal analysis: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/batch-process', methods=['POST'])
def batch_process():
    """Batch process multiple files"""
    try:
        # Check if files were uploaded
        if 'files[]' not in request.files:
            return jsonify({
                "success": False,
                "error": "No files provided"
            }), 400
        
        files = request.files.getlist('files[]')
        
        # Check if files are empty
        if len(files) == 0:
            return jsonify({
                "success": False,
                "error": "No files selected"
            }), 400
        
        # Generate unique batch ID
        batch_id = str(uuid.uuid4())
        
        # Process each file
        results = []
        for file in files:
            # Check if file is allowed
            if not allowed_file(file.filename):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
                })
                continue
            
            # Generate unique ID for this detection
            detection_id = str(uuid.uuid4())
            
            # Save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, f"{detection_id}_{filename}")
            file.save(file_path)
            
            logger.info(f"File saved to {file_path}")
            
            # Perform detection
            try:
                result = detector.detect(file_path, detection_id)
                
                # Add file info to result
                result["filename"] = filename
                result["detection_id"] = detection_id
                
                results.append({
                    "filename": filename,
                    "success": True,
                    "result": result
                })
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")
                results.append({
                    "filename": filename,
                    "success": False,
                    "error": str(e)
                })
        
        # Save batch results
        batch_result = {
            "batch_id": batch_id,
            "timestamp": time.time(),
            "total_files": len(files),
            "successful": sum(1 for r in results if r.get("success", False)),
            "failed": sum(1 for r in results if not r.get("success", False)),
            "results": results
        }
        
        batch_result_path = os.path.join(BATCH_FOLDER, f"{batch_id}.json")
        with open(batch_result_path, 'w') as f:
            json.dump(batch_result, f, indent=2)
        
        return jsonify({
            "success": True,
            "batch_id": batch_id,
            "results": results
        })
    
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/detection-settings', methods=['GET', 'PUT'])
def detection_settings():
    """Get or update detection settings"""
    try:
        if request.method == 'GET':
            # Get current settings
            settings = detector.get_settings()
            return jsonify({
                "success": True,
                "settings": settings
            })
        else:  # PUT
            # Update settings
            data = request.json
            if not data:
                return jsonify({
                    "success": False,
                    "error": "No settings provided"
                }), 400
            
            updated_settings = detector.update_settings(data)
            return jsonify({
                "success": True,
                "settings": updated_settings
            })
    
    except Exception as e:
        logger.error(f"Error handling detection settings: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/model-performance', methods=['GET'])
def model_performance():
    """Get model performance metrics"""
    try:
        # Get model performance metrics
        performance = detector.get_model_performance()
        
        return jsonify({
            "success": True,
            "performance": performance
        })
    
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/results/<detection_id>', methods=['GET'])
def get_results(detection_id):
    """Get detection results by ID"""
    try:
        # Check if results exist
        result_path = os.path.join(RESULTS_FOLDER, f"{detection_id}.json")
        if not os.path.exists(result_path):
            return jsonify({
                "success": False,
                "error": f"No results found for detection ID: {detection_id}"
            }), 404
        
        # Load results
        with open(result_path, 'r') as f:
            results = json.load(f)
        
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

@app.route('/api/report/<detection_id>', methods=['GET'])
def get_report(detection_id):
    """Generate and return a report for the detection"""
    try:
        # Check format parameter
        format_type = request.args.get('format', 'pdf').lower()
        if format_type not in ['pdf', 'html', 'json']:
            return jsonify({
                "success": False,
                "error": f"Unsupported format: {format_type}. Supported formats: pdf, html, json"
            }), 400
        
        # Check if results exist
        result_path = os.path.join(RESULTS_FOLDER, f"{detection_id}.json")
        if not os.path.exists(result_path):
            return jsonify({
                "success": False,
                "error": f"No results found for detection ID: {detection_id}"
            }), 404
        
        # Load results
        with open(result_path, 'r') as f:
            results = json.load(f)
        
        # Generate report
        report_generator = PDFReportGenerator()
        
        if format_type == 'pdf':
            # Generate PDF report
            report_path = os.path.join(REPORTS_FOLDER, f"{detection_id}.pdf")
            report_generator.generate_pdf_report(results, report_path)
            
            # Return the PDF file
            return send_file(report_path, as_attachment=True, download_name=f"deepfake_report_{detection_id}.pdf")
            
        elif format_type == 'html':
            # Generate HTML report
            html_content = report_generator.generate_html_report(results)
            
            # Save HTML report
            html_path = os.path.join(REPORTS_FOLDER, f"{detection_id}.html")
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            # Return the HTML content
            return html_content, 200, {'Content-Type': 'text/html'}
            
        else:  # JSON
            # Generate JSON report
            json_report = report_generator.generate_json_report(results)
            
            # Return the JSON report
            return jsonify({
                "success": True,
                "report": json_report
            })
    
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/visualization/<detection_id>', methods=['GET'])
def get_visualization(detection_id):
    """Generate and return a visualization for the detection"""
    try:
        # Check visualization type parameter
        vis_type = request.args.get('type', 'standard').lower()
        if vis_type not in ['standard', 'heatmap', 'comparison']:
            return jsonify({
                "success": False,
                "error": f"Unsupported visualization type: {vis_type}. Supported types: standard, heatmap, comparison"
            }), 400
        
        # Check if results exist
        result_path = os.path.join(RESULTS_FOLDER, f"{detection_id}.json")
        if not os.path.exists(result_path):
            return jsonify({
                "success": False,
                "error": f"No results found for detection ID: {detection_id}"
            }), 404
        
        # Load results
        with open(result_path, 'r') as f:
            results = json.load(f)
        
        # Get the original file path
        file_path = results.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return jsonify({
                "success": False,
                "error": f"Original file not found: {file_path}"
            }), 404
        
        # Generate visualization
        visualizer = DetectionVisualizer()
        
        if vis_type == 'standard':
            # Generate standard visualization
            vis_path = os.path.join(VISUALIZATIONS_FOLDER, f"{detection_id}_standard.jpg")
            visualizer.create_standard_visualization(file_path, results, vis_path)
            
        elif vis_type == 'heatmap':
            # Generate heatmap visualization
            vis_path = os.path.join(VISUALIZATIONS_FOLDER, f"{detection_id}_heatmap.jpg")
            visualizer.create_heatmap_visualization(file_path, results, vis_path)
            
        else:  # comparison
            # Generate comparison visualization
            vis_path = os.path.join(VISUALIZATIONS_FOLDER, f"{detection_id}_comparison.jpg")
            visualizer.create_comparison_visualization(file_path, results, vis_path)
        
        # Return the visualization image
        return send_file(vis_path, mimetype='image/jpeg')
    
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    """Get dashboard data for analytics"""
    try:
        # Get all result files
        result_files = [f for f in os.listdir(RESULTS_FOLDER) if f.endswith('.json')]
        
        # Load results
        results = []
        for file_name in result_files:
            try:
                with open(os.path.join(RESULTS_FOLDER, file_name), 'r') as f:
                    result = json.load(f)
                    results.append(result)
            except Exception as e:
                logger.error(f"Error loading result file {file_name}: {str(e)}")
        
        # Calculate dashboard metrics
        total_detections = len(results)
        
        # Count by verdict
        deepfakes = 0
        suspicious = 0
        authentic = 0
        
        for r in results:
            verdict = r.get('verdict', '')
            if verdict == 'deepfake':
                deepfakes += 1
            elif verdict == 'suspicious':
                suspicious += 1
            elif verdict == 'authentic':
                authentic += 1
        
        # Count by model
        models = {}
        for r in results:
            model_results = r.get('model_results', {})
            for model_name in model_results:
                models[model_name] = models.get(model_name, 0) + 1
        
        # Average confidence
        avg_confidence = sum(r.get('confidence', 0) for r in results) / total_detections if total_detections > 0 else 0
        
        # Average processing time
        avg_processing_time = sum(r.get('processing_time', 0) for r in results) / total_detections if total_detections > 0 else 0
        
        # Get recent detections (last 10)
        recent_detections = sorted(results, key=lambda x: x.get('timestamp', 0), reverse=True)[:10]
        
        # Format recent detections for the dashboard
        formatted_recent = []
        for r in recent_detections:
            formatted_recent.append({
                "id": r.get('detection_id', ''),
                "filename": r.get('filename', ''),
                "probability": r.get('probability', 0),
                "confidence": r.get('confidence', 0),
                "verdict": r.get('verdict', ''),
                "model": r.get('model_results', {}).keys(),
                "timestamp": r.get('timestamp', 0)
            })
        
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
                "recent_detections": formatted_recent
            }
        })
    
    except Exception as e:
        logger.error(f"Error getting dashboard data: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)"""
Enhanced API Endpoints for Deepfake Detector

This module implements enhanced API endpoints including:
- Advanced detection with multiple models
- Multi-modal analysis of media
- Batch processing of multiple files
- Detection settings management
- Model performance metrics
"""

import os
import json
import logging
import uuid
import time
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from advanced_detection import AdvancedDetector
from multimodal_analyzer import MultiModalAnalyzer
from pdf_report import PDFReportGenerator
from visualization import DetectionVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "results")
REPORTS_FOLDER = os.path.join(os.path.dirname(__file__), "reports")
VISUALIZATIONS_FOLDER = os.path.join(os.path.dirname(__file__), "visualizations")
BATCH_FOLDER = os.path.join(os.path.dirname(__file__), "batch_results")

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATIONS_FOLDER, exist_ok=True)
os.makedirs(BATCH_FOLDER, exist_ok=True)

# Configure allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'mp3', 'wav'}

# Initialize detector and analyzer
detector = AdvancedDetector()
analyzer = MultiModalAnalyzer()

# Add error handler
@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all exceptions"""
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({
        "success": False,
        "error": str(e)
    }), 500

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    try:
        # Get model performance metrics
        model_performance = detector.get_model_performance()
        
        # Format models for API response
        models = []
        for model_id, metrics in model_performance.items():
            models.append({
                "id": model_id,
                "name": metrics["name"],
                "type": metrics["type"],
                "accuracy": metrics["accuracy"],
                "avg_inference_time": metrics["avg_inference_time"],
                "description": f"{metrics['name']} model for {metrics['type']} analysis"
            })
        
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

@app.route('/api/advanced-detect', methods=['POST'])
def advanced_detect():
    """Perform advanced detection with multiple models"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file provided"
            }), 400
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Generate unique ID for this detection
        detection_id = str(uuid.uuid4())
        
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, f"{detection_id}_{filename}")
        file.save(file_path)
        
        logger.info(f"File saved to {file_path}")
        
        # Get selected model from request
        model_id = request.form.get('model', 'ensemble')
        
        # Update detector settings if provided
        if 'settings' in request.form:
            try:
                settings = json.loads(request.form['settings'])
                detector.update_settings(settings)
            except json.JSONDecodeError:
                logger.warning("Invalid settings JSON, using default settings")
        
        # Perform detection
        result = detector.detect(file_path, detection_id)
        
        # Add file info to result
        result["filename"] = filename
        result["detection_id"] = detection_id
        
        return jsonify({
            "success": True,
            "result": result
        })
    
    except Exception as e:
        logger.error(f"Error in advanced detection: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/multimodal-analyze', methods=['POST'])
def multimodal_analyze():
    """Perform multi-modal analysis of media"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file provided"
            }), 400
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Generate unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, f"{analysis_id}_{filename}")
        file.save(file_path)
        
        logger.info(f"File saved to {file_path}")
        
        # Update analyzer settings if provided
        if 'settings' in request.form:
            try:
                settings = json.loads(request.form['settings'])
                analyzer.update_settings(settings)
            except json.JSONDecodeError:
                logger.warning("Invalid settings JSON, using default settings")
        
        # Perform analysis
        result = analyzer.analyze(file_path, analysis_id)
        
        # Add file info to result
        result["filename"] = filename
        result["analysis_id"] = analysis_id
        
        return jsonify({
            "success": True,
            "result": result
        })
    
    except Exception as e:
        logger.error(f"Error in multi-modal analysis: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/batch-process', methods=['POST'])
def batch_process():
    """Batch process multiple files"""
    try:
        # Check if files were uploaded
        if 'files[]' not in request.files:
            return jsonify({
                "success": False,
                "error": "No files provided"
            }), 400
        
        files = request.files.getlist('files[]')
        
        # Check if files are empty
        if len(files) == 0:
            return jsonify({
                "success": False,
                "error": "No files selected"
            }), 400
        
        # Generate unique batch ID
        batch_id = str(uuid.uuid4())
        
        # Process each file
        results = []
        for file in files:
            # Check if file is allowed
            if not allowed_file(file.filename):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
                })
                continue
            
            # Generate unique ID for this detection
            detection_id = str(uuid.uuid4())
            
            # Save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, f"{detection_id}_{filename}")
            file.save(file_path)
            
            logger.info(f"File saved to {file_path}")
            
            # Perform detection
            try:
                result = detector.detect(file_path, detection_id)
                
                # Add file info to result
                result["filename"] = filename
                result["detection_id"] = detection_id
                
                results.append({
                    "filename": filename,
                    "success": True,
                    "result": result
                })
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")
                results.append({
                    "filename": filename,
                    "success": False,
                    "error": str(e)
                })
        
        # Save batch results
        batch_result = {
            "batch_id": batch_id,
            "timestamp": time.time(),
            "total_files": len(files),
            "successful": sum(1 for r in results if r.get("success", False)),
            "failed": sum(1 for r in results if not r.get("success", False)),
            "results": results
        }
        
        batch_result_path = os.path.join(BATCH_FOLDER, f"{batch_id}.json")
        with open(batch_result_path, 'w') as f:
            json.dump(batch_result, f, indent=2)
        
        return jsonify({
            "success": True,
            "batch_id": batch_id,
            "results": results
        })
    
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/detection-settings', methods=['GET', 'PUT'])
def detection_settings():
    """Get or update detection settings"""
    try:
        if request.method == 'GET':
            # Get current settings
            settings = detector.get_settings()
            return jsonify({
                "success": True,
                "settings": settings
            })
        else:  # PUT
            # Update settings
            data = request.json
            if not data:
                return jsonify({
                    "success": False,
                    "error": "No settings provided"
                }), 400
            
            updated_settings = detector.update_settings(data)
            return jsonify({
                "success": True,
                "settings": updated_settings
            })
    
    except Exception as e:
        logger.error(f"Error handling detection settings: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/model-performance', methods=['GET'])
def model_performance():
    """Get model performance metrics"""
    try:
        # Get model performance metrics
        performance = detector.get_model_performance()
        
        return jsonify({
            "success": True,
            "performance": performance
        })
    
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/results/<detection_id>', methods=['GET'])
def get_results(detection_id):
    """Get detection results by ID"""
    try:
        # Check if results exist
        result_path = os.path.join(RESULTS_FOLDER, f"{detection_id}.json")
        if not os.path.exists(result_path):
            return jsonify({
                "success": False,
                "error": f"No results found for detection ID: {detection_id}"
            }), 404
        
        # Load results
        with open(result_path, 'r') as f:
            results = json.load(f)
        
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

@app.route('/api/report/<detection_id>', methods=['GET'])
def get_report(detection_id):
    """Generate and return a report for the detection"""
    try:
        # Check format parameter
        format_type = request.args.get('format', 'pdf').lower()
        if format_type not in ['pdf', 'html', 'json']:
            return jsonify({
                "success": False,
                "error": f"Unsupported format: {format_type}. Supported formats: pdf, html, json"
            }), 400
        
        # Check if results exist
        result_path = os.path.join(RESULTS_FOLDER, f"{detection_id}.json")
        if not os.path.exists(result_path):
            return jsonify({
                "success": False,
                "error": f"No results found for detection ID: {detection_id}"
            }), 404
        
        # Load results
        with open(result_path, 'r') as f:
            results = json.load(f)
        
        # Generate report
        report_generator = PDFReportGenerator()
        
        if format_type == 'pdf':
            # Generate PDF report
            report_path = os.path.join(REPORTS_FOLDER, f"{detection_id}.pdf")
            report_generator.generate_pdf_report(results, report_path)
            
            # Return the PDF file
            return send_file(report_path, as_attachment=True, download_name=f"deepfake_report_{detection_id}.pdf")
            
        elif format_type == 'html':
            # Generate HTML report
            html_content = report_generator.generate_html_report(results)
            
            # Save HTML report
            html_path = os.path.join(REPORTS_FOLDER, f"{detection_id}.html")
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            # Return the HTML content
            return html_content, 200, {'Content-Type': 'text/html'}
            
        else:  # JSON
            # Generate JSON report
            json_report = report_generator.generate_json_report(results)
            
            # Return the JSON report
            return jsonify({
                "success": True,
                "report": json_report
            })
    
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/visualization/<detection_id>', methods=['GET'])
def get_visualization(detection_id):
    """Generate and return a visualization for the detection"""
    try:
        # Check visualization type parameter
        vis_type = request.args.get('type', 'standard').lower()
        if vis_type not in ['standard', 'heatmap', 'comparison']:
            return jsonify({
                "success": False,
                "error": f"Unsupported visualization type: {vis_type}. Supported types: standard, heatmap, comparison"
            }), 400
        
        # Check if results exist
        result_path = os.path.join(RESULTS_FOLDER, f"{detection_id}.json")
        if not os.path.exists(result_path):
            return jsonify({
                "success": False,
                "error": f"No results found for detection ID: {detection_id}"
            }), 404
        
        # Load results
        with open(result_path, 'r') as f:
            results = json.load(f)
        
        # Get the original file path
        file_path = results.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return jsonify({
                "success": False,
                "error": f"Original file not found: {file_path}"
            }), 404
        
        # Generate visualization
        visualizer = DetectionVisualizer()
        
        if vis_type == 'standard':
            # Generate standard visualization
            vis_path = os.path.join(VISUALIZATIONS_FOLDER, f"{detection_id}_standard.jpg")
            visualizer.create_standard_visualization(file_path, results, vis_path)
            
        elif vis_type == 'heatmap':
            # Generate heatmap visualization
            vis_path = os.path.join(VISUALIZATIONS_FOLDER, f"{detection_id}_heatmap.jpg")
            visualizer.create_heatmap_visualization(file_path, results, vis_path)
            
        else:  # comparison
            # Generate comparison visualization
            vis_path = os.path.join(VISUALIZATIONS_FOLDER, f"{detection_id}_comparison.jpg")
            visualizer.create_comparison_visualization(file_path, results, vis_path)
        
        # Return the visualization image
        return send_file(vis_path, mimetype='image/jpeg')
    
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    """Get dashboard data for analytics"""
    try:
        # Get all result files
        result_files = [f for f in os.listdir(RESULTS_FOLDER) if f.endswith('.json')]
        
        # Load results
        results = []
        for file_name in result_files:
            try:
                with open(os.path.join(RESULTS_FOLDER, file_name), 'r') as f:
                    result = json.load(f)
                    results.append(result)
            except Exception as e:
                logger.error(f"Error loading result file {file_name}: {str(e)}")
        
        # Calculate dashboard metrics
        total_detections = len(results)
        
        # Count by verdict
        deepfakes = 0
        suspicious = 0
        authentic = 0
        
        for r in results:
            verdict = r.get('verdict', '')
            if verdict == 'deepfake':
                deepfakes += 1
            elif verdict == 'suspicious':
                suspicious += 1
            elif verdict == 'authentic':
                authentic += 1
        
        # Count by model
        models = {}
        for r in results:
            model_results = r.get('model_results', {})
            for model_name in model_results:
                models[model_name] = models.get(model_name, 0) + 1
        
        # Average confidence
        avg_confidence = sum(r.get('confidence', 0) for r in results) / total_detections if total_detections > 0 else 0
        
        # Average processing time
        avg_processing_time = sum(r.get('processing_time', 0) for r in results) / total_detections if total_detections > 0 else 0
        
        # Get recent detections (last 10)
        recent_detections = sorted(results, key=lambda x: x.get('timestamp', 0), reverse=True)[:10]
        
        # Format recent detections for the dashboard
        formatted_recent = []
        for r in recent_detections:
            formatted_recent.append({
                "id": r.get('detection_id', ''),
                "filename": r.get('filename', ''),
                "probability": r.get('probability', 0),
                "confidence": r.get('confidence', 0),
                "verdict": r.get('verdict', ''),
                "model": r.get('model_results', {}).keys(),
                "timestamp": r.get('timestamp', 0)
            })
        
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
                "recent_detections": formatted_recent
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