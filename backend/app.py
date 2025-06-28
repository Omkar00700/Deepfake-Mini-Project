import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
DeepDefend API
Main application entry point
"""

from flask import Flask, jsonify, request, send_file, Response
from backend.api import api_bp
from backend.feedback_api import feedback_bp as feedback_api
from evaluation_api import evaluation_api
import logging
from backend.config import (
    DEBUG_MODE_ENABLED,
    LOG_INTERMEDIATE_RESULTS,
    VIDEO_ENSEMBLE_DEFAULT_WEIGHTS,
    RETRAINING_ENABLED,
    FEEDBACK_COLLECTION_ENABLED,
    RETRAINING_ENABLED,
    FEEDBACK_COLLECTION_ENABLED,
)

from metrics_logger import log_detection_metrics
from debug_utils import create_debug_report, generate_model_diagnostic_report
import os
import tempfile
import json
import time

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Configure CORS for development and testing
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# Register blueprints
app.register_blueprint(api_bp, url_prefix='/api')
app.register_blueprint(feedback_api, url_prefix='/api/feedback')
app.register_blueprint(evaluation_api, url_prefix='/api/evaluation')

@app.route('/status', methods=['GET'])
def status():
    """API status endpoint"""
    return jsonify({
        "status": "online",
        "version": "1.0.2",  # Updated version
        "retraining_enabled": RETRAINING_ENABLED,
        "feedback_enabled": FEEDBACK_COLLECTION_ENABLED,
        "debug_mode": DEBUG_MODE_ENABLED,
        "environment": os.environ.get("FLASK_ENV", "production")
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for load balancers and monitoring"""
    try:
        # Perform a simple check that critical components are available
        from inference import get_model_info
        model_info = get_model_info()
        
        return jsonify({
            "status": "healthy",
            "components": {
                "api": "available",
                "model": "available",
                "model_info": model_info
            }
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/api/report/<detection_id>', methods=['GET'])
def generate_report(detection_id):
    """Generate a detailed report for a detection result"""
    try:
        # Get the requested format
        report_format = request.args.get('format', 'json').lower()
        
        # Get detection result from database
        from database import get_detection_result
        detection_result = get_detection_result(detection_id)
        
        if not detection_result:
            return jsonify({
                "success": False,
                "error": f"Detection result with ID {detection_id} not found"
            }), 404
        
        # Generate the report
        report_data = {
            "report_id": f"report-{detection_id}-{int(time.time())}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "detection_id": detection_id,
            "detection_result": detection_result,
            "analysis": {
                "prediction": "deepfake" if detection_result.get("probability", 0) > 0.5 else "real",
                "confidence_level": get_confidence_level(detection_result.get("confidence", 0)),
                "detection_type": detection_result.get("detectionType", "unknown"),
                "processing_summary": {
                    "time_taken_ms": detection_result.get("processingTime", 0),
                    "frames_analyzed": detection_result.get("frameCount", 1),
                    "model_used": detection_result.get("model", "DeepDefend AI")
                }
            },
            "recommendations": generate_recommendations(detection_result),
            "technical_details": get_technical_details(detection_result)
        }
        
        # Return the report in the requested format
        if report_format == 'json':
            return jsonify({
                "success": True,
                "data": report_data
            })
        
        elif report_format == 'csv':
            # Create a simple CSV with key metrics
            csv_data = "Detection ID,Timestamp,Media Type,Prediction,Probability,Confidence,Frames Analyzed,Processing Time\n"
            csv_data += f"{detection_id},{report_data['timestamp']},{detection_result.get('detectionType', 'unknown')},"
            csv_data += f"{report_data['analysis']['prediction']},{detection_result.get('probability', 0):.4f},"
            csv_data += f"{detection_result.get('confidence', 0):.4f},{detection_result.get('frameCount', 1)},"
            csv_data += f"{detection_result.get('processingTime', 0)}\n"
            
            # Add face data if available
            if "regions" in detection_result and detection_result["regions"]:
                csv_data += "\nFace Analysis\n"
                csv_data += "Face,Probability,Confidence,Skin Tone\n"
                
                for i, region in enumerate(detection_result["regions"]):
                    # Get skin tone if available
                    skin_tone = "Unknown"
                    if "metadata" in region and "skin_tone" in region["metadata"] and region["metadata"]["skin_tone"].get("success", False):
                        if "indian_tone" in region["metadata"]["skin_tone"] and region["metadata"]["skin_tone"]["indian_tone"]:
                            skin_tone = region["metadata"]["skin_tone"]["indian_tone"].get("name", "Unknown")
                    elif "skin_tone" in region and region["skin_tone"].get("success", False):
                        if "indian_tone" in region["skin_tone"] and region["skin_tone"]["indian_tone"]:
                            skin_tone = region["skin_tone"]["indian_tone"].get("name", "Unknown")
                    
                    csv_data += f"Face {i+1},{region.get('probability', 0):.4f},{region.get('confidence', 0):.4f},{skin_tone}\n"
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
                temp.write(csv_data.encode('utf-8'))
                temp_path = temp.name
            
            # Return the CSV file
            return send_file(
                temp_path,
                mimetype='text/csv',
                as_attachment=True,
                download_name=f"deepdefend-report-{detection_id}.csv"
            )
        
        elif report_format == 'pdf':
            # Use our PDF report generator
            from pdf_report_generator import PDFReportGenerator
            
            # Create reports directory if it doesn't exist
            reports_dir = os.path.join(os.path.dirname(__file__), "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            # Generate PDF report
            pdf_generator = PDFReportGenerator(output_dir=reports_dir)
            pdf_path = pdf_generator.generate_report(detection_result, detection_id)
            
            # Return the PDF file
            return send_file(
                pdf_path,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f"deepdefend-report-{detection_id}.pdf"
            )
        
        else:
            return jsonify({
                "success": False,
                "error": f"Unsupported report format: {report_format}"
            }), 400
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Failed to generate report: {str(e)}"
        }), 500

@app.route('/api/report-status/<detection_id>', methods=['GET'])
def check_report_status(detection_id):
    """Check if a report is ready for the given detection"""
    try:
        # Get detection result from database
        from database import get_detection_result
        detection_result = get_detection_result(detection_id)
        
        if not detection_result:
            return jsonify({
                "ready": False,
                "error": f"Detection result with ID {detection_id} not found"
            }), 404
        
        # Check if we have enough data for a report
        is_ready = True
        missing_data = []
        
        # Basic checks for required fields
        if 'probability' not in detection_result:
            is_ready = False
            missing_data.append("probability")
        
        if 'confidence' not in detection_result:
            is_ready = False
            missing_data.append("confidence")
        
        # For videos, ensure we have enough processed frames
        if detection_result.get('detectionType') == 'video':
            if 'frameCount' not in detection_result or detection_result['frameCount'] < 3:
                is_ready = False
                missing_data.append(f"sufficient frame count (has {detection_result.get('frameCount', 0)})")
        
        # Check for processing metrics
        if 'regions' in detection_result and detection_result['regions']:
            first_region = detection_result['regions'][0]
            if 'metadata' not in first_region or 'processing_metrics' not in first_region.get('metadata', {}):
                is_ready = False
                missing_data.append("processing metrics")
        else:
            is_ready = False
            missing_data.append("region data")
        
        return jsonify({
            "ready": is_ready,
            "detection_id": detection_id,
            "missing_data": missing_data if not is_ready else []
        })
        
    except Exception as e:
        logger.error(f"Error checking report status: {str(e)}", exc_info=True)
        return jsonify({
            "ready": False,
            "error": f"Failed to check report status: {str(e)}"
        }), 500

@app.route('/api/debug/<detection_id>', methods=['GET'])
def debug_detection(detection_id):
    """Generate a debug report for a detection result"""
    if not DEBUG_MODE_ENABLED:
        return jsonify({
            "success": False,
            "error": "Debug mode is disabled"
        }), 403
    
    try:
        # Get detection result from database
        from database import get_detection_result
        detection_result = get_detection_result(detection_id)
        
        if not detection_result:
            return jsonify({
                "success": False,
                "error": f"Detection result with ID {detection_id} not found"
            }), 404
        
        # Generate debug report
        report_path = create_debug_report(detection_id, detection_result)
        
        return jsonify({
            "success": True,
            "message": f"Debug report generated: {report_path}",
            "detection_id": detection_id,
            "detection_data": detection_result
        })
        
    except Exception as e:
        logger.error(f"Error generating debug report: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Failed to generate debug report: {str(e)}"
        }), 500

@app.route('/api/diagnostics', methods=['GET'])
def get_system_diagnostics():
    """Get system diagnostic information"""
    if not DEBUG_MODE_ENABLED:
        return jsonify({
            "success": False,
            "error": "Debug mode is disabled"
        }), 403
    
    try:
        # Generate model diagnostic report
        report_path = generate_model_diagnostic_report()
        
        # Get system metrics
        from metrics import performance_metrics
        metrics = performance_metrics.get_all_metrics()
        
        # Get model information
        from inference import get_model_info
        model_info = get_model_info()
        
        # Get ensemble weights
        from inference import get_ensemble_weights
        ensemble_weights = get_ensemble_weights()
        
        return jsonify({
            "success": True,
            "diagnostics": {
                "report_path": report_path,
                "metrics": metrics,
                "model_info": model_info,
                "ensemble_weights": ensemble_weights,
                "system_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "debug_mode": DEBUG_MODE_ENABLED,
                "retraining_enabled": RETRAINING_ENABLED
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting system diagnostics: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Failed to get system diagnostics: {str(e)}"
        }), 500

def get_confidence_level(confidence_score):
    """Convert numerical confidence to descriptive level"""
    if confidence_score >= 0.9:
        return "Very High"
    elif confidence_score >= 0.75:
        return "High"
    elif confidence_score >= 0.6:
        return "Moderate"
    elif confidence_score >= 0.4:
        return "Low"
    else:
        return "Very Low"

def generate_recommendations(detection_result):
    """Generate recommendations based on detection result"""
    recommendations = []
    
    # General recommendations
    if detection_result.get('probability', 0) > 0.5:
        # Deepfake recommendations
        probability = detection_result.get('probability', 0)
        confidence = detection_result.get('confidence', 0)
        
        if probability > 0.9 and confidence > 0.8:
            recommendations.append("This media has very strong indicators of being a deepfake. We recommend treating it as synthetic content.")
        elif probability > 0.7:
            recommendations.append("This media shows significant signs of manipulation. Further verification is recommended before trusting this content.")
        else:
            recommendations.append("This media shows some signs of potential manipulation. Consider seeking additional verification.")
        
        # Add specific advice for high-confidence detections
        if confidence > 0.7:
            recommendations.append("Look for visual artifacts around faces, unnatural eye movements, or inconsistent lighting which are common in deepfakes.")
    else:
        # Real content recommendations
        confidence = detection_result.get('confidence', 0)
        
        if confidence > 0.8:
            recommendations.append("This media shows strong indicators of being authentic content.")
        elif confidence > 0.6:
            recommendations.append("This media appears to be authentic, but maintaining healthy skepticism is always recommended for online content.")
        else:
            recommendations.append("While this media appears authentic, the confidence is relatively low. Consider additional verification for critical decisions.")
    
    # Add media-specific recommendations
    if detection_result.get('detectionType') == 'video':
        recommendations.append("For videos, check for temporal consistency issues like sudden changes in facial expressions or unnatural movements.")
        
        # Check frame count for video-specific recommendations
        frame_count = detection_result.get('frameCount', 0)
        if frame_count < 10:
            recommendations.append("Note that this analysis was based on a limited number of video frames. Longer videos provide more reliable results.")
    
    # Add general advice
    recommendations.append("Remember that deepfake detection technology is constantly evolving, and no detection system is 100% accurate.")
    
    return recommendations

def get_technical_details(detection_result):
    """Extract technical details from detection result"""
    details = {
        "detection_time": detection_result.get('timestamp', 'unknown'),
        "processing_time_ms": detection_result.get('processingTime', 0),
        "model_used": detection_result.get('model', 'unknown')
    }
    
    # Add media-specific details
    if detection_result.get('detectionType') == 'video':
        details["frames_analyzed"] = detection_result.get('frameCount', 0)
    
    # Add regions information
    if 'regions' in detection_result and detection_result['regions']:
        details["regions_analyzed"] = len(detection_result['regions'])
        
        # Extract processing metrics if available
        first_region = detection_result['regions'][0]
        if 'metadata' in first_region and 'processing_metrics' in first_region['metadata']:
            metrics = first_region['metadata']['processing_metrics']
            details["processing_metrics"] = metrics
    
    return details

if __name__ == '__main__':
    logger.info("Starting DeepDefend API")
    
    # Log application startup
    log_detection_metrics({
        "type": "system",
        "event": "startup",
        "metrics": {
            "retraining_enabled": RETRAINING_ENABLED,
            "feedback_enabled": FEEDBACK_COLLECTION_ENABLED,
            "debug_mode": DEBUG_MODE_ENABLED
        }
    })
    
    # Start the application
    app.run(host='0.0.0.0', port=5000, debug=True)
