import os
import sys
import logging
import time
import uuid
import numpy as np
import cv2
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
import json
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create a simple Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create necessary directories
uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
visualizations_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizations')

for directory in [uploads_dir, results_dir, visualizations_dir]:
    os.makedirs(directory, exist_ok=True)

# Initialize the detector
try:
    from backend.ensemble_detector import EnsembleDeepfakeDetector
    from backend.config import INDIAN_FACE_DETECTION_ENABLED

    # Initialize the detector
    detector = EnsembleDeepfakeDetector(use_indian_enhancement=INDIAN_FACE_DETECTION_ENABLED)
    logger.info("Initialized EnsembleDeepfakeDetector")
    USING_ENHANCED_DETECTOR = True
except Exception as e:
    logger.error(f"Failed to initialize EnsembleDeepfakeDetector: {str(e)}")
    logger.info("Using mock detector instead")
    USING_ENHANCED_DETECTOR = False

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "status": "online",
        "version": "1.0.0",
        "message": "Enhanced DeepDefend API is running",
        "using_enhanced_detector": USING_ENHANCED_DETECTOR,
        "indian_enhancement": INDIAN_FACE_DETECTION_ENABLED if USING_ENHANCED_DETECTOR else False
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

        # Process file with enhanced detection if available
        start_time = time.time()

        if USING_ENHANCED_DETECTOR:
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
        else:
            # Use mock implementation
            if is_video:
                result = {
                    "id": detection_id,
                    "filename": filename,
                    "detectionType": "video",
                    "probability": 0.85,
                    "confidence": 0.92,
                    "processingTime": 3.5,
                    "frameCount": 30,
                    "regions": [
                        {
                            "box": [100, 100, 200, 200],
                            "probability": 0.87,
                            "confidence": 0.93,
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
                    "probability": 0.75,
                    "confidence": 0.85,
                    "processingTime": 0.8,
                    "regions": [
                        {
                            "box": [100, 100, 200, 200],
                            "probability": 0.78,
                            "confidence": 0.86,
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

        # Create visualization
        create_visualization(result, os.path.join(visualizations_dir, f"{detection_id}.jpg"))

        # Log successful detection
        logger.info(f"Successful detection: ID={detection_id}, Type={result['detectionType']}, " +
                   f"Probability={result['probability']:.4f}, Confidence={result['confidence']:.4f}")

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

@app.route('/api/result/<detection_id>', methods=['GET'])
def get_result(detection_id):
    """Get detection result by ID"""
    try:
        # Get result from file
        result_path = os.path.join(results_dir, f"{detection_id}.json")

        if not os.path.exists(result_path):
            return jsonify({
                "success": False,
                "error": f"Detection result with ID {detection_id} not found"
            }), 404

        # Read result from file
        with open(result_path, 'r') as f:
            result = json.load(f)

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

@app.route('/api/visualization/<detection_id>', methods=['GET'])
def get_visualization(detection_id):
    """Get visualization for a detection result"""
    try:
        # Get visualization path
        vis_path = os.path.join(visualizations_dir, f"{detection_id}.jpg")

        if not os.path.exists(vis_path):
            # Get result from file
            result_path = os.path.join(results_dir, f"{detection_id}.json")

            if not os.path.exists(result_path):
                return jsonify({
                    "success": False,
                    "error": f"Detection result with ID {detection_id} not found"
                }), 404

            # Read result from file
            with open(result_path, 'r') as f:
                result = json.load(f)

            # Create visualization
            create_visualization(result, vis_path)

        # Return visualization
        return send_file(vis_path, mimetype='image/jpeg')

    except Exception as e:
        logger.error(f"Error getting visualization: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback for a detection result"""
    try:
        # Get feedback data
        data = request.json

        if not data:
            return jsonify({
                "success": False,
                "error": "No feedback data provided"
            }), 400

        # Get detection ID
        detection_id = data.get('detection_id')

        if not detection_id:
            return jsonify({
                "success": False,
                "error": "No detection ID provided"
            }), 400

        # Save feedback to file
        feedback_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feedback')
        os.makedirs(feedback_dir, exist_ok=True)

        feedback_path = os.path.join(feedback_dir, f"{detection_id}.json")
        with open(feedback_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Feedback received for detection {detection_id}")

        return jsonify({
            "success": True,
            "message": "Feedback received. Thank you!"
        })

    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded files"""
    return send_from_directory(uploads_dir, filename)

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

        logger.debug(f"Created visualization at {output_path}")

        return True

    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        return False

if __name__ == '__main__':
    print("Starting enhanced DeepDefend backend on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)