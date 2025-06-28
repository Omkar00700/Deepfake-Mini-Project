"""
Simple Web Interface for Deepfake Detector
"""

import os
import logging
import json
import base64
from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from deepfake_detector import DeepfakeDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Create detector
detector = DeepfakeDetector(use_indian_enhancement=True)

# Create templates directory if it doesn't exist
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(templates_dir, exist_ok=True)

# Create static directory if it doesn't exist
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

# Create HTML template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Deepfake Detector</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #333;
        }
        .upload-form {
            margin: 20px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .result {
            margin: 20px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            display: none;
        }
        .result.show {
            display: block;
        }
        .result-image {
            max-width: 100%;
            margin: 10px 0;
        }
        .deepfake {
            color: red;
            font-weight: bold;
        }
        .real {
            color: green;
            font-weight: bold;
        }
        .loading {
            display: none;
            margin: 20px 0;
        }
        .loading.show {
            display: block;
        }
        .options {
            margin: 20px 0;
        }
        .options label {
            margin-right: 20px;
        }
    </style>
</head>
<body>
    <h1>Deepfake Detector</h1>
    <p>Upload an image to detect deepfakes.</p>
    
    <div class="upload-form">
        <form id="upload-form" enctype="multipart/form-data">
            <input type="file" name="file" id="file" accept="image/*">
            
            <div class="options">
                <label>
                    <input type="checkbox" name="use_ensemble" id="use_ensemble" checked>
                    Use Ensemble
                </label>
                <label>
                    <input type="checkbox" name="use_indian_enhancement" id="use_indian_enhancement" checked>
                    Use Indian Enhancement
                </label>
            </div>
            
            <button type="submit">Detect</button>
        </form>
    </div>
    
    <div class="loading" id="loading">
        <p>Processing... Please wait.</p>
    </div>
    
    <div class="result" id="result">
        <h2>Detection Result</h2>
        <p>Verdict: <span id="verdict"></span></p>
        <p>Probability: <span id="probability"></span></p>
        <p>Confidence: <span id="confidence"></span></p>
        <p>Processing Time: <span id="processing_time"></span> seconds</p>
        
        <h3>Detected Faces</h3>
        <div id="faces"></div>
        
        <img class="result-image" id="result_image">
    </div>
    
    <script>
        document.getElementById('upload-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show loading
            document.getElementById('loading').classList.add('show');
            document.getElementById('result').classList.remove('show');
            
            // Get form data
            var formData = new FormData();
            formData.append('file', document.getElementById('file').files[0]);
            formData.append('use_ensemble', document.getElementById('use_ensemble').checked);
            formData.append('use_indian_enhancement', document.getElementById('use_indian_enhancement').checked);
            
            // Send request
            fetch('/detect', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading
                document.getElementById('loading').classList.remove('show');
                
                if (data.success) {
                    // Show result
                    document.getElementById('result').classList.add('show');
                    
                    // Set result values
                    var verdict = data.result.probability > 0.5 ? 'DEEPFAKE' : 'REAL';
                    var verdictClass = data.result.probability > 0.5 ? 'deepfake' : 'real';
                    document.getElementById('verdict').textContent = verdict;
                    document.getElementById('verdict').className = verdictClass;
                    
                    document.getElementById('probability').textContent = (data.result.probability * 100).toFixed(2) + '%';
                    document.getElementById('confidence').textContent = (data.result.confidence * 100).toFixed(2) + '%';
                    document.getElementById('processing_time').textContent = data.result.processingTime.toFixed(2);
                    
                    // Set result image
                    document.getElementById('result_image').src = '/result_image/' + data.detection_id;
                    
                    // Show faces
                    var facesHtml = '';
                    for (var i = 0; i < data.result.regions.length; i++) {
                        var region = data.result.regions[i];
                        var faceVerdict = region.probability > 0.5 ? 'DEEPFAKE' : 'REAL';
                        var faceVerdictClass = region.probability > 0.5 ? 'deepfake' : 'real';
                        
                        facesHtml += '<div>';
                        facesHtml += '<p>Face ' + (i + 1) + ': <span class="' + faceVerdictClass + '">' + faceVerdict + '</span></p>';
                        facesHtml += '<p>Probability: ' + (region.probability * 100).toFixed(2) + '%</p>';
                        facesHtml += '<p>Confidence: ' + (region.confidence * 100).toFixed(2) + '%</p>';
                        
                        if (region.skin_tone && region.skin_tone.success) {
                            facesHtml += '<p>Skin Tone: ' + region.skin_tone.indian_tone.name + '</p>';
                        }
                        
                        facesHtml += '</div>';
                    }
                    
                    document.getElementById('faces').innerHTML = facesHtml;
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => {
                // Hide loading
                document.getElementById('loading').classList.remove('show');
                
                alert('Error: ' + error);
            });
        });
    </script>
</body>
</html>
"""

# Write HTML template to file
with open(os.path.join(templates_dir, "index.html"), 'w') as f:
    f.write(html_template)

@app.route('/')
def index():
    """Render index page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Detect deepfakes in uploaded image"""
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
        use_ensemble = request.form.get('use_ensemble', 'true').lower() == 'true'
        use_indian_enhancement = request.form.get('use_indian_enhancement', 'true').lower() == 'true'
        
        # Create upload directory if it doesn't exist
        upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)
        
        # Create detector with specified parameters
        detector = DeepfakeDetector(
            use_ensemble=use_ensemble,
            use_indian_enhancement=use_indian_enhancement
        )
        
        # Detect deepfakes
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

@app.route('/result_image/<detection_id>')
def result_image(detection_id):
    """Get result image for a detection"""
    try:
        # Create visualizations directory if it doesn't exist
        vis_dir = os.path.join(os.path.dirname(__file__), "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Check if visualization exists
        vis_path = os.path.join(vis_dir, f"{detection_id}.jpg")
        
        if not os.path.exists(vis_path):
            # Get result
            results_dir = os.path.join(os.path.dirname(__file__), "results")
            result_path = os.path.join(results_dir, f"{detection_id}.json")
            
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
        logger.error(f"Error getting result image: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def create_visualization(result):
    """
    Create visualization for a detection result
    
    Args:
        result: Detection result
    """
    try:
        # Create visualizations directory if it doesn't exist
        vis_dir = os.path.join(os.path.dirname(__file__), "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Create visualization file path
        vis_path = os.path.join(vis_dir, f"{result['id']}.jpg")
        
        # Check if original image exists
        upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
        image_path = os.path.join(upload_dir, result["filename"])
        
        if os.path.exists(image_path):
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
    app.run(host='0.0.0.0', port=5001, debug=True)
