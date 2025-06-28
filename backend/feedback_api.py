"""
Feedback API for DeepDefend
Provides endpoints for submitting and managing user feedback on detection results
"""

import logging
import time
import json
import os
from typing import Dict, Any, Optional
from flask import Blueprint, request, jsonify, current_app
from backend.auth import auth_required

# Import retraining manager (will be available after integration)
try:
    from backend.model_retraining import add_detection_feedback, get_retraining_status
    RETRAINING_AVAILABLE = True
except ImportError:
    RETRAINING_AVAILABLE = False
    # Create placeholder functions if retraining is not available
    def add_detection_feedback(detection_id, correct, actual_label, confidence=None, metadata=None):
        return False
    
    def get_retraining_status():
        return {"enabled": False, "feedback_collection_enabled": False}

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint for the feedback API
feedback_bp = Blueprint('feedback', __name__)

@feedback_bp.route('/feedback', methods=['POST'])
@auth_required
def submit_feedback():
    """
    Submit feedback for a detection result
    
    Request body:
    {
        "detection_id": "string",
        "correct": boolean,
        "actual_label": "string" (real/deepfake),
        "confidence": float (optional),
        "metadata": object (optional)
    }
    
    Returns:
        Success message or error
    """
    # Check if retraining is available
    if not RETRAINING_AVAILABLE:
        return jsonify({
            "success": False,
            "message": "Feedback collection is not available"
        }), 501
    
    # Check if feedback collection is enabled
    retraining_status = get_retraining_status()
    if not retraining_status.get("feedback_collection_enabled", False):
        return jsonify({
            "success": False,
            "message": "Feedback collection is disabled"
        }), 403
    
    # Get request data
    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "Missing request body"
        }), 400
    
    # Validate required fields
    required_fields = ["detection_id", "correct", "actual_label"]
    if not all(field in data for field in required_fields):
        return jsonify({
            "success": False,
            "message": f"Missing required fields: {', '.join(required_fields)}"
        }), 400
    
    # Validate field types
    if not isinstance(data["correct"], bool):
        return jsonify({
            "success": False,
            "message": "Field 'correct' must be a boolean"
        }), 400
    
    if data["actual_label"] not in ["real", "deepfake"]:
        return jsonify({
            "success": False,
            "message": "Field 'actual_label' must be 'real' or 'deepfake'"
        }), 400
    
    # Get user information if authenticated
    user_id = None
    try:
        user_id = request.user.get('sub', 'anonymous') if hasattr(request, 'user') else 'anonymous'
    except:
        pass
    
    # Add metadata
    metadata = data.get("metadata", {})
    if user_id:
        metadata["user_id"] = user_id
    
    metadata["timestamp"] = time.time()
    metadata["source"] = "api"
    
    # Check for region-specific data
    if "region" in data:
        metadata["region"] = data["region"]
    
    # Add feedback
    success = add_detection_feedback(
        detection_id=data["detection_id"],
        correct=data["correct"],
        actual_label=data["actual_label"],
        confidence=data.get("confidence"),
        metadata=metadata
    )
    
    if success:
        logger.info(f"Feedback submitted for detection {data['detection_id']}")
        return jsonify({
            "success": True,
            "message": "Feedback submitted successfully"
        })
    else:
        logger.error(f"Failed to submit feedback for detection {data['detection_id']}")
        return jsonify({
            "success": False,
            "message": "Failed to submit feedback"
        }), 500

@feedback_bp.route('/feedback/status', methods=['GET'])
@auth_required
def get_feedback_status():
    """
    Get feedback collection status
    
    Returns:
        Feedback collection status information
    """
    if not RETRAINING_AVAILABLE:
        return jsonify({
            "enabled": False,
            "message": "Feedback collection is not available"
        })
    
    retraining_status = get_retraining_status()
    
    return jsonify({
        "enabled": retraining_status.get("feedback_collection_enabled", False),
        "retraining_enabled": retraining_status.get("enabled", False),
        "samples_collected": retraining_status.get("feedback_samples_collected", 0),
        "last_evaluation_time": retraining_status.get("last_evaluation_time"),
        "last_retraining_time": retraining_status.get("last_retraining_time"),
        "accuracy_metrics": retraining_status.get("current_evaluation", {})
    })

@feedback_bp.route('/feedback/bulk-analysis', methods=['POST'])
@auth_required
def bulk_analysis():
    """
    Bulk test a set of images/videos and get accuracy metrics
    ```json
    {
        "items": [ ... ],
        "options": { "model": "string", "ensemble": boolean }
    }
    ```
    Returns:
        Results of bulk analysis with accuracy metrics
    """
    from backend.detection_handler import process_image, process_video
    
    # Get request data
    data = request.get_json()
    if not data or "items" not in data:
        return jsonify({
            "success": False,
            "message": "Missing required field: items"
        }), 400
    
    # Get options
    options = data.get("options", {})
    
    # Switch model if specified
    if "model" in options:
        from backend.inference import switch_model
        switch_model(options["model"])
    
    # Process items
    results = []
    accuracy_metrics = {
        "total": 0,
        "correct": 0,
        "accuracy": 0,
        "true_positives": 0,
        "true_negatives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "real_accuracy": 0,
        "fake_accuracy": 0
    }
    
    for item in data["items"]:
        try:
            # Validate item
            if "path" not in item or "type" not in item or "actual_label" not in item:
                logger.warning(f"Skipping invalid item: {item}")
                continue
            
            path = item["path"]
            item_type = item["type"]
            actual_label = item["actual_label"]
            
            # Ensure path exists
            if not os.path.exists(path):
                logger.warning(f"Skipping non-existent path: {path}")
                continue
            
            # Process item
            if item_type == "image":
                probability, confidence, regions = process_image(path)
            elif item_type == "video":
                probability, confidence, frame_count, regions = process_video(path)
            else:
                logger.warning(f"Skipping unknown item type: {item_type}")
                continue
            
            # Determine predicted label
            predicted_label = "deepfake" if probability > 0.5 else "real"
            
            # Update accuracy metrics
            accuracy_metrics["total"] += 1
            
            if predicted_label == actual_label:
                accuracy_metrics["correct"] += 1
                if actual_label == "deepfake":
                    accuracy_metrics["true_positives"] += 1
                else:
                    accuracy_metrics["true_negatives"] += 1
            else:
                if actual_label == "real":
                    accuracy_metrics["false_positives"] += 1
                else:
                    accuracy_metrics["false_negatives"] += 1
            
            # Add to results
            results.append({
                "path": path,
                "type": item_type,
                "actual_label": actual_label,
                "predicted_label": predicted_label,
                "probability": float(probability),
                "confidence": float(confidence),
                "correct": predicted_label == actual_label,
                "metadata": item.get("metadata", {})
            })
        except Exception as e:
            logger.error(f"Error processing item {item.get('path')}: {str(e)}")
            results.append({
                "path": item.get("path"),
                "type": item.get("type"),
                "actual_label": item.get("actual_label"),
                "error": str(e)
            })
    
    # Calculate final metrics
    if accuracy_metrics["total"] > 0:
        accuracy_metrics["accuracy"] = accuracy_metrics["correct"] / accuracy_metrics["total"]
        real_count = accuracy_metrics["true_negatives"] + accuracy_metrics["false_positives"]
        fake_count = accuracy_metrics["true_positives"] + accuracy_metrics["false_negatives"]
        if real_count > 0:
            accuracy_metrics["real_accuracy"] = accuracy_metrics["true_negatives"] / real_count
        if fake_count > 0:
            accuracy_metrics["fake_accuracy"] = accuracy_metrics["true_positives"] / fake_count
        if accuracy_metrics.get("precision") and accuracy_metrics.get("recall"):
            accuracy_metrics["f1"] = 2 * (accuracy_metrics["precision"] * accuracy_metrics["recall"]) / (accuracy_metrics["precision"] + accuracy_metrics["recall"])
    
    return jsonify({
        "success": True,
        "results": results,
        "metrics": accuracy_metrics
    })

def init_app(app):
    """Register the feedback API blueprint with the Flask app"""
    app.register_blueprint(feedback_bp, url_prefix='/api')
    logger.info("Feedback API initialized")
