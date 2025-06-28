
"""
API endpoints for model evaluation and performance metrics
"""

from flask import Blueprint, jsonify, request, current_app
from metrics_logger import (
    get_recent_metrics, get_recent_errors, calculate_performance_metrics
)
from backend.model_retraining import get_retraining_status, trigger_model_retraining, trigger_model_evaluation, trigger_model_evaluation
from inference import get_model_info, switch_model
from backend.auth import auth_required
import logging
import os
import json
import time
from backend.config import (
    DEBUG_MODE_ENABLED,
    LOG_INTERMEDIATE_RESULTS,
    VIDEO_ENSEMBLE_DEFAULT_WEIGHTS,
)


# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
evaluation_api = Blueprint('evaluation_api', __name__)

@evaluation_api.route('/status', methods=['GET'])
def get_evaluation_status():
    """Get current evaluation and retraining status"""
    try:
        # Get retraining status
        retraining_status = get_retraining_status()
        
        # Get performance metrics for last day
        performance = calculate_performance_metrics(days=1)
        
        # Get model info
        model_info = get_model_info()
        
        return jsonify({
            "success": True,
            "status": "online",
            "retraining": retraining_status,
            "performance": performance,
            "model": model_info
        })
    except Exception as e:
        logger.error(f"Error getting evaluation status: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@evaluation_api.route('/metrics', methods=['GET'])
@auauth_required
def get_metrics():
    """Get detection metrics"""
    try:
        # Parse query parameters
        limit = request.args.get('limit', default=100, type=int)
        days = request.args.get('days', default=1, type=int)
        metric_type = request.args.get('type')
        
        # Get metrics
        metrics = get_recent_metrics(limit=limit, metric_type=metric_type)
        
        # Get aggregate performance
        performance = calculate_performance_metrics(days=days)
        
        return jsonify({
            "success": True,
            "metrics": metrics,
            "performance": performance,
            "count": len(metrics)
        })
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@evaluation_api.route('/errors', methods=['GET'])
@auauth_required
def get_errors():
    """Get error logs"""
    try:
        # Parse query parameters
        limit = request.args.get('limit', default=50, type=int)
        error_type = request.args.get('type')
        
        # Get errors
        errors = get_recent_errors(limit=limit, error_type=error_type)
        
        return jsonify({
            "success": True,
            "errors": errors,
            "count": len(errors)
        })
    except Exception as e:
        logger.error(f"Error getting error logs: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@evaluation_api.route('/evaluate', methods=['POST'])
@auauth_required
def evaluate_model():
    """Trigger model evaluation"""
    try:
        # Trigger evaluation
        result = trigger_model_evaluation()
        
        if result:
            return jsonify({
                "success": True,
                "result": result
            })
        else:
            return jsonify({
                "success": False,
                "error": "Evaluation failed"
            }), 500
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@evaluation_api.route('/retrain', methods=['POST'])
@auauth_required
def retrain_model():
    """Trigger model retraining"""
    try:
        # Trigger retraining
        success = trigger_model_retraining()
        
        if success:
            return jsonify({
                "success": True,
                "message": "Retraining started successfully"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Retraining failed to start"
            }), 500
    except Exception as e:
        logger.error(f"Error starting retraining: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@evaluation_api.route('/models/switch', methods=['POST'])
@auauth_required
def switch_detection_model():
    """Switch to a different detection model"""
    try:
        # Get model name from request
        data = request.json
        model_name = data.get('model')
        
        if not model_name:
            return jsonify({
                "success": False,
                "error": "Missing model name"
            }), 400
        
        # Switch model
        success = switch_model(model_name)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Switched to model: {model_name}"
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Failed to switch to model: {model_name}"
            }), 500
    except Exception as e:
        logger.error(f"Error switching model: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@evaluation_api.route('/metrics/roc_auc', methods=['GET'])
@auauth_required
def get_roc_auc_metrics():
    """
    Get ROC curve and AUC metrics for each class
    
    Query parameters:
    - days: Number of days of data to include (default: 7)
    - model: Specific model to filter by (optional)
    """
    try:
        # Parse query parameters
        days = request.args.get('days', default=7, type=int)
        model = request.args.get('model')
        
        # Import necessary libraries
        from sklearn.metrics import roc_curve, auc
        import numpy as np
        
        # Get recent predictions with ground truth
        from metrics_logger import get_predictions_with_ground_truth
        predictions = get_predictions_with_ground_truth(days=days, model_name=model)
        
        if not predictions or len(predictions) < 10:
            return jsonify({
                "success": False,
                "error": f"Insufficient data for ROC/AUC calculation. Found {len(predictions) if predictions else 0} samples."
            }), 400
        
        # Separate real and fake classes
        y_true = np.array([p['ground_truth'] for p in predictions])
        y_pred = np.array([p['probability'] for p in predictions])
        
        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Create points for the ROC curve (limit to 100 points for response size)
        step = max(1, len(fpr) // 100)
        curve_points = [
            {"fpr": float(fpr[i]), "tpr": float(tpr[i]), "threshold": float(thresholds[i])}
            for i in range(0, len(fpr), step)
        ]
        
        # Calculate metrics at different thresholds
        thresholds_to_report = [0.1, 0.3, 0.5, 0.7, 0.9]
        threshold_metrics = []
        
        for threshold in thresholds_to_report:
            # Calculate metrics at this threshold
            y_pred_binary = (y_pred >= threshold).astype(int)
            
            # Calculate metrics
            tp = np.sum((y_pred_binary == 1) & (y_true == 1))
            fp = np.sum((y_pred_binary == 1) & (y_true == 0))
            tn = np.sum((y_pred_binary == 0) & (y_true == 0))
            fn = np.sum((y_pred_binary == 0) & (y_true == 1))
            
            # Calculate derived metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            threshold_metrics.append({
                "threshold": threshold,
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn)
            })
        
        # Return the results
        return jsonify({
            "success": True,
            "auc": float(roc_auc),
            "roc_curve": curve_points,
            "threshold_metrics": threshold_metrics,
            "sample_count": len(predictions),
            "days": days,
            "model": model or "all"
        })
    except Exception as e:
        logger.error(f"Error calculating ROC/AUC metrics: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@evaluation_api.route('/validate/dataset', methods=['POST'])
@token_required
def validate_on_dataset():
    """
    Validate model on a dataset
    
    Expected request format:
    {
        "dataset_path": "/path/to/dataset",
        "use_ensemble": true
    }
    """
    try:
        # Get parameters from request
        data = request.json
        dataset_path = data.get('dataset_path')
        use_ensemble = data.get('use_ensemble', True)
        
        if not dataset_path:
            return jsonify({
                "success": False,
                "error": "Missing dataset_path parameter"
            }), 400
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            return jsonify({
                "success": False,
                "error": f"Dataset path not found: {dataset_path}"
            }), 400
        
        # Load dataset metadata if exists
        metadata_path = os.path.join(dataset_path, "metadata.json")
        metadata = {}
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load dataset metadata: {str(e)}")
        
        # Import necessary modules
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        import numpy as np
        import os
        import cv2
        from detection_handler import process_image, process_video
        
        # Check if dataset has real and fake subdirectories
        real_dir = os.path.join(dataset_path, 'real')
        fake_dir = os.path.join(dataset_path, 'fake')
        
        if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
            return jsonify({
                "success": False,
                "error": "Dataset must contain 'real' and 'fake' subdirectories"
            }), 400
        
        # Get list of files
        real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov'))]
        fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov'))]
        
        # Limit the number of files to process if there are too many
        max_files = 100
        if len(real_files) > max_files:
            real_files = real_files[:max_files]
        if len(fake_files) > max_files:
            fake_files = fake_files[:max_files]
        
        logger.info(f"Validating on {len(real_files)} real and {len(fake_files)} fake files")
        
        # Process files and collect results
        results = []
        
        # Process real files
        for file_path in real_files:
            try:
                is_video = any(file_path.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov'])
                
                if is_video:
                    probability, confidence, frame_count, _ = process_video(file_path, use_ensemble=use_ensemble)
                else:
                    probability, confidence, _ = process_image(file_path, use_ensemble=use_ensemble)
                
                results.append({
                    'file': os.path.basename(file_path),
                    'type': 'real',
                    'ground_truth': 0,  # 0 for real
                    'probability': float(probability),
                    'confidence': float(confidence),
                    'is_video': is_video
                })
                
                logger.debug(f"Processed real file {os.path.basename(file_path)}: prob={probability:.4f}, conf={confidence:.4f}")
                
            except Exception as e:
                logger.error(f"Error processing real file {file_path}: {str(e)}")
        
        # Process fake files
        for file_path in fake_files:
            try:
                is_video = any(file_path.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov'])
                
                if is_video:
                    probability, confidence, frame_count, _ = process_video(file_path, use_ensemble=use_ensemble)
                else:
                    probability, confidence, _ = process_image(file_path, use_ensemble=use_ensemble)
                
                results.append({
                    'file': os.path.basename(file_path),
                    'type': 'fake',
                    'ground_truth': 1,  # 1 for fake
                    'probability': float(probability),
                    'confidence': float(confidence),
                    'is_video': is_video
                })
                
                logger.debug(f"Processed fake file {os.path.basename(file_path)}: prob={probability:.4f}, conf={confidence:.4f}")
                
            except Exception as e:
                logger.error(f"Error processing fake file {file_path}: {str(e)}")
        
        # Calculate metrics
        if not results:
            return jsonify({
                "success": False,
                "error": "No files were successfully processed"
            }), 500
        
        # Extract ground truth and predictions
        y_true = np.array([r['ground_truth'] for r in results])
        y_pred_prob = np.array([r['probability'] for r in results])
        y_pred = (y_pred_prob >= 0.5).astype(int)  # Binary predictions with threshold 0.5
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'auc': float(roc_auc_score(y_true, y_pred_prob)) if len(np.unique(y_true)) > 1 else 0.0
        }
        
        # Calculate separate metrics for images and videos
        image_results = [r for r in results if not r['is_video']]
        video_results = [r for r in results if r['is_video']]
        
        image_metrics = {}
        if image_results:
            y_true_img = np.array([r['ground_truth'] for r in image_results])
            y_pred_prob_img = np.array([r['probability'] for r in image_results])
            y_pred_img = (y_pred_prob_img >= 0.5).astype(int)
            
            image_metrics = {
                'accuracy': float(accuracy_score(y_true_img, y_pred_img)),
                'precision': float(precision_score(y_true_img, y_pred_img, zero_division=0)),
                'recall': float(recall_score(y_true_img, y_pred_img, zero_division=0)),
                'f1_score': float(f1_score(y_true_img, y_pred_img, zero_division=0)),
                'auc': float(roc_auc_score(y_true_img, y_pred_prob_img)) if len(np.unique(y_true_img)) > 1 else 0.0,
                'count': len(image_results)
            }
        
        video_metrics = {}
        if video_results:
            y_true_vid = np.array([r['ground_truth'] for r in video_results])
            y_pred_prob_vid = np.array([r['probability'] for r in video_results])
            y_pred_vid = (y_pred_prob_vid >= 0.5).astype(int)
            
            video_metrics = {
                'accuracy': float(accuracy_score(y_true_vid, y_pred_vid)),
                'precision': float(precision_score(y_true_vid, y_pred_vid, zero_division=0)),
                'recall': float(recall_score(y_true_vid, y_pred_vid, zero_division=0)),
                'f1_score': float(f1_score(y_true_vid, y_pred_vid, zero_division=0)),
                'auc': float(roc_auc_score(y_true_vid, y_pred_prob_vid)) if len(np.unique(y_true_vid)) > 1 else 0.0,
                'count': len(video_results)
            }
        
        # Save validation results
        from datetime import datetime
        validation_result = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': dataset_path,
            'use_ensemble': use_ensemble,
            'metrics': metrics,
            'image_metrics': image_metrics,
            'video_metrics': video_metrics,
            'file_count': len(results),
            'real_count': len(real_files),
            'fake_count': len(fake_files),
            'metadata': metadata
        }
        
        # Save to file
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'validation_results')
        os.makedirs(results_dir, exist_ok=True)
        
        result_file = os.path.join(results_dir, f'validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(result_file, 'w') as f:
            json.dump(validation_result, f, indent=2)
        
        return jsonify({
            "success": True,
            "metrics": metrics,
            "image_metrics": image_metrics,
            "video_metrics": video_metrics,
            "file_count": len(results),
            "results_file": result_file,
            "dataset": {
                "path": dataset_path,
                "metadata": metadata
            }
        })
    except Exception as e:
        logger.error(f"Error validating dataset: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@evaluation_api.route('/debug/detection', methods=['GET'])
def get_debug_info():
    """Get debug information about the detection system"""
    if not DEBUG_MODE_ENABLED:
        return jsonify({
            "success": False,
            "error": "Debug mode is disabled"
        }), 403
    
    try:
        # Get model info
        model_info = get_model_info()
        
        # Get ensemble weights
        from inference import get_ensemble_weights
        ensemble_weights = get_ensemble_weights()
        
        # Get config settings that affect detection
        config_settings = {
            "video_processing": {
                "max_frames": VIDEO_MAX_FRAMES,
                "frame_interval": VIDEO_FRAME_INTERVAL,
                "frame_similarity_threshold": FRAME_SIMILARITY_THRESHOLD,
                "scene_based_sampling": USE_SCENE_BASED_SAMPLING,
                "temporal_analysis": ENABLE_TEMPORAL_ANALYSIS
            },
            "ensemble": {
                "enabled": ENABLE_ENSEMBLE_DETECTION,
                "default_weights": VIDEO_ENSEMBLE_DEFAULT_WEIGHTS,
                "current_weights": ensemble_weights
            },
            "preprocessing": {
                "advanced_preprocessing": ENABLE_ADVANCED_PREPROCESSING,
                "adaptive_preprocessing": ENABLE_ADAPTIVE_PREPROCESSING,
                "indian_face_detection": INDIAN_FACE_DETECTION_ENABLED
            }
        }
        
        # Get recent detection metrics
        metrics = get_recent_metrics(limit=10, metric_type="video")
        
        return jsonify({
            "success": True,
            "timestamp": time.time(),
            "model_info": model_info,
            "config": config_settings,
            "recent_metrics": metrics
        })
    except Exception as e:
        logger.error(f"Error getting debug info: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@evaluation_api.route('/debug/test-detection', methods=['POST'])
@token_required
def test_detection_pipeline():
    """
    Test the detection pipeline with specific parameters
    
    This endpoint allows testing different configuration settings
    to diagnose detection issues.
    """
    if not DEBUG_MODE_ENABLED:
        return jsonify({
            "success": False,
            "error": "Debug mode is disabled"
        }), 403
    
    try:
        # Get parameters from request
        data = request.json
        media_path = data.get('media_path')
        config_overrides = data.get('config_overrides', {})
        
        if not media_path:
            return jsonify({
                "success": False,
                "error": "Missing media_path parameter"
            }), 400
        
        if not os.path.exists(media_path):
            return jsonify({
                "success": False,
                "error": f"Media path not found: {media_path}"
            }), 404
        
        # Determine if it's a video or image
        is_video = any(media_path.lower().endswith(ext) for ext in 
                     ['.mp4', '.avi', '.mov', '.mkv', '.webm'])
        
        logger.info(f"Testing detection on {'video' if is_video else 'image'}: {media_path}")
        
        # Apply temporary config overrides
        original_config = {}
        for key, value in config_overrides.items():
            if hasattr(import_module('config'), key):
                original_config[key] = getattr(import_module('config'), key)
                setattr(import_module('config'), key, value)
                logger.debug(f"Temporarily overriding config {key}={value}")
        
        try:
            # Process the media with detailed logging
            if is_video:
                from detection_handler import process_video
                probability, confidence, frame_count, regions = process_video(media_path)
                result = {
                    "media_type": "video",
                    "probability": float(probability),
                    "confidence": float(confidence),
                    "frames_processed": frame_count,
                    "regions_count": len(regions),
                    "regions": regions if LOG_INTERMEDIATE_RESULTS else "omitted for brevity"
                }
            else:
                from detection_handler import process_image
                probability, confidence, regions = process_image(media_path)
                result = {
                    "media_type": "image",
                    "probability": float(probability),
                    "confidence": float(confidence),
                    "regions_count": len(regions),
                    "regions": regions if LOG_INTERMEDIATE_RESULTS else "omitted for brevity"
                }
                
            # Include applied config overrides in result
            result["config_overrides"] = config_overrides
            
            return jsonify({
                "success": True,
                "result": result
            })
            
        finally:
            # Restore original config values
            for key, value in original_config.items():
                setattr(import_module('config'), key, value)
                logger.debug(f"Restored config {key}={value}")
    
    except Exception as e:
        logger.error(f"Error in test detection: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
