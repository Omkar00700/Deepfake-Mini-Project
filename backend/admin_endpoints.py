
from flask import Blueprint, request, jsonify
import logging
import os
from backend.auth import auth_required
from model_manager import ModelManager
from backend.config import (
    DEEPFAKE_THRESHOLD, CONFIDENCE_THRESHOLD, VIDEO_MAX_FRAMES,
    VIDEO_FRAME_INTERVAL, ENABLE_ENSEMBLE_DETECTION, CONFIDENCE_MIN_VALUE,
    CONFIDENCE_MAX_VALUE
)

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint for admin endpoints
admin_blueprint = Blueprint('admin', __name__, url_prefix='/api/admin')

# Get model manager instance
model_manager = ModelManager()

# Dictionary of configurable parameters with validation rules
CONFIGURABLE_PARAMS = {
    "deepfake_threshold": {
        "type": float,
        "min": 0.0,
        "max": 1.0,
        "default": DEEPFAKE_THRESHOLD,
        "env_var": "DEEPFAKE_THRESHOLD",
        "description": "Threshold above which an image is classified as deepfake"
    },
    "confidence_threshold": {
        "type": float,
        "min": 0.0,
        "max": 1.0,
        "default": CONFIDENCE_THRESHOLD,
        "env_var": "CONFIDENCE_THRESHOLD",
        "description": "Threshold for face detection confidence"
    },
    "video_max_frames": {
        "type": int,
        "min": 1,
        "max": 100,
        "default": VIDEO_MAX_FRAMES,
        "env_var": "VIDEO_MAX_FRAMES",
        "description": "Maximum number of frames to analyze in a video"
    },
    "video_frame_interval": {
        "type": float,
        "min": 0.1,
        "max": 10.0,
        "default": VIDEO_FRAME_INTERVAL,
        "env_var": "VIDEO_FRAME_INTERVAL",
        "description": "Interval between frames to analyze in seconds"
    },
    "enable_ensemble_detection": {
        "type": bool,
        "default": ENABLE_ENSEMBLE_DETECTION,
        "env_var": "ENABLE_ENSEMBLE_DETECTION",
        "description": "Whether to use ensemble of models for detection"
    },
    "confidence_min_value": {
        "type": float,
        "min": 0.0,
        "max": 0.5,
        "default": CONFIDENCE_MIN_VALUE,
        "env_var": "CONFIDENCE_MIN_VALUE",
        "description": "Minimum confidence value"
    },
    "confidence_max_value": {
        "type": float,
        "min": 0.5,
        "max": 1.0,
        "default": CONFIDENCE_MAX_VALUE,
        "env_var": "CONFIDENCE_MAX_VALUE",
        "description": "Maximum confidence value"
    }
}

# In-memory config to track runtime changes
runtime_config = {param: config["default"] for param, config in CONFIGURABLE_PARAMS.items()}

@admin_blueprint.route('/config', methods=['GET'])
@auth_required
def get_config():
    """
    Get the current configuration parameters
    
    Returns:
        JSON with all configurable parameters and their current values
    """
    try:
        # Get current model info
        model_info = model_manager.get_models_info()
        
        # Combine runtime config with model info
        config_info = {
            "runtime_config": runtime_config,
            "model_info": model_info,
            "configurable_params": CONFIGURABLE_PARAMS
        }
        
        return jsonify({
            "success": True,
            "config": config_info
        })
    except Exception as e:
        logger.error(f"Error getting config: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Error getting config: {str(e)}"
        }), 500

@admin_blueprint.route('/config', methods=['POST'])
@auth_required
def update_config():
    """
    Update configuration parameters at runtime
    
    Expects:
        JSON with parameter names and new values
        
    Returns:
        JSON with updated parameters
    """
    try:
        data = request.json or {}
        
        if not data:
            return jsonify({
                "success": False,
                "message": "No config updates provided"
            }), 400
        
        # Track updated parameters
        updated_params = {}
        validation_errors = {}
        
        # Process each parameter
        for param_name, new_value in data.items():
            # Skip if parameter is not configurable
            if param_name not in CONFIGURABLE_PARAMS:
                validation_errors[param_name] = f"Unknown parameter: {param_name}"
                continue
            
            # Get parameter config
            param_config = CONFIGURABLE_PARAMS[param_name]
            
            # Validate parameter type
            try:
                if param_config["type"] == bool and isinstance(new_value, str):
                    # Handle string representation of boolean
                    new_value = new_value.lower() in ('true', '1', 't', 'yes')
                else:
                    # Cast to correct type
                    new_value = param_config["type"](new_value)
            except (ValueError, TypeError):
                validation_errors[param_name] = f"Invalid type for {param_name}. Expected {param_config['type'].__name__}."
                continue
            
            # Validate min/max if applicable
            if "min" in param_config and new_value < param_config["min"]:
                validation_errors[param_name] = f"Value for {param_name} is below minimum ({param_config['min']})."
                continue
                
            if "max" in param_config and new_value > param_config["max"]:
                validation_errors[param_name] = f"Value for {param_name} is above maximum ({param_config['max']})."
                continue
            
            # Update runtime config
            runtime_config[param_name] = new_value
            updated_params[param_name] = new_value
            
            # Update environment variable (for processes that read directly from os.environ)
            if "env_var" in param_config:
                os.environ[param_config["env_var"]] = str(new_value)
            
            logger.info(f"Updated config parameter {param_name} to {new_value}")
        
        # Check if we have any updates
        if not updated_params:
            return jsonify({
                "success": False,
                "message": "No valid config updates provided",
                "validation_errors": validation_errors
            }), 400
        
        return jsonify({
            "success": True,
            "message": f"Updated {len(updated_params)} configuration parameters",
            "updated_params": updated_params,
            "validation_errors": validation_errors if validation_errors else None
        })
        
    except Exception as e:
        logger.error(f"Error updating config: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Error updating config: {str(e)}"
        }), 500

@admin_blueprint.route('/models/reload', methods=['POST'])
@auth_required
def reload_models():
    """
    Reload models from disk
    
    Returns:
        JSON with reload status
    """
    try:
        # Get model name from request
        data = request.json or {}
        model_name = data.get('model_name')
        
        # Check if we're reloading a specific model or all models
        if model_name:
            # Reload specific model
            result = model_manager.reload_model(model_name)
            message = f"Reloaded model: {model_name}" if result else f"Failed to reload model: {model_name}"
        else:
            # Reload all models
            result = model_manager.reload_all_models()
            message = f"Reloaded all models" if result else "Failed to reload all models"
        
        return jsonify({
            "success": result,
            "message": message,
            "models_info": model_manager.get_models_info()
        })
    except Exception as e:
        logger.error(f"Error reloading models: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Error reloading models: {str(e)}"
        }), 500

# Function to get the current runtime configuration value for a parameter
def get_config_value(param_name):
    """
    Get the current value of a configuration parameter
    
    Args:
        param_name: Name of the parameter
        
    Returns:
        Current value of the parameter
    """
    return runtime_config.get(param_name, CONFIGURABLE_PARAMS.get(param_name, {}).get("default"))
