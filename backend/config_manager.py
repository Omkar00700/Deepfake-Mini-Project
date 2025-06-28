
"""
Centralized Configuration Manager for DeepDefend
Provides runtime configuration updates and a centralized repository
for all application settings.
"""

import os
import json
import time
import logging
import threading
from typing import Dict, Any, Optional, List, Union, Callable
from flask import jsonify

# Configure logging
logger = logging.getLogger(__name__)

class ConfigurationManager:
    """
    Centralized configuration management system
    
    Features:
    - Runtime configuration updates
    - Observer pattern for configuration changes
    - Historical configuration tracking
    - Configuration validation
    - Default value handling
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigurationManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.config: Dict[str, Any] = {}
        self.default_config: Dict[str, Any] = {}
        self.schema: Dict[str, Dict[str, Any]] = {}
        self.config_lock = threading.RLock()
        self.observers: Dict[str, List[Callable[[str, Any], None]]] = {}
        self.history: List[Dict[str, Any]] = []
        self.initialized = False
        
        # Define schema for configuration settings
        self._define_schema()
    
    def _define_schema(self):
        """Define schema for configuration settings"""
        # Detection settings
        self.schema["DEEPFAKE_THRESHOLD"] = {
            "type": "float",
            "default": 0.5,
            "min": 0.0,
            "max": 1.0,
            "description": "Threshold above which an image is classified as deepfake",
            "category": "detection",
            "runtime_updatable": True,
            "requires_restart": False
        }
        
        self.schema["CONFIDENCE_THRESHOLD"] = {
            "type": "float",
            "default": 0.7,
            "min": 0.0,
            "max": 1.0,
            "description": "Minimum confidence threshold for considering a result reliable",
            "category": "detection",
            "runtime_updatable": True,
            "requires_restart": False
        }
        
        # Model settings
        self.schema["DEFAULT_MODEL"] = {
            "type": "string",
            "default": "efficientnet",
            "options": ["efficientnet", "xception", "mesonet"],
            "description": "Default model to use for detection",
            "category": "model",
            "runtime_updatable": True,
            "requires_restart": False
        }
        
        self.schema["ENABLE_ENSEMBLE_DETECTION"] = {
            "type": "boolean",
            "default": False,
            "description": "Enable ensemble detection using multiple models",
            "category": "model",
            "runtime_updatable": True,
            "requires_restart": False
        }
        
        self.schema["ENABLE_DYNAMIC_MODEL_SWITCHING"] = {
            "type": "boolean",
            "default": True,
            "description": "Allow automatic model switching based on confidence",
            "category": "model",
            "runtime_updatable": True,
            "requires_restart": False
        }
        
        # Video processing settings
        self.schema["VIDEO_MAX_FRAMES"] = {
            "type": "integer",
            "default": 30,
            "min": 1,
            "max": 1000,
            "description": "Maximum number of frames to analyze in videos",
            "category": "video",
            "runtime_updatable": True,
            "requires_restart": False
        }
        
        self.schema["VIDEO_FRAME_INTERVAL"] = {
            "type": "float",
            "default": 1.0,
            "min": 0.1,
            "max": 10.0,
            "description": "Interval between frames to analyze (in seconds)",
            "category": "video",
            "runtime_updatable": True,
            "requires_restart": False
        }
        
        self.schema["ENABLE_PARALLEL_PROCESSING"] = {
            "type": "boolean",
            "default": True,
            "description": "Enable parallel processing for videos",
            "category": "video",
            "runtime_updatable": True,
            "requires_restart": False
        }
        
        self.schema["MAX_WORKERS"] = {
            "type": "integer",
            "default": 4,
            "min": 1,
            "max": 16,
            "description": "Maximum number of worker threads",
            "category": "video",
            "runtime_updatable": True,
            "requires_restart": False
        }
        
        # Async processing settings
        self.schema["ENABLE_ASYNC_PROCESSING"] = {
            "type": "boolean",
            "default": False,
            "description": "Enable asynchronous processing for videos",
            "category": "async",
            "runtime_updatable": False,  # Requires restart
            "requires_restart": True
        }
        
        # Authentication settings
        self.schema["REQUIRE_AUTH"] = {
            "type": "boolean",
            "default": False,
            "description": "Require authentication for API endpoints",
            "category": "auth",
            "runtime_updatable": True,
            "requires_restart": False
        }
        
        self.schema["JWT_EXPIRATION_MINUTES"] = {
            "type": "integer",
            "default": 60,
            "min": 1,
            "max": 10080,  # 1 week
            "description": "JWT token expiration time in minutes",
            "category": "auth",
            "runtime_updatable": True,
            "requires_restart": False
        }
        
        # Rate limiting settings
        self.schema["ENABLE_RATE_LIMITING"] = {
            "type": "boolean",
            "default": False,
            "description": "Enable rate limiting for API endpoints",
            "category": "rate_limiting",
            "runtime_updatable": True,
            "requires_restart": False
        }
        
        self.schema["RATE_LIMIT_REQUESTS"] = {
            "type": "integer",
            "default": 100,
            "min": 1,
            "max": 10000,
            "description": "Maximum number of requests per time window",
            "category": "rate_limiting",
            "runtime_updatable": True,
            "requires_restart": False
        }
        
        self.schema["RATE_LIMIT_WINDOW"] = {
            "type": "integer",
            "default": 3600,
            "min": 60,
            "max": 86400,
            "description": "Rate limit window in seconds",
            "category": "rate_limiting",
            "runtime_updatable": True,
            "requires_restart": False
        }
        
        # Monitoring settings
        self.schema["ENABLE_PROMETHEUS_METRICS"] = {
            "type": "boolean",
            "default": False,
            "description": "Enable Prometheus metrics server",
            "category": "monitoring",
            "runtime_updatable": False,  # Requires restart
            "requires_restart": True
        }
        
        # Retraining settings
        self.schema["RETRAINING_ENABLED"] = {
            "type": "boolean",
            "default": False,
            "description": "Enable automatic model retraining",
            "category": "retraining",
            "runtime_updatable": True,
            "requires_restart": False
        }
        
        self.schema["RETRAINING_INTERVAL_HOURS"] = {
            "type": "integer",
            "default": 24,
            "min": 1,
            "max": 720,  # 30 days
            "description": "Interval between model evaluations (hours)",
            "category": "retraining",
            "runtime_updatable": True,
            "requires_restart": False
        }
        
        self.schema["RETRAINING_PERFORMANCE_THRESHOLD"] = {
            "type": "float",
            "default": 0.7,
            "min": 0.0,
            "max": 1.0,
            "description": "Performance threshold below which retraining is triggered",
            "category": "retraining",
            "runtime_updatable": True,
            "requires_restart": False
        }
        
        self.schema["FEEDBACK_COLLECTION_ENABLED"] = {
            "type": "boolean",
            "default": True,
            "description": "Enable collection of user feedback",
            "category": "retraining",
            "runtime_updatable": True,
            "requires_restart": False
        }
        
        # GraphQL settings
        self.schema["GRAPHQL_ENABLED"] = {
            "type": "boolean",
            "default": True,
            "description": "Enable GraphQL API",
            "category": "api",
            "runtime_updatable": False,  # Requires restart
            "requires_restart": True
        }
        
        self.schema["GRAPHQL_REQUIRE_AUTH"] = {
            "type": "boolean",
            "default": True,
            "description": "Require authentication for GraphQL API",
            "category": "api",
            "runtime_updatable": True,
            "requires_restart": False
        }
        
        # Auto scaling settings
        self.schema["AUTO_SCALING_ENABLED"] = {
            "type": "boolean",
            "default": False,
            "description": "Enable auto-scaling for Celery workers",
            "category": "scaling",
            "runtime_updatable": True,
            "requires_restart": False
        }
        
        self.schema["CELERY_MIN_WORKERS"] = {
            "type": "integer",
            "default": 2,
            "min": 1,
            "max": 100,
            "description": "Minimum number of Celery workers",
            "category": "scaling",
            "runtime_updatable": True,
            "requires_restart": False
        }
        
        self.schema["CELERY_MAX_WORKERS"] = {
            "type": "integer",
            "default": 10,
            "min": 1,
            "max": 100,
            "description": "Maximum number of Celery workers",
            "category": "scaling",
            "runtime_updatable": True,
            "requires_restart": False
        }
        
        # Set default values from schema
        for key, schema_item in self.schema.items():
            self.default_config[key] = schema_item["default"]
    
    def initialize(self):
        """Initialize configuration with values from environment variables"""
        if self.initialized:
            return
            
        with self.config_lock:
            # Start with default values
            self.config = self.default_config.copy()
            
            # Override with environment variables
            for key in self.schema:
                env_value = os.environ.get(key)
                if env_value is not None:
                    try:
                        # Convert to appropriate type
                        if self.schema[key]["type"] == "boolean":
                            self.config[key] = env_value.lower() == "true"
                        elif self.schema[key]["type"] == "integer":
                            self.config[key] = int(env_value)
                        elif self.schema[key]["type"] == "float":
                            self.config[key] = float(env_value)
                        else:
                            self.config[key] = env_value
                            
                        # Validate value
                        self._validate_setting(key, self.config[key])
                    except Exception as e:
                        logger.error(f"Error parsing environment variable {key}: {str(e)}")
                        # Keep default value
            
            # Record initial configuration
            self._record_history("initialization", self.config.copy())
            
            self.initialized = True
            logger.info("Configuration manager initialized")
    
    def _validate_setting(self, key: str, value: Any) -> bool:
        """
        Validate a configuration setting against its schema
        
        Args:
            key: Setting key
            value: Setting value
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ValueError: If validation fails
        """
        if key not in self.schema:
            raise ValueError(f"Unknown configuration key: {key}")
            
        schema_item = self.schema[key]
        
        # Check type
        if schema_item["type"] == "boolean" and not isinstance(value, bool):
            raise ValueError(f"Setting {key} must be a boolean")
        elif schema_item["type"] == "integer" and not isinstance(value, int):
            raise ValueError(f"Setting {key} must be an integer")
        elif schema_item["type"] == "float" and not isinstance(value, (int, float)):
            raise ValueError(f"Setting {key} must be a number")
        elif schema_item["type"] == "string" and not isinstance(value, str):
            raise ValueError(f"Setting {key} must be a string")
        
        # Check range for numeric values
        if schema_item["type"] in ["integer", "float"]:
            if "min" in schema_item and value < schema_item["min"]:
                raise ValueError(f"Setting {key} must be >= {schema_item['min']}")
            if "max" in schema_item and value > schema_item["max"]:
                raise ValueError(f"Setting {key} must be <= {schema_item['max']}")
        
        # Check options for string values
        if schema_item["type"] == "string" and "options" in schema_item:
            if value not in schema_item["options"]:
                raise ValueError(f"Setting {key} must be one of: {', '.join(schema_item['options'])}")
        
        return True
    
    def _record_history(self, action: str, data: Dict[str, Any]):
        """Record configuration change in history"""
        entry = {
            "timestamp": time.time(),
            "action": action,
            "data": data
        }
        self.history.append(entry)
        
        # Trim history if needed
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if not self.initialized:
            self.initialize()
            
        with self.config_lock:
            return self.config.get(key, default)
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values
        
        Returns:
            Dictionary of all configuration values
        """
        if not self.initialized:
            self.initialize()
            
        with self.config_lock:
            return self.config.copy()
    
    def update(self, key: str, value: Any) -> bool:
        """
        Update a configuration value
        
        Args:
            key: Configuration key
            value: New value
            
        Returns:
            True if update was successful, False otherwise
        """
        if not self.initialized:
            self.initialize()
            
        with self.config_lock:
            # Check if key exists
            if key not in self.schema:
                logger.error(f"Unknown configuration key: {key}")
                return False
                
            # Check if runtime updatable
            if not self.schema[key]["runtime_updatable"]:
                logger.error(f"Configuration key {key} cannot be updated at runtime")
                return False
                
            try:
                # Validate value
                self._validate_setting(key, value)
                
                # Record old value
                old_value = self.config.get(key)
                
                # Update value
                self.config[key] = value
                
                # Record history
                self._record_history("update", {
                    "key": key,
                    "old_value": old_value,
                    "new_value": value
                })
                
                # Notify observers
                self._notify_observers(key, value)
                
                logger.info(f"Configuration updated: {key} = {value}")
                return True
            except ValueError as e:
                logger.error(f"Invalid configuration value: {str(e)}")
                return False
    
    def update_many(self, updates: Dict[str, Any]) -> Dict[str, bool]:
        """
        Update multiple configuration values
        
        Args:
            updates: Dictionary of key-value pairs to update
            
        Returns:
            Dictionary of keys and update status
        """
        results = {}
        for key, value in updates.items():
            results[key] = self.update(key, value)
        return results
    
    def reset(self, key: str) -> bool:
        """
        Reset a configuration value to its default
        
        Args:
            key: Configuration key
            
        Returns:
            True if reset was successful, False otherwise
        """
        if not self.initialized:
            self.initialize()
            
        with self.config_lock:
            # Check if key exists
            if key not in self.schema:
                logger.error(f"Unknown configuration key: {key}")
                return False
                
            # Check if runtime updatable
            if not self.schema[key]["runtime_updatable"]:
                logger.error(f"Configuration key {key} cannot be updated at runtime")
                return False
                
            # Get default value
            default_value = self.default_config.get(key)
            
            # Record old value
            old_value = self.config.get(key)
            
            # Update value
            self.config[key] = default_value
            
            # Record history
            self._record_history("reset", {
                "key": key,
                "old_value": old_value,
                "new_value": default_value
            })
            
            # Notify observers
            self._notify_observers(key, default_value)
            
            logger.info(f"Configuration reset: {key} = {default_value}")
            return True
    
    def reset_all(self) -> bool:
        """
        Reset all configuration values to defaults
        
        Returns:
            True if reset was successful, False otherwise
        """
        if not self.initialized:
            self.initialize()
            
        with self.config_lock:
            # Record old values
            old_values = self.config.copy()
            
            # Update values
            for key, value in self.default_config.items():
                if self.schema[key]["runtime_updatable"]:
                    self.config[key] = value
                    self._notify_observers(key, value)
            
            # Record history
            self._record_history("reset_all", {
                "old_values": old_values,
                "new_values": self.config.copy()
            })
            
            logger.info("All configuration values reset to defaults")
            return True
    
    def get_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the configuration schema
        
        Returns:
            Dictionary of configuration schema
        """
        return self.schema.copy()
    
    def get_schema_for_key(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get the schema for a specific key
        
        Args:
            key: Configuration key
            
        Returns:
            Schema dictionary or None if key not found
        """
        return self.schema.get(key)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get configuration change history
        
        Returns:
            List of history entries
        """
        return self.history.copy()
    
    def register_observer(self, key: str, callback: Callable[[str, Any], None]):
        """
        Register an observer for a configuration key
        
        Args:
            key: Configuration key
            callback: Function to call when value changes
        """
        if key not in self.observers:
            self.observers[key] = []
        
        if callback not in self.observers[key]:
            self.observers[key].append(callback)
    
    def unregister_observer(self, key: str, callback: Callable[[str, Any], None]) -> bool:
        """
        Unregister an observer for a configuration key
        
        Args:
            key: Configuration key
            callback: Function to remove
            
        Returns:
            True if observer was removed, False otherwise
        """
        if key in self.observers and callback in self.observers[key]:
            self.observers[key].remove(callback)
            return True
        return False
    
    def _notify_observers(self, key: str, value: Any):
        """
        Notify observers of a configuration change
        
        Args:
            key: Configuration key
            value: New value
        """
        if key in self.observers:
            for callback in self.observers[key]:
                try:
                    callback(key, value)
                except Exception as e:
                    logger.error(f"Error in configuration observer: {str(e)}")
    
    def get_categories(self) -> Dict[str, List[str]]:
        """
        Get configuration keys grouped by category
        
        Returns:
            Dictionary of categories and keys
        """
        categories = {}
        for key, schema_item in self.schema.items():
            category = schema_item.get("category", "general")
            if category not in categories:
                categories[category] = []
            categories[category].append(key)
        return categories
    
    def get_settings_requiring_restart(self) -> List[str]:
        """
        Get list of settings that require restart to take effect
        
        Returns:
            List of setting keys
        """
        return [key for key, schema_item in self.schema.items() 
                if schema_item.get("requires_restart", False)]

# Create singleton instance
config_manager = ConfigurationManager()

def create_blueprint():
    """
    Create a Flask blueprint for configuration management API
    
    Returns:
        Flask Blueprint
    """
    from flask import Blueprint, request, jsonify
    from backend.auth import admin_required
    
    bp = Blueprint('config', __name__)
    
    @bp.route('/config', methods=['GET'])
    @admin_required
    def get_config():
        """Get all configuration values"""
        return jsonify({
            "config": config_manager.get_all(),
            "schema": config_manager.get_schema(),
            "categories": config_manager.get_categories(),
            "requires_restart": config_manager.get_settings_requiring_restart()
        })
    
    @bp.route('/config/<key>', methods=['GET'])
    @admin_required
    def get_config_key(key):
        """Get a specific configuration value"""
        value = config_manager.get(key)
        if value is None:
            return jsonify({"error": f"Unknown configuration key: {key}"}), 404
            
        schema = config_manager.get_schema_for_key(key)
        return jsonify({
            "key": key,
            "value": value,
            "schema": schema
        })
    
    @bp.route('/config/<key>', methods=['PUT'])
    @admin_required
    def update_config_key(key):
        """Update a specific configuration value"""
        data = request.get_json()
        if data is None or 'value' not in data:
            return jsonify({"error": "Missing required field: value"}), 400
            
        success = config_manager.update(key, data['value'])
        if not success:
            return jsonify({"error": f"Failed to update configuration key: {key}"}), 400
            
        return jsonify({
            "success": True,
            "key": key,
            "value": config_manager.get(key),
            "requires_restart": config_manager.get_schema_for_key(key).get("requires_restart", False)
        })
    
    @bp.route('/config', methods=['PUT'])
    @admin_required
    def update_config_many():
        """Update multiple configuration values"""
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No data provided"}), 400
            
        results = config_manager.update_many(data)
        
        # Check if any updates require restart
        requires_restart = any(
            config_manager.get_schema_for_key(key).get("requires_restart", False)
            for key, success in results.items()
            if success
        )
        
        return jsonify({
            "success": all(results.values()),
            "results": results,
            "requires_restart": requires_restart
        })
    
    @bp.route('/config/<key>/reset', methods=['POST'])
    @admin_required
    def reset_config_key(key):
        """Reset a specific configuration value to default"""
        success = config_manager.reset(key)
        if not success:
            return jsonify({"error": f"Failed to reset configuration key: {key}"}), 400
            
        return jsonify({
            "success": True,
            "key": key,
            "value": config_manager.get(key),
            "requires_restart": config_manager.get_schema_for_key(key).get("requires_restart", False)
        })
    
    @bp.route('/config/reset', methods=['POST'])
    @admin_required
    def reset_config_all():
        """Reset all configuration values to defaults"""
        success = config_manager.reset_all()
        
        # Check if any resets require restart
        requires_restart = any(
            schema_item.get("requires_restart", False) and schema_item.get("runtime_updatable", True)
            for key, schema_item in config_manager.get_schema().items()
        )
        
        return jsonify({
            "success": success,
            "requires_restart": requires_restart
        })
    
    @bp.route('/config/history', methods=['GET'])
    @admin_required
    def get_config_history():
        """Get configuration change history"""
        return jsonify({
            "history": config_manager.get_history()
        })
    
    @bp.route('/config/categories', methods=['GET'])
    @admin_required
    def get_config_categories():
        """Get configuration categories"""
        return jsonify({
            "categories": config_manager.get_categories()
        })
    
    return bp

def init_app(app):
    """Initialize configuration manager and register blueprint with Flask app"""
    # Make sure configuration is initialized
    config_manager.initialize()
    
    # Register blueprint
    app.register_blueprint(create_blueprint(), url_prefix='/api/admin')
    
    # Configure Celery based on configuration
    # This is just an example - in a real app, you'd have more sophisticated
    # integration with Celery
    if 'celery' in app.extensions:
        celery = app.extensions['celery']
        
        # Register observer for MAX_WORKERS
        def update_celery_concurrency(key, value):
            celery.conf.worker_concurrency = value
            
        config_manager.register_observer('MAX_WORKERS', update_celery_concurrency)
    
    logger.info("Configuration manager initialized")
