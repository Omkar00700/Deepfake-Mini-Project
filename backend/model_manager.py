
import os
import logging
import time
from typing import Dict, Any, List, Optional
import tensorflow as tf
import numpy as np
from model import DeepfakeDetectionModel
from backend.config import (
    MODEL_DIR, DEFAULT_MODEL, AVAILABLE_MODELS, ENABLE_DYNAMIC_MODEL_SWITCHING,
    AUTO_SWITCH_THRESHOLD, MODEL_SWITCH_COOLDOWN
)

# Configure logging
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Singleton class to manage deepfake detection models
    Handles loading, caching, and switching between different models
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.models = {}
        self.current_model_name = DEFAULT_MODEL
        self.last_switch_time = 0
        self.last_auto_switch_time = 0
        
        # Initialize default model
        self._load_model(self.current_model_name)
        
        # Preload other models if configured
        self._preload_models()
        
        logger.info(f"ModelManager initialized with default model: {self.current_model_name}")
    
    def _preload_models(self):
        """Preload all available models in a background thread to speed up first usage"""
        try:
            import threading
            
            def load_all_models():
                for model_name in AVAILABLE_MODELS:
                    if model_name != self.current_model_name:
                        try:
                            logger.info(f"Preloading model: {model_name}")
                            self._load_model(model_name)
                            logger.info(f"Successfully preloaded model: {model_name}")
                        except Exception as e:
                            logger.error(f"Failed to preload model {model_name}: {str(e)}")
                            
            # Start preloading in background thread
            thread = threading.Thread(target=load_all_models)
            thread.daemon = True
            thread.start()
        except Exception as e:
            logger.error(f"Error in model preloading: {str(e)}")
    
    def _load_model(self, model_name: str) -> Optional[DeepfakeDetectionModel]:
        """
        Load a model by name
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model or None if loading failed
        """
        if model_name in self.models:
            logger.debug(f"Model {model_name} already loaded")
            return self.models[model_name]
        
        try:
            logger.info(f"Loading model: {model_name}")
            model = DeepfakeDetectionModel(model_name=model_name, model_dir=MODEL_DIR)
            self.models[model_name] = model
            logger.info(f"Successfully loaded model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}", exc_info=True)
            
            # If this is the default model and it failed, try to load a different model
            if model_name == DEFAULT_MODEL and model_name != "mesonet":
                logger.warning(f"Default model {model_name} failed to load, trying mesonet as fallback")
                try:
                    model = DeepfakeDetectionModel(model_name="mesonet", model_dir=MODEL_DIR)
                    self.models["mesonet"] = model
                    self.current_model_name = "mesonet"
                    return model
                except Exception as e2:
                    logger.error(f"Failed to load fallback model: {str(e2)}", exc_info=True)
            return None
    
    def get_model(self) -> DeepfakeDetectionModel:
        """
        Get the current active model
        
        Returns:
            Current DeepfakeDetectionModel instance
        """
        # Ensure the current model is loaded
        if self.current_model_name not in self.models:
            self._load_model(self.current_model_name)
        
        # Return the current model
        return self.models.get(self.current_model_name)
    
    def get_model_by_name(self, model_name: str) -> Optional[DeepfakeDetectionModel]:
        """
        Get a specific model by name
        
        Args:
            model_name: Name of the model to get
            
        Returns:
            DeepfakeDetectionModel instance or None if not found
        """
        # Load the model if not already loaded
        if model_name not in self.models:
            return self._load_model(model_name)
        
        # Return the requested model
        return self.models.get(model_name)
    
    def get_all_models(self) -> Dict[str, DeepfakeDetectionModel]:
        """
        Get all loaded models
        
        Returns:
            Dict of model_name -> DeepfakeDetectionModel
        """
        return self.models
    
    def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different model
        
        Args:
            model_name: Name of the model to switch to
            
        Returns:
            True if switch was successful, False otherwise
        """
        # Check if model name is valid
        if model_name not in AVAILABLE_MODELS:
            logger.error(f"Invalid model name: {model_name}")
            return False
        
        # Check if we're already using this model
        if model_name == self.current_model_name:
            logger.info(f"Already using model: {model_name}")
            return True
        
        # Check cooldown period for model switching
        current_time = time.time()
        if current_time - self.last_switch_time < MODEL_SWITCH_COOLDOWN:
            logger.warning(f"Model switch cooldown in effect, cannot switch yet")
            return False
        
        # Try to load the model if it's not already loaded
        if model_name not in self.models:
            model = self._load_model(model_name)
            if model is None:
                logger.error(f"Failed to load model: {model_name}")
                return False
        
        # Switch to the new model
        self.current_model_name = model_name
        self.last_switch_time = current_time
        logger.info(f"Switched to model: {model_name}")
        return True
    
    def auto_switch_model(self, confidence: float) -> bool:
        """
        Automatically switch to a different model if confidence is low
        
        Args:
            confidence: Current confidence value
            
        Returns:
            True if a switch was performed, False otherwise
        """
        if not ENABLE_DYNAMIC_MODEL_SWITCHING:
            return False
        
        # Check if confidence is below threshold
        if confidence >= AUTO_SWITCH_THRESHOLD:
            return False
        
        # Check cooldown period
        current_time = time.time()
        if current_time - self.last_auto_switch_time < MODEL_SWITCH_COOLDOWN:
            return False
        
        # Try to switch to a different model
        current_index = AVAILABLE_MODELS.index(self.current_model_name)
        next_index = (current_index + 1) % len(AVAILABLE_MODELS)
        next_model = AVAILABLE_MODELS[next_index]
        
        logger.info(f"Auto-switching from {self.current_model_name} to {next_model} due to low confidence: {confidence}")
        
        # Update last auto-switch time even if switch fails
        self.last_auto_switch_time = current_time
        
        return self.switch_model(next_model)
    
    def get_current_model_name(self) -> str:
        """
        Get the name of the current model
        
        Returns:
            Current model name
        """
        return self.current_model_name
    
    def reload_model(self, model_name: str) -> bool:
        """
        Reload a specific model from disk
        
        Args:
            model_name: Name of the model to reload
            
        Returns:
            True if reload was successful, False otherwise
        """
        try:
            # Check if model name is valid
            if model_name not in AVAILABLE_MODELS:
                logger.error(f"Invalid model name: {model_name}")
                return False
            
            # Remove from cache if loaded
            if model_name in self.models:
                logger.info(f"Removing model {model_name} from cache")
                del self.models[model_name]
            
            # Force garbage collection to release resources
            try:
                import gc
                gc.collect()
            except Exception as e:
                logger.warning(f"Error during garbage collection: {str(e)}")
            
            # Load model again
            model = self._load_model(model_name)
            if model is None:
                logger.error(f"Failed to reload model: {model_name}")
                return False
            
            logger.info(f"Successfully reloaded model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error reloading model {model_name}: {str(e)}", exc_info=True)
            return False
    
    def reload_all_models(self) -> bool:
        """
        Reload all models from disk
        
        Returns:
            True if all reloads were successful, False if any failed
        """
        try:
            # Keep track of current model
            current_model = self.current_model_name
            
            # Clear models cache
            self.models = {}
            
            # Force garbage collection
            try:
                import gc
                gc.collect()
            except Exception as e:
                logger.warning(f"Error during garbage collection: {str(e)}")
            
            # Reload all models
            success = True
            for model_name in AVAILABLE_MODELS:
                try:
                    model = self._load_model(model_name)
                    if model is None:
                        logger.error(f"Failed to reload model: {model_name}")
                        success = False
                except Exception as e:
                    logger.error(f"Error reloading model {model_name}: {str(e)}", exc_info=True)
                    success = False
            
            # Set current model back
            self.current_model_name = current_model
            
            # Make sure current model is loaded
            if self.current_model_name not in self.models:
                # If current model failed to load, try to load another one
                for model_name in AVAILABLE_MODELS:
                    if model_name in self.models:
                        self.current_model_name = model_name
                        logger.warning(f"Current model failed to reload, switched to: {model_name}")
                        break
            
            logger.info(f"Reloaded all models. Success: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Error reloading all models: {str(e)}", exc_info=True)
            return False
    
    def get_models_info(self) -> Dict[str, Any]:
        """
        Get information about all models
        
        Returns:
            Dict with model information
        """
        info = {
            "available_models": AVAILABLE_MODELS,
            "current_model": self.current_model_name,
            "loaded_models": list(self.models.keys()),
            "auto_switch_enabled": ENABLE_DYNAMIC_MODEL_SWITCHING,
            "auto_switch_threshold": AUTO_SWITCH_THRESHOLD,
            "model_switch_cooldown": MODEL_SWITCH_COOLDOWN
        }
        
        # Add info about current model if available
        current_model = self.get_model()
        if current_model:
            info["current_model_details"] = {
                "name": current_model.model_name,
                "input_shape": current_model.input_shape,
                "description": current_model.get_model_description()
            }
        
        # Add info about all loaded models
        info["models_details"] = {}
        for name, model in self.models.items():
            info["models_details"][name] = {
                "name": model.model_name,
                "input_shape": model.input_shape,
                "description": model.get_model_description()
            }
        
        return info
