"""
Model Ensemble for DeepDefend
Combines multiple deepfake detection models for improved accuracy
"""

import os
import logging
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Tuple, Optional
import time

# Configure logging
logger = logging.getLogger(__name__)

class ModelEnsemble:
    """
    Ensemble of deepfake detection models for improved accuracy
    """
    
    def __init__(self, model_paths: List[str] = None, weights: List[float] = None):
        """
        Initialize the model ensemble
        
        Args:
            model_paths: List of paths to model files
            weights: List of weights for each model (must sum to 1.0)
        """
        self.models = []
        self.model_names = []
        self.weights = weights or []
        self.model_paths = model_paths or []
        
        # Load default models if none provided
        if not self.model_paths:
            self._load_default_models()
        else:
            self._load_models(self.model_paths)
        
        # Normalize weights if provided
        if self.weights:
            if len(self.weights) != len(self.models):
                raise ValueError(f"Number of weights ({len(self.weights)}) must match number of models ({len(self.models)})")
            
            # Normalize weights to sum to 1.0
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
        else:
            # Equal weights if none provided
            self.weights = [1.0 / len(self.models) for _ in self.models]
        
        logger.info(f"Initialized model ensemble with {len(self.models)} models")
        for i, name in enumerate(self.model_names):
            logger.info(f"  Model {i+1}: {name} (weight: {self.weights[i]:.2f})")
    
    def _load_default_models(self):
        """Load default models for the ensemble"""
        # Define default model paths
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        default_models = [
            os.path.join(model_dir, "efficientnet_b0"),
            os.path.join(model_dir, "resnet50"),
            os.path.join(model_dir, "xception")
        ]
        
        # Load models that exist
        existing_models = [path for path in default_models if os.path.exists(path)]
        if not existing_models:
            raise ValueError("No default models found. Please provide model paths.")
        
        self._load_models(existing_models)
    
    def _load_models(self, model_paths: List[str]):
        """
        Load models from paths
        
        Args:
            model_paths: List of paths to model files
        """
        for path in model_paths:
            try:
                # Load model
                model = tf.keras.models.load_model(path)
                
                # Get model name
                model_name = os.path.basename(path)
                
                # Add to list
                self.models.append(model)
                self.model_names.append(model_name)
                
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Error loading model {path}: {str(e)}")
    
    def predict(self, image: np.ndarray) -> Tuple[float, float, Dict[str, Any]]:
        """
        Make a prediction using the ensemble
        
        Args:
            image: Input image as numpy array (preprocessed)
            
        Returns:
            Tuple of (probability, confidence, metadata)
        """
        if not self.models:
            raise ValueError("No models loaded in ensemble")
        
        # Track timing
        start_time = time.time()
        
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Get predictions from each model
        predictions = []
        model_predictions = {}
        
        for i, model in enumerate(self.models):
            try:
                # Get prediction
                pred = model.predict(image, verbose=0)
                
                # Extract probability (assume binary classification)
                if isinstance(pred, list):
                    prob = pred[0][0]  # Some models return a list of arrays
                else:
                    prob = pred[0][0]  # Single output
                
                # Add to list
                predictions.append(prob)
                
                # Store in dict
                model_predictions[self.model_names[i]] = float(prob)
                
            except Exception as e:
                logger.error(f"Error predicting with model {self.model_names[i]}: {str(e)}")
                # Use default value on error
                predictions.append(0.5)
                model_predictions[self.model_names[i]] = 0.5
        
        # Calculate weighted average
        ensemble_prob = sum(p * w for p, w in zip(predictions, self.weights))
        
        # Calculate confidence based on agreement between models
        # Higher agreement = higher confidence
        agreement = 1.0 - np.std(predictions)
        confidence = agreement * 0.8 + 0.2  # Scale to 0.2-1.0 range
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create metadata
        metadata = {
            "model_predictions": model_predictions,
            "model_weights": {name: weight for name, weight in zip(self.model_names, self.weights)},
            "model_agreement": float(agreement),
            "processing_time": processing_time
        }
        
        return float(ensemble_prob), float(confidence), metadata
    
    def update_weights(self, new_weights: List[float]) -> bool:
        """
        Update the weights for each model
        
        Args:
            new_weights: New weights for each model (must sum to 1.0)
            
        Returns:
            True if successful, False otherwise
        """
        if len(new_weights) != len(self.models):
            logger.error(f"Number of weights ({len(new_weights)}) must match number of models ({len(self.models)})")
            return False
        
        # Normalize weights
        total = sum(new_weights)
        normalized_weights = [w / total for w in new_weights]
        
        # Update weights
        self.weights = normalized_weights
        
        logger.info(f"Updated ensemble weights: {self.weights}")
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the ensemble
        
        Returns:
            Dictionary with ensemble information
        """
        return {
            "ensemble_size": len(self.models),
            "models": self.model_names,
            "weights": self.weights,
            "description": "DeepDefend Model Ensemble"
        }


# Create a singleton instance
_ensemble_instance = None

def get_ensemble() -> ModelEnsemble:
    """
    Get the singleton ensemble instance
    
    Returns:
        ModelEnsemble instance
    """
    global _ensemble_instance
    if _ensemble_instance is None:
        _ensemble_instance = ModelEnsemble()
    return _ensemble_instance

def predict_with_ensemble(image: np.ndarray) -> Tuple[float, float, Dict[str, Any]]:
    """
    Make a prediction using the ensemble
    
    Args:
        image: Input image as numpy array (preprocessed)
        
    Returns:
        Tuple of (probability, confidence, metadata)
    """
    ensemble = get_ensemble()
    return ensemble.predict(image)

def update_ensemble_weights(new_weights: List[float]) -> bool:
    """
    Update the weights for each model in the ensemble
    
    Args:
        new_weights: New weights for each model
        
    Returns:
        True if successful, False otherwise
    """
    ensemble = get_ensemble()
    return ensemble.update_weights(new_weights)

def get_ensemble_info() -> Dict[str, Any]:
    """
    Get information about the ensemble
    
    Returns:
        Dictionary with ensemble information
    """
    ensemble = get_ensemble()
    return ensemble.get_model_info()