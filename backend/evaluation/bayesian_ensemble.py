"""
Bayesian model averaging and deep ensembles for deepfake detection
"""

import tensorflow as tf
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional, Union, Callable
import os
import json
import time
from pathlib import Path
import scipy.stats

# Configure logging
logger = logging.getLogger(__name__)

class DeepEnsemble:
    """
    Deep ensemble for deepfake detection
    Combines multiple models trained with different random initializations
    """
    
    def __init__(self, 
                 models: Optional[List[tf.keras.Model]] = None,
                 model_paths: Optional[List[str]] = None,
                 weights: Optional[List[float]] = None,
                 temperature: float = 1.0):
        """
        Initialize deep ensemble
        
        Args:
            models: List of trained models
            model_paths: List of paths to saved models
            weights: List of weights for each model
            temperature: Temperature for calibration
        """
        self.models = models or []
        self.weights = weights
        self.temperature = temperature
        
        # Load models from paths if provided
        if model_paths:
            self._load_models(model_paths)
        
        # Normalize weights if provided
        if self.weights is not None:
            self.weights = np.array(self.weights) / np.sum(self.weights)
        else:
            # Equal weights by default
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        logger.info(f"Initialized deep ensemble with {len(self.models)} models")
    
    def _load_models(self, model_paths: List[str]) -> None:
        """
        Load models from paths
        
        Args:
            model_paths: List of paths to saved models
        """
        for path in model_paths:
            try:
                model = tf.keras.models.load_model(path)
                self.models.append(model)
                logger.info(f"Loaded model from {path}")
            except Exception as e:
                logger.error(f"Failed to load model from {path}: {str(e)}")
    
    def predict(self, 
               x: np.ndarray, 
               return_individual: bool = False,
               return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Make predictions with the ensemble
        
        Args:
            x: Input data
            return_individual: Whether to return individual model predictions
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Ensemble predictions, and optionally individual predictions and uncertainty
        """
        # Get predictions from each model
        individual_preds = []
        
        for i, model in enumerate(self.models):
            # Make prediction
            pred = model.predict(x)
            
            # Apply temperature scaling
            if self.temperature != 1.0:
                pred = self._apply_temperature_scaling(pred, self.temperature)
            
            individual_preds.append(pred)
        
        # Stack predictions
        individual_preds = np.stack(individual_preds, axis=0)  # [n_models, n_samples, n_classes]
        
        # Calculate weighted average
        ensemble_pred = np.sum(individual_preds * self.weights[:, np.newaxis, np.newaxis], axis=0)
        
        if return_individual or return_uncertainty:
            # Calculate uncertainty
            uncertainty = np.std(individual_preds, axis=0)
            
            if return_individual:
                return ensemble_pred, individual_preds, uncertainty
            else:
                return ensemble_pred, uncertainty
        else:
            return ensemble_pred
    
    def _apply_temperature_scaling(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """
        Apply temperature scaling to logits
        
        Args:
            logits: Model logits
            temperature: Temperature parameter
            
        Returns:
            Calibrated probabilities
        """
        # If logits are already probabilities, convert to logits
        if np.all((logits >= 0) & (logits <= 1)):
            # Add small epsilon to avoid log(0)
            epsilon = 1e-7
            logits = np.log(logits + epsilon) - np.log(1 - logits + epsilon)
        
        # Apply temperature scaling
        scaled_logits = logits / temperature
        
        # Convert back to probabilities
        probabilities = 1 / (1 + np.exp(-scaled_logits))
        
        return probabilities
    
    def calibrate(self, 
                x_val: np.ndarray, 
                y_val: np.ndarray,
                temperature_range: Tuple[float, float] = (0.1, 10.0),
                n_steps: int = 100) -> float:
        """
        Calibrate ensemble using temperature scaling
        
        Args:
            x_val: Validation data
            y_val: Validation labels
            temperature_range: Range of temperatures to try
            n_steps: Number of temperature steps
            
        Returns:
            Optimal temperature
        """
        # Get raw predictions from each model
        individual_preds = []
        
        for model in self.models:
            pred = model.predict(x_val)
            individual_preds.append(pred)
        
        # Stack predictions
        individual_preds = np.stack(individual_preds, axis=0)  # [n_models, n_samples, n_classes]
        
        # Calculate weighted average
        ensemble_pred = np.sum(individual_preds * self.weights[:, np.newaxis, np.newaxis], axis=0)
        
        # Try different temperatures
        temperatures = np.linspace(temperature_range[0], temperature_range[1], n_steps)
        best_nll = float('inf')
        best_temperature = 1.0
        
        for temperature in temperatures:
            # Apply temperature scaling
            calibrated_pred = self._apply_temperature_scaling(ensemble_pred, temperature)
            
            # Calculate negative log likelihood
            nll = self._negative_log_likelihood(y_val, calibrated_pred)
            
            # Update best temperature
            if nll < best_nll:
                best_nll = nll
                best_temperature = temperature
        
        # Set temperature
        self.temperature = best_temperature
        
        logger.info(f"Calibrated ensemble with temperature {best_temperature:.4f}")
        
        return best_temperature
    
    def _negative_log_likelihood(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate negative log likelihood
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            
        Returns:
            Negative log likelihood
        """
        # Convert to one-hot if needed
        if len(y_true.shape) == 1:
            y_true = tf.keras.utils.to_categorical(y_true, num_classes=y_pred.shape[1])
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Calculate negative log likelihood
        nll = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        return nll
    
    def save(self, save_dir: str) -> None:
        """
        Save ensemble configuration
        
        Args:
            save_dir: Directory to save configuration
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        model_paths = []
        
        for i, model in enumerate(self.models):
            model_path = os.path.join(save_dir, f"model_{i}")
            model.save(model_path)
            model_paths.append(model_path)
        
        # Save configuration
        config = {
            "model_paths": model_paths,
            "weights": self.weights.tolist(),
            "temperature": float(self.temperature)
        }
        
        config_path = os.path.join(save_dir, "ensemble_config.json")
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved ensemble configuration to {config_path}")
    
    @classmethod
    def load(cls, config_path: str) -> "DeepEnsemble":
        """
        Load ensemble from configuration
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded ensemble
        """
        # Load configuration
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create ensemble
        ensemble = cls(
            model_paths=config["model_paths"],
            weights=config["weights"],
            temperature=config["temperature"]
        )
        
        return ensemble


class BayesianModelAveraging:
    """
    Bayesian model averaging for deepfake detection
    Combines multiple models with weights based on their evidence
    """
    
    def __init__(self, 
                 models: Optional[List[tf.keras.Model]] = None,
                 model_paths: Optional[List[str]] = None,
                 prior_weights: Optional[List[float]] = None,
                 temperature: float = 1.0):
        """
        Initialize Bayesian model averaging
        
        Args:
            models: List of trained models
            model_paths: List of paths to saved models
            prior_weights: List of prior weights for each model
            temperature: Temperature for calibration
        """
        self.models = models or []
        self.prior_weights = prior_weights
        self.temperature = temperature
        
        # Posterior weights (to be computed)
        self.posterior_weights = None
        
        # Load models from paths if provided
        if model_paths:
            self._load_models(model_paths)
        
        # Normalize prior weights if provided
        if self.prior_weights is not None:
            self.prior_weights = np.array(self.prior_weights) / np.sum(self.prior_weights)
        else:
            # Equal weights by default
            self.prior_weights = np.ones(len(self.models)) / len(self.models)
        
        logger.info(f"Initialized Bayesian model averaging with {len(self.models)} models")
    
    def _load_models(self, model_paths: List[str]) -> None:
        """
        Load models from paths
        
        Args:
            model_paths: List of paths to saved models
        """
        for path in model_paths:
            try:
                model = tf.keras.models.load_model(path)
                self.models.append(model)
                logger.info(f"Loaded model from {path}")
            except Exception as e:
                logger.error(f"Failed to load model from {path}: {str(e)}")
    
    def compute_posterior_weights(self, 
                                x_val: np.ndarray, 
                                y_val: np.ndarray) -> np.ndarray:
        """
        Compute posterior weights for each model
        
        Args:
            x_val: Validation data
            y_val: Validation labels
            
        Returns:
            Posterior weights
        """
        # Get predictions from each model
        log_likelihoods = []
        
        for model in self.models:
            # Make prediction
            pred = model.predict(x_val)
            
            # Apply temperature scaling
            if self.temperature != 1.0:
                pred = self._apply_temperature_scaling(pred, self.temperature)
            
            # Calculate log likelihood
            log_likelihood = self._log_likelihood(y_val, pred)
            log_likelihoods.append(log_likelihood)
        
        # Convert to numpy array
        log_likelihoods = np.array(log_likelihoods)
        
        # Calculate log evidence for each model
        log_evidence = log_likelihoods + np.log(self.prior_weights)
        
        # Normalize to get posterior weights
        # Use log-sum-exp trick for numerical stability
        max_log_evidence = np.max(log_evidence)
        evidence = np.exp(log_evidence - max_log_evidence)
        posterior_weights = evidence / np.sum(evidence)
        
        # Store posterior weights
        self.posterior_weights = posterior_weights
        
        logger.info(f"Computed posterior weights: {posterior_weights}")
        
        return posterior_weights
    
    def predict(self, 
               x: np.ndarray, 
               return_individual: bool = False,
               return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Make predictions with Bayesian model averaging
        
        Args:
            x: Input data
            return_individual: Whether to return individual model predictions
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Ensemble predictions, and optionally individual predictions and uncertainty
        """
        # Use posterior weights if available, otherwise use prior weights
        weights = self.posterior_weights if self.posterior_weights is not None else self.prior_weights
        
        # Get predictions from each model
        individual_preds = []
        
        for i, model in enumerate(self.models):
            # Make prediction
            pred = model.predict(x)
            
            # Apply temperature scaling
            if self.temperature != 1.0:
                pred = self._apply_temperature_scaling(pred, self.temperature)
            
            individual_preds.append(pred)
        
        # Stack predictions
        individual_preds = np.stack(individual_preds, axis=0)  # [n_models, n_samples, n_classes]
        
        # Calculate weighted average
        ensemble_pred = np.sum(individual_preds * weights[:, np.newaxis, np.newaxis], axis=0)
        
        if return_individual or return_uncertainty:
            # Calculate uncertainty
            # Total uncertainty = aleatoric + epistemic
            # Aleatoric: average of individual model uncertainties
            # Epistemic: variance of model predictions
            
            # Aleatoric uncertainty (predictive entropy)
            aleatoric = -np.sum(ensemble_pred * np.log(ensemble_pred + 1e-7), axis=1)
            
            # Epistemic uncertainty (mutual information)
            avg_entropy = -np.sum(
                np.sum(individual_preds * np.log(individual_preds + 1e-7), axis=2) * weights[:, np.newaxis],
                axis=0
            )
            epistemic = aleatoric - avg_entropy
            
            # Total uncertainty
            uncertainty = aleatoric + epistemic
            
            if return_individual:
                return ensemble_pred, individual_preds, uncertainty
            else:
                return ensemble_pred, uncertainty
        else:
            return ensemble_pred
    
    def _apply_temperature_scaling(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """
        Apply temperature scaling to logits
        
        Args:
            logits: Model logits
            temperature: Temperature parameter
            
        Returns:
            Calibrated probabilities
        """
        # If logits are already probabilities, convert to logits
        if np.all((logits >= 0) & (logits <= 1)):
            # Add small epsilon to avoid log(0)
            epsilon = 1e-7
            logits = np.log(logits + epsilon) - np.log(1 - logits + epsilon)
        
        # Apply temperature scaling
        scaled_logits = logits / temperature
        
        # Convert back to probabilities
        probabilities = 1 / (1 + np.exp(-scaled_logits))
        
        return probabilities
    
    def _log_likelihood(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate log likelihood
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            
        Returns:
            Log likelihood
        """
        # Convert to one-hot if needed
        if len(y_true.shape) == 1:
            y_true = tf.keras.utils.to_categorical(y_true, num_classes=y_pred.shape[1])
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Calculate log likelihood
        log_likelihood = np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        return log_likelihood
    
    def calibrate(self, 
                x_val: np.ndarray, 
                y_val: np.ndarray,
                temperature_range: Tuple[float, float] = (0.1, 10.0),
                n_steps: int = 100) -> float:
        """
        Calibrate ensemble using temperature scaling
        
        Args:
            x_val: Validation data
            y_val: Validation labels
            temperature_range: Range of temperatures to try
            n_steps: Number of temperature steps
            
        Returns:
            Optimal temperature
        """
        # Use posterior weights if available, otherwise use prior weights
        weights = self.posterior_weights if self.posterior_weights is not None else self.prior_weights
        
        # Get raw predictions from each model
        individual_preds = []
        
        for model in self.models:
            pred = model.predict(x_val)
            individual_preds.append(pred)
        
        # Stack predictions
        individual_preds = np.stack(individual_preds, axis=0)  # [n_models, n_samples, n_classes]
        
        # Calculate weighted average
        ensemble_pred = np.sum(individual_preds * weights[:, np.newaxis, np.newaxis], axis=0)
        
        # Try different temperatures
        temperatures = np.linspace(temperature_range[0], temperature_range[1], n_steps)
        best_nll = float('inf')
        best_temperature = 1.0
        
        for temperature in temperatures:
            # Apply temperature scaling
            calibrated_pred = self._apply_temperature_scaling(ensemble_pred, temperature)
            
            # Calculate negative log likelihood
            nll = -self._log_likelihood(y_val, calibrated_pred)
            
            # Update best temperature
            if nll < best_nll:
                best_nll = nll
                best_temperature = temperature
        
        # Set temperature
        self.temperature = best_temperature
        
        logger.info(f"Calibrated ensemble with temperature {best_temperature:.4f}")
        
        return best_temperature
    
    def save(self, save_dir: str) -> None:
        """
        Save ensemble configuration
        
        Args:
            save_dir: Directory to save configuration
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        model_paths = []
        
        for i, model in enumerate(self.models):
            model_path = os.path.join(save_dir, f"model_{i}")
            model.save(model_path)
            model_paths.append(model_path)
        
        # Save configuration
        config = {
            "model_paths": model_paths,
            "prior_weights": self.prior_weights.tolist(),
            "temperature": float(self.temperature)
        }
        
        if self.posterior_weights is not None:
            config["posterior_weights"] = self.posterior_weights.tolist()
        
        config_path = os.path.join(save_dir, "bma_config.json")
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved BMA configuration to {config_path}")
    
    @classmethod
    def load(cls, config_path: str) -> "BayesianModelAveraging":
        """
        Load BMA from configuration
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded BMA
        """
        # Load configuration
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create BMA
        bma = cls(
            model_paths=config["model_paths"],
            prior_weights=config["prior_weights"],
            temperature=config["temperature"]
        )
        
        # Set posterior weights if available
        if "posterior_weights" in config:
            bma.posterior_weights = np.array(config["posterior_weights"])
        
        return bma


class PostProcessingCalibrator:
    """
    Post-processing calibrator for deepfake detection
    Combines temperature scaling and isotonic regression
    """
    
    def __init__(self, 
                 temperature: float = 1.0,
                 use_isotonic: bool = True):
        """
        Initialize post-processing calibrator
        
        Args:
            temperature: Temperature for scaling
            use_isotonic: Whether to use isotonic regression
        """
        self.temperature = temperature
        self.use_isotonic = use_isotonic
        self.isotonic_calibrator = None
        
        if use_isotonic:
            from sklearn.isotonic import IsotonicRegression
            self.isotonic_calibrator = IsotonicRegression(out_of_bounds="clip")
        
        logger.info(f"Initialized post-processing calibrator with temperature={temperature}, "
                   f"use_isotonic={use_isotonic}")
    
    def fit(self, 
           y_pred: np.ndarray, 
           y_true: np.ndarray,
           temperature_range: Tuple[float, float] = (0.1, 10.0),
           n_steps: int = 100) -> None:
        """
        Fit calibrator to data
        
        Args:
            y_pred: Predicted probabilities
            y_true: True labels
            temperature_range: Range of temperatures to try
            n_steps: Number of temperature steps
        """
        # Find optimal temperature
        self.temperature = self._find_optimal_temperature(
            y_pred, y_true, temperature_range, n_steps
        )
        
        # Apply temperature scaling
        y_pred_temp = self._apply_temperature_scaling(y_pred, self.temperature)
        
        # Fit isotonic regression if enabled
        if self.use_isotonic and self.isotonic_calibrator is not None:
            # For binary classification, use the positive class probability
            if y_pred_temp.shape[1] == 2:
                y_pred_temp = y_pred_temp[:, 1]
            
            # Convert y_true to binary if needed
            if len(y_true.shape) > 1:
                y_true = np.argmax(y_true, axis=1)
            
            # Fit isotonic regression
            self.isotonic_calibrator.fit(y_pred_temp, y_true)
            
            logger.info("Fitted isotonic regression calibrator")
    
    def calibrate(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Calibrate predictions
        
        Args:
            y_pred: Predicted probabilities
            
        Returns:
            Calibrated probabilities
        """
        # Apply temperature scaling
        y_pred_temp = self._apply_temperature_scaling(y_pred, self.temperature)
        
        # Apply isotonic regression if enabled
        if self.use_isotonic and self.isotonic_calibrator is not None:
            # For binary classification, use the positive class probability
            if y_pred_temp.shape[1] == 2:
                y_pred_iso = self.isotonic_calibrator.transform(y_pred_temp[:, 1])
                
                # Convert back to two-class format
                y_pred_cal = np.zeros_like(y_pred_temp)
                y_pred_cal[:, 1] = y_pred_iso
                y_pred_cal[:, 0] = 1 - y_pred_iso
                
                return y_pred_cal
            else:
                # For multi-class, calibrate each class separately
                y_pred_cal = np.zeros_like(y_pred_temp)
                
                for i in range(y_pred_temp.shape[1]):
                    y_pred_cal[:, i] = self.isotonic_calibrator.transform(y_pred_temp[:, i])
                
                # Normalize to ensure sum to 1
                y_pred_cal = y_pred_cal / np.sum(y_pred_cal, axis=1, keepdims=True)
                
                return y_pred_cal
        else:
            return y_pred_temp
    
    def _find_optimal_temperature(self, 
                                y_pred: np.ndarray, 
                                y_true: np.ndarray,
                                temperature_range: Tuple[float, float],
                                n_steps: int) -> float:
        """
        Find optimal temperature for scaling
        
        Args:
            y_pred: Predicted probabilities
            y_true: True labels
            temperature_range: Range of temperatures to try
            n_steps: Number of temperature steps
            
        Returns:
            Optimal temperature
        """
        # Try different temperatures
        temperatures = np.linspace(temperature_range[0], temperature_range[1], n_steps)
        best_nll = float('inf')
        best_temperature = 1.0
        
        for temperature in temperatures:
            # Apply temperature scaling
            y_pred_temp = self._apply_temperature_scaling(y_pred, temperature)
            
            # Calculate negative log likelihood
            nll = self._negative_log_likelihood(y_true, y_pred_temp)
            
            # Update best temperature
            if nll < best_nll:
                best_nll = nll
                best_temperature = temperature
        
        logger.info(f"Found optimal temperature: {best_temperature:.4f}")
        
        return best_temperature
    
    def _apply_temperature_scaling(self, y_pred: np.ndarray, temperature: float) -> np.ndarray:
        """
        Apply temperature scaling to predictions
        
        Args:
            y_pred: Predicted probabilities
            temperature: Temperature parameter
            
        Returns:
            Calibrated probabilities
        """
        # If predictions are already probabilities, convert to logits
        if np.all((y_pred >= 0) & (y_pred <= 1)):
            # Add small epsilon to avoid log(0)
            epsilon = 1e-7
            logits = np.log(y_pred + epsilon) - np.log(1 - y_pred + epsilon)
        else:
            logits = y_pred
        
        # Apply temperature scaling
        scaled_logits = logits / temperature
        
        # Convert back to probabilities
        probabilities = 1 / (1 + np.exp(-scaled_logits))
        
        return probabilities
    
    def _negative_log_likelihood(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate negative log likelihood
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            
        Returns:
            Negative log likelihood
        """
        # Convert to one-hot if needed
        if len(y_true.shape) == 1:
            y_true = tf.keras.utils.to_categorical(y_true, num_classes=y_pred.shape[1])
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Calculate negative log likelihood
        nll = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        return nll
    
    def save(self, save_path: str) -> None:
        """
        Save calibrator configuration
        
        Args:
            save_path: Path to save configuration
        """
        import pickle
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save calibrator
        with open(save_path, "wb") as f:
            pickle.dump({
                "temperature": self.temperature,
                "use_isotonic": self.use_isotonic,
                "isotonic_calibrator": self.isotonic_calibrator
            }, f)
        
        logger.info(f"Saved calibrator to {save_path}")
    
    @classmethod
    def load(cls, load_path: str) -> "PostProcessingCalibrator":
        """
        Load calibrator from configuration
        
        Args:
            load_path: Path to configuration file
            
        Returns:
            Loaded calibrator
        """
        import pickle
        
        # Load calibrator
        with open(load_path, "rb") as f:
            config = pickle.load(f)
        
        # Create calibrator
        calibrator = cls(
            temperature=config["temperature"],
            use_isotonic=config["use_isotonic"]
        )
        
        # Set isotonic calibrator
        calibrator.isotonic_calibrator = config["isotonic_calibrator"]
        
        return calibrator


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy models and data
    models = [
        tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation="relu", input_shape=(5,)),
            tf.keras.layers.Dense(2, activation="softmax")
        ])
        for _ in range(3)
    ]
    
    # Create dummy data
    x = np.random.random((100, 5))
    y = np.random.randint(0, 2, size=(100,))
    y_onehot = tf.keras.utils.to_categorical(y, num_classes=2)
    
    # Create deep ensemble
    ensemble = DeepEnsemble(models=models)
    
    # Make predictions
    ensemble_pred = ensemble.predict(x)
    
    print(f"Ensemble predictions shape: {ensemble_pred.shape}")
    
    # Create Bayesian model averaging
    bma = BayesianModelAveraging(models=models)
    
    # Compute posterior weights
    bma.compute_posterior_weights(x, y_onehot)
    
    # Make predictions
    bma_pred = bma.predict(x)
    
    print(f"BMA predictions shape: {bma_pred.shape}")
    
    # Create post-processing calibrator
    calibrator = PostProcessingCalibrator(use_isotonic=True)
    
    # Fit calibrator
    calibrator.fit(ensemble_pred, y_onehot)
    
    # Calibrate predictions
    calibrated_pred = calibrator.calibrate(ensemble_pred)
    
    print(f"Calibrated predictions shape: {calibrated_pred.shape}")