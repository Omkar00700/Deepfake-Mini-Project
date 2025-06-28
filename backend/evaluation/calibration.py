"""
Calibration methods for deepfake detection
Implements temperature scaling and isotonic regression
"""

import tensorflow as tf
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional, Union, Callable
import os
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

# Configure logging
logger = logging.getLogger(__name__)

class TemperatureScaling:
    """
    Temperature scaling for calibrating model predictions
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize temperature scaling
        
        Args:
            temperature: Initial temperature
        """
        self.temperature = temperature
        
        logger.info(f"Initialized temperature scaling with temperature={temperature}")
    
    def fit(self, 
           logits: np.ndarray, 
           y_true: np.ndarray,
           temperature_range: Tuple[float, float] = (0.1, 10.0),
           n_steps: int = 100) -> float:
        """
        Find optimal temperature
        
        Args:
            logits: Model logits or probabilities
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
            scaled_probs = self._apply_temperature(logits, temperature)
            
            # Calculate negative log likelihood
            nll = self._negative_log_likelihood(y_true, scaled_probs)
            
            # Update best temperature
            if nll < best_nll:
                best_nll = nll
                best_temperature = temperature
        
        # Set temperature
        self.temperature = best_temperature
        
        logger.info(f"Found optimal temperature: {best_temperature:.4f}")
        
        return best_temperature
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling
        
        Args:
            logits: Model logits or probabilities
            
        Returns:
            Calibrated probabilities
        """
        return self._apply_temperature(logits, self.temperature)
    
    def _apply_temperature(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """
        Apply temperature scaling
        
        Args:
            logits: Model logits or probabilities
            temperature: Temperature parameter
            
        Returns:
            Calibrated probabilities
        """
        # If input is already probabilities, convert to logits
        if np.all((logits >= 0) & (logits <= 1)):
            # Add small epsilon to avoid log(0)
            epsilon = 1e-7
            logits = np.log(logits + epsilon) - np.log(1 - logits + epsilon)
        
        # Apply temperature scaling
        scaled_logits = logits / temperature
        
        # Convert to probabilities
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
        Save temperature scaling parameters
        
        Args:
            save_path: Path to save parameters
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save parameters
        with open(save_path, "w") as f:
            json.dump({"temperature": float(self.temperature)}, f, indent=2)
        
        logger.info(f"Saved temperature scaling parameters to {save_path}")
    
    @classmethod
    def load(cls, load_path: str) -> "TemperatureScaling":
        """
        Load temperature scaling parameters
        
        Args:
            load_path: Path to load parameters from
            
        Returns:
            Temperature scaling instance
        """
        # Load parameters
        with open(load_path, "r") as f:
            params = json.load(f)
        
        # Create instance
        return cls(temperature=params["temperature"])


class IsotonicCalibration:
    """
    Isotonic regression for calibrating model predictions
    """
    
    def __init__(self, out_of_bounds: str = "clip"):
        """
        Initialize isotonic calibration
        
        Args:
            out_of_bounds: Strategy for handling predictions outside training range
        """
        self.calibrator = IsotonicRegression(out_of_bounds=out_of_bounds)
        self.is_fitted = False
        
        logger.info(f"Initialized isotonic calibration with out_of_bounds={out_of_bounds}")
    
    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibration":
        """
        Fit isotonic regression
        
        Args:
            y_pred: Predicted probabilities
            y_true: True labels
            
        Returns:
            Self
        """
        # For binary classification, use the positive class probability
        if y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
        
        # Convert y_true to binary if needed
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        
        # Fit isotonic regression
        self.calibrator.fit(y_pred, y_true)
        self.is_fitted = True
        
        logger.info("Fitted isotonic calibration")
        
        return self
    
    def calibrate(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration
        
        Args:
            y_pred: Predicted probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            logger.warning("Isotonic calibration is not fitted yet")
            return y_pred
        
        # For binary classification, use the positive class probability
        if y_pred.shape[1] == 2:
            y_pred_cal = np.zeros_like(y_pred)
            y_pred_cal[:, 1] = self.calibrator.transform(y_pred[:, 1])
            y_pred_cal[:, 0] = 1 - y_pred_cal[:, 1]
            
            return y_pred_cal
        else:
            # For multi-class, calibrate each class separately
            y_pred_cal = np.zeros_like(y_pred)
            
            for i in range(y_pred.shape[1]):
                y_pred_cal[:, i] = self.calibrator.transform(y_pred[:, i])
            
            # Normalize to ensure sum to 1
            y_pred_cal = y_pred_cal / np.sum(y_pred_cal, axis=1, keepdims=True)
            
            return y_pred_cal
    
    def save(self, save_path: str) -> None:
        """
        Save isotonic calibration parameters
        
        Args:
            save_path: Path to save parameters
        """
        import pickle
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save parameters
        with open(save_path, "wb") as f:
            pickle.dump({"calibrator": self.calibrator, "is_fitted": self.is_fitted}, f)
        
        logger.info(f"Saved isotonic calibration parameters to {save_path}")
    
    @classmethod
    def load(cls, load_path: str) -> "IsotonicCalibration":
        """
        Load isotonic calibration parameters
        
        Args:
            load_path: Path to load parameters from
            
        Returns:
            Isotonic calibration instance
        """
        import pickle
        
        # Load parameters
        with open(load_path, "rb") as f:
            params = pickle.load(f)
        
        # Create instance
        instance = cls()
        instance.calibrator = params["calibrator"]
        instance.is_fitted = params["is_fitted"]
        
        return instance


class CombinedCalibration:
    """
    Combined calibration using temperature scaling and isotonic regression
    """
    
    def __init__(self, 
                 temperature: float = 1.0,
                 use_isotonic: bool = True):
        """
        Initialize combined calibration
        
        Args:
            temperature: Initial temperature
            use_isotonic: Whether to use isotonic regression
        """
        self.temperature_scaling = TemperatureScaling(temperature=temperature)
        self.use_isotonic = use_isotonic
        self.isotonic_calibration = IsotonicCalibration() if use_isotonic else None
        
        logger.info(f"Initialized combined calibration with temperature={temperature}, "
                   f"use_isotonic={use_isotonic}")
    
    def fit(self, 
           y_pred: np.ndarray, 
           y_true: np.ndarray,
           temperature_range: Tuple[float, float] = (0.1, 10.0),
           n_steps: int = 100) -> "CombinedCalibration":
        """
        Fit calibration
        
        Args:
            y_pred: Predicted probabilities
            y_true: True labels
            temperature_range: Range of temperatures to try
            n_steps: Number of temperature steps
            
        Returns:
            Self
        """
        # Fit temperature scaling
        self.temperature_scaling.fit(y_pred, y_true, temperature_range, n_steps)
        
        # Apply temperature scaling
        y_pred_temp = self.temperature_scaling.calibrate(y_pred)
        
        # Fit isotonic regression if enabled
        if self.use_isotonic and self.isotonic_calibration is not None:
            self.isotonic_calibration.fit(y_pred_temp, y_true)
        
        return self
    
    def calibrate(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Apply calibration
        
        Args:
            y_pred: Predicted probabilities
            
        Returns:
            Calibrated probabilities
        """
        # Apply temperature scaling
        y_pred_temp = self.temperature_scaling.calibrate(y_pred)
        
        # Apply isotonic regression if enabled
        if self.use_isotonic and self.isotonic_calibration is not None:
            return self.isotonic_calibration.calibrate(y_pred_temp)
        else:
            return y_pred_temp
    
    def save(self, save_dir: str) -> None:
        """
        Save calibration parameters
        
        Args:
            save_dir: Directory to save parameters
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save temperature scaling parameters
        self.temperature_scaling.save(os.path.join(save_dir, "temperature_scaling.json"))
        
        # Save isotonic calibration parameters if enabled
        if self.use_isotonic and self.isotonic_calibration is not None:
            self.isotonic_calibration.save(os.path.join(save_dir, "isotonic_calibration.pkl"))
        
        # Save configuration
        with open(os.path.join(save_dir, "calibration_config.json"), "w") as f:
            json.dump({"use_isotonic": self.use_isotonic}, f, indent=2)
        
        logger.info(f"Saved calibration parameters to {save_dir}")
    
    @classmethod
    def load(cls, load_dir: str) -> "CombinedCalibration":
        """
        Load calibration parameters
        
        Args:
            load_dir: Directory to load parameters from
            
        Returns:
            Combined calibration instance
        """
        # Load configuration
        with open(os.path.join(load_dir, "calibration_config.json"), "r") as f:
            config = json.load(f)
        
        # Load temperature scaling parameters
        temperature_scaling = TemperatureScaling.load(
            os.path.join(load_dir, "temperature_scaling.json")
        )
        
        # Create instance
        instance = cls(
            temperature=temperature_scaling.temperature,
            use_isotonic=config["use_isotonic"]
        )
        
        # Load isotonic calibration parameters if enabled
        if config["use_isotonic"]:
            isotonic_path = os.path.join(load_dir, "isotonic_calibration.pkl")
            if os.path.exists(isotonic_path):
                instance.isotonic_calibration = IsotonicCalibration.load(isotonic_path)
        
        return instance


def plot_calibration_curve(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         y_pred_calibrated: Optional[np.ndarray] = None,
                         n_bins: int = 10,
                         save_path: Optional[str] = None) -> None:
    """
    Plot calibration curve
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        y_pred_calibrated: Calibrated probabilities
        n_bins: Number of bins
        save_path: Path to save plot
    """
    # Convert to binary if needed
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # For binary classification, use the positive class probability
    if y_pred.shape[1] == 2:
        y_pred = y_pred[:, 1]
    
    if y_pred_calibrated is not None and y_pred_calibrated.shape[1] == 2:
        y_pred_calibrated = y_pred_calibrated[:, 1]
    
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
    
    # Calculate calibration curve for calibrated predictions if provided
    if y_pred_calibrated is not None:
        prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_pred_calibrated, n_bins=n_bins)
    
    # Plot calibration curve
    plt.figure(figsize=(10, 8))
    
    # Plot diagonal (perfect calibration)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    # Plot original predictions
    plt.plot(prob_pred, prob_true, "s-", label="Original")
    
    # Plot calibrated predictions if provided
    if y_pred_calibrated is not None:
        plt.plot(prob_pred_cal, prob_true_cal, "s-", label="Calibrated")
    
    # Add labels and legend
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curve")
    plt.legend()
    plt.grid(True)
    
    # Save plot if path is provided
    if save_path is not None:
        plt.savefig(save_path)
        logger.info(f"Saved calibration curve to {save_path}")
    
    plt.show()


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate random probabilities
    y_pred = np.random.beta(2, 5, size=(n_samples, 1))
    y_pred = np.hstack([1 - y_pred, y_pred])  # Convert to two-class format
    
    # Generate true labels with some noise
    y_true = (y_pred[:, 1] > 0.5).astype(int)
    y_true = np.logical_xor(y_true, np.random.random(n_samples) < 0.2).astype(int)
    
    # Create combined calibration
    calibration = CombinedCalibration(use_isotonic=True)
    
    # Fit calibration
    calibration.fit(y_pred, y_true)
    
    # Apply calibration
    y_pred_cal = calibration.calibrate(y_pred)
    
    # Plot calibration curve
    plot_calibration_curve(y_true, y_pred, y_pred_cal)