"""
Advanced Cross-Validation and Ensembling module for deepfake detection
Implements stratified k-fold cross-validation and model ensembling
"""

import os
import logging
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import StratifiedKFold
from typing import Tuple, List, Dict, Any, Optional, Union, Callable
import json
from pathlib import Path
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Configure logging
logger = logging.getLogger(__name__)

class CrossValidator:
    """
    Stratified k-fold cross-validation for deepfake detection models
    """
    
    def __init__(self, 
                 model_builder: Callable[[], Model],
                 n_splits: int = 5,
                 random_state: int = 42):
        """
        Initialize the cross-validator
        
        Args:
            model_builder: Function that builds and returns a model
            n_splits: Number of folds for cross-validation
            random_state: Random state for reproducibility
        """
        self.model_builder = model_builder
        self.n_splits = n_splits
        self.random_state = random_state
        self.models = []
        self.histories = []
        self.fold_metrics = []
        
        logger.info(f"Initialized cross-validator with {n_splits} folds")
    
    def cross_validate(self, 
                      x: np.ndarray,
                      y: np.ndarray,
                      epochs: int = 50,
                      batch_size: int = 32,
                      save_dir: str = "models/cross_validation",
                      callbacks: List[tf.keras.callbacks.Callback] = None) -> Dict[str, Any]:
        """
        Perform stratified k-fold cross-validation
        
        Args:
            x: Input data
            y: Target labels
            epochs: Number of epochs to train each fold
            batch_size: Batch size for training
            save_dir: Directory to save models and results
            callbacks: List of callbacks for training
            
        Returns:
            Cross-validation results
        """
        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize stratified k-fold
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # Initialize lists to store results
        self.models = []
        self.histories = []
        self.fold_metrics = []
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):
            logger.info(f"Training fold {fold+1}/{self.n_splits}")
            
            # Split data
            x_train, x_val = x[train_idx], x[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Build model
            model = self.model_builder()
            
            # Create fold-specific callbacks
            fold_callbacks = []
            if callbacks:
                fold_callbacks.extend(callbacks)
            
            # Add model checkpoint callback
            fold_callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(save_dir, f"model_fold_{fold+1}.h5"),
                    save_best_only=True,
                    monitor="val_loss"
                )
            )
            
            # Train model
            history = model.fit(
                x_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_val, y_val),
                callbacks=fold_callbacks
            )
            
            # Evaluate model
            metrics = model.evaluate(x_val, y_val)
            
            # Make predictions
            y_pred = model.predict(x_val)
            
            # Calculate additional metrics
            auc = roc_auc_score(y_val, y_pred)
            precision, recall, _ = precision_recall_curve(y_val, y_pred)
            ap = average_precision_score(y_val, y_pred)
            
            # Create fold metrics
            fold_metric = {
                "fold": fold + 1,
                "loss": float(metrics[0]),
                "accuracy": float(metrics[1]),
                "auc": float(auc),
                "average_precision": float(ap)
            }
            
            # Save model and history
            self.models.append(model)
            self.histories.append(history.history)
            self.fold_metrics.append(fold_metric)
            
            # Save fold history
            with open(os.path.join(save_dir, f"history_fold_{fold+1}.json"), "w") as f:
                json.dump({k: [float(val) for val in v] for k, v in history.history.items()}, f)
            
            logger.info(f"Fold {fold+1} metrics: {fold_metric}")
        
        # Calculate average metrics
        avg_metrics = {
            "avg_loss": np.mean([m["loss"] for m in self.fold_metrics]),
            "avg_accuracy": np.mean([m["accuracy"] for m in self.fold_metrics]),
            "avg_auc": np.mean([m["auc"] for m in self.fold_metrics]),
            "avg_average_precision": np.mean([m["average_precision"] for m in self.fold_metrics]),
            "std_loss": np.std([m["loss"] for m in self.fold_metrics]),
            "std_accuracy": np.std([m["accuracy"] for m in self.fold_metrics]),
            "std_auc": np.std([m["auc"] for m in self.fold_metrics]),
            "std_average_precision": np.std([m["average_precision"] for m in self.fold_metrics])
        }
        
        # Save cross-validation results
        results = {
            "fold_metrics": self.fold_metrics,
            "avg_metrics": avg_metrics,
            "n_splits": self.n_splits,
            "random_state": self.random_state,
            "timestamp": time.time()
        }
        
        with open(os.path.join(save_dir, "cross_validation_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Plot learning curves
        self._plot_learning_curves(save_dir)
        
        logger.info(f"Cross-validation completed with average metrics: {avg_metrics}")
        
        return results
    
    def _plot_learning_curves(self, save_dir: str):
        """
        Plot learning curves for all folds
        
        Args:
            save_dir: Directory to save plots
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        
        for fold, history in enumerate(self.histories):
            ax1.plot(history["loss"], label=f"Train (Fold {fold+1})")
            ax1.plot(history["val_loss"], label=f"Val (Fold {fold+1})", linestyle="--")
        
        ax1.legend()
        
        # Plot accuracy
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        
        for fold, history in enumerate(self.histories):
            ax2.plot(history["accuracy"], label=f"Train (Fold {fold+1})")
            ax2.plot(history["val_accuracy"], label=f"Val (Fold {fold+1})", linestyle="--")
        
        ax2.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "learning_curves.png"))
        plt.close()
    
    def create_ensemble(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create an ensemble from the trained models
        
        Args:
            save_path: Path to save the ensemble configuration
            
        Returns:
            Ensemble configuration
        """
        if not self.models:
            raise ValueError("No models available. Run cross_validate() first.")
        
        # Create ensemble configuration
        ensemble_config = {
            "n_models": len(self.models),
            "model_paths": [],
            "fold_metrics": self.fold_metrics,
            "timestamp": time.time()
        }
        
        # Save models if save_path is provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            
            for i, model in enumerate(self.models):
                model_path = os.path.join(save_path, f"ensemble_model_{i+1}.h5")
                model.save(model_path)
                ensemble_config["model_paths"].append(model_path)
            
            # Save ensemble configuration
            with open(os.path.join(save_path, "ensemble_config.json"), "w") as f:
                json.dump(ensemble_config, f, indent=2)
            
            logger.info(f"Saved ensemble with {len(self.models)} models to {save_path}")
        
        return ensemble_config


class ModelEnsemble:
    """
    Ensemble of deepfake detection models
    """
    
    def __init__(self, 
                 models: Optional[List[Model]] = None,
                 model_paths: Optional[List[str]] = None,
                 ensemble_type: str = "average"):
        """
        Initialize the model ensemble
        
        Args:
            models: List of models to ensemble
            model_paths: List of paths to load models from
            ensemble_type: Type of ensembling ('average', 'weighted', or 'stacking')
        """
        self.models = []
        self.ensemble_type = ensemble_type
        self.weights = None
        
        # Load models from paths if provided
        if model_paths:
            for path in model_paths:
                model = load_model(path)
                self.models.append(model)
        
        # Use provided models if available
        if models:
            self.models.extend(models)
        
        # Initialize weights for weighted ensemble
        if ensemble_type == "weighted" and self.models:
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        logger.info(f"Initialized {ensemble_type} ensemble with {len(self.models)} models")
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions with the ensemble
        
        Args:
            x: Input data
            
        Returns:
            Ensemble predictions
        """
        if not self.models:
            raise ValueError("No models in the ensemble")
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(x)
            predictions.append(pred)
        
        # Stack predictions
        stacked_preds = np.stack(predictions, axis=1)  # Shape: (n_samples, n_models, n_outputs)
        
        # Combine predictions based on ensemble type
        if self.ensemble_type == "average":
            # Simple averaging
            ensemble_pred = np.mean(stacked_preds, axis=1)
        elif self.ensemble_type == "weighted":
            # Weighted averaging
            ensemble_pred = np.sum(stacked_preds * self.weights, axis=1)
        elif self.ensemble_type == "stacking":
            # For stacking, we would need a meta-model
            # This is a placeholder for now
            ensemble_pred = np.mean(stacked_preds, axis=1)
        else:
            raise ValueError(f"Unsupported ensemble type: {self.ensemble_type}")
        
        return ensemble_pred
    
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the ensemble
        
        Args:
            x: Input data
            y: Target labels
            
        Returns:
            Evaluation metrics
        """
        # Get ensemble predictions
        y_pred = self.predict(x)
        
        # Calculate metrics
        loss = tf.keras.losses.binary_crossentropy(y, y_pred).numpy().mean()
        accuracy = np.mean((y_pred > 0.5) == y)
        auc = roc_auc_score(y, y_pred)
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y, y_pred)
        ap = average_precision_score(y, y_pred)
        
        # Calculate confusion matrix
        y_pred_binary = (y_pred > 0.5).astype(int)
        cm = confusion_matrix(y, y_pred_binary)
        
        # Calculate individual model metrics
        individual_metrics = []
        for i, model in enumerate(self.models):
            model_pred = model.predict(x)
            model_auc = roc_auc_score(y, model_pred)
            model_accuracy = np.mean((model_pred > 0.5) == y)
            
            individual_metrics.append({
                "model_idx": i,
                "auc": float(model_auc),
                "accuracy": float(model_accuracy)
            })
        
        # Create metrics dictionary
        metrics = {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "auc": float(auc),
            "average_precision": float(ap),
            "confusion_matrix": cm.tolist(),
            "individual_metrics": individual_metrics,
            "ensemble_type": self.ensemble_type,
            "n_models": len(self.models)
        }
        
        logger.info(f"Ensemble evaluation: accuracy={accuracy:.4f}, AUC={auc:.4f}")
        
        return metrics
    
    def optimize_weights(self, x: np.ndarray, y: np.ndarray, method: str = "grid_search") -> Dict[str, Any]:
        """
        Optimize the weights for weighted ensemble
        
        Args:
            x: Input data
            y: Target labels
            method: Optimization method ('grid_search' or 'bayesian')
            
        Returns:
            Optimization results
        """
        if self.ensemble_type != "weighted":
            raise ValueError("Weight optimization is only applicable for weighted ensemble")
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(x)
            predictions.append(pred)
        
        # Stack predictions
        stacked_preds = np.stack(predictions, axis=1)  # Shape: (n_samples, n_models, n_outputs)
        
        if method == "grid_search":
            # Simple grid search for weights
            best_auc = 0
            best_weights = None
            
            # Generate weight combinations
            n_models = len(self.models)
            weight_step = 0.1
            
            # For simplicity, we'll just try different weight distributions
            for i in range(10):
                # Generate random weights that sum to 1
                weights = np.random.dirichlet(np.ones(n_models))
                
                # Calculate weighted predictions
                weighted_pred = np.sum(stacked_preds * weights, axis=1)
                
                # Calculate AUC
                auc = roc_auc_score(y, weighted_pred)
                
                if auc > best_auc:
                    best_auc = auc
                    best_weights = weights
            
            # Update weights
            self.weights = best_weights
            
            # Create results dictionary
            results = {
                "method": method,
                "best_auc": float(best_auc),
                "best_weights": best_weights.tolist(),
                "n_models": n_models
            }
        elif method == "bayesian":
            # Placeholder for Bayesian optimization
            # This would require additional libraries like scikit-optimize
            results = {
                "method": method,
                "message": "Bayesian optimization not implemented yet"
            }
        else:
            raise ValueError(f"Unsupported optimization method: {method}")
        
        logger.info(f"Optimized weights with {method}: best_auc={results.get('best_auc', 'N/A')}")
        
        return results
    
    def save(self, save_dir: str):
        """
        Save the ensemble
        
        Args:
            save_dir: Directory to save the ensemble
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save individual models
        model_paths = []
        for i, model in enumerate(self.models):
            model_path = os.path.join(save_dir, f"model_{i+1}.h5")
            model.save(model_path)
            model_paths.append(model_path)
        
        # Save ensemble configuration
        config = {
            "ensemble_type": self.ensemble_type,
            "model_paths": model_paths,
            "weights": self.weights.tolist() if self.weights is not None else None,
            "n_models": len(self.models),
            "timestamp": time.time()
        }
        
        with open(os.path.join(save_dir, "ensemble_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved ensemble to {save_dir}")
    
    @classmethod
    def load(cls, config_path: str) -> "ModelEnsemble":
        """
        Load an ensemble from a configuration file
        
        Args:
            config_path: Path to the ensemble configuration file
            
        Returns:
            Loaded ensemble
        """
        # Load configuration
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Get model paths
        model_paths = config["model_paths"]
        
        # Create ensemble
        ensemble = cls(model_paths=model_paths, ensemble_type=config["ensemble_type"])
        
        # Set weights if available
        if config.get("weights") is not None:
            ensemble.weights = np.array(config["weights"])
        
        logger.info(f"Loaded ensemble with {len(model_paths)} models")
        
        return ensemble