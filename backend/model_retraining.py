
import os
import logging
import json
import time
import threading
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from model_manager import ModelManager
from metrics import performance_metrics
from backend.config import (
    RETRAINING_ENABLED,
    RETRAINING_INTERVAL_HOURS,
    RETRAINING_PERFORMANCE_THRESHOLD,
    FEEDBACK_COLLECTION_ENABLED,
    FEEDBACK_STORAGE_PATH,
    VALIDATION_DATASET_PATH,
    MODEL_SAVE_PATH
)
from inference_core import evaluate_input_quality, get_uncertainty_threshold

# Configure logging
logger = logging.getLogger(__name__)

class ModelRetrainingManager:
    """
    Manages continuous evaluation and retraining of deepfake detection models
    
    Features:
    - Periodic model performance evaluation
    - Automatic retraining when performance drops
    - Collection and storage of user feedback
    - Performance tracking and history
    - Uncertainty calibration and adaptive ensembling
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRetrainingManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.model_manager = ModelManager()
        self.retraining_lock = threading.Lock()
        self.feedback_data: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
        self.current_evaluation: Optional[Dict[str, Any]] = None
        self.last_evaluation_time: Optional[float] = None
        self.last_retraining_time: Optional[float] = None
        
        # New: Track uncertainty calibration
        self.uncertainty_calibration_data: List[Dict[str, Any]] = []
        self.adaptive_ensemble_weights: Dict[str, float] = {}
        
        # Load existing feedback data if available
        self._load_feedback_data()
        
        # Start periodic evaluation if enabled
        if RETRAINING_ENABLED:
            self._start_periodic_evaluation()
            logger.info("Model retraining system initialized with periodic evaluation")
        else:
            logger.info("Model retraining system initialized (periodic evaluation disabled)")
    
    def _load_feedback_data(self) -> None:
        """Load saved feedback data from storage"""
        if not FEEDBACK_COLLECTION_ENABLED:
            return
            
        try:
            feedback_file = os.path.join(FEEDBACK_STORAGE_PATH, "feedback_data.json")
            if os.path.exists(feedback_file):
                with open(feedback_file, "r") as f:
                    self.feedback_data = json.load(f)
                logger.info(f"Loaded {len(self.feedback_data)} feedback records")
        except Exception as e:
            logger.error(f"Error loading feedback data: {str(e)}")
    
    def _save_feedback_data(self) -> None:
        """Save feedback data to storage"""
        if not FEEDBACK_COLLECTION_ENABLED:
            return
            
        try:
            os.makedirs(FEEDBACK_STORAGE_PATH, exist_ok=True)
            feedback_file = os.path.join(FEEDBACK_STORAGE_PATH, "feedback_data.json")
            with open(feedback_file, "w") as f:
                json.dump(self.feedback_data, f)
            logger.debug(f"Saved {len(self.feedback_data)} feedback records")
        except Exception as e:
            logger.error(f"Error saving feedback data: {str(e)}")
    
    def _start_periodic_evaluation(self) -> None:
        """Start a background thread for periodic model evaluation"""
        def evaluation_worker():
            while True:
                try:
                    # Run evaluation
                    evaluation_result = self.evaluate_model_performance()
                    
                    # Update ensemble weights based on performance
                    if evaluation_result:
                        self._update_ensemble_weights(evaluation_result)
                    
                    # Check if retraining is needed
                    if evaluation_result and evaluation_result.get("performance", 1.0) < RETRAINING_PERFORMANCE_THRESHOLD:
                        logger.warning(f"Model performance below threshold ({evaluation_result['performance']:.4f} < {RETRAINING_PERFORMANCE_THRESHOLD}), triggering retraining")
                        self.retrain_model()
                    
                    # Sleep until next evaluation
                    time.sleep(RETRAINING_INTERVAL_HOURS * 3600)
                except Exception as e:
                    logger.error(f"Error in evaluation worker: {str(e)}", exc_info=True)
                    time.sleep(3600)  # Sleep for an hour on error
        
        # Start the evaluation thread
        thread = threading.Thread(target=evaluation_worker, daemon=True)
        thread.start()
        logger.info(f"Periodic model evaluation started (interval: {RETRAINING_INTERVAL_HOURS} hours)")
    
    def _update_ensemble_weights(self, evaluation_result: Dict[str, Any]) -> None:
        """
        Update ensemble weights based on model performance
        
        Args:
            evaluation_result: Dictionary with performance metrics for different models
        """
        try:
            # Extract per-model performance metrics
            model_metrics = evaluation_result.get("model_metrics", {})
            if not model_metrics:
                return
                
            # Calculate weights based on F1 scores
            f1_scores = {}
            for model_name, metrics in model_metrics.items():
                f1_scores[model_name] = metrics.get("f1", 0.5)
            
            # Adjust weights using softmax to enhance differences while keeping all weights positive
            scores = np.array(list(f1_scores.values()))
            adjusted_scores = np.exp(scores * 2) / sum(np.exp(scores * 2))
            
            # Update weights
            for i, model_name in enumerate(f1_scores.keys()):
                self.adaptive_ensemble_weights[model_name] = adjusted_scores[i]
                
            # Update model manager
            self.model_manager.update_model_performance(model_metrics)
            
            logger.info(f"Updated ensemble weights: {self.adaptive_ensemble_weights}")
            
        except Exception as e:
            logger.error(f"Error updating ensemble weights: {str(e)}")
    
    def evaluate_model_performance(self) -> Optional[Dict[str, Any]]:
        """
        Evaluate current model performance on validation dataset
        
        Returns:
            Dictionary with performance metrics or None if evaluation failed
        """
        try:
            with self.retraining_lock:
                logger.info("Starting model performance evaluation")
                self.last_evaluation_time = time.time()
                
                # Get current model
                model = self.model_manager.get_model()
                
                # Check if validation dataset exists
                if not os.path.exists(VALIDATION_DATASET_PATH):
                    logger.error(f"Validation dataset not found at {VALIDATION_DATASET_PATH}")
                    return None
                
                # Load validation dataset
                # This is a simplified implementation - in production, you'd have a more
                # sophisticated dataset loading mechanism
                try:
                    validation_data = np.load(os.path.join(VALIDATION_DATASET_PATH, "images.npy"))
                    validation_labels = np.load(os.path.join(VALIDATION_DATASET_PATH, "labels.npy"))
                except Exception as e:
                    logger.error(f"Error loading validation dataset: {str(e)}")
                    return None
                
                # Process validation data with current model
                predictions = []
                uncertainties = []
                quality_scores = []
                batch_size = 32
                total_samples = len(validation_data)
                
                for i in range(0, total_samples, batch_size):
                    batch = validation_data[i:i+batch_size]
                    batch_preds = []
                    batch_uncertainties = []
                    batch_qualities = []
                    
                    for img in batch:
                        # Get prediction with uncertainty
                        try:
                            pred_result = model.predict(img, include_uncertainty=True)
                            
                            # Handle dict or float result
                            if isinstance(pred_result, dict):
                                # Extract probability and uncertainty
                                pred_value = pred_result.get("probability", 0.5)
                                uncertainty = pred_result.get("uncertainty", 0.1)
                            else:
                                # Simple float result
                                pred_value = float(pred_result)
                                uncertainty = 0.1  # Default uncertainty
                                
                            # Evaluate image quality
                            quality = evaluate_input_quality(img).get("overall", 0.7)
                            
                            batch_preds.append(pred_value)
                            batch_uncertainties.append(uncertainty)
                            batch_qualities.append(quality)
                            
                        except Exception as e:
                            logger.error(f"Error processing validation sample: {str(e)}")
                            # Use default values on error
                            batch_preds.append(0.5)
                            batch_uncertainties.append(1.0)
                            batch_qualities.append(0.5)
                    
                    predictions.extend(batch_preds)
                    uncertainties.extend(batch_uncertainties)
                    quality_scores.extend(batch_qualities)
                
                predictions = np.array(predictions)
                
                # Evaluate each available model separately
                model_metrics = {}
                available_models = self.model_manager.get_available_models()
                
                for model_name in available_models:
                    try:
                        # Switch to this model
                        current_model = self.model_manager.get_model(model_name)
                        
                        # Sample a subset for efficiency
                        sample_indices = np.random.choice(
                            len(validation_data), 
                            min(100, len(validation_data)), 
                            replace=False
                        )
                        
                        sample_preds = []
                        for idx in sample_indices:
                            try:
                                pred_result = current_model.predict(validation_data[idx])
                                if isinstance(pred_result, dict):
                                    sample_preds.append(pred_result.get("probability", 0.5))
                                else:
                                    sample_preds.append(float(pred_result))
                            except:
                                sample_preds.append(0.5)
                        
                        # Convert to binary using threshold
                        from backend.config import DEEPFAKE_THRESHOLD
                        sample_preds = np.array(sample_preds)
                        sample_binary = (sample_preds > DEEPFAKE_THRESHOLD).astype(int)
                        sample_labels = validation_labels[sample_indices]
                        
                        # Calculate metrics
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                        
                        model_metrics[model_name] = {
                            "accuracy": float(accuracy_score(sample_labels, sample_binary)),
                            "precision": float(precision_score(sample_labels, sample_binary)),
                            "recall": float(recall_score(sample_labels, sample_binary)),
                            "f1": float(f1_score(sample_labels, sample_binary))
                        }
                        
                    except Exception as e:
                        logger.error(f"Error evaluating model {model_name}: {str(e)}")
                        model_metrics[model_name] = {"error": str(e)}
                
                # Calculate performance metrics for the default model
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                
                # Convert continuous predictions to binary using the current threshold
                from backend.config import DEEPFAKE_THRESHOLD
                binary_preds = (predictions > DEEPFAKE_THRESHOLD).astype(int)
                
                metrics = {
                    "accuracy": float(accuracy_score(validation_labels, binary_preds)),
                    "precision": float(precision_score(validation_labels, binary_preds)),
                    "recall": float(recall_score(validation_labels, binary_preds)),
                    "f1": float(f1_score(validation_labels, binary_preds)),
                    "auc": float(roc_auc_score(validation_labels, predictions)),
                    "timestamp": time.time(),
                    "model": model.model_name,
                    "samples": total_samples,
                    "model_metrics": model_metrics,
                    "mean_uncertainty": float(np.mean(uncertainties)),
                    "mean_quality": float(np.mean(quality_scores))
                }
                
                # We'll use F1 score as our primary performance metric
                metrics["performance"] = metrics["f1"]
                
                # Evaluate uncertainty calibration
                metrics["uncertainty_calibration"] = self._evaluate_uncertainty_calibration(
                    predictions, uncertainties, validation_labels
                )
                
                # Save evaluation result
                self.current_evaluation = metrics
                self.performance_history.append(metrics)
                
                # Trim history to keep the last 100 evaluations
                if len(self.performance_history) > 100:
                    self.performance_history = self.performance_history[-100:]
                
                logger.info(f"Model evaluation completed: f1={metrics['f1']:.4f}, accuracy={metrics['accuracy']:.4f}")
                return metrics
                
        except Exception as e:
            logger.error(f"Error evaluating model performance: {str(e)}", exc_info=True)
            return None
    
    def _evaluate_uncertainty_calibration(self, predictions, uncertainties, labels):
        """
        Evaluate how well the uncertainty estimates are calibrated
        
        Args:
            predictions: Model predictions (probabilities)
            uncertainties: Uncertainty estimates
            labels: True labels
            
        Returns:
            Dictionary with calibration metrics
        """
        try:
            # Convert to binary predictions
            from backend.config import DEEPFAKE_THRESHOLD
            binary_preds = (predictions > DEEPFAKE_THRESHOLD).astype(int)
            
            # Calculate errors
            errors = (binary_preds != labels).astype(int)
            
            # Group by uncertainty levels
            uncertainty_bins = np.linspace(0, 0.5, 6)  # 5 bins from 0 to 0.5
            bin_indices = np.digitize(uncertainties, uncertainty_bins) - 1
            
            # Calculate error rate per uncertainty bin
            bin_errors = []
            bin_uncertainties = []
            
            for i in range(len(uncertainty_bins) - 1):
                bin_mask = (bin_indices == i)
                if np.sum(bin_mask) > 10:  # Need enough samples
                    bin_error_rate = np.mean(errors[bin_mask])
                    bin_mean_uncertainty = np.mean(uncertainties[bin_mask])
                    bin_errors.append(float(bin_error_rate))
                    bin_uncertainties.append(float(bin_mean_uncertainty))
            
            # Calculate calibration metrics
            if len(bin_errors) > 1:
                # Expected calibration error: average |uncertainty - error_rate|
                calibration_errors = [abs(u - e) for u, e in zip(bin_uncertainties, bin_errors)]
                ece = float(np.mean(calibration_errors))
                
                # Correlation between uncertainty and error
                correlation = float(np.corrcoef(bin_uncertainties, bin_errors)[0, 1])
                
                # Uncertainty predictive power using AUC
                from sklearn.metrics import roc_auc_score
                try:
                    uncertainty_auc = float(roc_auc_score(errors, uncertainties))
                except:
                    uncertainty_auc = 0.5  # Default if calculation fails
            else:
                ece = 0.2  # Default value
                correlation = 0.0
                uncertainty_auc = 0.5
            
            return {
                "ece": ece,
                "correlation": correlation,
                "predictive_auc": uncertainty_auc,
                "bin_errors": bin_errors,
                "bin_uncertainties": bin_uncertainties
            }
            
        except Exception as e:
            logger.error(f"Error calculating uncertainty calibration: {str(e)}")
            return {"error": str(e)}
    
    def retrain_model(self) -> bool:
        """
        Retrain the current model using feedback data and validation dataset
        
        Returns:
            True if retraining was successful, False otherwise
        """
        try:
            with self.retraining_lock:
                logger.info("Starting model retraining process")
                self.last_retraining_time = time.time()
                
                # Get current model
                model = self.model_manager.get_model()
                model_name = model.model_name
                
                # Load training data (validation dataset + feedback data)
                # In a real implementation, you would have a more sophisticated
                # data pipeline for retraining
                
                # TODO: Implement actual model retraining using TensorFlow/Keras
                # This would involve:
                # 1. Loading the model architecture
                # 2. Loading pre-processed training data
                # 3. Fine-tuning the model with new data
                # 4. Saving the updated model
                
                # For this example, we'll simulate retraining with a sleep
                logger.info(f"Retraining model {model_name}...")
                time.sleep(10)  # Simulate training time
                
                # Save retraining metadata
                retraining_info = {
                    "model": model_name,
                    "timestamp": time.time(),
                    "feedback_samples_used": len(self.feedback_data),
                    "training_duration": 10,  # Simulated duration
                    "previous_performance": self.current_evaluation["performance"] if self.current_evaluation else None
                }
                
                # Run evaluation on the "retrained" model
                new_evaluation = self.evaluate_model_performance()
                if new_evaluation:
                    retraining_info["new_performance"] = new_evaluation["performance"]
                    improvement = (
                        new_evaluation["performance"] - self.current_evaluation["performance"]
                        if self.current_evaluation else 0
                    )
                    retraining_info["improvement"] = improvement
                    
                    logger.info(f"Model retraining completed with performance change: {improvement:.4f}")
                else:
                    logger.warning("Could not evaluate retrained model performance")
                
                # Update ensemble weights based on new performance
                self._update_ensemble_weights(new_evaluation or {})
                
                return True
                
        except Exception as e:
            logger.error(f"Error retraining model: {str(e)}", exc_info=True)
            return False
    
    def add_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """
        Add user feedback for a detection result
        
        Args:
            feedback_data: Dictionary containing feedback information
                Should include:
                - detection_id: ID of the detection
                - correct: Boolean indicating if the detection was correct
                - actual_label: The correct label according to the user
                - confidence: Original confidence score
                - image_id or video_id: Identifier for the media
                
        Returns:
            True if feedback was successfully added, False otherwise
        """
        if not FEEDBACK_COLLECTION_ENABLED:
            logger.warning("Feedback collection is disabled")
            return False
            
        try:
            # Validate required fields
            required_fields = ["detection_id", "correct", "actual_label"]
            if not all(field in feedback_data for field in required_fields):
                logger.error(f"Missing required fields in feedback data: {required_fields}")
                return False
            
            # Add timestamp and metadata
            feedback_data["timestamp"] = time.time()
            feedback_data["timestamp_human"] = datetime.datetime.now().isoformat()
            
            # Add feedback to collection
            self.feedback_data.append(feedback_data)
            
            # Save feedback data periodically (every 10 entries)
            if len(self.feedback_data) % 10 == 0:
                self._save_feedback_data()
            
            logger.info(f"Added feedback for detection {feedback_data['detection_id']}")
            
            # If we have enough new feedback, trigger evaluation
            if len(self.feedback_data) % 50 == 0:
                logger.info(f"Collected {len(self.feedback_data)} feedback samples, triggering model evaluation")
                # Run evaluation in a separate thread
                threading.Thread(target=self.evaluate_model_performance).start()
                
            return True
            
        except Exception as e:
            logger.error(f"Error adding feedback: {str(e)}")
            return False
    
    def get_retraining_status(self) -> Dict[str, Any]:
        """
        Get the current status of the retraining system
        
        Returns:
            Dictionary with retraining system status
        """
        return {
            "enabled": RETRAINING_ENABLED,
            "feedback_collection_enabled": FEEDBACK_COLLECTION_ENABLED,
            "feedback_samples_collected": len(self.feedback_data),
            "last_evaluation_time": self.last_evaluation_time,
            "last_retraining_time": self.last_retraining_time,
            "current_evaluation": self.current_evaluation,
            "retraining_threshold": RETRAINING_PERFORMANCE_THRESHOLD,
            "retraining_interval_hours": RETRAINING_INTERVAL_HOURS,
            "adaptive_ensemble_weights": self.adaptive_ensemble_weights,
            "uncertainty_calibration": self.current_evaluation.get("uncertainty_calibration", {}) 
                                     if self.current_evaluation else {}
        }
    
    def get_performance_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of model performance evaluations
        
        Returns:
            List of performance evaluation results
        """
        return self.performance_history
    
    def get_uncertainty_calibration_data(self) -> Dict[str, Any]:
        """
        Get uncertainty calibration data for analysis
        
        Returns:
            Dictionary with uncertainty calibration data
        """
        if not self.current_evaluation or "uncertainty_calibration" not in self.current_evaluation:
            return {"error": "No calibration data available"}
            
        return self.current_evaluation["uncertainty_calibration"]

# Create singleton instance
retraining_manager = ModelRetrainingManager()

def add_detection_feedback(detection_id: str, correct: bool, actual_label: str, confidence: float = None, 
                           uncertainty: float = None, metadata: Dict[str, Any] = None) -> bool:
    """
    Add feedback for a detection result
    
    Args:
        detection_id: ID of the detection
        correct: Whether the detection was correct
        actual_label: The correct label according to the user
        confidence: Original confidence score
        uncertainty: Uncertainty estimate if available
        metadata: Additional metadata about the detection
        
    Returns:
        True if feedback was successfully added, False otherwise
    """
    feedback_data = {
        "detection_id": detection_id,
        "correct": correct,
        "actual_label": actual_label
    }
    
    if confidence is not None:
        feedback_data["confidence"] = confidence
    
    if uncertainty is not None:
        feedback_data["uncertainty"] = uncertainty
        
    if metadata:
        feedback_data["metadata"] = metadata
    
    return retraining_manager.add_feedback(feedback_data)

def get_retraining_status() -> Dict[str, Any]:
    """
    Get the current status of the retraining system
    
    Returns:
        Dictionary with retraining system status
    """
    return retraining_manager.get_retraining_status()

def trigger_model_evaluation() -> Optional[Dict[str, Any]]:
    """
    Manually trigger a model performance evaluation
    
    Returns:
        Evaluation results if successful, None otherwise
    """
    return retraining_manager.evaluate_model_performance()

def trigger_model_retraining() -> bool:
    """
    Manually trigger model retraining
    
    Returns:
        True if retraining was successful, False otherwise
    """
    return retraining_manager.retrain_model()

def get_adaptive_weights() -> Dict[str, float]:
    """
    Get the current adaptive weights for ensemble models
    
    Returns:
        Dictionary mapping model names to weights
    """
    return retraining_manager.adaptive_ensemble_weights

def get_uncertainty_calibration() -> Dict[str, Any]:
    """
    Get uncertainty calibration metrics
    
    Returns:
        Dictionary with uncertainty calibration data
    """
    return retraining_manager.get_uncertainty_calibration_data()

