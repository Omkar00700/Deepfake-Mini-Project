"""
Adversarial Training module for deepfake detection
Implements FGSM and PGD attacks for adversarial training
"""

import tensorflow as tf
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Optional, Union, Callable

# Configure logging
logger = logging.getLogger(__name__)

class AdversarialTrainer:
    """
    Adversarial training implementation for deepfake detection models
    """
    
    def __init__(self, 
                 model: tf.keras.Model,
                 attack_type: str = "fgsm",
                 epsilon: float = 0.01,
                 pgd_steps: int = 7,
                 pgd_alpha: float = 0.01,
                 clip_min: float = 0.0,
                 clip_max: float = 1.0):
        """
        Initialize the adversarial trainer
        
        Args:
            model: Model to train adversarially
            attack_type: Type of attack to use ('fgsm' or 'pgd')
            epsilon: Maximum perturbation size
            pgd_steps: Number of steps for PGD attack
            pgd_alpha: Step size for PGD attack
            clip_min: Minimum value for clipping
            clip_max: Maximum value for clipping
        """
        self.model = model
        self.attack_type = attack_type.lower()
        self.epsilon = epsilon
        self.pgd_steps = pgd_steps
        self.pgd_alpha = pgd_alpha
        self.clip_min = clip_min
        self.clip_max = clip_max
        
        # Validate attack type
        if self.attack_type not in ["fgsm", "pgd"]:
            raise ValueError(f"Unsupported attack type: {attack_type}. Must be 'fgsm' or 'pgd'.")
        
        logger.info(f"Initialized adversarial trainer with {attack_type} attack, epsilon={epsilon}")
    
    def fgsm_attack(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Fast Gradient Sign Method (FGSM) attack
        
        Args:
            x: Input images
            y: Target labels
            
        Returns:
            Adversarial examples
        """
        # Cast inputs to float32
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            tape.watch(x)
            predictions = self.model(x, training=False)
            loss = tf.keras.losses.binary_crossentropy(y, predictions)
        
        # Get the gradients of the loss w.r.t to the input image
        gradients = tape.gradient(loss, x)
        
        # Get the sign of the gradients
        signed_gradients = tf.sign(gradients)
        
        # Create adversarial examples
        adv_x = x + self.epsilon * signed_gradients
        
        # Clip to valid range
        adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
        
        return adv_x
    
    def pgd_attack(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Projected Gradient Descent (PGD) attack
        
        Args:
            x: Input images
            y: Target labels
            
        Returns:
            Adversarial examples
        """
        # Cast inputs to float32
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        
        # Initialize adversarial examples with small random noise
        noise = tf.random.uniform(tf.shape(x), -self.epsilon, self.epsilon)
        adv_x = tf.clip_by_value(x + noise, self.clip_min, self.clip_max)
        
        # Iterative attack
        for i in range(self.pgd_steps):
            with tf.GradientTape() as tape:
                tape.watch(adv_x)
                predictions = self.model(adv_x, training=False)
                loss = tf.keras.losses.binary_crossentropy(y, predictions)
            
            # Get the gradients of the loss w.r.t to the input image
            gradients = tape.gradient(loss, adv_x)
            
            # Get the sign of the gradients
            signed_gradients = tf.sign(gradients)
            
            # Update adversarial examples
            adv_x = adv_x + self.pgd_alpha * signed_gradients
            
            # Project back to epsilon ball
            adv_x = tf.clip_by_value(adv_x, x - self.epsilon, x + self.epsilon)
            
            # Clip to valid range
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
        
        return adv_x
    
    def generate_adversarial_examples(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Generate adversarial examples using the specified attack
        
        Args:
            x: Input images
            y: Target labels
            
        Returns:
            Adversarial examples
        """
        if self.attack_type == "fgsm":
            return self.fgsm_attack(x, y)
        elif self.attack_type == "pgd":
            return self.pgd_attack(x, y)
        else:
            raise ValueError(f"Unsupported attack type: {self.attack_type}")
    
    def adversarial_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """
        Compute adversarial loss
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            x: Input images
            
        Returns:
            Adversarial loss
        """
        # Standard loss
        standard_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Generate adversarial examples
        adv_x = self.generate_adversarial_examples(x, y_true)
        
        # Get predictions on adversarial examples
        adv_pred = self.model(adv_x, training=True)
        
        # Adversarial loss
        adv_loss = tf.keras.losses.binary_crossentropy(y_true, adv_pred)
        
        # Combine losses
        total_loss = 0.5 * standard_loss + 0.5 * adv_loss
        
        return total_loss
    
    def train_step(self, x: tf.Tensor, y: tf.Tensor) -> Dict[str, float]:
        """
        Perform one adversarial training step
        
        Args:
            x: Input images
            y: Target labels
            
        Returns:
            Dictionary of metrics
        """
        # Generate adversarial examples
        adv_x = self.generate_adversarial_examples(x, y)
        
        # Combine original and adversarial examples
        combined_x = tf.concat([x, adv_x], axis=0)
        combined_y = tf.concat([y, y], axis=0)
        
        # Train on combined data
        with tf.GradientTape() as tape:
            predictions = self.model(combined_x, training=True)
            loss = tf.keras.losses.binary_crossentropy(combined_y, predictions)
            
            # Add regularization losses
            if self.model.losses:
                loss += tf.math.add_n(self.model.losses)
        
        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Apply gradients
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Compute metrics
        metrics = {
            "loss": loss,
            "accuracy": tf.keras.metrics.binary_accuracy(combined_y, predictions)
        }
        
        return metrics
    
    def fit(self, 
            train_dataset: tf.data.Dataset,
            validation_dataset: Optional[tf.data.Dataset] = None,
            epochs: int = 10,
            callbacks: List[tf.keras.callbacks.Callback] = None) -> Dict[str, List[float]]:
        """
        Train the model with adversarial training
        
        Args:
            train_dataset: Training dataset
            validation_dataset: Validation dataset
            epochs: Number of epochs to train
            callbacks: List of callbacks
            
        Returns:
            Training history
        """
        # Initialize history
        history = {"loss": [], "accuracy": []}
        if validation_dataset is not None:
            history["val_loss"] = []
            history["val_accuracy"] = []
        
        # Initialize callbacks
        if callbacks is None:
            callbacks = []
        
        for callback in callbacks:
            callback.set_model(self.model)
            callback.on_train_begin()
        
        # Training loop
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Initialize metrics
            epoch_metrics = {"loss": [], "accuracy": []}
            
            # Train on batches
            for batch in train_dataset:
                x, y = batch
                batch_metrics = self.train_step(x, y)
                
                # Update metrics
                for metric_name, metric_value in batch_metrics.items():
                    epoch_metrics[metric_name].append(metric_value.numpy())
            
            # Compute epoch metrics
            epoch_loss = np.mean(epoch_metrics["loss"])
            epoch_accuracy = np.mean(epoch_metrics["accuracy"])
            
            history["loss"].append(epoch_loss)
            history["accuracy"].append(epoch_accuracy)
            
            # Validate if validation dataset is provided
            if validation_dataset is not None:
                val_metrics = self.evaluate(validation_dataset)
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])
                
                logger.info(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - accuracy: {epoch_accuracy:.4f} - "
                           f"val_loss: {val_metrics['loss']:.4f} - val_accuracy: {val_metrics['accuracy']:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - accuracy: {epoch_accuracy:.4f}")
            
            # Call callbacks
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs={
                    "loss": epoch_loss,
                    "accuracy": epoch_accuracy,
                    "val_loss": val_metrics["loss"] if validation_dataset is not None else None,
                    "val_accuracy": val_metrics["accuracy"] if validation_dataset is not None else None
                })
        
        # Call callbacks on train end
        for callback in callbacks:
            callback.on_train_end()
        
        return history
    
    def evaluate(self, dataset: tf.data.Dataset) -> Dict[str, float]:
        """
        Evaluate the model on a dataset
        
        Args:
            dataset: Dataset to evaluate on
            
        Returns:
            Dictionary of metrics
        """
        # Initialize metrics
        metrics = {"loss": [], "accuracy": []}
        
        # Evaluate on batches
        for batch in dataset:
            x, y = batch
            
            # Get predictions
            predictions = self.model(x, training=False)
            
            # Compute loss
            loss = tf.keras.losses.binary_crossentropy(y, predictions)
            
            # Compute accuracy
            accuracy = tf.keras.metrics.binary_accuracy(y, predictions)
            
            # Update metrics
            metrics["loss"].append(loss.numpy().mean())
            metrics["accuracy"].append(accuracy.numpy().mean())
        
        # Compute average metrics
        metrics["loss"] = np.mean(metrics["loss"])
        metrics["accuracy"] = np.mean(metrics["accuracy"])
        
        return metrics
    
    def evaluate_robustness(self, dataset: tf.data.Dataset) -> Dict[str, float]:
        """
        Evaluate the model's robustness against adversarial examples
        
        Args:
            dataset: Dataset to evaluate on
            
        Returns:
            Dictionary of metrics
        """
        # Initialize metrics
        metrics = {
            "clean_loss": [], "clean_accuracy": [],
            "adv_loss": [], "adv_accuracy": []
        }
        
        # Evaluate on batches
        for batch in dataset:
            x, y = batch
            
            # Get predictions on clean examples
            clean_pred = self.model(x, training=False)
            
            # Compute clean metrics
            clean_loss = tf.keras.losses.binary_crossentropy(y, clean_pred)
            clean_accuracy = tf.keras.metrics.binary_accuracy(y, clean_pred)
            
            # Generate adversarial examples
            adv_x = self.generate_adversarial_examples(x, y)
            
            # Get predictions on adversarial examples
            adv_pred = self.model(adv_x, training=False)
            
            # Compute adversarial metrics
            adv_loss = tf.keras.losses.binary_crossentropy(y, adv_pred)
            adv_accuracy = tf.keras.metrics.binary_accuracy(y, adv_pred)
            
            # Update metrics
            metrics["clean_loss"].append(clean_loss.numpy().mean())
            metrics["clean_accuracy"].append(clean_accuracy.numpy().mean())
            metrics["adv_loss"].append(adv_loss.numpy().mean())
            metrics["adv_accuracy"].append(adv_accuracy.numpy().mean())
        
        # Compute average metrics
        for key in metrics:
            metrics[key] = np.mean(metrics[key])
        
        # Compute robustness score (ratio of adversarial to clean accuracy)
        metrics["robustness_score"] = metrics["adv_accuracy"] / metrics["clean_accuracy"] if metrics["clean_accuracy"] > 0 else 0
        
        return metrics