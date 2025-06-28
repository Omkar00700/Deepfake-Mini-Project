"""
Knowledge Distillation module for deepfake detection
Trains a smaller student model using a larger teacher model
"""

import os
import logging
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Lambda
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow_addons as tfa
from typing import Tuple, List, Dict, Any, Optional, Union
import json
from pathlib import Path
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeDistiller:
    """
    Knowledge distillation for training smaller deepfake detection models
    """
    
    def __init__(self, 
                 teacher_model: Model,
                 student_model_name: str = "mobilenetv2",
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 temperature: float = 4.0,
                 alpha: float = 0.1,
                 dropout_rate: float = 0.5):
        """
        Initialize the knowledge distiller
        
        Args:
            teacher_model: Teacher model to distill knowledge from
            student_model_name: Name of the student model architecture
            input_shape: Input shape for the models
            temperature: Temperature for softening the teacher's predictions
            alpha: Weight for the distillation loss
            dropout_rate: Dropout rate for the student model
        """
        self.teacher_model = teacher_model
        self.student_model_name = student_model_name
        self.input_shape = input_shape
        self.temperature = temperature
        self.alpha = alpha
        self.dropout_rate = dropout_rate
        self.student_model = None
        
        logger.info(f"Initialized knowledge distiller with {student_model_name} student model, "
                   f"temperature={temperature}, alpha={alpha}")
    
    def build_student_model(self) -> Model:
        """
        Build the student model
        
        Returns:
            Student model
        """
        if self.student_model_name == "mobilenetv2":
            # MobileNetV2 base model
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            feature_size = 1280
        elif self.student_model_name == "efficientnet_b0":
            # EfficientNetB0 base model
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            feature_size = 1280
        else:
            raise ValueError(f"Unsupported student model name: {self.student_model_name}")
        
        # Create student model
        inputs = Input(shape=self.input_shape)
        x = base_model(inputs)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(self.dropout_rate * 0.5)(x)
        
        # Output layer with temperature scaling
        logits = Dense(1, activation=None)(x)
        outputs = Lambda(lambda x: tf.nn.sigmoid(x))(logits)
        
        # Create model
        self.student_model = Model(inputs=inputs, outputs=outputs, name="student_model")
        
        logger.info(f"Built student model with {self.student_model_name} architecture")
        return self.student_model
    
    def distillation_loss(self, y_true, y_pred):
        """
        Compute the distillation loss
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Distillation loss
        """
        # Split the labels into ground truth and teacher predictions
        y_true_onehot, y_teacher = tf.split(y_true, [1, 1], axis=1)
        
        # Standard cross-entropy loss
        ce_loss = tf.keras.losses.binary_crossentropy(y_true_onehot, y_pred)
        
        # Distillation loss (KL divergence)
        kl_loss = tf.keras.losses.kullback_leibler_divergence(y_teacher, y_pred)
        
        # Combine losses
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * kl_loss
        
        return total_loss
    
    def prepare_distillation_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Prepare a dataset for distillation by adding teacher predictions
        
        Args:
            dataset: Original dataset
            
        Returns:
            Dataset with teacher predictions
        """
        def add_teacher_predictions(x, y):
            # Get teacher predictions
            teacher_pred = self.teacher_model(x, training=False)
            
            # Combine ground truth and teacher predictions
            y_combined = tf.concat([y, teacher_pred], axis=1)
            
            return x, y_combined
        
        # Apply the transformation to the dataset
        distillation_dataset = dataset.map(add_teacher_predictions)
        
        return distillation_dataset
    
    def train(self, 
              train_dataset: tf.data.Dataset,
              validation_dataset: Optional[tf.data.Dataset] = None,
              epochs: int = 50,
              batch_size: int = 32,
              learning_rate: float = 1e-4,
              save_dir: str = "models/distilled",
              early_stopping_patience: int = 10):
        """
        Train the student model with knowledge distillation
        
        Args:
            train_dataset: Training dataset
            validation_dataset: Validation dataset
            epochs: Number of epochs to train
            batch_size: Batch size for training
            learning_rate: Learning rate for training
            save_dir: Directory to save the distilled model
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        # Build the student model if not already built
        if self.student_model is None:
            self.build_student_model()
        
        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare datasets for distillation
        train_distill_dataset = self.prepare_distillation_dataset(train_dataset)
        
        if validation_dataset is not None:
            val_distill_dataset = self.prepare_distillation_dataset(validation_dataset)
        else:
            val_distill_dataset = None
        
        # Compile the student model with the distillation loss
        self.student_model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=self.distillation_loss,
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        # Create callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(save_dir, f"{self.student_model_name}_distilled.h5"),
                save_best_only=True,
                monitor="val_loss" if val_distill_dataset is not None else "loss"
            ),
            EarlyStopping(
                monitor="val_loss" if val_distill_dataset is not None else "loss",
                patience=early_stopping_patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss" if val_distill_dataset is not None else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train the student model
        history = self.student_model.fit(
            train_distill_dataset,
            epochs=epochs,
            validation_data=val_distill_dataset,
            callbacks=callbacks
        )
        
        # Save training history
        history_path = os.path.join(save_dir, f"{self.student_model_name}_distilled_history.json")
        with open(history_path, "w") as f:
            json.dump({k: [float(val) for val in v] for k, v in history.history.items()}, f)
        
        logger.info(f"Trained student model and saved to {save_dir}")
        
        return history
    
    def evaluate(self, dataset: tf.data.Dataset) -> Dict[str, float]:
        """
        Evaluate the student model
        
        Args:
            dataset: Dataset to evaluate on
            
        Returns:
            Evaluation metrics
        """
        # Evaluate the student model
        student_metrics = self.student_model.evaluate(dataset)
        
        # Evaluate the teacher model
        teacher_metrics = self.teacher_model.evaluate(dataset)
        
        # Create metrics dictionary
        metrics = {
            "student_loss": float(student_metrics[0]),
            "student_accuracy": float(student_metrics[1]),
            "student_auc": float(student_metrics[2]),
            "teacher_loss": float(teacher_metrics[0]),
            "teacher_accuracy": float(teacher_metrics[1]),
            "teacher_auc": float(teacher_metrics[2])
        }
        
        # Calculate compression ratio
        student_params = self.student_model.count_params()
        teacher_params = self.teacher_model.count_params()
        compression_ratio = teacher_params / student_params
        
        metrics["student_params"] = student_params
        metrics["teacher_params"] = teacher_params
        metrics["compression_ratio"] = float(compression_ratio)
        
        logger.info(f"Student model has {student_params:,} parameters, "
                   f"teacher model has {teacher_params:,} parameters, "
                   f"compression ratio: {compression_ratio:.2f}x")
        
        return metrics
    
    def benchmark_inference_speed(self, batch_size: int = 1, num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark inference speed of the student and teacher models
        
        Args:
            batch_size: Batch size for inference
            num_iterations: Number of iterations to run
            
        Returns:
            Inference speed metrics
        """
        # Create random input data
        input_data = np.random.random((batch_size, *self.input_shape))
        
        # Warm up
        self.teacher_model.predict(input_data)
        self.student_model.predict(input_data)
        
        # Benchmark teacher model
        teacher_start_time = tf.timestamp()
        for _ in range(num_iterations):
            self.teacher_model.predict(input_data)
        teacher_end_time = tf.timestamp()
        teacher_time = (teacher_end_time - teacher_start_time) / num_iterations
        
        # Benchmark student model
        student_start_time = tf.timestamp()
        for _ in range(num_iterations):
            self.student_model.predict(input_data)
        student_end_time = tf.timestamp()
        student_time = (student_end_time - student_start_time) / num_iterations
        
        # Calculate speedup
        speedup = teacher_time / student_time
        
        # Create metrics dictionary
        metrics = {
            "teacher_inference_time": float(teacher_time),
            "student_inference_time": float(student_time),
            "speedup": float(speedup),
            "batch_size": batch_size,
            "num_iterations": num_iterations
        }
        
        logger.info(f"Teacher inference time: {teacher_time*1000:.2f} ms, "
                   f"student inference time: {student_time*1000:.2f} ms, "
                   f"speedup: {speedup:.2f}x")
        
        return metrics
    
    def save_model(self, save_path: str):
        """
        Save the student model
        
        Args:
            save_path: Path to save the model
        """
        if self.student_model is None:
            raise ValueError("Student model not initialized. Call build_student_model() or train() first.")
        
        self.student_model.save(save_path)
        logger.info(f"Saved student model to {save_path}")
    
    @classmethod
    def load_model(cls, model_path: str) -> Model:
        """
        Load a distilled model
        
        Args:
            model_path: Path to the distilled model
            
        Returns:
            Loaded model
        """
        logger.info(f"Loading distilled model from {model_path}")
        return tf.keras.models.load_model(model_path, compile=False)