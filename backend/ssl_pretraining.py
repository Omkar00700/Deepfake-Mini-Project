"""
Self-Supervised Learning module for pretraining deepfake detection models
Implements SimCLR and MoCo approaches for representation learning
"""

import os
import logging
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Lambda
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3, ResNet50V2
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa
from typing import Tuple, List, Dict, Any, Optional, Union
import albumentations as A
import cv2
import random
import json
import math
from pathlib import Path
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

class SimCLRPretrainer:
    """
    SimCLR pretraining implementation for self-supervised learning
    """
    
    def __init__(self, 
                 model_name: str = "efficientnet_b3", 
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 projection_dim: int = 128,
                 temperature: float = 0.1,
                 batch_size: int = 32,
                 learning_rate: float = 1e-4):
        """
        Initialize the SimCLR pretrainer
        
        Args:
            model_name: Base model architecture to use
            input_shape: Input shape for the model
            projection_dim: Dimension of the projection head
            temperature: Temperature parameter for contrastive loss
            batch_size: Batch size for training
            learning_rate: Learning rate for training
        """
        self.model_name = model_name
        self.input_shape = input_shape
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.base_model = None
        self.encoder = None
        self.projection_head = None
        self.model = None
        
        # Initialize augmentation pipeline for SimCLR
        self.augmentation_1 = A.Compose([
            A.RandomResizedCrop(height=input_shape[0], width=input_shape[1], scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.8, brightness_limit=0.4, contrast_limit=0.4),
            A.HueSaturationValue(p=0.8, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.ToGray(p=0.2),
        ])
        
        self.augmentation_2 = A.Compose([
            A.RandomResizedCrop(height=input_shape[0], width=input_shape[1], scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.8, brightness_limit=0.4, contrast_limit=0.4),
            A.HueSaturationValue(p=0.8, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.ToGray(p=0.2),
        ])
        
        logger.info(f"Initialized SimCLR pretrainer with model: {model_name}")
    
    def _create_base_encoder(self) -> Model:
        """
        Create the base encoder model
        
        Returns:
            Base encoder model
        """
        if self.model_name == "efficientnet_b0":
            base_model = EfficientNetB0(
                weights=None,
                include_top=False,
                input_shape=self.input_shape
            )
            feature_size = 1280
        elif self.model_name == "efficientnet_b3":
            base_model = EfficientNetB3(
                weights=None,
                include_top=False,
                input_shape=self.input_shape
            )
            feature_size = 1536
        elif self.model_name == "resnet50v2":
            base_model = ResNet50V2(
                weights=None,
                include_top=False,
                input_shape=self.input_shape
            )
            feature_size = 2048
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
        inputs = Input(shape=self.input_shape)
        features = base_model(inputs)
        features = GlobalAveragePooling2D()(features)
        
        self.base_model = base_model
        self.encoder = Model(inputs=inputs, outputs=features, name="encoder")
        
        return self.encoder
    
    def _create_projection_head(self, encoder: Model) -> Model:
        """
        Create the projection head for SimCLR
        
        Args:
            encoder: Encoder model
            
        Returns:
            Model with projection head
        """
        inputs = Input(shape=self.input_shape)
        features = encoder(inputs)
        
        # Non-linear projection head
        projection = Dense(2048, activation='relu')(features)
        projection = Dense(self.projection_dim)(projection)
        
        self.projection_head = Model(inputs=inputs, outputs=projection, name="projection_head")
        return self.projection_head
    
    def _contrastive_loss(self, projections):
        """
        Compute the contrastive loss for SimCLR
        
        Args:
            projections: Projected features from two augmented views
            
        Returns:
            Contrastive loss value
        """
        # Get the two projections
        z1, z2 = projections
        
        # Normalize the projections
        z1 = tf.math.l2_normalize(z1, axis=1)
        z2 = tf.math.l2_normalize(z2, axis=1)
        
        # Compute similarity matrix
        batch_size = tf.shape(z1)[0]
        similarity_matrix = tf.matmul(z1, z2, transpose_b=True) / self.temperature
        
        # The positive samples are the diagonal elements
        positive_samples = tf.linalg.diag_part(similarity_matrix)
        
        # Create labels: positives are 1s, negatives are 0s
        labels = tf.eye(batch_size)
        
        # Compute the loss
        loss = tf.keras.losses.categorical_crossentropy(
            y_true=labels,
            y_pred=tf.nn.softmax(similarity_matrix),
            from_logits=False
        )
        
        return tf.reduce_mean(loss)
    
    def build_model(self) -> Model:
        """
        Build the SimCLR model
        
        Returns:
            SimCLR model
        """
        # Create the encoder
        encoder = self._create_base_encoder()
        
        # Create the projection head
        projection_head = self._create_projection_head(encoder)
        
        # Create the SimCLR model with two inputs
        input_1 = Input(shape=self.input_shape, name="input_1")
        input_2 = Input(shape=self.input_shape, name="input_2")
        
        projection_1 = projection_head(input_1)
        projection_2 = projection_head(input_2)
        
        # Create a model with two inputs and two outputs
        self.model = Model(
            inputs=[input_1, input_2],
            outputs=[projection_1, projection_2],
            name="simclr_model"
        )
        
        # Compile the model with the contrastive loss
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=self._contrastive_loss
        )
        
        logger.info(f"Built SimCLR model with {self.model_name} encoder")
        return self.model
    
    def _generate_augmented_pairs(self, images):
        """
        Generate augmented pairs for SimCLR training
        
        Args:
            images: Batch of images
            
        Returns:
            Two batches of augmented images
        """
        augmented_1 = []
        augmented_2 = []
        
        for image in images:
            # Apply two different augmentations to the same image
            aug_1 = self.augmentation_1(image=image.numpy())["image"]
            aug_2 = self.augmentation_2(image=image.numpy())["image"]
            
            # Normalize to [0, 1]
            aug_1 = aug_1.astype(np.float32) / 255.0
            aug_2 = aug_2.astype(np.float32) / 255.0
            
            augmented_1.append(aug_1)
            augmented_2.append(aug_2)
        
        return np.array(augmented_1), np.array(augmented_2)
    
    def train(self, 
              data_dir: str, 
              epochs: int = 100,
              save_dir: str = "models/ssl_pretrained",
              validation_split: float = 0.1,
              early_stopping_patience: int = 10):
        """
        Train the SimCLR model
        
        Args:
            data_dir: Directory containing unlabeled images
            epochs: Number of epochs to train
            save_dir: Directory to save the pretrained model
            validation_split: Fraction of data to use for validation
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        # Build the model if not already built
        if self.model is None:
            self.build_model()
        
        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Load unlabeled images
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(list(Path(data_dir).glob(f"**/{ext}")))
        
        logger.info(f"Found {len(image_files)} images for pretraining")
        
        # Split into training and validation
        random.shuffle(image_files)
        val_size = int(len(image_files) * validation_split)
        train_files = image_files[val_size:]
        val_files = image_files[:val_size]
        
        logger.info(f"Training on {len(train_files)} images, validating on {len(val_files)} images")
        
        # Create data generators
        def load_and_preprocess_image(file_path):
            img = cv2.imread(str(file_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        
        def train_generator():
            while True:
                # Shuffle files for each epoch
                random.shuffle(train_files)
                
                for i in range(0, len(train_files), self.batch_size):
                    batch_files = train_files[i:i+self.batch_size]
                    batch_images = [load_and_preprocess_image(f) for f in batch_files]
                    
                    # Generate augmented pairs
                    aug_1, aug_2 = self._generate_augmented_pairs(batch_images)
                    
                    # SimCLR doesn't use labels, so we yield a dummy label
                    yield [aug_1, aug_2], [np.zeros((len(aug_1), self.projection_dim)), 
                                          np.zeros((len(aug_1), self.projection_dim))]
        
        def val_generator():
            while True:
                for i in range(0, len(val_files), self.batch_size):
                    batch_files = val_files[i:i+self.batch_size]
                    batch_images = [load_and_preprocess_image(f) for f in batch_files]
                    
                    # Generate augmented pairs
                    aug_1, aug_2 = self._generate_augmented_pairs(batch_images)
                    
                    # SimCLR doesn't use labels, so we yield a dummy label
                    yield [aug_1, aug_2], [np.zeros((len(aug_1), self.projection_dim)), 
                                          np.zeros((len(aug_1), self.projection_dim))]
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(save_dir, f"{self.model_name}_simclr_checkpoint.h5"),
                save_best_only=True,
                monitor="val_loss"
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(save_dir, "logs"),
                histogram_freq=1
            )
        ]
        
        # Train the model
        steps_per_epoch = len(train_files) // self.batch_size
        validation_steps = max(1, len(val_files) // self.batch_size)
        
        history = self.model.fit(
            train_generator(),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator(),
            validation_steps=validation_steps,
            callbacks=callbacks
        )
        
        # Save the encoder model (without projection head)
        encoder_save_path = os.path.join(save_dir, f"{self.model_name}_simclr_encoder.h5")
        self.encoder.save(encoder_save_path)
        logger.info(f"Saved pretrained encoder to {encoder_save_path}")
        
        # Save training history
        history_path = os.path.join(save_dir, f"{self.model_name}_simclr_history.json")
        with open(history_path, "w") as f:
            json.dump({k: [float(val) for val in v] for k, v in history.history.items()}, f)
        
        return history
    
    def get_pretrained_encoder(self) -> Model:
        """
        Get the pretrained encoder model
        
        Returns:
            Pretrained encoder model
        """
        if self.encoder is None:
            raise ValueError("Encoder not initialized. Call build_model() or train() first.")
        
        return self.encoder
    
    @classmethod
    def load_pretrained_encoder(cls, model_path: str) -> Model:
        """
        Load a pretrained encoder model
        
        Args:
            model_path: Path to the pretrained encoder model
            
        Returns:
            Pretrained encoder model
        """
        logger.info(f"Loading pretrained encoder from {model_path}")
        return tf.keras.models.load_model(model_path)


class MoCoPretrainer:
    """
    Momentum Contrast (MoCo) pretraining implementation
    """
    
    def __init__(self, 
                 model_name: str = "efficientnet_b3", 
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 projection_dim: int = 128,
                 queue_size: int = 4096,
                 momentum: float = 0.999,
                 temperature: float = 0.07,
                 batch_size: int = 32,
                 learning_rate: float = 1e-4):
        """
        Initialize the MoCo pretrainer
        
        Args:
            model_name: Base model architecture to use
            input_shape: Input shape for the model
            projection_dim: Dimension of the projection head
            queue_size: Size of the queue for negative samples
            momentum: Momentum coefficient for updating the key encoder
            temperature: Temperature parameter for contrastive loss
            batch_size: Batch size for training
            learning_rate: Learning rate for training
        """
        self.model_name = model_name
        self.input_shape = input_shape
        self.projection_dim = projection_dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Initialize models
        self.query_encoder = None
        self.key_encoder = None
        self.model = None
        
        # Initialize queue for negative samples
        self.queue = tf.Variable(
            tf.math.l2_normalize(tf.random.normal([queue_size, projection_dim]), axis=1),
            trainable=False
        )
        self.queue_ptr = tf.Variable(0, trainable=False)
        
        # Initialize augmentation pipeline
        self.augmentation = A.Compose([
            A.RandomResizedCrop(height=input_shape[0], width=input_shape[1], scale=(0.2, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.8, brightness_limit=0.4, contrast_limit=0.4),
            A.HueSaturationValue(p=0.8, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.ToGray(p=0.2),
        ])
        
        logger.info(f"Initialized MoCo pretrainer with model: {model_name}")
    
    def _create_encoder(self) -> Tuple[Model, Model]:
        """
        Create the query and key encoders
        
        Returns:
            Query and key encoder models
        """
        # Create base model
        if self.model_name == "efficientnet_b0":
            base_model = EfficientNetB0(
                weights=None,
                include_top=False,
                input_shape=self.input_shape
            )
            feature_size = 1280
        elif self.model_name == "efficientnet_b3":
            base_model = EfficientNetB3(
                weights=None,
                include_top=False,
                input_shape=self.input_shape
            )
            feature_size = 1536
        elif self.model_name == "resnet50v2":
            base_model = ResNet50V2(
                weights=None,
                include_top=False,
                input_shape=self.input_shape
            )
            feature_size = 2048
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
        # Create query encoder
        query_inputs = Input(shape=self.input_shape)
        query_features = base_model(query_inputs)
        query_features = GlobalAveragePooling2D()(query_features)
        query_projection = Dense(2048, activation='relu')(query_features)
        query_projection = Dense(self.projection_dim)(query_projection)
        query_projection = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(query_projection)
        
        self.query_encoder = Model(inputs=query_inputs, outputs=query_projection, name="query_encoder")
        
        # Create key encoder (same architecture, different weights)
        key_inputs = Input(shape=self.input_shape)
        key_features = base_model(key_inputs)
        key_features = GlobalAveragePooling2D()(key_features)
        key_projection = Dense(2048, activation='relu')(key_features)
        key_projection = Dense(self.projection_dim)(key_projection)
        key_projection = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(key_projection)
        
        self.key_encoder = Model(inputs=key_inputs, outputs=key_projection, name="key_encoder")
        
        # Initialize key encoder with query encoder weights
        for query_layer, key_layer in zip(self.query_encoder.layers, self.key_encoder.layers):
            key_layer.set_weights(query_layer.get_weights())
            # Freeze key encoder weights
            key_layer.trainable = False
        
        return self.query_encoder, self.key_encoder
    
    def _update_key_encoder(self):
        """
        Update the key encoder using momentum update
        """
        for query_layer, key_layer in zip(self.query_encoder.layers, self.key_encoder.layers):
            key_weights = key_layer.get_weights()
            query_weights = query_layer.get_weights()
            
            # Skip layers without weights
            if not key_weights:
                continue
            
            # Momentum update
            new_weights = []
            for key_w, query_w in zip(key_weights, query_weights):
                new_weights.append(self.momentum * key_w + (1 - self.momentum) * query_w)
            
            key_layer.set_weights(new_weights)
    
    def _update_queue(self, keys):
        """
        Update the queue of negative samples
        
        Args:
            keys: Batch of key features to add to the queue
        """
        batch_size = tf.shape(keys)[0]
        
        # Enqueue the current batch
        ptr = self.queue_ptr.read_value()
        
        # Handle queue wrapping
        if ptr + batch_size > self.queue_size:
            # Fill to the end of the queue
            remaining = self.queue_size - ptr
            self.queue[ptr:].assign(keys[:remaining])
            
            # Wrap around to the beginning
            self.queue[:batch_size-remaining].assign(keys[remaining:])
            new_ptr = batch_size - remaining
        else:
            # Normal case
            self.queue[ptr:ptr+batch_size].assign(keys)
            new_ptr = ptr + batch_size
        
        # Update pointer
        self.queue_ptr.assign(new_ptr % self.queue_size)
    
    def _contrastive_loss(self, query, key):
        """
        Compute the InfoNCE contrastive loss for MoCo
        
        Args:
            query: Query features
            key: Key features
            
        Returns:
            Contrastive loss value
        """
        # Compute positive logits
        positive_logits = tf.reduce_sum(query * key, axis=1, keepdims=True) / self.temperature
        
        # Compute negative logits
        negative_logits = tf.matmul(query, self.queue, transpose_b=True) / self.temperature
        
        # Concatenate positive and negative logits
        logits = tf.concat([positive_logits, negative_logits], axis=1)
        
        # Labels are 0 (first column is the positive)
        labels = tf.zeros(tf.shape(query)[0], dtype=tf.int64)
        
        # Compute cross entropy loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true=labels,
            y_pred=logits,
            from_logits=True
        )
        
        return tf.reduce_mean(loss)
    
    def build_model(self) -> Model:
        """
        Build the MoCo model
        
        Returns:
            MoCo model
        """
        # Create the encoders
        query_encoder, key_encoder = self._create_encoder()
        
        # Create the MoCo model
        query_input = Input(shape=self.input_shape, name="query_input")
        key_input = Input(shape=self.input_shape, name="key_input")
        
        query_output = query_encoder(query_input)
        key_output = key_encoder(key_input)
        
        # Create a model with two inputs and two outputs
        self.model = Model(
            inputs=[query_input, key_input],
            outputs=[query_output, key_output],
            name="moco_model"
        )
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=self._contrastive_loss
        )
        
        logger.info(f"Built MoCo model with {self.model_name} encoder")
        return self.model
    
    def _generate_augmented_pairs(self, images):
        """
        Generate augmented pairs for MoCo training
        
        Args:
            images: Batch of images
            
        Returns:
            Two batches of augmented images
        """
        augmented_1 = []
        augmented_2 = []
        
        for image in images:
            # Apply two different augmentations to the same image
            aug_1 = self.augmentation(image=image.numpy())["image"]
            aug_2 = self.augmentation(image=image.numpy())["image"]
            
            # Normalize to [0, 1]
            aug_1 = aug_1.astype(np.float32) / 255.0
            aug_2 = aug_2.astype(np.float32) / 255.0
            
            augmented_1.append(aug_1)
            augmented_2.append(aug_2)
        
        return np.array(augmented_1), np.array(augmented_2)
    
    def train(self, 
              data_dir: str, 
              epochs: int = 100,
              save_dir: str = "models/ssl_pretrained",
              validation_split: float = 0.1,
              early_stopping_patience: int = 10):
        """
        Train the MoCo model
        
        Args:
            data_dir: Directory containing unlabeled images
            epochs: Number of epochs to train
            save_dir: Directory to save the pretrained model
            validation_split: Fraction of data to use for validation
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        # Build the model if not already built
        if self.model is None:
            self.build_model()
        
        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Load unlabeled images
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(list(Path(data_dir).glob(f"**/{ext}")))
        
        logger.info(f"Found {len(image_files)} images for pretraining")
        
        # Split into training and validation
        random.shuffle(image_files)
        val_size = int(len(image_files) * validation_split)
        train_files = image_files[val_size:]
        val_files = image_files[:val_size]
        
        logger.info(f"Training on {len(train_files)} images, validating on {len(val_files)} images")
        
        # Create data generators
        def load_and_preprocess_image(file_path):
            img = cv2.imread(str(file_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        
        # Custom training loop for MoCo
        train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
        train_dataset = train_dataset.shuffle(buffer_size=10000)
        train_dataset = train_dataset.batch(self.batch_size)
        
        val_dataset = tf.data.Dataset.from_tensor_slices(val_files)
        val_dataset = val_dataset.batch(self.batch_size)
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(save_dir, f"{self.model_name}_moco_checkpoint.h5"),
                save_best_only=True,
                monitor="val_loss"
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(save_dir, "logs"),
                histogram_freq=1
            )
        ]
        
        # Initialize history
        history = {"loss": [], "val_loss": []}
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Training
            train_losses = []
            train_progbar = tqdm(train_dataset, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_files in train_progbar:
                batch_images = [load_and_preprocess_image(f.numpy()) for f in batch_files]
                
                # Generate augmented pairs
                query_images, key_images = self._generate_augmented_pairs(batch_images)
                
                # Forward pass
                with tf.GradientTape() as tape:
                    query_features = self.query_encoder(query_images)
                    key_features = self.key_encoder(key_images)
                    
                    # Compute loss
                    loss = self._contrastive_loss(query_features, key_features)
                
                # Backward pass
                gradients = tape.gradient(loss, self.query_encoder.trainable_variables)
                self.model.optimizer.apply_gradients(zip(gradients, self.query_encoder.trainable_variables))
                
                # Update key encoder
                self._update_key_encoder()
                
                # Update queue
                self._update_queue(key_features)
                
                # Update progress bar
                train_losses.append(loss.numpy())
                train_progbar.set_postfix({"loss": np.mean(train_losses)})
            
            # Validation
            val_losses = []
            
            for batch_files in val_dataset:
                batch_images = [load_and_preprocess_image(f.numpy()) for f in batch_files]
                
                # Generate augmented pairs
                query_images, key_images = self._generate_augmented_pairs(batch_images)
                
                # Forward pass
                query_features = self.query_encoder(query_images)
                key_features = self.key_encoder(key_images)
                
                # Compute loss
                loss = self._contrastive_loss(query_features, key_features)
                val_losses.append(loss.numpy())
            
            # Update history
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            
            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save the best model
                self.query_encoder.save(os.path.join(save_dir, f"{self.model_name}_moco_encoder.h5"))
                logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Save training history
        history_path = os.path.join(save_dir, f"{self.model_name}_moco_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f)
        
        return history
    
    def get_pretrained_encoder(self) -> Model:
        """
        Get the pretrained encoder model
        
        Returns:
            Pretrained encoder model
        """
        if self.query_encoder is None:
            raise ValueError("Encoder not initialized. Call build_model() or train() first.")
        
        return self.query_encoder
    
    @classmethod
    def load_pretrained_encoder(cls, model_path: str) -> Model:
        """
        Load a pretrained encoder model
        
        Args:
            model_path: Path to the pretrained encoder model
            
        Returns:
            Pretrained encoder model
        """
        logger.info(f"Loading pretrained encoder from {model_path}")
        return tf.keras.models.load_model(model_path)