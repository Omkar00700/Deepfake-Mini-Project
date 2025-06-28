
"""
Enhanced Model Trainer for DeepDefend
This module provides functionality for training and fine-tuning deepfake detection models
with special focus on Indian faces and improved generalization.
Includes advanced data augmentation, model architecture enhancements, and hyperparameter tuning.
"""

import os
import logging
import time
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Input, Dropout, LSTM, Bidirectional,
    BatchNormalization, Conv2D, MaxPooling2D, Flatten, TimeDistributed, Lambda
)
from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB3, Xception, ResNet50V2, InceptionV3
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, LearningRateScheduler
)
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa
from typing import Tuple, List, Dict, Any, Optional, Union, Callable
from backend.config import MODEL_DIR, VALIDATION_DATASET_PATH, MODEL_SAVE_PATH, MODEL_INPUT_SIZE
from indian_face_utils import IndianFacePreprocessor
from face_detector import FaceDetector
import albumentations as A
import cv2
import random
import json
import math
import optuna
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Configure TensorFlow to use GPU memory efficiently
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        logger.error(f"GPU memory growth setting failed: {e}")

class DeepfakeModelTrainer:
    """
    Enhanced trainer for deepfake detection models with focus on Indian faces
    and generalization to global populations. Includes advanced data augmentation,
    model architecture improvements, and hyperparameter tuning.
    """
    
    def __init__(self, model_name="efficientnet_b3", input_shape=(224, 224, 3), 
                 use_temporal=False, use_ensemble=False, use_hyperparameter_tuning=False):
        """
        Initialize the model trainer
        
        Args:
            model_name: Name of the base model to use
            input_shape: Input shape for the model
            use_temporal: Whether to use temporal features for video analysis
            use_ensemble: Whether to use model ensembling
            use_hyperparameter_tuning: Whether to use hyperparameter tuning
        """
        self.model_name = model_name
        self.input_shape = input_shape
        self.model = None
        self.use_temporal = use_temporal
        self.use_ensemble = use_ensemble
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        self.face_detector = FaceDetector(detector_type="auto")  # Use enhanced face detector
        self.face_preprocessor = IndianFacePreprocessor(self.face_detector)
        self.training_metrics = {}
        self.class_weights = None
        self.hyperparameters = {
            'learning_rate': 1e-4,
            'dropout_rate': 0.5,
            'weight_decay': 1e-5,
            'batch_size': 32,
            'scheduler': 'cosine'  # 'cosine' or 'step'
        }
        self.experiment_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create directories if they don't exist
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(os.path.join(MODEL_SAVE_PATH, 'experiments'), exist_ok=True)
        os.makedirs(os.path.join(MODEL_SAVE_PATH, 'logs'), exist_ok=True)
        
        # Initialize enhanced data augmentation with more transformations
        self.augmentation = A.Compose([
            # Basic augmentations
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.3, contrast_limit=0.3),
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.5),
            
            # Advanced augmentations
            A.GaussNoise(p=0.3, var_limit=(10.0, 50.0)),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.RandomRotate90(p=0.2),
            A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_REPLICATE),
            A.RandomScale(scale_limit=0.2, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.3, border_mode=cv2.BORDER_REPLICATE),
            
            # Color transformations
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
            
            # Specific augmentations for deepfake detection
            A.OneOf([
                A.JpegCompression(quality_lower=70, quality_upper=90, p=0.5),
                A.Downscale(scale_min=0.7, scale_max=0.9, p=0.5),
            ], p=0.3),
            
            # Cutout/CoarseDropout to help model focus on different facial regions
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=2, min_height=8, min_width=8, p=0.3),
        ])
        
        # Separate augmentation for validation (lighter)
        self.val_augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
        ])
        
        logger.info(f"Initialized Enhanced DeepfakeModelTrainer with model: {model_name}, "
                   f"temporal: {use_temporal}, ensemble: {use_ensemble}, "
                   f"hyperparameter tuning: {use_hyperparameter_tuning}")
    
    def build_model(self, hyperparams=None) -> Model:
        """
        Build or load a model for deepfake detection with enhanced architectures
        
        Args:
            hyperparams: Optional hyperparameters dictionary for model configuration
        
        Returns:
            Keras Model object
        """
        logger.info(f"Building enhanced model: {self.model_name}")
        
        # Use provided hyperparameters or defaults
        hp = hyperparams or self.hyperparameters
        dropout_rate = hp.get('dropout_rate', 0.5)
        weight_decay = hp.get('weight_decay', 1e-5)
        learning_rate = hp.get('learning_rate', 1e-4)
        
        # Create base model based on selected architecture
        if self.model_name == "efficientnet_b0":
            # EfficientNetB0 base model
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            feature_size = 1280
            
        elif self.model_name == "efficientnet_b3":
            # EfficientNetB3 base model (better performance)
            base_model = EfficientNetB3(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            feature_size = 1536
            
        elif self.model_name == "xception":
            # Xception base model
            base_model = Xception(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            feature_size = 2048
            
        elif self.model_name == "resnet50v2":
            # ResNet50V2 base model
            base_model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            feature_size = 2048
            
        elif self.model_name == "inception_v3":
            # InceptionV3 base model
            base_model = InceptionV3(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            feature_size = 2048
            
        elif self.model_name == "mesonet":
            # MesoNet - custom architecture specifically for deepfake detection
            inputs = Input(shape=self.input_shape)
            
            # First block
            x = Conv2D(8, (3, 3), padding='same', activation='relu')(inputs)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
            
            # Second block
            x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
            
            # Third block
            x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
            
            # Fourth block
            x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
            
            # Classification layers
            x = Flatten()(x)
            x = Dropout(dropout_rate)(x)
            x = Dense(128, activation='relu', kernel_regularizer=l2(weight_decay))(x)
            x = Dropout(dropout_rate * 0.5)(x)
            outputs = Dense(1, activation='sigmoid')(x)
            
            # Assemble final model
            model = Model(inputs=inputs, outputs=outputs)
            
            # Compile the model
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    tf.keras.metrics.AUC(),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tfa.metrics.F1Score(num_classes=1, threshold=0.5)
                ]
            )
            
            self.model = model
            logger.info(f"MesoNet model built successfully")
            return model
        
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
        
        # For CNN-based models, create the classification head
        if self.use_temporal:
            # For temporal models, we'll return the base model without top layers
            # to be used in build_temporal_model
            return base_model
        
        # Standard CNN model with improved classification head
        # Extract features from base model
        inputs = Input(shape=self.input_shape)
        x = base_model(inputs)
        
        # Global pooling
        x = GlobalAveragePooling2D()(x)
        
        # Add batch normalization for better training stability
        x = BatchNormalization()(x)
        
        # First dense layer with dropout and L2 regularization
        x = Dense(
            1024, 
            activation='relu',
            kernel_regularizer=l2(weight_decay)
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Second dense layer (smaller)
        x = Dense(
            512, 
            activation='relu',
            kernel_regularizer=l2(weight_decay)
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate * 0.5)(x)  # Less dropout in later layers
        
        # Temperature scaling layer for better calibration
        # We'll use a Lambda layer to apply temperature scaling during inference
        temperature = hp.get('temperature', 1.0)
        logits = Dense(1, activation=None)(x)  # Linear activation for logits
        
        # Apply temperature scaling
        scaled_logits = Lambda(lambda x: x / temperature)(logits)
        
        # Final sigmoid activation
        outputs = tf.keras.layers.Activation('sigmoid')(scaled_logits)
        
        # Assemble final model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tfa.metrics.F1Score(num_classes=1, threshold=0.5)
            ]
        )
        
        self.model = model
        logger.info(f"Model {self.model_name} built successfully with dropout={dropout_rate}, "
                   f"weight_decay={weight_decay}, learning_rate={learning_rate}")
        
        return model
        
    def build_temporal_model(self, base_model=None, sequence_length=16, hyperparams=None) -> Model:
        """
        Build a temporal model for video deepfake detection using LSTM/Bi-LSTM
        
        Args:
            base_model: Base CNN model to use as feature extractor
            sequence_length: Number of frames to process in each sequence
            hyperparams: Optional hyperparameters dictionary
            
        Returns:
            Keras Model for temporal analysis
        """
        logger.info(f"Building temporal model with base model: {self.model_name}")
        
        # Use provided hyperparameters or defaults
        hp = hyperparams or self.hyperparameters
        dropout_rate = hp.get('dropout_rate', 0.5)
        weight_decay = hp.get('weight_decay', 1e-5)
        learning_rate = hp.get('learning_rate', 1e-4)
        lstm_units = hp.get('lstm_units', 128)
        bidirectional = hp.get('bidirectional', True)
        
        # Get base model if not provided
        if base_model is None:
            base_model = self.build_model(hyperparams)
        
        # Create a feature extraction model from the base model
        # We'll remove the classification head and use only the CNN part
        if isinstance(base_model, tf.keras.Model) and not base_model.layers[-1].name.startswith('global_average_pooling'):
            # Find the global average pooling layer or equivalent
            for i, layer in enumerate(base_model.layers):
                if isinstance(layer, GlobalAveragePooling2D) or 'global_average_pooling' in layer.name:
                    feature_extractor = Model(inputs=base_model.inputs, outputs=base_model.layers[i].output)
                    break
            else:
                # If no pooling layer found, use the base model's output
                feature_extractor = Model(inputs=base_model.inputs, outputs=base_model.outputs)
        else:
            # Base model is already a feature extractor
            feature_extractor = base_model
        
        # Create input for sequences of frames
        frame_input_shape = self.input_shape
        sequence_input = Input(shape=(sequence_length, *frame_input_shape))
        
        # Apply the CNN to each frame in the sequence
        encoded_frames = TimeDistributed(feature_extractor)(sequence_input)
        
        # Add LSTM layers for temporal analysis
        if bidirectional:
            x = Bidirectional(LSTM(lstm_units, return_sequences=True))(encoded_frames)
            x = Bidirectional(LSTM(lstm_units // 2))(x)  # Second layer with fewer units
        else:
            x = LSTM(lstm_units, return_sequences=True)(encoded_frames)
            x = LSTM(lstm_units // 2)(x)  # Second layer with fewer units
        
        # Add classification layers
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate * 0.5)(x)
        
        # Temperature scaling for better calibration
        temperature = hp.get('temperature', 1.0)
        logits = Dense(1, activation=None)(x)
        scaled_logits = Lambda(lambda x: x / temperature)(logits)
        outputs = tf.keras.layers.Activation('sigmoid')(scaled_logits)
        
        # Create and compile the model
        temporal_model = Model(inputs=sequence_input, outputs=outputs)
        
        # Compile with appropriate metrics
        temporal_model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tfa.metrics.F1Score(num_classes=1, threshold=0.5)
            ]
        )
        
        self.model = temporal_model
        logger.info(f"Temporal model built successfully with LSTM units={lstm_units}, "
                   f"bidirectional={bidirectional}")
        
        return temporal_model
    
    def load_model(self, model_path: str) -> Optional[Model]:
        """
        Load a pre-trained model for fine-tuning
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model or None if loading failed
        """
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            return None
    
    def tune_hyperparameters(self, 
                           train_dir: str,
                           validation_dir: str,
                           n_trials: int = 20,
                           epochs_per_trial: int = 5) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using Optuna
        
        Args:
            train_dir: Directory containing training data
            validation_dir: Directory containing validation data
            n_trials: Number of hyperparameter combinations to try
            epochs_per_trial: Number of epochs to train each trial
            
        Returns:
            Dictionary of best hyperparameters
        """
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials")
        
        def objective(trial):
            """Optuna objective function for hyperparameter tuning"""
            # Define hyperparameters to search
            hyperparams = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'scheduler': trial.suggest_categorical('scheduler', ['cosine', 'step']),
                'temperature': trial.suggest_float('temperature', 0.5, 2.0)
            }
            
            if self.use_temporal:
                hyperparams['lstm_units'] = trial.suggest_categorical('lstm_units', [64, 128, 256])
                hyperparams['bidirectional'] = trial.suggest_categorical('bidirectional', [True, False])
            
            # Log hyperparameters
            logger.info(f"Trial {trial.number} hyperparameters: {hyperparams}")
            
            # Build model with these hyperparameters
            if self.use_temporal:
                model = self.build_temporal_model(hyperparams=hyperparams)
            else:
                model = self.build_model(hyperparams=hyperparams)
            
            # Create data generators
            batch_size = hyperparams['batch_size']
            train_gen, val_gen = self.prepare_data_generators(
                train_dir, validation_dir, batch_size=batch_size
            )
            
            # Create callbacks for this trial
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
            
            # Train for a few epochs
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs_per_trial,
                callbacks=callbacks,
                class_weight=self.class_weights,
                verbose=1
            )
            
            # Return best validation accuracy
            best_val_acc = max(history.history['val_accuracy'])
            
            # Log results
            logger.info(f"Trial {trial.number} best validation accuracy: {best_val_acc:.4f}")
            
            return best_val_acc
        
        # Create Optuna study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get best hyperparameters
        best_hyperparams = study.best_params
        best_accuracy = study.best_value
        
        # Log best hyperparameters
        logger.info(f"Best hyperparameters: {best_hyperparams}")
        logger.info(f"Best validation accuracy: {best_accuracy:.4f}")
        
        # Save best hyperparameters
        self.hyperparameters = best_hyperparams
        
        # Save hyperparameter tuning results
        results_path = os.path.join(MODEL_SAVE_PATH, 'experiments', f'hparam_tuning_{self.experiment_id}.json')
        with open(results_path, 'w') as f:
            json.dump({
                'best_hyperparameters': best_hyperparams,
                'best_accuracy': best_accuracy,
                'model_name': self.model_name,
                'n_trials': n_trials,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        return best_hyperparams
    
    def prepare_data_generators(self, 
                              train_dir: str, 
                              validation_dir: str, 
                              batch_size: int = 32) -> Tuple[tf.keras.utils.Sequence, tf.keras.utils.Sequence]:
        """
        Create balanced data generators for training and validation
        
        Args:
            train_dir: Directory containing training data
            validation_dir: Directory containing validation data
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_generator, validation_generator)
        """
        # Custom data generator with augmentation and balanced sampling
        class BalancedDataGenerator(tf.keras.utils.Sequence):
            def __init__(self, 
                        directory, 
                        batch_size, 
                        augment=False, 
                        shuffle=True, 
                        face_preprocessor=None):
                
                self.batch_size = batch_size
                self.shuffle = shuffle
                self.augment = augment
                self.face_preprocessor = face_preprocessor
                
                # Find all images
                self.real_images = []
                self.fake_images = []
                
                # Load real images
                real_dir = os.path.join(directory, 'real')
                if os.path.exists(real_dir):
                    for filename in os.listdir(real_dir):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.real_images.append(os.path.join(real_dir, filename))
                
                # Load fake images
                fake_dir = os.path.join(directory, 'fake')
                if os.path.exists(fake_dir):
                    for filename in os.listdir(fake_dir):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.fake_images.append(os.path.join(fake_dir, filename))
                
                # Calculate class weights
                total_images = len(self.real_images) + len(self.fake_images)
                self.class_weights = {
                    0: total_images / (2 * len(self.real_images)) if len(self.real_images) > 0 else 1.0,
                    1: total_images / (2 * len(self.fake_images)) if len(self.fake_images) > 0 else 1.0
                }
                
                # Calculate number of batches
                self.n_real = len(self.real_images)
                self.n_fake = len(self.fake_images)
                self.n = self.n_real + self.n_fake
                self.indices = np.arange(self.n)
                self.steps = math.ceil(self.n / self.batch_size)
                
                logger.info(f"Data generator created with {self.n_real} real and {self.n_fake} fake images")
                
                # Shuffle initially
                if self.shuffle:
                    np.random.shuffle(self.indices)
            
            def __len__(self):
                return self.steps
            
            def __getitem__(self, idx):
                # Get batch indices
                batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
                
                # Initialize batch arrays
                batch_x = np.zeros((len(batch_indices), *MODEL_INPUT_SIZE, 3), dtype=np.float32)
                batch_y = np.zeros(len(batch_indices), dtype=np.float32)
                
                # Fill batch
                for i, idx in enumerate(batch_indices):
                    # Determine if this is a real or fake image
                    if idx < self.n_real:
                        image_path = self.real_images[idx]
                        label = 0  # Real
                    else:
                        image_path = self.fake_images[idx - self.n_real]
                        label = 1  # Fake
                    
                    # Load and preprocess image
                    try:
                        image = cv2.imread(image_path)
                        if image is None:
                            logger.warning(f"Failed to load image: {image_path}")
                            # Use a blank image as fallback
                            image = np.zeros((*MODEL_INPUT_SIZE, 3), dtype=np.uint8)
                        else:
                            # Convert BGR to RGB
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            
                            # Apply face preprocessing if available
                            if self.face_preprocessor is not None:
                                # Detect faces
                                faces = self.face_preprocessor.detect_faces(image)
                                if faces:
                                    # Use the largest face
                                    largest_face = max(faces, key=lambda face: face[2] * face[3])
                                    face_img = self.face_preprocessor.preprocess_face(
                                        image, largest_face, MODEL_INPUT_SIZE[:2]
                                    )
                                    if face_img is not None:
                                        image = face_img
                                    else:
                                        # Resize if face preprocessing failed
                                        image = cv2.resize(image, MODEL_INPUT_SIZE[:2])
                                else:
                                    # Resize if no faces detected
                                    image = cv2.resize(image, MODEL_INPUT_SIZE[:2])
                            else:
                                # Resize if no face preprocessor
                                image = cv2.resize(image, MODEL_INPUT_SIZE[:2])
                        
                        # Apply augmentation if enabled
                        if self.augment:
                            augmented = self.trainer.augmentation(image=image)
                            image = augmented['image']
                        
                        # Normalize to [0, 1]
                        image = image.astype(np.float32) / 255.0
                        
                        # Add to batch
                        batch_x[i] = image
                        batch_y[i] = label
                    except Exception as e:
                        logger.error(f"Error processing image {image_path}: {str(e)}")
                        # Use a blank image as fallback
                        batch_x[i] = np.zeros((*MODEL_INPUT_SIZE, 3), dtype=np.float32)
                        batch_y[i] = label
                
                return batch_x, batch_y
            
            def on_epoch_end(self):
                if self.shuffle:
                    np.random.shuffle(self.indices)
            
            def get_class_weights(self):
                return self.class_weights
        
        # Create data generators
        self_aug = self  # For accessing self.augmentation in the class
        train_generator = BalancedDataGenerator(
            train_dir, 
            batch_size, 
            augment=True, 
            shuffle=True,
            face_preprocessor=self.face_preprocessor
        )
        validation_generator = BalancedDataGenerator(
            validation_dir, 
            batch_size, 
            augment=False, 
            shuffle=False,
            face_preprocessor=self.face_preprocessor
        )
        
        # Store class weights
        self.class_weights = train_generator.get_class_weights()
        
        return train_generator, validation_generator
    
    def train(self, 
             train_dir: str, 
             validation_dir: str, 
             epochs: int = 20, 
             batch_size: int = 32, 
             fine_tune: bool = False, 
             unfreeze_layers: int = 0) -> Dict[str, Any]:
        """
        Train or fine-tune the model
        
        Args:
            train_dir: Directory containing training data
            validation_dir: Directory containing validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
            fine_tune: Whether to fine-tune a pre-trained model
            unfreeze_layers: Number of layers to unfreeze for fine-tuning
            
        Returns:
            Dictionary with training history and metrics
        """
        # Initialize model if not already done
        if self.model is None:
            self.build_model()
        
        # Set up callbacks
        checkpoint_path = os.path.join(MODEL_SAVE_PATH, f"{self.model_name}_best.h5")
        callbacks = [
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            TensorBoard(
                log_dir=os.path.join(MODEL_SAVE_PATH, 'logs', f"{self.model_name}_{int(time.time())}"),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]
        
        # Prepare data generators
        train_generator, validation_generator = self.prepare_data_generators(
            train_dir, validation_dir, batch_size
        )
        
        # If fine-tuning, freeze/unfreeze layers
        if fine_tune and unfreeze_layers > 0:
            logger.info(f"Fine-tuning with {unfreeze_layers} unfrozen layers")
            
            # Freeze all layers first
            for layer in self.model.layers:
                layer.trainable = False
            
            # Unfreeze the top layers
            for layer in self.model.layers[-unfreeze_layers:]:
                layer.trainable = True
        
        # Train the model
        logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
        start_time = time.time()
        
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Save final model
        final_model_path = os.path.join(MODEL_SAVE_PATH, f"{self.model_name}_final.h5")
        self.model.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        # Evaluate on validation set
        logger.info("Evaluating on validation set")
        evaluation = self.model.evaluate(validation_generator, verbose=1)
        metrics = dict(zip(self.model.metrics_names, evaluation))
        
        # Save training metrics
        self.training_metrics = {
            "model_name": self.model_name,
            "training_time": training_time,
            "epochs": epochs,
            "batch_size": batch_size,
            "fine_tuned": fine_tune,
            "unfrozen_layers": unfreeze_layers,
            "class_weights": self.class_weights,
            "metrics": metrics,
            "history": {k: [float(x) for x in v] for k, v in history.history.items()}
        }
        
        # Save metrics to file
        metrics_path = os.path.join(MODEL_SAVE_PATH, f"{self.model_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Validation metrics: {metrics}")
        
        return self.training_metrics
    
    def test(self, test_dir: str, batch_size: int = 32) -> Dict[str, Any]:
        """
        Test the model on a separate test set
        
        Args:
            test_dir: Directory containing test data
            batch_size: Batch size for testing
            
        Returns:
            Dictionary with test metrics
        """
        if self.model is None:
            logger.error("Model not initialized, cannot test")
            return {}
        
        # Create test generator
        _, test_generator = self.prepare_data_generators(
            test_dir, test_dir, batch_size
        )
        
        # Evaluate on test set
        logger.info("Evaluating on test set")
        evaluation = self.model.evaluate(test_generator, verbose=1)
        metrics = dict(zip(self.model.metrics_names, evaluation))
        
        # Add to training metrics
        self.training_metrics["test_metrics"] = metrics
        
        # Update metrics file
        metrics_path = os.path.join(MODEL_SAVE_PATH, f"{self.model_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        logger.info(f"Test metrics: {metrics}")
        
        return metrics
    
    def save_model_summary(self) -> None:
        """Save model summary to a text file"""
        if self.model is None:
            logger.error("Model not initialized, cannot save summary")
            return
        
        summary_path = os.path.join(MODEL_SAVE_PATH, f"{self.model_name}_summary.txt")
        
        # Capture model summary to string
        from io import StringIO
        import sys
        
        original_stdout = sys.stdout
        string_io = StringIO()
        sys.stdout = string_io
        self.model.summary()
        sys.stdout = original_stdout
        
        # Save to file
        with open(summary_path, 'w') as f:
            f.write(string_io.getvalue())
        
        logger.info(f"Model summary saved to {summary_path}")
