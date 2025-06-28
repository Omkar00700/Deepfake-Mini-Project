"""
Fine-tuning Script for DeepDefend
Specializes deepfake detection models for Indian faces
"""

import os
import sys
import logging
import json
import time
import random
import shutil
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetPreparation:
    """
    Prepares datasets for fine-tuning
    """
    
    def __init__(self, dataset_dir: str = "training_data"):
        """
        Initialize the dataset preparation
        
        Args:
            dataset_dir: Directory for prepared datasets
        """
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(dataset_dir, "train")
        self.val_dir = os.path.join(dataset_dir, "val")
        self.test_dir = os.path.join(dataset_dir, "test")
        
        # Create directories if they don't exist
        for directory in [self.dataset_dir, self.train_dir, self.val_dir, self.test_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
        # Create class subdirectories
        for class_dir in ["real", "fake"]:
            for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
                class_path = os.path.join(split_dir, class_dir)
                if not os.path.exists(class_path):
                    os.makedirs(class_path)
        
        logger.info(f"Initialized dataset preparation with directory: {dataset_dir}")
    
    def prepare_from_directories(self, real_dir: str, fake_dir: str, 
                                train_ratio: float = 0.7, val_ratio: float = 0.15,
                                test_ratio: float = 0.15):
        """
        Prepare a dataset from directories of real and fake images
        
        Args:
            real_dir: Directory containing real images
            fake_dir: Directory containing fake images
            train_ratio: Ratio of images to use for training
            val_ratio: Ratio of images to use for validation
            test_ratio: Ratio of images to use for testing
        """
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("Ratios must sum to 1.0")
        
        # Process real images
        self._process_directory(real_dir, "real", train_ratio, val_ratio, test_ratio)
        
        # Process fake images
        self._process_directory(fake_dir, "fake", train_ratio, val_ratio, test_ratio)
        
        # Count images
        train_real = len(os.listdir(os.path.join(self.train_dir, "real")))
        train_fake = len(os.listdir(os.path.join(self.train_dir, "fake")))
        val_real = len(os.listdir(os.path.join(self.val_dir, "real")))
        val_fake = len(os.listdir(os.path.join(self.val_dir, "fake")))
        test_real = len(os.listdir(os.path.join(self.test_dir, "real")))
        test_fake = len(os.listdir(os.path.join(self.test_dir, "fake")))
        
        logger.info(f"Dataset preparation complete:")
        logger.info(f"  Training: {train_real} real, {train_fake} fake")
        logger.info(f"  Validation: {val_real} real, {val_fake} fake")
        logger.info(f"  Testing: {test_real} real, {test_fake} fake")
    
    def _process_directory(self, source_dir: str, class_name: str, 
                          train_ratio: float, val_ratio: float, test_ratio: float):
        """
        Process a directory of images and split into train/val/test
        
        Args:
            source_dir: Source directory containing images
            class_name: Class name ("real" or "fake")
            train_ratio: Ratio of images to use for training
            val_ratio: Ratio of images to use for validation
            test_ratio: Ratio of images to use for testing
        """
        # Get all image files
        image_files = [f for f in os.listdir(source_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Shuffle files
        random.shuffle(image_files)
        
        # Calculate split indices
        n_train = int(len(image_files) * train_ratio)
        n_val = int(len(image_files) * val_ratio)
        
        # Split files
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train+n_val]
        test_files = image_files[n_train+n_val:]
        
        # Copy files to respective directories
        self._copy_files(source_dir, train_files, os.path.join(self.train_dir, class_name))
        self._copy_files(source_dir, val_files, os.path.join(self.val_dir, class_name))
        self._copy_files(source_dir, test_files, os.path.join(self.test_dir, class_name))
    
    def _copy_files(self, source_dir: str, files: List[str], target_dir: str):
        """
        Copy files from source to target directory
        
        Args:
            source_dir: Source directory
            files: List of files to copy
            target_dir: Target directory
        """
        for file in files:
            source_path = os.path.join(source_dir, file)
            target_path = os.path.join(target_dir, file)
            shutil.copy(source_path, target_path)
    
    def create_data_generators(self, batch_size: int = 32, img_size: Tuple[int, int] = (224, 224)):
        """
        Create data generators for training, validation, and testing
        
        Args:
            batch_size: Batch size for training
            img_size: Image size (height, width)
            
        Returns:
            Tuple of (train_generator, val_generator, test_generator)
        """
        # Create data generators with augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Create data generators without augmentation for validation and testing
        val_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        return train_generator, val_generator, test_generator


class ModelFinetuner:
    """
    Fine-tunes deepfake detection models for Indian faces
    """
    
    def __init__(self, model_dir: str = "fine_tuned_models"):
        """
        Initialize the model fine-tuner
        
        Args:
            model_dir: Directory to save fine-tuned models
        """
        self.model_dir = model_dir
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Set up TensorFlow memory growth to avoid OOM errors
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                logger.info(f"Found {len(physical_devices)} GPU(s), enabled memory growth")
            except Exception as e:
                logger.warning(f"Error setting memory growth: {str(e)}")
        
        logger.info(f"Initialized model fine-tuner with model directory: {model_dir}")
    
    def create_model(self, model_type: str = "efficientnet", img_size: Tuple[int, int] = (224, 224)):
        """
        Create a model for fine-tuning
        
        Args:
            model_type: Type of model to create ("efficientnet" or "resnet")
            img_size: Image size (height, width)
            
        Returns:
            Keras model
        """
        input_shape = (*img_size, 3)
        
        if model_type.lower() == "efficientnet":
            # Create EfficientNet model
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
            
            # Add custom top layers
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.3)(x)
            predictions = Dense(1, activation='sigmoid')(x)
            
            model = Model(inputs=base_model.input, outputs=predictions)
            
        elif model_type.lower() == "resnet":
            # Create ResNet model
            base_model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
            
            # Add custom top layers
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.3)(x)
            predictions = Dense(1, activation='sigmoid')(x)
            
            model = Model(inputs=base_model.input, outputs=predictions)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        logger.info(f"Created {model_type} model for fine-tuning")
        
        return model
    
    def fine_tune(self, model, train_generator, val_generator, 
                 epochs: int = 10, fine_tune_epochs: int = 5,
                 model_name: str = "deepfake_detector"):
        """
        Fine-tune a model
        
        Args:
            model: Keras model to fine-tune
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of epochs for initial training
            fine_tune_epochs: Number of epochs for fine-tuning
            model_name: Name for the saved model
            
        Returns:
            Fine-tuned model and training history
        """
        # Create callbacks
        checkpoint_path = os.path.join(self.model_dir, f"{model_name}_checkpoint.h5")
        callbacks = [
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
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
                log_dir=os.path.join(self.model_dir, 'logs', model_name),
                histogram_freq=1
            )
        ]
        
        # Calculate class weights to handle imbalanced data
        class_weights = {
            0: 1.0,  # Real
            1: 1.0   # Fake
        }
        
        # If classes are imbalanced, calculate weights
        if train_generator.class_indices:
            n_real = len(os.listdir(os.path.join(train_generator.directory, 'real')))
            n_fake = len(os.listdir(os.path.join(train_generator.directory, 'fake')))
            total = n_real + n_fake
            
            if total > 0:
                class_weights = {
                    0: total / (2 * n_real) if n_real > 0 else 1.0,
                    1: total / (2 * n_fake) if n_fake > 0 else 1.0
                }
        
        logger.info(f"Starting initial training for {epochs} epochs")
        logger.info(f"Class weights: {class_weights}")
        
        # Initial training with frozen base model
        history1 = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Unfreeze some layers for fine-tuning
        if fine_tune_epochs > 0:
            logger.info(f"Fine-tuning model for {fine_tune_epochs} epochs")
            
            # Unfreeze the last 30% of the base model layers
            base_model = model.layers[0]
            n_layers = len(base_model.layers)
            for layer in base_model.layers[int(n_layers * 0.7):]:
                layer.trainable = True
            
            # Recompile model with a lower learning rate
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            
            # Fine-tune the model
            history2 = model.fit(
                train_generator,
                epochs=fine_tune_epochs,
                validation_data=val_generator,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
            # Combine histories
            history = {}
            for key in history1.history:
                history[key] = history1.history[key] + history2.history[key]
        else:
            history = history1.history
        
        # Save the final model
        final_model_path = os.path.join(self.model_dir, f"{model_name}.h5")
        model.save(final_model_path)
        logger.info(f"Model saved to {final_model_path}")
        
        # Save model architecture and training history
        with open(os.path.join(self.model_dir, f"{model_name}_history.json"), 'w') as f:
            json.dump(history, f)
        
        with open(os.path.join(self.model_dir, f"{model_name}_architecture.json"), 'w') as f:
            json.dump(model.to_json(), f)
        
        return model, history
    
    def evaluate_model(self, model, test_generator):
        """
        Evaluate a model on the test set
        
        Args:
            model: Keras model to evaluate
            test_generator: Test data generator
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating model on test set")
        
        # Evaluate model
        results = model.evaluate(test_generator, verbose=1)
        
        # Get metric names
        metric_names = model.metrics_names
        
        # Create results dictionary
        evaluation = {}
        for name, value in zip(metric_names, results):
            evaluation[name] = float(value)
        
        logger.info(f"Evaluation results: {evaluation}")
        
        # Get predictions
        predictions = model.predict(test_generator, verbose=1)
        
        # Get true labels
        y_true = test_generator.classes
        
        # Calculate predictions
        y_pred = (predictions > 0.5).astype(int).flatten()
        
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate classification report
        report = classification_report(y_true, y_pred, target_names=['real', 'fake'], output_dict=True)
        
        # Add to evaluation
        evaluation['confusion_matrix'] = cm.tolist()
        evaluation['classification_report'] = report
        
        logger.info(f"Confusion matrix:\n{cm}")
        logger.info(f"Classification report:\n{classification_report(y_true, y_pred, target_names=['real', 'fake'])}")
        
        return evaluation
    
    def plot_training_history(self, history, model_name: str = "deepfake_detector"):
        """
        Plot training history
        
        Args:
            history: Training history
            model_name: Name of the model
        """
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot accuracy
        plt.subplot(2, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        
        # Plot loss
        plt.subplot(2, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        
        # Plot AUC
        if 'auc' in history:
            plt.subplot(2, 2, 3)
            plt.plot(history['auc'], label='Training AUC')
            plt.plot(history['val_auc'], label='Validation AUC')
            plt.title('Model AUC')
            plt.ylabel('AUC')
            plt.xlabel('Epoch')
            plt.legend(loc='lower right')
        
        # Plot precision and recall
        if 'precision' in history and 'recall' in history:
            plt.subplot(2, 2, 4)
            plt.plot(history['precision'], label='Training Precision')
            plt.plot(history['val_precision'], label='Validation Precision')
            plt.plot(history['recall'], label='Training Recall')
            plt.plot(history['val_recall'], label='Validation Recall')
            plt.title('Precision and Recall')
            plt.ylabel('Score')
            plt.xlabel('Epoch')
            plt.legend(loc='lower right')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, f"{model_name}_training_history.png"))
        logger.info(f"Training history plot saved to {os.path.join(self.model_dir, f'{model_name}_training_history.png')}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune deepfake detection models for Indian faces")
    parser.add_argument("--prepare", action="store_true", help="Prepare dataset")
    parser.add_argument("--real-dir", type=str, help="Directory containing real images")
    parser.add_argument("--fake-dir", type=str, help="Directory containing fake images")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--model-type", type=str, default="efficientnet", choices=["efficientnet", "resnet"], help="Model type")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for initial training")
    parser.add_argument("--fine-tune-epochs", type=int, default=5, help="Number of epochs for fine-tuning")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--model-name", type=str, default="indian_deepfake_detector", help="Model name")
    
    args = parser.parse_args()
    
    # Initialize components
    dataset_prep = DatasetPreparation()
    model_finetuner = ModelFinetuner()
    
    if args.prepare and args.real_dir and args.fake_dir:
        logger.info(f"Preparing dataset from {args.real_dir} and {args.fake_dir}")
        dataset_prep.prepare_from_directories(args.real_dir, args.fake_dir)
    
    if args.train:
        # Create data generators
        train_generator, val_generator, test_generator = dataset_prep.create_data_generators(
            batch_size=args.batch_size
        )
        
        # Create model
        model = model_finetuner.create_model(model_type=args.model_type)
        
        # Fine-tune model
        model, history = model_finetuner.fine_tune(
            model,
            train_generator,
            val_generator,
            epochs=args.epochs,
            fine_tune_epochs=args.fine_tune_epochs,
            model_name=args.model_name
        )
        
        # Evaluate model
        evaluation = model_finetuner.evaluate_model(model, test_generator)
        
        # Plot training history
        model_finetuner.plot_training_history(history, args.model_name)
        
        # Save evaluation results
        with open(os.path.join(model_finetuner.model_dir, f"{args.model_name}_evaluation.json"), 'w') as f:
            json.dump(evaluation, f, indent=2)
    
    if not (args.prepare or args.train):
        parser.print_help()


if __name__ == "__main__":
    main()