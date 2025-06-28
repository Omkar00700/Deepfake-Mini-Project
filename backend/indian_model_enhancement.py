"""
Indian-specific model enhancement for DeepDefend
This module provides specialized model enhancements for Indian faces
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Dropout
from tensorflow.keras.applications import EfficientNetB3, Xception
import logging

# Configure logging
logger = logging.getLogger(__name__)

def create_indian_specialized_model(input_shape=(224, 224, 3), weights='imagenet'):
    """
    Create a specialized model for Indian face deepfake detection
    
    Args:
        input_shape: Input shape for the model
        weights: Pre-trained weights to use
        
    Returns:
        Compiled model specialized for Indian faces
    """
    try:
        # Create base model (EfficientNetB3)
        base_model = EfficientNetB3(
            include_top=False,
            weights=weights,
            input_shape=input_shape
        )
        
        # Freeze early layers
        for layer in base_model.layers[:100]:
            layer.trainable = False
        
        # Create main branch
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # Create skin tone analysis branch
        # This branch focuses on skin tone features which are important for Indian faces
        skin_branch = base_model.get_layer('block5c_add').output
        skin_branch = GlobalAveragePooling2D()(skin_branch)
        skin_branch = Dense(256, activation='relu')(skin_branch)
        skin_branch = Dropout(0.3)(skin_branch)
        
        # Combine branches
        combined = Concatenate()([x, skin_branch])
        combined = Dense(256, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        
        # Output layer
        predictions = Dense(1, activation='sigmoid', name='predictions')(combined)
        
        # Create model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    except Exception as e:
        logger.error(f"Error creating Indian specialized model: {str(e)}")
        raise

def create_ensemble_model(models_dict, input_shape=(224, 224, 3)):
    """
    Create an ensemble model that combines multiple models
    
    Args:
        models_dict: Dictionary of models to ensemble
        input_shape: Input shape for the model
        
    Returns:
        Ensemble model
    """
    try:
        # Create input layer
        input_tensor = Input(shape=input_shape)
        
        # Get predictions from each model
        predictions = []
        for model_name, model in models_dict.items():
            # Clone the model to avoid issues with shared layers
            cloned_model = tf.keras.models.clone_model(model)
            cloned_model.set_weights(model.get_weights())
            
            # Get prediction
            pred = cloned_model(input_tensor)
            predictions.append(pred)
        
        # Average predictions
        if len(predictions) > 1:
            ensemble_pred = tf.keras.layers.Average()(predictions)
        else:
            ensemble_pred = predictions[0]
        
        # Create ensemble model
        ensemble_model = Model(inputs=input_tensor, outputs=ensemble_pred)
        
        # Compile model
        ensemble_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return ensemble_model
    
    except Exception as e:
        logger.error(f"Error creating ensemble model: {str(e)}")
        raise

def fine_tune_on_indian_faces(model, train_data, validation_data, epochs=10, batch_size=32):
    """
    Fine-tune a model on Indian faces
    
    Args:
        model: Model to fine-tune
        train_data: Training data generator
        validation_data: Validation data generator
        epochs: Number of epochs to train
        batch_size: Batch size
        
    Returns:
        Fine-tuned model and training history
    """
    try:
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # Train model
        history = model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        return model, history
    
    except Exception as e:
        logger.error(f"Error fine-tuning model on Indian faces: {str(e)}")
        raise

def create_data_generator(data_dir, target_size=(224, 224), batch_size=32, augmentation=True):
    """
    Create a data generator for training
    
    Args:
        data_dir: Directory containing the data
        target_size: Target size for the images
        batch_size: Batch size
        augmentation: Whether to apply data augmentation
        
    Returns:
        Data generator
    """
    try:
        # Create data generator with augmentation
        if augmentation:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=0.2
            )
        else:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2
            )
        
        # Create train generator
        train_generator = datagen.flow_from_directory(
            data_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='training'
        )
        
        # Create validation generator
        validation_generator = datagen.flow_from_directory(
            data_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='validation'
        )
        
        return train_generator, validation_generator
    
    except Exception as e:
        logger.error(f"Error creating data generator: {str(e)}")
        raise

def save_model(model, model_path):
    """
    Save model to disk
    
    Args:
        model: Model to save
        model_path: Path to save the model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        model.save(model_path)
        
        logger.info(f"Model saved to {model_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return False

def load_and_enhance_model(model_path, enhance_for_indian=True):
    """
    Load a model and enhance it for Indian faces
    
    Args:
        model_path: Path to the model
        enhance_for_indian: Whether to enhance the model for Indian faces
        
    Returns:
        Enhanced model
    """
    try:
        # Load model
        model = load_model(model_path)
        
        if not enhance_for_indian:
            return model
        
        # Get the base model (up to the last convolutional layer)
        base_model = model
        
        # Add Indian-specific enhancements
        # Extract features before the final dense layer
        x = base_model.layers[-2].output
        
        # Add a specialized layer for Indian skin tones
        indian_features = Dense(128, activation='relu', name='indian_features')(x)
        
        # Add dropout for regularization
        indian_features = Dropout(0.3)(indian_features)
        
        # Combine with original features
        combined = Concatenate()([x, indian_features])
        
        # Final prediction layer
        predictions = Dense(1, activation='sigmoid', name='enhanced_predictions')(combined)
        
        # Create enhanced model
        enhanced_model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        enhanced_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return enhanced_model
    
    except Exception as e:
        logger.error(f"Error loading and enhancing model: {str(e)}")
        raise

def test_model_creation():
    """
    Test function to verify model creation works
    """
    try:
        # Create a small test model
        model = create_indian_specialized_model(input_shape=(224, 224, 3))
        
        # Print model summary
        model.summary()
        
        print("Model creation test passed!")
        return True
    
    except Exception as e:
        print(f"Model creation test error: {str(e)}")
        return False

if __name__ == "__main__":
    # Run test
    test_model_creation()