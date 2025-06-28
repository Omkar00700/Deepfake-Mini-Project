"""
Mixup and Manifold Mixup implementation for deepfake detection
Implements data augmentation by linearly interpolating samples and labels
"""

import tensorflow as tf
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional, Union, Callable
import random

# Configure logging
logger = logging.getLogger(__name__)

class Mixup:
    """
    Standard mixup implementation
    Linearly interpolates between pairs of samples and labels
    """
    
    def __init__(self, 
                 alpha: float = 0.2,
                 prob: float = 1.0):
        """
        Initialize mixup
        
        Args:
            alpha: Alpha parameter for beta distribution
            prob: Probability of applying mixup
        """
        self.alpha = alpha
        self.prob = prob
        
        logger.info(f"Initialized Mixup with alpha={alpha}, prob={prob}")
    
    def __call__(self, 
                x: np.ndarray, 
                y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mixup to batch
        
        Args:
            x: Input batch
            y: Labels
            
        Returns:
            Mixed batch and labels
        """
        # Apply mixup with probability prob
        if random.random() > self.prob:
            return x, y
        
        batch_size = x.shape[0]
        
        # Sample lambda from beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Ensure lambda is not too small
        lam = max(lam, 1 - lam)
        
        # Generate random indices for mixing
        indices = np.random.permutation(batch_size)
        
        # Mix inputs
        mixed_x = lam * x + (1 - lam) * x[indices]
        
        # Mix labels
        if isinstance(y, np.ndarray) and len(y.shape) == 2:
            # One-hot encoded labels
            mixed_y = lam * y + (1 - lam) * y[indices]
        else:
            # Integer labels - return both labels and mixing coefficient
            mixed_y = (y, y[indices], lam)
        
        return mixed_x, mixed_y


class ManifoldMixup:
    """
    Manifold Mixup implementation
    Linearly interpolates between intermediate feature representations
    """
    
    def __init__(self, 
                 alpha: float = 0.2,
                 prob: float = 1.0,
                 layers: Optional[List[str]] = None):
        """
        Initialize manifold mixup
        
        Args:
            alpha: Alpha parameter for beta distribution
            prob: Probability of applying mixup
            layers: List of layer names to apply mixup to (if None, randomly select)
        """
        self.alpha = alpha
        self.prob = prob
        self.layers = layers
        
        logger.info(f"Initialized ManifoldMixup with alpha={alpha}, prob={prob}")
    
    def get_random_layer(self, model: tf.keras.Model) -> str:
        """
        Get a random layer from the model
        
        Args:
            model: Keras model
            
        Returns:
            Layer name
        """
        if self.layers is not None:
            return random.choice(self.layers)
        
        # Get all eligible layers
        eligible_layers = []
        
        for layer in model.layers:
            # Skip input, output, and non-trainable layers
            if (layer.name == model.layers[0].name or 
                layer.name == model.layers[-1].name or 
                not layer.trainable):
                continue
            
            # Skip layers without weights
            if len(layer.weights) == 0:
                continue
            
            eligible_layers.append(layer.name)
        
        if not eligible_layers:
            # Fallback to standard mixup if no eligible layers
            return model.layers[0].name
        
        return random.choice(eligible_layers)
    
    def create_mixed_model(self, 
                          model: tf.keras.Model, 
                          layer_name: str) -> tf.keras.Model:
        """
        Create a model that applies mixup at the specified layer
        
        Args:
            model: Original model
            layer_name: Name of layer to apply mixup to
            
        Returns:
            Model with mixup applied
        """
        # Get the specified layer
        layer = model.get_layer(layer_name)
        
        # Create a model up to the specified layer
        input_layer = model.input
        output_layer = layer.output
        
        # Create the first part of the model
        first_part = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        
        # Create the second part of the model
        second_part_input = tf.keras.Input(shape=layer.output_shape[1:])
        x = second_part_input
        
        # Find the index of the specified layer
        layer_idx = [i for i, l in enumerate(model.layers) if l.name == layer_name][0]
        
        # Add all layers after the specified layer
        for i in range(layer_idx + 1, len(model.layers)):
            x = model.layers[i](x)
        
        second_part = tf.keras.Model(inputs=second_part_input, outputs=x)
        
        return first_part, second_part
    
    def __call__(self, 
                model: tf.keras.Model,
                x: np.ndarray, 
                y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply manifold mixup to batch
        
        Args:
            model: Keras model
            x: Input batch
            y: Labels
            
        Returns:
            Mixed batch and labels
        """
        # Apply mixup with probability prob
        if random.random() > self.prob:
            return x, y
        
        batch_size = x.shape[0]
        
        # Sample lambda from beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Ensure lambda is not too small
        lam = max(lam, 1 - lam)
        
        # Generate random indices for mixing
        indices = np.random.permutation(batch_size)
        
        # Get a random layer to apply mixup to
        layer_name = self.get_random_layer(model)
        
        # Create models for manifold mixup
        first_part, second_part = self.create_mixed_model(model, layer_name)
        
        # Get intermediate representations
        h = first_part.predict(x)
        
        # Mix intermediate representations
        mixed_h = lam * h + (1 - lam) * h[indices]
        
        # Get outputs from mixed representations
        mixed_x = second_part.predict(mixed_h)
        
        # Mix labels
        if isinstance(y, np.ndarray) and len(y.shape) == 2:
            # One-hot encoded labels
            mixed_y = lam * y + (1 - lam) * y[indices]
        else:
            # Integer labels - return both labels and mixing coefficient
            mixed_y = (y, y[indices], lam)
        
        return mixed_x, mixed_y


class MixupLayer(tf.keras.layers.Layer):
    """
    Mixup layer for TensorFlow models
    Can be integrated directly into the model
    """
    
    def __init__(self, 
                 alpha: float = 0.2,
                 **kwargs):
        """
        Initialize mixup layer
        
        Args:
            alpha: Alpha parameter for beta distribution
        """
        super(MixupLayer, self).__init__(**kwargs)
        self.alpha = alpha
    
    def call(self, inputs, training=None):
        """
        Forward pass
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Mixed tensor
        """
        if not training:
            return inputs
        
        x = inputs
        batch_size = tf.shape(x)[0]
        
        # Sample lambda from beta distribution
        if self.alpha > 0:
            lam = tf.random.uniform(
                shape=[], 
                minval=0, 
                maxval=1, 
                dtype=tf.float32
            )
            lam = tf.maximum(lam, 1 - lam)
        else:
            lam = tf.constant(1.0, dtype=tf.float32)
        
        # Generate random indices for mixing
        indices = tf.random.shuffle(tf.range(batch_size))
        
        # Mix inputs
        mixed_x = lam * x + (1 - lam) * tf.gather(x, indices)
        
        # Store indices and lambda for loss calculation
        self.add_loss(0.0)  # Dummy loss to ensure the layer is included in the model
        
        # Store mixing parameters as non-trainable weights
        if not hasattr(self, 'lam'):
            self.lam = tf.Variable(lam, trainable=False, name='lam')
            self.indices = tf.Variable(indices, trainable=False, name='indices')
        else:
            self.lam.assign(lam)
            self.indices.assign(indices)
        
        return mixed_x
    
    def get_config(self):
        config = super(MixupLayer, self).get_config()
        config.update({'alpha': self.alpha})
        return config


class MixupLoss(tf.keras.losses.Loss):
    """
    Mixup loss function
    Applies mixup to labels during loss calculation
    """
    
    def __init__(self, 
                 base_loss: tf.keras.losses.Loss,
                 mixup_layer: MixupLayer,
                 **kwargs):
        """
        Initialize mixup loss
        
        Args:
            base_loss: Base loss function
            mixup_layer: Mixup layer
        """
        super(MixupLoss, self).__init__(**kwargs)
        self.base_loss = base_loss
        self.mixup_layer = mixup_layer
    
    def call(self, y_true, y_pred):
        """
        Calculate loss
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Loss value
        """
        # Get mixing parameters
        lam = self.mixup_layer.lam
        indices = self.mixup_layer.indices
        
        # Mix labels
        y_true_mixed = tf.gather(y_true, indices)
        
        # Calculate mixed loss
        loss1 = self.base_loss(y_true, y_pred)
        loss2 = self.base_loss(y_true_mixed, y_pred)
        
        return lam * loss1 + (1 - lam) * loss2


class MixupCallback(tf.keras.callbacks.Callback):
    """
    Callback for applying mixup during training
    """
    
    def __init__(self, 
                 mixup: Union[Mixup, ManifoldMixup],
                 apply_to_val: bool = False):
        """
        Initialize mixup callback
        
        Args:
            mixup: Mixup or ManifoldMixup instance
            apply_to_val: Whether to apply mixup to validation data
        """
        super(MixupCallback, self).__init__()
        self.mixup = mixup
        self.apply_to_val = apply_to_val
    
    def on_train_batch_begin(self, batch, logs=None):
        """
        Apply mixup at the beginning of each training batch
        
        Args:
            batch: Batch index
            logs: Training logs
        """
        # Get current batch data
        x, y = self.model._train_counter.get_batch()
        
        # Apply mixup
        if isinstance(self.mixup, ManifoldMixup):
            mixed_x, mixed_y = self.mixup(self.model, x, y)
        else:
            mixed_x, mixed_y = self.mixup(x, y)
        
        # Update batch data
        self.model._train_counter.set_batch((mixed_x, mixed_y))
    
    def on_test_batch_begin(self, batch, logs=None):
        """
        Apply mixup at the beginning of each validation batch
        
        Args:
            batch: Batch index
            logs: Training logs
        """
        if not self.apply_to_val:
            return
        
        # Get current batch data
        x, y = self.model._test_counter.get_batch()
        
        # Apply mixup
        if isinstance(self.mixup, ManifoldMixup):
            mixed_x, mixed_y = self.mixup(self.model, x, y)
        else:
            mixed_x, mixed_y = self.mixup(x, y)
        
        # Update batch data
        self.model._test_counter.set_batch((mixed_x, mixed_y))


def mixup_loss(y_true, y_pred, lam):
    """
    Calculate mixup loss
    
    Args:
        y_true: Tuple of (y1, y2, lam) or tensor of mixed labels
        y_pred: Predicted labels
        lam: Mixing coefficient (if y_true is not a tuple)
        
    Returns:
        Loss value
    """
    if isinstance(y_true, tuple):
        y1, y2, lam = y_true
        loss1 = tf.keras.losses.categorical_crossentropy(y1, y_pred)
        loss2 = tf.keras.losses.categorical_crossentropy(y2, y_pred)
        return lam * loss1 + (1 - lam) * loss2
    else:
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data
    batch_size = 32
    x = np.random.random((batch_size, 224, 224, 3))
    y = np.random.randint(0, 2, size=(batch_size, 1))
    y_onehot = tf.keras.utils.to_categorical(y, num_classes=2)
    
    # Create mixup
    mixup = Mixup(alpha=0.2)
    
    # Apply mixup
    mixed_x, mixed_y = mixup(x, y_onehot)
    
    print(f"Original shape: {x.shape}, {y_onehot.shape}")
    print(f"Mixed shape: {mixed_x.shape}, {mixed_y.shape}")
    
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    # Create manifold mixup
    manifold_mixup = ManifoldMixup(alpha=0.2)
    
    # Apply manifold mixup
    mixed_x, mixed_y = manifold_mixup(model, x, y_onehot)
    
    print(f"Manifold mixed shape: {mixed_x.shape}, {mixed_y.shape}")