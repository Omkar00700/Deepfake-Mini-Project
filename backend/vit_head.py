"""
Vision Transformer (ViT) head module for deepfake detection
Implements a lightweight ViT head that can be added on top of CNN features
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Dropout, LayerNormalization, MultiHeadAttention, 
    Reshape, GlobalAveragePooling1D, Input, Concatenate, Add
)
from tensorflow.keras.models import Model
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

class ViTHead:
    """
    Vision Transformer head that can be added on top of CNN features
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int],
                 patch_size: int = 2,
                 num_heads: int = 8,
                 transformer_layers: int = 2,
                 mlp_dim: int = 256,
                 dropout_rate: float = 0.1):
        """
        Initialize the ViT head
        
        Args:
            input_shape: Shape of the input feature map (height, width, channels)
            patch_size: Size of patches to extract from the feature map
            num_heads: Number of attention heads
            transformer_layers: Number of transformer layers
            mlp_dim: Dimension of the MLP layer in the transformer
            dropout_rate: Dropout rate
        """
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        
        # Calculate number of patches
        self.num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        self.projection_dim = input_shape[2]  # Use the same dimension as input channels
        
        logger.info(f"Initialized ViT head with {self.num_patches} patches, "
                   f"{num_heads} heads, {transformer_layers} transformer layers")
    
    def _extract_patches(self, feature_map):
        """
        Extract patches from the feature map
        
        Args:
            feature_map: Input feature map
            
        Returns:
            Extracted patches
        """
        batch_size = tf.shape(feature_map)[0]
        
        # Reshape to extract patches
        patches = tf.image.extract_patches(
            images=feature_map,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        
        # Reshape to (batch_size, num_patches, patch_dim)
        patch_dim = self.patch_size * self.patch_size * self.input_shape[2]
        patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dim])
        
        return patches
    
    def _mlp(self, x, hidden_units, dropout_rate):
        """
        MLP block for transformer
        
        Args:
            x: Input tensor
            hidden_units: Number of hidden units
            dropout_rate: Dropout rate
            
        Returns:
            Output tensor
        """
        for units in hidden_units:
            x = Dense(units, activation=tf.nn.gelu)(x)
            x = Dropout(dropout_rate)(x)
        return x
    
    def _add_position_embedding(self, patches):
        """
        Add position embeddings to patches
        
        Args:
            patches: Input patches
            
        Returns:
            Patches with position embeddings
        """
        # Create position embeddings
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        position_embedding = tf.keras.layers.Embedding(
            input_dim=self.num_patches, output_dim=patches.shape[-1]
        )(positions)
        
        # Add position embeddings to patches
        return patches + position_embedding
    
    def _transformer_block(self, x, transformer_layers, heads, dropout_rate, mlp_dim):
        """
        Transformer block
        
        Args:
            x: Input tensor
            transformer_layers: Number of transformer layers
            heads: Number of attention heads
            dropout_rate: Dropout rate
            mlp_dim: Dimension of the MLP layer
            
        Returns:
            Output tensor
        """
        for _ in range(transformer_layers):
            # Layer normalization 1
            x1 = LayerNormalization(epsilon=1e-6)(x)
            
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=heads, key_dim=self.projection_dim // heads
            )(x1, x1)
            
            # Skip connection 1
            x2 = Add()([attention_output, x])
            
            # Layer normalization 2
            x3 = LayerNormalization(epsilon=1e-6)(x2)
            
            # MLP
            x3 = self._mlp(x3, hidden_units=[mlp_dim, self.projection_dim], dropout_rate=dropout_rate)
            
            # Skip connection 2
            x = Add()([x3, x2])
        
        return x
    
    def build_model(self, cnn_model: Optional[Model] = None) -> Model:
        """
        Build the ViT head model
        
        Args:
            cnn_model: Optional CNN model to use as feature extractor
            
        Returns:
            ViT head model
        """
        # Create input layer
        if cnn_model is None:
            inputs = Input(shape=self.input_shape)
            x = inputs
        else:
            inputs = cnn_model.input
            x = cnn_model.output
            
            # Ensure the output has the expected shape
            if len(x.shape) != 4:
                raise ValueError(f"CNN output shape must be 4D (batch_size, height, width, channels), got {x.shape}")
        
        # Extract patches
        patches = self._extract_patches(x)
        
        # Project patches to embedding dimension if needed
        if patches.shape[-1] != self.projection_dim:
            patches = Dense(self.projection_dim)(patches)
        
        # Add position embeddings
        encoded_patches = self._add_position_embedding(patches)
        
        # Apply transformer blocks
        transformer_output = self._transformer_block(
            encoded_patches,
            transformer_layers=self.transformer_layers,
            heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            mlp_dim=self.mlp_dim
        )
        
        # Global average pooling
        representation = LayerNormalization(epsilon=1e-6)(transformer_output)
        representation = GlobalAveragePooling1D()(representation)
        
        # Create the model
        if cnn_model is None:
            model = Model(inputs=inputs, outputs=representation)
        else:
            # Get CNN features using global average pooling
            cnn_features = GlobalAveragePooling1D()(transformer_output)
            
            # Concatenate CNN and ViT features
            features = Concatenate()([cnn_features, representation])
            
            # Final dense layer
            features = Dense(512, activation="relu")(features)
            features = Dropout(self.dropout_rate)(features)
            
            # Output layer
            outputs = Dense(1, activation="sigmoid")(features)
            
            model = Model(inputs=inputs, outputs=outputs)
        
        logger.info(f"Built ViT head model with {self.transformer_layers} transformer layers")
        return model
    
    def add_to_model(self, base_model: Model) -> Model:
        """
        Add ViT head to an existing model
        
        Args:
            base_model: Base model to add ViT head to
            
        Returns:
            Model with ViT head
        """
        # Get the output of the base model
        base_output = base_model.output
        
        # If the base output is not a feature map, we can't add a ViT head
        if len(base_output.shape) != 4:
            raise ValueError(f"Base model output must be a feature map, got shape {base_output.shape}")
        
        # Extract patches
        patches = self._extract_patches(base_output)
        
        # Project patches to embedding dimension if needed
        if patches.shape[-1] != self.projection_dim:
            patches = Dense(self.projection_dim)(patches)
        
        # Add position embeddings
        encoded_patches = self._add_position_embedding(patches)
        
        # Apply transformer blocks
        transformer_output = self._transformer_block(
            encoded_patches,
            transformer_layers=self.transformer_layers,
            heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            mlp_dim=self.mlp_dim
        )
        
        # Global average pooling for transformer output
        vit_features = LayerNormalization(epsilon=1e-6)(transformer_output)
        vit_features = GlobalAveragePooling1D()(vit_features)
        
        # Get CNN features using global average pooling
        cnn_features = GlobalAveragePooling1D()(transformer_output)
        
        # Concatenate CNN and ViT features
        features = Concatenate()([cnn_features, vit_features])
        
        # Final dense layer
        features = Dense(512, activation="relu")(features)
        features = Dropout(self.dropout_rate)(features)
        
        # Output layer
        outputs = Dense(1, activation="sigmoid")(features)
        
        # Create the model
        model = Model(inputs=base_model.input, outputs=outputs)
        
        logger.info(f"Added ViT head to base model")
        return model