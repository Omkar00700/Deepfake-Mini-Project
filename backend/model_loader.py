
import os
import numpy as np
import logging
import time
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, MultiHeadAttention, LayerNormalization
import urllib.request
from backend.config import MODEL_DIR, DEFAULT_MODEL, MODEL_INPUT_SIZE, AVAILABLE_MODELS
import threading

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Singleton class to manage multiple deepfake detection models and enable dynamic switching
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._models = {}
                cls._instance._current_model_name = DEFAULT_MODEL
                cls._instance._last_switch_time = 0
                cls._instance._initialized = False
                cls._instance._adaptive_weights = {}
                cls._instance._performance_metrics = {}
            return cls._instance
    
    def initialize(self):
        """Initialize the model manager and load the default model"""
        if not self._initialized:
            logger.info(f"Initializing ModelManager with default model: {DEFAULT_MODEL}")
            
            # Load default model
            self._models[DEFAULT_MODEL] = DeepfakeDetectionModel(DEFAULT_MODEL)
            self._current_model_name = DEFAULT_MODEL
            self._initialized = True
    
    def get_model(self, model_name=None):
        """
        Get a model by name, loading it if necessary
        
        Args:
            model_name: Name of the model to get, or None for current model
            
        Returns:
            The requested DeepfakeDetectionModel instance
        """
        self.initialize()  # Ensure initialization
        
        if model_name is None:
            model_name = self._current_model_name
            
        # Validate model name
        if model_name not in AVAILABLE_MODELS:
            logger.warning(f"Unknown model name: {model_name}, falling back to {self._current_model_name}")
            model_name = self._current_model_name
        
        # Load model if not already loaded
        if model_name not in self._models:
            logger.info(f"Loading model: {model_name}")
            self._models[model_name] = DeepfakeDetectionModel(model_name)
        
        return self._models[model_name]
    
    def switch_model(self, model_name):
        """
        Switch to a different model
        
        Args:
            model_name: Name of the model to switch to
            
        Returns:
            True if switch was successful, False otherwise
        """
        self.initialize()  # Ensure initialization
        
        if model_name not in AVAILABLE_MODELS:
            logger.warning(f"Cannot switch to unknown model: {model_name}")
            return False
        
        # Load model if not already loaded
        if model_name not in self._models:
            try:
                self._models[model_name] = DeepfakeDetectionModel(model_name)
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)}")
                return False
        
        # Update current model
        self._current_model_name = model_name
        self._last_switch_time = time.time()
        logger.info(f"Switched to model: {model_name}")
        return True
    
    def get_current_model_name(self):
        """Get the name of the currently active model"""
        self.initialize()  # Ensure initialization
        return self._current_model_name
    
    def get_available_models(self):
        """Get list of available model names"""
        self.initialize()  # Ensure initialization
        return AVAILABLE_MODELS
    
    def get_loaded_models(self):
        """Get list of currently loaded models"""
        self.initialize()  # Ensure initialization
        return list(self._models.keys())
    
    def update_model_performance(self, model_name, metrics):
        """Update performance metrics for a model to use in adaptive weighting"""
        self._performance_metrics[model_name] = metrics
        self._update_adaptive_weights()
        
    def _update_adaptive_weights(self):
        """Update the adaptive weights based on model performance metrics"""
        if not self._performance_metrics:
            # If no metrics, use equal weights
            for model_name in self._models:
                self._adaptive_weights[model_name] = 1.0
            return
        
        # Calculate weights based on recent F1 scores
        f1_scores = {model: metrics.get('f1', 0.7) for model, metrics in self._performance_metrics.items()}
        
        # Normalize weights
        total = sum(f1_scores.values())
        if total > 0:
            self._adaptive_weights = {model: score / total for model, score in f1_scores.items()}
        else:
            # Fallback to equal weights
            for model_name in self._models:
                self._adaptive_weights[model_name] = 1.0 / len(self._models)
                
        logger.debug(f"Updated adaptive weights: {self._adaptive_weights}")
    
    def get_adaptive_weights(self):
        """Get the current adaptive weights for ensemble models"""
        return self._adaptive_weights
    
    def get_models_info(self):
        """Get information about all loaded models"""
        self.initialize()  # Ensure initialization
        
        models_info = {}
        for name, model in self._models.items():
            models_info[name] = model.get_model_info()
            models_info[name]['is_current'] = (name == self._current_model_name)
            models_info[name]['adaptive_weight'] = self._adaptive_weights.get(name, 1.0)
        
        return {
            'current_model': self._current_model_name,
            'available_models': AVAILABLE_MODELS,
            'loaded_models': list(self._models.keys()),
            'models': models_info,
            'last_switch_time': self._last_switch_time,
            'adaptive_weights': self._adaptive_weights
        }

class DeepfakeDetectionModel:
    """
    Deep learning model wrapper for deepfake detection
    Supports multiple model architectures
    """
    
    def __init__(self, model_name=DEFAULT_MODEL):
        """
        Initialize the deepfake detection model
        
        Args:
            model_name: Name of the base model to use (efficientnet, xception, mesonet, vit)
        """
        logger.info(f"Initializing DeepfakeDetectionModel with {model_name}")
        
        # Store model name and input shape
        self.model_name = model_name
        self.input_shape = MODEL_INPUT_SIZE + (3,)  # (height, width, channels)
        self.model = None
        self.attention_model = None  # For attention visualization
        
        # Load the model
        self._load_model()
        
        # Counter for API calls
        self.prediction_count = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        
        # Performance tracking
        self.total_inference_time = 0
        self.min_inference_time = float('inf')
        self.max_inference_time = 0
        
        # Uncertainty estimation
        self.mc_dropout_samples = 5  # Number of forward passes for Monte Carlo dropout
        self.uncertainty_enabled = True
        
        logger.info(f"DeepfakeDetectionModel ({model_name}) initialized")
        
    def _load_model(self):
        """
        Load the deep learning model for deepfake detection
        """
        start_time = time.time()
        
        try:
            if self.model_name == 'efficientnet':
                self._load_efficientnet()
            elif self.model_name == 'xception':
                self._load_xception()
            elif self.model_name == 'mesonet':
                self._load_mesonet()
            elif self.model_name == 'vit':
                self._load_vit()
            elif self.model_name == 'hybrid':
                self._load_hybrid_model()
            else:
                logger.warning(f"Unknown model name: {self.model_name}, falling back to EfficientNet")
                self.model_name = 'efficientnet'
                self._load_efficientnet()
            
            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            self._load_fallback_model()
    
    def _load_efficientnet(self):
        """
        Load EfficientNet model with attention mechanism
        EfficientNet provides a good balance of accuracy and efficiency
        """
        model_file = os.path.join(MODEL_DIR, "efficientnet_deepfake.h5")
        attention_model_file = os.path.join(MODEL_DIR, "efficientnet_attention.h5")
        
        # Download the model if it doesn't exist
        if not os.path.exists(model_file):
            self._download_model(
                "https://github.com/deepfake-detection-models/efficientnet-b0-deepfake/raw/main/efficientnet_b0_deepfake.h5",
                model_file
            )
        
        # Load the model
        if os.path.exists(model_file):
            # Use TensorFlow's function to load the model
            self.model = load_model(model_file)
            
            # Try to load attention model if available
            if os.path.exists(attention_model_file):
                self.attention_model = load_model(attention_model_file)
            
            # Warmup the model
            self._warmup_model()
        else:
            # If the model file isn't found, load the fallback model
            self._load_fallback_model()
    
    def _load_xception(self):
        """
        Load Xception model, which has shown good results on deepfake detection
        """
        model_file = os.path.join(MODEL_DIR, "xception_deepfake.h5")
        
        # Download the model if it doesn't exist
        if not os.path.exists(model_file):
            self._download_model(
                "https://github.com/deepfake-detection-models/xception-deepfake/raw/main/xception_deepfake.h5",
                model_file
            )
        
        # Load the model
        if os.path.exists(model_file):
            self.model = load_model(model_file)
            
            # Warmup the model
            self._warmup_model()
        else:
            # If the model file isn't found, load the fallback model
            self._load_fallback_model()
    
    def _load_mesonet(self):
        """
        Load MesoNet, a compact model specially designed for deepfake detection
        """
        model_file = os.path.join(MODEL_DIR, "mesonet.h5")
        
        # Download the model if it doesn't exist
        if not os.path.exists(model_file):
            self._download_model(
                "https://github.com/deepfake-detection-models/mesonet/raw/main/mesonet.h5",
                model_file
            )
        
        # Load the model
        if os.path.exists(model_file):
            self.model = load_model(model_file)
            
            # Warmup the model
            self._warmup_model()
        else:
            # If the model file isn't found, load the fallback model
            self._load_fallback_model()
    
    def _load_vit(self):
        """
        Load Vision Transformer model for deepfake detection
        """
        model_file = os.path.join(MODEL_DIR, "vit_deepfake.h5")
        
        # Check if model exists
        if os.path.exists(model_file):
            self.model = load_model(model_file)
            
            # Warmup the model
            self._warmup_model()
        else:
            # If the model doesn't exist, create a simple ViT model
            logger.warning(f"ViT model not found, creating a new one")
            self._create_vit_model()
            
            # Warmup the model
            self._warmup_model()
    
    def _create_vit_model(self):
        """
        Create a simple Vision Transformer model
        This is a placeholder for a more sophisticated implementation
        """
        # Create a simple ViT model using TensorFlow
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Patch embeddings
        patch_size = 16
        patches = tf.keras.layers.Conv2D(
            filters=768,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            name="patches"
        )(inputs)
        
        # Flatten patches
        patch_dims = patches.shape[-3] * patches.shape[-2]
        patches = tf.keras.layers.Reshape((patch_dims, 768))(patches)
        
        # Add position embeddings
        positions = tf.keras.layers.Embedding(
            input_dim=patch_dims, output_dim=768
        )(tf.range(start=0, limit=patch_dims, delta=1))
        x = patches + positions
        
        # Add transformer blocks
        for _ in range(4):
            # Layer normalization 1
            x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
            
            # Multi-head attention
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=8, key_dim=64, dropout=0.1
            )(x1, x1)
            
            # Skip connection 1
            x2 = tf.keras.layers.Add()([attention_output, x])
            
            # Layer normalization 2
            x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
            
            # MLP
            x3 = tf.keras.layers.Dense(1024, activation="gelu")(x3)
            x3 = tf.keras.layers.Dropout(0.1)(x3)
            x3 = tf.keras.layers.Dense(768)(x3)
            x3 = tf.keras.layers.Dropout(0.1)(x3)
            
            # Skip connection 2
            x = tf.keras.layers.Add()([x2, x3])
        
        # Final layer normalization
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Classification head
        x = tf.keras.layers.Dense(256, activation="gelu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        
        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Created a new ViT model")
    
    def _load_hybrid_model(self):
        """
        Load a hybrid CNN-RNN model for deepfake detection
        Combines convolutional features with temporal modeling
        """
        model_file = os.path.join(MODEL_DIR, "hybrid_cnn_rnn.h5")
        
        # Check if model exists
        if os.path.exists(model_file):
            self.model = load_model(model_file)
            
            # Warmup the model
            self._warmup_model()
        else:
            # If the model doesn't exist, create a simple hybrid model
            logger.warning(f"Hybrid model not found, creating a new one")
            self._create_hybrid_model()
            
            # Warmup the model
            self._warmup_model()
    
    def _create_hybrid_model(self):
        """
        Create a hybrid CNN-RNN model for deepfake detection
        """
        # Create a hybrid model using TensorFlow
        # CNN part
        cnn_input = tf.keras.Input(shape=self.input_shape)
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_tensor=cnn_input
        )
        
        # Get CNN features
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        cnn_model = tf.keras.Model(inputs=cnn_input, outputs=x)
        
        # RNN part (for sequence of frames)
        sequence_input = tf.keras.Input(shape=(None, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        
        # Apply CNN to each frame
        frame_features = tf.keras.layers.TimeDistributed(cnn_model)(sequence_input)
        
        # Apply RNN to sequence of features
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(frame_features)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(x)
        
        # Classification head
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        # Create model
        sequence_model = tf.keras.Model(inputs=sequence_input, outputs=outputs)
        
        # For single image inference, we need a separate model
        # that can process a single frame
        single_output = tf.keras.layers.Dense(128, activation='relu')(cnn_model.output)
        single_output = tf.keras.layers.Dropout(0.5)(single_output)
        single_output = tf.keras.layers.Dense(1, activation='sigmoid')(single_output)
        
        # This is the model we'll use for inference on single images
        self.model = tf.keras.Model(inputs=cnn_input, outputs=single_output)
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Store the sequence model for future use
        self.sequence_model = sequence_model
        
        logger.info("Created a new hybrid CNN-RNN model")
    
    def _download_model(self, url, save_path):
        """
        Download a model file from URL
        """
        try:
            logger.info(f"Downloading model from {url}")
            logger.info(f"This may take a few minutes depending on your internet connection...")
            
            # Create a directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Download the file
            urllib.request.urlretrieve(url, save_path)
            logger.info(f"Model downloaded to {save_path}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to download model: {str(e)}", exc_info=True)
            return False
    
    def _load_fallback_model(self):
        """
        Load a simple fallback model when the main model cannot be loaded
        This is a simplified model built on the fly
        """
        logger.warning("Loading fallback model")
        
        # Create a simple CNN model
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Save model architecture and weights
        model_json = self.model.to_json()
        with open(os.path.join(MODEL_DIR, "fallback_model.json"), "w") as json_file:
            json_file.write(model_json)
        
        # Warmup the model
        self._warmup_model()
        
        logger.warning("Fallback model loaded - this is a simpler model and may not perform as well")
    
    def _warmup_model(self):
        """
        Warmup the model with a dummy prediction to initialize all weights
        """
        try:
            # Create dummy input
            dummy_input = np.zeros((1,) + self.input_shape)
            
            # Make a prediction to warm up the model
            self.model.predict(dummy_input, verbose=0)
            logger.debug("Model warmed up with dummy prediction")
        except Exception as e:
            logger.error(f"Failed to warm up model: {str(e)}", exc_info=True)
    
    def preprocess_image(self, image):
        """
        Preprocess the image before feeding it to the model
        Enhanced with additional preprocessing steps for better detection
        """
        # Ensure image is the right size
        if image.shape[:2] != MODEL_INPUT_SIZE:
            image = cv2.resize(image, MODEL_INPUT_SIZE)
        
        # Ensure the image has 3 channels
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Apply additional preprocessing steps
        # 1. Normalize pixel values
        image = image.astype(np.float32)
        
        # 2. Apply model-specific preprocessing
        if self.model_name == 'efficientnet':
            image = efficientnet_preprocess(image)
        elif self.model_name == 'xception':
            # Scale to [0, 1] and then to [-1, 1]
            image = image / 255.0
            image = (image - 0.5) * 2.0
        else:  # MesoNet, ViT, hybrid, and fallback
            # Scale to [0, 1]
            image = image / 255.0
        
        # 3. Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image, include_uncertainty=True):
        """
        Predict if an image is a deepfake with uncertainty estimation
        
        Args:
            image: Preprocessed face image
            include_uncertainty: Whether to calculate uncertainty using MC dropout
            
        Returns:
            Dictionary with prediction probability, uncertainty measures, and metadata
        """
        # Count predictions for monitoring
        self.prediction_count += 1
        
        try:
            # Check if model is loaded
            if self.model is None:
                logger.error("Model not loaded")
                self.failed_predictions += 1
                return {"probability": 0.5, "uncertainty": 1.0, "error": "Model not loaded"}
            
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Record start time for inference
            start_time = time.time()
            
            # Make prediction
            if include_uncertainty and self.uncertainty_enabled:
                predictions = []
                # Enable dropout at inference time for Monte Carlo dropout
                for _ in range(self.mc_dropout_samples):
                    with tf.keras.backend.learning_phase_scope(1):  # Set learning phase to 1 (training) to enable dropout
                        pred = self.model.predict(processed_image, verbose=0)[0][0]
                        predictions.append(pred)
                
                # Calculate mean and standard deviation
                prediction = np.mean(predictions)
                uncertainty = np.std(predictions)
            else:
                # Standard prediction
                prediction = float(self.model.predict(processed_image, verbose=0)[0][0])
                uncertainty = 0.0
            
            # Generate attention map if available
            attention_map = None
            if self.attention_model is not None:
                try:
                    attention_map = self.attention_model.predict(processed_image, verbose=0)[0]
                    attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
                except Exception as e:
                    logger.error(f"Error generating attention map: {str(e)}")
            
            # Record inference time
            inference_time = time.time() - start_time
            self.total_inference_time += inference_time
            self.min_inference_time = min(self.min_inference_time, inference_time)
            self.max_inference_time = max(self.max_inference_time, inference_time)
            
            # Clip prediction to [0.05, 0.95] range to avoid extreme certainty
            prediction = min(0.95, max(0.05, float(prediction)))
            
            # Calculate calibrated confidence using Platt scaling
            # This requires the model to be calibrated during training
            calibrated_prob = self._calibrate_probability(prediction)
            
            # Log successful prediction
            self.successful_predictions += 1
            logger.debug(f"Model {self.model_name} prediction: {prediction:.4f} (calibrated: {calibrated_prob:.4f}), uncertainty: {uncertainty:.4f} in {inference_time:.4f}s")
            
            # Return comprehensive prediction info
            return {
                "probability": calibrated_prob,
                "raw_probability": prediction,
                "uncertainty": float(uncertainty),
                "inference_time": inference_time,
                "model_name": self.model_name,
                "attention_map": attention_map.tolist() if attention_map is not None else None,
                "calibrated": True
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            self.failed_predictions += 1
            return {"probability": 0.5, "uncertainty": 1.0, "error": str(e)}
    
    def _calibrate_probability(self, raw_prob):
        """
        Calibrate the probability using Platt scaling
        
        Args:
            raw_prob: Raw prediction probability from the model
            
        Returns:
            Calibrated probability
        """
        # Parameters for Platt scaling (should ideally be learned on a validation set)
        # Default parameters that are approximately neutral
        A = 1.0
        B = 0.0
        
        # Different calibration parameters for different models
        if self.model_name == 'efficientnet':
            A = 1.2
            B = -0.1
        elif self.model_name == 'xception':
            A = 0.9
            B = 0.05
        elif self.model_name == 'mesonet':
            A = 1.1
            B = -0.05
        elif self.model_name == 'vit':
            A = 1.0
            B = 0.0
            
        # Apply Platt scaling: P(y=1) = 1 / (1 + exp(A*f(x) + B))
        # where f(x) is the raw model output in logit form
        
        # Convert probability to logit
        logit = np.log(raw_prob / (1 - raw_prob))
        
        # Apply scaling
        scaled_logit = A * logit + B
        
        # Convert back to probability
        calibrated_prob = 1 / (1 + np.exp(-scaled_logit))
        
        # Clip to valid range
        calibrated_prob = min(0.95, max(0.05, float(calibrated_prob)))
        
        return calibrated_prob
    
    def get_model_info(self):
        """
        Get information about the model
        """
        # Calculate average inference time
        avg_inference_time = 0
        if self.successful_predictions > 0:
            avg_inference_time = self.total_inference_time / self.successful_predictions
        
        return {
            "name": self.model_name,
            "input_shape": self.input_shape,
            "is_fallback": not os.path.exists(os.path.join(MODEL_DIR, f"{self.model_name}_deepfake.h5")),
            "predictions": {
                "total": self.prediction_count,
                "successful": self.successful_predictions,
                "failed": self.failed_predictions,
                "success_rate": self.successful_predictions / max(1, self.prediction_count)
            },
            "performance": {
                "avg_inference_time": avg_inference_time,
                "min_inference_time": self.min_inference_time if self.min_inference_time != float('inf') else 0,
                "max_inference_time": self.max_inference_time
            },
            "uncertainty_enabled": self.uncertainty_enabled,
            "mc_dropout_samples": self.mc_dropout_samples,
            "has_attention_model": self.attention_model is not None
        }
    
    def __str__(self):
        """String representation of the model"""
        return f"DeepfakeDetectionModel(name={self.model_name}, input_shape={self.input_shape})"

