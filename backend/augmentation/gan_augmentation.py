"""
GAN-based augmentation for deepfake detection
Implements StyleGAN2-based face swaps and domain randomization
"""

import os
import numpy as np
import cv2
import tensorflow as tf
import logging
from typing import List, Dict, Tuple, Optional, Union
import random
from pathlib import Path
import albumentations as A
from PIL import Image
import time

# Configure logging
logger = logging.getLogger(__name__)

class StyleGAN2Augmenter:
    """
    Implements StyleGAN2-based face swaps and domain randomization
    for ultra-realistic deepfake augmentation
    """
    
    def __init__(self, 
                 stylegan_model_path: str = None,
                 cache_dir: str = "cache/stylegan2",
                 device: str = "auto"):
        """
        Initialize StyleGAN2 augmenter
        
        Args:
            stylegan_model_path: Path to StyleGAN2 model checkpoint
            cache_dir: Directory to cache generated faces
            device: Device to use for inference ('cpu', 'gpu', or 'auto')
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Set device
        if device == "auto":
            self.device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
        else:
            self.device = device.upper()
        
        logger.info(f"Initializing StyleGAN2Augmenter on {self.device}")
        
        # Load StyleGAN2 model
        self.model = None
        self.model_loaded = False
        
        if stylegan_model_path and os.path.exists(stylegan_model_path):
            self._load_model(stylegan_model_path)
        else:
            logger.warning("StyleGAN2 model path not provided or invalid. "
                          "Model will be downloaded on first use.")
        
        # Initialize latent cache for faster generation
        self.latent_cache = {}
        self.max_cache_size = 1000
        
        # Initialize domain randomization parameters
        self.domain_params = self._initialize_domain_params()
    
    def _load_model(self, model_path: str) -> None:
        """
        Load StyleGAN2 model from checkpoint
        
        Args:
            model_path: Path to StyleGAN2 model checkpoint
        """
        try:
            # Use TF-Hub for easier loading if available
            import tensorflow_hub as hub
            
            logger.info(f"Loading StyleGAN2 model from {model_path}")
            with tf.device(f"/{self.device}:0"):
                self.model = hub.load(model_path)
            
            # Test model with a random latent vector
            _ = self.model.generate_images(
                tf.random.normal(shape=[1, 512]), 
                randomize_noise=False
            )
            
            self.model_loaded = True
            logger.info("StyleGAN2 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading StyleGAN2 model: {str(e)}", exc_info=True)
            logger.info("Will attempt to download model on first use")
    
    def _ensure_model_loaded(self) -> bool:
        """
        Ensure StyleGAN2 model is loaded, downloading if necessary
        
        Returns:
            True if model is loaded successfully, False otherwise
        """
        if self.model_loaded:
            return True
        
        try:
            # Try to load from TF-Hub
            import tensorflow_hub as hub
            
            logger.info("Downloading StyleGAN2 model from TensorFlow Hub")
            with tf.device(f"/{self.device}:0"):
                # Use the FFHQ trained model
                self.model = hub.load("https://tfhub.dev/google/stylegan2/1")
            
            # Test model with a random latent vector
            _ = self.model.generate_images(
                tf.random.normal(shape=[1, 512]), 
                randomize_noise=False
            )
            
            self.model_loaded = True
            logger.info("StyleGAN2 model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading StyleGAN2 model: {str(e)}", exc_info=True)
            return False
    
    def _initialize_domain_params(self) -> Dict:
        """
        Initialize domain randomization parameters
        
        Returns:
            Dictionary of domain randomization parameters
        """
        return {
            # Background parameters
            "backgrounds": {
                "enabled": True,
                "prob": 0.7,
                "types": ["solid", "gradient", "noise", "image"],
                "image_dir": "data/backgrounds"
            },
            
            # Lighting parameters
            "lighting": {
                "enabled": True,
                "prob": 0.8,
                "types": ["brightness", "contrast", "shadows", "highlights", "color_shift"],
                "intensity_range": (0.2, 0.8)
            },
            
            # Compression artifacts
            "compression": {
                "enabled": True,
                "prob": 0.6,
                "types": ["jpeg", "webp", "h264"],
                "quality_range": (50, 95)
            },
            
            # Noise parameters
            "noise": {
                "enabled": True,
                "prob": 0.5,
                "types": ["gaussian", "poisson", "speckle", "s&p"],
                "intensity_range": (0.01, 0.1)
            },
            
            # Blur parameters
            "blur": {
                "enabled": True,
                "prob": 0.4,
                "types": ["gaussian", "motion", "defocus"],
                "intensity_range": (0.5, 3.0)
            }
        }
    
    def generate_synthetic_face(self, 
                               seed: Optional[int] = None,
                               truncation: float = 0.7) -> np.ndarray:
        """
        Generate a synthetic face using StyleGAN2
        
        Args:
            seed: Random seed for reproducibility
            truncation: Truncation parameter for latent space (0.5-1.0 for realistic faces)
            
        Returns:
            Generated face image as numpy array (RGB)
        """
        if not self._ensure_model_loaded():
            raise RuntimeError("StyleGAN2 model could not be loaded")
        
        # Set seed for reproducibility
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            # Check if we have this seed cached
            if seed in self.latent_cache:
                latent = self.latent_cache[seed]
            else:
                # Generate new latent vector
                latent = tf.random.normal(shape=[1, 512], seed=seed)
                
                # Cache latent vector
                if len(self.latent_cache) < self.max_cache_size:
                    self.latent_cache[seed] = latent
        else:
            # Generate random latent vector
            latent = tf.random.normal(shape=[1, 512])
        
        # Generate image
        with tf.device(f"/{self.device}:0"):
            generated_image = self.model.generate_images(
                latent, 
                truncation=truncation,
                randomize_noise=True
            )
        
        # Convert to numpy array and normalize to [0, 255]
        image = generated_image[0].numpy()
        image = ((image + 1) * 127.5).astype(np.uint8)
        
        return image
    
    def face_swap(self, 
                 source_face: np.ndarray, 
                 target_face: Optional[np.ndarray] = None,
                 blend_factor: float = 0.8,
                 preserve_identity: bool = True) -> np.ndarray:
        """
        Perform face swapping using StyleGAN2
        
        Args:
            source_face: Source face image
            target_face: Target face image (if None, a synthetic face will be generated)
            blend_factor: Blending factor between source and target (0.0-1.0)
            preserve_identity: Whether to preserve identity features of source face
            
        Returns:
            Face-swapped image
        """
        if not self._ensure_model_loaded():
            raise RuntimeError("StyleGAN2 model could not be loaded")
        
        # If target face is not provided, generate one
        if target_face is None:
            target_face = self.generate_synthetic_face()
        
        # Resize faces to 1024x1024 (StyleGAN2 output size)
        source_face_resized = cv2.resize(source_face, (1024, 1024))
        target_face_resized = cv2.resize(target_face, (1024, 1024))
        
        # Convert to RGB if grayscale
        if len(source_face_resized.shape) == 2:
            source_face_resized = cv2.cvtColor(source_face_resized, cv2.COLOR_GRAY2RGB)
        if len(target_face_resized.shape) == 2:
            target_face_resized = cv2.cvtColor(target_face_resized, cv2.COLOR_GRAY2RGB)
        
        # Normalize to [-1, 1] for StyleGAN2
        source_face_norm = source_face_resized.astype(np.float32) / 127.5 - 1.0
        target_face_norm = target_face_resized.astype(np.float32) / 127.5 - 1.0
        
        # Project source face to latent space
        with tf.device(f"/{self.device}:0"):
            source_latent = self._project_to_latent(source_face_norm)
            target_latent = self._project_to_latent(target_face_norm)
            
            # Blend latents
            if preserve_identity:
                # Preserve identity by keeping early layers from source
                blended_latent = self._blend_latents_with_identity(
                    source_latent, target_latent, blend_factor
                )
            else:
                # Simple linear interpolation
                blended_latent = (1 - blend_factor) * source_latent + blend_factor * target_latent
            
            # Generate swapped face
            swapped_face = self.model.generate_images(
                blended_latent,
                randomize_noise=False
            )
        
        # Convert to numpy array and normalize to [0, 255]
        swapped_face = swapped_face[0].numpy()
        swapped_face = ((swapped_face + 1) * 127.5).astype(np.uint8)
        
        return swapped_face
    
    def _project_to_latent(self, face_image: np.ndarray) -> tf.Tensor:
        """
        Project a face image to StyleGAN2 latent space
        This is a simplified implementation - a real system would use an encoder network
        
        Args:
            face_image: Face image normalized to [-1, 1]
            
        Returns:
            Latent vector
        """
        # In a real implementation, this would use a proper encoder network
        # For simplicity, we'll use a random latent and optimize it
        
        # Create a random initial latent
        latent = tf.Variable(tf.random.normal(shape=[1, 512]))
        
        # Convert input to tensor
        face_tensor = tf.convert_to_tensor(face_image[np.newaxis, ...])
        
        # Optimize latent to match input face
        optimizer = tf.optimizers.Adam(learning_rate=0.01)
        
        # Simple optimization loop (in practice, this would be more sophisticated)
        for i in range(10):  # Very few steps for demo purposes
            with tf.GradientTape() as tape:
                # Generate image from latent
                generated = self.model.generate_images(latent, randomize_noise=False)
                
                # Calculate loss (L2 distance)
                loss = tf.reduce_mean(tf.square(generated - face_tensor))
            
            # Update latent
            gradients = tape.gradient(loss, [latent])
            optimizer.apply_gradients(zip(gradients, [latent]))
        
        return latent
    
    def _blend_latents_with_identity(self, 
                                    source_latent: tf.Tensor, 
                                    target_latent: tf.Tensor,
                                    blend_factor: float) -> tf.Tensor:
        """
        Blend latents while preserving identity features
        
        Args:
            source_latent: Source face latent
            target_latent: Target face latent
            blend_factor: Blending factor
            
        Returns:
            Blended latent
        """
        # In StyleGAN2, early layers control identity, later layers control details
        # This is a simplified implementation
        
        # Convert to numpy for easier manipulation
        source_np = source_latent.numpy()
        target_np = target_latent.numpy()
        
        # Create blended latent
        blended_np = source_np.copy()
        
        # Identity features (first 20% of dimensions)
        identity_dims = int(source_np.shape[1] * 0.2)
        
        # Keep identity features from source, blend the rest
        for i in range(identity_dims, source_np.shape[1]):
            blended_np[0, i] = (1 - blend_factor) * source_np[0, i] + blend_factor * target_np[0, i]
        
        return tf.convert_to_tensor(blended_np)
    
    def apply_domain_randomization(self, 
                                  image: np.ndarray,
                                  params: Optional[Dict] = None) -> np.ndarray:
        """
        Apply domain randomization to an image
        
        Args:
            image: Input image
            params: Domain randomization parameters (if None, use default)
            
        Returns:
            Augmented image
        """
        if params is None:
            params = self.domain_params
        
        # Make a copy of the image
        result = image.copy()
        
        # Apply background randomization
        if params["backgrounds"]["enabled"] and random.random() < params["backgrounds"]["prob"]:
            result = self._randomize_background(result)
        
        # Apply lighting randomization
        if params["lighting"]["enabled"] and random.random() < params["lighting"]["prob"]:
            result = self._randomize_lighting(result)
        
        # Apply compression artifacts
        if params["compression"]["enabled"] and random.random() < params["compression"]["prob"]:
            result = self._add_compression_artifacts(result)
        
        # Apply noise
        if params["noise"]["enabled"] and random.random() < params["noise"]["prob"]:
            result = self._add_noise(result)
        
        # Apply blur
        if params["blur"]["enabled"] and random.random() < params["blur"]["prob"]:
            result = self._add_blur(result)
        
        return result
    
    def _randomize_background(self, image: np.ndarray) -> np.ndarray:
        """
        Randomize the background of an image
        
        Args:
            image: Input image
            
        Returns:
            Image with randomized background
        """
        # Create a face mask (in a real implementation, this would use a segmentation model)
        # For simplicity, we'll use a simple elliptical mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        center = (image.shape[1] // 2, image.shape[0] // 2)
        axes = (int(image.shape[1] * 0.4), int(image.shape[0] * 0.5))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # Blur the mask for smoother blending
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        # Choose background type
        bg_type = random.choice(self.domain_params["backgrounds"]["types"])
        
        if bg_type == "solid":
            # Solid color background
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            background = np.ones_like(image) * np.array(color, dtype=np.uint8)
            
        elif bg_type == "gradient":
            # Gradient background
            color1 = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            color2 = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            
            # Create gradient
            background = np.zeros_like(image)
            h, w = image.shape[:2]
            
            if random.random() < 0.5:
                # Horizontal gradient
                for i in range(w):
                    ratio = i / w
                    color = tuple(int((1 - ratio) * c1 + ratio * c2) for c1, c2 in zip(color1, color2))
                    background[:, i] = color
            else:
                # Vertical gradient
                for i in range(h):
                    ratio = i / h
                    color = tuple(int((1 - ratio) * c1 + ratio * c2) for c1, c2 in zip(color1, color2))
                    background[i, :] = color
            
        elif bg_type == "noise":
            # Noise background
            background = np.random.randint(0, 255, size=image.shape, dtype=np.uint8)
            
            # Optionally blur the noise
            if random.random() < 0.5:
                blur_size = random.choice([3, 5, 7, 9])
                background = cv2.GaussianBlur(background, (blur_size, blur_size), 0)
            
        else:  # "image"
            # Image background
            # In a real implementation, this would load from a directory of background images
            # For simplicity, we'll create a patterned background
            
            pattern_size = random.choice([10, 20, 30, 40])
            background = np.zeros_like(image)
            
            for y in range(0, image.shape[0], pattern_size):
                for x in range(0, image.shape[1], pattern_size):
                    color = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )
                    y_end = min(y + pattern_size, image.shape[0])
                    x_end = min(x + pattern_size, image.shape[1])
                    background[y:y_end, x:x_end] = color
        
        # Blend face with background using mask
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2) / 255.0
        result = (image * mask_3d + background * (1 - mask_3d)).astype(np.uint8)
        
        return result
    
    def _randomize_lighting(self, image: np.ndarray) -> np.ndarray:
        """
        Randomize lighting conditions in an image
        
        Args:
            image: Input image
            
        Returns:
            Image with randomized lighting
        """
        # Choose lighting type
        lighting_type = random.choice(self.domain_params["lighting"]["types"])
        
        # Get intensity
        min_intensity, max_intensity = self.domain_params["lighting"]["intensity_range"]
        intensity = random.uniform(min_intensity, max_intensity)
        
        if lighting_type == "brightness":
            # Adjust brightness
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Increase or decrease brightness
            if random.random() < 0.5:
                # Increase brightness
                v = cv2.add(v, int(intensity * 255))
            else:
                # Decrease brightness
                v = cv2.subtract(v, int(intensity * 255))
            
            # Merge channels and convert back to BGR
            hsv = cv2.merge([h, s, v])
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        elif lighting_type == "contrast":
            # Adjust contrast
            alpha = 1.0 + intensity * 2  # Contrast factor
            result = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
            
        elif lighting_type == "shadows":
            # Darken shadows
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Create a mask for shadow regions (lower L values)
            shadow_mask = l < 100
            
            # Darken shadow regions
            l[shadow_mask] = l[shadow_mask] * (1 - intensity)
            
            # Merge channels and convert back to BGR
            lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        elif lighting_type == "highlights":
            # Brighten highlights
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Create a mask for highlight regions (higher L values)
            highlight_mask = l > 150
            
            # Brighten highlight regions
            l[highlight_mask] = np.minimum(255, l[highlight_mask] * (1 + intensity))
            
            # Merge channels and convert back to BGR
            lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        else:  # "color_shift"
            # Shift color balance
            b, g, r = cv2.split(image)
            
            # Randomly adjust color channels
            channels = [b, g, r]
            channel_idx = random.randint(0, 2)
            
            if random.random() < 0.5:
                # Increase selected channel
                channels[channel_idx] = cv2.add(channels[channel_idx], int(intensity * 255))
            else:
                # Decrease selected channel
                channels[channel_idx] = cv2.subtract(channels[channel_idx], int(intensity * 255))
            
            # Merge channels
            result = cv2.merge(channels)
        
        return result
    
    def _add_compression_artifacts(self, image: np.ndarray) -> np.ndarray:
        """
        Add compression artifacts to an image
        
        Args:
            image: Input image
            
        Returns:
            Image with compression artifacts
        """
        # Choose compression type
        compression_type = random.choice(self.domain_params["compression"]["types"])
        
        # Get quality
        min_quality, max_quality = self.domain_params["compression"]["quality_range"]
        quality = random.randint(min_quality, max_quality)
        
        if compression_type == "jpeg":
            # Add JPEG compression artifacts
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, buffer = cv2.imencode(".jpg", image, encode_param)
            result = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            
        elif compression_type == "webp":
            # Add WebP compression artifacts
            encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
            _, buffer = cv2.imencode(".webp", image, encode_param)
            result = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            
        else:  # "h264"
            # Simulate H.264 compression artifacts
            # This is a simplified simulation
            
            # First apply JPEG compression to simulate DCT artifacts
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, buffer = cv2.imencode(".jpg", image, encode_param)
            result = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            
            # Then add some blur to simulate loss of high-frequency details
            blur_size = 3
            result = cv2.GaussianBlur(result, (blur_size, blur_size), 0)
            
            # Finally add some noise to simulate quantization errors
            noise = np.random.normal(0, 3, result.shape).astype(np.int8)
            result = cv2.add(result, noise)
        
        return result
    
    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Add noise to an image
        
        Args:
            image: Input image
            
        Returns:
            Image with added noise
        """
        # Choose noise type
        noise_type = random.choice(self.domain_params["noise"]["types"])
        
        # Get intensity
        min_intensity, max_intensity = self.domain_params["noise"]["intensity_range"]
        intensity = random.uniform(min_intensity, max_intensity)
        
        # Convert to float for noise addition
        image_float = image.astype(np.float32)
        
        if noise_type == "gaussian":
            # Add Gaussian noise
            mean = 0
            sigma = intensity * 255
            noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
            noisy_image = image_float + noise
            
        elif noise_type == "poisson":
            # Add Poisson noise
            noise = np.random.poisson(image_float * intensity) / intensity
            noisy_image = noise
            
        elif noise_type == "speckle":
            # Add speckle noise
            noise = np.random.normal(0, intensity, image.shape).astype(np.float32)
            noisy_image = image_float + image_float * noise
            
        else:  # "s&p" (salt and pepper)
            # Add salt and pepper noise
            noisy_image = image_float.copy()
            
            # Salt
            salt_prob = intensity / 2
            salt_mask = np.random.random(image.shape[:2]) < salt_prob
            noisy_image[salt_mask] = 255
            
            # Pepper
            pepper_prob = intensity / 2
            pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
            noisy_image[pepper_mask] = 0
        
        # Clip values to valid range and convert back to uint8
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    def _add_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Add blur to an image
        
        Args:
            image: Input image
            
        Returns:
            Blurred image
        """
        # Choose blur type
        blur_type = random.choice(self.domain_params["blur"]["types"])
        
        # Get intensity
        min_intensity, max_intensity = self.domain_params["blur"]["intensity_range"]
        intensity = random.uniform(min_intensity, max_intensity)
        
        if blur_type == "gaussian":
            # Add Gaussian blur
            kernel_size = int(intensity * 10) | 1  # Ensure odd kernel size
            kernel_size = max(3, kernel_size)  # Minimum size of 3
            blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
        elif blur_type == "motion":
            # Add motion blur
            kernel_size = int(intensity * 20) | 1  # Ensure odd kernel size
            kernel_size = max(3, kernel_size)  # Minimum size of 3
            
            # Create motion blur kernel
            kernel = np.zeros((kernel_size, kernel_size))
            
            if random.random() < 0.5:
                # Horizontal motion blur
                kernel[kernel_size // 2, :] = 1.0
            else:
                # Vertical motion blur
                kernel[:, kernel_size // 2] = 1.0
            
            kernel = kernel / kernel_size
            
            # Apply motion blur
            blurred_image = cv2.filter2D(image, -1, kernel)
            
        else:  # "defocus"
            # Add defocus blur (disk kernel)
            kernel_size = int(intensity * 10) | 1  # Ensure odd kernel size
            kernel_size = max(3, kernel_size)  # Minimum size of 3
            
            # Create disk kernel
            y, x = np.ogrid[-kernel_size//2:kernel_size//2+1, -kernel_size//2:kernel_size//2+1]
            mask = x*x + y*y <= (kernel_size//2)*(kernel_size//2)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[mask] = 1.0
            kernel = kernel / np.sum(kernel)
            
            # Apply defocus blur
            blurred_image = cv2.filter2D(image, -1, kernel)
        
        return blurred_image


class AdvancedAugmentationPipeline:
    """
    Advanced augmentation pipeline for deepfake detection
    Combines GAN-based face swaps, domain randomization, and traditional augmentations
    """
    
    def __init__(self, 
                 stylegan_model_path: str = None,
                 cache_dir: str = "cache/augmentation",
                 device: str = "auto"):
        """
        Initialize advanced augmentation pipeline
        
        Args:
            stylegan_model_path: Path to StyleGAN2 model checkpoint
            cache_dir: Directory to cache generated faces
            device: Device to use for inference ('cpu', 'gpu', or 'auto')
        """
        # Initialize StyleGAN2 augmenter
        self.gan_augmenter = StyleGAN2Augmenter(
            stylegan_model_path=stylegan_model_path,
            cache_dir=cache_dir,
            device=device
        )
        
        # Initialize Albumentations pipeline for traditional augmentations
        self.albumentations_pipeline = self._create_albumentations_pipeline()
        
        # Cache for synthetic faces
        self.synthetic_faces_cache = []
        self.max_cache_size = 100
        
        logger.info("Advanced augmentation pipeline initialized")
    
    def _create_albumentations_pipeline(self) -> A.Compose:
        """
        Create Albumentations pipeline for traditional augmentations
        
        Returns:
            Albumentations Compose object
        """
        return A.Compose([
            # Spatial augmentations
            A.OneOf([
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), p=0.5),
            ], p=0.7),
            
            # Color augmentations
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.ColorJitter(p=0.5),
                A.ToGray(p=0.2),
            ], p=0.7),
            
            # Quality degradation
            A.OneOf([
                A.GaussianBlur(p=0.3),
                A.MotionBlur(p=0.3),
                A.ImageCompression(quality_lower=50, quality_upper=99, p=0.5),
                A.GaussNoise(p=0.3),
            ], p=0.5),
            
            # Advanced augmentations
            A.OneOf([
                A.RandomSunFlare(p=0.2),
                A.RandomShadow(p=0.2),
                A.RandomFog(p=0.1),
                A.RandomRain(p=0.1),
            ], p=0.3),
            
            # Cutout/CoarseDropout for robustness
            A.OneOf([
                A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, p=0.3),
                A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.3),
            ], p=0.2),
        ])
    
    def augment_batch(self, 
                     images: List[np.ndarray], 
                     labels: List[int],
                     gan_swap_prob: float = 0.3,
                     domain_rand_prob: float = 0.5) -> Tuple[List[np.ndarray], List[int]]:
        """
        Apply advanced augmentation to a batch of images
        
        Args:
            images: List of input images
            labels: List of labels (0 for real, 1 for fake)
            gan_swap_prob: Probability of applying GAN-based face swap
            domain_rand_prob: Probability of applying domain randomization
            
        Returns:
            Tuple of (augmented_images, augmented_labels)
        """
        augmented_images = []
        augmented_labels = []
        
        for image, label in zip(images, labels):
            # Apply GAN-based face swap with probability gan_swap_prob
            if random.random() < gan_swap_prob:
                try:
                    # For real faces, create a deepfake
                    if label == 0:
                        # Create a deepfake by swapping with a synthetic face
                        swapped_face = self.gan_augmenter.face_swap(
                            source_face=image,
                            target_face=self._get_synthetic_face(),
                            blend_factor=random.uniform(0.7, 0.9),
                            preserve_identity=random.random() < 0.7
                        )
                        
                        # Add to augmented batch
                        augmented_images.append(swapped_face)
                        augmented_labels.append(1)  # Label as fake
                        
                        # Also add the original image sometimes
                        if random.random() < 0.5:
                            # Apply traditional augmentations to original
                            aug_image = self._apply_traditional_augmentations(image)
                            augmented_images.append(aug_image)
                            augmented_labels.append(label)
                    else:
                        # For fake faces, just apply domain randomization
                        aug_image = self._apply_traditional_augmentations(image)
                        augmented_images.append(aug_image)
                        augmented_labels.append(label)
                        
                except Exception as e:
                    logger.warning(f"GAN augmentation failed: {str(e)}")
                    # Fall back to traditional augmentation
                    aug_image = self._apply_traditional_augmentations(image)
                    augmented_images.append(aug_image)
                    augmented_labels.append(label)
            else:
                # Apply domain randomization with probability domain_rand_prob
                if random.random() < domain_rand_prob:
                    try:
                        # Apply domain randomization
                        randomized_image = self.gan_augmenter.apply_domain_randomization(image)
                        
                        # Apply traditional augmentations
                        aug_image = self._apply_traditional_augmentations(randomized_image)
                        
                        augmented_images.append(aug_image)
                        augmented_labels.append(label)
                    except Exception as e:
                        logger.warning(f"Domain randomization failed: {str(e)}")
                        # Fall back to traditional augmentation
                        aug_image = self._apply_traditional_augmentations(image)
                        augmented_images.append(aug_image)
                        augmented_labels.append(label)
                else:
                    # Apply only traditional augmentations
                    aug_image = self._apply_traditional_augmentations(image)
                    augmented_images.append(aug_image)
                    augmented_labels.append(label)
        
        return augmented_images, augmented_labels
    
    def _apply_traditional_augmentations(self, image: np.ndarray) -> np.ndarray:
        """
        Apply traditional augmentations using Albumentations
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        # Apply Albumentations pipeline
        augmented = self.albumentations_pipeline(image=image)
        return augmented["image"]
    
    def _get_synthetic_face(self) -> np.ndarray:
        """
        Get a synthetic face from cache or generate a new one
        
        Returns:
            Synthetic face image
        """
        # If cache is empty or with low probability, generate a new face
        if not self.synthetic_faces_cache or random.random() < 0.3:
            synthetic_face = self.gan_augmenter.generate_synthetic_face(
                seed=random.randint(0, 10000),
                truncation=random.uniform(0.5, 0.9)
            )
            
            # Add to cache if not full
            if len(self.synthetic_faces_cache) < self.max_cache_size:
                self.synthetic_faces_cache.append(synthetic_face)
            
            return synthetic_face
        else:
            # Return a random face from cache
            return random.choice(self.synthetic_faces_cache)
    
    def generate_synthetic_dataset(self, 
                                  num_samples: int, 
                                  output_dir: str,
                                  real_faces_dir: Optional[str] = None) -> None:
        """
        Generate a synthetic dataset of deepfakes
        
        Args:
            num_samples: Number of synthetic samples to generate
            output_dir: Directory to save generated samples
            real_faces_dir: Directory containing real faces for swapping
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "real"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "fake"), exist_ok=True)
        
        # Load real faces if provided
        real_faces = []
        if real_faces_dir and os.path.exists(real_faces_dir):
            # Get all image files
            image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(list(Path(real_faces_dir).glob(f"*{ext}")))
            
            # Load images
            for img_path in image_files[:min(100, len(image_files))]:  # Limit to 100 images
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        real_faces.append(img)
                except Exception as e:
                    logger.warning(f"Failed to load image {img_path}: {str(e)}")
        
        logger.info(f"Loaded {len(real_faces)} real faces for swapping")
        
        # Generate synthetic samples
        for i in range(num_samples):
            try:
                # Generate a synthetic face
                synthetic_face = self.gan_augmenter.generate_synthetic_face(
                    seed=i,
                    truncation=random.uniform(0.5, 0.9)
                )
                
                # Save as real
                cv2.imwrite(
                    os.path.join(output_dir, "real", f"synthetic_real_{i:04d}.png"),
                    synthetic_face
                )
                
                # Create a deepfake by swapping with a real face or another synthetic face
                if real_faces and random.random() < 0.7:
                    # Swap with a real face
                    source_face = random.choice(real_faces)
                else:
                    # Swap with another synthetic face
                    source_face = self.gan_augmenter.generate_synthetic_face(
                        seed=i + 10000,
                        truncation=random.uniform(0.5, 0.9)
                    )
                
                # Perform face swap
                swapped_face = self.gan_augmenter.face_swap(
                    source_face=source_face,
                    target_face=synthetic_face,
                    blend_factor=random.uniform(0.7, 0.9),
                    preserve_identity=random.random() < 0.7
                )
                
                # Apply domain randomization
                if random.random() < 0.5:
                    swapped_face = self.gan_augmenter.apply_domain_randomization(swapped_face)
                
                # Save as fake
                cv2.imwrite(
                    os.path.join(output_dir, "fake", f"synthetic_fake_{i:04d}.png"),
                    swapped_face
                )
                
                if i % 10 == 0:
                    logger.info(f"Generated {i}/{num_samples} synthetic samples")
                
            except Exception as e:
                logger.error(f"Failed to generate synthetic sample {i}: {str(e)}")
        
        logger.info(f"Generated {num_samples} synthetic samples in {output_dir}")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create augmentation pipeline
    augmentation_pipeline = AdvancedAugmentationPipeline()
    
    # Generate synthetic dataset
    augmentation_pipeline.generate_synthetic_dataset(
        num_samples=100,
        output_dir="data/synthetic_dataset",
        real_faces_dir="data/real_faces"
    )