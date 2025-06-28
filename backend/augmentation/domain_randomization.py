"""
Domain randomization for deepfake detection
Implements advanced augmentation techniques for backgrounds, lighting, and compression artifacts
"""

import os
import numpy as np
import cv2
import logging
import random
from typing import List, Dict, Tuple, Optional, Union
import albumentations as A
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class DomainRandomizer:
    """
    Implements domain randomization techniques for deepfake detection
    Focuses on realistic variations in backgrounds, lighting, and compression artifacts
    """
    
    def __init__(self, 
                 background_dir: str = None,
                 cache_dir: str = "cache/domain_randomization"):
        """
        Initialize domain randomizer
        
        Args:
            background_dir: Directory containing background images
            cache_dir: Directory to cache processed backgrounds
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize background images
        self.background_images = []
        if background_dir and os.path.exists(background_dir):
            self._load_backgrounds(background_dir)
        
        # Initialize domain randomization parameters
        self.domain_params = self._initialize_domain_params()
        
        # Initialize Albumentations pipeline
        self.albumentations_pipeline = self._create_albumentations_pipeline()
        
        logger.info("Domain randomizer initialized")
    
    def _load_backgrounds(self, background_dir: str) -> None:
        """
        Load background images from directory
        
        Args:
            background_dir: Directory containing background images
        """
        # Get all image files
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(Path(background_dir).glob(f"*{ext}")))
        
        # Load images
        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    self.background_images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load background image {img_path}: {str(e)}")
        
        logger.info(f"Loaded {len(self.background_images)} background images")
    
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
                "blur_prob": 0.5,
                "blur_range": (3, 15)
            },
            
            # Lighting parameters
            "lighting": {
                "enabled": True,
                "prob": 0.8,
                "types": ["brightness", "contrast", "shadows", "highlights", "color_shift", "vignette"],
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
            },
            
            # Weather effects
            "weather": {
                "enabled": True,
                "prob": 0.3,
                "types": ["rain", "snow", "fog", "sun_flare"],
                "intensity_range": (0.2, 0.6)
            }
        }
    
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
    
    def randomize_domain(self, 
                        image: np.ndarray,
                        face_mask: Optional[np.ndarray] = None,
                        params: Optional[Dict] = None) -> np.ndarray:
        """
        Apply domain randomization to an image
        
        Args:
            image: Input image
            face_mask: Optional mask for face region (255 for face, 0 for background)
            params: Domain randomization parameters (if None, use default)
            
        Returns:
            Augmented image
        """
        if params is None:
            params = self.domain_params
        
        # Make a copy of the image
        result = image.copy()
        
        # Generate face mask if not provided
        if face_mask is None:
            face_mask = self._generate_face_mask(image)
        
        # Apply background randomization
        if params["backgrounds"]["enabled"] and random.random() < params["backgrounds"]["prob"]:
            result = self._randomize_background(result, face_mask)
        
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
        
        # Apply weather effects
        if params["weather"]["enabled"] and random.random() < params["weather"]["prob"]:
            result = self._add_weather_effects(result)
        
        return result
    
    def _generate_face_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generate a face mask using a simple elliptical approximation
        In a real implementation, this would use a face segmentation model
        
        Args:
            image: Input image
            
        Returns:
            Face mask (255 for face, 0 for background)
        """
        # Create a face mask using a simple elliptical approximation
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        center = (image.shape[1] // 2, image.shape[0] // 2)
        axes = (int(image.shape[1] * 0.4), int(image.shape[0] * 0.5))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # Blur the mask for smoother blending
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    def _randomize_background(self, 
                             image: np.ndarray, 
                             face_mask: np.ndarray) -> np.ndarray:
        """
        Randomize the background of an image
        
        Args:
            image: Input image
            face_mask: Mask for face region (255 for face, 0 for background)
            
        Returns:
            Image with randomized background
        """
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
            if random.random() < self.domain_params["backgrounds"]["blur_prob"]:
                blur_size = random.randrange(
                    self.domain_params["backgrounds"]["blur_range"][0],
                    self.domain_params["backgrounds"]["blur_range"][1],
                    2  # Ensure odd kernel size
                )
                background = cv2.GaussianBlur(background, (blur_size, blur_size), 0)
            
        else:  # "image"
            # Image background
            if self.background_images:
                # Use a random background image
                bg_img = random.choice(self.background_images)
                
                # Resize to match input image
                background = cv2.resize(bg_img, (image.shape[1], image.shape[0]))
            else:
                # Fallback to pattern background if no images available
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
        mask_3d = np.repeat(face_mask[:, :, np.newaxis], 3, axis=2) / 255.0
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
            
        elif lighting_type == "color_shift":
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
            
        else:  # "vignette"
            # Add vignette effect
            height, width = image.shape[:2]
            
            # Create radial gradient mask
            x = np.linspace(-1, 1, width)
            y = np.linspace(-1, 1, height)
            x_grid, y_grid = np.meshgrid(x, y)
            radius = np.sqrt(x_grid**2 + y_grid**2)
            
            # Create vignette mask
            vignette = np.clip(1.0 - radius * intensity * 1.5, 0, 1)
            vignette = vignette[:, :, np.newaxis]
            
            # Apply vignette
            result = (image * vignette).astype(np.uint8)
        
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
    
    def _add_weather_effects(self, image: np.ndarray) -> np.ndarray:
        """
        Add weather effects to an image
        
        Args:
            image: Input image
            
        Returns:
            Image with weather effects
        """
        # Choose weather type
        weather_type = random.choice(self.domain_params["weather"]["types"])
        
        # Get intensity
        min_intensity, max_intensity = self.domain_params["weather"]["intensity_range"]
        intensity = random.uniform(min_intensity, max_intensity)
        
        if weather_type == "rain":
            # Add rain effect
            rain_layer = np.zeros_like(image)
            
            # Number of raindrops
            num_drops = int(image.shape[0] * image.shape[1] * 0.01 * intensity)
            
            # Rain properties
            drop_length = int(15 * intensity) + 1
            drop_width = 1
            drop_color = (200, 200, 200)  # Light gray
            
            # Add raindrops
            for _ in range(num_drops):
                x = random.randint(0, image.shape[1] - 1)
                y = random.randint(0, image.shape[0] - drop_length - 1)
                
                # Draw raindrop (angled line)
                angle = random.uniform(-0.2, 0.2)  # Slight angle variation
                x_end = int(x + drop_length * np.sin(angle))
                y_end = int(y + drop_length * np.cos(angle))
                
                # Ensure endpoints are within image bounds
                x_end = max(0, min(x_end, image.shape[1] - 1))
                y_end = max(0, min(y_end, image.shape[0] - 1))
                
                cv2.line(rain_layer, (x, y), (x_end, y_end), drop_color, drop_width)
            
            # Blend rain with image
            alpha = 0.7  # Rain opacity
            result = cv2.addWeighted(image, 1.0, rain_layer, alpha, 0)
            
        elif weather_type == "snow":
            # Add snow effect
            snow_layer = np.zeros_like(image)
            
            # Number of snowflakes
            num_flakes = int(image.shape[0] * image.shape[1] * 0.005 * intensity)
            
            # Snowflake properties
            flake_sizes = [1, 2, 3]
            flake_color = (250, 250, 250)  # White
            
            # Add snowflakes
            for _ in range(num_flakes):
                x = random.randint(0, image.shape[1] - 1)
                y = random.randint(0, image.shape[0] - 1)
                size = random.choice(flake_sizes)
                
                # Draw snowflake (circle)
                cv2.circle(snow_layer, (x, y), size, flake_color, -1)
            
            # Add slight blur to snow layer
            snow_layer = cv2.GaussianBlur(snow_layer, (3, 3), 0)
            
            # Blend snow with image
            alpha = 0.7  # Snow opacity
            result = cv2.addWeighted(image, 1.0, snow_layer, alpha, 0)
            
        elif weather_type == "fog":
            # Add fog effect
            fog_layer = np.ones_like(image) * 200  # Light gray fog
            
            # Create fog mask with random variation
            mask = np.random.uniform(0, 1, size=image.shape[:2])
            mask = cv2.GaussianBlur(mask, (51, 51), 0)
            
            # Normalize mask to control fog intensity
            mask = mask * intensity
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            
            # Blend fog with image
            result = (image * (1 - mask) + fog_layer * mask).astype(np.uint8)
            
        else:  # "sun_flare"
            # Add sun flare effect
            flare_layer = np.zeros_like(image)
            
            # Flare center position (typically in upper portion of image)
            center_x = random.randint(0, image.shape[1] - 1)
            center_y = random.randint(0, int(image.shape[0] * 0.6))
            
            # Flare properties
            flare_radius = int(min(image.shape) * 0.2 * intensity)
            flare_color = (100, 200, 255)  # Yellow-orange
            
            # Create radial gradient for flare
            y, x = np.ogrid[:image.shape[0], :image.shape[1]]
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Normalize distances
            max_dist = np.sqrt(image.shape[0]**2 + image.shape[1]**2)
            norm_dist = dist_from_center / max_dist
            
            # Create flare mask
            flare_mask = np.maximum(0, 1 - norm_dist / (intensity * 0.5))
            flare_mask = flare_mask[:, :, np.newaxis]
            
            # Create flare
            flare = np.ones_like(image) * np.array(flare_color)
            flare_effect = (flare * flare_mask).astype(np.uint8)
            
            # Add lens streaks
            num_streaks = random.randint(3, 7)
            for i in range(num_streaks):
                angle = i * (360 / num_streaks)
                angle_rad = np.radians(angle)
                
                end_x = int(center_x + np.cos(angle_rad) * flare_radius)
                end_y = int(center_y + np.sin(angle_rad) * flare_radius)
                
                cv2.line(flare_effect, (center_x, center_y), (end_x, end_y), 
                         flare_color, 2)
            
            # Blend flare with image using screen blending
            result = 255 - ((255 - image) * (255 - flare_effect) / 255)
            result = result.astype(np.uint8)
        
        return result
    
    def apply_traditional_augmentations(self, image: np.ndarray) -> np.ndarray:
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
    
    def augment_batch(self, 
                     images: List[np.ndarray],
                     domain_rand_prob: float = 0.7,
                     traditional_aug_prob: float = 0.8) -> List[np.ndarray]:
        """
        Apply domain randomization to a batch of images
        
        Args:
            images: List of input images
            domain_rand_prob: Probability of applying domain randomization
            traditional_aug_prob: Probability of applying traditional augmentations
            
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        for image in images:
            # Apply domain randomization with probability domain_rand_prob
            if random.random() < domain_rand_prob:
                try:
                    # Apply domain randomization
                    randomized_image = self.randomize_domain(image)
                    
                    # Apply traditional augmentations with probability traditional_aug_prob
                    if random.random() < traditional_aug_prob:
                        randomized_image = self.apply_traditional_augmentations(randomized_image)
                    
                    augmented_images.append(randomized_image)
                except Exception as e:
                    logger.warning(f"Domain randomization failed: {str(e)}")
                    # Fall back to traditional augmentation
                    aug_image = self.apply_traditional_augmentations(image)
                    augmented_images.append(aug_image)
            else:
                # Apply only traditional augmentations with probability traditional_aug_prob
                if random.random() < traditional_aug_prob:
                    aug_image = self.apply_traditional_augmentations(image)
                    augmented_images.append(aug_image)
                else:
                    # No augmentation
                    augmented_images.append(image.copy())
        
        return augmented_images


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create domain randomizer
    randomizer = DomainRandomizer(background_dir="data/backgrounds")
    
    # Test with a sample image
    sample_image = np.ones((224, 224, 3), dtype=np.uint8) * 128
    
    # Apply domain randomization
    randomized_image = randomizer.randomize_domain(sample_image)
    
    # Save result
    cv2.imwrite("domain_randomized.jpg", randomized_image)