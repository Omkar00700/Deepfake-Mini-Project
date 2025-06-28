"""
Advanced Data Augmentation for Deepfake Detection
Implements specialized augmentations to improve model robustness and accuracy
"""

import os
import numpy as np
import cv2
import tensorflow as tf
import logging
from typing import List, Dict, Tuple, Optional, Union, Callable
import random
import albumentations as A
from pathlib import Path
import json
import time
from PIL import Image, ImageFilter, ImageEnhance
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from augmentation.gan_augmentation import StyleGAN2Augmenter

# Configure logging
logger = logging.getLogger(__name__)

class DeepfakeAugmenter:
    """
    Advanced data augmentation specifically designed for deepfake detection
    Implements specialized techniques to improve model robustness
    """
    
    def __init__(self, 
                 cache_dir: str = "cache/augmentation",
                 use_gan: bool = True,
                 use_mixup: bool = True,
                 use_cutmix: bool = True,
                 use_domain_randomization: bool = True):
        """
        Initialize the deepfake augmenter
        
        Args:
            cache_dir: Directory to cache augmented images
            use_gan: Whether to use GAN-based augmentation
            use_mixup: Whether to use mixup augmentation
            use_cutmix: Whether to use cutmix augmentation
            use_domain_randomization: Whether to use domain randomization
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.use_gan = use_gan
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.use_domain_randomization = use_domain_randomization
        
        # Initialize GAN augmenter if enabled
        self.gan_augmenter = None
        if use_gan:
            try:
                self.gan_augmenter = StyleGAN2Augmenter(cache_dir=os.path.join(cache_dir, "stylegan2"))
                logger.info("GAN augmenter initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize GAN augmenter: {str(e)}")
                self.use_gan = False
        
        # Initialize basic augmentation pipeline
        self.basic_augmentation = A.Compose([
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            
            # Color transformations
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            
            # Blur and noise
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MotionBlur(blur_limit=(3, 7), p=0.5),
            ], p=0.3),
            
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            ], p=0.3),
            
            # Compression artifacts
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
        ])
        
        # Initialize deepfake-specific augmentation pipeline
        self.deepfake_augmentation = A.Compose([
            # Facial region manipulations
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, min_height=8, min_width=8, p=0.3),
            
            # Compression artifacts (common in deepfakes)
            A.OneOf([
                A.JpegCompression(quality_lower=50, quality_upper=85, p=0.5),
                A.Downscale(scale_min=0.7, scale_max=0.9, p=0.5),
            ], p=0.5),
            
            # Blur (often present in deepfakes)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 9), p=0.5),
                A.MotionBlur(blur_limit=(3, 9), p=0.5),
                A.MedianBlur(blur_limit=5, p=0.5),
            ], p=0.5),
            
            # Color inconsistencies
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.ToSepia(p=0.3),
                A.ChannelShuffle(p=0.3),
            ], p=0.3),
            
            # Noise patterns
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 80.0), p=0.5),
                A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            ], p=0.4),
        ])
        
        # Initialize domain randomization
        self.domain_randomization = A.Compose([
            # Background variations
            A.OneOf([
                A.ToGray(p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            ], p=0.5),
            
            # Lighting variations
            A.OneOf([
                A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.5),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1, num_flare_circles_lower=1, num_flare_circles_upper=3, src_radius=100, src_color=(255, 255, 255), p=0.3),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.2),
            ], p=0.3),
            
            # Texture and pattern variations
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                A.OpticalDistortion(distort_limit=0.2, shift_limit=0.05, p=0.3),
            ], p=0.2),
        ])
        
        logger.info("DeepfakeAugmenter initialized with advanced augmentation pipelines")
    
    def augment_image(self, 
                     image: np.ndarray, 
                     is_fake: bool = False,
                     augmentation_strength: float = 1.0) -> np.ndarray:
        """
        Apply augmentation to a single image
        
        Args:
            image: Input image
            is_fake: Whether the image is a deepfake
            augmentation_strength: Strength of augmentation (0.0-1.0)
            
        Returns:
            Augmented image
        """
        # Apply basic augmentation
        augmented = self.basic_augmentation(image=image)["image"]
        
        # Apply deepfake-specific augmentation if it's a fake image
        if is_fake and random.random() < augmentation_strength:
            augmented = self.deepfake_augmentation(image=augmented)["image"]
        
        # Apply domain randomization
        if self.use_domain_randomization and random.random() < 0.3 * augmentation_strength:
            augmented = self.domain_randomization(image=augmented)["image"]
        
        return augmented
    
    def augment_batch(self, 
                     images: np.ndarray, 
                     labels: np.ndarray,
                     augmentation_strength: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation to a batch of images
        
        Args:
            images: Batch of images
            labels: Batch of labels (0 for real, 1 for fake)
            augmentation_strength: Strength of augmentation (0.0-1.0)
            
        Returns:
            Tuple of (augmented_images, augmented_labels)
        """
        augmented_images = []
        augmented_labels = []
        
        # Apply standard augmentation to each image
        for i, (image, label) in enumerate(zip(images, labels)):
            # Apply basic augmentation
            augmented = self.augment_image(
                image, 
                is_fake=(label > 0.5),
                augmentation_strength=augmentation_strength
            )
            
            augmented_images.append(augmented)
            augmented_labels.append(label)
        
        # Apply mixup if enabled
        if self.use_mixup and random.random() < 0.3 * augmentation_strength:
            mixed_images, mixed_labels = self._apply_mixup(
                np.array(augmented_images), 
                np.array(augmented_labels)
            )
            augmented_images.extend(mixed_images)
            augmented_labels.extend(mixed_labels)
        
        # Apply cutmix if enabled
        if self.use_cutmix and random.random() < 0.3 * augmentation_strength:
            mixed_images, mixed_labels = self._apply_cutmix(
                np.array(augmented_images), 
                np.array(augmented_labels)
            )
            augmented_images.extend(mixed_images)
            augmented_labels.extend(mixed_labels)
        
        # Apply GAN augmentation if enabled
        if self.use_gan and self.gan_augmenter and random.random() < 0.2 * augmentation_strength:
            gan_images, gan_labels = self._apply_gan_augmentation(
                np.array(augmented_images), 
                np.array(augmented_labels)
            )
            augmented_images.extend(gan_images)
            augmented_labels.extend(gan_labels)
        
        return np.array(augmented_images), np.array(augmented_labels)
    
    def _apply_mixup(self, 
                    images: np.ndarray, 
                    labels: np.ndarray,
                    alpha: float = 0.2,
                    num_to_generate: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mixup augmentation
        
        Args:
            images: Batch of images
            labels: Batch of labels
            alpha: Mixup alpha parameter
            num_to_generate: Number of mixed samples to generate
            
        Returns:
            Tuple of (mixed_images, mixed_labels)
        """
        if num_to_generate is None:
            num_to_generate = max(1, int(len(images) * 0.3))
        
        mixed_images = []
        mixed_labels = []
        
        for _ in range(num_to_generate):
            # Select two random images
            idx1, idx2 = np.random.choice(len(images), 2, replace=False)
            
            # Generate mixup weight
            lam = np.random.beta(alpha, alpha)
            
            # Mix images
            mixed_image = lam * images[idx1] + (1 - lam) * images[idx2]
            mixed_image = mixed_image.astype(np.uint8)
            
            # Mix labels
            mixed_label = lam * labels[idx1] + (1 - lam) * labels[idx2]
            
            mixed_images.append(mixed_image)
            mixed_labels.append(mixed_label)
        
        return mixed_images, mixed_labels
    
    def _apply_cutmix(self, 
                     images: np.ndarray, 
                     labels: np.ndarray,
                     alpha: float = 0.2,
                     num_to_generate: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply cutmix augmentation
        
        Args:
            images: Batch of images
            labels: Batch of labels
            alpha: CutMix alpha parameter
            num_to_generate: Number of mixed samples to generate
            
        Returns:
            Tuple of (mixed_images, mixed_labels)
        """
        if num_to_generate is None:
            num_to_generate = max(1, int(len(images) * 0.3))
        
        mixed_images = []
        mixed_labels = []
        
        for _ in range(num_to_generate):
            # Select two random images
            idx1, idx2 = np.random.choice(len(images), 2, replace=False)
            
            # Get image dimensions
            h, w = images[idx1].shape[0], images[idx1].shape[1]
            
            # Generate random box
            lam = np.random.beta(alpha, alpha)
            
            # Calculate box size
            cut_rat = np.sqrt(1.0 - lam)
            cut_w = int(w * cut_rat)
            cut_h = int(h * cut_rat)
            
            # Calculate box center
            cx = np.random.randint(w)
            cy = np.random.randint(h)
            
            # Calculate box boundaries
            bbx1 = np.clip(cx - cut_w // 2, 0, w)
            bby1 = np.clip(cy - cut_h // 2, 0, h)
            bbx2 = np.clip(cx + cut_w // 2, 0, w)
            bby2 = np.clip(cy + cut_h // 2, 0, h)
            
            # Create mixed image
            mixed_image = images[idx1].copy()
            mixed_image[bby1:bby2, bbx1:bbx2] = images[idx2][bby1:bby2, bbx1:bbx2]
            
            # Calculate area ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
            
            # Mix labels
            mixed_label = lam * labels[idx1] + (1 - lam) * labels[idx2]
            
            mixed_images.append(mixed_image)
            mixed_labels.append(mixed_label)
        
        return mixed_images, mixed_labels
    
    def _apply_gan_augmentation(self, 
                               images: np.ndarray, 
                               labels: np.ndarray,
                               num_to_generate: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply GAN-based augmentation
        
        Args:
            images: Batch of images
            labels: Batch of labels
            num_to_generate: Number of GAN samples to generate
            
        Returns:
            Tuple of (gan_images, gan_labels)
        """
        if not self.gan_augmenter:
            return [], []
        
        if num_to_generate is None:
            num_to_generate = max(1, int(len(images) * 0.2))
        
        gan_images = []
        gan_labels = []
        
        try:
            # Generate synthetic faces
            for _ in range(num_to_generate // 2):
                # Generate a synthetic face
                synthetic_face = self.gan_augmenter.generate_synthetic_face()
                
                # Apply additional augmentation
                synthetic_face = self.deepfake_augmentation(image=synthetic_face)["image"]
                
                gan_images.append(synthetic_face)
                gan_labels.append(1.0)  # Synthetic faces are labeled as fake
            
            # Generate face swaps
            for _ in range(num_to_generate - len(gan_images)):
                # Select a random real face
                real_idx = np.random.choice(np.where(labels < 0.5)[0])
                
                # Perform face swap
                swapped_face = self.gan_augmenter.face_swap(
                    source_face=images[real_idx],
                    blend_factor=random.uniform(0.7, 0.9)
                )
                
                gan_images.append(swapped_face)
                gan_labels.append(1.0)  # Swapped faces are labeled as fake
        
        except Exception as e:
            logger.error(f"Error in GAN augmentation: {str(e)}")
        
        return gan_images, gan_labels
    
    def generate_hard_examples(self, 
                              images: np.ndarray, 
                              labels: np.ndarray,
                              model: tf.keras.Model,
                              num_to_generate: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate hard examples that the model struggles with
        
        Args:
            images: Batch of images
            labels: Batch of labels
            model: Model to evaluate difficulty
            num_to_generate: Number of hard examples to generate
            
        Returns:
            Tuple of (hard_images, hard_labels)
        """
        hard_images = []
        hard_labels = []
        
        # Get model predictions
        predictions = model.predict(images)
        if predictions.shape[-1] == 1:
            predictions = predictions.flatten()
        
        # Calculate difficulty scores (how close prediction is to decision boundary)
        difficulties = 1.0 - 2.0 * np.abs(predictions - 0.5)
        
        # Sort by difficulty
        sorted_indices = np.argsort(difficulties)[::-1]  # Descending order
        
        # Select hardest examples
        hard_indices = sorted_indices[:num_to_generate]
        
        # Generate augmented versions of hard examples
        for idx in hard_indices:
            # Apply strong augmentation
            augmented = self.augment_image(
                images[idx], 
                is_fake=(labels[idx] > 0.5),
                augmentation_strength=1.0
            )
            
            hard_images.append(augmented)
            hard_labels.append(labels[idx])
        
        return np.array(hard_images), np.array(hard_labels)
    
    def apply_style_transfer(self, 
                            content_image: np.ndarray, 
                            style_image: np.ndarray,
                            alpha: float = 0.5) -> np.ndarray:
        """
        Apply simple style transfer between images
        
        Args:
            content_image: Content image
            style_image: Style image
            alpha: Style weight
            
        Returns:
            Styled image
        """
        # Convert to PIL images
        content_pil = Image.fromarray(content_image)
        style_pil = Image.fromarray(style_image)
        
        # Resize style image to match content
        style_pil = style_pil.resize(content_pil.size)
        
        # Convert to arrays
        content_array = np.array(content_pil).astype(np.float32)
        style_array = np.array(style_pil).astype(np.float32)
        
        # Simple color transfer
        content_mean = content_array.mean(axis=(0, 1))
        content_std = content_array.std(axis=(0, 1))
        style_mean = style_array.mean(axis=(0, 1))
        style_std = style_array.std(axis=(0, 1))
        
        # Normalize content image
        normalized = (content_array - content_mean) / (content_std + 1e-7)
        
        # Apply style statistics
        styled = normalized * (style_std * alpha + content_std * (1 - alpha)) + \
                (style_mean * alpha + content_mean * (1 - alpha))
        
        # Clip to valid range
        styled = np.clip(styled, 0, 255).astype(np.uint8)
        
        return styled
    
    def create_augmentation_pipeline(self, is_training: bool = True) -> Callable:
        """
        Create an augmentation pipeline function
        
        Args:
            is_training: Whether this is for training or validation
            
        Returns:
            Augmentation function
        """
        def augment_fn(images, labels):
            if is_training:
                return self.augment_batch(images, labels)
            else:
                # For validation, only apply basic augmentation
                augmented_images = []
                for image in images:
                    augmented = self.basic_augmentation(image=image)["image"]
                    augmented_images.append(augmented)
                return np.array(augmented_images), labels
        
        return augment_fn