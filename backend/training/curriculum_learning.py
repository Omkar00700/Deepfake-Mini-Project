"""
Curriculum learning implementation for deepfake detection
Implements progressive training from easy to hard samples
"""

import tensorflow as tf
import numpy as np
import logging
import os
import cv2
from typing import Tuple, List, Dict, Optional, Union, Callable
import random
from pathlib import Path
import time

# Configure logging
logger = logging.getLogger(__name__)

class SampleDifficulty:
    """
    Utility class for assessing sample difficulty
    """
    
    @staticmethod
    def assess_image_quality(image: np.ndarray) -> float:
        """
        Assess image quality to determine difficulty
        Higher quality = easier sample
        
        Args:
            image: Input image
            
        Returns:
            Quality score (0.0-1.0)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate contrast
        contrast = gray.std()
        
        # Normalize scores
        sharpness_score = min(1.0, sharpness / 1000)
        contrast_score = min(1.0, contrast / 80)
        
        # Combine scores
        quality_score = 0.6 * sharpness_score + 0.4 * contrast_score
        
        return quality_score
    
    @staticmethod
    def assess_compression_level(image: np.ndarray) -> float:
        """
        Assess compression level to determine difficulty
        Higher compression = harder sample
        
        Args:
            image: Input image
            
        Returns:
            Compression score (0.0-1.0), higher means more compression
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply DCT to detect compression artifacts
        h, w = gray.shape
        h_8 = h // 8 * 8
        w_8 = w // 8 * 8
        gray = gray[:h_8, :w_8]
        
        # Compute DCT coefficients
        dct_blocks = []
        for i in range(0, h_8, 8):
            for j in range(0, w_8, 8):
                block = gray[i:i+8, j:j+8]
                dct_block = cv2.dct(block.astype(np.float32))
                dct_blocks.append(dct_block)
        
        # Analyze DCT coefficients
        high_freq_energy = 0
        total_energy = 0
        
        for block in dct_blocks:
            # High frequency components (bottom right of DCT block)
            high_freq = block[4:, 4:]
            high_freq_energy += np.sum(np.abs(high_freq))
            
            # Total energy
            total_energy += np.sum(np.abs(block))
        
        # Calculate compression score
        if total_energy > 0:
            compression_score = 1.0 - (high_freq_energy / total_energy)
        else:
            compression_score = 0.5
        
        return compression_score
    
    @staticmethod
    def assess_face_size(face_bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int]) -> float:
        """
        Assess face size to determine difficulty
        Smaller faces = harder samples
        
        Args:
            face_bbox: Face bounding box (x, y, w, h)
            image_shape: Image shape (height, width)
            
        Returns:
            Size score (0.0-1.0)
        """
        x, y, w, h = face_bbox
        image_height, image_width = image_shape
        
        # Calculate face area ratio
        face_area = w * h
        image_area = image_height * image_width
        
        # Normalize score
        size_score = min(1.0, face_area / (image_area * 0.5))
        
        return size_score
    
    @staticmethod
    def assess_sample_difficulty(image: np.ndarray, 
                               face_bbox: Optional[Tuple[int, int, int, int]] = None,
                               is_fake: bool = False) -> float:
        """
        Assess overall sample difficulty
        
        Args:
            image: Input image
            face_bbox: Face bounding box (x, y, w, h)
            is_fake: Whether the sample is fake
            
        Returns:
            Difficulty score (0.0-1.0), higher means more difficult
        """
        # Assess image quality
        quality_score = SampleDifficulty.assess_image_quality(image)
        
        # Assess compression level
        compression_score = SampleDifficulty.assess_compression_level(image)
        
        # Assess face size if bbox is provided
        if face_bbox is not None:
            size_score = SampleDifficulty.assess_face_size(face_bbox, image.shape[:2])
        else:
            size_score = 0.5  # Default value
        
        # Combine scores
        # For fake samples, high quality fakes are more difficult
        if is_fake:
            difficulty_score = 0.4 * quality_score + 0.3 * compression_score + 0.3 * (1.0 - size_score)
        else:
            difficulty_score = 0.4 * (1.0 - quality_score) + 0.3 * compression_score + 0.3 * (1.0 - size_score)
        
        return difficulty_score


class CurriculumSampler:
    """
    Curriculum sampler for progressive training
    """
    
    def __init__(self, 
                 difficulty_fn: Callable[[np.ndarray, Optional[Tuple[int, int, int, int]], bool], float] = None,
                 num_stages: int = 3,
                 stage_epochs: List[int] = None,
                 difficulty_thresholds: List[float] = None):
        """
        Initialize curriculum sampler
        
        Args:
            difficulty_fn: Function to assess sample difficulty
            num_stages: Number of curriculum stages
            stage_epochs: List of epochs for each stage
            difficulty_thresholds: List of difficulty thresholds for each stage
        """
        self.difficulty_fn = difficulty_fn or SampleDifficulty.assess_sample_difficulty
        self.num_stages = num_stages
        
        # Set default stage epochs if not provided
        if stage_epochs is None:
            self.stage_epochs = [10, 20, 30]  # Default epochs for each stage
        else:
            self.stage_epochs = stage_epochs
        
        # Set default difficulty thresholds if not provided
        if difficulty_thresholds is None:
            self.difficulty_thresholds = [0.3, 0.6, 1.0]  # Default thresholds
        else:
            self.difficulty_thresholds = difficulty_thresholds
        
        # Current stage
        self.current_stage = 0
        self.current_epoch = 0
        
        logger.info(f"Initialized curriculum sampler with {num_stages} stages")
        logger.info(f"Stage epochs: {self.stage_epochs}")
        logger.info(f"Difficulty thresholds: {self.difficulty_thresholds}")
    
    def update_stage(self, epoch: int) -> int:
        """
        Update curriculum stage based on current epoch
        
        Args:
            epoch: Current epoch
            
        Returns:
            Current stage
        """
        self.current_epoch = epoch
        
        # Determine current stage
        for i, stage_epoch in enumerate(self.stage_epochs):
            if epoch < stage_epoch:
                self.current_stage = i
                break
        else:
            # If we've passed all stage epochs, use the final stage
            self.current_stage = self.num_stages - 1
        
        logger.info(f"Curriculum updated to stage {self.current_stage} at epoch {epoch}")
        logger.info(f"Current difficulty threshold: {self.difficulty_thresholds[self.current_stage]}")
        
        return self.current_stage
    
    def filter_samples(self, 
                      images: List[np.ndarray],
                      labels: List[int],
                      face_bboxes: Optional[List[Tuple[int, int, int, int]]] = None) -> Tuple[List[np.ndarray], List[int]]:
        """
        Filter samples based on current curriculum stage
        
        Args:
            images: List of input images
            labels: List of labels (0 for real, 1 for fake)
            face_bboxes: List of face bounding boxes
            
        Returns:
            Filtered images and labels
        """
        # Get current difficulty threshold
        threshold = self.difficulty_thresholds[self.current_stage]
        
        filtered_images = []
        filtered_labels = []
        
        for i, (image, label) in enumerate(zip(images, labels)):
            # Get face bbox if available
            face_bbox = face_bboxes[i] if face_bboxes is not None else None
            
            # Assess sample difficulty
            difficulty = self.difficulty_fn(image, face_bbox, label == 1)
            
            # Filter based on difficulty threshold
            if difficulty <= threshold:
                filtered_images.append(image)
                filtered_labels.append(label)
        
        logger.debug(f"Filtered {len(filtered_images)}/{len(images)} samples for stage {self.current_stage}")
        
        return filtered_images, filtered_labels
    
    def get_sampling_weights(self,
                           images: List[np.ndarray],
                           labels: List[int],
                           face_bboxes: Optional[List[Tuple[int, int, int, int]]] = None) -> np.ndarray:
        """
        Get sampling weights based on current curriculum stage
        
        Args:
            images: List of input images
            labels: List of labels (0 for real, 1 for fake)
            face_bboxes: List of face bounding boxes
            
        Returns:
            Array of sampling weights
        """
        # Get current difficulty threshold
        threshold = self.difficulty_thresholds[self.current_stage]
        
        weights = []
        
        for i, (image, label) in enumerate(zip(images, labels)):
            # Get face bbox if available
            face_bbox = face_bboxes[i] if face_bboxes is not None else None
            
            # Assess sample difficulty
            difficulty = self.difficulty_fn(image, face_bbox, label == 1)
            
            # Calculate weight based on difficulty and current threshold
            if difficulty <= threshold:
                # For samples within threshold, weight by difficulty
                # Harder samples (closer to threshold) get higher weight
                weight = difficulty / threshold
            else:
                # For samples beyond threshold, assign low weight
                weight = 0.1
            
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        return weights


class CurriculumDataGenerator(tf.keras.utils.Sequence):
    """
    Data generator with curriculum learning
    """
    
    def __init__(self,
                images: List[np.ndarray],
                labels: List[int],
                face_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
                batch_size: int = 32,
                shuffle: bool = True,
                curriculum_sampler: Optional[CurriculumSampler] = None,
                preprocessing_fn: Optional[Callable] = None):
        """
        Initialize curriculum data generator
        
        Args:
            images: List of input images
            labels: List of labels (0 for real, 1 for fake)
            face_bboxes: List of face bounding boxes
            batch_size: Batch size
            shuffle: Whether to shuffle data
            curriculum_sampler: Curriculum sampler
            preprocessing_fn: Preprocessing function
        """
        self.images = images
        self.labels = labels
        self.face_bboxes = face_bboxes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.curriculum_sampler = curriculum_sampler or CurriculumSampler()
        self.preprocessing_fn = preprocessing_fn
        
        # Initial indices
        self.indices = np.arange(len(images))
        
        # Shuffle if needed
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Get number of batches per epoch"""
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __getitem__(self, idx):
        """Get batch at index idx"""
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Get batch data
        batch_images = [self.images[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]
        
        # Get face bboxes if available
        if self.face_bboxes is not None:
            batch_bboxes = [self.face_bboxes[i] for i in batch_indices]
        else:
            batch_bboxes = None
        
        # Apply curriculum filtering
        filtered_images, filtered_labels = self.curriculum_sampler.filter_samples(
            batch_images, batch_labels, batch_bboxes
        )
        
        # If filtering removed all samples, use original batch
        if len(filtered_images) == 0:
            filtered_images = batch_images
            filtered_labels = batch_labels
        
        # Convert to numpy arrays
        x = np.array(filtered_images)
        y = np.array(filtered_labels)
        
        # Apply preprocessing if provided
        if self.preprocessing_fn is not None:
            x = self.preprocessing_fn(x)
        
        return x, y
    
    def on_epoch_end(self):
        """Called at the end of each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def update_curriculum(self, epoch: int):
        """
        Update curriculum stage
        
        Args:
            epoch: Current epoch
        """
        self.curriculum_sampler.update_stage(epoch)


class CurriculumCallback(tf.keras.callbacks.Callback):
    """
    Callback for curriculum learning
    """
    
    def __init__(self, data_generator: CurriculumDataGenerator):
        """
        Initialize curriculum callback
        
        Args:
            data_generator: Curriculum data generator
        """
        super(CurriculumCallback, self).__init__()
        self.data_generator = data_generator
    
    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the beginning of each epoch
        
        Args:
            epoch: Current epoch
            logs: Training logs
        """
        self.data_generator.update_curriculum(epoch)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data
    num_samples = 100
    images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(num_samples)]
    labels = [random.randint(0, 1) for _ in range(num_samples)]
    
    # Create curriculum sampler
    sampler = CurriculumSampler(
        num_stages=3,
        stage_epochs=[5, 10, 15],
        difficulty_thresholds=[0.3, 0.6, 1.0]
    )
    
    # Create data generator
    data_gen = CurriculumDataGenerator(
        images=images,
        labels=labels,
        batch_size=16,
        curriculum_sampler=sampler
    )
    
    # Create callback
    callback = CurriculumCallback(data_gen)
    
    # Test data generator
    for epoch in range(20):
        sampler.update_stage(epoch)
        print(f"Epoch {epoch}, Stage {sampler.current_stage}")
        
        for i in range(len(data_gen)):
            x, y = data_gen[i]
            print(f"  Batch {i}: {x.shape}, {y.shape}")