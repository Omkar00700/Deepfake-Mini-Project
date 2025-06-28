"""
Stratified K-fold cross-validation with group-based splits for deepfake detection
"""

import numpy as np
import logging
import os
import cv2
from typing import Tuple, List, Dict, Optional, Union, Callable
import random
from pathlib import Path
import time
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold
import tensorflow as tf

# Configure logging
logger = logging.getLogger(__name__)

class StratifiedGroupKFoldCV:
    """
    Stratified K-fold cross-validation with group-based splits
    Ensures that samples from the same group (e.g., same video or actor) are not split across folds
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 shuffle: bool = True,
                 random_state: int = 42):
        """
        Initialize stratified group k-fold cross-validation
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle data
            random_state: Random state for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
        # Initialize splitters
        self.stratified_kfold = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        
        self.group_kfold = GroupKFold(n_splits=n_splits)
        
        # If scikit-learn version supports StratifiedGroupKFold, use it
        try:
            self.stratified_group_kfold = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state
            )
            self.has_stratified_group_kfold = True
        except AttributeError:
            self.has_stratified_group_kfold = False
            logger.warning("StratifiedGroupKFold not available in scikit-learn version. "
                          "Using custom implementation.")
        
        logger.info(f"Initialized StratifiedGroupKFoldCV with {n_splits} splits")
    
    def split(self, 
             X: np.ndarray, 
             y: np.ndarray, 
             groups: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Split data into train and validation sets
        
        Args:
            X: Input data
            y: Labels
            groups: Group labels
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        if groups is None:
            # If no groups provided, use standard stratified k-fold
            logger.info("No groups provided, using standard StratifiedKFold")
            return list(self.stratified_kfold.split(X, y))
        
        if self.has_stratified_group_kfold:
            # If scikit-learn version supports StratifiedGroupKFold, use it
            logger.info("Using scikit-learn StratifiedGroupKFold")
            return list(self.stratified_group_kfold.split(X, y, groups))
        else:
            # Otherwise, use custom implementation
            logger.info("Using custom StratifiedGroupKFold implementation")
            return self._custom_stratified_group_kfold(X, y, groups)
    
    def _custom_stratified_group_kfold(self, 
                                     X: np.ndarray, 
                                     y: np.ndarray, 
                                     groups: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Custom implementation of stratified group k-fold
        
        Args:
            X: Input data
            y: Labels
            groups: Group labels
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        # Convert to pandas DataFrame for easier manipulation
        df = pd.DataFrame({
            'index': np.arange(len(X)),
            'y': y,
            'group': groups
        })
        
        # Get unique groups and their label distribution
        group_stats = df.groupby('group')['y'].agg(['mean', 'count'])
        
        # Sort groups by size and label distribution
        group_stats = group_stats.sort_values(['mean', 'count'])
        
        # Assign groups to folds
        group_to_fold = {}
        fold_stats = [{'count': 0, 'pos_count': 0} for _ in range(self.n_splits)]
        
        # Assign each group to the fold with the lowest positive sample count
        for group, stats in group_stats.iterrows():
            count = stats['count']
            pos_count = stats['mean'] * count
            
            # Find fold with lowest positive sample count
            fold_idx = np.argmin([fold['pos_count'] for fold in fold_stats])
            
            # Assign group to fold
            group_to_fold[group] = fold_idx
            
            # Update fold stats
            fold_stats[fold_idx]['count'] += count
            fold_stats[fold_idx]['pos_count'] += pos_count
        
        # Create train/val splits
        splits = []
        
        for fold_idx in range(self.n_splits):
            # Get groups for validation fold
            val_groups = [group for group, fold in group_to_fold.items() if fold == fold_idx]
            
            # Get indices for train and validation
            val_indices = df[df['group'].isin(val_groups)]['index'].values
            train_indices = df[~df['group'].isin(val_groups)]['index'].values
            
            splits.append((train_indices, val_indices))
        
        return splits


class DeepfakeDatasetSplitter:
    """
    Splits deepfake detection dataset with consideration for videos and actors
    """
    
    def __init__(self, 
                 dataset_path: str,
                 n_splits: int = 5,
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialize deepfake dataset splitter
        
        Args:
            dataset_path: Path to dataset
            n_splits: Number of folds
            test_size: Test set size
            random_state: Random state for reproducibility
        """
        self.dataset_path = dataset_path
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        
        # Set random seed
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Initialize splitter
        self.splitter = StratifiedGroupKFoldCV(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
        
        logger.info(f"Initialized DeepfakeDatasetSplitter for {dataset_path}")
    
    def load_dataset_metadata(self) -> pd.DataFrame:
        """
        Load dataset metadata
        
        Returns:
            DataFrame with metadata
        """
        # This is a placeholder - in a real implementation, you would load actual metadata
        # For example, from a CSV file or by scanning the dataset directory
        
        # Check if dataset path exists
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"Dataset path {self.dataset_path} does not exist")
        
        # Scan dataset directory
        real_dir = os.path.join(self.dataset_path, "real")
        fake_dir = os.path.join(self.dataset_path, "fake")
        
        if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
            raise ValueError(f"Dataset should have 'real' and 'fake' subdirectories")
        
        # Get all image files
        real_files = list(Path(real_dir).glob("**/*.jpg")) + list(Path(real_dir).glob("**/*.png"))
        fake_files = list(Path(fake_dir).glob("**/*.jpg")) + list(Path(fake_dir).glob("**/*.png"))
        
        # Extract metadata
        metadata = []
        
        # Process real files
        for file_path in real_files:
            # Extract video ID and actor ID from filename or path
            # This is dataset-specific and should be adapted to your dataset
            file_name = file_path.name
            
            # Example: extract video ID from filename (e.g., video_001_frame_01.jpg)
            video_id = file_name.split("_")[1] if "_" in file_name else "unknown"
            
            # Example: extract actor ID from parent directory
            actor_id = file_path.parent.name if file_path.parent.name != "real" else "unknown"
            
            metadata.append({
                "file_path": str(file_path),
                "label": 0,  # Real
                "video_id": video_id,
                "actor_id": actor_id
            })
        
        # Process fake files
        for file_path in fake_files:
            # Extract video ID and actor ID from filename or path
            file_name = file_path.name
            
            # Example: extract video ID from filename
            video_id = file_name.split("_")[1] if "_" in file_name else "unknown"
            
            # Example: extract actor ID from parent directory
            actor_id = file_path.parent.name if file_path.parent.name != "fake" else "unknown"
            
            metadata.append({
                "file_path": str(file_path),
                "label": 1,  # Fake
                "video_id": video_id,
                "actor_id": actor_id
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(metadata)
        
        logger.info(f"Loaded metadata for {len(df)} samples "
                   f"({len(real_files)} real, {len(fake_files)} fake)")
        
        return df
    
    def split_dataset(self, 
                     metadata: Optional[pd.DataFrame] = None,
                     group_by: str = "video_id") -> Dict[str, Dict[str, List[str]]]:
        """
        Split dataset into train, validation, and test sets
        
        Args:
            metadata: DataFrame with metadata
            group_by: Column to group by ('video_id' or 'actor_id')
            
        Returns:
            Dictionary with train, validation, and test file paths for each fold
        """
        # Load metadata if not provided
        if metadata is None:
            metadata = self.load_dataset_metadata()
        
        # Validate group_by column
        if group_by not in metadata.columns:
            raise ValueError(f"Column {group_by} not found in metadata")
        
        # Split into train+val and test
        unique_groups = metadata[group_by].unique()
        n_groups = len(unique_groups)
        n_test_groups = int(n_groups * self.test_size)
        
        # Shuffle groups
        np.random.shuffle(unique_groups)
        
        # Select test groups
        test_groups = unique_groups[:n_test_groups]
        train_val_groups = unique_groups[n_test_groups:]
        
        # Get test samples
        test_mask = metadata[group_by].isin(test_groups)
        test_metadata = metadata[test_mask].copy()
        train_val_metadata = metadata[~test_mask].copy()
        
        # Reset indices
        train_val_metadata = train_val_metadata.reset_index(drop=True)
        
        # Get labels and groups for stratified group k-fold
        X = np.arange(len(train_val_metadata))
        y = train_val_metadata["label"].values
        groups = train_val_metadata[group_by].values
        
        # Split into folds
        splits = self.splitter.split(X, y, groups)
        
        # Create result dictionary
        result = {}
        
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            # Get file paths for train and validation
            train_files = train_val_metadata.iloc[train_indices]["file_path"].tolist()
            val_files = train_val_metadata.iloc[val_indices]["file_path"].tolist()
            test_files = test_metadata["file_path"].tolist()
            
            # Get labels
            train_labels = train_val_metadata.iloc[train_indices]["label"].tolist()
            val_labels = train_val_metadata.iloc[val_indices]["label"].tolist()
            test_labels = test_metadata["label"].tolist()
            
            # Store in result dictionary
            result[f"fold_{fold_idx}"] = {
                "train_files": train_files,
                "train_labels": train_labels,
                "val_files": val_files,
                "val_labels": val_labels,
                "test_files": test_files,
                "test_labels": test_labels
            }
        
        logger.info(f"Split dataset into {self.n_splits} folds")
        logger.info(f"Test set: {len(test_files)} samples")
        logger.info(f"Average train set: {np.mean([len(result[f]['train_files']) for f in result])} samples")
        logger.info(f"Average validation set: {np.mean([len(result[f]['val_files']) for f in result])} samples")
        
        return result
    
    def create_tf_datasets(self, 
                         split_data: Dict[str, Dict[str, List[str]]],
                         batch_size: int = 32,
                         preprocessing_fn: Optional[Callable] = None) -> Dict[str, Dict[str, tf.data.Dataset]]:
        """
        Create TensorFlow datasets for each fold
        
        Args:
            split_data: Split data from split_dataset
            batch_size: Batch size
            preprocessing_fn: Preprocessing function
            
        Returns:
            Dictionary with TensorFlow datasets for each fold
        """
        result = {}
        
        for fold_name, fold_data in split_data.items():
            # Create datasets for train, validation, and test
            train_ds = self._create_dataset(
                fold_data["train_files"],
                fold_data["train_labels"],
                batch_size,
                preprocessing_fn,
                shuffle=True
            )
            
            val_ds = self._create_dataset(
                fold_data["val_files"],
                fold_data["val_labels"],
                batch_size,
                preprocessing_fn,
                shuffle=False
            )
            
            test_ds = self._create_dataset(
                fold_data["test_files"],
                fold_data["test_labels"],
                batch_size,
                preprocessing_fn,
                shuffle=False
            )
            
            result[fold_name] = {
                "train": train_ds,
                "val": val_ds,
                "test": test_ds
            }
        
        return result
    
    def _create_dataset(self, 
                      file_paths: List[str],
                      labels: List[int],
                      batch_size: int,
                      preprocessing_fn: Optional[Callable],
                      shuffle: bool) -> tf.data.Dataset:
        """
        Create TensorFlow dataset
        
        Args:
            file_paths: List of file paths
            labels: List of labels
            batch_size: Batch size
            preprocessing_fn: Preprocessing function
            shuffle: Whether to shuffle data
            
        Returns:
            TensorFlow dataset
        """
        # Create dataset from file paths and labels
        ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        
        # Map file paths to images
        ds = ds.map(
            lambda file_path, label: (self._load_image(file_path), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply preprocessing if provided
        if preprocessing_fn is not None:
            ds = ds.map(
                lambda image, label: (preprocessing_fn(image), label),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Shuffle if needed
        if shuffle:
            ds = ds.shuffle(buffer_size=len(file_paths))
        
        # Batch and prefetch
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return ds
    
    def _load_image(self, file_path: tf.Tensor) -> tf.Tensor:
        """
        Load image from file path
        
        Args:
            file_path: File path
            
        Returns:
            Image tensor
        """
        # Read file
        file_content = tf.io.read_file(file_path)
        
        # Decode image
        image = tf.image.decode_image(
            file_content,
            channels=3,
            expand_animations=False
        )
        
        # Set shape
        image.set_shape([None, None, 3])
        
        return image


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create dataset splitter
    splitter = DeepfakeDatasetSplitter(
        dataset_path="/path/to/dataset",
        n_splits=5
    )
    
    # Split dataset
    try:
        split_data = splitter.split_dataset(group_by="video_id")
        
        # Print statistics
        for fold_name, fold_data in split_data.items():
            print(f"{fold_name}:")
            print(f"  Train: {len(fold_data['train_files'])} samples")
            print(f"  Validation: {len(fold_data['val_files'])} samples")
            print(f"  Test: {len(fold_data['test_files'])} samples")
    except ValueError as e:
        print(f"Error: {e}")