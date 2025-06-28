"""
Multi-Modal Fusion module for deepfake detection
Combines visual and audio features for improved detection
"""

import os
import logging
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Input, Dropout, LSTM, Bidirectional,
    BatchNormalization, Conv1D, MaxPooling1D, Flatten, Concatenate
)
import librosa
import cv2
from typing import Tuple, List, Dict, Any, Optional, Union
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class AudioFeatureExtractor:
    """
    Extract audio features from video files
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_mfcc: int = 40,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 max_audio_length: float = 10.0):
        """
        Initialize the audio feature extractor
        
        Args:
            sample_rate: Sample rate for audio processing
            n_mfcc: Number of MFCC features to extract
            n_fft: FFT window size
            hop_length: Hop length for STFT
            max_audio_length: Maximum audio length in seconds
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_audio_length = max_audio_length
        self.max_frames = int(max_audio_length * sample_rate / hop_length)
        
        logger.info(f"Initialized audio feature extractor with {n_mfcc} MFCC features")
    
    def extract_from_video(self, video_path: str) -> np.ndarray:
        """
        Extract audio features from a video file
        
        Args:
            video_path: Path to the video file
            
        Returns:
            MFCC features
        """
        try:
            # Extract audio from video
            audio_path = self._extract_audio(video_path)
            
            # Extract features from audio
            features = self.extract_from_audio(audio_path)
            
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return features
        except Exception as e:
            logger.error(f"Error extracting audio features from video: {str(e)}")
            # Return empty features
            return np.zeros((self.max_frames, self.n_mfcc))
    
    def _extract_audio(self, video_path: str) -> str:
        """
        Extract audio from a video file
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to the extracted audio file
        """
        # Create temporary audio file
        audio_path = os.path.splitext(video_path)[0] + "_temp_audio.wav"
        
        # Extract audio using ffmpeg
        os.system(f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y -loglevel error")
        
        return audio_path
    
    def extract_from_audio(self, audio_path: str) -> np.ndarray:
        """
        Extract MFCC features from an audio file
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            MFCC features
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.max_audio_length)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length
            )
            
            # Transpose to (time, features)
            mfcc = mfcc.T
            
            # Pad or truncate to max_frames
            if mfcc.shape[0] < self.max_frames:
                # Pad with zeros
                pad_width = self.max_frames - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))
            elif mfcc.shape[0] > self.max_frames:
                # Truncate
                mfcc = mfcc[:self.max_frames, :]
            
            return mfcc
        except Exception as e:
            logger.error(f"Error extracting MFCC features: {str(e)}")
            # Return empty features
            return np.zeros((self.max_frames, self.n_mfcc))
    
    def extract_from_frames(self, frames: List[np.ndarray], fps: float = 30.0) -> np.ndarray:
        """
        Extract audio features from video frames
        
        Args:
            frames: List of video frames
            fps: Frames per second
            
        Returns:
            MFCC features (empty array as frames don't contain audio)
        """
        logger.warning("Cannot extract audio features from frames without audio")
        return np.zeros((self.max_frames, self.n_mfcc))


class AudioModel:
    """
    Audio model for deepfake detection
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int] = (313, 40),  # (time_steps, n_mfcc)
                 use_lstm: bool = True,
                 lstm_units: int = 64,
                 dropout_rate: float = 0.5):
        """
        Initialize the audio model
        
        Args:
            input_shape: Shape of the input MFCC features
            use_lstm: Whether to use LSTM layers
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate
        """
        self.input_shape = input_shape
        self.use_lstm = use_lstm
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        
        logger.info(f"Initialized audio model with input shape {input_shape}")
    
    def build_model(self) -> Model:
        """
        Build the audio model
        
        Returns:
            Audio model
        """
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        if self.use_lstm:
            # LSTM-based model
            x = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(inputs)
            x = Bidirectional(LSTM(self.lstm_units))(x)
        else:
            # CNN-based model
            x = Conv1D(64, kernel_size=3, activation='relu')(inputs)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=2)(x)
            
            x = Conv1D(128, kernel_size=3, activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=2)(x)
            
            x = Conv1D(256, kernel_size=3, activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=2)(x)
            
            x = Flatten()(x)
        
        # Dense layers
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = Dense(128, activation='relu')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name="audio_model")
        
        logger.info(f"Built audio model with {'LSTM' if self.use_lstm else 'CNN'} architecture")
        return self.model


class MultiModalFusion:
    """
    Multi-modal fusion for deepfake detection
    """
    
    def __init__(self, 
                 visual_model: Model,
                 audio_input_shape: Tuple[int, int] = (313, 40),
                 fusion_type: str = "concat",
                 use_lstm_audio: bool = True,
                 dropout_rate: float = 0.5):
        """
        Initialize the multi-modal fusion model
        
        Args:
            visual_model: Visual model for feature extraction
            audio_input_shape: Shape of the audio input features
            fusion_type: Type of fusion ('concat', 'attention', or 'weighted')
            use_lstm_audio: Whether to use LSTM for audio processing
            dropout_rate: Dropout rate
        """
        self.visual_model = visual_model
        self.audio_input_shape = audio_input_shape
        self.fusion_type = fusion_type
        self.use_lstm_audio = use_lstm_audio
        self.dropout_rate = dropout_rate
        self.model = None
        self.audio_model = None
        self.audio_extractor = AudioFeatureExtractor(n_mfcc=audio_input_shape[1])
        
        logger.info(f"Initialized multi-modal fusion with {fusion_type} fusion")
    
    def build_model(self) -> Model:
        """
        Build the multi-modal fusion model
        
        Returns:
            Multi-modal fusion model
        """
        # Build audio model
        audio_model = AudioModel(
            input_shape=self.audio_input_shape,
            use_lstm=self.use_lstm_audio,
            dropout_rate=self.dropout_rate
        )
        self.audio_model = audio_model.build_model()
        
        # Visual input
        visual_input = self.visual_model.input
        visual_features = self.visual_model.layers[-2].output  # Get features before the final layer
        
        # Audio input
        audio_input = Input(shape=self.audio_input_shape, name="audio_input")
        audio_features = self.audio_model(audio_input)
        
        # Fusion
        if self.fusion_type == "concat":
            # Simple concatenation
            fused_features = Concatenate()([visual_features, audio_features])
        elif self.fusion_type == "attention":
            # Attention-based fusion
            attention = Dense(1, activation='sigmoid')(Concatenate()([visual_features, audio_features]))
            visual_weighted = tf.multiply(visual_features, attention)
            audio_weighted = tf.multiply(audio_features, 1 - attention)
            fused_features = Concatenate()([visual_weighted, audio_weighted])
        elif self.fusion_type == "weighted":
            # Weighted sum
            visual_weight = Dense(1, activation='sigmoid')(visual_features)
            audio_weight = Dense(1, activation='sigmoid')(audio_features)
            
            # Normalize weights
            total_weight = tf.add(visual_weight, audio_weight)
            visual_weight = tf.divide(visual_weight, total_weight)
            audio_weight = tf.divide(audio_weight, total_weight)
            
            # Weighted features
            visual_weighted = tf.multiply(visual_features, visual_weight)
            audio_weighted = tf.multiply(audio_features, audio_weight)
            
            # Concatenate weighted features
            fused_features = Concatenate()([visual_weighted, audio_weighted])
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
        
        # Final classification layers
        x = Dense(512, activation='relu')(fused_features)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate * 0.5)(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid')(x)
        
        # Create model
        self.model = Model(inputs=[visual_input, audio_input], outputs=outputs, name="multimodal_model")
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        logger.info(f"Built multi-modal fusion model with {self.fusion_type} fusion")
        return self.model
    
    def process_video(self, video_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a video file to extract visual and audio features
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Visual frames and audio features
        """
        # Extract audio features
        audio_features = self.audio_extractor.extract_from_video(video_path)
        
        # Extract video frames
        frames = self._extract_frames(video_path)
        
        return frames, audio_features
    
    def _extract_frames(self, video_path: str, max_frames: int = 30) -> List[np.ndarray]:
        """
        Extract frames from a video file
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of video frames
        """
        frames = []
        
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame interval to extract max_frames evenly
            if frame_count <= max_frames:
                frame_interval = 1
            else:
                frame_interval = frame_count // max_frames
            
            # Extract frames
            frame_idx = 0
            while cap.isOpened() and len(frames) < max_frames:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    # Convert to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                
                frame_idx += 1
            
            # Release video capture
            cap.release()
        except Exception as e:
            logger.error(f"Error extracting frames from video: {str(e)}")
        
        return frames
    
    def predict(self, video_path: str) -> float:
        """
        Predict whether a video is deepfake
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Probability of being deepfake
        """
        try:
            # Process video
            frames, audio_features = self.process_video(video_path)
            
            if not frames:
                logger.error(f"No frames extracted from video: {video_path}")
                return 0.5
            
            # Preprocess frames for visual model
            visual_input = self._preprocess_frames(frames)
            
            # Reshape audio features for model input
            audio_input = np.expand_dims(audio_features, axis=0)
            
            # Make prediction
            prediction = self.model.predict([visual_input, audio_input])
            
            return float(prediction[0][0])
        except Exception as e:
            logger.error(f"Error predicting video: {str(e)}")
            return 0.5
    
    def _preprocess_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess frames for the visual model
        
        Args:
            frames: List of video frames
            
        Returns:
            Preprocessed frames
        """
        # Get input shape for visual model
        input_shape = self.visual_model.input_shape[1:3]
        
        # Resize frames
        resized_frames = [cv2.resize(frame, input_shape) for frame in frames]
        
        # Convert to numpy array
        frames_array = np.array(resized_frames)
        
        # Normalize
        frames_array = frames_array.astype(np.float32) / 255.0
        
        # If the visual model expects a single frame, use the middle frame
        if len(self.visual_model.input_shape) == 4:  # (batch_size, height, width, channels)
            middle_idx = len(frames_array) // 2
            return np.expand_dims(frames_array[middle_idx], axis=0)
        
        # Otherwise, return all frames
        return frames_array