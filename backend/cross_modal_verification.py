"""
Cross-Modal Verification for Deepfake Detection
Implements multi-modal analysis to improve detection accuracy
"""

import os
import numpy as np
import tensorflow as tf
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import cv2
import time
from pathlib import Path
import json
import librosa
import soundfile as sf
from scipy.signal import spectrogram
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Input, Dropout, LSTM, Bidirectional,
    BatchNormalization, Conv1D, MaxPooling1D, Flatten, Concatenate
)

# Configure logging
logger = logging.getLogger(__name__)

class AudioFeatureExtractor:
    """
    Extract audio features for deepfake detection
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
    
    def extract_from_video(self, video_path: str) -> Dict[str, np.ndarray]:
        """
        Extract audio features from a video file
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary of audio features
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
            return {
                "mfcc": np.zeros((self.max_frames, self.n_mfcc)),
                "spectral_contrast": np.zeros((self.max_frames, 7)),
                "chroma": np.zeros((self.max_frames, 12)),
                "spectral_bandwidth": np.zeros((self.max_frames, 1)),
                "spectral_flatness": np.zeros((self.max_frames, 1)),
                "rms": np.zeros((self.max_frames, 1)),
                "zero_crossing_rate": np.zeros((self.max_frames, 1))
            }
    
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
    
    def extract_from_audio(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        Extract audio features from an audio file
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary of audio features
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.max_audio_length)
            
            # Initialize features dictionary
            features = {}
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length
            )
            features["mfcc"] = self._pad_or_truncate(mfcc.T)
            
            # Extract spectral contrast
            contrast = librosa.feature.spectral_contrast(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            features["spectral_contrast"] = self._pad_or_truncate(contrast.T)
            
            # Extract chroma features
            chroma = librosa.feature.chroma_stft(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            features["chroma"] = self._pad_or_truncate(chroma.T)
            
            # Extract spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            features["spectral_bandwidth"] = self._pad_or_truncate(bandwidth.T)
            
            # Extract spectral flatness
            flatness = librosa.feature.spectral_flatness(
                y=y, n_fft=self.n_fft, hop_length=self.hop_length
            )
            features["spectral_flatness"] = self._pad_or_truncate(flatness.T)
            
            # Extract RMS energy
            rms = librosa.feature.rms(
                y=y, frame_length=self.n_fft, hop_length=self.hop_length
            )
            features["rms"] = self._pad_or_truncate(rms.T)
            
            # Extract zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                y=y, frame_length=self.n_fft, hop_length=self.hop_length
            )
            features["zero_crossing_rate"] = self._pad_or_truncate(zcr.T)
            
            return features
        except Exception as e:
            logger.error(f"Error extracting audio features: {str(e)}")
            # Return empty features
            return {
                "mfcc": np.zeros((self.max_frames, self.n_mfcc)),
                "spectral_contrast": np.zeros((self.max_frames, 7)),
                "chroma": np.zeros((self.max_frames, 12)),
                "spectral_bandwidth": np.zeros((self.max_frames, 1)),
                "spectral_flatness": np.zeros((self.max_frames, 1)),
                "rms": np.zeros((self.max_frames, 1)),
                "zero_crossing_rate": np.zeros((self.max_frames, 1))
            }
    
    def _pad_or_truncate(self, features: np.ndarray) -> np.ndarray:
        """
        Pad or truncate features to max_frames
        
        Args:
            features: Feature array of shape (time, features)
            
        Returns:
            Padded or truncated features
        """
        # Pad or truncate to max_frames
        if features.shape[0] < self.max_frames:
            # Pad with zeros
            pad_width = self.max_frames - features.shape[0]
            features = np.pad(features, ((0, pad_width), (0, 0)))
        elif features.shape[0] > self.max_frames:
            # Truncate
            features = features[:self.max_frames, :]
        
        return features


class LipSyncAnalyzer:
    """
    Analyze lip synchronization for deepfake detection
    """
    
    def __init__(self, 
                 face_detector_path: str = None,
                 landmark_detector_path: str = None):
        """
        Initialize the lip sync analyzer
        
        Args:
            face_detector_path: Path to face detector model
            landmark_detector_path: Path to landmark detector model
        """
        self.face_detector = None
        self.landmark_detector = None
        
        # Initialize face detector
        try:
            if face_detector_path and os.path.exists(face_detector_path):
                self.face_detector = cv2.dnn.readNetFromCaffe(
                    face_detector_path,
                    face_detector_path.replace(".prototxt", ".caffemodel")
                )
            else:
                # Use OpenCV's built-in face detector
                self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            logger.error(f"Error initializing face detector: {str(e)}")
        
        # Initialize landmark detector
        try:
            if landmark_detector_path and os.path.exists(landmark_detector_path):
                self.landmark_detector = cv2.face.createFacemarkLBF()
                self.landmark_detector.loadModel(landmark_detector_path)
            else:
                # Try to use dlib's landmark detector
                import dlib
                self.landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
                self.use_dlib = True
        except Exception as e:
            logger.error(f"Error initializing landmark detector: {str(e)}")
            self.landmark_detector = None
        
        logger.info("Initialized lip sync analyzer")
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze lip synchronization in a video
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary of lip sync analysis results
        """
        try:
            # Extract audio features
            audio_extractor = AudioFeatureExtractor()
            audio_features = audio_extractor.extract_from_video(video_path)
            
            # Extract lip movements
            lip_movements = self._extract_lip_movements(video_path)
            
            # Calculate correlation between audio and lip movements
            correlation = self._calculate_correlation(audio_features, lip_movements)
            
            # Calculate inconsistency score
            inconsistency_score = self._calculate_inconsistency_score(correlation)
            
            return {
                "correlation": correlation,
                "inconsistency_score": inconsistency_score,
                "is_inconsistent": inconsistency_score > 0.5
            }
        except Exception as e:
            logger.error(f"Error analyzing lip sync: {str(e)}")
            return {
                "correlation": 0.0,
                "inconsistency_score": 0.5,
                "is_inconsistent": False,
                "error": str(e)
            }
    
    def _extract_lip_movements(self, video_path: str) -> np.ndarray:
        """
        Extract lip movements from a video
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Array of lip movement features
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize lip movement array
        lip_movements = []
        
        # Process frames
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process every 3rd frame to reduce computation
            if frame_idx % 3 == 0:
                # Detect face
                faces = self._detect_faces(frame)
                
                # Extract lip landmarks
                lip_features = self._extract_lip_landmarks(frame, faces)
                
                if lip_features is not None:
                    lip_movements.append(lip_features)
            
            frame_idx += 1
        
        # Release video capture
        cap.release()
        
        # Convert to numpy array
        if lip_movements:
            lip_movements = np.array(lip_movements)
        else:
            # Return empty array if no lip movements detected
            lip_movements = np.zeros((1, 20))
        
        return lip_movements
    
    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame
        
        Args:
            frame: Video frame
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        if self.face_detector is None:
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        return faces
    
    def _extract_lip_landmarks(self, 
                              frame: np.ndarray, 
                              faces: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Extract lip landmarks from a frame
        
        Args:
            frame: Video frame
            faces: List of face bounding boxes
            
        Returns:
            Array of lip landmark features
        """
        if not faces or self.landmark_detector is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get the largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Detect landmarks
        try:
            if hasattr(self, 'use_dlib') and self.use_dlib:
                # Convert to dlib rectangle
                import dlib
                rect = dlib.rectangle(x, y, x+w, y+h)
                
                # Detect landmarks
                shape = self.landmark_detector(gray, rect)
                
                # Extract lip landmarks (indices 48-67)
                lip_points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(48, 68)])
            else:
                # Detect landmarks using OpenCV
                _, landmarks = self.landmark_detector.fit(gray, np.array([face]))
                
                # Extract lip landmarks (indices 48-67)
                lip_points = landmarks[0][0][48:68]
            
            # Normalize coordinates
            lip_points = lip_points - np.array([x, y])
            lip_points = lip_points / np.array([w, h])
            
            # Flatten to feature vector
            lip_features = lip_points.flatten()
            
            return lip_features
        except Exception as e:
            logger.error(f"Error extracting lip landmarks: {str(e)}")
            return None
    
    def _calculate_correlation(self, 
                              audio_features: Dict[str, np.ndarray],
                              lip_movements: np.ndarray) -> float:
        """
        Calculate correlation between audio and lip movements
        
        Args:
            audio_features: Dictionary of audio features
            lip_movements: Array of lip movement features
            
        Returns:
            Correlation score
        """
        if lip_movements.shape[0] <= 1:
            return 0.0
        
        # Use RMS energy as the main audio feature for correlation
        if "rms" in audio_features and audio_features["rms"].shape[0] > 1:
            audio_energy = audio_features["rms"].flatten()
            
            # Resample to match lip movement length
            audio_resampled = np.interp(
                np.linspace(0, 1, lip_movements.shape[0]),
                np.linspace(0, 1, audio_energy.shape[0]),
                audio_energy
            )
            
            # Calculate lip movement energy (sum of absolute differences)
            lip_energy = np.sum(np.abs(np.diff(lip_movements, axis=0)), axis=1)
            if lip_energy.shape[0] < audio_resampled.shape[0]:
                lip_energy = np.pad(lip_energy, (0, audio_resampled.shape[0] - lip_energy.shape[0]))
            else:
                lip_energy = lip_energy[:audio_resampled.shape[0]]
            
            # Calculate correlation
            correlation = np.corrcoef(audio_resampled, lip_energy)[0, 1]
            
            # Handle NaN
            if np.isnan(correlation):
                correlation = 0.0
            
            return correlation
        else:
            return 0.0
    
    def _calculate_inconsistency_score(self, correlation: float) -> float:
        """
        Calculate inconsistency score from correlation
        
        Args:
            correlation: Correlation between audio and lip movements
            
        Returns:
            Inconsistency score (0-1, higher means more inconsistent)
        """
        # Invert and scale correlation to get inconsistency
        # Correlation of 1 means perfectly consistent (inconsistency = 0)
        # Correlation of -1 means perfectly inconsistent (inconsistency = 1)
        # Correlation of 0 means no relation (inconsistency = 0.5)
        inconsistency = (1 - correlation) / 2
        
        return inconsistency


class CrossModalVerifier:
    """
    Cross-modal verification for deepfake detection
    """
    
    def __init__(self, 
                 visual_model: tf.keras.Model = None,
                 audio_model_path: str = None,
                 use_lip_sync: bool = True,
                 use_audio_analysis: bool = True):
        """
        Initialize the cross-modal verifier
        
        Args:
            visual_model: Visual deepfake detection model
            audio_model_path: Path to audio deepfake detection model
            use_lip_sync: Whether to use lip sync analysis
            use_audio_analysis: Whether to use audio analysis
        """
        self.visual_model = visual_model
        self.audio_model = None
        self.use_lip_sync = use_lip_sync
        self.use_audio_analysis = use_audio_analysis
        
        # Initialize audio model if path is provided
        if audio_model_path and os.path.exists(audio_model_path):
            try:
                self.audio_model = load_model(audio_model_path)
                logger.info(f"Loaded audio model from {audio_model_path}")
            except Exception as e:
                logger.error(f"Error loading audio model: {str(e)}")
        
        # Initialize lip sync analyzer if enabled
        self.lip_sync_analyzer = None
        if use_lip_sync:
            self.lip_sync_analyzer = LipSyncAnalyzer()
        
        # Initialize audio feature extractor
        self.audio_extractor = AudioFeatureExtractor()
        
        logger.info(f"Initialized cross-modal verifier with lip_sync={use_lip_sync}, audio_analysis={use_audio_analysis}")
    
    def verify(self, 
              video_path: str,
              visual_prediction: float = None) -> Dict[str, Any]:
        """
        Perform cross-modal verification on a video
        
        Args:
            video_path: Path to the video file
            visual_prediction: Visual model prediction (if already computed)
            
        Returns:
            Dictionary of verification results
        """
        start_time = time.time()
        
        # Initialize results
        results = {
            "visual_prediction": visual_prediction,
            "audio_prediction": None,
            "lip_sync_score": None,
            "cross_modal_score": None,
            "is_deepfake": None,
            "confidence": None,
            "processing_time": None
        }
        
        # Extract audio features
        audio_features = None
        if self.use_audio_analysis or (self.use_lip_sync and self.lip_sync_analyzer):
            audio_features = self.audio_extractor.extract_from_video(video_path)
        
        # Perform audio analysis if enabled
        if self.use_audio_analysis and self.audio_model and audio_features:
            # Prepare audio features for model input
            mfcc_features = audio_features["mfcc"]
            mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Add batch dimension
            
            # Get audio model prediction
            audio_prediction = self.audio_model.predict(mfcc_features)[0][0]
            results["audio_prediction"] = float(audio_prediction)
        
        # Perform lip sync analysis if enabled
        if self.use_lip_sync and self.lip_sync_analyzer:
            lip_sync_results = self.lip_sync_analyzer.analyze_video(video_path)
            results["lip_sync_score"] = lip_sync_results["inconsistency_score"]
        
        # Calculate cross-modal score
        cross_modal_score = self._calculate_cross_modal_score(results)
        results["cross_modal_score"] = cross_modal_score
        
        # Determine if deepfake
        if visual_prediction is not None:
            # Combine visual prediction with cross-modal score
            combined_score = self._combine_scores(visual_prediction, cross_modal_score)
            results["is_deepfake"] = combined_score > 0.5
            results["confidence"] = self._calculate_confidence(visual_prediction, cross_modal_score)
        else:
            # Use cross-modal score alone
            results["is_deepfake"] = cross_modal_score > 0.5
            results["confidence"] = 0.5 + abs(cross_modal_score - 0.5)
        
        # Calculate processing time
        results["processing_time"] = time.time() - start_time
        
        return results
    
    def _calculate_cross_modal_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate cross-modal score from individual modality results
        
        Args:
            results: Dictionary of verification results
            
        Returns:
            Cross-modal score (0-1, higher means more likely to be deepfake)
        """
        scores = []
        weights = []
        
        # Add audio prediction if available
        if results["audio_prediction"] is not None:
            scores.append(results["audio_prediction"])
            weights.append(0.3)  # 30% weight for audio
        
        # Add lip sync score if available
        if results["lip_sync_score"] is not None:
            scores.append(results["lip_sync_score"])
            weights.append(0.2)  # 20% weight for lip sync
        
        # If no scores available, return neutral score
        if not scores:
            return 0.5
        
        # Normalize weights
        weights = [w / sum(weights) for w in weights]
        
        # Calculate weighted average
        cross_modal_score = sum(s * w for s, w in zip(scores, weights))
        
        return cross_modal_score
    
    def _combine_scores(self, 
                       visual_prediction: float,
                       cross_modal_score: float) -> float:
        """
        Combine visual prediction with cross-modal score
        
        Args:
            visual_prediction: Visual model prediction
            cross_modal_score: Cross-modal score
            
        Returns:
            Combined score
        """
        # Visual prediction has higher weight (70%)
        combined_score = 0.7 * visual_prediction + 0.3 * cross_modal_score
        
        return combined_score
    
    def _calculate_confidence(self, 
                             visual_prediction: float,
                             cross_modal_score: float) -> float:
        """
        Calculate confidence based on agreement between modalities
        
        Args:
            visual_prediction: Visual model prediction
            cross_modal_score: Cross-modal score
            
        Returns:
            Confidence score (0-1)
        """
        # Calculate agreement between visual and cross-modal
        agreement = 1.0 - abs(visual_prediction - cross_modal_score)
        
        # Calculate extremity (how far from decision boundary)
        combined_score = self._combine_scores(visual_prediction, cross_modal_score)
        extremity = abs(combined_score - 0.5) * 2  # Scale to [0, 1]
        
        # Combine agreement and extremity
        confidence = 0.5 + (agreement * 0.25) + (extremity * 0.25)
        
        # Ensure valid range
        confidence = min(0.95, max(0.5, confidence))
        
        return confidence