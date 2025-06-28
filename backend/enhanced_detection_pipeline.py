"""
Enhanced detection pipeline for DeepDefend
Specialized for high-accuracy deepfake detection with focus on Indian faces
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import time
import logging
from typing import Dict, List, Tuple, Any, Optional
import concurrent.futures
from mtcnn import MTCNN

# Import custom modules
from backend.indian_face_enhancement import detect_and_enhance_indian_faces, analyze_indian_skin_tone
from backend.skin_tone_analyzer import SkinToneAnalyzer

# Configure logging
logger = logging.getLogger(__name__)

# Initialize face detector
detector = MTCNN()

# Initialize skin tone analyzer
skin_analyzer = SkinToneAnalyzer()

class EnhancedDetectionPipeline:
    """
    Enhanced detection pipeline for deepfake detection
    """
    
    def __init__(self, model_paths=None, ensemble_weights=None, use_ensemble=True, use_indian_enhancement=True):
        """
        Initialize the detection pipeline
        
        Args:
            model_paths: Dictionary of model paths
            ensemble_weights: Dictionary of ensemble weights
            use_ensemble: Whether to use ensemble of models
            use_indian_enhancement: Whether to use Indian-specific enhancements
        """
        self.models = {}
        self.ensemble_weights = ensemble_weights or {
            "efficientnet": 0.4,
            "xception": 0.3,
            "indian_specialized": 0.3
        }
        self.use_ensemble = use_ensemble
        self.use_indian_enhancement = use_indian_enhancement
        
        # Default model paths
        self.model_paths = model_paths or {
            "efficientnet": "models/efficientnet_deepfake.h5",
            "xception": "models/xception_deepfake.h5",
            "indian_specialized": "models/indian_specialized_deepfake.h5"
        }
        
        # Load models
        self._load_models()
        
        logger.info("Enhanced detection pipeline initialized")
    
    def _load_models(self):
        """
        Load models for detection
        """
        try:
            # Load each model
            for model_name, model_path in self.model_paths.items():
                if os.path.exists(model_path):
                    logger.info(f"Loading model: {model_name} from {model_path}")
                    self.models[model_name] = tf.keras.models.load_model(model_path)
                else:
                    logger.warning(f"Model not found: {model_path}")
            
            if not self.models:
                logger.error("No models loaded. Detection will not work.")
        
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    def preprocess_image(self, image, target_size=(224, 224)):
        """
        Preprocess image for model input
        
        Args:
            image: Input image (BGR format)
            target_size: Target size for the model
            
        Returns:
            Preprocessed image
        """
        # Resize image
        resized = cv2.resize(image, target_size)
        
        # Convert to RGB (models expect RGB)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Expand dimensions for batch
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def detect_faces(self, image, min_face_size=20):
        """
        Detect faces in an image
        
        Args:
            image: Input image (BGR format)
            min_face_size: Minimum face size to detect
            
        Returns:
            List of detected faces with their locations
        """
        # Convert to RGB (MTCNN expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detector.detect_faces(image_rgb)
        
        # Filter small faces
        faces = [face for face in faces if face['box'][2] >= min_face_size and face['box'][3] >= min_face_size]
        
        return faces
    
    def extract_face(self, image, face_box, margin=0.2, target_size=(224, 224)):
        """
        Extract face from image with margin
        
        Args:
            image: Input image (BGR format)
            face_box: Face bounding box [x, y, width, height]
            margin: Margin around the face as a fraction of face size
            target_size: Target size for the face
            
        Returns:
            Extracted face image
        """
        x, y, w, h = face_box
        
        # Add margin
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        # Calculate coordinates with margin
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        
        # Extract face
        face_img = image[y1:y2, x1:x2]
        
        # Resize to target size
        face_img = cv2.resize(face_img, target_size)
        
        return face_img
    
    def predict_single_model(self, face_image, model_name):
        """
        Make prediction with a single model
        
        Args:
            face_image: Face image (BGR format)
            model_name: Name of the model to use
            
        Returns:
            Prediction probability and confidence
        """
        try:
            # Check if model exists
            if model_name not in self.models:
                logger.warning(f"Model not found: {model_name}")
                return 0.5, 0.1
            
            # Preprocess image
            preprocessed = self.preprocess_image(face_image)
            
            # Make prediction
            prediction = self.models[model_name].predict(preprocessed, verbose=0)[0][0]
            
            # Calculate confidence
            # Confidence is higher when prediction is far from 0.5
            confidence = abs(prediction - 0.5) * 2
            
            return float(prediction), float(confidence)
        
        except Exception as e:
            logger.error(f"Error predicting with model {model_name}: {str(e)}")
            return 0.5, 0.1
    
    def predict_ensemble(self, face_image):
        """
        Make prediction with ensemble of models
        
        Args:
            face_image: Face image (BGR format)
            
        Returns:
            Ensemble prediction probability and confidence
        """
        try:
            # Make predictions with each model
            predictions = {}
            confidences = {}
            
            for model_name in self.models:
                prob, conf = self.predict_single_model(face_image, model_name)
                predictions[model_name] = prob
                confidences[model_name] = conf
            
            # Calculate weighted ensemble prediction
            if not predictions:
                return 0.5, 0.1
            
            # Normalize weights for available models
            weights = {k: v for k, v in self.ensemble_weights.items() if k in predictions}
            total_weight = sum(weights.values())
            normalized_weights = {k: v / total_weight for k, v in weights.items()} if total_weight > 0 else {k: 1.0 / len(weights) for k in weights}
            
            # Calculate weighted prediction
            ensemble_prediction = sum(predictions[model] * normalized_weights.get(model, 0) for model in predictions)
            
            # Calculate ensemble confidence
            # Higher confidence for models with higher individual confidence
            ensemble_confidence = sum(confidences[model] * normalized_weights.get(model, 0) for model in confidences)
            
            return float(ensemble_prediction), float(ensemble_confidence)
        
        except Exception as e:
            logger.error(f"Error predicting with ensemble: {str(e)}")
            return 0.5, 0.1
    
    def analyze_face(self, face_image):
        """
        Analyze a face for deepfake detection
        
        Args:
            face_image: Face image (BGR format)
            
        Returns:
            Analysis results including prediction, confidence, and metadata
        """
        try:
            start_time = time.time()
            
            # Make prediction
            if self.use_ensemble and len(self.models) > 1:
                probability, confidence = self.predict_ensemble(face_image)
            else:
                # Use the first available model
                model_name = next(iter(self.models))
                probability, confidence = self.predict_single_model(face_image, model_name)
            
            # Analyze skin tone
            skin_analysis = None
            if self.use_indian_enhancement:
                skin_analysis = skin_analyzer.analyze_skin_tone(face_image)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Compile results
            result = {
                "probability": probability,
                "confidence": confidence,
                "processing_time": processing_time,
                "metadata": {
                    "skin_tone": skin_analysis if skin_analysis and skin_analysis.get("success", False) else None,
                    "processing_metrics": {
                        "time_ms": processing_time * 1000,
                        "ensemble_used": self.use_ensemble and len(self.models) > 1,
                        "indian_enhancement_used": self.use_indian_enhancement
                    }
                }
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error analyzing face: {str(e)}")
            return {
                "probability": 0.5,
                "confidence": 0.1,
                "error": str(e)
            }
    
    def process_image(self, image_path):
        """
        Process an image for deepfake detection
        
        Args:
            image_path: Path to the image
            
        Returns:
            Detection results including probability, confidence, and regions
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Detect faces
            faces = self.detect_faces(image)
            
            if not faces:
                logger.warning(f"No faces detected in {image_path}")
                return 0.5, 0.1, []
            
            # Process each face
            regions = []
            probabilities = []
            confidences = []
            
            for face in faces:
                # Extract face
                face_img = self.extract_face(image, face['box'])
                
                # Analyze face
                result = self.analyze_face(face_img)
                
                # Add to results
                probabilities.append(result["probability"])
                confidences.append(result["confidence"])
                
                # Add region
                regions.append({
                    "box": face['box'],
                    "probability": result["probability"],
                    "confidence": result["confidence"],
                    "keypoints": face['keypoints'],
                    "metadata": result.get("metadata", {})
                })
            
            # Calculate overall probability and confidence
            # Weight by confidence
            if confidences:
                weights = np.array(confidences) / sum(confidences) if sum(confidences) > 0 else np.ones(len(confidences)) / len(confidences)
                overall_probability = sum(p * w for p, w in zip(probabilities, weights))
                overall_confidence = sum(confidences) / len(confidences)
            else:
                overall_probability = 0.5
                overall_confidence = 0.1
            
            return overall_probability, overall_confidence, regions
        
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return 0.5, 0.1, []
    
    def extract_frames(self, video_path, max_frames=30, frame_interval=1.0):
        """
        Extract frames from a video
        
        Args:
            video_path: Path to the video
            max_frames: Maximum number of frames to extract
            frame_interval: Interval between frames in seconds
            
        Returns:
            List of extracted frames
        """
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            # Calculate frame interval in frames
            frame_interval_frames = int(fps * frame_interval)
            
            # Calculate number of frames to extract
            num_frames = min(max_frames, int(duration / frame_interval) + 1)
            
            # Extract frames
            frames = []
            frame_positions = []
            
            for i in range(num_frames):
                # Calculate frame position
                frame_pos = int(i * frame_interval_frames)
                if frame_pos >= frame_count:
                    break
                
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames.append(frame)
                frame_positions.append(frame_pos)
            
            # Release video
            cap.release()
            
            return frames, frame_positions
        
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {str(e)}")
            return [], []
    
    def process_video(self, video_path, max_frames=30, frame_interval=1.0, parallel=True):
        """
        Process a video for deepfake detection
        
        Args:
            video_path: Path to the video
            max_frames: Maximum number of frames to process
            frame_interval: Interval between frames in seconds
            parallel: Whether to process frames in parallel
            
        Returns:
            Detection results including probability, confidence, frame count, and regions
        """
        try:
            # Extract frames
            frames, frame_positions = self.extract_frames(video_path, max_frames, frame_interval)
            
            if not frames:
                logger.warning(f"No frames extracted from {video_path}")
                return 0.5, 0.1, 0, []
            
            # Process frames
            frame_results = []
            
            if parallel:
                # Process frames in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(frames))) as executor:
                    future_to_frame = {executor.submit(self._process_frame, frame, i): i for i, frame in enumerate(frames)}
                    
                    for future in concurrent.futures.as_completed(future_to_frame):
                        frame_idx = future_to_frame[future]
                        try:
                            result = future.result()
                            if result:
                                result["frame_idx"] = frame_idx
                                result["frame_position"] = frame_positions[frame_idx]
                                frame_results.append(result)
                        except Exception as e:
                            logger.error(f"Error processing frame {frame_idx}: {str(e)}")
            else:
                # Process frames sequentially
                for i, frame in enumerate(frames):
                    try:
                        result = self._process_frame(frame, i)
                        if result:
                            result["frame_idx"] = i
                            result["frame_position"] = frame_positions[i]
                            frame_results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing frame {i}: {str(e)}")
            
            # Calculate overall probability and confidence
            if not frame_results:
                logger.warning(f"No valid results from frames in {video_path}")
                return 0.5, 0.1, len(frames), []
            
            # Extract probabilities and confidences
            probabilities = [r["probability"] for r in frame_results if "probability" in r]
            confidences = [r["confidence"] for r in frame_results if "confidence" in r]
            
            # Weight by confidence
            if confidences:
                weights = np.array(confidences) / sum(confidences) if sum(confidences) > 0 else np.ones(len(confidences)) / len(confidences)
                overall_probability = sum(p * w for p, w in zip(probabilities, weights))
                overall_confidence = sum(confidences) / len(confidences)
            else:
                overall_probability = 0.5
                overall_confidence = 0.1
            
            # Collect all regions
            all_regions = []
            for result in frame_results:
                if "regions" in result and result["regions"]:
                    for region in result["regions"]:
                        region["frame_idx"] = result["frame_idx"]
                        region["frame_position"] = result["frame_position"]
                        all_regions.append(region)
            
            return overall_probability, overall_confidence, len(frames), all_regions
        
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            return 0.5, 0.1, 0, []
    
    def _process_frame(self, frame, frame_idx):
        """
        Process a single video frame
        
        Args:
            frame: Video frame
            frame_idx: Frame index
            
        Returns:
            Frame processing results
        """
        try:
            # Detect faces
            faces = self.detect_faces(frame)
            
            if not faces:
                return None
            
            # Process each face
            regions = []
            probabilities = []
            confidences = []
            
            for face in faces:
                # Extract face
                face_img = self.extract_face(frame, face['box'])
                
                # Analyze face
                result = self.analyze_face(face_img)
                
                # Add to results
                probabilities.append(result["probability"])
                confidences.append(result["confidence"])
                
                # Add region
                regions.append({
                    "box": face['box'],
                    "probability": result["probability"],
                    "confidence": result["confidence"],
                    "keypoints": face['keypoints'],
                    "metadata": result.get("metadata", {})
                })
            
            # Calculate overall probability and confidence
            # Weight by confidence
            if confidences:
                weights = np.array(confidences) / sum(confidences) if sum(confidences) > 0 else np.ones(len(confidences)) / len(confidences)
                overall_probability = sum(p * w for p, w in zip(probabilities, weights))
                overall_confidence = sum(confidences) / len(confidences)
            else:
                overall_probability = 0.5
                overall_confidence = 0.1
            
            return {
                "probability": overall_probability,
                "confidence": overall_confidence,
                "regions": regions
            }
        
        except Exception as e:
            logger.error(f"Error processing frame {frame_idx}: {str(e)}")
            return None

# Function to create and initialize the pipeline
def create_detection_pipeline(model_paths=None, ensemble_weights=None, use_ensemble=True, use_indian_enhancement=True):
    """
    Create and initialize the enhanced detection pipeline
    
    Args:
        model_paths: Dictionary of model paths
        ensemble_weights: Dictionary of ensemble weights
        use_ensemble: Whether to use ensemble of models
        use_indian_enhancement: Whether to use Indian-specific enhancements
        
    Returns:
        Initialized detection pipeline
    """
    return EnhancedDetectionPipeline(
        model_paths=model_paths,
        ensemble_weights=ensemble_weights,
        use_ensemble=use_ensemble,
        use_indian_enhancement=use_indian_enhancement
    )

# Convenience functions for external use
def process_image(image_path, use_ensemble=True, use_indian_enhancement=True):
    """
    Process an image for deepfake detection
    
    Args:
        image_path: Path to the image
        use_ensemble: Whether to use ensemble of models
        use_indian_enhancement: Whether to use Indian-specific enhancements
        
    Returns:
        Detection results including probability, confidence, and regions
    """
    pipeline = create_detection_pipeline(use_ensemble=use_ensemble, use_indian_enhancement=use_indian_enhancement)
    return pipeline.process_image(image_path)

def process_video(video_path, max_frames=30, frame_interval=1.0, use_ensemble=True, use_indian_enhancement=True, parallel=True):
    """
    Process a video for deepfake detection
    
    Args:
        video_path: Path to the video
        max_frames: Maximum number of frames to process
        frame_interval: Interval between frames in seconds
        use_ensemble: Whether to use ensemble of models
        use_indian_enhancement: Whether to use Indian-specific enhancements
        parallel: Whether to process frames in parallel
        
    Returns:
        Detection results including probability, confidence, frame count, and regions
    """
    pipeline = create_detection_pipeline(use_ensemble=use_ensemble, use_indian_enhancement=use_indian_enhancement)
    return pipeline.process_video(video_path, max_frames, frame_interval, parallel)

# Test function
def test_pipeline():
    """
    Test function to verify the detection pipeline works
    """
    try:
        # Create pipeline
        pipeline = create_detection_pipeline(use_ensemble=False, use_indian_enhancement=True)
        
        # Create a sample image (just for testing)
        sample_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Save sample image
        sample_path = "test_sample.jpg"
        cv2.imwrite(sample_path, sample_img)
        
        # Test image processing
        try:
            prob, conf, regions = pipeline.process_image(sample_path)
            print(f"Image processing test: prob={prob}, conf={conf}, regions={len(regions)}")
        except Exception as e:
            print(f"Image processing test error: {str(e)}")
        
        # Clean up
        if os.path.exists(sample_path):
            os.remove(sample_path)
        
        print("Detection pipeline test completed!")
        return True
    
    except Exception as e:
        print(f"Detection pipeline test error: {str(e)}")
        return False

if __name__ == "__main__":
    # Run test
    test_pipeline()