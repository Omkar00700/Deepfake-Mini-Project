"""
Multi-Modal Analyzer for Deepfake Detector

This module implements multi-modal analysis including:
- Combined image and audio analysis
- Cross-modal verification
- Contextual analysis of metadata
- Confidence scoring across modalities
"""

import os
import cv2
import numpy as np
import logging
import json
import time
from typing import Dict, Any, List, Tuple, Optional
import threading
from concurrent.futures import ThreadPoolExecutor
from advanced_detection import AdvancedDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

class MultiModalAnalyzer:
    """Multi-modal analysis for deepfake detection"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the multi-modal analyzer"""
        self.config = config or {}
        self.detector = AdvancedDetector(config)
        
        # Default settings
        self.settings = {
            "cross_modal_verification": True,
            "metadata_analysis": True,
            "confidence_threshold": 0.6,
            "modality_weights": {
                "image": 0.5,
                "audio": 0.3,
                "metadata": 0.2
            },
            "parallel_processing": True,
            "max_workers": 4
        }
        
        # Update settings from config
        if config and "multimodal_settings" in config:
            self.settings.update(config["multimodal_settings"])
        
        logger.info(f"Multi-modal analyzer initialized with settings: {self.settings}")
    
    def analyze(self, file_path: str, analysis_id: str) -> Dict[str, Any]:
        """
        Perform multi-modal analysis on the input file
        
        Args:
            file_path: Path to the input file
            analysis_id: Unique ID for this analysis
            
        Returns:
            Analysis results
        """
        start_time = time.time()
        logger.info(f"Starting multi-modal analysis for {file_path} (ID: {analysis_id})")
        
        # Determine file type
        file_type = self._get_file_type(file_path)
        logger.info(f"Detected file type: {file_type}")
        
        # Extract components based on file type
        components = self._extract_components(file_path, file_type)
        
        # Perform analysis on each modality
        modality_results = {}
        
        if self.settings.get("parallel_processing", True):
            modality_results = self._parallel_analyze(components, file_type)
        else:
            modality_results = self._sequential_analyze(components, file_type)
        
        # Perform cross-modal verification if enabled
        cross_modal_score = 0
        if self.settings.get("cross_modal_verification", True) and len(modality_results) > 1:
            cross_modal_score = self._cross_modal_verify(modality_results)
            logger.info(f"Cross-modal verification score: {cross_modal_score}")
        
        # Analyze metadata if enabled
        metadata_score = 0
        metadata_findings = []
        if self.settings.get("metadata_analysis", True):
            metadata_score, metadata_findings = self._analyze_metadata(file_path, components)
            logger.info(f"Metadata analysis score: {metadata_score}")
        
        # Combine results from all modalities
        combined_result = self._combine_results(modality_results, cross_modal_score, metadata_score)
        
        # Add metadata
        combined_result["analysis_id"] = analysis_id
        combined_result["file_path"] = file_path
        combined_result["file_type"] = file_type
        combined_result["processing_time"] = time.time() - start_time
        combined_result["timestamp"] = time.time()
        combined_result["modalities_analyzed"] = list(modality_results.keys())
        
        if metadata_findings:
            combined_result["metadata_findings"] = metadata_findings
        
        # Save results
        self._save_results(combined_result, analysis_id)
        
        logger.info(f"Multi-modal analysis completed in {combined_result['processing_time']:.2f} seconds")
        return combined_result
    
    def _get_file_type(self, file_path: str) -> str:
        """Determine the type of the input file"""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext in ['.mp4', '.avi', '.mov', '.wmv']:
            return "video"
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            return "image"
        elif ext in ['.mp3', '.wav', '.ogg']:
            return "audio"
        else:
            return "unknown"
    
    def _extract_components(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """Extract components from the input file"""
        components = {}
        
        try:
            if file_type == "video":
                # Extract frames and audio
                frames, audio = self._extract_video_components(file_path)
                components["image"] = frames
                components["audio"] = audio
                
                # Extract metadata
                components["metadata"] = self._extract_metadata(file_path)
                
            elif file_type == "image":
                # Load image
                image = cv2.imread(file_path)
                if image is not None:
                    components["image"] = [image]
                
                # Extract metadata
                components["metadata"] = self._extract_metadata(file_path)
                
            elif file_type == "audio":
                # In a real implementation, we would extract audio using a library like librosa
                # For this demo, we'll simulate audio extraction
                components["audio"] = np.zeros((1, 1000))  # Placeholder for audio data
                
                # Extract metadata
                components["metadata"] = self._extract_metadata(file_path)
            
            logger.info(f"Extracted components: {list(components.keys())}")
            
        except Exception as e:
            logger.error(f"Error extracting components: {str(e)}")
        
        return components
    
    def _extract_video_components(self, video_path: str) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """Extract frames and audio from a video file"""
        frames = []
        audio = None
        
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            
            # Check if video opened successfully
            if not cap.isOpened():
                logger.error(f"Error opening video file: {video_path}")
                return frames, audio
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            logger.info(f"Video properties: {fps} fps, {frame_count} frames, {duration:.2f} seconds")
            
            # Extract frames (sample at 1 fps for efficiency)
            sample_interval = int(fps) if fps > 0 else 1
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_interval == 0:
                    frames.append(frame)
                
                frame_idx += 1
            
            cap.release()
            
            # In a real implementation, we would extract audio using a library like librosa
            # For this demo, we'll simulate audio extraction
            audio = np.zeros((1, 1000))  # Placeholder for audio data
            
            logger.info(f"Extracted {len(frames)} frames at 1 fps")
            
        except Exception as e:
            logger.error(f"Error extracting video components: {str(e)}")
        
        return frames, audio
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from the input file"""
        # In a real implementation, this would extract actual metadata
        # For this demo, we'll return simulated metadata
        
        file_type = self._get_file_type(file_path)
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        
        metadata = {
            "file_name": file_name,
            "file_size": file_size,
            "file_type": file_type,
            "creation_date": time.time() - np.random.randint(1, 100) * 86400,  # Random date in the past
            "last_modified": time.time() - np.random.randint(0, 10) * 86400,  # Random date in the past
        }
        
        if file_type == "image":
            metadata.update({
                "width": np.random.randint(800, 4000),
                "height": np.random.randint(600, 3000),
                "color_space": "RGB",
                "bit_depth": 24,
                "compression": "JPEG",
                "camera_make": np.random.choice(["Canon", "Nikon", "Sony", "iPhone", "Samsung"]),
                "camera_model": f"Model-{np.random.randint(1000, 9999)}",
                "software": np.random.choice(["Photoshop", "Lightroom", "GIMP", "Camera Raw", None]),
                "has_exif": np.random.choice([True, False], p=[0.7, 0.3])
            })
        elif file_type == "video":
            metadata.update({
                "width": np.random.randint(800, 4000),
                "height": np.random.randint(600, 3000),
                "duration": np.random.uniform(5, 120),
                "fps": np.random.choice([24, 25, 30, 60]),
                "codec": np.random.choice(["H.264", "H.265", "VP9", "AV1"]),
                "audio_codec": np.random.choice(["AAC", "MP3", "Opus", None]),
                "bitrate": np.random.randint(1000000, 10000000),
                "camera_make": np.random.choice(["Canon", "Nikon", "Sony", "iPhone", "Samsung"]),
                "camera_model": f"Model-{np.random.randint(1000, 9999)}",
                "software": np.random.choice(["Premiere Pro", "Final Cut Pro", "DaVinci Resolve", None]),
                "has_metadata": np.random.choice([True, False], p=[0.7, 0.3])
            })
        elif file_type == "audio":
            metadata.update({
                "duration": np.random.uniform(5, 300),
                "sample_rate": np.random.choice([8000, 16000, 44100, 48000]),
                "channels": np.random.choice([1, 2]),
                "codec": np.random.choice(["MP3", "AAC", "WAV", "FLAC"]),
                "bitrate": np.random.randint(64000, 320000),
                "software": np.random.choice(["Audacity", "Adobe Audition", "Logic Pro", None]),
                "has_metadata": np.random.choice([True, False], p=[0.7, 0.3])
            })
        
        return metadata
    
    def _parallel_analyze(self, components: Dict[str, Any], file_type: str) -> Dict[str, Any]:
        """Perform analysis in parallel"""
        results = {}
        max_workers = self.settings.get("max_workers", 4)
        
        # Process each modality in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_modality = {}
            
            if "image" in components:
                future_to_modality[executor.submit(self._analyze_image, components["image"])] = "image"
            
            if "audio" in components:
                future_to_modality[executor.submit(self._analyze_audio, components["audio"])] = "audio"
            
            for future in future_to_modality:
                modality = future_to_modality[future]
                try:
                    results[modality] = future.result()
                except Exception as e:
                    logger.error(f"Error in parallel analysis of {modality}: {str(e)}")
        
        return results
    
    def _sequential_analyze(self, components: Dict[str, Any], file_type: str) -> Dict[str, Any]:
        """Perform analysis sequentially"""
        results = {}
        
        if "image" in components:
            try:
                results["image"] = self._analyze_image(components["image"])
            except Exception as e:
                logger.error(f"Error in image analysis: {str(e)}")
        
        if "audio" in components:
            try:
                results["audio"] = self._analyze_audio(components["audio"])
            except Exception as e:
                logger.error(f"Error in audio analysis: {str(e)}")
        
        return results
    
    def _analyze_image(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze image frames"""
        # In a real implementation, this would use the detector to analyze the frames
        # For this demo, we'll simulate image analysis
        
        if not frames:
            return {"probability": 0, "confidence": 0, "error": "No frames to analyze"}
        
        # Simulate different analysis results for different frames
        frame_results = []
        for i, frame in enumerate(frames):
            probability = np.random.uniform(0.6, 0.9)
            confidence = np.random.uniform(0.7, 0.95)
            frame_results.append({
                "frame_idx": i,
                "probability": probability,
                "confidence": confidence
            })
        
        # Aggregate frame results
        avg_probability = sum(r["probability"] for r in frame_results) / len(frame_results)
        avg_confidence = sum(r["confidence"] for r in frame_results) / len(frame_results)
        
        # Determine verdict
        verdict = "deepfake" if avg_probability >= 0.7 else "suspicious" if avg_probability >= 0.4 else "authentic"
        
        return {
            "probability": avg_probability,
            "confidence": avg_confidence,
            "verdict": verdict,
            "frame_results": frame_results,
            "processing_time": np.random.uniform(0.5, 2.0)
        }
    
    def _analyze_audio(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze audio"""
        # In a real implementation, this would use specialized audio analysis
        # For this demo, we'll simulate audio analysis
        
        if audio is None:
            return {"probability": 0, "confidence": 0, "error": "No audio to analyze"}
        
        # Simulate audio analysis
        probability = np.random.uniform(0.4, 0.8)
        confidence = np.random.uniform(0.6, 0.9)
        
        # Determine verdict
        verdict = "deepfake" if probability >= 0.7 else "suspicious" if probability >= 0.4 else "authentic"
        
        # Simulate segment results
        segment_results = []
        num_segments = np.random.randint(5, 15)
        
        for i in range(num_segments):
            segment_prob = np.clip(probability + np.random.uniform(-0.2, 0.2), 0, 1)
            segment_conf = np.clip(confidence + np.random.uniform(-0.1, 0.1), 0, 1)
            segment_results.append({
                "segment_idx": i,
                "start_time": i * np.random.uniform(0.5, 2.0),
                "end_time": (i + 1) * np.random.uniform(0.5, 2.0),
                "probability": segment_prob,
                "confidence": segment_conf
            })
        
        return {
            "probability": probability,
            "confidence": confidence,
            "verdict": verdict,
            "segment_results": segment_results,
            "processing_time": np.random.uniform(0.3, 1.5)
        }
    
    def _cross_modal_verify(self, modality_results: Dict[str, Any]) -> float:
        """Perform cross-modal verification"""
        # In a real implementation, this would compare results across modalities
        # For this demo, we'll simulate cross-modal verification
        
        if len(modality_results) < 2:
            return 0.0
        
        # Check if both image and audio results are available
        if "image" in modality_results and "audio" in modality_results:
            image_prob = modality_results["image"].get("probability", 0)
            audio_prob = modality_results["audio"].get("probability", 0)
            
            # Calculate consistency between modalities
            consistency = 1.0 - abs(image_prob - audio_prob)
            
            # Higher consistency means more reliable results
            return consistency
        
        return 0.5  # Default consistency
    
    def _analyze_metadata(self, file_path: str, components: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Analyze metadata for signs of manipulation"""
        # In a real implementation, this would analyze actual metadata
        # For this demo, we'll simulate metadata analysis
        
        if "metadata" not in components:
            return 0.0, []
        
        metadata = components["metadata"]
        findings = []
        manipulation_score = 0.0
        
        # Check for common signs of manipulation
        if "software" in metadata and metadata["software"] in ["Photoshop", "GIMP", "Premiere Pro"]:
            findings.append(f"File was edited with {metadata['software']}")
            manipulation_score += 0.2
        
        if "has_exif" in metadata and not metadata["has_exif"]:
            findings.append("EXIF data is missing, which is unusual for camera photos")
            manipulation_score += 0.3
        
        if "has_metadata" in metadata and not metadata["has_metadata"]:
            findings.append("Standard metadata is missing")
            manipulation_score += 0.3
        
        # Check creation and modification dates
        if "creation_date" in metadata and "last_modified" in metadata:
            time_diff = metadata["last_modified"] - metadata["creation_date"]
            if time_diff > 86400:  # More than a day difference
                findings.append(f"File was modified {int(time_diff/86400)} days after creation")
                manipulation_score += 0.1
        
        # Limit the score to 1.0
        manipulation_score = min(manipulation_score, 1.0)
        
        return manipulation_score, findings
    
    def _combine_results(self, modality_results: Dict[str, Any], cross_modal_score: float, metadata_score: float) -> Dict[str, Any]:
        """Combine results from all modalities"""
        if not modality_results:
            return {"probability": 0, "confidence": 0, "error": "No results to combine"}
        
        # Get weights for each modality
        weights = self.settings.get("modality_weights", {
            "image": 0.5,
            "audio": 0.3,
            "metadata": 0.2
        })
        
        # Calculate weighted probability and confidence
        weighted_prob_sum = 0
        weighted_conf_sum = 0
        total_weight = 0
        
        if "image" in modality_results and "image" in weights:
            image_result = modality_results["image"]
            image_weight = weights["image"]
            weighted_prob_sum += image_result.get("probability", 0) * image_weight
            weighted_conf_sum += image_result.get("confidence", 0) * image_weight
            total_weight += image_weight
        
        if "audio" in modality_results and "audio" in weights:
            audio_result = modality_results["audio"]
            audio_weight = weights["audio"]
            weighted_prob_sum += audio_result.get("probability", 0) * audio_weight
            weighted_conf_sum += audio_result.get("confidence", 0) * audio_weight
            total_weight += audio_weight
        
        if metadata_score > 0 and "metadata" in weights:
            metadata_weight = weights["metadata"]
            weighted_prob_sum += metadata_score * metadata_weight
            weighted_conf_sum += 0.8 * metadata_weight  # Assume 0.8 confidence for metadata
            total_weight += metadata_weight
        
        # Calculate final probability and confidence
        if total_weight > 0:
            probability = weighted_prob_sum / total_weight
            confidence = weighted_conf_sum / total_weight
            
            # Adjust confidence based on cross-modal verification
            if cross_modal_score > 0:
                confidence = confidence * (0.7 + 0.3 * cross_modal_score)
        else:
            probability = 0
            confidence = 0
        
        # Determine verdict
        verdict = "deepfake" if probability >= 0.7 else "suspicious" if probability >= 0.4 else "authentic"
        
        # Create combined result
        combined_result = {
            "probability": probability,
            "confidence": confidence,
            "verdict": verdict,
            "cross_modal_score": cross_modal_score,
            "metadata_score": metadata_score,
            "modality_results": modality_results
        }
        
        return combined_result
    
    def _save_results(self, results: Dict[str, Any], analysis_id: str) -> None:
        """Save analysis results to file"""
        result_path = os.path.join(RESULTS_DIR, f"multimodal_{analysis_id}.json")
        
        try:
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved analysis results to {result_path}")
        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}")
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current analysis settings"""
        return self.settings
    
    def update_settings(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update analysis settings"""
        self.settings.update(new_settings)
        logger.info(f"Updated analysis settings: {self.settings}")
        return self.settings"""
Multi-Modal Analyzer for Deepfake Detector

This module implements multi-modal analysis including:
- Combined image and audio analysis
- Cross-modal verification
- Contextual analysis of metadata
- Confidence scoring across modalities
"""

import os
import cv2
import numpy as np
import logging
import json
import time
from typing import Dict, Any, List, Tuple, Optional
import threading
from concurrent.futures import ThreadPoolExecutor
from advanced_detection import AdvancedDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

class MultiModalAnalyzer:
    """Multi-modal analysis for deepfake detection"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the multi-modal analyzer"""
        self.config = config or {}
        self.detector = AdvancedDetector(config)
        
        # Default settings
        self.settings = {
            "cross_modal_verification": True,
            "metadata_analysis": True,
            "confidence_threshold": 0.6,
            "modality_weights": {
                "image": 0.5,
                "audio": 0.3,
                "metadata": 0.2
            },
            "parallel_processing": True,
            "max_workers": 4
        }
        
        # Update settings from config
        if config and "multimodal_settings" in config:
            self.settings.update(config["multimodal_settings"])
        
        logger.info(f"Multi-modal analyzer initialized with settings: {self.settings}")
    
    def analyze(self, file_path: str, analysis_id: str) -> Dict[str, Any]:
        """
        Perform multi-modal analysis on the input file
        
        Args:
            file_path: Path to the input file
            analysis_id: Unique ID for this analysis
            
        Returns:
            Analysis results
        """
        start_time = time.time()
        logger.info(f"Starting multi-modal analysis for {file_path} (ID: {analysis_id})")
        
        # Determine file type
        file_type = self._get_file_type(file_path)
        logger.info(f"Detected file type: {file_type}")
        
        # Extract components based on file type
        components = self._extract_components(file_path, file_type)
        
        # Perform analysis on each modality
        modality_results = {}
        
        if self.settings.get("parallel_processing", True):
            modality_results = self._parallel_analyze(components, file_type)
        else:
            modality_results = self._sequential_analyze(components, file_type)
        
        # Perform cross-modal verification if enabled
        cross_modal_score = 0
        if self.settings.get("cross_modal_verification", True) and len(modality_results) > 1:
            cross_modal_score = self._cross_modal_verify(modality_results)
            logger.info(f"Cross-modal verification score: {cross_modal_score}")
        
        # Analyze metadata if enabled
        metadata_score = 0
        metadata_findings = []
        if self.settings.get("metadata_analysis", True):
            metadata_score, metadata_findings = self._analyze_metadata(file_path, components)
            logger.info(f"Metadata analysis score: {metadata_score}")
        
        # Combine results from all modalities
        combined_result = self._combine_results(modality_results, cross_modal_score, metadata_score)
        
        # Add metadata
        combined_result["analysis_id"] = analysis_id
        combined_result["file_path"] = file_path
        combined_result["file_type"] = file_type
        combined_result["processing_time"] = time.time() - start_time
        combined_result["timestamp"] = time.time()
        combined_result["modalities_analyzed"] = list(modality_results.keys())
        
        if metadata_findings:
            combined_result["metadata_findings"] = metadata_findings
        
        # Save results
        self._save_results(combined_result, analysis_id)
        
        logger.info(f"Multi-modal analysis completed in {combined_result['processing_time']:.2f} seconds")
        return combined_result
    
    def _get_file_type(self, file_path: str) -> str:
        """Determine the type of the input file"""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext in ['.mp4', '.avi', '.mov', '.wmv']:
            return "video"
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            return "image"
        elif ext in ['.mp3', '.wav', '.ogg']:
            return "audio"
        else:
            return "unknown"
    
    def _extract_components(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """Extract components from the input file"""
        components = {}
        
        try:
            if file_type == "video":
                # Extract frames and audio
                frames, audio = self._extract_video_components(file_path)
                components["image"] = frames
                components["audio"] = audio
                
                # Extract metadata
                components["metadata"] = self._extract_metadata(file_path)
                
            elif file_type == "image":
                # Load image
                image = cv2.imread(file_path)
                if image is not None:
                    components["image"] = [image]
                
                # Extract metadata
                components["metadata"] = self._extract_metadata(file_path)
                
            elif file_type == "audio":
                # In a real implementation, we would extract audio using a library like librosa
                # For this demo, we'll simulate audio extraction
                components["audio"] = np.zeros((1, 1000))  # Placeholder for audio data
                
                # Extract metadata
                components["metadata"] = self._extract_metadata(file_path)
            
            logger.info(f"Extracted components: {list(components.keys())}")
            
        except Exception as e:
            logger.error(f"Error extracting components: {str(e)}")
        
        return components
    
    def _extract_video_components(self, video_path: str) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """Extract frames and audio from a video file"""
        frames = []
        audio = None
        
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            
            # Check if video opened successfully
            if not cap.isOpened():
                logger.error(f"Error opening video file: {video_path}")
                return frames, audio
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            logger.info(f"Video properties: {fps} fps, {frame_count} frames, {duration:.2f} seconds")
            
            # Extract frames (sample at 1 fps for efficiency)
            sample_interval = int(fps) if fps > 0 else 1
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_interval == 0:
                    frames.append(frame)
                
                frame_idx += 1
            
            cap.release()
            
            # In a real implementation, we would extract audio using a library like librosa
            # For this demo, we'll simulate audio extraction
            audio = np.zeros((1, 1000))  # Placeholder for audio data
            
            logger.info(f"Extracted {len(frames)} frames at 1 fps")
            
        except Exception as e:
            logger.error(f"Error extracting video components: {str(e)}")
        
        return frames, audio
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from the input file"""
        # In a real implementation, this would extract actual metadata
        # For this demo, we'll return simulated metadata
        
        file_type = self._get_file_type(file_path)
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        
        metadata = {
            "file_name": file_name,
            "file_size": file_size,
            "file_type": file_type,
            "creation_date": time.time() - np.random.randint(1, 100) * 86400,  # Random date in the past
            "last_modified": time.time() - np.random.randint(0, 10) * 86400,  # Random date in the past
        }
        
        if file_type == "image":
            metadata.update({
                "width": np.random.randint(800, 4000),
                "height": np.random.randint(600, 3000),
                "color_space": "RGB",
                "bit_depth": 24,
                "compression": "JPEG",
                "camera_make": np.random.choice(["Canon", "Nikon", "Sony", "iPhone", "Samsung"]),
                "camera_model": f"Model-{np.random.randint(1000, 9999)}",
                "software": np.random.choice(["Photoshop", "Lightroom", "GIMP", "Camera Raw", None]),
                "has_exif": np.random.choice([True, False], p=[0.7, 0.3])
            })
        elif file_type == "video":
            metadata.update({
                "width": np.random.randint(800, 4000),
                "height": np.random.randint(600, 3000),
                "duration": np.random.uniform(5, 120),
                "fps": np.random.choice([24, 25, 30, 60]),
                "codec": np.random.choice(["H.264", "H.265", "VP9", "AV1"]),
                "audio_codec": np.random.choice(["AAC", "MP3", "Opus", None]),
                "bitrate": np.random.randint(1000000, 10000000),
                "camera_make": np.random.choice(["Canon", "Nikon", "Sony", "iPhone", "Samsung"]),
                "camera_model": f"Model-{np.random.randint(1000, 9999)}",
                "software": np.random.choice(["Premiere Pro", "Final Cut Pro", "DaVinci Resolve", None]),
                "has_metadata": np.random.choice([True, False], p=[0.7, 0.3])
            })
        elif file_type == "audio":
            metadata.update({
                "duration": np.random.uniform(5, 300),
                "sample_rate": np.random.choice([8000, 16000, 44100, 48000]),
                "channels": np.random.choice([1, 2]),
                "codec": np.random.choice(["MP3", "AAC", "WAV", "FLAC"]),
                "bitrate": np.random.randint(64000, 320000),
                "software": np.random.choice(["Audacity", "Adobe Audition", "Logic Pro", None]),
                "has_metadata": np.random.choice([True, False], p=[0.7, 0.3])
            })
        
        return metadata
    
    def _parallel_analyze(self, components: Dict[str, Any], file_type: str) -> Dict[str, Any]:
        """Perform analysis in parallel"""
        results = {}
        max_workers = self.settings.get("max_workers", 4)
        
        # Process each modality in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_modality = {}
            
            if "image" in components:
                future_to_modality[executor.submit(self._analyze_image, components["image"])] = "image"
            
            if "audio" in components:
                future_to_modality[executor.submit(self._analyze_audio, components["audio"])] = "audio"
            
            for future in future_to_modality:
                modality = future_to_modality[future]
                try:
                    results[modality] = future.result()
                except Exception as e:
                    logger.error(f"Error in parallel analysis of {modality}: {str(e)}")
        
        return results
    
    def _sequential_analyze(self, components: Dict[str, Any], file_type: str) -> Dict[str, Any]:
        """Perform analysis sequentially"""
        results = {}
        
        if "image" in components:
            try:
                results["image"] = self._analyze_image(components["image"])
            except Exception as e:
                logger.error(f"Error in image analysis: {str(e)}")
        
        if "audio" in components:
            try:
                results["audio"] = self._analyze_audio(components["audio"])
            except Exception as e:
                logger.error(f"Error in audio analysis: {str(e)}")
        
        return results
    
    def _analyze_image(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze image frames"""
        # In a real implementation, this would use the detector to analyze the frames
        # For this demo, we'll simulate image analysis
        
        if not frames:
            return {"probability": 0, "confidence": 0, "error": "No frames to analyze"}
        
        # Simulate different analysis results for different frames
        frame_results = []
        for i, frame in enumerate(frames):
            probability = np.random.uniform(0.6, 0.9)
            confidence = np.random.uniform(0.7, 0.95)
            frame_results.append({
                "frame_idx": i,
                "probability": probability,
                "confidence": confidence
            })
        
        # Aggregate frame results
        avg_probability = sum(r["probability"] for r in frame_results) / len(frame_results)
        avg_confidence = sum(r["confidence"] for r in frame_results) / len(frame_results)
        
        # Determine verdict
        verdict = "deepfake" if avg_probability >= 0.7 else "suspicious" if avg_probability >= 0.4 else "authentic"
        
        return {
            "probability": avg_probability,
            "confidence": avg_confidence,
            "verdict": verdict,
            "frame_results": frame_results,
            "processing_time": np.random.uniform(0.5, 2.0)
        }
    
    def _analyze_audio(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze audio"""
        # In a real implementation, this would use specialized audio analysis
        # For this demo, we'll simulate audio analysis
        
        if audio is None:
            return {"probability": 0, "confidence": 0, "error": "No audio to analyze"}
        
        # Simulate audio analysis
        probability = np.random.uniform(0.4, 0.8)
        confidence = np.random.uniform(0.6, 0.9)
        
        # Determine verdict
        verdict = "deepfake" if probability >= 0.7 else "suspicious" if probability >= 0.4 else "authentic"
        
        # Simulate segment results
        segment_results = []
        num_segments = np.random.randint(5, 15)
        
        for i in range(num_segments):
            segment_prob = np.clip(probability + np.random.uniform(-0.2, 0.2), 0, 1)
            segment_conf = np.clip(confidence + np.random.uniform(-0.1, 0.1), 0, 1)
            segment_results.append({
                "segment_idx": i,
                "start_time": i * np.random.uniform(0.5, 2.0),
                "end_time": (i + 1) * np.random.uniform(0.5, 2.0),
                "probability": segment_prob,
                "confidence": segment_conf
            })
        
        return {
            "probability": probability,
            "confidence": confidence,
            "verdict": verdict,
            "segment_results": segment_results,
            "processing_time": np.random.uniform(0.3, 1.5)
        }
    
    def _cross_modal_verify(self, modality_results: Dict[str, Any]) -> float:
        """Perform cross-modal verification"""
        # In a real implementation, this would compare results across modalities
        # For this demo, we'll simulate cross-modal verification
        
        if len(modality_results) < 2:
            return 0.0
        
        # Check if both image and audio results are available
        if "image" in modality_results and "audio" in modality_results:
            image_prob = modality_results["image"].get("probability", 0)
            audio_prob = modality_results["audio"].get("probability", 0)
            
            # Calculate consistency between modalities
            consistency = 1.0 - abs(image_prob - audio_prob)
            
            # Higher consistency means more reliable results
            return consistency
        
        return 0.5  # Default consistency
    
    def _analyze_metadata(self, file_path: str, components: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Analyze metadata for signs of manipulation"""
        # In a real implementation, this would analyze actual metadata
        # For this demo, we'll simulate metadata analysis
        
        if "metadata" not in components:
            return 0.0, []
        
        metadata = components["metadata"]
        findings = []
        manipulation_score = 0.0
        
        # Check for common signs of manipulation
        if "software" in metadata and metadata["software"] in ["Photoshop", "GIMP", "Premiere Pro"]:
            findings.append(f"File was edited with {metadata['software']}")
            manipulation_score += 0.2
        
        if "has_exif" in metadata and not metadata["has_exif"]:
            findings.append("EXIF data is missing, which is unusual for camera photos")
            manipulation_score += 0.3
        
        if "has_metadata" in metadata and not metadata["has_metadata"]:
            findings.append("Standard metadata is missing")
            manipulation_score += 0.3
        
        # Check creation and modification dates
        if "creation_date" in metadata and "last_modified" in metadata:
            time_diff = metadata["last_modified"] - metadata["creation_date"]
            if time_diff > 86400:  # More than a day difference
                findings.append(f"File was modified {int(time_diff/86400)} days after creation")
                manipulation_score += 0.1
        
        # Limit the score to 1.0
        manipulation_score = min(manipulation_score, 1.0)
        
        return manipulation_score, findings
    
    def _combine_results(self, modality_results: Dict[str, Any], cross_modal_score: float, metadata_score: float) -> Dict[str, Any]:
        """Combine results from all modalities"""
        if not modality_results:
            return {"probability": 0, "confidence": 0, "error": "No results to combine"}
        
        # Get weights for each modality
        weights = self.settings.get("modality_weights", {
            "image": 0.5,
            "audio": 0.3,
            "metadata": 0.2
        })
        
        # Calculate weighted probability and confidence
        weighted_prob_sum = 0
        weighted_conf_sum = 0
        total_weight = 0
        
        if "image" in modality_results and "image" in weights:
            image_result = modality_results["image"]
            image_weight = weights["image"]
            weighted_prob_sum += image_result.get("probability", 0) * image_weight
            weighted_conf_sum += image_result.get("confidence", 0) * image_weight
            total_weight += image_weight
        
        if "audio" in modality_results and "audio" in weights:
            audio_result = modality_results["audio"]
            audio_weight = weights["audio"]
            weighted_prob_sum += audio_result.get("probability", 0) * audio_weight
            weighted_conf_sum += audio_result.get("confidence", 0) * audio_weight
            total_weight += audio_weight
        
        if metadata_score > 0 and "metadata" in weights:
            metadata_weight = weights["metadata"]
            weighted_prob_sum += metadata_score * metadata_weight
            weighted_conf_sum += 0.8 * metadata_weight  # Assume 0.8 confidence for metadata
            total_weight += metadata_weight
        
        # Calculate final probability and confidence
        if total_weight > 0:
            probability = weighted_prob_sum / total_weight
            confidence = weighted_conf_sum / total_weight
            
            # Adjust confidence based on cross-modal verification
            if cross_modal_score > 0:
                confidence = confidence * (0.7 + 0.3 * cross_modal_score)
        else:
            probability = 0
            confidence = 0
        
        # Determine verdict
        verdict = "deepfake" if probability >= 0.7 else "suspicious" if probability >= 0.4 else "authentic"
        
        # Create combined result
        combined_result = {
            "probability": probability,
            "confidence": confidence,
            "verdict": verdict,
            "cross_modal_score": cross_modal_score,
            "metadata_score": metadata_score,
            "modality_results": modality_results
        }
        
        return combined_result
    
    def _save_results(self, results: Dict[str, Any], analysis_id: str) -> None:
        """Save analysis results to file"""
        result_path = os.path.join(RESULTS_DIR, f"multimodal_{analysis_id}.json")
        
        try:
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved analysis results to {result_path}")
        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}")
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current analysis settings"""
        return self.settings
    
    def update_settings(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update analysis settings"""
        self.settings.update(new_settings)
        logger.info(f"Updated analysis settings: {self.settings}")
        return self.settings