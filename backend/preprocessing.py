
import cv2
import numpy as np
import time
import os
import logging
import threading
import hashlib
import json
from typing import Tuple, List, Dict, Any, Optional
from collections import OrderedDict
from face_detector import FaceDetector
from backend.config import (
    FACE_DETECTION_CONFIDENCE, FACE_MARGIN_PERCENT,
    FRAME_SIMILARITY_THRESHOLD, ENABLE_FRAME_CACHING, FRAME_CACHE_SIZE,
    ENABLE_ADVANCED_PREPROCESSING
)

# Configure logging
logger = logging.getLogger(__name__)

# Initialize the face detector
face_detector = FaceDetector(confidence_threshold=FACE_DETECTION_CONFIDENCE)

# Frame cache for video processing
frame_cache = OrderedDict()
frame_cache_lock = threading.Lock()

class DiagnosticLog:
    """
    Class to track diagnostic information during preprocessing
    for debugging and performance analysis
    """
    def __init__(self):
        self.start_time = time.time()
        self.logs = []
        self.stages = {}
        self.current_stage = None
        self.metrics = {
            "frame_counts": {
                "total": 0,
                "processed": 0,
                "cached": 0,
                "skipped": 0
            },
            "timing": {},
            "quality": {},
            "errors": []
        }
    
    def start_stage(self, stage_name: str):
        """Start timing a new processing stage"""
        self.current_stage = stage_name
        self.stages[stage_name] = {
            "start": time.time(),
            "end": None,
            "duration": None
        }
        self.log(f"Starting {stage_name}")
    
    def end_stage(self, stage_name: str):
        """End timing for a processing stage"""
        if stage_name in self.stages:
            self.stages[stage_name]["end"] = time.time()
            self.stages[stage_name]["duration"] = (
                self.stages[stage_name]["end"] - self.stages[stage_name]["start"]
            )
            self.metrics["timing"][stage_name] = self.stages[stage_name]["duration"]
            self.log(f"Completed {stage_name} in {self.stages[stage_name]['duration']:.4f}s")
    
    def log(self, message: str, data: Dict[str, Any] = None):
        """Add a log entry with timestamp"""
        entry = {
            "timestamp": time.time(),
            "elapsed": time.time() - self.start_time,
            "message": message,
            "stage": self.current_stage,
            "data": data
        }
        self.logs.append(entry)
        logger.debug(f"[DIAG] {message}" + (f" {data}" if data else ""))
    
    def add_error(self, error_type: str, message: str, details: Dict[str, Any] = None):
        """Log an error or warning"""
        error_entry = {
            "type": error_type,
            "message": message,
            "timestamp": time.time(),
            "details": details
        }
        self.metrics["errors"].append(error_entry)
        self.log(f"ERROR: {message}", {"error_type": error_type, "details": details})
    
    def add_quality_metric(self, metric_name: str, value: float):
        """Add a quality metric"""
        self.metrics["quality"][metric_name] = value
        self.log(f"Quality metric: {metric_name} = {value:.4f}")
    
    def get_report(self) -> Dict[str, Any]:
        """Get the complete diagnostic report"""
        total_time = time.time() - self.start_time
        report = {
            "overall": {
                "total_processing_time": total_time,
                "success": len(self.metrics["errors"]) == 0,
                "error_count": len(self.metrics["errors"])
            },
            "stages": self.stages,
            "metrics": self.metrics,
            "logs": self.logs
        }
        
        # Calculate stage percentages
        total_stage_time = sum(s["duration"] or 0 for s in self.stages.values())
        if total_stage_time > 0:
            report["overall"]["stage_breakdown"] = {
                name: {"duration": data["duration"], "percentage": (data["duration"] / total_stage_time) * 100}
                for name, data in self.stages.items() if data["duration"]
            }
        
        return report
    
    def to_json(self) -> str:
        """Convert the report to JSON string"""
        return json.dumps(self.get_report(), indent=2)
    
    def __str__(self) -> str:
        """String representation of the diagnostic log summary"""
        errors = len(self.metrics["errors"])
        stages = len(self.stages)
        total_time = time.time() - self.start_time
        return f"DiagnosticLog: {stages} stages, {errors} errors, {total_time:.2f}s total"

def get_frame_hash(frame):
    """
    Generate a hash for a frame to use as a cache key
    Enhanced with logging
    """
    # Downscale frame to reduce hash computation time
    small_frame = cv2.resize(frame, (32, 32))
    # Compute hash
    return hashlib.md5(small_frame.tobytes()).hexdigest()

def is_similar_frame(frame1, frame2, threshold=FRAME_SIMILARITY_THRESHOLD, diagnostic_log=None):
    """
    Check if two frames are similar based on histogram comparison
    Enhanced with quality metrics and diagnostics
    """
    if frame1.shape != frame2.shape:
        if diagnostic_log:
            diagnostic_log.log("Frame shape mismatch, cannot compare similarity")
        return False
    
    start_time = time.time()
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate histograms
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [0, 256], [0, 256])
    
    # Normalize histograms
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    
    # Compare histograms
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    if diagnostic_log:
        diagnostic_log.log(f"Frame similarity: {similarity:.4f} (threshold: {threshold})",
                         {"similarity": similarity, "time": time.time() - start_time})
    
    return similarity >= threshold

def cache_frame_result(frame, result, diagnostic_log=None):
    """
    Cache a frame processing result
    Enhanced with diagnostics
    """
    if not ENABLE_FRAME_CACHING:
        return
    
    with frame_cache_lock:
        # Get frame hash
        frame_hash = get_frame_hash(frame)
        
        # Add to cache
        frame_cache[frame_hash] = result
        
        # Remove oldest items if cache is too large
        while len(frame_cache) > FRAME_CACHE_SIZE:
            frame_cache.popitem(last=False)
        
        if diagnostic_log:
            diagnostic_log.log(f"Frame result cached (hash: {frame_hash})", 
                            {"cache_size": len(frame_cache)})

def get_cached_frame_result(frame, diagnostic_log=None):
    """
    Get a cached frame processing result if available
    Enhanced with diagnostics
    """
    if not ENABLE_FRAME_CACHING:
        return None
    
    start_time = time.time()
    
    with frame_cache_lock:
        # Get frame hash
        frame_hash = get_frame_hash(frame)
        
        # Check cache
        if frame_hash in frame_cache:
            # Move to end (most recently used)
            result = frame_cache.pop(frame_hash)
            frame_cache[frame_hash] = result
            
            if diagnostic_log:
                diagnostic_log.log("Using exact cached result", 
                                {"hash": frame_hash, "time": time.time() - start_time})
                # Increment cached frame count
                diagnostic_log.metrics["frame_counts"]["cached"] += 1
            
            return result
        
        # Check for similar frames
        similar_frame_found = False
        for cached_hash, cached_result in frame_cache.items():
            if 'frame' in cached_result and is_similar_frame(frame, cached_result['frame'], 
                                                            diagnostic_log=diagnostic_log):
                logger.debug("Using cached result for similar frame")
                similar_frame_found = True
                
                # Move to end (most recently used)
                frame_cache.pop(cached_hash)
                frame_cache[frame_hash] = cached_result
                
                if diagnostic_log:
                    diagnostic_log.log("Using similar frame cached result", 
                                    {"original_hash": cached_hash,
                                     "new_hash": frame_hash,
                                     "time": time.time() - start_time})
                    # Increment cached frame count
                    diagnostic_log.metrics["frame_counts"]["cached"] += 1
                
                return cached_result
    
    if diagnostic_log:
        diagnostic_log.log("No cached result found", 
                        {"time": time.time() - start_time})
    
    return None

def detect_faces(image, diagnostic_log=None) -> List[Tuple[int, int, int, int]]:
    """
    Detects faces in an image using the face detector
    Returns list of (x, y, width, height) tuples
    Enhanced with diagnostics
    """
    if diagnostic_log:
        diagnostic_log.start_stage("face_detection")
    
    start_time = time.time()
    faces = face_detector.detect_faces(image)
    
    if diagnostic_log:
        detection_time = time.time() - start_time
        diagnostic_log.log(f"Detected {len(faces)} faces in {detection_time:.4f}s", 
                         {"face_count": len(faces), "detection_time": detection_time})
        diagnostic_log.end_stage("face_detection")
    
    return faces

def enhance_image_quality(image, diagnostic_log=None):
    """
    Apply advanced image enhancement for better feature extraction
    - Adjusts lighting
    - Reduces noise
    - Enhances details
    
    Enhanced with quality metrics and diagnostics
    """
    if not ENABLE_ADVANCED_PREPROCESSING:
        return image
    
    if diagnostic_log:
        diagnostic_log.start_stage("image_enhancement")
    
    try:
        # Original image quality assessment
        if diagnostic_log:
            original_blur = measure_blur(image)
            original_contrast = measure_contrast(image)
            original_noise = estimate_noise_level(image)
            
            diagnostic_log.log("Original image quality metrics", {
                "blur": original_blur,
                "contrast": original_contrast,
                "noise": original_noise
            })
        
        # Convert to LAB color space for better lighting adjustment
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel for lighting normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge enhanced L-channel with original A and B channels
        enhanced_lab = cv2.merge((cl, a, b))
        
        # Convert back to BGR color space
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Apply subtle denoising
        enhanced_image = cv2.fastNlMeansDenoisingColored(
            enhanced_image, 
            None, 
            h=5,  # Filter strength (5-10 is usually good)
            hColor=5,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # Enhance details with subtle sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)
        
        # Enhanced image quality assessment
        if diagnostic_log:
            enhanced_blur = measure_blur(enhanced_image)
            enhanced_contrast = measure_contrast(enhanced_image)
            enhanced_noise = estimate_noise_level(enhanced_image)
            
            improved_blur = enhanced_blur > original_blur
            improved_contrast = enhanced_contrast > original_contrast
            improved_noise = enhanced_noise < original_noise
            
            diagnostic_log.log("Enhanced image quality metrics", {
                "blur": enhanced_blur,
                "contrast": enhanced_contrast,
                "noise": enhanced_noise,
                "improvements": {
                    "blur": improved_blur,
                    "contrast": improved_contrast,
                    "noise": improved_noise
                }
            })
            
            # Add to overall quality metrics
            diagnostic_log.add_quality_metric("original_blur", original_blur)
            diagnostic_log.add_quality_metric("enhanced_blur", enhanced_blur)
            diagnostic_log.add_quality_metric("original_contrast", original_contrast)
            diagnostic_log.add_quality_metric("enhanced_contrast", enhanced_contrast)
            diagnostic_log.add_quality_metric("original_noise", original_noise)
            diagnostic_log.add_quality_metric("enhanced_noise", enhanced_noise)
        
        if diagnostic_log:
            diagnostic_log.end_stage("image_enhancement")
        
        return enhanced_image
    except Exception as e:
        if diagnostic_log:
            diagnostic_log.add_error("enhancement_error", f"Image enhancement failed: {str(e)}")
            diagnostic_log.end_stage("image_enhancement")
        
        logger.warning(f"Image enhancement failed: {str(e)}")
        return image  # Return original image on failure

def measure_blur(image):
    """
    Measure the amount of blur in an image
    Higher values indicate less blur (sharper image)
    """
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.uint8)
    
    # Measure 1: Laplacian variance (standard method)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var

def measure_contrast(image):
    """
    Measure image contrast
    Returns a value between 0 and 1, where higher values indicate more contrast
    """
    if len(image.shape) == 3:
        # For color images, convert to grayscale
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.uint8)
    
    # Calculate standard deviation and normalize
    std_dev = np.std(gray)
    # Normalize to [0, 1] range (approximate, as most natural images have std_dev < 80)
    return min(1.0, std_dev / 80.0)

def estimate_noise_level(image):
    """
    Estimate the amount of noise in an image
    Returns a value between 0 and 1, where higher values indicate more noise
    """
    if len(image.shape) == 3:
        # For color images, use luminance
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.uint8)
    
    # Use median filter difference to estimate noise
    filtered = cv2.medianBlur(gray, 3)
    noise = np.mean(np.abs(gray.astype(np.float32) - filtered.astype(np.float32))) / 255.0
    
    return noise

def preprocess_face(image, face, target_size, diagnostic_log=None) -> np.ndarray:
    """
    Extract and preprocess a face for the deepfake detection model
    Enhanced with advanced preprocessing and diagnostics
    """
    if diagnostic_log:
        diagnostic_log.start_stage("face_preprocessing")
        diagnostic_log.log("Preprocessing face", {
            "face": face,
            "target_size": target_size
        })
    
    # Extract face with margin
    extracted_face = face_detector.extract_face(
        image, 
        face, 
        margin_percent=FACE_MARGIN_PERCENT,
        target_size=target_size
    )
    
    if extracted_face is None:
        if diagnostic_log:
            diagnostic_log.add_error("face_extraction", "Failed to extract face from image")
            diagnostic_log.end_stage("face_preprocessing")
        return None
    
    # Apply advanced preprocessing if enabled
    if ENABLE_ADVANCED_PREPROCESSING:
        try:
            # Enhance image quality
            extracted_face = enhance_image_quality(extracted_face, diagnostic_log)
            
            # Normalize pixel values to [0, 1]
            extracted_face = extracted_face.astype(np.float32) / 255.0
            
            # Apply data augmentation during inference for robustness
            # This is a simplified version - in production you'd use a more sophisticated approach
            if np.random.random() < 0.3:  # 30% chance of augmentation during inference
                # Apply random slight rotation (+/- 5 degrees)
                angle = np.random.uniform(-5, 5)
                h, w = extracted_face.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                extracted_face = cv2.warpAffine(extracted_face, M, (w, h))
                
                if diagnostic_log:
                    diagnostic_log.log("Applied random augmentation", {"rotation_angle": angle})
            
            # Re-scale to [0, 255] and convert back to uint8 if needed for the model
            extracted_face = (extracted_face * 255).astype(np.uint8)
            
        except Exception as e:
            if diagnostic_log:
                diagnostic_log.add_error("preprocessing_error", f"Advanced face preprocessing failed: {str(e)}")
            
            logger.warning(f"Advanced face preprocessing failed: {str(e)}")
            # If advanced preprocessing fails, return the basic extracted face
    
    if diagnostic_log:
        # Calculate quality metrics for the preprocessed face
        blur = measure_blur(extracted_face)
        contrast = measure_contrast(extracted_face)
        noise = estimate_noise_level(extracted_face)
        
        # Log quality metrics
        diagnostic_log.log("Preprocessed face quality metrics", {
            "blur": blur,
            "contrast": contrast,
            "noise": noise
        })
        
        # Add quality metrics
        diagnostic_log.add_quality_metric("face_blur", blur)
        diagnostic_log.add_quality_metric("face_contrast", contrast)
        diagnostic_log.add_quality_metric("face_noise", noise)
        
        diagnostic_log.end_stage("face_preprocessing")
    
    return extracted_face

def extract_scene_based_frames(video_path, max_frames=30, diagnostic_log=None):
    """
    Extract frames based on scene changes rather than fixed intervals
    for more intelligent video sampling
    Enhanced with diagnostics
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract
        diagnostic_log: Optional diagnostic log object
        
    Returns:
        List of (frame, frame_number) tuples
    """
    logger.info(f"Extracting scene-based frames from {video_path}")
    frames = []
    
    if diagnostic_log:
        diagnostic_log.start_stage("scene_based_extraction")
        diagnostic_log.log(f"Starting scene-based extraction from {video_path}", 
                         {"max_frames": max_frames})
    
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f"Failed to open video: {video_path}"
            if diagnostic_log:
                diagnostic_log.add_error("video_open_error", error_msg)
            raise ValueError(error_msg)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if diagnostic_log:
            diagnostic_log.log("Video metadata", {
                "total_frames": total_frames,
                "fps": fps,
                "duration": total_frames / max(1, fps)
            })
        
        if total_frames == 0 or fps == 0:
            logger.warning(f"Invalid video metadata: frames={total_frames}, fps={fps}")
            if diagnostic_log:
                diagnostic_log.add_error("video_metadata_error", 
                                     f"Invalid video metadata: frames={total_frames}, fps={fps}")
                diagnostic_log.end_stage("scene_based_extraction")
            return []
        
        # Initialize variables for scene detection
        prev_frame = None
        frame_count = 0
        selected_frames = 0
        min_scene_change_threshold = 30.0  # Minimum difference to consider as scene change
        
        # Set initial reading position
        next_frame_to_read = 0
        
        # Process frames to detect scene changes
        scene_changes = []
        
        while selected_frames < max_frames and frame_count < total_frames:
            # Set position to next frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_to_read)
            ret, frame = cap.read()
            
            if not ret:
                if diagnostic_log:
                    diagnostic_log.add_error("frame_read_error", 
                                         f"Failed to read frame {next_frame_to_read}")
                break
            
            # Convert to grayscale for scene detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # If this is the first frame, always include it
            if prev_frame is None:
                frames.append((frame, next_frame_to_read))
                selected_frames += 1
                prev_frame = gray
                
                if diagnostic_log:
                    diagnostic_log.log(f"Selected first frame: {next_frame_to_read}")
            else:
                # Calculate frame difference to detect scene changes
                frame_diff = cv2.absdiff(prev_frame, gray)
                mean_diff = np.mean(frame_diff)
                
                # Record scene change info
                scene_changes.append({
                    "frame": next_frame_to_read,
                    "difference": float(mean_diff)
                })
                
                # If significant change or if we've gone too many frames without selecting one
                if mean_diff > min_scene_change_threshold or (next_frame_to_read - frame_count) > total_frames / (max_frames * 2):
                    frames.append((frame, next_frame_to_read))
                    selected_frames += 1
                    prev_frame = gray
                    
                    if diagnostic_log:
                        diagnostic_log.log(f"Selected frame {next_frame_to_read} (diff: {mean_diff:.2f})", 
                                       {"difference": mean_diff, 
                                        "is_scene_change": mean_diff > min_scene_change_threshold})
                
            frame_count = next_frame_to_read
            
            # Calculate next frame to check based on remaining frames and needed samples
            frames_remaining = total_frames - frame_count - 1
            samples_needed = max_frames - selected_frames
            
            if samples_needed <= 0:
                break
                
            # Calculate approximate gap between frames
            if frames_remaining <= 0 or samples_needed <= 0:
                break
                
            frame_gap = max(1, frames_remaining // samples_needed)
            next_frame_to_read = frame_count + frame_gap
            
            # Ensure we don't exceed total frames
            if next_frame_to_read >= total_frames:
                next_frame_to_read = total_frames - 1
        
        # Release video capture
        cap.release()
        
        if diagnostic_log:
            # Add scene change data to diagnostic log
            diagnostic_log.log(f"Scene detection complete", {
                "frames_selected": len(frames),
                "scene_changes_detected": len([s for s in scene_changes 
                                              if s["difference"] > min_scene_change_threshold]),
                "scene_changes": scene_changes[:10]  # Just include the first 10 to avoid huge logs
            })
            
            # Update frame counts in metrics
            diagnostic_log.metrics["frame_counts"]["total"] = total_frames
            diagnostic_log.metrics["frame_counts"]["processed"] = len(frames)
            
            diagnostic_log.end_stage("scene_based_extraction")
        
        logger.info(f"Successfully extracted {len(frames)} scene-based frames")
        return frames
        
    except Exception as e:
        logger.error(f"Error extracting scene-based frames: {str(e)}", exc_info=True)
        
        if diagnostic_log:
            diagnostic_log.add_error("scene_extraction_error", 
                                 f"Error extracting scene-based frames: {str(e)}")
            diagnostic_log.end_stage("scene_based_extraction")
        
        # Return any frames we've managed to extract so far
        return frames

def extract_video_frames(video_path, target_frame_positions, diagnostic_log=None):
    """
    Extract frames from a video at specified positions
    Enhanced with diagnostics
    
    Args:
        video_path: Path to the video file
        target_frame_positions: List of frame numbers to extract
        diagnostic_log: Optional diagnostic log object
        
    Returns:
        List of (frame, frame_number) tuples
    """
    logger.info(f"Extracting {len(target_frame_positions)} frames from {video_path}")
    frames = []
    
    if diagnostic_log:
        diagnostic_log.start_stage("frame_extraction")
        diagnostic_log.log(f"Starting frame extraction from {video_path}", 
                         {"frame_positions": target_frame_positions[:10]})  # Log first 10 positions
    
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f"Failed to open video: {video_path}"
            if diagnostic_log:
                diagnostic_log.add_error("video_open_error", error_msg)
            raise ValueError(error_msg)
        
        # Get video metadata
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if diagnostic_log:
            diagnostic_log.log("Video metadata", {
                "total_frames": total_frames,
                "fps": fps,
                "duration": total_frames / max(1, fps)
            })
        
        # Extract each requested frame
        for frame_number in target_frame_positions:
            if frame_number >= total_frames:
                if diagnostic_log:
                    diagnostic_log.add_error("invalid_frame", 
                                         f"Frame {frame_number} exceeds total frames {total_frames}")
                continue
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                frames.append((frame, frame_number))
                
                if diagnostic_log:
                    # Only log brief info to avoid huge logs
                    if len(frames) <= 5 or len(frames) % 10 == 0:
                        frame_quality = {
                            "blur": measure_blur(frame),
                            "contrast": measure_contrast(frame),
                            "noise": estimate_noise_level(frame)
                        }
                        diagnostic_log.log(f"Extracted frame {frame_number}", frame_quality)
            else:
                logger.warning(f"Failed to read frame {frame_number}")
                if diagnostic_log:
                    diagnostic_log.add_error("frame_read_error", 
                                         f"Failed to read frame {frame_number}")
        
        # Release video capture
        cap.release()
        
        if diagnostic_log:
            # Update frame counts in metrics
            diagnostic_log.metrics["frame_counts"]["total"] = total_frames
            diagnostic_log.metrics["frame_counts"]["requested"] = len(target_frame_positions)
            diagnostic_log.metrics["frame_counts"]["processed"] = len(frames)
            
            diagnostic_log.log(f"Frame extraction complete", {
                "requested_frames": len(target_frame_positions),
                "extracted_frames": len(frames),
                "success_rate": len(frames) / max(1, len(target_frame_positions))
            })
            
            diagnostic_log.end_stage("frame_extraction")
        
        logger.info(f"Successfully extracted {len(frames)}/{len(target_frame_positions)} frames")
        return frames
        
    except Exception as e:
        logger.error(f"Error extracting video frames: {str(e)}", exc_info=True)
        
        if diagnostic_log:
            diagnostic_log.add_error("frame_extraction_error", 
                                 f"Error extracting video frames: {str(e)}")
            diagnostic_log.end_stage("frame_extraction")
        
        # Return any frames we've managed to extract so far
        return frames

def get_video_metadata(video_path, diagnostic_log=None):
    """
    Get metadata from a video file
    Enhanced with diagnostics
    
    Args:
        video_path: Path to the video file
        diagnostic_log: Optional diagnostic log object
        
    Returns:
        Dictionary with video metadata
    """
    if diagnostic_log:
        diagnostic_log.start_stage("video_metadata")
    
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f"Failed to open video: {video_path}"
            if diagnostic_log:
                diagnostic_log.add_error("video_open_error", error_msg)
                diagnostic_log.end_stage("video_metadata")
            return {
                "error": error_msg,
                "total_frames": 0,
                "fps": 0,
                "width": 0,
                "height": 0,
                "duration": 0
            }
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / max(1, fps)
        
        # Release the video capture
        cap.release()
        
        metadata = {
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "duration": duration,
            "aspect_ratio": width / max(1, height),
            "codec": "unknown"  # OpenCV doesn't provide codec info easily
        }
        
        if diagnostic_log:
            diagnostic_log.log("Video metadata retrieved", metadata)
            diagnostic_log.end_stage("video_metadata")
        
        return metadata
    
    except Exception as e:
        error_msg = f"Error getting video metadata: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        if diagnostic_log:
            diagnostic_log.add_error("metadata_error", error_msg)
            diagnostic_log.end_stage("video_metadata")
        
        return {
            "error": error_msg,
            "total_frames": 0,
            "fps": 0,
            "width": 0,
            "height": 0,
            "duration": 0
        }

def calculate_frame_positions(video_metadata, max_frames, frame_interval=0.0, diagnostic_log=None):
    """
    Calculate frame positions to extract from a video
    Enhanced with diagnostics
    
    Args:
        video_metadata: Dictionary with video metadata
        max_frames: Maximum number of frames to extract
        frame_interval: Interval between frames (0.0 for uniform distribution)
        diagnostic_log: Optional diagnostic log object
        
    Returns:
        List of frame positions to extract
    """
    if diagnostic_log:
        diagnostic_log.start_stage("frame_position_calculation")
    
    total_frames = video_metadata.get("total_frames", 0)
    
    if total_frames <= 0:
        if diagnostic_log:
            diagnostic_log.add_error("invalid_metadata", 
                                 f"Invalid total_frames: {total_frames}")
            diagnostic_log.end_stage("frame_position_calculation")
        return []
    
    # Adjust max_frames if the video has fewer frames
    max_frames = min(max_frames, total_frames)
    
    # If frame_interval is specified, use it to calculate frame positions
    if frame_interval > 0.0:
        # Convert interval to frames
        fps = video_metadata.get("fps", 30.0)
        interval_frames = int(frame_interval * fps)
        
        # Ensure interval is at least 1 frame
        interval_frames = max(1, interval_frames)
        
        # Calculate frames to extract
        frame_positions = list(range(0, total_frames, interval_frames))
        
        # Limit to max_frames
        frame_positions = frame_positions[:max_frames]
    else:
        # Calculate evenly spaced frame positions
        if max_frames == 1:
            # For a single frame, take the middle frame
            frame_positions = [total_frames // 2]
        else:
            # For multiple frames, distribute evenly
            step = total_frames / (max_frames - 1)
            frame_positions = [int(i * step) for i in range(max_frames - 1)]
            # Add the last frame
            frame_positions.append(total_frames - 1)
    
    if diagnostic_log:
        diagnostic_log.log("Calculated frame positions", {
            "method": "interval" if frame_interval > 0.0 else "uniform",
            "frame_count": len(frame_positions),
            "first_frame": frame_positions[0] if frame_positions else None,
            "last_frame": frame_positions[-1] if frame_positions else None,
            "interval": frame_interval
        })
        diagnostic_log.end_stage("frame_position_calculation")
    
    return frame_positions
