
import cv2
import numpy as np
import time
import os
import logging
from typing import Tuple, List, Dict, Any, Optional
import concurrent.futures
from preprocessing import (
    detect_faces, get_video_metadata, calculate_frame_positions,
    extract_video_frames, extract_scene_based_frames, enhance_image_quality
)
from inference import (
    process_image_faces, process_frame, get_model_info, switch_model,
    get_ensemble_weights, update_ensemble_weights
)
from postprocessing import (
    calculate_image_confidence, aggregate_video_results,
    analyze_temporal_consistency
)
from backend.config import (
    VIDEO_MAX_FRAMES, VIDEO_FRAME_INTERVAL, ENABLE_PARALLEL_PROCESSING, MAX_WORKERS,
    ENABLE_ENSEMBLE_DETECTION, USE_SCENE_BASED_SAMPLING, ENABLE_TEMPORAL_ANALYSIS,
    MIN_FRAMES_FOR_VALID_DETECTION, TEMPORAL_CONSISTENCY_WEIGHT
)
from indian_face_utils import IndianFacePreprocessor
from face_detector import FaceDetector
from metrics_logger import log_detection_metrics, log_error
from debug_utils import save_debug_image, log_inference_data

# Configure logging
logger = logging.getLogger(__name__)

# Initialize specialized face preprocessor
face_detector = FaceDetector()
indian_face_preprocessor = IndianFacePreprocessor(face_detector)

def process_image(image_path) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Process an image and return deepfake probability
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (probability, confidence, regions)
    """
    logger.info(f"Processing image: {image_path}")
    
    try:
        # Record processing start time
        start_time = time.time()
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        # Save original image for debugging
        debug_path = save_debug_image(image, "original_image")
        logger.debug(f"Saved original image to {debug_path}")
        
        # Apply advanced image enhancement
        enhanced_image = enhance_image_quality(image)
        
        # Save enhanced image for debugging
        debug_path = save_debug_image(enhanced_image, "enhanced_image")
        logger.debug(f"Saved enhanced image to {debug_path}")
        
        # Try using the Indian-specific face detector first
        faces = indian_face_preprocessor.detect_faces(enhanced_image)
        
        # Fall back to regular face detection if no faces detected
        if len(faces) == 0:
            logger.info(f"No faces detected using Indian-specific detector, falling back to standard detector")
            faces = detect_faces(enhanced_image)
        
        if len(faces) == 0:
            logger.warning(f"No faces detected in {image_path}")
            return 0.1, 0.5, []  # Low probability, medium confidence if no faces detected
        
        # Save image with detected faces for debugging
        debug_path = save_debug_image(enhanced_image, "detected_faces", 
                             [{"x": x, "y": y, "width": w, "height": h} for x, y, w, h in faces])
        logger.debug(f"Saved image with detected faces to {debug_path}")
        
        # Process faces with ensemble if enabled
        face_probabilities, face_confidences, regions, prediction_consistency = process_image_faces(
            enhanced_image, faces, use_ensemble=ENABLE_ENSEMBLE_DETECTION
        )
        
        # If no faces were successfully processed
        if not face_probabilities:
            logger.warning(f"No faces were successfully processed in {image_path}")
            return 0.1, 0.5, []
        
        # Calculate overall probability
        # For single face, use that probability
        # For multiple faces, use weighted average based on face size
        if len(face_probabilities) == 1:
            overall_probability = face_probabilities[0]
            overall_confidence = face_confidences[0]
        else:
            # Weight by face area (larger faces contribute more)
            weights = [r["width"] * r["height"] for r in regions if "width" in r and "height" in r]
            if sum(weights) > 0:
                overall_probability = np.average(face_probabilities, weights=weights)
                # Adjust confidence based on consistency between faces
                # If faces disagree, lower the confidence
                face_consistency = 1.0 - np.std(face_probabilities) 
                overall_confidence = np.average(face_confidences, weights=weights) * face_consistency
            else:
                # Simple average if weights are invalid
                overall_probability = np.mean(face_probabilities)
                overall_confidence = np.mean(face_confidences)
        
        # Calibrate confidence for more realistic scores
        # This addresses the tendency to have inflated confidence on real images
        calibrated_confidence = calculate_image_confidence(
            overall_probability, 
            face_count=len(face_probabilities),
            face_consistency=prediction_consistency if prediction_consistency is not None else 1.0
        )
        
        # Log processing time
        processing_time = time.time() - start_time
        logger.info(f"Processed image in {processing_time:.2f} seconds, detected {len(faces)} faces")
        logger.info(f"Result: probability={overall_probability:.4f}, confidence={calibrated_confidence:.4f}")
        
        # Add processing time to the first region metadata
        if regions:
            regions[0]["metadata"] = regions[0].get("metadata", {})
            regions[0]["metadata"]["processing_time"] = processing_time
            regions[0]["metadata"]["face_count"] = len(faces)
            regions[0]["metadata"]["processor"] = "indian_optimized"
            regions[0]["metadata"]["faces_processed"] = len(face_probabilities)
            
        # Log metrics for continuous evaluation
        log_detection_metrics({
            "type": "image",
            "faces_detected": len(faces),
            "faces_processed": len(face_probabilities),
            "processing_time": processing_time,
            "probability": overall_probability,
            "confidence": calibrated_confidence
        })
            
        return overall_probability, calibrated_confidence, regions
        
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}", exc_info=True)
        log_error("image_processing", str(e), {"path": image_path})
        # Return a default low probability with low confidence on error
        return 0.1, 0.1, []

def process_video(video_path, max_frames=VIDEO_MAX_FRAMES) -> Tuple[float, float, int, List[Dict[str, Any]]]:
    """
    Process a video and return deepfake probability by analyzing frames
    Enhanced with scene-based frame extraction and temporal consistency analysis
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to analyze
        
    Returns:
        Tuple of (probability, confidence, processed_frame_count, regions)
    """
    logger.info(f"Processing video: {video_path}")
    
    try:
        # Record processing start time
        start_time = time.time()
        
        # Get video metadata
        metadata = get_video_metadata(video_path)
        if metadata["total_frames"] == 0:
            raise ValueError(f"Failed to get metadata for video: {video_path}")
        
        logger.info(f"Video info: {metadata['total_frames']} frames, {metadata['fps']} fps, {metadata['duration']:.2f} seconds")
        
        # Extract frames using scene-based detection if enabled
        frames = []
        if USE_SCENE_BASED_SAMPLING:
            logger.info("Using scene-based frame sampling")
            frames = extract_scene_based_frames(video_path, max_frames)
        
        # Fall back to standard frame extraction if scene-based extraction failed or is disabled
        if not frames:
            # Calculate which frames to process
            target_frame_positions = calculate_frame_positions(
                metadata, 
                max_frames,
                frame_interval=VIDEO_FRAME_INTERVAL
            )
            
            if not target_frame_positions:
                logger.warning(f"No frames to process in {video_path}")
                return 0.1, 0.5, 0, []
            
            # Extract frames
            logger.info(f"Using standard frame sampling at positions: {target_frame_positions}")
            frames = extract_video_frames(video_path, target_frame_positions)
        
        if not frames:
            logger.warning(f"Failed to extract any frames from {video_path}")
            return 0.1, 0.5, 0, []
        
        logger.info(f"Successfully extracted {len(frames)} frames for processing")
        
        # Process frames with enhanced logging
        frame_results = []
        
        if ENABLE_PARALLEL_PROCESSING and len(frames) > 1:
            # Process frames in parallel
            logger.info(f"Processing {len(frames)} frames in parallel with {min(MAX_WORKERS, len(frames))} workers")
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(frames))) as executor:
                future_to_frame = {
                    executor.submit(process_frame, frame, frame_number, ENABLE_ENSEMBLE_DETECTION): (frame, frame_number)
                    for frame, frame_number in frames
                }
                
                # Collect results
                for future in concurrent.futures.as_completed(future_to_frame):
                    try:
                        result = future.result()
                        frame_results.append(result)
                    except Exception as e:
                        frame, frame_number = future_to_frame[future]
                        logger.error(f"Error processing frame {frame_number}: {str(e)}")
        else:
            # Process frames sequentially
            logger.info(f"Processing {len(frames)} frames sequentially")
            for frame, frame_number in frames:
                try:
                    result = process_frame(frame, frame_number, ENABLE_ENSEMBLE_DETECTION)
                    frame_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing frame {frame_number}: {str(e)}")
        
        # Count successfully processed frames
        processed_frame_count = sum(1 for r in frame_results if r.get('status') == 'processed')
        logger.info(f"Successfully processed {processed_frame_count}/{len(frames)} frames")
        
        # Check if we have enough processed frames
        if processed_frame_count < MIN_FRAMES_FOR_VALID_DETECTION:
            logger.warning(f"Insufficient processed frames: {processed_frame_count} < {MIN_FRAMES_FOR_VALID_DETECTION}")
            
            # If we have too few processed frames, try alternative processing
            # Fallback to process more frames with different parameters
            if processed_frame_count < MIN_FRAMES_FOR_VALID_DETECTION/2 and metadata["total_frames"] > max_frames:
                logger.info(f"Attempting fallback processing with different parameters")
                
                # Try with more frames and different sampling
                new_max_frames = min(max_frames * 2, 60)  # Double frames, max 60
                fallback_frames = []
                
                # Try different frame extraction method than the first attempt
                if USE_SCENE_BASED_SAMPLING:
                    # Calculate evenly spaced frames
                    logger.info(f"Fallback to uniform sampling with {new_max_frames} frames")
                    fallback_positions = calculate_frame_positions(metadata, new_max_frames, frame_interval=0.0)
                    fallback_frames = extract_video_frames(video_path, fallback_positions)
                else:
                    # Try scene-based sampling
                    logger.info(f"Fallback to scene-based sampling with {new_max_frames} frames")
                    fallback_frames = extract_scene_based_frames(video_path, new_max_frames)
                
                if fallback_frames:
                    logger.info(f"Fallback extracted {len(fallback_frames)} frames")
                    
                    # Process these frames (always sequential for stability)
                    fallback_results = []
                    for frame, frame_number in fallback_frames:
                        try:
                            result = process_frame(frame, frame_number, ENABLE_ENSEMBLE_DETECTION)
                            fallback_results.append(result)
                        except Exception as e:
                            logger.error(f"Error in fallback processing frame {frame_number}: {str(e)}")
                    
                    # Count successfully processed fallback frames
                    fallback_processed = sum(1 for r in fallback_results if r.get('status') == 'processed')
                    logger.info(f"Fallback successfully processed {fallback_processed}/{len(fallback_frames)} frames")
                    
                    # If fallback helped, use those results instead
                    if fallback_processed > processed_frame_count:
                        logger.info(f"Using fallback results: {fallback_processed} vs {processed_frame_count} frames")
                        frame_results = fallback_results
                        processed_frame_count = fallback_processed
        
        # Perform temporal consistency analysis if enabled and we have enough frames
        temporal_consistency = None
        if ENABLE_TEMPORAL_ANALYSIS and processed_frame_count >= 3:
            temporal_consistency = analyze_temporal_consistency(frame_results)
            logger.info(f"Temporal consistency score: {temporal_consistency:.4f}")
            
            # Log detailed temporal metrics
            log_inference_data("temporal_analysis", {
                "consistency_score": float(temporal_consistency),
                "frame_count": processed_frame_count,
                "frame_probabilities": [r.get('probability', 0.5) for r in frame_results if r.get('status') == 'processed']
            })
        
        # Aggregate results with temporal consistency if available
        overall_probability, confidence, all_regions = aggregate_video_results(
            frame_results, 
            processed_frame_count, 
            len(frames),
            temporal_consistency
        )
        
        # Apply additional calibration for videos
        # Videos sometimes need different calibration than images
        if overall_probability > 0.7:
            # High probability (likely deepfake)
            # We're more confident in high deepfake scores
            confidence = min(confidence * 1.1, 0.95)
        elif overall_probability < 0.3:
            # Low probability (likely real)
            # We're slightly less confident in real videos due to potential for manipulation
            confidence = min(confidence * 0.95, 0.95)
            
        # If temporal consistency is very poor, reduce confidence
        if temporal_consistency is not None and temporal_consistency < 0.5:
            logger.warning(f"Poor temporal consistency: {temporal_consistency:.4f}, reducing confidence")
            confidence = confidence * (0.5 + temporal_consistency / 2)
            
        # Log processing time
        processing_time = time.time() - start_time
        logger.info(f"Processed video in {processing_time:.2f} seconds")
        logger.info(f"Final result: probability={overall_probability:.4f}, confidence={confidence:.4f}")
        
        # Add detailed metrics to output
        processing_metrics = {
            "total_time": processing_time,
            "frames": {
                "requested": len(frames),
                "processed": processed_frame_count,
                "coverage": processed_frame_count / max(1, len(frames))
            },
            "faces": {
                "total": sum(result.get('face_count', 0) for result in frame_results),
                "per_frame": sum(result.get('face_count', 0) for result in frame_results) / max(1, processed_frame_count)
            },
            "temporal_consistency": temporal_consistency if temporal_consistency is not None else "not_analyzed",
            "ensemble_used": ENABLE_ENSEMBLE_DETECTION,
            "model": get_model_info().get('models', {}).get('current_model', 'unknown'),
            "processor": "indian_optimized",
            "scene_based_sampling": USE_SCENE_BASED_SAMPLING,
            "fallback_processing": len(frames) != len(frame_results)
        }
        
        # Add frame scores for debugging
        processing_metrics["frame_scores"] = [
            {"frame": r.get("frame_number"), 
             "probability": r.get("probability", 0), 
             "confidence": r.get("confidence", 0)}
            for r in frame_results if r.get("status") == "processed"
        ]
        
        # Add metadata to the first region if we have any
        if all_regions:
            if 'metadata' not in all_regions[0]:
                all_regions[0]['metadata'] = {}
            all_regions[0]['metadata']["processing_metrics"] = processing_metrics
        
        # Log metrics for continuous evaluation
        log_detection_metrics({
            "type": "video",
            "frames_requested": len(frames),
            "frames_processed": processed_frame_count,
            "processing_time": processing_time,
            "probability": overall_probability,
            "confidence": confidence,
            "temporal_consistency": temporal_consistency
        })
        
        return overall_probability, confidence, processed_frame_count, all_regions
        
    except Exception as e:
        logger.error(f"Error in process_video: {str(e)}", exc_info=True)
        log_error("video_processing", str(e), {"path": video_path})
        # Return a default low probability with low confidence on error
        return 0.1, 0.1, 0, []
