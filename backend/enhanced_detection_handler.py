"""
Enhanced Detection Handler for DeepDefend
Specialized for Indian faces and skin tones with improved accuracy
"""

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
from skin_tone_analyzer import SkinToneAnalyzer
from metrics_logger import log_detection_metrics, log_error
from debug_utils import save_debug_image, log_inference_data

# Configure logging
logger = logging.getLogger(__name__)

# Initialize specialized face preprocessor and skin tone analyzer
face_detector = FaceDetector()
indian_face_preprocessor = IndianFacePreprocessor(face_detector)
skin_tone_analyzer = SkinToneAnalyzer()

def process_image_enhanced(image_path) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Process an image with enhanced Indian face detection and skin tone analysis
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (probability, confidence, regions)
    """
    logger.info(f"Processing image with enhanced Indian face detection: {image_path}")
    
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
        
        # Apply advanced image enhancement using Indian face preprocessor
        enhanced_image = indian_face_preprocessor.enhance_image(image)
        
        # Save enhanced image for debugging
        debug_path = save_debug_image(enhanced_image, "enhanced_image")
        logger.debug(f"Saved enhanced image to {debug_path}")
        
        # Detect faces using Indian-specific face detector
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
        
        # Process each face with enhanced Indian face detection
        processed_faces = []
        for face in faces:
            face_data = indian_face_preprocessor.preprocess_face(
                enhanced_image, 
                face, 
                target_size=(224, 224)
            )
            
            if face_data:
                processed_faces.append(face_data)
        
        # If no faces were successfully processed
        if not processed_faces:
            logger.warning(f"No faces were successfully processed in {image_path}")
            return 0.1, 0.5, []
        
        # Process each face with the model and enhance with skin tone analysis
        regions = []
        face_probabilities = []
        face_confidences = []
        
        for face_data in processed_faces:
            # Get the face image and bounding box
            face_img = face_data["image"]
            bbox = face_data["bbox"]
            
            # Process with deepfake detection model
            # We'll use a single face at a time for more accurate per-face analysis
            model_results = process_image_faces(
                enhanced_image, 
                [bbox], 
                use_ensemble=ENABLE_ENSEMBLE_DETECTION
            )
            
            # Skip if no results
            if not model_results:
                continue
                
            # Get the model result for this face
            model_result = model_results[0]
            
            # Add skin tone analysis to the result
            model_result["metadata"] = model_result.get("metadata", {})
            model_result["metadata"]["skin_tone"] = face_data["skin_tone"]
            model_result["metadata"]["skin_anomalies"] = face_data["skin_anomalies"]
            
            # Analyze face authenticity based on skin tone
            authenticity_analysis = indian_face_preprocessor.analyze_face_authenticity(face_data)
            model_result["metadata"]["authenticity_analysis"] = authenticity_analysis
            
            # Adjust probability and confidence based on skin analysis
            if authenticity_analysis["authenticity_score"] < 0.8:
                # If skin analysis suggests it might be fake, increase the deepfake probability
                adjustment = (1.0 - authenticity_analysis["authenticity_score"]) * 0.2
                model_result["probability"] = min(0.98, model_result["probability"] + adjustment)
            
            # Add to results
            regions.append(model_result)
            face_probabilities.append(model_result["probability"])
            face_confidences.append(model_result["confidence"])
        
        # If no faces were successfully processed
        if not regions:
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
        prediction_consistency = 1.0 - np.std(face_probabilities) if len(face_probabilities) > 1 else 1.0
        calibrated_confidence = calculate_image_confidence(
            overall_probability, 
            face_count=len(face_probabilities),
            face_consistency=prediction_consistency
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
            regions[0]["metadata"]["processor"] = "enhanced_indian_detection"
            regions[0]["metadata"]["faces_processed"] = len(face_probabilities)
            
        # Log metrics for continuous evaluation
        log_detection_metrics({
            "type": "image",
            "faces_detected": len(faces),
            "faces_processed": len(face_probabilities),
            "processing_time": processing_time,
            "probability": overall_probability,
            "confidence": calibrated_confidence,
            "enhanced_indian_detection": True
        })
            
        return overall_probability, calibrated_confidence, regions
        
    except Exception as e:
        logger.error(f"Error in process_image_enhanced: {str(e)}", exc_info=True)
        log_error("enhanced_image_processing", str(e), {"path": image_path})
        # Return a default low probability with low confidence on error
        return 0.1, 0.1, []

def process_video_enhanced(video_path, max_frames=VIDEO_MAX_FRAMES) -> Tuple[float, float, int, List[Dict[str, Any]]]:
    """
    Process a video with enhanced Indian face detection and skin tone analysis
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to analyze
        
    Returns:
        Tuple of (probability, confidence, processed_frame_count, regions)
    """
    logger.info(f"Processing video with enhanced Indian face detection: {video_path}")
    
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
        
        # Process frames with enhanced Indian face detection
        frame_results = []
        
        # Process frames sequentially for better consistency with skin tone analysis
        logger.info(f"Processing {len(frames)} frames sequentially with enhanced Indian detection")
        for frame, frame_number in frames:
            try:
                # Enhance frame with Indian face preprocessor
                enhanced_frame = indian_face_preprocessor.enhance_image(frame)
                
                # Detect faces using Indian face preprocessor
                faces = indian_face_preprocessor.detect_faces(enhanced_frame)
                
                # Fall back to standard detection if needed
                if not faces:
                    faces = detect_faces(enhanced_frame)
                
                if not faces:
                    logger.debug(f"No faces detected in frame {frame_number}")
                    continue
                
                # Process each face with enhanced Indian face detection
                processed_faces = []
                for face in faces:
                    face_data = indian_face_preprocessor.preprocess_face(
                        enhanced_frame, 
                        face, 
                        target_size=(224, 224)
                    )
                    
                    if face_data:
                        processed_faces.append(face_data)
                
                if not processed_faces:
                    logger.debug(f"No faces successfully preprocessed in frame {frame_number}")
                    continue
                
                # Process each face with the model
                frame_face_results = []
                
                for face_data in processed_faces:
                    # Get the face image and bounding box
                    face_img = face_data["image"]
                    bbox = face_data["bbox"]
                    
                    # Process with deepfake detection model
                    model_results = process_image_faces(
                        enhanced_frame, 
                        [bbox], 
                        use_ensemble=ENABLE_ENSEMBLE_DETECTION
                    )
                    
                    # Skip if no results
                    if not model_results:
                        continue
                        
                    # Get the model result for this face
                    model_result = model_results[0]
                    
                    # Add skin tone analysis to the result
                    model_result["metadata"] = model_result.get("metadata", {})
                    model_result["metadata"]["skin_tone"] = face_data["skin_tone"]
                    model_result["metadata"]["skin_anomalies"] = face_data["skin_anomalies"]
                    
                    # Analyze face authenticity based on skin tone
                    authenticity_analysis = indian_face_preprocessor.analyze_face_authenticity(face_data)
                    model_result["metadata"]["authenticity_analysis"] = authenticity_analysis
                    
                    # Adjust probability and confidence based on skin analysis
                    if authenticity_analysis["authenticity_score"] < 0.8:
                        # If skin analysis suggests it might be fake, increase the deepfake probability
                        adjustment = (1.0 - authenticity_analysis["authenticity_score"]) * 0.2
                        model_result["probability"] = min(0.98, model_result["probability"] + adjustment)
                    
                    # Add to frame face results
                    frame_face_results.append(model_result)
                
                # Calculate frame result
                if frame_face_results:
                    # Calculate weighted average based on face size
                    weights = [r["width"] * r["height"] for r in frame_face_results if "width" in r and "height" in r]
                    
                    if sum(weights) > 0:
                        frame_probability = np.average([r["probability"] for r in frame_face_results], weights=weights)
                        frame_confidence = np.average([r["confidence"] for r in frame_face_results], weights=weights)
                    else:
                        frame_probability = np.mean([r["probability"] for r in frame_face_results])
                        frame_confidence = np.mean([r["confidence"] for r in frame_face_results])
                    
                    # Create frame result
                    frame_result = {
                        "frame": frame_number,
                        "probability": frame_probability,
                        "confidence": frame_confidence,
                        "faces": frame_face_results,
                        "status": "processed"
                    }
                    
                    frame_results.append(frame_result)
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
                            # Use the same enhanced processing as above
                            # Enhance frame with Indian face preprocessor
                            enhanced_frame = indian_face_preprocessor.enhance_image(frame)
                            
                            # Detect faces using Indian face preprocessor
                            faces = indian_face_preprocessor.detect_faces(enhanced_frame)
                            
                            # Fall back to standard detection if needed
                            if not faces:
                                faces = detect_faces(enhanced_frame)
                            
                            if not faces:
                                logger.debug(f"No faces detected in fallback frame {frame_number}")
                                continue
                            
                            # Process each face with enhanced Indian face detection
                            processed_faces = []
                            for face in faces:
                                face_data = indian_face_preprocessor.preprocess_face(
                                    enhanced_frame, 
                                    face, 
                                    target_size=(224, 224)
                                )
                                
                                if face_data:
                                    processed_faces.append(face_data)
                            
                            if not processed_faces:
                                logger.debug(f"No faces successfully preprocessed in fallback frame {frame_number}")
                                continue
                            
                            # Process each face with the model
                            frame_face_results = []
                            
                            for face_data in processed_faces:
                                # Get the face image and bounding box
                                face_img = face_data["image"]
                                bbox = face_data["bbox"]
                                
                                # Process with deepfake detection model
                                model_results = process_image_faces(
                                    enhanced_frame, 
                                    [bbox], 
                                    use_ensemble=ENABLE_ENSEMBLE_DETECTION
                                )
                                
                                # Skip if no results
                                if not model_results:
                                    continue
                                    
                                # Get the model result for this face
                                model_result = model_results[0]
                                
                                # Add skin tone analysis to the result
                                model_result["metadata"] = model_result.get("metadata", {})
                                model_result["metadata"]["skin_tone"] = face_data["skin_tone"]
                                model_result["metadata"]["skin_anomalies"] = face_data["skin_anomalies"]
                                
                                # Analyze face authenticity based on skin tone
                                authenticity_analysis = indian_face_preprocessor.analyze_face_authenticity(face_data)
                                model_result["metadata"]["authenticity_analysis"] = authenticity_analysis
                                
                                # Adjust probability and confidence based on skin analysis
                                if authenticity_analysis["authenticity_score"] < 0.8:
                                    # If skin analysis suggests it might be fake, increase the deepfake probability
                                    adjustment = (1.0 - authenticity_analysis["authenticity_score"]) * 0.2
                                    model_result["probability"] = min(0.98, model_result["probability"] + adjustment)
                                
                                # Add to frame face results
                                frame_face_results.append(model_result)
                            
                            # Calculate frame result
                            if frame_face_results:
                                # Calculate weighted average based on face size
                                weights = [r["width"] * r["height"] for r in frame_face_results if "width" in r and "height" in r]
                                
                                if sum(weights) > 0:
                                    frame_probability = np.average([r["probability"] for r in frame_face_results], weights=weights)
                                    frame_confidence = np.average([r["confidence"] for r in frame_face_results], weights=weights)
                                else:
                                    frame_probability = np.mean([r["probability"] for r in frame_face_results])
                                    frame_confidence = np.mean([r["confidence"] for r in frame_face_results])
                                
                                # Create frame result
                                frame_result = {
                                    "frame": frame_number,
                                    "probability": frame_probability,
                                    "confidence": frame_confidence,
                                    "faces": frame_face_results,
                                    "status": "processed"
                                }
                                
                                fallback_results.append(frame_result)
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
            logger.info(f"Temporal consistency analysis: {temporal_consistency:.4f}")
        
        # Aggregate results across frames
        overall_probability, overall_confidence, regions = aggregate_video_results(
            frame_results, 
            temporal_consistency=temporal_consistency,
            temporal_weight=TEMPORAL_CONSISTENCY_WEIGHT
        )
        
        # Log processing time
        processing_time = time.time() - start_time
        logger.info(f"Processed video in {processing_time:.2f} seconds, analyzed {processed_frame_count} frames")
        logger.info(f"Result: probability={overall_probability:.4f}, confidence={overall_confidence:.4f}")
        
        # Add metadata to the first region
        if regions:
            regions[0]["metadata"] = regions[0].get("metadata", {})
            regions[0]["metadata"]["processing_time"] = processing_time
            regions[0]["metadata"]["frames_processed"] = processed_frame_count
            regions[0]["metadata"]["frames_extracted"] = len(frames)
            regions[0]["metadata"]["processor"] = "enhanced_indian_detection"
            regions[0]["metadata"]["temporal_consistency"] = temporal_consistency
            
            # Add temporal analysis data
            regions[0]["metadata"]["temporal_analysis"] = {
                "frame_count": processed_frame_count,
                "frame_consistency": temporal_consistency if temporal_consistency is not None else 1.0,
                "frame_predictions": [
                    {"frame": r["frame"], "probability": r["probability"], "confidence": r["confidence"]}
                    for r in frame_results if r.get("status") == "processed"
                ]
            }
        
        # Log metrics for continuous evaluation
        log_detection_metrics({
            "type": "video",
            "frames_processed": processed_frame_count,
            "frames_extracted": len(frames),
            "processing_time": processing_time,
            "probability": overall_probability,
            "confidence": overall_confidence,
            "temporal_consistency": temporal_consistency if temporal_consistency is not None else 1.0,
            "enhanced_indian_detection": True
        })
        
        return overall_probability, overall_confidence, processed_frame_count, regions
        
    except Exception as e:
        logger.error(f"Error in process_video_enhanced: {str(e)}", exc_info=True)
        log_error("enhanced_video_processing", str(e), {"path": video_path})
        # Return a default low probability with low confidence on error
        return 0.1, 0.1, 0, []