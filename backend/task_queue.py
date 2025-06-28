
import celery
import os
import time
import logging
import uuid
import json
from celery import Celery
import numpy as np
from backend.config import DEBUG_MODE_ENABLED, MIN_FRAMES_FOR_VALID_DETECTION

# Configure logging
logger = logging.getLogger(__name__)

# Configure Celery
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
celery_app = Celery('deepdefend', broker=redis_url, backend=redis_url)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour timeout for tasks
    task_soft_time_limit=3000,  # 50 minutes soft timeout
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks
    worker_prefetch_multiplier=1,  # Don't prefetch tasks
)

# Store task metadata in memory (for demonstration, in production use Redis/DB)
task_store = {}

@celery_app.task(bind=True, name='process_video')
def process_video_task(self, video_path, options=None):
    """
    Celery task to process a video asynchronously
    
    Args:
        video_path: Path to the video file
        options: Dictionary of processing options
        
    Returns:
        Dictionary with detection results
    """
    # Import here to avoid circular imports
    from detection_handler import process_video
    
    try:
        logger.info(f"Starting async video processing task for {video_path}")
        debug_mode = options.get('debug', DEBUG_MODE_ENABLED) if options else DEBUG_MODE_ENABLED
        
        # Record task metadata
        task_id = self.request.id
        task_store[task_id] = {
            'status': 'processing',
            'start_time': time.time(),
            'video_path': video_path,
            'progress': 0,
            'debug_mode': debug_mode
        }
        
        # Process the video with detailed logging if debug mode is enabled
        if debug_mode:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled for this task")
        
        # Process the video
        probability, confidence, frame_count, regions = process_video(video_path)
        
        # Calculate processing time
        processing_time = int((time.time() - task_store[task_id]['start_time']) * 1000)  # in milliseconds
        
        # Validate results
        if frame_count < MIN_FRAMES_FOR_VALID_DETECTION:
            logger.warning(f"Not enough frames processed ({frame_count}), results may be unreliable")
        
        # Record model info
        from inference import get_model_info
        model_info = get_model_info()
        current_model = model_info.get('models', {}).get('current_model', 'unknown')
        
        # Log detailed info for debugging
        if debug_mode:
            logger.debug(f"Video processing results: probability={probability:.4f}, confidence={confidence:.4f}")
            logger.debug(f"Processed {frame_count} frames in {processing_time}ms using {current_model}")
            
        # Prepare the result
        result = {
            "imageName": os.path.basename(video_path),
            "probability": float(probability),
            "confidence": float(confidence),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "detectionType": "video",
            "model": current_model,
            "frameCount": frame_count,
            "processingTime": processing_time,
            "regions": [r for r in regions if isinstance(r, dict)]  # Ensure regions are serializable
        }
        
        # Update task metadata
        task_store[task_id] = {
            'status': 'completed',
            'start_time': task_store[task_id]['start_time'],
            'end_time': time.time(),
            'video_path': video_path,
            'progress': 100,
            'result': result
        }
        
        # Record metrics
        from metrics import performance_metrics
        performance_metrics.record_video_metrics(
            processing_time/1000,  # Convert to seconds
            frame_count,
            sum(1 for r in regions if r.get('status', '') == 'success'),
            confidence,
            probability,
            current_model
        )
        
        # Save to database if possible
        try:
            from database import save_detection_result
            save_detection_result(result)
        except Exception as db_error:
            logger.error(f"Error saving task result to database: {str(db_error)}")
        
        logger.info(f"Completed async video processing task for {video_path}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in async video processing task: {str(e)}", exc_info=True)
        
        # Update task metadata
        if self.request.id in task_store:
            task_store[self.request.id].update({
                'status': 'failed',
                'end_time': time.time(),
                'error': str(e)
            })
        
        # Record error in metrics
        try:
            from metrics import performance_metrics
            performance_metrics.record_error("async_video_processing")
        except Exception:
            pass
            
        # Re-raise exception for Celery to handle
        raise

def submit_video_task(video_path, options=None):
    """
    Submit a video processing task to the Celery queue
    
    Args:
        video_path: Path to the video file
        options: Dictionary of processing options
        
    Returns:
        Dictionary with task ID and status
    """
    try:
        # Submit task to Celery
        task = process_video_task.delay(video_path, options)
        
        # Store initial task metadata
        task_store[task.id] = {
            'status': 'submitted',
            'start_time': time.time(),
            'video_path': video_path,
            'progress': 0,
            'debug_mode': options.get('debug', DEBUG_MODE_ENABLED) if options else DEBUG_MODE_ENABLED
        }
        
        logger.info(f"Submitted video processing task: {task.id}")
        
        return {
            'task_id': task.id,
            'status': 'submitted',
            'video_path': video_path
        }
        
    except Exception as e:
        logger.error(f"Error submitting video task: {str(e)}", exc_info=True)
        raise

def get_task_status(task_id):
    """
    Get the status of a task
    
    Args:
        task_id: Celery task ID
        
    Returns:
        Dictionary with task status and metadata
    """
    try:
        # Check if we have local metadata
        if task_id in task_store:
            status_data = task_store[task_id]
        else:
            # Check Celery for task status
            task = celery_app.AsyncResult(task_id)
            
            if task.state == 'PENDING':
                status_data = {
                    'status': 'pending',
                    'progress': 0
                }
            elif task.state == 'STARTED':
                status_data = {
                    'status': 'processing',
                    'progress': 10  # Assuming early stage of processing
                }
            elif task.state == 'SUCCESS':
                status_data = {
                    'status': 'completed',
                    'progress': 100,
                    'result': task.result
                }
            else:  # FAILURE, REVOKED, etc.
                status_data = {
                    'status': 'failed',
                    'progress': 0,
                    'error': str(task.result) if task.result else 'Unknown error'
                }
        
        # Calculate time elapsed if possible
        if 'start_time' in status_data:
            end_time = status_data.get('end_time', time.time())
            status_data['elapsed_seconds'] = round(end_time - status_data['start_time'], 2)
        
        # Return a copy to prevent modification
        return dict(status_data)
        
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}", exc_info=True)
        return {
            'status': 'unknown',
            'error': str(e)
        }
