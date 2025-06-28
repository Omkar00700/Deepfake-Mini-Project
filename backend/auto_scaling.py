
"""
Auto-scaling for DeepDefend
Provides auto-scaling capabilities for Celery workers based on queue size
"""

import os
import logging
import time
import subprocess
import threading
import json
from typing import Dict, Any, Optional, List
from backend.config import (
    AUTO_SCALING_ENABLED,
    CELERY_MIN_WORKERS,
    CELERY_MAX_WORKERS,
    CELERY_WORKER_SCALE_UP_THRESHOLD,
    CELERY_WORKER_SCALE_DOWN_THRESHOLD,
    CELERY_WORKER_SCALE_FACTOR,
    CELERY_WORKER_SCALE_DOWN_DELAY
)
from monitoring_service import record_metric, send_alert

# Configure logging
logger = logging.getLogger(__name__)

class CeleryAutoScaler:
    """
    Auto-scaler for Celery workers
    
    Features:
    - Scale workers based on queue size
    - Respect minimum and maximum worker counts
    - Scale up quickly, scale down gradually
    - Support different scaling strategies
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CeleryAutoScaler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.enabled = AUTO_SCALING_ENABLED
        self.min_workers = CELERY_MIN_WORKERS
        self.max_workers = CELERY_MAX_WORKERS
        self.scale_up_threshold = CELERY_WORKER_SCALE_UP_THRESHOLD
        self.scale_down_threshold = CELERY_WORKER_SCALE_DOWN_THRESHOLD
        self.scale_factor = CELERY_WORKER_SCALE_FACTOR
        self.scale_down_delay = CELERY_WORKER_SCALE_DOWN_DELAY
        
        self.current_workers = self.min_workers
        self.last_scale_up = 0
        self.last_scale_down = 0
        self.running = False
        self.lock = threading.RLock()
        
        # Start monitoring if enabled
        if self.enabled:
            self.start()
    
    def start(self):
        """Start the auto-scaler"""
        if self.running:
            return
            
        self.running = True
        
        # Start monitoring thread
        thread = threading.Thread(target=self._monitor_loop, daemon=True)
        thread.start()
        
        logger.info(f"Celery auto-scaler started (min: {self.min_workers}, max: {self.max_workers})")
    
    def stop(self):
        """Stop the auto-scaler"""
        self.running = False
        logger.info("Celery auto-scaler stopped")
    
    def update_config(self, config: Dict[str, Any]):
        """
        Update auto-scaler configuration
        
        Args:
            config: New configuration values
        """
        with self.lock:
            if "min_workers" in config:
                self.min_workers = max(1, int(config["min_workers"]))
                
            if "max_workers" in config:
                self.max_workers = max(self.min_workers, int(config["max_workers"]))
                
            if "scale_up_threshold" in config:
                self.scale_up_threshold = max(1, int(config["scale_up_threshold"]))
                
            if "scale_down_threshold" in config:
                self.scale_down_threshold = max(0, int(config["scale_down_threshold"]))
                
            if "scale_factor" in config:
                self.scale_factor = max(1.0, float(config["scale_factor"]))
                
            if "scale_down_delay" in config:
                self.scale_down_delay = max(0, int(config["scale_down_delay"]))
            
            logger.info(f"Auto-scaler configuration updated: min={self.min_workers}, max={self.max_workers}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current auto-scaler configuration
        
        Returns:
            Dictionary with current configuration
        """
        with self.lock:
            return {
                "enabled": self.enabled,
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
                "scale_up_threshold": self.scale_up_threshold,
                "scale_down_threshold": self.scale_down_threshold,
                "scale_factor": self.scale_factor,
                "scale_down_delay": self.scale_down_delay,
                "current_workers": self.current_workers,
                "last_scale_up": self.last_scale_up,
                "last_scale_down": self.last_scale_down
            }
    
    def _monitor_loop(self):
        """Main monitoring loop for auto-scaling"""
        while self.running:
            try:
                # Get queue size
                queue_size = self._get_queue_size()
                
                # Record metric
                record_metric("task_queue_size", queue_size)
                
                # Check if scaling is needed
                self._check_scaling(queue_size)
                
                # Sleep for a while
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in auto-scaler monitoring loop: {str(e)}")
                time.sleep(60)  # Sleep longer on error
    
    def _get_queue_size(self) -> int:
        """
        Get current queue size
        
        Returns:
            Number of tasks in the queue
        """
        try:
            # Try to get queue size from Celery API
            # This requires the Celery app to be available
            from celery.task.control import inspect
            insp = inspect()
            
            # Check active tasks
            active = insp.active() or {}
            active_count = sum(len(tasks) for worker, tasks in active.items())
            
            # Check reserved tasks (acknowledged but not yet executed)
            reserved = insp.reserved() or {}
            reserved_count = sum(len(tasks) for worker, tasks in reserved.items())
            
            # Check scheduled tasks
            scheduled = insp.scheduled() or {}
            scheduled_count = sum(len(tasks) for worker, tasks in scheduled.items())
            
            # Check tasks in main queue
            # This requires additional configuration to access the broker directly
            # For now, we'll use the sum of active, reserved, and scheduled
            
            return active_count + reserved_count + scheduled_count
            
        except Exception as e:
            logger.error(f"Error getting queue size: {str(e)}")
            return 0
    
    def _check_scaling(self, queue_size: int):
        """
        Check if scaling is needed based on queue size
        
        Args:
            queue_size: Current queue size
        """
        with self.lock:
            current_time = time.time()
            
            # Check if scale up is needed
            if queue_size > self.scale_up_threshold and self.current_workers < self.max_workers:
                # Calculate target workers
                target_workers = min(
                    self.max_workers,
                    int(self.current_workers * self.scale_factor)
                )
                target_workers = max(target_workers, self.current_workers + 1)
                
                # Scale up
                if self._scale_to(target_workers):
                    logger.info(f"Scaled up workers: {self.current_workers} -> {target_workers} (queue size: {queue_size})")
                    self.last_scale_up = current_time
                    
                    # Record metrics
                    record_metric("scale_up_events", 1)
                    record_metric("worker_count", target_workers)
                    
                    # Send alert
                    send_alert(
                        alert_type="worker_scale_up",
                        message=f"Scaled up Celery workers to {target_workers}",
                        severity="info",
                        details={
                            "previous_count": self.current_workers,
                            "new_count": target_workers,
                            "queue_size": queue_size,
                            "threshold": self.scale_up_threshold
                        }
                    )
                    
                    # Update current workers
                    self.current_workers = target_workers
            
            # Check if scale down is needed
            elif (queue_size < self.scale_down_threshold and 
                  self.current_workers > self.min_workers and
                  current_time - self.last_scale_up > self.scale_down_delay):
                  
                # Calculate target workers
                target_workers = max(
                    self.min_workers,
                    int(self.current_workers / self.scale_factor)
                )
                
                # Scale down
                if self._scale_to(target_workers):
                    logger.info(f"Scaled down workers: {self.current_workers} -> {target_workers} (queue size: {queue_size})")
                    self.last_scale_down = current_time
                    
                    # Record metrics
                    record_metric("scale_down_events", 1)
                    record_metric("worker_count", target_workers)
                    
                    # Send alert
                    send_alert(
                        alert_type="worker_scale_down",
                        message=f"Scaled down Celery workers to {target_workers}",
                        severity="info",
                        details={
                            "previous_count": self.current_workers,
                            "new_count": target_workers,
                            "queue_size": queue_size,
                            "threshold": self.scale_down_threshold
                        }
                    )
                    
                    # Update current workers
                    self.current_workers = target_workers
    
    def _scale_to(self, target_workers: int) -> bool:
        """
        Scale to the target number of workers
        
        Args:
            target_workers: Target number of workers
            
        Returns:
            True if scaling was successful, False otherwise
        """
        # Implementation depends on deployment environment
        
        # For Kubernetes, you would update the deployment
        if self._is_kubernetes():
            return self._scale_kubernetes(target_workers)
            
        # For Docker Compose, you would update the service
        elif self._is_docker_compose():
            return self._scale_docker_compose(target_workers)
            
        # For local deployment, you would start/stop worker processes
        else:
            return self._scale_local(target_workers)
    
    def _is_kubernetes(self) -> bool:
        """
        Check if running in Kubernetes
        
        Returns:
            True if running in Kubernetes, False otherwise
        """
        return os.path.exists("/var/run/secrets/kubernetes.io")
    
    def _is_docker_compose(self) -> bool:
        """
        Check if running in Docker Compose
        
        Returns:
            True if running in Docker Compose, False otherwise
        """
        return os.environ.get("COMPOSE_PROJECT_NAME") is not None
    
    def _scale_kubernetes(self, target_workers: int) -> bool:
        """
        Scale workers in Kubernetes
        
        Args:
            target_workers: Target number of workers
            
        Returns:
            True if scaling was successful, False otherwise
        """
        try:
            # Use kubectl to scale the deployment
            deployment_name = os.environ.get("K8S_WORKER_DEPLOYMENT", "deepdefend-worker")
            namespace = os.environ.get("K8S_NAMESPACE", "default")
            
            cmd = [
                "kubectl", "scale", "deployment", deployment_name,
                "--replicas", str(target_workers),
                "--namespace", namespace
            ]
            
            # Run the command
            process = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                timeout=30
            )
            
            # Check if successful
            if process.returncode == 0:
                logger.info(f"Successfully scaled Kubernetes deployment {deployment_name} to {target_workers} replicas")
                return True
            else:
                error = process.stderr.decode("utf-8")
                logger.error(f"Failed to scale Kubernetes deployment: {error}")
                return False
                
        except Exception as e:
            logger.error(f"Error scaling Kubernetes deployment: {str(e)}")
            return False
    
    def _scale_docker_compose(self, target_workers: int) -> bool:
        """
        Scale workers in Docker Compose
        
        Args:
            target_workers: Target number of workers
            
        Returns:
            True if scaling was successful, False otherwise
        """
        try:
            # Use docker-compose to scale the service
            service_name = os.environ.get("COMPOSE_SERVICE_NAME", "worker")
            project_name = os.environ.get("COMPOSE_PROJECT_NAME", "deepdefend")
            
            cmd = [
                "docker-compose", "-p", project_name,
                "up", "-d", "--scale", f"{service_name}={target_workers}"
            ]
            
            # Run the command
            process = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                timeout=60
            )
            
            # Check if successful
            if process.returncode == 0:
                logger.info(f"Successfully scaled Docker Compose service {service_name} to {target_workers} replicas")
                return True
            else:
                error = process.stderr.decode("utf-8")
                logger.error(f"Failed to scale Docker Compose service: {error}")
                return False
                
        except Exception as e:
            logger.error(f"Error scaling Docker Compose service: {str(e)}")
            return False
    
    def _scale_local(self, target_workers: int) -> bool:
        """
        Scale workers locally
        
        Args:
            target_workers: Target number of workers
            
        Returns:
            True if scaling was successful, False otherwise
        """
        # For local scaling, we would need to manage worker processes directly
        # This is a simplified implementation
        logger.warning(f"Local worker scaling not fully implemented (target: {target_workers})")
        return True

# Create singleton instance
auto_scaler = CeleryAutoScaler()

def get_auto_scaler():
    """
    Get the auto-scaler instance
    
    Returns:
        CeleryAutoScaler instance
    """
    return auto_scaler

def update_auto_scaler_config(config: Dict[str, Any]) -> bool:
    """
    Update auto-scaler configuration
    
    Args:
        config: New configuration values
        
    Returns:
        True if update was successful, False otherwise
    """
    try:
        auto_scaler.update_config(config)
        return True
    except Exception as e:
        logger.error(f"Error updating auto-scaler configuration: {str(e)}")
        return False

def get_auto_scaler_config() -> Dict[str, Any]:
    """
    Get current auto-scaler configuration
    
    Returns:
        Dictionary with current configuration
    """
    return auto_scaler.get_config()

def start_auto_scaler() -> bool:
    """
    Start the auto-scaler
    
    Returns:
        True if starting was successful, False otherwise
    """
    try:
        auto_scaler.start()
        return True
    except Exception as e:
        logger.error(f"Error starting auto-scaler: {str(e)}")
        return False

def stop_auto_scaler() -> bool:
    """
    Stop the auto-scaler
    
    Returns:
        True if stopping was successful, False otherwise
    """
    try:
        auto_scaler.stop()
        return True
    except Exception as e:
        logger.error(f"Error stopping auto-scaler: {str(e)}")
        return False

def create_blueprint():
    """
    Create a Flask blueprint for auto-scaler API
    
    Returns:
        Flask Blueprint
    """
    from flask import Blueprint, request, jsonify
    from backend.auth import admin_required
    
    bp = Blueprint('auto_scaler', __name__)
    
    @bp.route('/auto-scaler', methods=['GET'])
    @admin_required
    def get_config():
        """Get auto-scaler configuration"""
        return jsonify(get_auto_scaler_config())
    
    @bp.route('/auto-scaler', methods=['PUT'])
    @admin_required
    def update_config():
        """Update auto-scaler configuration"""
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request body"}), 400
            
        success = update_auto_scaler_config(data)
        if not success:
            return jsonify({"error": "Failed to update auto-scaler configuration"}), 500
            
        return jsonify({
            "success": True,
            "config": get_auto_scaler_config()
        })
    
    @bp.route('/auto-scaler/start', methods=['POST'])
    @admin_required
    def start():
        """Start the auto-scaler"""
        success = start_auto_scaler()
        if not success:
            return jsonify({"error": "Failed to start auto-scaler"}), 500
            
        return jsonify({
            "success": True,
            "message": "Auto-scaler started",
            "config": get_auto_scaler_config()
        })
    
    @bp.route('/auto-scaler/stop', methods=['POST'])
    @admin_required
    def stop():
        """Stop the auto-scaler"""
        success = stop_auto_scaler()
        if not success:
            return jsonify({"error": "Failed to stop auto-scaler"}), 500
            
        return jsonify({
            "success": True,
            "message": "Auto-scaler stopped",
            "config": get_auto_scaler_config()
        })
    
    return bp

def init_app(app):
    """Initialize auto-scaler and register blueprint with Flask app"""
    # Register blueprint
    app.register_blueprint(create_blueprint(), url_prefix='/api/admin')
    
    logger.info("Auto-scaler initialized")
