
"""
Monitoring system for DeepDefend
This module is a thin wrapper around the advanced monitoring functionality
provided by monitoring_service.py
"""

import os
import time
import logging
import json
import threading
import traceback
from typing import Dict, Any, Optional, List, Callable
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
from apscheduler.schedulers.background import BackgroundScheduler
from backend.config import ENABLE_PROMETHEUS_METRICS, PROMETHEUS_PORT

# Import advanced monitoring services
from monitoring_service import (
    sentry_monitoring, 
    alerting_service, 
    metrics_manager,
    capture_exception,
    capture_message,
    start_transaction,
    send_alert,
    record_metric,
    get_metrics,
    get_alert_history
)

# Configure logging
logger = logging.getLogger(__name__)

# Initialize metrics
class DeepDefendMonitoring:
    """
    Comprehensive monitoring system for DeepDefend
    Provides metrics, alerts, and health checks
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeepDefendMonitoring, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        
        # Initialize counters
        self.detection_counter = Counter(
            'deepdefend_detections_total', 
            'Total number of detections performed',
            ['detection_type', 'model', 'result_category']
        )
        
        self.error_counter = Counter(
            'deepdefend_errors_total',
            'Total number of errors',
            ['error_type', 'severity']
        )
        
        self.request_counter = Counter(
            'deepdefend_requests_total',
            'Total number of API requests',
            ['endpoint', 'method', 'status']
        )
        
        # Initialize gauges
        self.model_loaded_gauge = Gauge(
            'deepdefend_models_loaded',
            'Number of models currently loaded in memory',
            ['model_type']
        )
        
        self.active_tasks_gauge = Gauge(
            'deepdefend_active_tasks',
            'Number of active tasks',
            ['task_type']
        )
        
        # Initialize histograms
        self.detection_time_histogram = Histogram(
            'deepdefend_detection_seconds',
            'Time spent on detection',
            ['detection_type', 'model'],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
        )
        
        self.request_size_histogram = Histogram(
            'deepdefend_request_bytes',
            'Size of requests in bytes',
            ['endpoint'],
            buckets=(1024, 10*1024, 50*1024, 100*1024, 500*1024, 1024*1024, 5*1024*1024)
        )
        
        # Initialize summaries
        self.face_count_summary = Summary(
            'deepdefend_face_count',
            'Number of faces detected',
            ['detection_type']
        )
        
        # Set up alert rules (these rules will be checked periodically)
        self._setup_alert_rules()
        
        # Start background metrics server if enabled
        if ENABLE_PROMETHEUS_METRICS:
            try:
                start_http_server(PROMETHEUS_PORT)
                logger.info(f"Started Prometheus metrics server on port {PROMETHEUS_PORT}")
            except Exception as e:
                logger.error(f"Failed to start Prometheus metrics server: {str(e)}")
                capture_exception(e)
        
        # Start health check scheduler
        self._start_health_check_scheduler()
        
        logger.info("Monitoring system initialized")
    
    def _setup_alert_rules(self):
        """Set up standard alert rules"""
        thresholds = alerting_service.get_thresholds()
        
        # High error rate rule
        alerting_service.create_alert_rule(
            rule_id="high_error_rate",
            condition=lambda ctx: ctx.get("error_rate", 0) > thresholds["error_rate"],
            alert_data={
                "type": "high_error_rate",
                "severity": "critical",
                "message": "High error rate detected"
            }
        )
        
        # Slow detection rule
        alerting_service.create_alert_rule(
            rule_id="slow_detection",
            condition=lambda ctx: ctx.get("detection_time_avg", 0) > thresholds["detection_time_avg"],
            alert_data={
                "type": "slow_detection",
                "severity": "warning",
                "message": "Detection processing is slower than expected"
            }
        )
        
        # Memory usage rule
        alerting_service.create_alert_rule(
            rule_id="high_memory_usage",
            condition=lambda ctx: ctx.get("memory_usage", 0) > thresholds["memory_usage"],
            alert_data={
                "type": "high_memory_usage",
                "severity": "warning",
                "message": "High memory usage detected"
            }
        )
        
        # Model performance rule
        alerting_service.create_alert_rule(
            rule_id="low_model_performance",
            condition=lambda ctx: ctx.get("model_performance", 1.0) < thresholds["model_performance"],
            alert_data={
                "type": "low_model_performance",
                "severity": "warning",
                "message": "Model performance is below threshold"
            }
        )
        
        # Task queue size rule
        alerting_service.create_alert_rule(
            rule_id="large_task_queue",
            condition=lambda ctx: ctx.get("queue_size", 0) > thresholds["queue_size"],
            alert_data={
                "type": "large_task_queue",
                "severity": "warning",
                "message": "Task queue is larger than expected"
            }
        )
    
    def _start_health_check_scheduler(self):
        """Start scheduler for periodic health checks"""
        try:
            scheduler = BackgroundScheduler()
            scheduler.add_job(self.run_health_checks, 'interval', minutes=5)
            scheduler.start()
            logger.info("Health check scheduler started")
        except Exception as e:
            logger.error(f"Failed to start health check scheduler: {str(e)}")
            capture_exception(e)
    
    def run_health_checks(self):
        """Run periodic health checks"""
        try:
            # Check memory usage
            import psutil
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            
            record_metric("memory_usage", memory_usage, {"unit": "percent"})
            
            # Check CPU usage
            cpu_usage = psutil.cpu_percent() / 100.0
            record_metric("cpu_usage", cpu_usage, {"unit": "percent"})
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent / 100.0
            record_metric("disk_usage", disk_usage, {"unit": "percent"})
            
            # Check model health
            from model_manager import ModelManager
            model_manager = ModelManager()
            models_info = model_manager.get_models_info()
            
            record_metric("loaded_models", len(models_info.get("loaded_models", [])))
            
            # Check for model health
            current_model = models_info.get("current_model")
            if current_model and current_model not in models_info.get("loaded_models", []):
                send_alert(
                    alert_type="model_not_loaded",
                    message=f"Current model {current_model} is not loaded",
                    severity="critical",
                    details=models_info
                )
            
            # Check task queue size
            try:
                from celery.task.control import inspect
                insp = inspect()
                active_tasks = insp.active() or {}
                reserved_tasks = insp.reserved() or {}
                
                total_tasks = sum(len(tasks) for worker, tasks in active_tasks.items())
                total_tasks += sum(len(tasks) for worker, tasks in reserved_tasks.items())
                
                record_metric("task_queue_size", total_tasks)
                
                # Update active tasks gauge
                self.active_tasks_gauge.labels(task_type="all").set(total_tasks)
            except Exception as e:
                logger.error(f"Error checking task queue: {str(e)}")
            
            # Check detection performance metrics 
            try:
                from metrics import performance_metrics
                metrics_data = performance_metrics.get_all_metrics()
                
                # Calculate error rate
                error_rate = metrics_data.get("error_rate", 0)
                record_metric("error_rate", error_rate, {"unit": "percent"})
                
                # Get detection time metrics
                image_metrics = metrics_data.get("images", {})
                video_metrics = metrics_data.get("videos", {})
                
                if image_metrics:
                    image_proc_time = image_metrics.get("processing_time", {}).get("avg", 0)
                    record_metric("image_detection_time_avg", image_proc_time, {"unit": "seconds"})
                    
                if video_metrics:
                    video_proc_time = video_metrics.get("processing_time", {}).get("avg", 0)
                    record_metric("video_detection_time_avg", video_proc_time, {"unit": "seconds"})
                
                # Get model performance from retraining module if available
                try:
                    from model_retraining import get_retraining_status
                    retraining_status = get_retraining_status()
                    
                    if retraining_status.get("current_evaluation"):
                        model_performance = retraining_status["current_evaluation"].get("performance", 1.0)
                        record_metric("model_performance", model_performance, {
                            "model": retraining_status["current_evaluation"].get("model", "unknown")
                        })
                except ImportError:
                    pass  # Retraining module not available
                
                # Check alert rules with context
                context = {
                    "error_rate": error_rate,
                    "memory_usage": memory_usage,
                    "cpu_usage": cpu_usage,
                    "disk_usage": disk_usage,
                    "detection_time_avg": video_proc_time if video_metrics else image_proc_time,
                    "queue_size": total_tasks if 'total_tasks' in locals() else 0
                }
                
                # Add model performance if available
                if 'model_performance' in locals():
                    context["model_performance"] = model_performance
                
                # Check rules
                alerting_service.check_rules(context)
                
            except Exception as e:
                logger.error(f"Error checking performance metrics: {str(e)}")
                capture_exception(e)
            
            logger.debug("Health checks completed successfully")
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}", exc_info=True)
            capture_exception(e)
    
    def record_request(self, endpoint: str, method: str, status: int, request_size: int = 0):
        """
        Record an API request
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            status: HTTP status code
            request_size: Size of request in bytes
        """
        try:
            # Increment request counter
            self.request_counter.labels(endpoint=endpoint, method=method, status=status).inc()
            
            # Record request size
            if request_size > 0:
                self.request_size_histogram.labels(endpoint=endpoint).observe(request_size)
                
            # Record metrics in the metrics manager
            record_metric("api_requests", 1, {
                "endpoint": endpoint,
                "method": method,
                "status": str(status)
            })
                
        except Exception as e:
            logger.error(f"Error recording request: {str(e)}")
            capture_exception(e)
    
    def record_detection(self, detection_type: str, model: str, probability: float, 
                         time_taken: float, face_count: int = 0, error: Optional[str] = None):
        """
        Record a detection operation
        
        Args:
            detection_type: Type of detection (image/video)
            model: Model used for detection
            probability: Detection probability
            time_taken: Time taken in seconds
            face_count: Number of faces detected
            error: Error message if detection failed
        """
        try:
            # Start a transaction for performance monitoring
            transaction = start_transaction(
                name=f"detection.{detection_type}",
                op="deepfake_detection"
            )
            
            # Determine result category based on probability
            if error:
                result_category = "error"
            elif probability > 0.7:
                result_category = "deepfake_high"
            elif probability > 0.5:
                result_category = "deepfake_medium"
            elif probability < 0.3:
                result_category = "real_high"
            else:
                result_category = "real_medium"
            
            # Increment detection counter
            self.detection_counter.labels(
                detection_type=detection_type, 
                model=model,
                result_category=result_category
            ).inc()
            
            # Record detection time
            self.detection_time_histogram.labels(
                detection_type=detection_type,
                model=model
            ).observe(time_taken)
            
            # Record face count
            if face_count > 0:
                self.face_count_summary.labels(detection_type=detection_type).observe(face_count)
            
            # Record error if any
            if error:
                self.error_counter.labels(error_type=f"detection_{detection_type}", severity="error").inc()
                capture_message(f"Detection error ({detection_type}): {error}", level="error")
            
            # Record metrics in the metrics manager
            record_metric(f"{detection_type}_detection_time", time_taken, {
                "model": model,
                "result": result_category
            })
            
            record_metric(f"{detection_type}_detection_probability", probability, {
                "model": model,
                "result": result_category
            })
            
            if face_count > 0:
                record_metric(f"{detection_type}_face_count", face_count, {
                    "model": model
                })
                
            # Finish the transaction if started
            if transaction:
                transaction.finish()
                
        except Exception as e:
            logger.error(f"Error recording detection: {str(e)}")
            capture_exception(e)
    
    def record_error(self, error_type: str, severity: str = "error", traceback_info: Optional[str] = None):
        """
        Record an error
        
        Args:
            error_type: Type of error
            severity: Error severity (critical, error, warning)
            traceback_info: Exception traceback
        """
        try:
            # Increment error counter
            self.error_counter.labels(error_type=error_type, severity=severity).inc()
            
            # Log error with traceback
            log_message = f"Error: {error_type}"
            if traceback_info:
                log_message += f"\nTraceback: {traceback_info}"
                
            if severity == "critical":
                logger.critical(log_message)
                
                # Send alert for critical errors
                send_alert(
                    alert_type="critical_error",
                    message=f"Critical error: {error_type}",
                    severity="critical",
                    details={
                        "error_type": error_type,
                        "traceback": traceback_info
                    }
                )
                
            elif severity == "error":
                logger.error(log_message)
                capture_message(log_message, level="error")
            else:
                logger.warning(log_message)
                
            # Record metric
            record_metric("errors", 1, {
                "error_type": error_type,
                "severity": severity
            })
                
        except Exception as e:
            logger.error(f"Error recording error: {str(e)}")
            capture_exception(e)
    
    def update_model_loaded_metric(self, model_type: str, count: int):
        """
        Update model loaded gauge
        
        Args:
            model_type: Type of model
            count: Number of models loaded
        """
        try:
            self.model_loaded_gauge.labels(model_type=model_type).set(count)
            record_metric("models_loaded", count, {"model_type": model_type})
        except Exception as e:
            logger.error(f"Error updating model loaded gauge: {str(e)}")
            capture_exception(e)
    
    def update_active_tasks_metric(self, task_type: str, count: int):
        """
        Update active tasks gauge
        
        Args:
            task_type: Type of task
            count: Number of active tasks
        """
        try:
            self.active_tasks_gauge.labels(task_type=task_type).set(count)
            record_metric("active_tasks", count, {"task_type": task_type})
        except Exception as e:
            logger.error(f"Error updating active tasks gauge: {str(e)}")
            capture_exception(e)
            
# Initialize monitoring singleton
monitoring = DeepDefendMonitoring()

# Create a custom exception handler
def global_exception_handler(exctype, value, tb):
    """
    Global exception handler to log uncaught exceptions
    """
    # Get traceback as string
    traceback_str = ''.join(traceback.format_exception(exctype, value, tb))
    
    # Log error
    monitoring.record_error(
        error_type="uncaught_exception",
        severity="critical",
        traceback_info=traceback_str
    )
    
    # Send to monitoring service
    capture_exception(value)
    
    # Call original exception handler
    sys.__excepthook__(exctype, value, tb)

# Install global exception handler
import sys
sys.excepthook = global_exception_handler

# Create a Flask request handler
def request_metrics_middleware(app):
    """
    Middleware to record request metrics
    
    Usage:
        request_metrics_middleware(app)
    """
    @app.before_request
    def before_request():
        request.start_time = time.time()
    
    @app.after_request
    def after_request(response):
        try:
            # Calculate request duration
            duration = time.time() - getattr(request, 'start_time', time.time())
            
            # Get endpoint name
            if request.endpoint:
                endpoint = request.endpoint
            else:
                endpoint = request.path
            
            # Record request
            monitoring.record_request(
                endpoint=endpoint,
                method=request.method,
                status=response.status_code,
                request_size=request.content_length or 0
            )
            
            # Record API latency
            record_metric("api_latency", duration, {
                "endpoint": endpoint,
                "method": request.method,
                "status": str(response.status_code)
            })
            
            # Add response time header
            response.headers['X-Response-Time'] = f"{duration:.3f}s"
            
        except Exception as e:
            logger.error(f"Error in request metrics middleware: {str(e)}")
            capture_exception(e)
            
        return response
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        # Record error
        monitoring.record_error(
            error_type="http_exception",
            severity="error",
            traceback_info=traceback.format_exc()
        )
        
        # Send to monitoring service
        capture_exception(e)
        
        # Re-raise for Flask to handle
        raise e
