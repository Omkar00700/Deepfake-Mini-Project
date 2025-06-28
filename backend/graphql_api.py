
"""
GraphQL API for DeepDefend
Provides flexible querying of detection results, system metrics,
and model performance data
"""

import os
import logging
import json
import time
from typing import Dict, Any, Optional, List, Union
from flask import Blueprint, request, jsonify, current_app
import graphene
from graphene import ObjectType, Field, String, Float, Int, Boolean, List as GrapheneList
from graphql import GraphQLError
from datetime import datetime

# Import required modules
from model_manager import ModelManager
from metrics import performance_metrics
from backend.auth import token_required, get_token_identity
from monitoring_service import get_metrics, get_alert_history
from backend.config import GRAPHQL_REQUIRE_AUTH

# Configure logging
logger = logging.getLogger(__name__)

# Initialize dependencies
model_manager = ModelManager()

# Define GraphQL schema

class Detection(ObjectType):
    """GraphQL object for detection results"""
    id = String(description="Detection ID")
    timestamp = Float(description="Detection timestamp")
    detection_type = String(description="Type of detection (image/video)")
    model = String(description="Model used for detection")
    probability = Float(description="Probability of being a deepfake")
    confidence = Float(description="Confidence score of the detection")
    processing_time = Float(description="Time taken to process the detection")
    media_id = String(description="ID of the processed media")
    face_count = Int(description="Number of faces detected")
    deepfake = Boolean(description="Whether the media was classified as deepfake")
    user_id = String(description="ID of the user who requested the detection")
    regions = GrapheneList(lambda: FaceRegion, description="Detected face regions")

class FaceRegion(ObjectType):
    """GraphQL object for detected face regions"""
    x = Int(description="X coordinate")
    y = Int(description="Y coordinate")
    width = Int(description="Width of the region")
    height = Int(description="Height of the region")
    probability = Float(description="Probability for this region")
    confidence = Float(description="Confidence for this region")
    frame = Int(description="Frame number (for videos)")

class ModelInfo(ObjectType):
    """GraphQL object for model information"""
    name = String(description="Model name")
    type = String(description="Model type")
    input_shape = GrapheneList(Int, description="Input shape required by the model")
    version = String(description="Model version")
    is_loaded = Boolean(description="Whether the model is currently loaded")
    is_current = Boolean(description="Whether this is the current active model")
    last_used = Float(description="Timestamp when the model was last used")
    performance_metrics = Field(lambda: ModelPerformance, description="Model performance metrics")

class ModelPerformance(ObjectType):
    """GraphQL object for model performance metrics"""
    accuracy = Float(description="Accuracy on validation dataset")
    precision = Float(description="Precision on validation dataset")
    recall = Float(description="Recall on validation dataset")
    f1_score = Float(description="F1 score on validation dataset")
    auc = Float(description="Area under ROC curve")
    avg_processing_time = Float(description="Average processing time per image")
    last_evaluation = Float(description="Timestamp of last evaluation")

class SystemMetric(ObjectType):
    """GraphQL object for system metrics"""
    name = String(description="Metric name")
    value = Float(description="Current metric value")
    min = Float(description="Minimum value recorded")
    max = Float(description="Maximum value recorded")
    count = Int(description="Number of measurements")
    last_updated = Float(description="Timestamp of last update")
    unit = String(description="Unit of measurement")

class TimeSeriesPoint(ObjectType):
    """GraphQL object for time series data points"""
    timestamp = Float(description="Timestamp of measurement")
    value = Float(description="Metric value")
    tags = String(description="Associated tags in JSON format")

class Alert(ObjectType):
    """GraphQL object for system alerts"""
    alert_id = String(description="Alert ID")
    type = String(description="Alert type")
    message = String(description="Alert message")
    severity = String(description="Alert severity (critical, warning, info)")
    timestamp = String(description="Alert timestamp")
    details = String(description="Alert details in JSON format")

class Task(ObjectType):
    """GraphQL object for asynchronous tasks"""
    id = String(description="Task ID")
    status = String(description="Task status")
    created_at = Float(description="Creation timestamp")
    updated_at = Float(description="Last update timestamp")
    task_type = String(description="Type of task")
    progress = Float(description="Task progress (0-1)")
    result_id = String(description="ID of the result if completed")
    error = String(description="Error message if failed")

class User(ObjectType):
    """GraphQL object for user information"""
    id = String(description="User ID")
    username = String(description="Username")
    email = String(description="Email address")
    role = String(description="User role")
    created_at = Float(description="Account creation timestamp")
    last_active = Float(description="Last activity timestamp")
    detection_count = Int(description="Number of detections performed")

class RetrainingStatus(ObjectType):
    """GraphQL object for model retraining status"""
    enabled = Boolean(description="Whether automatic retraining is enabled")
    feedback_collection_enabled = Boolean(description="Whether feedback collection is enabled")
    feedback_samples_collected = Int(description="Number of feedback samples collected")
    last_evaluation_time = Float(description="Timestamp of last evaluation")
    last_retraining_time = Float(description="Timestamp of last retraining")
    current_performance = Float(description="Current model performance")
    retraining_threshold = Float(description="Performance threshold for triggering retraining")
    retraining_interval_hours = Int(description="Interval between evaluations in hours")

class Query(ObjectType):
    """Root query object for the GraphQL API"""
    # Detection queries
    detection = Field(Detection, id=String(required=True), description="Get a detection by ID")
    detections = GrapheneList(Detection, 
        limit=Int(default_value=10), 
        offset=Int(default_value=0),
        user_id=String(),
        detection_type=String(),
        min_probability=Float(),
        max_probability=Float(),
        min_confidence=Float(),
        from_timestamp=Float(),
        to_timestamp=Float(),
        description="Get a list of detections with optional filters"
    )
    
    # Model queries
    model = Field(ModelInfo, name=String(required=True), description="Get information about a specific model")
    models = GrapheneList(ModelInfo, description="Get information about all available models")
    current_model = Field(ModelInfo, description="Get information about the currently active model")
    
    # Metrics queries
    metric = Field(SystemMetric, name=String(required=True), description="Get a specific metric")
    metrics = GrapheneList(SystemMetric, description="Get all system metrics")
    time_series = GrapheneList(TimeSeriesPoint, 
        metric_name=String(required=True),
        start_time=Float(),
        end_time=Float(),
        limit=Int(default_value=100),
        description="Get time series data for a metric"
    )
    
    # Alert queries
    alerts = GrapheneList(Alert, 
        limit=Int(default_value=50),
        severity=String(),
        description="Get system alerts"
    )
    
    # Task queries
    task = Field(Task, id=String(required=True), description="Get information about a specific task")
    tasks = GrapheneList(Task, 
        status=String(),
        task_type=String(),
        limit=Int(default_value=20),
        offset=Int(default_value=0),
        description="Get a list of tasks with optional filters"
    )
    
    # User queries (admin only)
    user = Field(User, id=String(required=True), description="Get information about a specific user")
    users = GrapheneList(User, 
        limit=Int(default_value=20),
        offset=Int(default_value=0),
        role=String(),
        description="Get a list of users with optional filters"
    )
    
    # Retraining queries
    retraining_status = Field(RetrainingStatus, description="Get model retraining status")
    
    def resolve_detection(self, info, id):
        """Resolve a single detection by ID"""
        # This is a placeholder - in a real implementation, you would
        # retrieve the detection from your database
        return None
    
    def resolve_detections(self, info, **kwargs):
        """Resolve a list of detections with filters"""
        # This is a placeholder - in a real implementation, you would
        # retrieve detections from your database with filters
        return []
    
    def resolve_model(self, info, name):
        """Resolve a specific model's information"""
        models_info = model_manager.get_models_info()
        
        if name not in models_info["available_models"]:
            raise GraphQLError(f"Model '{name}' not found")
            
        is_loaded = name in models_info["loaded_models"]
        is_current = name == models_info["current_model"]
        
        return ModelInfo(
            name=name,
            type="deepfake_detection",
            input_shape=[224, 224, 3],  # Placeholder
            version="1.0",  # Placeholder
            is_loaded=is_loaded,
            is_current=is_current,
            last_used=time.time() if is_current else None,
            performance_metrics=ModelPerformance(
                accuracy=0.95,  # Placeholder
                precision=0.93,  # Placeholder
                recall=0.91,  # Placeholder
                f1_score=0.92,  # Placeholder
                auc=0.97,  # Placeholder
                avg_processing_time=0.5,  # Placeholder
                last_evaluation=time.time()  # Placeholder
            )
        )
    
    def resolve_models(self, info):
        """Resolve all available models information"""
        models_info = model_manager.get_models_info()
        
        result = []
        for name in models_info["available_models"]:
            is_loaded = name in models_info["loaded_models"]
            is_current = name == models_info["current_model"]
            
            result.append(ModelInfo(
                name=name,
                type="deepfake_detection",
                input_shape=[224, 224, 3],  # Placeholder
                version="1.0",  # Placeholder
                is_loaded=is_loaded,
                is_current=is_current,
                last_used=time.time() if is_current else None,
                performance_metrics=ModelPerformance(
                    accuracy=0.95,  # Placeholder
                    precision=0.93,  # Placeholder
                    recall=0.91,  # Placeholder
                    f1_score=0.92,  # Placeholder
                    auc=0.97,  # Placeholder
                    avg_processing_time=0.5,  # Placeholder
                    last_evaluation=time.time()  # Placeholder
                )
            ))
            
        return result
    
    def resolve_current_model(self, info):
        """Resolve the current model's information"""
        models_info = model_manager.get_models_info()
        current_model = models_info["current_model"]
        
        return ModelInfo(
            name=current_model,
            type="deepfake_detection",
            input_shape=[224, 224, 3],  # Placeholder
            version="1.0",  # Placeholder
            is_loaded=True,
            is_current=True,
            last_used=time.time(),
            performance_metrics=ModelPerformance(
                accuracy=0.95,  # Placeholder
                precision=0.93,  # Placeholder
                recall=0.91,  # Placeholder
                f1_score=0.92,  # Placeholder
                auc=0.97,  # Placeholder
                avg_processing_time=0.5,  # Placeholder
                last_evaluation=time.time()  # Placeholder
            )
        )
    
    def resolve_metric(self, info, name):
        """Resolve a specific metric"""
        metrics = get_metrics()
        
        if name not in metrics:
            raise GraphQLError(f"Metric '{name}' not found")
            
        metric_data = metrics[name]
        
        return SystemMetric(
            name=name,
            value=metric_data.get("current"),
            min=metric_data.get("min"),
            max=metric_data.get("max"),
            count=metric_data.get("count"),
            last_updated=metric_data.get("last_updated"),
            unit="unknown"  # Placeholder
        )
    
    def resolve_metrics(self, info):
        """Resolve all system metrics"""
        metrics = get_metrics()
        
        result = []
        for name, metric_data in metrics.items():
            result.append(SystemMetric(
                name=name,
                value=metric_data.get("current"),
                min=metric_data.get("min"),
                max=metric_data.get("max"),
                count=metric_data.get("count"),
                last_updated=metric_data.get("last_updated"),
                unit="unknown"  # Placeholder
            ))
            
        return result
    
    def resolve_time_series(self, info, metric_name, **kwargs):
        """Resolve time series data for a metric"""
        # This is a placeholder - in a real implementation, you would
        # retrieve time series data from your metrics storage
        return []
    
    def resolve_alerts(self, info, **kwargs):
        """Resolve system alerts"""
        limit = kwargs.get("limit", 50)
        severity = kwargs.get("severity")
        
        alerts = get_alert_history(limit, severity)
        
        result = []
        for alert in alerts:
            result.append(Alert(
                alert_id=alert.get("alert_id", "unknown"),
                type=alert.get("type", "unknown"),
                message=alert.get("message", ""),
                severity=alert.get("severity", "info"),
                timestamp=alert.get("timestamp", ""),
                details=json.dumps(alert.get("details", {}))
            ))
            
        return result
    
    def resolve_task(self, info, id):
        """Resolve a specific task's information"""
        # This is a placeholder - in a real implementation, you would
        # retrieve the task from your task queue or database
        return None
    
    def resolve_tasks(self, info, **kwargs):
        """Resolve a list of tasks with filters"""
        # This is a placeholder - in a real implementation, you would
        # retrieve tasks from your task queue or database with filters
        return []
    
    def resolve_user(self, info, id):
        """Resolve a specific user's information"""
        # This is a placeholder - in a real implementation, you would
        # retrieve the user from your database
        # This should be admin-only
        return None
    
    def resolve_users(self, info, **kwargs):
        """Resolve a list of users with filters"""
        # This is a placeholder - in a real implementation, you would
        # retrieve users from your database with filters
        # This should be admin-only
        return []
    
    def resolve_retraining_status(self, info):
        """Resolve model retraining status"""
        try:
            from model_retraining import get_retraining_status
            status = get_retraining_status()
            
            return RetrainingStatus(
                enabled=status.get("enabled", False),
                feedback_collection_enabled=status.get("feedback_collection_enabled", False),
                feedback_samples_collected=status.get("feedback_samples_collected", 0),
                last_evaluation_time=status.get("last_evaluation_time"),
                last_retraining_time=status.get("last_retraining_time"),
                current_performance=status.get("current_evaluation", {}).get("performance") if status.get("current_evaluation") else None,
                retraining_threshold=status.get("retraining_threshold"),
                retraining_interval_hours=status.get("retraining_interval_hours")
            )
        except ImportError:
            # Retraining module not available
            return RetrainingStatus(
                enabled=False,
                feedback_collection_enabled=False,
                feedback_samples_collected=0,
                retraining_threshold=0.7,
                retraining_interval_hours=24
            )

class FeedbackInput(graphene.InputObjectType):
    """Input type for submitting detection feedback"""
    detection_id = String(required=True)
    correct = Boolean(required=True)
    actual_label = String(required=True)
    confidence = Float()
    
class DetectionInput(graphene.InputObjectType):
    """Input type for creating a detection (used in testing)"""
    detection_type = String(required=True)
    media_id = String(required=True)
    model = String()

class Mutation(ObjectType):
    """Root mutation object for the GraphQL API"""
    # Model mutations
    switch_model = Field(Boolean, model_name=String(required=True), description="Switch to a different model")
    
    # Feedback mutations
    submit_feedback = Field(Boolean, 
        feedback_data=FeedbackInput(required=True),
        description="Submit feedback for a detection result"
    )
    
    # Retraining mutations
    trigger_evaluation = Field(Boolean, description="Manually trigger model evaluation")
    trigger_retraining = Field(Boolean, description="Manually trigger model retraining")
    
    # Testing mutations
    create_test_detection = Field(Detection,
        input=DetectionInput(required=True),
        description="Create a test detection (for development/testing)"
    )
    
    def resolve_switch_model(self, info, model_name):
        """Switch to a different model"""
        # Check authentication if required
        if GRAPHQL_REQUIRE_AUTH:
            auth_header = info.context.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                raise GraphQLError("Authentication required")
                
            # In a real implementation, you would validate the token
            
        success = model_manager.switch_model(model_name)
        if not success:
            raise GraphQLError(f"Failed to switch to model '{model_name}'")
            
        return True
    
    def resolve_submit_feedback(self, info, feedback_data):
        """Submit feedback for a detection result"""
        try:
            from model_retraining import add_detection_feedback
            
            success = add_detection_feedback(
                detection_id=feedback_data.detection_id,
                correct=feedback_data.correct,
                actual_label=feedback_data.actual_label,
                confidence=feedback_data.confidence
            )
            
            if not success:
                raise GraphQLError("Failed to submit feedback")
                
            return True
        except ImportError:
            # Retraining module not available
            raise GraphQLError("Feedback collection is not available")
    
    def resolve_trigger_evaluation(self, info):
        """Manually trigger model evaluation"""
        # This should be admin-only
        try:
            from model_retraining import trigger_model_evaluation
            
            result = trigger_model_evaluation()
            if result is None:
                raise GraphQLError("Failed to trigger model evaluation")
                
            return True
        except ImportError:
            # Retraining module not available
            raise GraphQLError("Model retraining is not available")
    
    def resolve_trigger_retraining(self, info):
        """Manually trigger model retraining"""
        # This should be admin-only
        try:
            from model_retraining import trigger_model_retraining
            
            success = trigger_model_retraining()
            if not success:
                raise GraphQLError("Failed to trigger model retraining")
                
            return True
        except ImportError:
            # Retraining module not available
            raise GraphQLError("Model retraining is not available")
    
    def resolve_create_test_detection(self, info, input):
        """Create a test detection (for development/testing)"""
        # This should be development-only
        return Detection(
            id="test-" + str(int(time.time())),
            timestamp=time.time(),
            detection_type=input.detection_type,
            model=input.model or "efficientnet",
            probability=0.75,
            confidence=0.85,
            processing_time=0.5,
            media_id=input.media_id,
            face_count=1,
            deepfake=True,
            user_id="test-user",
            regions=[
                FaceRegion(
                    x=100,
                    y=100,
                    width=200,
                    height=200,
                    probability=0.75,
                    confidence=0.85,
                    frame=0
                )
            ]
        )

# Create the schema
schema = graphene.Schema(query=Query, mutation=Mutation)

# Create blueprint for the GraphQL API
graphql_bp = Blueprint('graphql', __name__)

@graphql_bp.route('/graphql', methods=['POST'])
def graphql_endpoint():
    """GraphQL API endpoint"""
    # Check for authentication if required
    if GRAPHQL_REQUIRE_AUTH:
        try:
            token_required()(lambda: None)()
        except Exception as e:
            return jsonify({"errors": [{"message": str(e)}]}), 401
    
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({"errors": [{"message": "No GraphQL query provided"}]}), 400
        
    # Execute the query
    result = schema.execute(
        data['query'],
        context=request,
        variables=data.get('variables')
    )
    
    # Handle errors
    if result.errors:
        logger.error(f"GraphQL errors: {result.errors}")
        return jsonify({"errors": [{"message": str(error)} for error in result.errors]}), 400
        
    return jsonify(result.data)

def init_app(app):
    """Register the GraphQL blueprint with the Flask app"""
    app.register_blueprint(graphql_bp, url_prefix='/api')
    logger.info("GraphQL API initialized")
