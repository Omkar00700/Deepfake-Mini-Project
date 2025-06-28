"""
Configuration settings for DeepDefend
"""

import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
load_dotenv()

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "models"))
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", os.path.join(BASE_DIR, "uploads"))
RESULTS_FOLDER = os.getenv("RESULTS_FOLDER", os.path.join(BASE_DIR, "results"))
VISUALIZATION_OUTPUT_DIR = os.getenv("VISUALIZATION_OUTPUT_DIR", os.path.join(BASE_DIR, "visualizations"))
FEEDBACK_FOLDER = os.getenv("FEEDBACK_FOLDER", os.path.join(BASE_DIR, "feedback"))
DATASET_DIR = os.getenv("DATASET_DIR", os.path.join(BASE_DIR, "data"))

# Create directories if they don't exist
for directory in [MODEL_DIR, UPLOAD_FOLDER, RESULTS_FOLDER, VISUALIZATION_OUTPUT_DIR, FEEDBACK_FOLDER, DATASET_DIR]:
    os.makedirs(directory, exist_ok=True)
    logger.debug(f"Created directory: {directory}")

# Model Configuration
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "efficientnet")
DEEPFAKE_THRESHOLD = float(os.getenv("DEEPFAKE_THRESHOLD", "0.5"))
CONFIDENCE_MIN_VALUE = float(os.getenv("CONFIDENCE_MIN_VALUE", "0.1"))
CONFIDENCE_MAX_VALUE = float(os.getenv("CONFIDENCE_MAX_VALUE", "0.95"))
MODEL_INPUT_SIZE = (224, 224)  # (height, width)
AVAILABLE_MODELS = ["efficientnet", "xception", "mesonet", "resnet", "indian_specialized"]

# Ensemble configuration
ENABLE_ENSEMBLE_DETECTION = os.getenv("ENABLE_ENSEMBLE", "true").lower() == "true"
ENSEMBLE_WEIGHTS = {
    "efficientnet": 1.0,
    "xception": 0.9,
    "mesonet": 0.7,
    "resnet": 0.8,
    "indian_specialized": 1.0
}

# Indian face detection configuration
INDIAN_FACE_DETECTION_ENABLED = os.getenv("INDIAN_FACE_DETECTION", "true").lower() == "true"

# Video processing configuration
MAX_VIDEO_FRAMES = int(os.getenv("MAX_VIDEO_FRAMES", "30"))
VIDEO_FRAME_INTERVAL = int(os.getenv("VIDEO_FRAME_INTERVAL", "10"))  # Process every Nth frame

# Authentication
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_MINUTES = int(os.getenv("JWT_EXPIRATION_MINUTES", "60"))
JWT_REFRESH_EXPIRATION_DAYS = int(os.getenv("JWT_REFRESH_EXPIRATION_DAYS", "7"))
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "False").lower() == "true"

# API configuration
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(16 * 1024 * 1024)))  # 16 MB max upload size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'wmv'}

# Debug mode
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# Log configuration
logger.info(f"DeepDefend backend configuration loaded")
logger.info(f"Debug mode: {DEBUG}")
logger.info(f"Ensemble detection: {ENABLE_ENSEMBLE_DETECTION}")
logger.info(f"Indian face detection: {INDIAN_FACE_DETECTION_ENABLED}")

# Face Detection
FACE_DETECTION_CONFIDENCE = float(os.getenv("FACE_DETECTION_CONFIDENCE", "0.5"))
FACE_MARGIN_PERCENT = float(os.getenv("FACE_MARGIN_PERCENT", "0.2"))

# Video Processing - Improved parameters
VIDEO_MAX_FRAMES = int(os.getenv("VIDEO_MAX_FRAMES", "30"))
VIDEO_FRAME_INTERVAL = float(os.getenv("VIDEO_FRAME_INTERVAL", "0.5"))
FRAME_SIMILARITY_THRESHOLD = float(os.getenv("FRAME_SIMILARITY_THRESHOLD", "0.85"))
ENABLE_FRAME_CACHING = os.getenv("ENABLE_FRAME_CACHING", "True").lower() == "true"
FRAME_CACHE_SIZE = int(os.getenv("FRAME_CACHE_SIZE", "100"))

# Video Analysis Improvements
MIN_FRAMES_FOR_VALID_DETECTION = int(os.getenv("MIN_FRAMES_FOR_VALID_DETECTION", "5"))
TEMPORAL_CONSISTENCY_WEIGHT = float(os.getenv("TEMPORAL_CONSISTENCY_WEIGHT", "0.25"))
CONFIDENCE_THRESHOLD_FOR_ENSEMBLE = float(os.getenv("CONFIDENCE_THRESHOLD_FOR_ENSEMBLE", "0.65"))
DETECTION_BIAS_CORRECTION = float(os.getenv("DETECTION_BIAS_CORRECTION", "0.05"))
VIDEO_ENSEMBLE_DEFAULT_WEIGHTS = {
    "efficientnet": 0.45,
    "xception": 0.35,
    "mesonet": 0.2
}

# Indian Face Detection Enhancement
INDIAN_FACE_DETECTION_ENABLED = os.getenv("INDIAN_FACE_DETECTION_ENABLED", "True").lower() == "true"
FACE_CONTRAST_ADJUSTMENT = float(os.getenv("FACE_CONTRAST_ADJUSTMENT", "1.2"))
FACE_BRIGHTNESS_ADJUSTMENT = float(os.getenv("FACE_BRIGHTNESS_ADJUSTMENT", "10.0"))
NOISE_REDUCTION_STRENGTH = int(os.getenv("NOISE_REDUCTION_STRENGTH", "5"))
ENABLE_ADAPTIVE_PREPROCESSING = os.getenv("ENABLE_ADAPTIVE_PREPROCESSING", "True").lower() == "true"

# Performance Optimization
ENABLE_PARALLEL_PROCESSING = os.getenv("ENABLE_PARALLEL_PROCESSING", "True").lower() == "true"
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))

# Enhanced Detection Features
ENABLE_ENSEMBLE_DETECTION = os.getenv("ENABLE_ENSEMBLE_DETECTION", "True").lower() == "true"
ENABLE_ADVANCED_PREPROCESSING = os.getenv("ENABLE_ADVANCED_PREPROCESSING", "True").lower() == "true"
USE_SCENE_BASED_SAMPLING = os.getenv("USE_SCENE_BASED_SAMPLING", "True").lower() == "true"
ENABLE_TEMPORAL_ANALYSIS = os.getenv("ENABLE_TEMPORAL_ANALYSIS", "True").lower() == "true"
ENABLE_DYNAMIC_MODEL_SWITCHING = os.getenv("ENABLE_DYNAMIC_MODEL_SWITCHING", "True").lower() == "true"

# ─── Newly added for auto-switching ─────────────────────────────────────────────
AUTO_SWITCH_THRESHOLD   = float(os.getenv("AUTO_SWITCH_THRESHOLD", "0.2"))
MODEL_SWITCH_COOLDOWN   = int(os.getenv("MODEL_SWITCH_COOLDOWN", "3600"))  # in seconds
# ────────────────────────────────────────────────────────────────────────────────

# Debug and Logging
DEBUG_MODE_ENABLED       = os.getenv("DEBUG_MODE_ENABLED", "False").lower() == "true"
DETAILED_LOGGING         = os.getenv("DETAILED_LOGGING", "True").lower() == "true"
LOG_INTERMEDIATE_RESULTS = os.getenv("LOG_INTERMEDIATE_RESULTS", "True").lower() == "true"

# Validation Dataset
VALIDATION_DATASET_PATH = os.getenv("VALIDATION_DATASET_PATH", "data/validation")

# Path Configuration
MODEL_SAVE_PATH   = os.getenv("MODEL_SAVE_PATH", "models/trained")
MODEL_INPUT_SIZE  = (224, 224)

# Continuous Retraining
RETRAINING_ENABLED               = os.getenv("RETRAINING_ENABLED", "True").lower() == "true"
RETRAINING_INTERVAL_HOURS        = int(os.getenv("RETRAINING_INTERVAL_HOURS", "24"))
RETRAINING_PERFORMANCE_THRESHOLD = float(os.getenv("RETRAINING_PERFORMANCE_THRESHOLD", "0.7"))

# Feedback Collection
FEEDBACK_COLLECTION_ENABLED = os.getenv("FEEDBACK_COLLECTION_ENABLED", "True").lower() == "true"
FEEDBACK_STORAGE_PATH       = os.getenv("FEEDBACK_STORAGE_PATH", "data/feedback")

# Metrics Logging
METRICS_STORAGE_PATH   = os.getenv("METRICS_STORAGE_PATH", "data/metrics")
METRICS_RETENTION_DAYS = int(os.getenv("METRICS_RETENTION_DAYS", "30"))
ERROR_LOG_MAX_ENTRIES  = int(os.getenv("ERROR_LOG_MAX_ENTRIES", "1000"))
