
"""
Additional configuration parameters for DeepDefend
These parameters extend the base configuration in config.py
"""

import os
from typing import Dict, Any, Optional, List, Union

# Continuous Model Retraining Configuration
RETRAINING_ENABLED = os.environ.get('RETRAINING_ENABLED', 'true').lower() == 'true'
RETRAINING_INTERVAL_HOURS = int(os.environ.get('RETRAINING_INTERVAL_HOURS', '24'))
RETRAINING_PERFORMANCE_THRESHOLD = float(os.environ.get('RETRAINING_PERFORMANCE_THRESHOLD', '0.7'))
FEEDBACK_COLLECTION_ENABLED = os.environ.get('FEEDBACK_COLLECTION_ENABLED', 'true').lower() == 'true'
FEEDBACK_STORAGE_PATH = os.environ.get('FEEDBACK_STORAGE_PATH', 'data/feedback')
VALIDATION_DATASET_PATH = os.environ.get('VALIDATION_DATASET_PATH', 'data/validation')
MODEL_SAVE_PATH = os.environ.get('MODEL_SAVE_PATH', 'models')

# Indian Face Detection Enhancement Configuration
INDIAN_FACE_DETECTION_ENABLED = os.environ.get('INDIAN_FACE_DETECTION_ENABLED', 'true').lower() == 'true'
FACE_CONTRAST_ADJUSTMENT = float(os.environ.get('FACE_CONTRAST_ADJUSTMENT', '1.2'))
FACE_BRIGHTNESS_ADJUSTMENT = float(os.environ.get('FACE_BRIGHTNESS_ADJUSTMENT', '10.0'))
NOISE_REDUCTION_STRENGTH = int(os.environ.get('NOISE_REDUCTION_STRENGTH', '5'))
ENABLE_ADAPTIVE_PREPROCESSING = os.environ.get('ENABLE_ADAPTIVE_PREPROCESSING', 'true').lower() == 'true'

# Advanced Monitoring Configuration
ENABLE_SENTRY = os.environ.get('ENABLE_SENTRY', 'false').lower() == 'true'
SENTRY_DSN = os.environ.get('SENTRY_DSN', '')
SENTRY_ENVIRONMENT = os.environ.get('SENTRY_ENVIRONMENT', 'production')
ENABLE_EMAIL_ALERTS = os.environ.get('ENABLE_EMAIL_ALERTS', 'false').lower() == 'true'
SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.example.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))
SMTP_USER = os.environ.get('SMTP_USER', '')
SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD', '')
ALERT_FROM_EMAIL = os.environ.get('ALERT_FROM_EMAIL', 'alerts@deepdefend.ai')
ALERT_TO_EMAILS = os.environ.get('ALERT_TO_EMAILS', '').split(',')
ENABLE_WEBHOOK_ALERTS = os.environ.get('ENABLE_WEBHOOK_ALERTS', 'false').lower() == 'true'
WEBHOOK_URL = os.environ.get('WEBHOOK_URL', '')

# Enhanced Confidence Calibration 
ENABLE_CONFIDENCE_CALIBRATION = os.environ.get('ENABLE_CONFIDENCE_CALIBRATION', 'true').lower() == 'true'
REAL_CONFIDENCE_MULTIPLIER = float(os.environ.get('REAL_CONFIDENCE_MULTIPLIER', '0.9'))  # Slightly reduce confidence for real predictions
UNCERTAIN_CONFIDENCE_MULTIPLIER = float(os.environ.get('UNCERTAIN_CONFIDENCE_MULTIPLIER', '0.7'))  # Reduce confidence in uncertain region

# GraphQL API Configuration
GRAPHQL_ENABLED = os.environ.get('GRAPHQL_ENABLED', 'true').lower() == 'true'
GRAPHQL_REQUIRE_AUTH = os.environ.get('GRAPHQL_REQUIRE_AUTH', 'true').lower() == 'true'

# Auto Scaling Configuration
AUTO_SCALING_ENABLED = os.environ.get('AUTO_SCALING_ENABLED', 'false').lower() == 'true'
CELERY_MIN_WORKERS = int(os.environ.get('CELERY_MIN_WORKERS', '2'))
CELERY_MAX_WORKERS = int(os.environ.get('CELERY_MAX_WORKERS', '10'))
CELERY_WORKER_SCALE_UP_THRESHOLD = int(os.environ.get('CELERY_WORKER_SCALE_UP_THRESHOLD', '50'))  # Tasks in queue
CELERY_WORKER_SCALE_DOWN_THRESHOLD = int(os.environ.get('CELERY_WORKER_SCALE_DOWN_THRESHOLD', '10'))  # Tasks in queue
CELERY_WORKER_SCALE_FACTOR = float(os.environ.get('CELERY_WORKER_SCALE_FACTOR', '1.5'))  # Multiplier for scaling
CELERY_WORKER_SCALE_DOWN_DELAY = int(os.environ.get('CELERY_WORKER_SCALE_DOWN_DELAY', '300'))  # Seconds to wait before scaling down

# Bulk Testing Configuration
BULK_TESTING_MAX_ITEMS = int(os.environ.get('BULK_TESTING_MAX_ITEMS', '100'))
ENABLE_BATCH_PROCESSING = os.environ.get('ENABLE_BATCH_PROCESSING', 'true').lower() == 'true'
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '16'))

# Dataset Configuration
INDIAN_DATASET_PATH = os.environ.get('INDIAN_DATASET_PATH', 'data/indian_dataset')
GLOBAL_DATASET_PATH = os.environ.get('GLOBAL_DATASET_PATH', 'data/global_dataset')
BALANCED_TRAINING_ENABLED = os.environ.get('BALANCED_TRAINING_ENABLED', 'true').lower() == 'true'
DATA_AUGMENTATION_STRENGTH = float(os.environ.get('DATA_AUGMENTATION_STRENGTH', '1.0'))

# Enhanced Validation Configuration
CROSS_VALIDATION_FOLDS = int(os.environ.get('CROSS_VALIDATION_FOLDS', '5'))
VALIDATION_SPLIT = float(os.environ.get('VALIDATION_SPLIT', '0.2'))
STORE_VALIDATION_RESULTS = os.environ.get('STORE_VALIDATION_RESULTS', 'true').lower() == 'true'
VALIDATION_RESULTS_PATH = os.environ.get('VALIDATION_RESULTS_PATH', 'data/validation_results')
