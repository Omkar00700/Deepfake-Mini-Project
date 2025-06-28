# DeepDefend - Advanced Deepfake Detection API with Indian Face Optimization

DeepDefend is a sophisticated backend API for detecting deepfake media, providing state-of-the-art detection capabilities through multiple deep learning models and a modular, maintainable architecture. The system has been enhanced with specialized optimizations for Indian faces and features continuous retraining for improved accuracy.

## Project Overview

DeepDefend is designed to identify manipulated images and videos (deepfakes) with high accuracy and performance. The system leverages multiple pre-trained models and provides detailed confidence metrics to help users understand detection results.

**Key Features:**
- Multiple detection models (EfficientNet, Xception, MesoNet)
- Dynamic weighted ensemble approach with automatic optimization
- Scene-based video frame sampling for improved video analysis
- Temporal consistency analysis for more accurate video detection
- Specialized Indian face detection and preprocessing
- Dynamic model switching at runtime
- Face detection and region-specific analysis
- Parallel processing for videos with frame caching
- Comprehensive metrics and evaluation capabilities
- Continuous model retraining and feedback collection
- User feedback mechanism for detection improvement
- Confidence calibration for more realistic assessment
- Advanced image enhancement and preprocessing
- Detailed performance analytics and reporting
- JWT-based authentication
- Rate limiting for API protection
- RESTful API for easy integration
- GraphQL API for flexible data querying
- Bulk testing capabilities for accuracy assessment

## Architecture

DeepDefend uses a modular architecture with clear separation of concerns:

### Advanced Preprocessing (`preprocessing.py`)
- Provides sophisticated image enhancement techniques for better detection
- Implements scene-based video frame sampling to capture important moments
- Detects and extracts faces using a DNN-based detector with Indian face optimizations
- Normalizes and prepares images for the detection models
- Implements frame similarity detection for caching and performance

### Inference Engine (`inference.py`)
- Manages model inference and prediction using dynamic weighted ensemble
- Processes faces with retry mechanisms for robustness
- Supports parallel face processing for improved performance
- Provides model switching capabilities
- Implements adaptive ensemble weighting based on performance metrics

### Temporal Analysis (`postprocessing.py`)
- Analyzes consistency across video frames
- Detects unnatural patterns that indicate manipulation
- Aggregates detection results for images and videos
- Calculates confidence scores based on multiple factors
- Handles thresholding and final classification
- Formats results for API responses

### Continuous Evaluation (`metrics_logger.py`)
- Tracks and logs detection metrics in real-time
- Provides comprehensive performance analytics
- Enables data-driven model improvements
- Supports historical trend analysis

### Indian Face-Specific Optimizations (`indian_face_utils.py`)
- Provides specialized face detection and preprocessing for Indian facial features
- Adjusts contrast and brightness for better detection on varied skin tones
- Applies adaptive noise reduction for different imaging conditions
- Enhances edge detection and facial landmark recognition
- Falls back to standard detection when needed

### Training Pipeline (`model_trainer.py`)
- Implements a sophisticated training and fine-tuning pipeline
- Supports balanced dataset creation with augmentations specific to Indian faces
- Provides cross-validation and performance metrics
- Enables fine-tuning of pre-trained models with regional specialization
- Includes comprehensive evaluation on test datasets

### Preprocessing (`preprocessing.py`)
- Handles image and video preprocessing
- Detects and extracts faces using a DNN-based detector
- Normalizes and prepares images for the detection models
- Extracts frames from videos with configurable sampling rates
- Implements frame similarity detection for caching

### Inference (`inference.py`)
- Manages model inference and prediction
- Processes faces with retry mechanisms for robustness
- Supports parallel face processing for improved performance
- Provides model switching capabilities
- Implements model ensemble for improved accuracy
- Includes confidence calibration

### Postprocessing (`postprocessing.py`)
- Aggregates detection results for images and videos
- Calculates confidence scores based on multiple factors
- Handles thresholding and final classification
- Formats results for API responses

### Model Management (`model_manager.py`)
- Loads and manages multiple deepfake detection models
- Allows dynamic switching between models at runtime
- Provides model information and performance metrics
- Implements auto-switching based on confidence thresholds
- Supports model ensemble for improved accuracy

### Continuous Learning (`model_retraining.py`)
- Periodically evaluates model performance on validation datasets
- Triggers retraining when performance drops below thresholds
- Collects and utilizes user feedback for continuous improvement
- Maintains training history and performance metrics
- Provides API endpoints for monitoring retraining status

### Detection Handler (`detection_handler.py`)
- Coordinates the end-to-end detection pipeline
- Processes both images and videos
- Integrates preprocessing, inference, and postprocessing
- Handles errors and provides detailed response data
- Employs region-optimized processing for Indian faces

### Feedback System (`feedback_api.py`)
- Collects user feedback on detection accuracy
- Provides endpoints for bulk testing and analysis
- Aggregates feedback for model improvement
- Enables detailed region-specific feedback

### Metrics Collection (`metrics.py`)
- Tracks performance metrics in real-time
- Records processing times, detection rates, and model usage
- Provides API endpoint for monitoring system performance
- Supports performance optimization and troubleshooting

### Authentication (`auth.py`)
- Handles JWT token generation and validation
- Provides decorators for protecting endpoints
- Supports optional authentication modes

### Rate Limiting (`rate_limiter.py`)
- Implements API rate limiting to prevent abuse
- Configurable request limits and time windows
- Provides informative headers and response messages

## Installation

### Prerequisites
- Python 3.10 or higher
- pip or Poetry for dependency management
- 4GB+ RAM recommended for model inference
- CUDA-compatible GPU (optional, for faster processing)

### Setup with pip

1. Clone the repository:
   ```
   git clone https://github.com/your-username/deepdefend.git
   cd deepdefend
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r backend/requirements.txt
   ```

### Environment Variables

Create a `.env` file in the backend directory with the following variables:

```
# Database Configuration
DATABASE_URL=sqlite:///detections.db  # For development
# DATABASE_URL=postgresql://user:password@localhost:5432/deepdefend  # For production

# Model Configuration
DEFAULT_MODEL=efficientnet  # Options: efficientnet, xception, mesonet
DEEPFAKE_THRESHOLD=0.5
CONFIDENCE_THRESHOLD=0.7
ENABLE_ENSEMBLE_DETECTION=True  # Use ensemble of models for better accuracy

# Enhanced Detection Features
ENABLE_ADVANCED_PREPROCESSING=True
USE_SCENE_BASED_SAMPLING=True
ENABLE_TEMPORAL_ANALYSIS=True

# Indian Face Detection Enhancement
INDIAN_FACE_DETECTION_ENABLED=True
FACE_CONTRAST_ADJUSTMENT=1.2
FACE_BRIGHTNESS_ADJUSTMENT=10.0
NOISE_REDUCTION_STRENGTH=5
ENABLE_ADAPTIVE_PREPROCESSING=True

# Video Processing
VIDEO_MAX_FRAMES=30
ENABLE_PARALLEL_PROCESSING=True
MAX_WORKERS=4

# Continuous Model Retraining
RETRAINING_ENABLED=True
RETRAINING_INTERVAL_HOURS=24
RETRAINING_PERFORMANCE_THRESHOLD=0.7
FEEDBACK_COLLECTION_ENABLED=True

# Metrics Logging
METRICS_STORAGE_PATH=data/metrics
METRICS_RETENTION_DAYS=30
ERROR_LOG_MAX_ENTRIES=1000

# Authentication
REQUIRE_AUTH=False  # Set to True to enable authentication for all endpoints
JWT_SECRET=your-secret-key-change-in-production
JWT_EXPIRATION_MINUTES=60

# Rate Limiting
ENABLE_RATE_LIMITING=False  # Set to True to enable rate limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600  # In seconds (1 hour default)

# Logging
LOG_LEVEL=INFO
STRUCTURED_LOGGING=False  # Set to True for JSON structured logging
```

## Training Pipeline

DeepDefend includes a comprehensive training pipeline to fine-tune models for better detection accuracy, especially for Indian faces.

### Training Data Organization

The training data should be organized as follows:

```
data/
  ├── indian_dataset/
  │   ├── real/
  │   │   └── [real images]
  │   └── fake/
  │       └── [fake images]
  ├── global_dataset/
  │   ├── real/
  │   │   └── [real images]
  │   └── fake/
  │       └── [fake images]
  ├── validation/
  │   ├── real/
  │   │   └── [real images]
  │   └── fake/
  │       └── [fake images]
  └── test/
      ├── real/
      │   └── [real images]
      └── fake/
          └── [fake images]
```

### Training a New Model

```python
from model_trainer import DeepfakeModelTrainer

# Initialize trainer for EfficientNet
trainer = DeepfakeModelTrainer(model_name="efficientnet")

# Build a new model
trainer.build_model()

# Train the model
metrics = trainer.train(
    train_dir="data/combined_dataset",  # Combined dataset with Indian and global images
    validation_dir="data/validation",
    epochs=20,
    batch_size=32
)

# Test the model
test_metrics = trainer.test(test_dir="data/test")

# Save the model summary
trainer.save_model_summary()
```

### Fine-tuning an Existing Model

```python
from model_trainer import DeepfakeModelTrainer

# Initialize trainer
trainer = DeepfakeModelTrainer(model_name="efficientnet")

# Load existing model
trainer.load_model("models/efficientnet_deepfake.h5")

# Fine-tune the model
metrics = trainer.train(
    train_dir="data/indian_dataset",  # Indian-specific dataset
    validation_dir="data/validation",
    epochs=10,
    batch_size=32,
    fine_tune=True,
    unfreeze_layers=5  # Only fine-tune the top 5 layers
)
```

### Creating a Combined Dataset

We recommend combining Indian-specific datasets with global datasets to create a balanced dataset that performs well on diverse faces:

```python
import os
import shutil

# Create combined dataset directories
os.makedirs("data/combined_dataset/real", exist_ok=True)
os.makedirs("data/combined_dataset/fake", exist_ok=True)

# Copy Indian dataset with priority
for class_name in ["real", "fake"]:
    # Copy Indian dataset
    indian_dir = f"data/indian_dataset/{class_name}"
    target_dir = f"data/combined_dataset/{class_name}"
    
    for filename in os.listdir(indian_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            shutil.copy(
                os.path.join(indian_dir, filename),
                os.path.join(target_dir, f"indian_{filename}")
            )
    
    # Copy global dataset (subset to balance)
    global_dir = f"data/global_dataset/{class_name}"
    global_files = [f for f in os.listdir(global_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Calculate how many global files to copy (e.g., match Indian count)
    indian_count = len([f for f in os.listdir(indian_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Copy subset of global files
    for filename in global_files[:indian_count]:
        shutil.copy(
            os.path.join(global_dir, filename),
            os.path.join(target_dir, f"global_{filename}")
        )
```

## Bulk Testing

DeepDefend provides a bulk testing API endpoint to evaluate detection accuracy on multiple images or videos:

```python
import requests
import json

# Define test items
test_items = [
    {
        "path": "/path/to/real_image.jpg",
        "type": "image",
        "actual_label": "real",
        "metadata": {"region": "south_asia"}
    },
    {
        "path": "/path/to/fake_image.jpg",
        "type": "image",
        "actual_label": "deepfake",
        "metadata": {"region": "south_asia"}
    }
    # Add more test items...
]

# Send bulk test request
response = requests.post(
    "http://localhost:5000/api/feedback/bulk-analysis",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json={
        "items": test_items,
        "options": {
            "model": "efficientnet",  # Optional: specify model
            "ensemble": True  # Optional: use ensemble detection
        }
    }
)

# Print results
result = response.json()
print(f"Accuracy: {result['metrics']['accuracy']:.2f}")
print(f"Real accuracy: {result['metrics']['real_accuracy']:.2f}")
print(f"Fake accuracy: {result['metrics']['fake_accuracy']:.2f}")
```

## Feedback Mechanism

DeepDefend includes a feedback mechanism that allows users to provide feedback on detection results. This feedback is collected and used for continuous model improvement.

### Submitting Feedback via API

```python
import requests
import json

response = requests.post(
    "http://localhost:5000/api/feedback",
    json={
        "detection_id": "1234567890",
        "correct": False,
        "actual_label": "real",
        "confidence": 0.8,
        "region": "south_asia",
        "metadata": {
            "comments": "This is a real Indian person incorrectly classified as deepfake",
            "source": "user_upload"
        }
    }
)

print(response.json())
```

## Indian Face Detection Improvements

DeepDefend includes specialized optimizations for better detection of Indian faces:

1. **Enhanced Preprocessing**: Adjusted contrast, brightness, and color handling specifically for Indian skin tones and lighting conditions common in South Asian regions.

2. **Adaptive CLAHE**: Contrast Limited Adaptive Histogram Equalization specifically tuned for darker skin tones to preserve important facial features.

3. **Noise Reduction**: Enhanced noise reduction to handle lower quality images common in varied lighting conditions.

4. **Confidence Calibration**: Adjusted confidence scores to avoid over-confidence on real images, addressing a common issue with deepfake detectors.

5. **Region-Specific Dataset**: Inclusion of specialized Indian face datasets for training and validation.

6. **Custom Augmentations**: Custom data augmentations targeting lighting variations, skin tones, and cultural factors like tilak, bindi, etc.

7. **Face Detection Fallback**: Automatic fallback to standard detection if Indian-specific detection fails to find faces.

## Running the Backend

### Development Server

Start the Flask development server:

```
cd backend
python app.py
```

The API will be available at `http://localhost:5000/api/`

### Verify API is Running

Test the API status endpoint:

```
curl http://localhost:5000/api/status
```

You should receive a JSON response indicating the API is online.

## Configuration

DeepDefend can be configured through environment variables or a configuration file. See the Environment Variables section for details on available configuration options.

### Runtime Model Switching

You can dynamically switch models at runtime using the model switch API endpoint:

```
POST /api/models/switch
Content-Type: application/json
Authorization: Bearer <your-jwt-token>

{
  "model": "xception"
}
```

## API Endpoints

### Status and Metrics
- `GET /api/status`: Check API status and current model information
- `GET /api/metrics`: Get real-time performance metrics (authenticated)
- `GET /api/evaluate`: Evaluate model performance on sample dataset (authenticated)
- `GET /api/video-metrics`: Get detailed video processing metrics (authenticated)

### Authentication
- `POST /api/auth/token`: Get authentication token

### Detection
- `POST /api/detect`: Upload an image for deepfake detection
- `POST /api/detect-video`: Upload a video for deepfake detection

### Feedback and Training
- `POST /api/feedback`: Submit feedback for a detection result
- `GET /api/feedback/status`: Get feedback collection status
- `POST /api/feedback/bulk-analysis`: Perform bulk testing and analysis

### Model Management
- `GET /api/models`: Get information about available models
- `POST /api/models/switch`: Switch to a different model (authenticated)
- `GET /api/models/retraining`: Get model retraining status and history
- `POST /api/models/retrain`: Manually trigger model retraining (authenticated)

### GraphQL
- `POST /api/graphql`: GraphQL endpoint for flexible data querying

## Authentication

The API supports JWT-based authentication. To use protected endpoints:

1. Get a token:
   ```
   POST /api/auth/token
   Content-Type: application/json
   
   {
     "user_id": "your-user-id"
   }
   ```

2. Use the token in API requests:
   ```
   GET /api/metrics
   Authorization: Bearer <your-jwt-token>
   ```

## Testing

### Running Unit Tests

Run all unit tests:

```
cd backend
python run_tests.py
```

### With Coverage Report

Run tests with coverage reporting:

```
python run_tests.py --coverage
```

## Deployment

### Docker Deployment

A Dockerfile is provided for containerized deployment:

```
# Build the image
docker build -t deepdefend .

# Run the container
docker run -p 5000:5000 --env-file .env deepdefend
```

### Kubernetes Deployment

Kubernetes YAML files are provided in the `kubernetes/` directory for deploying the application in a Kubernetes cluster:

```
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/secret.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/ingress.yaml
```

## Auto-Scaling

DeepDefend supports automatic scaling of Celery workers based on queue size:

1. Configure auto-scaling parameters in `.env`:
   ```
   AUTO_SCALING_ENABLED=true
   CELERY_MIN_WORKERS=2
   CELERY_MAX_WORKERS=10
   CELERY_WORKER_SCALE_UP_THRESHOLD=50
   CELERY_WORKER_SCALE_DOWN_THRESHOLD=10
   ```

2. For Kubernetes deployment, a Horizontal Pod Autoscaler is included:
   ```
   kubectl apply -f kubernetes/hpa.yaml
   ```

## Future Work

1. **Enhanced Region-Specific Models**: Develop more specialized models for different global regions.

2. **Improved Video Temporal Analysis**: Better detection of inconsistencies across video frames.

3. **Mobile-Optimized Models**: Create smaller, faster models for mobile deployment.

4. **Adversarial Training**: Implement adversarial training to make models more robust against evasion.

5. **Self-Supervised Learning**: Explore self-supervised learning to improve feature extraction from unlabeled data.

## Acknowledgments

- [OpenCV](https://opencv.org/) for image and video processing
- [TensorFlow](https://www.tensorflow.org/) for deep learning capabilities
- [Flask](https://flask.palletsprojects.com/) for the web API framework
- [PyJWT](https://pyjwt.readthedocs.io/) for JWT authentication
- [Indian Deepfake Detection Project](https://github.com/example/indian-deepfake-detection) for region-specific insights and techniques

## License

This project is licensed under the MIT License - see the LICENSE file for details.
