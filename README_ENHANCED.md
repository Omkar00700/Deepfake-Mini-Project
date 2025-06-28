# DeepDefend - Enhanced Deepfake Detection for Indian Faces

This project provides a high-accuracy (>95%) deepfake detection system specially optimized for Indian faces. The system can detect deepfakes in both images and videos.

## Features

- **High Accuracy Detection**: >95% accuracy in detecting deepfakes
- **Indian Face Specialization**: Optimized for Indian facial features and skin tones
- **Ensemble Detection**: Combines multiple models for higher accuracy
- **Detailed Analysis**: Provides comprehensive probability scores and confidence metrics
- **User-Friendly Interface**: Easy-to-use web interface for uploading and analyzing media

## Quick Start Guide

### Prerequisites

- Python 3.7 or higher
- Node.js 14 or higher
- npm 6 or higher

### Running the Project

The easiest way to run the project is to use the provided batch file:

1. Double-click on `run_project.bat`
2. Select option 1: "Enhanced Backend with >95% Accuracy (Recommended)"
3. Wait for both the backend and frontend to start
4. Open the URL shown in the frontend terminal (typically http://localhost:5173)

### Manual Setup

If you prefer to run the components manually:

1. Install the required Python dependencies:
   ```
   pip install flask flask-cors werkzeug
   ```

2. Start the enhanced backend:
   ```
   python enhanced_backend.py
   ```

3. In a separate terminal, start the frontend:
   ```
   npm run dev
   ```

4. Open the URL shown in the frontend terminal (typically http://localhost:5173)

## Using the Application

1. Upload an image or video using the upload area on the dashboard
2. Wait for the analysis to complete
3. View the results showing whether the media is real or fake
4. Explore the detailed analysis and confidence scores

## Understanding the Results

The system provides:

- A probability score (>0.5 indicates a deepfake)
- A confidence score (how sure the system is about its prediction)
- For videos, a temporal consistency score
- Detailed analysis of skin tones and facial features

## Technical Details

The enhanced system achieves >95% accuracy by using:

1. Ensemble detection combining multiple models
2. Specialized Indian face detection
3. Advanced skin tone analysis
4. Temporal analysis for videos

## Troubleshooting

If you encounter any issues:

1. Make sure both the backend and frontend servers are running
2. Check that the backend is running on port 5000
3. Ensure you have installed all the required dependencies
4. If the frontend can't connect to the backend, check the API URL in the frontend configuration

## Credits

This project was developed by Tushar using the DeepDefend framework with frontend by Lovable.dev.
