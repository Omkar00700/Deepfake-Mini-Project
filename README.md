HEAD
# Deepfake-Mini-Project

# DeepDefend - Advanced Deepfake Detection Platform for Indian Faces

DeepDefend is an AI-powered solution for detecting deepfake images and videos with high precision, specifically optimized for Indian faces, skin tones, and cultural contexts.

![DeepDefend Logo](./src/assets/logo.png)

## Features

- **Advanced Deepfake Detection**: Utilizes ensemble learning with multiple specialized models
- **Indian Face Specialization**: Optimized for Indian skin tones and facial features
- **Skin Tone Analysis**: Advanced skin tone analysis to detect inconsistencies in Indian skin tones
- **GAN Detection**: Specialized detection for GAN-generated images with artifact identification
- **Temporal Analysis**: For video deepfake detection with frame consistency measurement
- **Uncertainty Estimation**: Provides confidence scores and uncertainty metrics for more trustworthy results
- **Comprehensive Reporting**: Generate detailed PDF, JSON, or CSV reports with complete detection metrics
- **Diagnostic Visualization**: Visual tools to understand model decisions and prediction confidence

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- NPM or Yarn
- Python 3.8+ for backend API functionality
- CUDA-compatible GPU (recommended for faster processing)

### Installation

#### Backend Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/deepdefend.git
cd deepdefend
```

2. Create a Python virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install backend dependencies:
```bash
pip install -r requirements.txt
```

4. Start the backend server:
```bash
python app.py
```

#### Frontend Setup

1. Install frontend dependencies:
```bash
cd ../
npm install
# or
yarn install
```

2. Start the development server:
```bash
npm run dev
# or
yarn dev
```

3. Open http://localhost:3000 to view the application in your browser.

### Configuration

DeepDefend supports several configuration options:

- **API Endpoint**: Configure the API URL in `.env` or `src/services/config.ts`
- **Model Selection**: Choose between different detection models in the API configuration
- **Debugging**: Enable detailed logs and diagnostics in development mode

## Usage

1. **Upload an Image or Video**: Use the upload area on the main page to select a file for analysis
2. **View Results**: Examine detection probability, confidence metrics, and uncertainty visualization
3. **Download Report**: Generate a comprehensive report in PDF, JSON, or CSV format

## Advanced Features

### Ensemble Detection

DeepDefend utilizes multiple specialized models in an ensemble to improve detection accuracy:

- **Face-specific Models**: Specialized for different Indian demographics and skin tones
- **Artifact Detection**: Models focused on common deepfake artifacts
- **Multi-frame Analysis**: Temporal consistency checking for videos
- **GAN Fingerprint Detection**: Specialized detection for GAN-generated images

### Indian Skin Tone Analysis

Our advanced skin tone analyzer is specialized for Indian skin tones:

- **Classification**: Accurate classification into 6 Indian skin tone categories
- **Anomaly Detection**: Identification of unnatural skin smoothness and texture
- **Consistency Analysis**: Detection of inconsistent lighting and coloration
- **Authenticity Scoring**: Specialized scoring for Indian facial features

### Confidence Estimation

The system provides confidence scores based on:

- **Model Uncertainty**: Using Monte Carlo dropout and ensemble variance
- **Input Quality**: Evaluating image/video quality metrics
- **Temporal Consistency**: For video inputs, measuring frame-to-frame consistency
- **Skin Tone Consistency**: Analyzing consistency of skin tones across the face

## Troubleshooting

Common issues and solutions:

- **Report Download Issues**: Ensure file-saver package is installed correctly
- **API Connection Errors**: Check network connectivity and API endpoint configuration
- **Performance Issues**: Reduce video resolution or frame count for faster processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Roadmap

See the [ROADMAP.md](ROADMAP.md) file for our development plans and upcoming features.

## Acknowledgements

- Developed by the DeepDefend AI Research Team
- Special thanks to our contributors and advisors
- [DeepFake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge) for datasets and inspiration
- [FaceForensics++](https://github.com/ondyari/FaceForensics) for research and benchmarks
- [Indian Face Database](https://cvit.iiit.ac.in/projects/IMFDB/) for Indian face datasets

## Contact

For questions or support, please contact support@deepdefend.ai
365131d (Initial commit of my deepfake detection project)
