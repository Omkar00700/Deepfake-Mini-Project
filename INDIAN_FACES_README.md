# DeepDefend Indian Face Enhancement Guide

This guide provides step-by-step instructions for enhancing DeepDefend's deepfake detection capabilities for Indian faces.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Step 1: Collect Indian Face Data](#step-1-collect-indian-face-data)
4. [Step 2: Run Initial Tests](#step-2-run-initial-tests)
5. [Step 3: Fine-tune Models](#step-3-fine-tune-models)
6. [Step 4: Validate Improvements](#step-4-validate-improvements)
7. [Step 5: Deploy Enhanced Reports](#step-5-deploy-enhanced-reports)
8. [Troubleshooting](#troubleshooting)

## Introduction

DeepDefend's deepfake detection system has been enhanced with specialized capabilities for Indian faces. This includes:

- Improved face detection optimized for Indian facial features
- Skin tone analysis specific to Indian skin tones
- Fine-tuned models trained on diverse Indian face datasets
- Enhanced PDF reports with Indian face-specific information

## Prerequisites

Before you begin, ensure you have the following:

- Python 3.8 or higher
- TensorFlow 2.6 or higher
- OpenCV 4.5 or higher
- All dependencies listed in `backend/requirements.txt`

Install the required dependencies:

```bash
pip install -r backend/requirements.txt
```

## Step 1: Collect Indian Face Data

The first step is to collect and organize a diverse dataset of Indian faces.

### Using the Collection Script

```bash
python backend/collect_indian_faces.py --real-dir /path/to/real/indian/faces --fake-dir /path/to/fake/indian/faces --organize --analyze
```

### Parameters:

- `--real-dir`: Directory containing real Indian face images
- `--fake-dir`: Directory containing fake (deepfake) Indian face images
- `--organize`: Organize the collected faces into train/val/test splits
- `--analyze`: Analyze the dataset and generate statistics
- `--output-dir`: Directory to save the collected faces (default: `data/indian_faces`)

### Expected Output:

- Organized dataset in `data/indian_faces` with train/val/test splits
- Dataset statistics in `data/indian_faces/dataset_stats.json`

### Manual Collection

If you prefer to collect data manually, ensure you have:

1. A diverse set of real Indian face images covering different:
   - Skin tones (fair, wheatish, medium, dusky, dark)
   - Ages (young, middle-aged, elderly)
   - Genders
   - Facial features

2. A corresponding set of deepfake images of Indian faces

Organize them in the following structure:

```
data/indian_faces/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

## Step 2: Run Initial Tests

Before fine-tuning, evaluate the current system to identify areas for improvement.

### Using the Testing Script

```bash
python backend/test_indian_faces.py --dataset-dir data/indian_faces/test --output-dir test_results
```

### Parameters:

- `--dataset-dir`: Directory containing test data (should have 'real' and 'fake' subdirectories)
- `--model-path`: Path to the model to test (optional, uses default model if not specified)
- `--output-dir`: Directory to save test results (default: `test_results`)

### Expected Output:

- Test results in `test_results/test_results.json`
- Confusion matrix plot in `test_results/confusion_matrix.png`
- Skin tone performance plot in `test_results/skin_tone_performance.png`

## Step 3: Fine-tune Models

Fine-tune the deepfake detection models on Indian faces.

### Using the Fine-tuning Script

```bash
python backend/run_indian_finetuning.py --dataset-dir data/indian_faces --model-type efficientnet --epochs 10 --fine-tune-epochs 5
```

### Parameters:

- `--dataset-dir`: Directory containing the dataset (with train/val/test splits)
- `--model-type`: Type of model to fine-tune (`efficientnet` or `resnet`)
- `--epochs`: Number of epochs for initial training (default: 10)
- `--fine-tune-epochs`: Number of epochs for fine-tuning (default: 5)
- `--batch-size`: Batch size for training (default: 32)
- `--model-name`: Name for the saved model (default: `indian_deepfake_detector`)

### Expected Output:

- Fine-tuned model in `fine_tuned_models/indian_deepfake_detector.h5`
- Training history plot in `fine_tuned_models/indian_deepfake_detector_training_history.png`
- Evaluation results in `fine_tuned_models/indian_deepfake_detector_evaluation.json`
- Fine-tuning results in `fine_tuned_models/indian_deepfake_detector_finetuning_results.json`

## Step 4: Validate Improvements

Validate the improvements by comparing the original and fine-tuned models.

### Using the Validation Script

```bash
python backend/validate_improvements.py --original-model /path/to/original/model.h5 --fine-tuned-model fine_tuned_models/indian_deepfake_detector.h5 --test-dataset data/indian_faces/test
```

### Parameters:

- `--original-model`: Path to the original model
- `--fine-tuned-model`: Path to the fine-tuned model
- `--test-dataset`: Directory containing test data
- `--output-dir`: Directory to save validation results (default: `validation_results`)

### Expected Output:

- Validation results in `validation_results/validation_results.json`
- Metric comparison plot in `validation_results/metric_comparison.png`
- Skin tone comparison plot in `validation_results/skin_tone_comparison.png`

## Step 5: Deploy Enhanced Reports

Deploy the enhanced PDF reports with Indian face-specific information.

### Using the Report Enhancement Script

```bash
python backend/enhance_pdf_reports.py --integrate --generate-sample
```

### Parameters:

- `--integrate`: Integrate enhanced PDF reports into the main application
- `--generate-sample`: Generate a sample enhanced PDF report
- `--output-dir`: Directory to save sample reports (default: `enhanced_reports`)

### Expected Output:

- Modified `app.py` to use enhanced PDF reports
- Sample enhanced PDF report in `enhanced_reports/`

## Troubleshooting

### Common Issues and Solutions

1. **Missing dependencies**
   - Ensure all dependencies are installed: `pip install -r backend/requirements.txt`
   - For MTCNN issues, try: `pip install mtcnn tensorflow`

2. **GPU memory errors**
   - Reduce batch size: `--batch-size 16` or lower
   - Enable memory growth: Set environment variable `TF_FORCE_GPU_ALLOW_GROWTH=true`

3. **Dataset issues**
   - Ensure dataset structure is correct (train/val/test with real/fake subdirectories)
   - Check image formats (JPG, PNG)
   - Verify face detection works on your images

4. **Model loading errors**
   - Check model path is correct
   - Ensure TensorFlow version compatibility

### Getting Help

If you encounter issues not covered here, please:

1. Check the logs for detailed error messages
2. Consult the main DeepDefend documentation
3. Contact support with the error details and logs

## Next Steps

After completing these steps, your DeepDefend system will be enhanced for Indian faces with:

1. Improved detection accuracy across diverse Indian skin tones
2. Better handling of Indian facial features
3. Enhanced reporting with skin tone analysis
4. More reliable deepfake detection for Indian faces

Continue to collect feedback and data to further improve the system over time.