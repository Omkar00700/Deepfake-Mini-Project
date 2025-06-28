# DeepDefend Testing and Fine-tuning Guide

This guide provides step-by-step instructions for testing and fine-tuning the DeepDefend deepfake detection system with a focus on Indian faces and skin tones.

## Table of Contents

1. [Introduction](#introduction)
2. [Testing Framework](#testing-framework)
   - [Setting Up Test Datasets](#setting-up-test-datasets)
   - [Running Tests](#running-tests)
   - [Analyzing Test Results](#analyzing-test-results)
3. [Model Fine-tuning](#model-fine-tuning)
   - [Preparing Training Data](#preparing-training-data)
   - [Fine-tuning Models](#fine-tuning-models)
   - [Evaluating Fine-tuned Models](#evaluating-fine-tuned-models)
4. [Enhanced PDF Reports](#enhanced-pdf-reports)
   - [Generating Reports](#generating-reports)
   - [Customizing Reports](#customizing-reports)
5. [Troubleshooting](#troubleshooting)

## Introduction

DeepDefend's specialized detection system for Indian faces requires thorough testing and fine-tuning to achieve optimal performance. This guide will help you:

1. Test the system with diverse Indian faces to validate its performance
2. Fine-tune the models on Indian face datasets to improve accuracy
3. Generate enhanced PDF reports with clear visualizations

## Testing Framework

The testing framework (`test_framework.py`) provides tools for organizing test datasets, running tests, and analyzing results.

### Setting Up Test Datasets

#### 1. Collect Test Images

First, collect a diverse set of test images:

- **Real images**: Authentic photographs of Indian faces
- **Fake images**: Deepfake or GAN-generated images of Indian faces

Organize these into separate directories.

#### 2. Organize Test Dataset

Use the `TestDatasetManager` to organize your test images by skin tone:

```bash
python backend/test_framework.py --organize --source /path/to/real/images --real
python backend/test_framework.py --organize --source /path/to/fake/images
```

This will:
- Analyze each image to detect skin tone
- Organize images into categories based on skin tone
- Create a structured test dataset

#### 3. Create a Balanced Test Set

Create a balanced test set with equal representation of each skin tone category:

```bash
python backend/test_framework.py --create-test-set --samples 20
```

This creates a test set with 20 samples per skin tone category for both real and fake images.

### Running Tests

#### 1. Run the Test Evaluator

Evaluate the system's performance on your test set:

```bash
python backend/test_framework.py --evaluate --test-dir balanced_test_set
```

This will:
- Process each image in the test set
- Calculate accuracy, false positives, and false negatives
- Generate detailed metrics by skin tone
- Save results to the `test_results` directory

#### 2. Batch Testing

For large-scale testing, you can use the batch testing functionality:

```python
from test_framework import TestEvaluator

evaluator = TestEvaluator()
metrics = evaluator.evaluate_test_set("path/to/test_set")
evaluator.generate_report(metrics)
```

### Analyzing Test Results

The test framework generates several outputs to help you analyze results:

1. **JSON Results**: Detailed results for each image
2. **CSV Reports**: Breakdown of performance by skin tone
3. **Visual Reports**: Charts showing accuracy, confusion matrix, etc.

Key metrics to analyze:

- **Overall accuracy**: Should be >85% for a well-performing system
- **Accuracy by skin tone**: Look for variations across different skin tones
- **False positives/negatives**: Identify which types of images cause errors
- **Processing time**: Ensure performance is acceptable

## Model Fine-tuning

The fine-tuning script (`finetune_indian_faces.py`) helps you improve model performance on Indian faces.

### Preparing Training Data

#### 1. Collect Training Data

Collect a large dataset of Indian faces:
- Real images: At least 1000 authentic photographs
- Fake images: At least 1000 deepfake or GAN-generated images

#### 2. Prepare Dataset for Training

Prepare your dataset for training:

```bash
python backend/finetune_indian_faces.py --prepare --real-dir /path/to/real/images --fake-dir /path/to/fake/images
```

This will:
- Split your data into training (70%), validation (15%), and testing (15%) sets
- Organize the data in the required directory structure

### Fine-tuning Models

#### 1. Fine-tune the Model

Fine-tune the model on your prepared dataset:

```bash
python backend/finetune_indian_faces.py --train --model-type efficientnet --epochs 10 --fine-tune-epochs 5 --batch-size 32 --model-name indian_deepfake_detector
```

Options:
- `--model-type`: Choose between `efficientnet` or `resnet`
- `--epochs`: Number of initial training epochs
- `--fine-tune-epochs`: Number of fine-tuning epochs
- `--batch-size`: Batch size for training
- `--model-name`: Name for the saved model

#### 2. Advanced Fine-tuning

For more advanced fine-tuning, you can use the Python API:

```python
from finetune_indian_faces import DatasetPreparation, ModelFinetuner

# Prepare dataset
dataset_prep = DatasetPreparation()
dataset_prep.prepare_from_directories("real_images", "fake_images")
train_gen, val_gen, test_gen = dataset_prep.create_data_generators(batch_size=32)

# Fine-tune model
finetuner = ModelFinetuner()
model = finetuner.create_model(model_type="efficientnet")
model, history = finetuner.fine_tune(model, train_gen, val_gen, epochs=10, fine_tune_epochs=5)

# Evaluate model
evaluation = finetuner.evaluate_model(model, test_gen)
finetuner.plot_training_history(history)
```

### Evaluating Fine-tuned Models

After fine-tuning, evaluate your model:

1. **Training History**: Check the training and validation accuracy/loss curves
2. **Test Set Performance**: Evaluate on the test set
3. **Confusion Matrix**: Analyze false positives and false negatives
4. **Real-world Testing**: Test on new, unseen images

## Enhanced PDF Reports

The enhanced PDF report generator (`enhanced_pdf_report.py`) creates visually appealing and informative reports.

### Generating Reports

#### 1. Generate a Report from Detection Results

Generate a PDF report from detection results:

```bash
python backend/enhanced_pdf_report.py --result /path/to/detection_result.json --image /path/to/original_image.jpg --output-dir reports
```

#### 2. Programmatic Report Generation

Generate reports from your code:

```python
from enhanced_pdf_report import EnhancedPDFReportGenerator

# Create report generator
report_generator = EnhancedPDFReportGenerator(output_dir="reports")

# Generate report
report_path = report_generator.generate_report(detection_result, image_path)
print(f"Report generated: {report_path}")
```

### Customizing Reports

You can customize the reports by modifying the `enhanced_pdf_report.py` file:

1. **Add new sections**: Create new methods like `_add_custom_section`
2. **Modify styles**: Update the styles in the `__init__` method
3. **Change visualizations**: Modify the chart generation methods

## Troubleshooting

### Common Testing Issues

1. **No faces detected**:
   - Ensure images have clear, visible faces
   - Check face detector confidence threshold in `config.py`

2. **Low accuracy**:
   - Verify test dataset quality
   - Ensure balanced representation of different skin tones
   - Check for biases in your test data

3. **Slow processing**:
   - Reduce image resolution for faster processing
   - Enable parallel processing in `config.py`

### Fine-tuning Issues

1. **Training errors**:
   - Check GPU memory usage
   - Reduce batch size if out of memory
   - Verify dataset structure

2. **Overfitting**:
   - Increase data augmentation
   - Add dropout layers
   - Reduce model complexity

3. **Poor generalization**:
   - Ensure diverse training data
   - Add more varied examples
   - Use cross-validation

### Report Generation Issues

1. **Missing visualizations**:
   - Check for required libraries (matplotlib, reportlab)
   - Verify image paths

2. **PDF errors**:
   - Check disk space
   - Verify write permissions to output directory

## Conclusion

By following this guide, you can thoroughly test and fine-tune DeepDefend for optimal performance on Indian faces. Regular testing and fine-tuning will help maintain and improve the system's accuracy over time.

For additional help, refer to the project documentation or contact the DeepDefend support team.