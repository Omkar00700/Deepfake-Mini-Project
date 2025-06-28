# Advanced Deepfake Detection Features

This document describes the advanced features implemented to improve the deepfake detection system's accuracy, robustness, and explainability.

## Table of Contents

1. [Self-Supervised Pretraining](#1-self-supervised-pretraining)
2. [Vision Transformer (ViT) Head](#2-vision-transformer-vit-head)
3. [Adversarial Training](#3-adversarial-training)
4. [Multi-Modal Fusion](#4-multi-modal-fusion)
5. [Knowledge Distillation](#5-knowledge-distillation)
6. [Advanced Cross-Validation & Ensembling](#6-advanced-cross-validation--ensembling)
7. [Explainability & Monitoring](#7-explainability--monitoring)
8. [Quantization & Edge Deployment](#8-quantization--edge-deployment)
9. [Installation & Setup](#9-installation--setup)
10. [Usage Examples](#10-usage-examples)
11. [Expected Performance Gains](#11-expected-performance-gains)

## 1. Self-Supervised Pretraining

Self-supervised pretraining helps the model learn robust features from unlabeled data before fine-tuning on labeled real/fake images.

### Implemented Methods:

- **SimCLR (Simple Framework for Contrastive Learning of Visual Representations)**
  - Uses contrastive learning to train the model to distinguish between different augmentations of the same image
  - Helps the model learn invariant features that are useful for deepfake detection

- **MoCo (Momentum Contrast)**
  - Maintains a queue of negative samples for contrastive learning
  - Uses a momentum encoder to provide consistent feature representations

### Usage:

```bash
python backend/train_advanced.py --ssl-pretrain --ssl-method simclr --train-dir /path/to/train --val-dir /path/to/val --unlabeled-dir /path/to/unlabeled
```

## 2. Vision Transformer (ViT) Head

Adds a Vision Transformer head on top of CNN features to capture global relationships between image patches.

### Features:

- Lightweight ViT head that processes feature maps from CNN backbone
- Self-attention mechanism to capture long-range dependencies
- Fusion of CNN and ViT features for improved performance

### Usage:

```bash
python backend/train_advanced.py --vit-head --train-dir /path/to/train --val-dir /path/to/val
```

## 3. Adversarial Training

Implements adversarial training to make the model robust against subtle perturbations.

### Implemented Methods:

- **FGSM (Fast Gradient Sign Method)**
  - Fast, single-step adversarial attack
  - Adds perturbations in the direction of the gradient of the loss

- **PGD (Projected Gradient Descent)**
  - More powerful, multi-step adversarial attack
  - Iteratively applies FGSM with small step sizes

### Usage:

```bash
python backend/train_advanced.py --adversarial-train --adversarial-method pgd --train-dir /path/to/train --val-dir /path/to/val
```

## 4. Multi-Modal Fusion

Extracts audio features from videos and fuses them with visual features for improved detection.

### Features:

- MFCC (Mel-Frequency Cepstral Coefficients) extraction from audio
- 1D-CNN or LSTM for audio feature processing
- Attention-based fusion of audio and visual features

### Usage:

```bash
python backend/train_advanced.py --multimodal --train-dir /path/to/train --val-dir /path/to/val
```

## 5. Knowledge Distillation

Trains a smaller, faster model using the outputs of a larger, more accurate model.

### Features:

- Teacher-student training paradigm
- Temperature scaling for soft targets
- Support for MobileNetV2 and EfficientNetB0 student models

### Usage:

```bash
python backend/train_advanced.py --distill --student-model mobilenetv2 --train-dir /path/to/train --val-dir /path/to/val
```

## 6. Advanced Cross-Validation & Ensembling

Implements stratified k-fold cross-validation and model ensembling for improved performance.

### Features:

- Stratified k-fold cross-validation
- Model ensembling with different combination methods:
  - Simple averaging
  - Weighted averaging
  - Stacking

### Usage:

```bash
python backend/train_advanced.py --cross-validation --n-folds 5 --train-dir /path/to/train --val-dir /path/to/val
```

## 7. Explainability & Monitoring

Provides tools for understanding model predictions and monitoring performance.

### Features:

- **Explainability Methods:**
  - Grad-CAM for visual explanations
  - Integrated Gradients for feature attribution
  - SHAP (SHapley Additive exPlanations) for feature importance

- **Monitoring:**
  - Performance metrics tracking
  - Drift detection
  - Visualization tools

### Usage:

```bash
# API endpoint for explanations
curl -X POST -F "file=@image.jpg" -F "method=grad_cam" http://localhost:5000/api/explain/image

# Get explanation for a specific frame
curl -X GET http://localhost:5000/api/explain/frame/12345?method=shap
```

## 8. Quantization & Edge Deployment

Tools for optimizing models for deployment on edge devices.

### Features:

- TensorFlow Lite conversion with quantization
- ONNX export for cross-platform deployment
- Benchmarking tools for size, latency, and accuracy

### Usage:

```bash
# Export to TFLite with INT8 quantization
python backend/export_quantized.py --format tflite --quantization-type int8 --model-path /path/to/model.h5

# Export to ONNX
python backend/export_quantized.py --format onnx --model-path /path/to/model.h5
```

## 9. Installation & Setup

### Backend Dependencies

Install the required Python packages:

```bash
cd backend
pip install -r requirements.txt
```

### Frontend Dependencies

Install the required npm packages:

```bash
npm install
```

## 10. Usage Examples

### Complete Training Pipeline

```bash
# 1. Self-supervised pretraining
python backend/train_advanced.py --ssl-pretrain --ssl-method simclr --unlabeled-dir /path/to/unlabeled --output-dir models/ssl

# 2. Full training with all advanced features
python backend/train_advanced.py \
  --model-name efficientnet_b3 \
  --ssl-pretrain \
  --vit-head \
  --adversarial-train \
  --multimodal \
  --cross-validation \
  --n-folds 5 \
  --train-dir /path/to/train \
  --val-dir /path/to/val \
  --output-dir models/advanced

# 3. Knowledge distillation
python backend/train_advanced.py \
  --distill \
  --student-model mobilenetv2 \
  --train-dir /path/to/train \
  --val-dir /path/to/val \
  --output-dir models/distilled

# 4. Export for edge deployment
python backend/export_quantized.py \
  --format tflite \
  --quantization-type int8 \
  --model-path models/distilled/final_model.h5 \
  --output models/deployed/model_int8.tflite
```

### Running the Application

```bash
# Start the backend
cd backend
python app.py

# Start the frontend
npm run dev
```

## 11. Expected Performance Gains

The implemented advanced features are expected to provide the following performance improvements:

| Feature | Expected Gain | Notes |
|---------|---------------|-------|
| Self-Supervised Pretraining | +2-3% AUC | Especially helpful with limited labeled data |
| Vision Transformer Head | +1-2% AUC | Improves detection of global manipulations |
| Adversarial Training | +1-2% AUC | Enhances robustness against adversarial examples |
| Multi-Modal Fusion | +2-3% AUC | Significant for videos with audio-visual inconsistencies |
| Knowledge Distillation | Minimal AUC loss, 2-4x speedup | Enables deployment on resource-constrained devices |
| Cross-Validation & Ensembling | +1-2% AUC | Reduces variance and improves generalization |
| Combined Approach | +5-10% AUC | With all features enabled |

These gains are estimated based on similar approaches in the literature and may vary depending on the specific dataset and implementation details.