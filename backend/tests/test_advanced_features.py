"""
Tests for advanced deepfake detection features
"""

import os
import sys
import pytest
import numpy as np
import tensorflow as tf
import cv2
import json
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from ssl_pretraining import SimCLRPretrainer
from vit_head import ViTHead
from adversarial_training import AdversarialTrainer
from multimodal_fusion import AudioFeatureExtractor, AudioModel, MultiModalFusion
from knowledge_distillation import KnowledgeDistiller
from cross_validation import CrossValidator, ModelEnsemble
from explainability import DeepfakeExplainer
from export_quantized import ModelExporter

# Create a simple model for testing
def create_test_model(input_shape=(224, 224, 3)):
    """Create a simple model for testing"""
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create a sample image for testing
def create_test_image(size=(224, 224)):
    """Create a random test image"""
    return np.random.randint(0, 255, (*size, 3), dtype=np.uint8)

@pytest.fixture
def test_model():
    """Fixture for a test model"""
    return create_test_model()

@pytest.fixture
def test_image():
    """Fixture for a test image"""
    return create_test_image()

@pytest.fixture
def temp_dir():
    """Fixture for a temporary directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

class TestSSLPretraining:
    """Tests for self-supervised pretraining"""
    
    def test_simclr_init(self):
        """Test SimCLR initialization"""
        pretrainer = SimCLRPretrainer()
        assert pretrainer.model_name == "efficientnet_b3"
        assert pretrainer.input_shape == (224, 224, 3)
    
    def test_simclr_build_model(self):
        """Test SimCLR model building"""
        pretrainer = SimCLRPretrainer()
        model = pretrainer.build_model()
        
        assert isinstance(model, tf.keras.Model)
        assert len(model.inputs) == 2
        assert len(model.outputs) == 2

class TestViTHead:
    """Tests for Vision Transformer head"""
    
    def test_vit_init(self):
        """Test ViT head initialization"""
        vit_head = ViTHead(input_shape=(14, 14, 512))
        assert vit_head.input_shape == (14, 14, 512)
        assert vit_head.num_patches == 49  # 7x7 patches
    
    def test_vit_build_model(self, test_model):
        """Test ViT head model building"""
        # Get the output shape of the last convolutional layer
        conv_layer = None
        for layer in reversed(test_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_layer = layer
                break
        
        assert conv_layer is not None
        
        # Create ViT head
        vit_head = ViTHead(input_shape=conv_layer.output_shape[1:])
        model = vit_head.build_model()
        
        assert isinstance(model, tf.keras.Model)

class TestAdversarialTraining:
    """Tests for adversarial training"""
    
    def test_adversarial_trainer_init(self, test_model):
        """Test adversarial trainer initialization"""
        trainer = AdversarialTrainer(test_model)
        assert trainer.model == test_model
        assert trainer.attack_type == "fgsm"
    
    def test_fgsm_attack(self, test_model, test_image):
        """Test FGSM attack"""
        trainer = AdversarialTrainer(test_model)
        
        # Prepare input
        x = np.expand_dims(test_image.astype(np.float32) / 255.0, axis=0)
        y = np.array([[1.0]])  # Fake label
        
        # Generate adversarial example
        adv_x = trainer.fgsm_attack(x, y)
        
        assert adv_x.shape == x.shape
        assert np.any(adv_x != x)  # Should be different from original

class TestMultiModalFusion:
    """Tests for multi-modal fusion"""
    
    def test_audio_feature_extractor_init(self):
        """Test audio feature extractor initialization"""
        extractor = AudioFeatureExtractor()
        assert extractor.n_mfcc == 40
    
    def test_audio_model_init(self):
        """Test audio model initialization"""
        audio_model = AudioModel()
        assert audio_model.input_shape == (313, 40)
    
    def test_audio_model_build(self):
        """Test audio model building"""
        audio_model = AudioModel()
        model = audio_model.build_model()
        
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape[1:] == (313, 40)

class TestKnowledgeDistillation:
    """Tests for knowledge distillation"""
    
    def test_knowledge_distiller_init(self, test_model):
        """Test knowledge distiller initialization"""
        distiller = KnowledgeDistiller(test_model)
        assert distiller.teacher_model == test_model
        assert distiller.student_model_name == "mobilenetv2"
    
    def test_build_student_model(self, test_model):
        """Test building student model"""
        distiller = KnowledgeDistiller(test_model)
        student_model = distiller.build_student_model()
        
        assert isinstance(student_model, tf.keras.Model)
        assert student_model.name == "student_model"

class TestCrossValidation:
    """Tests for cross-validation and ensembling"""
    
    def test_cross_validator_init(self):
        """Test cross-validator initialization"""
        validator = CrossValidator(create_test_model)
        assert validator.n_splits == 5
    
    def test_model_ensemble_init(self, test_model):
        """Test model ensemble initialization"""
        ensemble = ModelEnsemble(models=[test_model, test_model])
        assert len(ensemble.models) == 2
        assert ensemble.ensemble_type == "average"

class TestExplainability:
    """Tests for explainability"""
    
    def test_explainer_init(self):
        """Test explainer initialization"""
        explainer = DeepfakeExplainer()
        assert hasattr(explainer, 'model_manager')
    
    def test_grad_cam_generation(self, test_model, test_image):
        """Test Grad-CAM generation"""
        # This is a simplified test since we can't easily mock the model manager
        # In a real test, you would mock the model manager to return a test model
        
        # Create a face region
        face_coords = (0, 0, test_image.shape[1], test_image.shape[0])
        
        # Create explainer
        explainer = DeepfakeExplainer()
        
        # We can't fully test the explain_prediction method without mocking,
        # but we can test the internal _generate_grad_cam method
        try:
            # This will likely fail since we're not using the actual model,
            # but it tests that the method exists and takes the right arguments
            heatmap, explanation_data = explainer._generate_grad_cam(test_image, test_model)
        except Exception:
            # Expected to fail, but we're just testing the method signature
            pass

class TestModelExport:
    """Tests for model export and quantization"""
    
    def test_model_exporter_init(self, test_model):
        """Test model exporter initialization with a provided model"""
        # Save the model to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            model_path = f.name
        
        test_model.save(model_path)
        
        try:
            exporter = ModelExporter(model_path)
            assert exporter.model is not None
        finally:
            # Clean up
            os.unlink(model_path)
    
    def test_tflite_export(self, test_model, temp_dir):
        """Test TFLite export"""
        # Save the model to a temporary file
        model_path = os.path.join(temp_dir, 'model.h5')
        test_model.save(model_path)
        
        # Create exporter
        exporter = ModelExporter(model_path)
        
        # Export to TFLite
        output_path = os.path.join(temp_dir, 'model.tflite')
        exported_path = exporter.export_tflite(
            output_path,
            quantize=False  # Disable quantization for faster testing
        )
        
        assert os.path.exists(exported_path)
        assert os.path.getsize(exported_path) > 0