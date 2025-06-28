"""
Advanced training script for deepfake detection models
Implements self-supervised pretraining, ViT head, adversarial training, and more
"""

import os
import logging
import argparse
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime

# Import modules
from ssl_pretraining import SimCLRPretrainer, MoCoPretrainer
from vit_head import ViTHead
from adversarial_training import AdversarialTrainer
from multimodal_fusion import MultiModalFusion, AudioFeatureExtractor, AudioModel
from knowledge_distillation import KnowledgeDistiller
from cross_validation import CrossValidator, ModelEnsemble
from model_trainer import DeepfakeModelTrainer
from backend.config import MODEL_DIR, DATASET_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Advanced training for deepfake detection")
    
    # General options
    parser.add_argument("--model-name", type=str, default="efficientnet_b3",
                       choices=["efficientnet_b0", "efficientnet_b3", "xception", "resnet50v2", "inception_v3"],
                       help="Base model architecture (default: efficientnet_b3)")
    parser.add_argument("--input-shape", type=int, nargs=3, default=[224, 224, 3],
                       help="Input shape for the model (default: 224 224 3)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training (default: 32)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs to train (default: 50)")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate for training (default: 1e-4)")
    parser.add_argument("--output-dir", type=str, default=os.path.join(MODEL_DIR, "advanced"),
                       help="Directory to save models and results")
    
    # Dataset options
    parser.add_argument("--train-dir", type=str, required=True,
                       help="Directory containing training data")
    parser.add_argument("--val-dir", type=str, required=True,
                       help="Directory containing validation data")
    parser.add_argument("--unlabeled-dir", type=str, default=None,
                       help="Directory containing unlabeled data for self-supervised learning")
    
    # Feature options
    parser.add_argument("--ssl-pretrain", action="store_true",
                       help="Use self-supervised pretraining")
    parser.add_argument("--ssl-method", type=str, default="simclr", choices=["simclr", "moco"],
                       help="Self-supervised learning method (default: simclr)")
    parser.add_argument("--vit-head", action="store_true",
                       help="Add Vision Transformer head")
    parser.add_argument("--adversarial-train", action="store_true",
                       help="Use adversarial training")
    parser.add_argument("--adversarial-method", type=str, default="fgsm", choices=["fgsm", "pgd"],
                       help="Adversarial training method (default: fgsm)")
    parser.add_argument("--multimodal", action="store_true",
                       help="Use multi-modal fusion with audio")
    parser.add_argument("--temporal", action="store_true",
                       help="Use temporal features for video")
    parser.add_argument("--cross-validation", action="store_true",
                       help="Use cross-validation")
    parser.add_argument("--n-folds", type=int, default=5,
                       help="Number of folds for cross-validation (default: 5)")
    parser.add_argument("--distill", action="store_true",
                       help="Use knowledge distillation")
    parser.add_argument("--student-model", type=str, default="mobilenetv2",
                       choices=["mobilenetv2", "efficientnet_b0"],
                       help="Student model for knowledge distillation (default: mobilenetv2)")
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Convert input shape to tuple
    input_shape = tuple(args.input_shape)
    
    # Step 1: Self-supervised pretraining (if enabled)
    if args.ssl_pretrain:
        logger.info(f"Starting self-supervised pretraining with {args.ssl_method}")
        
        # Check if unlabeled data directory is provided
        if args.unlabeled_dir is None:
            logger.warning("No unlabeled data directory provided for self-supervised learning. Using training data.")
            unlabeled_dir = args.train_dir
        else:
            unlabeled_dir = args.unlabeled_dir
        
        # Create SSL pretrainer
        if args.ssl_method == "simclr":
            ssl_pretrainer = SimCLRPretrainer(
                model_name=args.model_name,
                input_shape=input_shape,
                batch_size=args.batch_size
            )
        else:  # moco
            ssl_pretrainer = MoCoPretrainer(
                model_name=args.model_name,
                input_shape=input_shape,
                batch_size=args.batch_size
            )
        
        # Train SSL model
        ssl_save_dir = os.path.join(args.output_dir, "ssl_pretrained")
        ssl_pretrainer.train(
            data_dir=unlabeled_dir,
            epochs=min(100, args.epochs * 2),  # SSL typically needs more epochs
            save_dir=ssl_save_dir
        )
        
        # Get pretrained encoder
        pretrained_encoder = ssl_pretrainer.get_pretrained_encoder()
        
        logger.info(f"Self-supervised pretraining completed. Model saved to {ssl_save_dir}")
    else:
        pretrained_encoder = None
    
    # Step 2: Create model trainer
    model_trainer = DeepfakeModelTrainer(
        model_name=args.model_name,
        input_shape=input_shape,
        use_temporal=args.temporal,
        use_ensemble=args.cross_validation,
        use_hyperparameter_tuning=True  # Always use hyperparameter tuning for advanced training
    )
    
    # Step 3: Build base model
    if pretrained_encoder is not None:
        # Use pretrained encoder as base
        logger.info("Using pretrained encoder as base model")
        base_model = model_trainer.build_model_from_encoder(pretrained_encoder)
    else:
        # Build model from scratch
        logger.info("Building model from scratch")
        base_model = model_trainer.build_model()
    
    # Step 4: Add Vision Transformer head (if enabled)
    if args.vit_head:
        logger.info("Adding Vision Transformer head")
        
        # Get the output shape of the last convolutional layer
        conv_layer = None
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_layer = layer
                break
        
        if conv_layer is None:
            logger.warning("No convolutional layer found in the model. Cannot add ViT head.")
        else:
            # Create ViT head
            vit_head = ViTHead(input_shape=conv_layer.output_shape[1:])
            
            # Add ViT head to model
            model = vit_head.add_to_model(base_model)
            
            # Update base model
            base_model = model
    
    # Step 5: Set up multi-modal fusion (if enabled)
    if args.multimodal:
        logger.info("Setting up multi-modal fusion with audio")
        
        # Create multi-modal fusion
        fusion = MultiModalFusion(
            visual_model=base_model,
            fusion_type="attention"  # Use attention-based fusion
        )
        
        # Build fusion model
        model = fusion.build_model()
        
        # Update base model
        base_model = model
    
    # Step 6: Set up adversarial training (if enabled)
    if args.adversarial_train:
        logger.info(f"Setting up adversarial training with {args.adversarial_method}")
        
        # Create adversarial trainer
        adv_trainer = AdversarialTrainer(
            model=base_model,
            attack_type=args.adversarial_method,
            epsilon=0.01 if args.adversarial_method == "fgsm" else 0.005
        )
    else:
        adv_trainer = None
    
    # Step 7: Set up cross-validation (if enabled)
    if args.cross_validation:
        logger.info(f"Setting up {args.n_folds}-fold cross-validation")
        
        # Create model builder function
        def model_builder():
            if pretrained_encoder is not None:
                # Use pretrained encoder as base
                model = model_trainer.build_model_from_encoder(pretrained_encoder)
            else:
                # Build model from scratch
                model = model_trainer.build_model()
            
            # Add ViT head if enabled
            if args.vit_head:
                conv_layer = None
                for layer in reversed(model.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        conv_layer = layer
                        break
                
                if conv_layer is not None:
                    vit_head = ViTHead(input_shape=conv_layer.output_shape[1:])
                    model = vit_head.add_to_model(model)
            
            return model
        
        # Create cross-validator
        validator = CrossValidator(
            model_builder=model_builder,
            n_splits=args.n_folds
        )
        
        # Load data
        x_train, y_train = model_trainer.load_data(args.train_dir)
        
        # Perform cross-validation
        cv_save_dir = os.path.join(args.output_dir, "cross_validation")
        cv_results = validator.cross_validate(
            x=x_train,
            y=y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_dir=cv_save_dir
        )
        
        # Create ensemble
        ensemble_save_dir = os.path.join(args.output_dir, "ensemble")
        ensemble_config = validator.create_ensemble(save_path=ensemble_save_dir)
        
        logger.info(f"Cross-validation completed. Results saved to {cv_save_dir}")
        logger.info(f"Ensemble created. Config saved to {ensemble_save_dir}")
        
        # Use the first model from cross-validation for distillation
        base_model = validator.models[0] if validator.models else base_model
    
    # Step 8: Knowledge distillation (if enabled)
    if args.distill:
        logger.info(f"Setting up knowledge distillation with {args.student_model}")
        
        # Create knowledge distiller
        distiller = KnowledgeDistiller(
            teacher_model=base_model,
            student_model_name=args.student_model,
            input_shape=input_shape
        )
        
        # Build student model
        student_model = distiller.build_student_model()
        
        # Load data
        train_dataset = tf.data.Dataset.from_tensor_slices(model_trainer.load_data(args.train_dir))
        train_dataset = train_dataset.batch(args.batch_size)
        
        val_dataset = tf.data.Dataset.from_tensor_slices(model_trainer.load_data(args.val_dir))
        val_dataset = val_dataset.batch(args.batch_size)
        
        # Train student model
        distill_save_dir = os.path.join(args.output_dir, "distilled")
        distiller.train(
            train_dataset=train_dataset,
            validation_dataset=val_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_dir=distill_save_dir
        )
        
        # Benchmark models
        benchmark_results = distiller.benchmark_inference_speed(batch_size=1, num_iterations=100)
        
        # Save benchmark results
        benchmark_path = os.path.join(distill_save_dir, "benchmark_results.json")
        with open(benchmark_path, "w") as f:
            json.dump(benchmark_results, f, indent=2)
        
        logger.info(f"Knowledge distillation completed. Model saved to {distill_save_dir}")
        logger.info(f"Benchmark results: {benchmark_results}")
    
    # Step 9: Regular training (if not using cross-validation)
    if not args.cross_validation:
        logger.info("Starting regular training")
        
        # Prepare data generators
        train_gen, val_gen = model_trainer.prepare_data_generators(
            train_dir=args.train_dir,
            validation_dir=args.val_dir,
            batch_size=args.batch_size
        )
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(args.output_dir, "best_model.h5"),
                save_best_only=True,
                monitor="val_loss"
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(args.output_dir, "logs"),
                histogram_freq=1
            )
        ]
        
        # Train model
        if args.adversarial_train and adv_trainer is not None:
            # Use adversarial training
            history = adv_trainer.fit(
                train_dataset=train_gen,
                validation_dataset=val_gen,
                epochs=args.epochs,
                callbacks=callbacks
            )
        else:
            # Use regular training
            history = base_model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=args.epochs,
                callbacks=callbacks
            )
        
        # Save training history
        history_path = os.path.join(args.output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump({k: [float(val) for val in v] for k, v in history.history.items()}, f)
        
        # Save final model
        final_model_path = os.path.join(args.output_dir, "final_model.h5")
        base_model.save(final_model_path)
        
        logger.info(f"Training completed. Final model saved to {final_model_path}")
    
    logger.info("Advanced training completed successfully")


if __name__ == "__main__":
    main()