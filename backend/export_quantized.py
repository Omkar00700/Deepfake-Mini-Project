"""
Export Quantized Model for Edge Deployment
Converts TensorFlow models to TensorFlow Lite or ONNX format with quantization
"""

import os
import logging
import tensorflow as tf
import numpy as np
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import onnx
import tf2onnx
from model_manager import ModelManager
from backend.config import MODEL_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelExporter:
    """
    Export and quantize models for edge deployment
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the model exporter
        
        Args:
            model_path: Path to the model to export (optional)
        """
        self.model_manager = ModelManager()
        self.model_path = model_path
        self.model = None
        
        # Load model
        if model_path:
            logger.info(f"Loading model from {model_path}")
            self.model = tf.keras.models.load_model(model_path)
        else:
            logger.info("Using default model from model manager")
            self.model = self.model_manager.get_model().model
        
        logger.info(f"Model loaded with input shape: {self.model.input_shape}")
    
    def export_tflite(self, 
                     output_path: str, 
                     quantize: bool = True,
                     quantization_type: str = "int8",
                     representative_dataset: Optional[tf.data.Dataset] = None) -> str:
        """
        Export model to TensorFlow Lite format
        
        Args:
            output_path: Path to save the exported model
            quantize: Whether to quantize the model
            quantization_type: Type of quantization ('int8', 'float16', or 'dynamic')
            representative_dataset: Dataset for quantization calibration
            
        Returns:
            Path to the exported model
        """
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Set optimization flags
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Configure quantization
        if quantize:
            if quantization_type == "int8":
                logger.info("Using INT8 quantization")
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                
                # Representative dataset is required for full integer quantization
                if representative_dataset is None:
                    logger.warning("No representative dataset provided for INT8 quantization. "
                                  "Creating a synthetic dataset.")
                    representative_dataset = self._create_synthetic_dataset()
                
                def representative_dataset_gen():
                    for data in representative_dataset.take(100):
                        yield [data]
                
                converter.representative_dataset = representative_dataset_gen
                
            elif quantization_type == "float16":
                logger.info("Using float16 quantization")
                converter.target_spec.supported_types = [tf.float16]
                
            elif quantization_type == "dynamic":
                logger.info("Using dynamic range quantization")
                # Dynamic range quantization is enabled by default with optimizations
                
            else:
                logger.warning(f"Unknown quantization type: {quantization_type}. "
                              "Using default quantization.")
        
        # Convert model
        logger.info("Converting model to TFLite format...")
        tflite_model = converter.convert()
        
        # Save model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Get model size
        model_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
        logger.info(f"Model exported to {output_path} (Size: {model_size:.2f} MB)")
        
        return output_path
    
    def export_onnx(self, output_path: str, opset: int = 12) -> str:
        """
        Export model to ONNX format
        
        Args:
            output_path: Path to save the exported model
            opset: ONNX opset version
            
        Returns:
            Path to the exported model
        """
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get model input and output specs
        input_signature = [tf.TensorSpec(self.model.input_shape, tf.float32)]
        
        # Convert model to ONNX
        logger.info(f"Converting model to ONNX format (opset {opset})...")
        onnx_model, _ = tf2onnx.convert.from_keras(self.model, input_signature, opset=opset)
        
        # Save model
        onnx.save(onnx_model, output_path)
        
        # Get model size
        model_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
        logger.info(f"Model exported to {output_path} (Size: {model_size:.2f} MB)")
        
        return output_path
    
    def _create_synthetic_dataset(self) -> tf.data.Dataset:
        """
        Create a synthetic dataset for quantization calibration
        
        Returns:
            TensorFlow dataset with synthetic data
        """
        # Get input shape
        input_shape = self.model.input_shape
        
        # Remove batch dimension if present
        if input_shape[0] is None:
            input_shape = input_shape[1:]
        
        # Create random data
        synthetic_data = np.random.random((100, *input_shape)).astype(np.float32)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(synthetic_data)
        dataset = dataset.batch(1)
        
        return dataset
    
    def benchmark_model(self, 
                       model_path: str, 
                       model_format: str = "tflite",
                       num_runs: int = 100) -> Dict[str, Any]:
        """
        Benchmark a model's inference performance
        
        Args:
            model_path: Path to the model
            model_format: Format of the model ('tflite' or 'onnx')
            num_runs: Number of inference runs
            
        Returns:
            Benchmark results
        """
        # Load model
        if model_format == "tflite":
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Create input data
            input_shape = input_details[0]['shape']
            input_data = np.random.random(input_shape).astype(np.float32)
            
            # Warm up
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_runs):
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
            end_time = time.time()
            
        elif model_format == "onnx":
            # For ONNX, we'll use ONNX Runtime
            try:
                import onnxruntime as ort
                
                # Create ONNX Runtime session
                session = ort.InferenceSession(model_path)
                
                # Get input name
                input_name = session.get_inputs()[0].name
                
                # Create input data
                input_shape = session.get_inputs()[0].shape
                if input_shape[0] == 'batch' or input_shape[0] is None:
                    input_shape = (1, *input_shape[1:])
                input_data = np.random.random(input_shape).astype(np.float32)
                
                # Warm up
                session.run(None, {input_name: input_data})
                
                # Benchmark
                start_time = time.time()
                for _ in range(num_runs):
                    output = session.run(None, {input_name: input_data})
                end_time = time.time()
                
            except ImportError:
                logger.error("ONNX Runtime not installed. Please install with: pip install onnxruntime")
                return {"error": "ONNX Runtime not installed"}
        else:
            logger.error(f"Unsupported model format: {model_format}")
            return {"error": f"Unsupported model format: {model_format}"}
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        fps = num_runs / total_time
        
        # Get model size
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
        
        # Create results
        results = {
            "model_path": model_path,
            "model_format": model_format,
            "model_size_mb": float(model_size),
            "num_runs": num_runs,
            "total_time_sec": float(total_time),
            "avg_inference_time_ms": float(avg_time * 1000),
            "fps": float(fps),
            "timestamp": time.time()
        }
        
        logger.info(f"Benchmark results: {results}")
        
        return results
    
    def compare_models(self, 
                      original_model: tf.keras.Model,
                      quantized_model_path: str,
                      quantized_model_format: str = "tflite",
                      test_images: Optional[np.ndarray] = None,
                      num_test_images: int = 10) -> Dict[str, Any]:
        """
        Compare original and quantized models for accuracy
        
        Args:
            original_model: Original TensorFlow model
            quantized_model_path: Path to the quantized model
            quantized_model_format: Format of the quantized model ('tflite' or 'onnx')
            test_images: Test images for comparison
            num_test_images: Number of test images to generate if not provided
            
        Returns:
            Comparison results
        """
        # Create test images if not provided
        if test_images is None:
            input_shape = original_model.input_shape
            if input_shape[0] is None:
                input_shape = (1, *input_shape[1:])
            else:
                input_shape = (num_test_images, *input_shape[1:])
            
            test_images = np.random.random(input_shape).astype(np.float32)
        
        # Get predictions from original model
        original_predictions = original_model.predict(test_images)
        
        # Get predictions from quantized model
        if quantized_model_format == "tflite":
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=quantized_model_path)
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Make predictions
            quantized_predictions = []
            for i in range(len(test_images)):
                interpreter.set_tensor(input_details[0]['index'], np.expand_dims(test_images[i], axis=0))
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                quantized_predictions.append(output[0])
            
            quantized_predictions = np.array(quantized_predictions)
            
        elif quantized_model_format == "onnx":
            # For ONNX, we'll use ONNX Runtime
            try:
                import onnxruntime as ort
                
                # Create ONNX Runtime session
                session = ort.InferenceSession(quantized_model_path)
                
                # Get input name
                input_name = session.get_inputs()[0].name
                
                # Make predictions
                quantized_predictions = []
                for i in range(len(test_images)):
                    output = session.run(None, {input_name: np.expand_dims(test_images[i], axis=0)})
                    quantized_predictions.append(output[0][0])
                
                quantized_predictions = np.array(quantized_predictions)
                
            except ImportError:
                logger.error("ONNX Runtime not installed. Please install with: pip install onnxruntime")
                return {"error": "ONNX Runtime not installed"}
        else:
            logger.error(f"Unsupported model format: {quantized_model_format}")
            return {"error": f"Unsupported model format: {quantized_model_format}"}
        
        # Calculate metrics
        mae = np.mean(np.abs(original_predictions - quantized_predictions))
        mse = np.mean(np.square(original_predictions - quantized_predictions))
        
        # Calculate accuracy drop
        original_classes = (original_predictions > 0.5).astype(int)
        quantized_classes = (quantized_predictions > 0.5).astype(int)
        accuracy_match = np.mean(original_classes == quantized_classes)
        
        # Create results
        results = {
            "num_test_images": len(test_images),
            "mean_absolute_error": float(mae),
            "mean_squared_error": float(mse),
            "accuracy_match": float(accuracy_match),
            "timestamp": time.time()
        }
        
        logger.info(f"Model comparison results: {results}")
        
        return results


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description="Export and quantize models for edge deployment")
    
    # Input model options
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to the model to export (default: use model manager)")
    
    # Export options
    parser.add_argument("--format", type=str, choices=["tflite", "onnx"], default="tflite",
                       help="Format to export the model to (default: tflite)")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save the exported model (default: auto-generate)")
    
    # Quantization options
    parser.add_argument("--quantize", action="store_true", default=True,
                       help="Quantize the model (default: True)")
    parser.add_argument("--no-quantize", action="store_false", dest="quantize",
                       help="Don't quantize the model")
    parser.add_argument("--quantization-type", type=str, 
                       choices=["int8", "float16", "dynamic"], default="int8",
                       help="Type of quantization to use (default: int8)")
    
    # Benchmark options
    parser.add_argument("--benchmark", action="store_true", default=True,
                       help="Benchmark the exported model (default: True)")
    parser.add_argument("--no-benchmark", action="store_false", dest="benchmark",
                       help="Don't benchmark the exported model")
    parser.add_argument("--num-runs", type=int, default=100,
                       help="Number of inference runs for benchmarking (default: 100)")
    
    # Comparison options
    parser.add_argument("--compare", action="store_true", default=True,
                       help="Compare original and quantized models (default: True)")
    parser.add_argument("--no-compare", action="store_false", dest="compare",
                       help="Don't compare original and quantized models")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create exporter
    exporter = ModelExporter(args.model_path)
    
    # Generate output path if not provided
    if args.output is None:
        timestamp = int(time.time())
        if args.format == "tflite":
            args.output = os.path.join(MODEL_DIR, "exported", f"model_{timestamp}.tflite")
        else:
            args.output = os.path.join(MODEL_DIR, "exported", f"model_{timestamp}.onnx")
    
    # Export model
    if args.format == "tflite":
        exported_path = exporter.export_tflite(
            args.output,
            quantize=args.quantize,
            quantization_type=args.quantization_type
        )
    else:
        exported_path = exporter.export_onnx(args.output)
    
    # Benchmark model
    if args.benchmark:
        benchmark_results = exporter.benchmark_model(
            exported_path,
            model_format=args.format,
            num_runs=args.num_runs
        )
        
        # Save benchmark results
        benchmark_path = os.path.splitext(exported_path)[0] + "_benchmark.json"
        with open(benchmark_path, "w") as f:
            json.dump(benchmark_results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {benchmark_path}")
    
    # Compare models
    if args.compare:
        comparison_results = exporter.compare_models(
            exporter.model,
            exported_path,
            quantized_model_format=args.format
        )
        
        # Save comparison results
        comparison_path = os.path.splitext(exported_path)[0] + "_comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(comparison_results, f, indent=2)
        
        logger.info(f"Comparison results saved to {comparison_path}")
    
    logger.info("Export completed successfully")


if __name__ == "__main__":
    main()