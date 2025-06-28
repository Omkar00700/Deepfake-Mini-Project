"""
SHAP and Captum integration for explainable deepfake detection
"""

import tensorflow as tf
import numpy as np
import logging
import os
import cv2
import shap
import json
import base64
from typing import Tuple, List, Dict, Optional, Union, Callable, Any
from pathlib import Path
import matplotlib.pyplot as plt
import io
import time

# Configure logging
logger = logging.getLogger(__name__)

try:
    import torch
    import captum
    from captum.attr import IntegratedGradients, GradientShap, DeepLift, Occlusion, NoiseTunnel
    from captum.attr import visualization as viz
    CAPTUM_AVAILABLE = True
except ImportError:
    logger.warning("Captum not available. Some explainability features will be disabled.")
    CAPTUM_AVAILABLE = False

class DeepfakeExplainer:
    """
    Explainer for deepfake detection models
    Combines SHAP and Captum for comprehensive explanations
    """
    
    def __init__(self, 
                 model: tf.keras.Model,
                 model_type: str = "tensorflow",
                 preprocess_fn: Optional[Callable] = None,
                 class_names: Optional[List[str]] = None,
                 output_dir: str = "explanations"):
        """
        Initialize deepfake explainer
        
        Args:
            model: Model to explain
            model_type: Model type ('tensorflow' or 'pytorch')
            preprocess_fn: Preprocessing function
            class_names: Class names
            output_dir: Directory to save explanations
        """
        self.model = model
        self.model_type = model_type
        self.preprocess_fn = preprocess_fn
        self.class_names = class_names or ["Real", "Fake"]
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize SHAP explainer
        self.shap_explainer = None
        
        # Initialize Captum explainers
        self.captum_explainers = {}
        
        if CAPTUM_AVAILABLE and model_type == "pytorch":
            self._init_captum_explainers()
        
        logger.info(f"Initialized deepfake explainer for {model_type} model")
    
    def _init_captum_explainers(self) -> None:
        """
        Initialize Captum explainers
        """
        if not CAPTUM_AVAILABLE:
            logger.warning("Captum not available. Skipping initialization.")
            return
        
        if self.model_type != "pytorch":
            logger.warning("Captum only works with PyTorch models. Skipping initialization.")
            return
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Initialize explainers
        self.captum_explainers = {
            "integrated_gradients": IntegratedGradients(self.model),
            "gradient_shap": GradientShap(self.model),
            "deep_lift": DeepLift(self.model),
            "occlusion": Occlusion(self.model)
        }
        
        logger.info("Initialized Captum explainers")
    
    def _init_shap_explainer(self, background_data: Optional[np.ndarray] = None) -> None:
        """
        Initialize SHAP explainer
        
        Args:
            background_data: Background data for SHAP
        """
        if self.shap_explainer is not None:
            return
        
        if background_data is None:
            # Create simple background (black image)
            if hasattr(self.model, "input_shape"):
                input_shape = self.model.input_shape[1:]  # Remove batch dimension
                background_data = np.zeros((1, *input_shape))
            else:
                logger.warning("Model has no input_shape attribute and no background data provided. "
                              "Using default background.")
                background_data = np.zeros((1, 224, 224, 3))
        
        # Initialize SHAP explainer based on model type
        if self.model_type == "tensorflow":
            self.shap_explainer = shap.DeepExplainer(self.model, background_data)
        elif self.model_type == "pytorch":
            self.shap_explainer = shap.DeepExplainer(self.model, torch.tensor(background_data))
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        logger.info(f"Initialized SHAP explainer with background data shape {background_data.shape}")
    
    def explain_with_shap(self, 
                         image: np.ndarray,
                         target_class: Optional[int] = None,
                         background_data: Optional[np.ndarray] = None,
                         n_samples: int = 100) -> Dict[str, Any]:
        """
        Explain prediction with SHAP
        
        Args:
            image: Input image
            target_class: Target class index (if None, use predicted class)
            background_data: Background data for SHAP
            n_samples: Number of samples for SHAP
            
        Returns:
            Dictionary with explanation data
        """
        # Initialize SHAP explainer if needed
        if self.shap_explainer is None:
            self._init_shap_explainer(background_data)
        
        # Preprocess image if needed
        if self.preprocess_fn is not None:
            processed_image = self.preprocess_fn(image)
        else:
            processed_image = image.copy()
        
        # Add batch dimension if needed
        if len(processed_image.shape) == 3:
            processed_image = np.expand_dims(processed_image, axis=0)
        
        # Get model prediction
        if self.model_type == "tensorflow":
            prediction = self.model.predict(processed_image)[0]
        else:  # pytorch
            with torch.no_grad():
                prediction = self.model(torch.tensor(processed_image)).numpy()[0]
        
        # Get predicted class if target class is not specified
        if target_class is None:
            target_class = np.argmax(prediction)
        
        # Calculate SHAP values
        try:
            if self.model_type == "tensorflow":
                shap_values = self.shap_explainer.shap_values(processed_image, nsamples=n_samples)
            else:  # pytorch
                shap_values = self.shap_explainer.shap_values(
                    torch.tensor(processed_image), nsamples=n_samples
                )
                
                # Convert to numpy if needed
                if isinstance(shap_values, list):
                    shap_values = [sv.numpy() if isinstance(sv, torch.Tensor) else sv for sv in shap_values]
            
            # Get SHAP values for target class
            if isinstance(shap_values, list):
                target_shap_values = shap_values[target_class][0]
            else:
                target_shap_values = shap_values[0, target_class]
            
            # Generate SHAP visualization
            plt.figure(figsize=(10, 6))
            
            # For RGB images, use shap.image_plot
            if len(target_shap_values.shape) == 3 and target_shap_values.shape[2] == 3:
                shap.image_plot(target_shap_values, processed_image[0])
            else:
                # For grayscale or other formats, use custom visualization
                plt.imshow(np.sum(np.abs(target_shap_values), axis=2), cmap='hot')
                plt.colorbar(label='SHAP value magnitude')
                plt.title(f"SHAP values for class {self.class_names[target_class]}")
            
            # Save visualization
            vis_path = os.path.join(self.output_dir, f"shap_vis_{int(time.time())}.png")
            plt.savefig(vis_path)
            plt.close()
            
            # Convert SHAP values to heatmap
            heatmap = self._shap_values_to_heatmap(target_shap_values)
            
            # Save heatmap
            heatmap_path = os.path.join(self.output_dir, f"shap_heatmap_{int(time.time())}.png")
            cv2.imwrite(heatmap_path, heatmap)
            
            # Create overlay
            overlay = self._create_overlay(image, heatmap)
            
            # Save overlay
            overlay_path = os.path.join(self.output_dir, f"shap_overlay_{int(time.time())}.png")
            cv2.imwrite(overlay_path, overlay)
            
            # Convert to base64 for web display
            overlay_base64 = self._image_to_base64(overlay)
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(target_shap_values)
            
            # Create explanation data
            explanation = {
                "method": "shap",
                "target_class": int(target_class),
                "target_class_name": self.class_names[target_class],
                "prediction": prediction.tolist(),
                "visualization_path": vis_path,
                "heatmap_path": heatmap_path,
                "overlay_path": overlay_path,
                "overlay_base64": overlay_base64,
                "feature_importance": feature_importance,
                "success": True
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error in SHAP explanation: {str(e)}", exc_info=True)
            
            return {
                "method": "shap",
                "error": str(e),
                "success": False
            }
    
    def explain_with_captum(self, 
                          image: np.ndarray,
                          method: str = "integrated_gradients",
                          target_class: Optional[int] = None,
                          n_steps: int = 50) -> Dict[str, Any]:
        """
        Explain prediction with Captum
        
        Args:
            image: Input image
            method: Captum method to use
            target_class: Target class index (if None, use predicted class)
            n_steps: Number of steps for methods like IntegratedGradients
            
        Returns:
            Dictionary with explanation data
        """
        if not CAPTUM_AVAILABLE:
            return {
                "method": method,
                "error": "Captum not available",
                "success": False
            }
        
        if self.model_type != "pytorch":
            return {
                "method": method,
                "error": "Captum only works with PyTorch models",
                "success": False
            }
        
        # Check if method is available
        if method not in self.captum_explainers:
            return {
                "method": method,
                "error": f"Method {method} not available",
                "success": False
            }
        
        # Preprocess image if needed
        if self.preprocess_fn is not None:
            processed_image = self.preprocess_fn(image)
        else:
            processed_image = image.copy()
        
        # Convert to PyTorch tensor
        input_tensor = torch.tensor(processed_image, dtype=torch.float32)
        
        # Add batch dimension if needed
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Set requires_grad
        input_tensor.requires_grad = True
        
        # Get model prediction
        with torch.no_grad():
            prediction = self.model(input_tensor).detach().numpy()[0]
        
        # Get predicted class if target class is not specified
        if target_class is None:
            target_class = np.argmax(prediction)
        
        try:
            # Get explainer
            explainer = self.captum_explainers[method]
            
            # Calculate attributions
            if method == "integrated_gradients":
                attributions = explainer.attribute(
                    input_tensor,
                    target=target_class,
                    n_steps=n_steps
                )
            elif method == "gradient_shap":
                # Create baseline
                baseline = torch.zeros_like(input_tensor)
                
                attributions = explainer.attribute(
                    input_tensor,
                    baselines=baseline,
                    target=target_class,
                    n_samples=n_steps
                )
            elif method == "deep_lift":
                # Create baseline
                baseline = torch.zeros_like(input_tensor)
                
                attributions = explainer.attribute(
                    input_tensor,
                    baselines=baseline,
                    target=target_class
                )
            elif method == "occlusion":
                attributions = explainer.attribute(
                    input_tensor,
                    target=target_class,
                    sliding_window_shapes=(3, 3, 3),
                    strides=(1, 1, 1)
                )
            else:
                return {
                    "method": method,
                    "error": f"Method {method} not implemented",
                    "success": False
                }
            
            # Convert attributions to numpy
            attributions = attributions.detach().numpy()[0]
            
            # Generate visualization
            plt.figure(figsize=(10, 6))
            
            # For RGB images
            if len(attributions.shape) == 3 and attributions.shape[0] == 3:
                # Transpose to HWC format
                attributions = np.transpose(attributions, (1, 2, 0))
                processed_image_np = np.transpose(processed_image, (1, 2, 0))
                
                # Use Captum visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                
                viz.visualize_image_attr(
                    np.abs(attributions),
                    processed_image_np,
                    method="heat_map",
                    sign="absolute_value",
                    show_colorbar=True,
                    title=f"{method} attributions for class {self.class_names[target_class]}",
                    plt_fig_axis=(fig, ax)
                )
            else:
                # For grayscale or other formats, use custom visualization
                plt.imshow(np.sum(np.abs(attributions), axis=0), cmap='hot')
                plt.colorbar(label='Attribution magnitude')
                plt.title(f"{method} attributions for class {self.class_names[target_class]}")
            
            # Save visualization
            vis_path = os.path.join(self.output_dir, f"{method}_vis_{int(time.time())}.png")
            plt.savefig(vis_path)
            plt.close()
            
            # Convert attributions to heatmap
            if len(attributions.shape) == 3 and attributions.shape[0] == 3:
                # Already transposed above
                heatmap = self._attributions_to_heatmap(attributions)
            else:
                heatmap = self._attributions_to_heatmap(np.transpose(attributions, (1, 2, 0)))
            
            # Save heatmap
            heatmap_path = os.path.join(self.output_dir, f"{method}_heatmap_{int(time.time())}.png")
            cv2.imwrite(heatmap_path, heatmap)
            
            # Create overlay
            overlay = self._create_overlay(image, heatmap)
            
            # Save overlay
            overlay_path = os.path.join(self.output_dir, f"{method}_overlay_{int(time.time())}.png")
            cv2.imwrite(overlay_path, overlay)
            
            # Convert to base64 for web display
            overlay_base64 = self._image_to_base64(overlay)
            
            # Calculate feature importance
            if len(attributions.shape) == 3 and attributions.shape[0] == 3:
                # Already transposed above
                feature_importance = self._calculate_feature_importance(attributions)
            else:
                feature_importance = self._calculate_feature_importance(
                    np.transpose(attributions, (1, 2, 0))
                )
            
            # Create explanation data
            explanation = {
                "method": method,
                "target_class": int(target_class),
                "target_class_name": self.class_names[target_class],
                "prediction": prediction.tolist(),
                "visualization_path": vis_path,
                "heatmap_path": heatmap_path,
                "overlay_path": overlay_path,
                "overlay_base64": overlay_base64,
                "feature_importance": feature_importance,
                "success": True
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error in Captum explanation: {str(e)}", exc_info=True)
            
            return {
                "method": method,
                "error": str(e),
                "success": False
            }
    
    def _shap_values_to_heatmap(self, shap_values: np.ndarray) -> np.ndarray:
        """
        Convert SHAP values to heatmap
        
        Args:
            shap_values: SHAP values
            
        Returns:
            Heatmap image
        """
        # Calculate absolute values
        abs_values = np.abs(shap_values)
        
        # Sum across channels if needed
        if len(abs_values.shape) == 3:
            abs_values = np.sum(abs_values, axis=2)
        
        # Normalize to [0, 255]
        abs_values = abs_values - np.min(abs_values)
        max_val = np.max(abs_values)
        
        if max_val > 0:
            abs_values = abs_values / max_val * 255
        
        # Convert to uint8
        heatmap = abs_values.astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return heatmap
    
    def _attributions_to_heatmap(self, attributions: np.ndarray) -> np.ndarray:
        """
        Convert Captum attributions to heatmap
        
        Args:
            attributions: Captum attributions
            
        Returns:
            Heatmap image
        """
        # Calculate absolute values
        abs_values = np.abs(attributions)
        
        # Sum across channels if needed
        if len(abs_values.shape) == 3:
            abs_values = np.sum(abs_values, axis=2)
        
        # Normalize to [0, 255]
        abs_values = abs_values - np.min(abs_values)
        max_val = np.max(abs_values)
        
        if max_val > 0:
            abs_values = abs_values / max_val * 255
        
        # Convert to uint8
        heatmap = abs_values.astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return heatmap
    
    def _create_overlay(self, image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """
        Create overlay of image and heatmap
        
        Args:
            image: Original image
            heatmap: Heatmap image
            
        Returns:
            Overlay image
        """
        # Resize heatmap to match image size if needed
        if image.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert image to BGR if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if image is RGB (not BGR)
            if self.model_type == "tensorflow":
                # TensorFlow models typically use RGB
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                # PyTorch models typically use RGB as well
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            # Convert grayscale to BGR if needed
            if len(image.shape) == 2:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image_bgr = image.copy()
        
        # Create overlay
        overlay = cv2.addWeighted(image_bgr, 0.7, heatmap, 0.3, 0)
        
        return overlay
    
    def _calculate_feature_importance(self, values: np.ndarray) -> Dict[str, float]:
        """
        Calculate feature importance from attribution values
        
        Args:
            values: Attribution values
            
        Returns:
            Dictionary with feature importance
        """
        # Define regions of interest
        h, w = values.shape[:2]
        
        regions = {
            "eyes": (0, h // 3, 0, w),
            "nose": (h // 3, 2 * h // 3, w // 4, 3 * w // 4),
            "mouth": (2 * h // 3, h, 0, w),
            "forehead": (0, h // 4, 0, w),
            "cheeks": (h // 3, 2 * h // 3, 0, w // 4),
            "chin": (2 * h // 3, h, w // 4, 3 * w // 4)
        }
        
        # Calculate importance for each region
        importance = {}
        
        for region_name, (y1, y2, x1, x2) in regions.items():
            # Extract region
            region_values = values[y1:y2, x1:x2]
            
            # Calculate importance (sum of absolute values)
            if len(region_values.shape) == 3:
                importance[region_name] = float(np.sum(np.abs(region_values)))
            else:
                importance[region_name] = float(np.sum(np.abs(region_values)))
        
        # Normalize importance
        total_importance = sum(importance.values())
        
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """
        Convert image to base64 string
        
        Args:
            image: Image to convert
            
        Returns:
            Base64 string
        """
        # Encode image as PNG
        _, buffer = cv2.imencode(".png", image)
        
        # Convert to base64
        base64_str = base64.b64encode(buffer).decode("utf-8")
        
        return f"data:image/png;base64,{base64_str}"
    
    def explain(self, 
               image: np.ndarray,
               method: str = "shap",
               target_class: Optional[int] = None,
               **kwargs) -> Dict[str, Any]:
        """
        Explain prediction
        
        Args:
            image: Input image
            method: Explanation method
            target_class: Target class index
            **kwargs: Additional arguments for specific methods
            
        Returns:
            Dictionary with explanation data
        """
        if method == "shap":
            return self.explain_with_shap(
                image,
                target_class=target_class,
                background_data=kwargs.get("background_data"),
                n_samples=kwargs.get("n_samples", 100)
            )
        elif method in ["integrated_gradients", "gradient_shap", "deep_lift", "occlusion"]:
            return self.explain_with_captum(
                image,
                method=method,
                target_class=target_class,
                n_steps=kwargs.get("n_steps", 50)
            )
        else:
            return {
                "method": method,
                "error": f"Unsupported method: {method}",
                "success": False
            }


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple model for testing
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax")
    ])
    
    # Create explainer
    explainer = DeepfakeExplainer(
        model=model,
        model_type="tensorflow",
        output_dir="explanations"
    )
    
    # Create a random image
    image = np.random.random((224, 224, 3))
    
    # Explain with SHAP
    explanation = explainer.explain(image, method="shap")
    
    print(f"SHAP explanation success: {explanation['success']}")
    if explanation['success']:
        print(f"Visualization saved to: {explanation['visualization_path']}")
        print(f"Feature importance: {explanation['feature_importance']}")
    else:
        print(f"Error: {explanation.get('error', 'Unknown error')}")