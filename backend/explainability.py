
import numpy as np
import cv2
import tensorflow as tf
import logging
from typing import Dict, Any, Tuple, List, Optional
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os
import time
import shap
from model_manager import ModelManager
from preprocessing import preprocess_face
from backend.config import VISUALIZATION_OUTPUT_DIR

# Configure logging
logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)

class DeepfakeExplainer:
    """
    Provides explainability features for deepfake detection models
    using techniques like Grad-CAM and integrated gradients.
    """
    
    def __init__(self):
        self.model_manager = ModelManager()
    
    def explain_prediction(self, image, face_coords, method="grad_cam") -> Dict[str, Any]:
        """
        Generate an explanation for a deepfake prediction
        
        Args:
            image: Original image
            face_coords: Face coordinates (x, y, w, h)
            method: Explanation method to use (grad_cam, integrated_gradients, shap)
            
        Returns:
            Dictionary containing explanation data including heatmap image
        """
        try:
            # Get current model
            model = self.model_manager.get_model()
            model_name = model.model_name
            
            # Extract and preprocess face
            x, y, w, h = face_coords
            processed_face = preprocess_face(image, face_coords, model.input_shape[:2])
            
            if processed_face is None:
                return {
                    "success": False,
                    "message": "Failed to preprocess face for explanation",
                    "method": method
                }
            
            # Select explanation method
            if method == "grad_cam":
                heatmap, explanation_data = self._generate_grad_cam(processed_face, model)
            elif method == "integrated_gradients":
                heatmap, explanation_data = self._generate_integrated_gradients(processed_face, model)
            elif method == "shap":
                heatmap, explanation_data = self._generate_shap(processed_face, model)
            else:
                return {
                    "success": False,
                    "message": f"Unsupported explanation method: {method}",
                    "method": method
                }
            
            # Apply heatmap to original face for visualization
            superimposed_img = self._apply_heatmap_to_image(processed_face, heatmap)
            
            # Get heatmap as base64 string
            heatmap_base64 = self._convert_image_to_base64(superimposed_img)
            
            # Get explanation results
            return {
                "success": True,
                "method": method,
                "model": model_name,
                "face_region": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                },
                "visualization": {
                    "heatmap_base64": heatmap_base64,
                    "format": "png"
                },
                "explanation_data": explanation_data
            }
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to generate explanation: {str(e)}",
                "method": method
            }
    
    def _generate_grad_cam(self, preprocessed_image, model) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate Grad-CAM visualization for deepfake detection
        
        Args:
            preprocessed_image: Preprocessed face image
            model: Model for inference
            
        Returns:
            Tuple of (heatmap, explanation_data)
        """
        try:
            # Get the model's last convolutional layer
            # Note: This is specific to each model architecture
            last_conv_layer = None
            if model.model_name == "efficientnet":
                last_conv_layer = model.model.get_layer("top_conv")
            elif model.model_name == "xception":
                last_conv_layer = model.model.get_layer("block14_sepconv2_act")
            elif model.model_name == "mesonet":
                # MesoNet usually has simple conv layers
                for layer in reversed(model.model.layers):
                    if 'conv' in layer.name.lower():
                        last_conv_layer = model.model.get_layer(layer.name)
                        break
                        
            if last_conv_layer is None:
                raise ValueError(f"Could not find appropriate conv layer for model {model.model_name}")
            
            # Create Grad-CAM model
            grad_model = tf.keras.models.Model(
                inputs=[model.model.inputs],
                outputs=[last_conv_layer.output, model.model.output]
            )
            
            # Process image
            img_array = np.expand_dims(preprocessed_image, axis=0)
            
            # Record gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                class_idx = 1  # Class index for "deepfake"
                loss = predictions[:, class_idx]
                
            # Gradients of the class output with respect to the conv layer outputs
            grads = tape.gradient(loss, conv_outputs)
            
            # Pool gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight feature maps with gradients
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap).numpy()
            
            # Normalize heatmap
            heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
            
            # Get top activation areas
            top_regions = []
            if np.any(heatmap > 0.7):  # Areas with >70% activation
                high_activation_mask = heatmap > 0.7
                labeled_mask, num_features = cv2.connectedComponents(high_activation_mask.astype(np.uint8))
                for label in range(1, num_features + 1):
                    region_mask = labeled_mask == label
                    region_size = np.sum(region_mask)
                    avg_activation = np.mean(heatmap[region_mask])
                    if region_size > 5:  # Only include reasonably sized regions
                        # Find region centroid
                        y_indices, x_indices = np.where(region_mask)
                        centroid_y = int(np.mean(y_indices))
                        centroid_x = int(np.mean(x_indices))
                        
                        # Add to top regions
                        top_regions.append({
                            "centroid_x": centroid_x,
                            "centroid_y": centroid_y,
                            "size": int(region_size),
                            "activation": float(avg_activation)
                        })
            
            # Sort regions by activation
            top_regions.sort(key=lambda x: x["activation"], reverse=True)
            
            # Calculate average activation for different facial regions
            # (This is a simplification - in a real system, you'd map to actual facial regions)
            regions_of_interest = {
                "eyes": float(np.mean(heatmap[0:int(heatmap.shape[0]/3), :])),
                "nose": float(np.mean(heatmap[int(heatmap.shape[0]/3):int(2*heatmap.shape[0]/3), 
                                      int(heatmap.shape[1]/4):int(3*heatmap.shape[1]/4)])),
                "mouth": float(np.mean(heatmap[int(2*heatmap.shape[0]/3):, :]))
            }
            
            # Calculate overall statistics
            explanation_data = {
                "top_activation_regions": top_regions[:5],  # Top 5 regions
                "regions_of_interest": regions_of_interest,
                "overall_activation": {
                    "mean": float(np.mean(heatmap)),
                    "max": float(np.max(heatmap)),
                    "min": float(np.min(heatmap)),
                    "std": float(np.std(heatmap))
                }
            }
            
            return heatmap, explanation_data
            
        except Exception as e:
            logger.error(f"Error in Grad-CAM generation: {str(e)}", exc_info=True)
            # Return a blank heatmap in case of failure
            blank_heatmap = np.zeros(preprocessed_image.shape[:2])
            return blank_heatmap, {"error": str(e)}
    
    def _generate_shap(self, preprocessed_image, model) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate SHAP visualization for deepfake detection
        
        Args:
            preprocessed_image: Preprocessed face image
            model: Model for inference
            
        Returns:
            Tuple of (heatmap, explanation_data)
        """
        try:
            # Create a background dataset (black image)
            background = np.zeros((1, *preprocessed_image.shape))
            
            # Prepare the image
            img_array = np.expand_dims(preprocessed_image, axis=0)
            
            # Create the SHAP explainer
            explainer = shap.DeepExplainer(model.model, background)
            
            # Get SHAP values
            shap_values = explainer.shap_values(img_array)
            
            # For binary classification, shap_values is a list with one element
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Sum across channels for RGB images
            if preprocessed_image.shape[-1] == 3:
                attributions = np.sum(np.abs(shap_values[0]), axis=-1)
            else:
                attributions = shap_values[0]
            
            # Normalize for heatmap
            attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min() + 1e-10)
            
            # Get top attribution areas
            top_regions = []
            if np.any(attributions > 0.7):  # Areas with >70% attribution
                high_attribution_mask = attributions > 0.7
                labeled_mask, num_features = cv2.connectedComponents(high_attribution_mask.astype(np.uint8))
                for label in range(1, num_features + 1):
                    region_mask = labeled_mask == label
                    region_size = np.sum(region_mask)
                    avg_attribution = np.mean(attributions[region_mask])
                    if region_size > 5:  # Only include reasonably sized regions
                        # Find region centroid
                        y_indices, x_indices = np.where(region_mask)
                        centroid_y = int(np.mean(y_indices))
                        centroid_x = int(np.mean(x_indices))
                        
                        # Add to top regions
                        top_regions.append({
                            "centroid_x": centroid_x,
                            "centroid_y": centroid_y,
                            "size": int(region_size),
                            "attribution": float(avg_attribution)
                        })
            
            # Sort regions by attribution
            top_regions.sort(key=lambda x: x["attribution"], reverse=True)
            
            # Calculate average attribution for different facial regions
            regions_of_interest = {
                "eyes": float(np.mean(attributions[0:int(attributions.shape[0]/3), :])),
                "nose": float(np.mean(attributions[int(attributions.shape[0]/3):int(2*attributions.shape[0]/3), 
                                      int(attributions.shape[1]/4):int(3*attributions.shape[1]/4)])),
                "mouth": float(np.mean(attributions[int(2*attributions.shape[0]/3):, :]))
            }
            
            # Calculate overall statistics
            explanation_data = {
                "top_attribution_regions": top_regions[:5],  # Top 5 regions
                "regions_of_interest": regions_of_interest,
                "overall_attribution": {
                    "mean": float(np.mean(attributions)),
                    "max": float(np.max(attributions)),
                    "min": float(np.min(attributions)),
                    "std": float(np.std(attributions))
                }
            }
            
            return attributions, explanation_data
            
        except Exception as e:
            logger.error(f"Error in SHAP generation: {str(e)}", exc_info=True)
            # Return a blank heatmap in case of failure
            blank_heatmap = np.zeros(preprocessed_image.shape[:2])
            return blank_heatmap, {"error": str(e)}
    
    def _generate_integrated_gradients(self, preprocessed_image, model) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate integrated gradients visualization for deepfake detection
        
        Args:
            preprocessed_image: Preprocessed face image
            model: Model for inference
            
        Returns:
            Tuple of (heatmap, explanation_data)
        """
        try:
            # Define baseline (black image)
            baseline = np.zeros_like(preprocessed_image)
            
            # Number of steps for integral approximation
            steps = 20
            
            # Create batch of interpolated images
            alphas = np.linspace(0, 1, steps)
            interpolated_images = np.array([baseline + alpha * (preprocessed_image - baseline) 
                                           for alpha in alphas])
            
            # Process batch and compute gradients
            interpolated_batch = tf.convert_to_tensor(interpolated_images)
            with tf.GradientTape() as tape:
                tape.watch(interpolated_batch)
                preds = model.model(interpolated_batch)
                class_idx = 1  # Class index for "deepfake"
                outputs = preds[:, class_idx]
            
            # Get gradients
            grads = tape.gradient(outputs, interpolated_batch)
            grads = tf.convert_to_tensor(grads)
            
            # Approximate integral using trapezoidal rule
            avg_grads = tf.reduce_mean(grads[:-1] + grads[1:], axis=0) / 2
            integrated_gradients = (preprocessed_image - baseline) * avg_grads
            
            # Sum across color channels for visualization
            attributions = tf.reduce_sum(integrated_gradients, axis=-1).numpy()
            
            # Normalize for heatmap
            attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min() + 1e-10)
            
            # Get top attribution areas
            top_regions = []
            if np.any(attributions > 0.7):  # Areas with >70% attribution
                high_attribution_mask = attributions > 0.7
                labeled_mask, num_features = cv2.connectedComponents(high_attribution_mask.astype(np.uint8))
                for label in range(1, num_features + 1):
                    region_mask = labeled_mask == label
                    region_size = np.sum(region_mask)
                    avg_attribution = np.mean(attributions[region_mask])
                    if region_size > 5:  # Only include reasonably sized regions
                        # Find region centroid
                        y_indices, x_indices = np.where(region_mask)
                        centroid_y = int(np.mean(y_indices))
                        centroid_x = int(np.mean(x_indices))
                        
                        # Add to top regions
                        top_regions.append({
                            "centroid_x": centroid_x,
                            "centroid_y": centroid_y,
                            "size": int(region_size),
                            "attribution": float(avg_attribution)
                        })
            
            # Sort regions by attribution
            top_regions.sort(key=lambda x: x["attribution"], reverse=True)
            
            # Calculate average attribution for different facial regions
            regions_of_interest = {
                "eyes": float(np.mean(attributions[0:int(attributions.shape[0]/3), :])),
                "nose": float(np.mean(attributions[int(attributions.shape[0]/3):int(2*attributions.shape[0]/3), 
                                      int(attributions.shape[1]/4):int(3*attributions.shape[1]/4)])),
                "mouth": float(np.mean(attributions[int(2*attributions.shape[0]/3):, :]))
            }
            
            # Calculate overall statistics
            explanation_data = {
                "top_attribution_regions": top_regions[:5],  # Top 5 regions
                "regions_of_interest": regions_of_interest,
                "overall_attribution": {
                    "mean": float(np.mean(attributions)),
                    "max": float(np.max(attributions)),
                    "min": float(np.min(attributions)),
                    "std": float(np.std(attributions))
                }
            }
            
            return attributions, explanation_data
            
        except Exception as e:
            logger.error(f"Error in integrated gradients generation: {str(e)}", exc_info=True)
            # Return a blank heatmap in case of failure
            blank_heatmap = np.zeros(preprocessed_image.shape[:2])
            return blank_heatmap, {"error": str(e)}
    
    def _apply_heatmap_to_image(self, image, heatmap) -> np.ndarray:
        """
        Apply a heatmap to the original image
        
        Args:
            image: Original image
            heatmap: Generated heatmap
            
        Returns:
            Image with superimposed heatmap
        """
        # Resize heatmap to match image dimensions
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose heatmap on original image
        if len(image.shape) == 2:  # Convert grayscale to RGB if needed
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # Make sure image is uint8
        if image.dtype != np.uint8:
            image = np.uint8(image * 255)
            
        # Superimpose with opacity
        superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        
        return superimposed_img
    
    def _convert_image_to_base64(self, image) -> str:
        """
        Convert an image to base64 string
        
        Args:
            image: Image to convert
            
        Returns:
            Base64 encoded string
        """
        # Convert to RGB if needed
        if len(image.shape) > 2 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Save to buffer
        buffer = BytesIO()
        plt.figure(figsize=(5, 5))
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Convert to base64
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return image_base64
        
    def save_explanation_visualization(self, image, heatmap, output_path):
        """
        Save the explanation visualization to disk
        
        Args:
            image: Original image
            heatmap: Generated heatmap
            output_path: Path to save the visualization
            
        Returns:
            Path to saved file
        """
        try:
            # Apply heatmap to image
            superimposed_img = self._apply_heatmap_to_image(image, heatmap)
            
            # Save to disk
            cv2.imwrite(output_path, superimposed_img)
            
            return output_path
        except Exception as e:
            logger.error(f"Error saving explanation visualization: {str(e)}", exc_info=True)
            return None

# Initialize explainer singleton
explainer = DeepfakeExplainer()
