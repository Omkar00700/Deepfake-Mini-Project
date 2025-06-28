"""
Visualization Module for Deepfake Detection Results
Provides functions to visualize detection results with heatmaps and annotations
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from typing import Dict, Any, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectionVisualizer:
    """
    Class for visualizing deepfake detection results
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize the visualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Visualization output directory: {output_dir}")
    
    def visualize_detection(self, image_path: str, result: Dict[str, Any], 
                           detection_id: str, save: bool = True) -> np.ndarray:
        """
        Visualize detection result on an image
        
        Args:
            image_path: Path to the input image
            result: Detection result dictionary
            detection_id: Unique ID for the detection
            save: Whether to save the visualization
            
        Returns:
            Annotated image as numpy array
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Create a copy for visualization
            vis_image = image.copy()
            
            # Add overall probability text
            probability = result.get("probability", 0)
            confidence = result.get("confidence", 0)
            
            # Determine text color based on probability
            if probability > 0.7:
                color = (0, 0, 255)  # Red for high probability (likely deepfake)
                label = "DEEPFAKE"
            elif probability > 0.4:
                color = (0, 165, 255)  # Orange for medium probability
                label = "SUSPICIOUS"
            else:
                color = (0, 255, 0)  # Green for low probability (likely real)
                label = "AUTHENTIC"
            
            # Add border based on detection result
            border_thickness = 15
            h, w = vis_image.shape[:2]
            vis_image = cv2.copyMakeBorder(
                vis_image, 
                border_thickness, border_thickness, border_thickness, border_thickness,
                cv2.BORDER_CONSTANT, 
                value=color
            )
            
            # Add header with detection info
            header_height = 60
            header = np.zeros((header_height, w + 2*border_thickness, 3), dtype=np.uint8)
            header[:] = color
            
            # Add text to header
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(header, f"{label} - {probability:.2f} ({confidence:.2f} confidence)", 
                      (20, 40), font, 1, (255, 255, 255), 2)
            
            # Combine header and image
            vis_image = np.vstack([header, vis_image])
            
            # Draw regions if available
            regions = result.get("regions", [])
            for region in regions:
                box = region.get("box")
                if box and len(box) == 4:
                    x1, y1, x2, y2 = box
                    # Adjust coordinates for the added border and header
                    y1 += border_thickness + header_height
                    y2 += border_thickness + header_height
                    x1 += border_thickness
                    x2 += border_thickness
                    
                    # Draw rectangle
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Add region probability
                    region_prob = region.get("probability", 0)
                    cv2.putText(vis_image, f"{region_prob:.2f}", 
                              (x1, y1-5), font, 0.6, color, 2)
                    
                    # Add skin tone if available
                    skin_tone = region.get("skin_tone", {})
                    if skin_tone and skin_tone.get("success"):
                        indian_tone = skin_tone.get("indian_tone", {})
                        tone_name = indian_tone.get("name", "Unknown")
                        cv2.putText(vis_image, f"Tone: {tone_name}", 
                                  (x1, y2+20), font, 0.6, color, 2)
            
            # Add footer with additional info
            footer_height = 40
            footer = np.zeros((footer_height, w + 2*border_thickness, 3), dtype=np.uint8)
            footer[:] = (50, 50, 50)  # Dark gray
            
            # Add text to footer
            model_name = result.get("model", "Unknown")
            processing_time = result.get("processingTime", 0)
            cv2.putText(footer, f"Model: {model_name} | Processing Time: {processing_time:.2f}ms", 
                      (20, 25), font, 0.6, (255, 255, 255), 1)
            
            # Combine image and footer
            vis_image = np.vstack([vis_image, footer])
            
            # Save visualization if requested
            if save:
                output_path = os.path.join(self.output_dir, f"{detection_id}_visualization.jpg")
                cv2.imwrite(output_path, vis_image)
                logger.info(f"Saved visualization to {output_path}")
            
            return vis_image
        
        except Exception as e:
            logger.error(f"Error visualizing detection: {str(e)}")
            return None
    
    def create_heatmap(self, image_path: str, result: Dict[str, Any], 
                      detection_id: str, save: bool = True) -> np.ndarray:
        """
        Create a heatmap visualization of the detection result
        
        Args:
            image_path: Path to the input image
            result: Detection result dictionary
            detection_id: Unique ID for the detection
            save: Whether to save the visualization
            
        Returns:
            Heatmap image as numpy array
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Convert to RGB for matplotlib
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create figure and axes
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Display the image
            ax.imshow(image_rgb)
            
            # Create a heatmap overlay
            h, w = image.shape[:2]
            heatmap = np.zeros((h, w), dtype=np.float32)
            
            # Add heat based on regions
            regions = result.get("regions", [])
            for region in regions:
                box = region.get("box")
                if box and len(box) == 4:
                    x1, y1, x2, y2 = box
                    region_prob = region.get("probability", 0)
                    
                    # Create a gradient around the region
                    y_indices, x_indices = np.mgrid[0:h, 0:w]
                    
                    # Calculate distance from center of region
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    distance = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
                    
                    # Create a radial gradient
                    region_size = max(x2 - x1, y2 - y1)
                    influence = np.exp(-0.5 * (distance / (region_size/2))**2)
                    
                    # Scale by probability
                    influence *= region_prob
                    
                    # Add to heatmap
                    heatmap = np.maximum(heatmap, influence)
            
            # If no regions, use a simple gradient based on overall probability
            if not regions:
                probability = result.get("probability", 0)
                y_indices, x_indices = np.mgrid[0:h, 0:w]
                
                # Create a center-weighted gradient
                center_x, center_y = w/2, h/2
                distance = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
                max_distance = np.sqrt(center_x**2 + center_y**2)
                
                # Normalize distance and create gradient
                normalized_distance = distance / max_distance
                heatmap = (1 - normalized_distance) * probability
            
            # Apply colormap
            norm = Normalize(vmin=0, vmax=1)
            cmap = cm.get_cmap('jet')
            heatmap_colored = cmap(norm(heatmap))
            
            # Display heatmap with transparency
            ax.imshow(heatmap_colored, alpha=0.6)
            
            # Add colorbar
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
            cbar.set_label('Deepfake Probability')
            
            # Add title
            probability = result.get("probability", 0)
            confidence = result.get("confidence", 0)
            plt.title(f"Deepfake Probability: {probability:.2f} (Confidence: {confidence:.2f})")
            
            # Remove axes
            plt.axis('off')
            
            # Save heatmap if requested
            if save:
                output_path = os.path.join(self.output_dir, f"{detection_id}_heatmap.jpg")
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                logger.info(f"Saved heatmap to {output_path}")
                
                # Close the figure to free memory
                plt.close(fig)
                
                # Return the saved image
                return cv2.imread(output_path)
            else:
                # Convert figure to image
                fig.canvas.draw()
                heatmap_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                heatmap_image = heatmap_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                
                # Close the figure to free memory
                plt.close(fig)
                
                return heatmap_image
        
        except Exception as e:
            logger.error(f"Error creating heatmap: {str(e)}")
            return None
    
    def create_comparison_visualization(self, original_path: str, result: Dict[str, Any], 
                                      detection_id: str, save: bool = True) -> np.ndarray:
        """
        Create a side-by-side comparison of original image and visualization
        
        Args:
            original_path: Path to the original image
            result: Detection result dictionary
            detection_id: Unique ID for the detection
            save: Whether to save the visualization
            
        Returns:
            Comparison image as numpy array
        """
        try:
            # Create visualization
            vis_image = self.visualize_detection(original_path, result, detection_id, save=False)
            if vis_image is None:
                return None
            
            # Create heatmap
            heatmap_image = self.create_heatmap(original_path, result, detection_id, save=False)
            if heatmap_image is None:
                return None
            
            # Resize heatmap to match visualization height
            h_vis, w_vis = vis_image.shape[:2]
            h_heat, w_heat = heatmap_image.shape[:2]
            
            # Calculate new width to maintain aspect ratio
            new_w_heat = int(w_heat * (h_vis / h_heat))
            
            # Resize heatmap
            heatmap_resized = cv2.resize(heatmap_image, (new_w_heat, h_vis))
            
            # Convert heatmap to BGR if needed
            if heatmap_resized.shape[2] == 3:
                heatmap_resized = cv2.cvtColor(heatmap_resized, cv2.COLOR_RGB2BGR)
            
            # Combine images horizontally
            comparison = np.hstack([vis_image, heatmap_resized])
            
            # Save comparison if requested
            if save:
                output_path = os.path.join(self.output_dir, f"{detection_id}_comparison.jpg")
                cv2.imwrite(output_path, comparison)
                logger.info(f"Saved comparison to {output_path}")
            
            return comparison
        
        except Exception as e:
            logger.error(f"Error creating comparison visualization: {str(e)}")
            return None