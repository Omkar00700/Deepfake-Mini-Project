"""
Skin Tone Analyzer for DeepDefend
This module provides specialized utilities for analyzing skin tones
with a focus on Indian skin tones and characteristics
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional
import os
import json
from sklearn.cluster import KMeans

# Configure logging
logger = logging.getLogger(__name__)

# Fitzpatrick skin type reference values (approximate RGB values)
FITZPATRICK_REFERENCE = {
    1: {"name": "Type I (Very Light)", "rgb": (255, 236, 228), "range": (0.0, 0.15)},
    2: {"name": "Type II (Light)", "rgb": (241, 194, 167), "range": (0.15, 0.30)},
    3: {"name": "Type III (Medium)", "rgb": (224, 172, 138), "range": (0.30, 0.45)},
    4: {"name": "Type IV (Olive)", "rgb": (198, 134, 94), "range": (0.45, 0.60)},
    5: {"name": "Type V (Brown)", "rgb": (141, 85, 54), "range": (0.60, 0.80)},
    6: {"name": "Type VI (Dark Brown/Black)", "rgb": (80, 51, 36), "range": (0.80, 1.0)}
}

# Indian skin tone reference (more specific to Indian subcontinent)
INDIAN_SKIN_TONES = {
    "fair": {"name": "Fair", "rgb": (241, 194, 167), "range": (0.15, 0.30)},
    "wheatish": {"name": "Wheatish", "rgb": (224, 172, 138), "range": (0.30, 0.45)},
    "medium": {"name": "Medium", "rgb": (198, 134, 94), "range": (0.45, 0.60)},
    "dusky": {"name": "Dusky", "rgb": (172, 112, 76), "range": (0.60, 0.75)},
    "dark": {"name": "Dark", "rgb": (141, 85, 54), "range": (0.75, 0.90)}
}

class SkinToneAnalyzer:
    """
    Analyzes skin tones in facial images with a focus on Indian skin tones
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the skin tone analyzer
        
        Args:
            model_path: Optional path to a custom skin tone model
        """
        self.model_path = model_path
        self.skin_tone_model = None
        
        # Load custom model if provided
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'r') as f:
                    self.skin_tone_model = json.load(f)
                logger.info(f"Loaded custom skin tone model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load skin tone model: {str(e)}")
        
        logger.info("Skin tone analyzer initialized")
    
    def extract_skin_mask(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract a mask of skin pixels from a face image
        
        Args:
            face_image: Face image in BGR format
            
        Returns:
            Binary mask of skin pixels
        """
        # Convert to different color spaces for better skin detection
        img_hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        img_ycrcb = cv2.cvtColor(face_image, cv2.COLOR_BGR2YCrCb)
        
        # HSV skin color range for diverse skin tones
        # Wider range to accommodate Indian skin tones
        lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
        upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
        
        # YCrCb skin color range
        lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
        upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
        
        # Create masks
        mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
        mask_ycrcb = cv2.inRange(img_ycrcb, lower_ycrcb, upper_ycrcb)
        
        # Combine masks for better results
        skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
        
        return skin_mask
    
    def extract_skin_pixels(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract skin pixels from a face image
        
        Args:
            face_image: Face image in BGR format
            
        Returns:
            Array of skin pixels
        """
        # Get skin mask
        skin_mask = self.extract_skin_mask(face_image)
        
        # Apply mask to original image
        skin = cv2.bitwise_and(face_image, face_image, mask=skin_mask)
        
        # Extract non-zero pixels (skin pixels)
        skin_pixels = skin[np.where((skin_mask > 0))]
        
        return skin_pixels
    
    def analyze_skin_tone(self, face_image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the skin tone of a face
        
        Args:
            face_image: Face image in BGR format
            
        Returns:
            Dictionary with skin tone analysis results
        """
        try:
            # Extract skin pixels
            skin_pixels = self.extract_skin_pixels(face_image)
            
            if skin_pixels.size == 0:
                logger.warning("No skin pixels detected in the image")
                return {
                    "success": False,
                    "error": "No skin pixels detected",
                    "fitzpatrick_type": None,
                    "indian_tone": None,
                    "dominant_color": None,
                    "skin_percentage": 0.0
                }
            
            # Calculate dominant skin color using K-means clustering
            skin_pixels_rgb = cv2.cvtColor(skin_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(skin_pixels_rgb)
            
            # Get dominant color (cluster center with most points)
            cluster_sizes = np.bincount(kmeans.labels_)
            dominant_cluster = np.argmax(cluster_sizes)
            dominant_color_rgb = kmeans.cluster_centers_[dominant_cluster].astype(int)
            
            # Calculate skin percentage
            skin_percentage = np.count_nonzero(self.extract_skin_mask(face_image)) / (face_image.shape[0] * face_image.shape[1])
            
            # Calculate skin tone darkness score (0 to 1, where 1 is darkest)
            # Convert RGB to L*a*b* color space and use L* channel (lightness)
            dominant_color_lab = cv2.cvtColor(np.uint8([[dominant_color_rgb]]), cv2.COLOR_RGB2Lab)[0][0]
            lightness = dominant_color_lab[0] / 255.0  # L* channel normalized to [0,1]
            darkness_score = 1.0 - lightness
            
            # Determine Fitzpatrick skin type
            fitzpatrick_type = None
            min_distance = float('inf')
            
            for skin_type, data in FITZPATRICK_REFERENCE.items():
                if darkness_score >= data["range"][0] and darkness_score < data["range"][1]:
                    fitzpatrick_type = {
                        "type": skin_type,
                        "name": data["name"],
                        "score": darkness_score
                    }
                    break
            
            # If no match found, find closest match
            if fitzpatrick_type is None:
                for skin_type, data in FITZPATRICK_REFERENCE.items():
                    mid_range = (data["range"][0] + data["range"][1]) / 2
                    distance = abs(darkness_score - mid_range)
                    if distance < min_distance:
                        min_distance = distance
                        fitzpatrick_type = {
                            "type": skin_type,
                            "name": data["name"],
                            "score": darkness_score
                        }
            
            # Determine Indian skin tone
            indian_tone = None
            min_distance = float('inf')
            
            for tone_id, data in INDIAN_SKIN_TONES.items():
                if darkness_score >= data["range"][0] and darkness_score < data["range"][1]:
                    indian_tone = {
                        "type": tone_id,
                        "name": data["name"],
                        "score": darkness_score
                    }
                    break
            
            # If no match found, find closest match
            if indian_tone is None:
                for tone_id, data in INDIAN_SKIN_TONES.items():
                    mid_range = (data["range"][0] + data["range"][1]) / 2
                    distance = abs(darkness_score - mid_range)
                    if distance < min_distance:
                        min_distance = distance
                        indian_tone = {
                            "type": tone_id,
                            "name": data["name"],
                            "score": darkness_score
                        }
            
            # Calculate skin tone evenness (standard deviation of skin pixels)
            skin_std = np.std(skin_pixels_rgb, axis=0).mean()
            evenness_score = 1.0 - min(1.0, skin_std / 50.0)  # Normalize to [0,1]
            
            # Calculate skin texture score based on local variance
            skin_mask = self.extract_skin_mask(face_image)
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            skin_gray = cv2.bitwise_and(gray_face, gray_face, mask=skin_mask)
            
            # Apply Gaussian blur and calculate local variance
            blurred = cv2.GaussianBlur(skin_gray, (5, 5), 0)
            variance = cv2.Laplacian(blurred, cv2.CV_64F).var()
            texture_score = min(1.0, variance / 500.0)  # Normalize to [0,1]
            
            return {
                "success": True,
                "fitzpatrick_type": fitzpatrick_type,
                "indian_tone": indian_tone,
                "dominant_color": {
                    "rgb": dominant_color_rgb.tolist(),
                    "hex": '#{:02x}{:02x}{:02x}'.format(*dominant_color_rgb)
                },
                "skin_percentage": float(skin_percentage),
                "darkness_score": float(darkness_score),
                "evenness_score": float(evenness_score),
                "texture_score": float(texture_score)
            }
        
        except Exception as e:
            logger.error(f"Error analyzing skin tone: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "fitzpatrick_type": None,
                "indian_tone": None,
                "dominant_color": None,
                "skin_percentage": 0.0
            }
    
    def analyze_skin_tone_batch(self, face_images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Analyze skin tones for multiple faces
        
        Args:
            face_images: List of face images
            
        Returns:
            List of skin tone analysis results
        """
        results = []
        for face in face_images:
            result = self.analyze_skin_tone(face)
            results.append(result)
        
        return results
    
    def get_skin_tone_distribution(self, face_image: np.ndarray) -> Dict[str, Any]:
        """
        Get the distribution of skin tones in a face image
        
        Args:
            face_image: Face image in BGR format
            
        Returns:
            Dictionary with skin tone distribution
        """
        try:
            # Extract skin pixels
            skin_pixels = self.extract_skin_pixels(face_image)
            
            if skin_pixels.size == 0:
                logger.warning("No skin pixels detected in the image")
                return {
                    "success": False,
                    "error": "No skin pixels detected",
                    "distribution": {}
                }
            
            # Convert to RGB for analysis
            skin_pixels_rgb = cv2.cvtColor(skin_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
            
            # Convert to L*a*b* color space
            skin_pixels_lab = cv2.cvtColor(skin_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2Lab).reshape(-1, 3)
            
            # Get lightness values (L* channel)
            lightness = skin_pixels_lab[:, 0] / 255.0  # Normalize to [0,1]
            darkness = 1.0 - lightness
            
            # Calculate distribution across Fitzpatrick scale
            fitzpatrick_distribution = {}
            for skin_type, data in FITZPATRICK_REFERENCE.items():
                lower, upper = data["range"]
                count = np.sum((darkness >= lower) & (darkness < upper))
                percentage = count / len(darkness) if len(darkness) > 0 else 0
                fitzpatrick_distribution[str(skin_type)] = {
                    "name": data["name"],
                    "percentage": float(percentage)
                }
            
            # Calculate distribution across Indian skin tones
            indian_distribution = {}
            for tone_id, data in INDIAN_SKIN_TONES.items():
                lower, upper = data["range"]
                count = np.sum((darkness >= lower) & (darkness < upper))
                percentage = count / len(darkness) if len(darkness) > 0 else 0
                indian_distribution[tone_id] = {
                    "name": data["name"],
                    "percentage": float(percentage)
                }
            
            return {
                "success": True,
                "fitzpatrick_distribution": fitzpatrick_distribution,
                "indian_distribution": indian_distribution,
                "pixel_count": int(len(darkness))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing skin tone distribution: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "distribution": {}
            }
    
    def detect_skin_anomalies(self, face_image: np.ndarray) -> Dict[str, Any]:
        """
        Detect anomalies in skin tone that might indicate deepfakes
        
        Args:
            face_image: Face image in BGR format
            
        Returns:
            Dictionary with skin anomaly detection results
        """
        try:
            # Extract skin mask
            skin_mask = self.extract_skin_mask(face_image)
            
            if np.count_nonzero(skin_mask) == 0:
                logger.warning("No skin pixels detected in the image")
                return {
                    "success": False,
                    "error": "No skin pixels detected",
                    "anomalies": []
                }
            
            # Convert to different color spaces for analysis
            face_hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
            face_lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2Lab)
            
            # Apply skin mask
            skin_hsv = cv2.bitwise_and(face_hsv, face_hsv, mask=skin_mask)
            skin_lab = cv2.bitwise_and(face_lab, face_lab, mask=skin_mask)
            
            # Extract channels
            h, s, v = cv2.split(skin_hsv)
            l, a, b = cv2.split(skin_lab)
            
            # Calculate statistics for each channel
            h_stats = self._calculate_channel_stats(h, skin_mask)
            s_stats = self._calculate_channel_stats(s, skin_mask)
            v_stats = self._calculate_channel_stats(v, skin_mask)
            l_stats = self._calculate_channel_stats(l, skin_mask)
            a_stats = self._calculate_channel_stats(a, skin_mask)
            b_stats = self._calculate_channel_stats(b, skin_mask)
            
            # Detect anomalies
            anomalies = []
            
            # Check for unnatural hue variation
            if h_stats["std"] > 20:
                anomalies.append({
                    "type": "unnatural_hue_variation",
                    "description": "Unusual variation in skin hue",
                    "severity": min(1.0, (h_stats["std"] - 20) / 20)
                })
            
            # Check for unnatural saturation
            if s_stats["mean"] > 150:
                anomalies.append({
                    "type": "oversaturated_skin",
                    "description": "Skin appears unnaturally saturated",
                    "severity": min(1.0, (s_stats["mean"] - 150) / 100)
                })
            
            # Check for unnatural brightness variation
            if v_stats["std"] > 50:
                anomalies.append({
                    "type": "uneven_brightness",
                    "description": "Unusual variation in skin brightness",
                    "severity": min(1.0, (v_stats["std"] - 50) / 50)
                })
            
            # Check for unnatural skin tone transitions
            # Calculate gradient magnitude
            sobelx = cv2.Sobel(l, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(l, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Apply skin mask to gradient
            gradient_skin = cv2.bitwise_and(gradient_magnitude.astype(np.uint8), 
                                          gradient_magnitude.astype(np.uint8), 
                                          mask=skin_mask)
            
            # Calculate gradient statistics
            gradient_mean = np.mean(gradient_skin[gradient_skin > 0]) if np.count_nonzero(gradient_skin) > 0 else 0
            
            if gradient_mean > 20:
                anomalies.append({
                    "type": "unnatural_transitions",
                    "description": "Unnatural transitions in skin tone",
                    "severity": min(1.0, (gradient_mean - 20) / 30)
                })
            
            # Check for color inconsistency in a/b channels (indicates unnatural skin color)
            if a_stats["std"] > 10 or b_stats["std"] > 15:
                anomalies.append({
                    "type": "color_inconsistency",
                    "description": "Inconsistent skin color",
                    "severity": min(1.0, max((a_stats["std"] - 10) / 10, (b_stats["std"] - 15) / 15))
                })
            
            # Calculate overall anomaly score
            anomaly_score = 0.0
            if anomalies:
                anomaly_score = sum(a["severity"] for a in anomalies) / len(anomalies)
            
            return {
                "success": True,
                "anomalies": anomalies,
                "anomaly_score": float(anomaly_score),
                "channel_stats": {
                    "hue": h_stats,
                    "saturation": s_stats,
                    "value": v_stats,
                    "lightness": l_stats,
                    "a_channel": a_stats,
                    "b_channel": b_stats
                }
            }
            
        except Exception as e:
            logger.error(f"Error detecting skin anomalies: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "anomalies": []
            }
    
    def analyze_face_authenticity(self, face_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze face authenticity based on skin tone and other features
        
        Args:
            face_data: Dictionary with face data including skin tone analysis
            
        Returns:
            Dictionary with authenticity analysis results
        """
        try:
            # Extract skin tone data
            skin_tone = face_data.get("skin_tone", {})
            if not skin_tone or not skin_tone.get("success", False):
                return {
                    "authenticity_score": 0.5,
                    "confidence": 0.3,
                    "analysis": "Insufficient skin tone data for authenticity analysis"
                }
            
            # Extract skin anomalies
            skin_anomalies = face_data.get("skin_anomalies", {})
            has_anomalies = skin_anomalies.get("success", False) and len(skin_anomalies.get("anomalies", [])) > 0
            
            # Initialize authenticity score (higher = more authentic)
            authenticity_score = 0.8  # Start with assumption of authenticity
            
            # Factors that reduce authenticity score
            factors = []
            
            # Check skin evenness (too even is suspicious)
            evenness_score = skin_tone.get("evenness_score", 0.5)
            if evenness_score > 0.95:  # Unnaturally even
                authenticity_score -= 0.2
                factors.append("Unnaturally even skin texture")
            
            # Check texture score (too smooth is suspicious)
            texture_score = skin_tone.get("texture_score", 0.5)
            if texture_score < 0.2:  # Unnaturally smooth
                authenticity_score -= 0.15
                factors.append("Unnaturally smooth skin")
            
            # Check for skin anomalies
            if has_anomalies:
                anomaly_score = skin_anomalies.get("anomaly_score", 0.0)
                authenticity_score -= anomaly_score * 0.3  # Weight anomalies
                
                # Add specific anomalies to factors
                for anomaly in skin_anomalies.get("anomalies", []):
                    factors.append(f"{anomaly.get('description', 'Unknown anomaly')}")
            
            # Ensure score is in valid range
            authenticity_score = max(0.1, min(0.99, authenticity_score))
            
            # Calculate confidence in this analysis
            confidence = 0.7  # Base confidence
            if not has_anomalies:
                confidence -= 0.2  # Lower confidence if no anomaly detection
            
            # Return results
            return {
                "authenticity_score": float(authenticity_score),
                "confidence": float(confidence),
                "factors": factors,
                "analysis": "High authenticity score indicates likely real face" if authenticity_score > 0.7 else "Low authenticity score suggests possible manipulation"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing face authenticity: {str(e)}", exc_info=True)
            return {
                "authenticity_score": 0.5,
                "confidence": 0.3,
                "analysis": f"Analysis failed: {str(e)}"
            }
    
    def _calculate_channel_stats(self, channel: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistics for a color channel within a mask
        
        Args:
            channel: Color channel
            mask: Binary mask
            
        Returns:
            Dictionary with channel statistics
        """
        # Extract non-zero pixels
        pixels = channel[mask > 0]
        
        if pixels.size == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0
            }
        
        return {
            "mean": float(np.mean(pixels)),
            "std": float(np.std(pixels)),
            "min": float(np.min(pixels)),
            "max": float(np.max(pixels)),
            "median": float(np.median(pixels))
        }