
import numpy as np
import logging
import time
import cv2
from model_loader import DeepfakeDetectionModel
from inference_core import evaluate_input_quality, get_uncertainty_threshold, PredictionResult

# Configure logging
logger = logging.getLogger(__name__)

# This file is maintained for backward compatibility
# The actual model implementation has been moved to model_loader.py

# For backward compatibility, we extend the class and add the legacy methods
class DeepfakeDetectionModelLegacy(DeepfakeDetectionModel):
    """
    Legacy wrapper for DeepfakeDetectionModel for backward compatibility
    Enhanced with advanced calibration and uncertainty estimation
    """
    
    def __init__(self):
        super().__init__()
        logger.info("Using enhanced legacy model wrapper with improved calibration")
        
        # Enable Monte Carlo dropout for uncertainty estimation
        self.uncertainty_enabled = True
        self.mc_dropout_samples = 10  # Increased from 5 for better uncertainty estimates
        
        # Calibration parameters (learned from validation data)
        self.calibration_params = {
            'efficientnet': {'A': 1.2, 'B': -0.1},
            'xception': {'A': 0.9, 'B': 0.05},
            'mesonet': {'A': 1.1, 'B': -0.05},
            'vit': {'A': 1.0, 'B': 0.0},
            'hybrid': {'A': 1.1, 'B': 0.0},
            'default': {'A': 1.0, 'B': 0.0}
        }
        
    def predict(self, image):
        """
        Predict the probability that an image is a deepfake.
        
        Enhanced version with quality assessment, uncertainty estimation,
        and improved calibration.
        
        Args:
            image: Preprocessed face image
            
        Returns:
            Probability value between 0 and 1, where higher values indicate greater
            likelihood of being a deepfake.
        """
        # Assess image quality for confidence weighting
        quality_scores = evaluate_input_quality(image)
        
        # If image quality is too low, log a warning
        if quality_scores['overall'] < 0.4:
            logger.warning(f"Low quality input image (score: {quality_scores['overall']:.3f}), "
                          f"prediction may be less reliable")
        
        # Try to use the enhanced model with uncertainty estimation
        try:
            # Get prediction with uncertainty
            pred_info = super().predict(image, include_uncertainty=True)
            
            # Extract values from prediction result
            if isinstance(pred_info, dict):
                probability = pred_info.get('probability', 0.5)
                uncertainty = pred_info.get('uncertainty', None)
                
                # Apply additional calibration based on image quality
                calibrated_prob = self._advanced_calibrate(
                    probability, 
                    quality_scores=quality_scores,
                    uncertainty=uncertainty
                )
                
                # Log detailed prediction info
                logger.debug(f"Enhanced prediction: raw={probability:.4f}, "
                            f"calibrated={calibrated_prob:.4f}, "
                            f"uncertainty={uncertainty:.4f if uncertainty else 'N/A'}, "
                            f"quality={quality_scores['overall']:.4f}")
                
                return calibrated_prob
            else:
                # If it returns a single value, use it directly
                return pred_info
                
        except Exception as e:
            logger.error(f"Enhanced prediction failed, falling back to legacy: {str(e)}")
            return self._legacy_predict(image)
    
    def _advanced_calibrate(self, probability, quality_scores=None, uncertainty=None):
        """
        Apply advanced calibration to raw model output
        
        Args:
            probability: Raw model probability
            quality_scores: Dictionary of image quality scores
            uncertainty: Uncertainty estimate from Monte Carlo dropout
            
        Returns:
            Calibrated probability
        """
        # 1. Apply Platt scaling based on model type
        model_params = self.calibration_params.get(
            self.model_name, self.calibration_params['default']
        )
        
        # Convert probability to logit for scaling
        eps = 1e-7  # To avoid log(0) or log(1)
        prob_clipped = min(1 - eps, max(eps, probability))
        logit = np.log(prob_clipped / (1 - prob_clipped))
        
        # Apply scaling
        A, B = model_params['A'], model_params['B']
        scaled_logit = A * logit + B
        
        # Convert back to probability
        calibrated_prob = 1 / (1 + np.exp(-scaled_logit))
        
        # 2. Adjust based on image quality if available
        if quality_scores:
            # Higher quality allows probabilities to be more extreme
            # Lower quality pulls probabilities toward 0.5
            quality_factor = quality_scores['overall']
            quality_adjustment = 0.5 + (calibrated_prob - 0.5) * min(1.5, quality_factor * 1.2)
            calibrated_prob = quality_adjustment
        
        # 3. Consider uncertainty if available
        if uncertainty is not None:
            # Higher uncertainty pulls probabilities toward 0.5
            uncertainty_threshold = get_uncertainty_threshold(calibrated_prob)
            if uncertainty > uncertainty_threshold:
                # Pull toward 0.5 based on how much uncertainty exceeds the threshold
                excess_uncertainty = (uncertainty - uncertainty_threshold) / (1 - uncertainty_threshold)
                uncertainty_adjustment = 0.5 + (calibrated_prob - 0.5) * (1 - min(0.8, excess_uncertainty))
                calibrated_prob = uncertainty_adjustment
        
        # 4. Final clipping to valid range with less extreme bounds
        return min(0.95, max(0.05, calibrated_prob))
    
    def _legacy_predict(self, image):
        """
        Legacy prediction method from the original implementation
        Enhanced with improved image quality metrics and calibration
        """
        # Get enhanced image quality metrics
        blur_measure = cv2_blur_measure(image)
        contrast = image_contrast(image)
        noise_level = noise_estimate(image)
        color_coherence = color_coherence_score(image)
        texture_score = texture_consistency(image)
        
        # Additional metrics for improved detection
        image_quality = overall_quality_score(image)
        facial_coherence = facial_feature_coherence(image)
        compression_artifacts = detect_compression_artifacts(image)
        lighting_consistency = lighting_consistency_score(image)
        
        # Baseline score - neutral starting point
        base_score = 0.45
        score_components = []
        
        # Analyze blur (excessive blur may indicate deepfake)
        # Normalize blur measure to [0,1] range (higher = more suspicious)
        norm_blur = min(1.0, max(0, 1.0 - (blur_measure / 150)))
        score_components.append(("blur", norm_blur * 0.15))  # 15% weight
        
        # Analyze contrast (poor contrast often in deepfakes)
        # High contrast is less suspicious (subtract from 1.0)
        norm_contrast = min(1.0, max(0, 1.0 - contrast))
        score_components.append(("contrast", norm_contrast * 0.1))  # 10% weight
        
        # Analyze noise (inconsistent noise patterns in deepfakes)
        # Normalize noise to [0,1] range
        norm_noise = min(1.0, max(0, noise_level * 10))
        score_components.append(("noise", norm_noise * 0.15))  # 15% weight
        
        # Color coherence (deepfakes may have color inconsistencies)
        # Lower coherence is more suspicious
        score_components.append(("color", (1.0 - color_coherence) * 0.15))  # 15% weight
        
        # Texture consistency (deepfakes often have texture issues)
        # Lower score is more suspicious
        score_components.append(("texture", (1.0 - texture_score) * 0.15))  # 15% weight
        
        # New metrics with enhanced weighting
        score_components.append(("quality", (1.0 - image_quality) * 0.1))  # 10% weight
        score_components.append(("facial", (1.0 - facial_coherence) * 0.1))  # 10% weight
        score_components.append(("compression", compression_artifacts * 0.05))  # 5% weight
        score_components.append(("lighting", (1.0 - lighting_consistency) * 0.05))  # 5% weight
        
        # Compute final score by adding all weighted components to base score
        final_score = base_score
        for component_name, component_value in score_components:
            final_score += component_value
            logger.debug(f"Component {component_name}: {component_value:.4f}")
        
        # Apply sigmoid function for better calibration
        final_score = 1.0 / (1.0 + np.exp(-4 * (final_score - 0.5)))
        
        # Ensure prediction is between 0.05 and 0.95 (avoid extreme certainty)
        prediction = max(0.05, min(0.95, final_score))
        
        logger.debug(f"Final legacy prediction: {prediction:.4f}")
        return prediction


# The legacy utility functions from the original model.py are enhanced below
# These are used by the _legacy_predict method

def cv2_blur_measure(image):
    """
    Measure the amount of blur in an image
    Higher values indicate less blur (sharper image)
    Enhanced with multiple blur metrics
    """
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.uint8)
        
    # Measure 1: Laplacian variance (standard method)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Measure 2: Sobel gradient magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2).mean()
    
    # Combine measures with weights
    combined_score = lap_var * 0.7 + sobel_mag * 0.3
    
    return combined_score

def image_contrast(image):
    """
    Measure image contrast using multiple metrics
    """
    if len(image.shape) == 3:
        # For color images, convert to grayscale
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.uint8)
        
    # Metric 1: Simple range contrast
    min_val = np.min(gray)
    max_val = np.max(gray)
    range_contrast = (max_val - min_val) / max(1, max_val + min_val)
    
    # Metric 2: Standard deviation contrast
    std_contrast = np.std(gray) / 128  # Normalized to [0,1] range (approximately)
    
    # Metric 3: Local contrast using local standard deviation
    local_std = cv2.blur(
        np.float32((gray - cv2.blur(gray, (11, 11))) ** 2), 
        (11, 11)
    )
    local_std = np.sqrt(local_std)
    local_contrast = np.mean(local_std) / 128
    
    # Combine metrics
    combined_contrast = range_contrast * 0.4 + std_contrast * 0.4 + local_contrast * 0.2
    
    return min(1.0, combined_contrast)  # Ensure result is in [0,1]

def noise_estimate(image):
    """
    Estimate the amount of noise in an image
    Enhanced with multiple noise estimation techniques
    """
    if len(image.shape) == 3:
        # For color images, use luminance
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.uint8)
    
    # Technique 1: Median filter difference
    filtered = cv2.medianBlur(gray, 3)
    noise1 = np.mean(np.abs(gray.astype(np.float32) - filtered.astype(np.float32))) / 255.0
    
    # Technique 2: High-frequency components
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    highpass = cv2.filter2D(gray, -1, kernel)
    noise2 = np.mean(np.abs(highpass)) / 255.0
    
    # Technique 3: Wavelet-based noise estimation (simplified)
    # Use the smallest-scale wavelet coefficients
    gray_float = gray.astype(np.float32) / 255.0
    noise3 = np.std(gray_float - cv2.GaussianBlur(gray_float, (5, 5), 0))
    
    # Combine noise estimates
    combined_noise = noise1 * 0.4 + noise2 * 0.3 + noise3 * 0.3
    
    return min(1.0, combined_noise)  # Ensure result is in [0,1]

def color_coherence_score(image):
    """
    Analyze color coherence across the image
    Returns a value between 0 and 1, where higher is more coherent (less suspicious)
    Enhanced with additional color metrics
    """
    if len(image.shape) < 3:
        return 1.0  # Grayscale images are considered coherent
    
    # Convert to uint8 and extract channels
    img = image.astype(np.uint8)
    b, g, r = cv2.split(img)
    
    # Calculate standard deviation of each channel
    std_r = np.std(r) / 255.0
    std_g = np.std(g) / 255.0
    std_b = np.std(b) / 255.0
    
    # Calculate correlation between channels (higher correlation = more natural)
    corr_rg = np.corrcoef(r.flatten(), g.flatten())[0, 1]
    corr_rb = np.corrcoef(r.flatten(), b.flatten())[0, 1]
    corr_gb = np.corrcoef(g.flatten(), b.flatten())[0, 1]
    
    # Calculate mean correlation
    mean_corr = (corr_rg + corr_rb + corr_gb) / 3.0
    
    # Convert to [0,1] scale where higher is more natural
    correlation_coherence = (mean_corr + 1.0) / 2.0
    
    # Additional metric: Channel ratio consistency
    # Natural images often have a certain balance between channels
    mean_r = np.mean(r) / 255.0
    mean_g = np.mean(g) / 255.0
    mean_b = np.mean(b) / 255.0
    
    # Calculate the variance of the means (lower = more balanced)
    channel_balance = 1.0 - min(1.0, 3.0 * np.var([mean_r, mean_g, mean_b]))
    
    # Combine metrics
    coherence = correlation_coherence * 0.7 + channel_balance * 0.3
    
    return coherence

def texture_consistency(image):
    """
    Analyze texture consistency across the image
    Returns a value between 0 and 1, where higher is more consistent (less suspicious)
    Enhanced with multiple texture analysis metrics
    """
    if len(image.shape) == 3:
        # For color images, convert to grayscale
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.uint8)
    
    # Metric 1: Gabor filter consistency
    kernel = cv2.getGaborKernel((21, 21), 5, 1, 10, 1, 0, cv2.CV_32F)
    filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
    mean, stddev = cv2.meanStdDev(filtered)
    gabor_consistency = 1.0 - min(1.0, stddev[0][0] / 50.0)
    
    # Metric 2: LBP (Local Binary Pattern) histogram variance
    # Simplified implementation for computation efficiency
    # Higher variance indicates inconsistent textures
    lbp = np.zeros_like(gray)
    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            center = gray[i, j]
            code = 0
            code |= (gray[i-1, j-1] >= center) << 7
            code |= (gray[i-1, j] >= center) << 6
            code |= (gray[i-1, j+1] >= center) << 5
            code |= (gray[i, j+1] >= center) << 4
            code |= (gray[i+1, j+1] >= center) << 3
            code |= (gray[i+1, j] >= center) << 2
            code |= (gray[i+1, j-1] >= center) << 1
            code |= (gray[i, j-1] >= center) << 0
            lbp[i, j] = code
    
    # Calculate LBP histogram for different regions
    regions = [
        lbp[:lbp.shape[0]//2, :lbp.shape[1]//2],
        lbp[:lbp.shape[0]//2, lbp.shape[1]//2:],
        lbp[lbp.shape[0]//2:, :lbp.shape[1]//2],
        lbp[lbp.shape[0]//2:, lbp.shape[1]//2:]
    ]
    
    histograms = []
    for region in regions:
        hist, _ = np.histogram(region, bins=32, range=(0, 256))
        histograms.append(hist / max(1, np.sum(hist)))
    
    # Calculate average pairwise correlation between histograms
    correlations = []
    for i in range(len(histograms)):
        for j in range(i+1, len(histograms)):
            corr = np.corrcoef(histograms[i], histograms[j])[0, 1]
            correlations.append((corr + 1) / 2)  # Convert from [-1,1] to [0,1]
    
    lbp_consistency = np.mean(correlations) if correlations else 0.5
    
    # Metric 3: Gradient direction consistency
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_dirs = np.arctan2(sobely, sobelx) * 180 / np.pi
    
    # Convert to histogram
    grad_hist, _ = np.histogram(gradient_dirs, bins=36, range=(-180, 180))
    grad_hist = grad_hist / max(1, np.sum(grad_hist))
    
    # Calculate entropy (lower entropy = more consistent directions)
    non_zero = grad_hist[grad_hist > 0]
    entropy = -np.sum(non_zero * np.log2(non_zero)) if len(non_zero) > 0 else 0
    max_entropy = np.log2(36)  # Maximum possible entropy
    gradient_consistency = 1.0 - (entropy / max_entropy)
    
    # Combine consistency scores
    consistency = (
        gabor_consistency * 0.4 +
        lbp_consistency * 0.4 +
        gradient_consistency * 0.2
    )
    
    return consistency

def overall_quality_score(image):
    """
    Calculate an overall image quality score combining multiple metrics
    Returns a value between 0 and 1, where higher is better quality
    """
    # Get existing metrics
    blur = cv2_blur_measure(image)
    contrast_val = image_contrast(image)
    noise = noise_estimate(image)
    
    # Calculate normalized blur score (higher = better)
    blur_score = min(1.0, blur / 300)
    
    # Calculate noise score (higher = better)
    noise_score = 1.0 - min(1.0, noise * 5)
    
    # Assess resolution adequacy
    h, w = image.shape[:2]
    resolution_score = min(1.0, (h * w) / (250 * 250))
    
    # Combined quality score
    quality = (
        blur_score * 0.4 +
        contrast_val * 0.2 +
        noise_score * 0.3 +
        resolution_score * 0.1
    )
    
    return quality

def facial_feature_coherence(image):
    """
    Analyze coherence of facial features
    Simple implementation that checks for feature consistency
    """
    # This is a simplified implementation
    # In practice, would use a face landmark detector and analyze relationships
    
    # For now, use a proxy metric based on edge consistency around face regions
    if len(image.shape) == 3:
        # For color images, convert to grayscale
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.uint8)
    
    # Calculate edges
    edges = cv2.Canny(gray, 100, 200)
    
    # Divide image into potential facial regions (simplified)
    h, w = gray.shape
    regions = [
        edges[h//4:3*h//4, w//4:3*w//4],  # Center (face)
        edges[:h//3, w//3:2*w//3],  # Top (forehead)
        edges[2*h//3:, w//3:2*w//3]  # Bottom (chin)
    ]
    
    # Calculate edge density for each region
    densities = [np.mean(region > 0) for region in regions]
    
    # Calculate coherence as inverse of variance in edge densities
    # Higher variance = less coherent
    variance = np.var(densities)
    coherence = 1.0 - min(1.0, variance * 10)
    
    return coherence

def detect_compression_artifacts(image):
    """
    Detect compression artifacts that might indicate manipulation
    Returns a value between 0 and 1, where higher indicates more artifacts
    """
    if len(image.shape) == 3:
        # For color images, use Y channel (luminance)
        ycrcb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:,:,0].astype(np.float32)
    else:
        y = image.astype(np.float32)
    
    # Check for block artifacts (common in JPEG compression)
    # Calculate horizontal and vertical differences
    h_diff = np.abs(y[:, 1:] - y[:, :-1]).mean()
    v_diff = np.abs(y[1:, :] - y[:-1, :]).mean()
    
    # Calculate block boundary differences (every 8 pixels for JPEG)
    h_block = np.abs(y[:, 8::8] - y[:, 7::8]).mean() if y.shape[1] >= 8 else 0
    v_block = np.abs(y[8::8, :] - y[7::8, :]).mean() if y.shape[0] >= 8 else 0
    
    # Calculate ratio of block boundary differences to average differences
    h_ratio = h_block / max(0.0001, h_diff)
    v_ratio = v_block / max(0.0001, v_diff)
    
    # Higher ratio indicates stronger block artifacts
    block_artifact_score = min(1.0, (h_ratio + v_ratio) / 4)
    
    # Detect ringing artifacts around edges
    edges = cv2.Canny(image.astype(np.uint8), 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    edge_dilated = cv2.dilate(edges, kernel, iterations=1)
    edge_region = edge_dilated > 0
    
    # Apply high-pass filter to image
    highpass = y - cv2.GaussianBlur(y, (5, 5), 0)
    
    # Calculate variance of high-frequency components near edges
    if np.sum(edge_region) > 0:
        edge_highpass = highpass[edge_region]
        ringing_score = min(1.0, np.var(edge_highpass) * 5)
    else:
        ringing_score = 0
    
    # Combine scores
    artifact_score = block_artifact_score * 0.7 + ringing_score * 0.3
    
    return artifact_score

def lighting_consistency_score(image):
    """
    Analyze lighting consistency across the image
    Inconsistent lighting can indicate manipulation
    """
    if len(image.shape) < 3:
        # Convert grayscale to "color" for consistent processing
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    # Convert to LAB color space - L channel represents lightness
    lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2LAB)
    L = lab[:,:,0].astype(np.float32)
    
    # Divide image into regions
    h, w = L.shape
    regions = [
        L[:h//2, :w//2],    # Top-left
        L[:h//2, w//2:],    # Top-right
        L[h//2:, :w//2],    # Bottom-left
        L[h//2:, w//2:],    # Bottom-right
        L[h//4:3*h//4, w//4:3*w//4]  # Center
    ]
    
    # Calculate mean and standard deviation for each region
    means = [np.mean(region) for region in regions]
    stds = [np.std(region) for region in regions]
    
    # Calculate lighting gradient across image
    grad_h = np.mean(L[h//2:, :]) - np.mean(L[:h//2, :])  # Vertical gradient
    grad_v = np.mean(L[:, w//2:]) - np.mean(L[:, :w//2])  # Horizontal gradient
    grad_magnitude = np.sqrt(grad_h**2 + grad_v**2)
    
    # Calculate consistency metrics:
    # 1. Variance of means (lower is more consistent)
    mean_variance = np.var(means)
    mean_consistency = 1.0 - min(1.0, mean_variance / 300)
    
    # 2. Similarity of standard deviations (higher is more consistent)
    std_min = min(stds)
    std_max = max(stds)
    std_ratio = std_min / max(0.0001, std_max)
    std_consistency = std_ratio
    
    # 3. Gradient magnitude (lower is more consistent)
    gradient_consistency = 1.0 - min(1.0, grad_magnitude / 50)
    
    # Combine metrics
    consistency = (
        mean_consistency * 0.5 +
        std_consistency * 0.2 +
        gradient_consistency * 0.3
    )
    
    return consistency

