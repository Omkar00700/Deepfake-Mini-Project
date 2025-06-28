"""
PDF Report Generator for Deepfake Detection Results
Generates detailed PDF reports with visualizations and analysis
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import cv2
import numpy as np
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from visualization import DetectionVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDF(FPDF):
    """
    Custom PDF class with header and footer
    """
    def header(self):
        # Logo
        # self.image('logo.png', 10, 8, 33)
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'DeepDefend - Deepfake Detection Report', 0, 0, 'C')
        # Line break
        self.ln(20)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cell(0, 10, timestamp, 0, 0, 'R')

class PDFReportGenerator:
    """
    Class for generating PDF reports from detection results
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the PDF report generator
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualizer
        self.visualizer = DetectionVisualizer()
        
        logger.info(f"PDF report output directory: {output_dir}")
    
    def generate_report(self, image_path: str, result: Dict[str, Any], 
                       detection_id: str) -> str:
        """
        Generate a PDF report for a detection result
        
        Args:
            image_path: Path to the input image
            result: Detection result dictionary
            detection_id: Unique ID for the detection
            
        Returns:
            Path to the generated PDF report
        """
        try:
            # Create PDF object
            pdf = PDF()
            pdf.alias_nb_pages()
            pdf.add_page()
            
            # Add title
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Deepfake Detection Analysis Report', 0, 1, 'C')
            pdf.ln(5)
            
            # Add detection summary
            self._add_detection_summary(pdf, result)
            
            # Add visualization
            self._add_visualization(pdf, image_path, result, detection_id)
            
            # Add detailed analysis
            self._add_detailed_analysis(pdf, result)
            
            # Add technical details
            self._add_technical_details(pdf, result)
            
            # Add recommendations
            self._add_recommendations(pdf, result)
            
            # Save the PDF
            output_path = os.path.join(self.output_dir, f"{detection_id}_report.pdf")
            pdf.output(output_path)
            
            logger.info(f"Generated PDF report: {output_path}")
            
            return output_path
        
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            return None
    
    def _add_detection_summary(self, pdf: FPDF, result: Dict[str, Any]):
        """
        Add detection summary to the PDF
        
        Args:
            pdf: PDF object
            result: Detection result dictionary
        """
        # Add section title
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Detection Summary', 0, 1)
        
        # Add summary table
        pdf.set_font('Arial', '', 10)
        
        # Get result values
        probability = result.get("probability", 0)
        confidence = result.get("confidence", 0)
        model = result.get("model", "Unknown")
        processing_time = result.get("processingTime", 0)
        detection_type = result.get("detectionType", "image")
        
        # Determine verdict
        if probability > 0.7:
            verdict = "DEEPFAKE DETECTED"
            verdict_color = (255, 0, 0)  # Red
        elif probability > 0.4:
            verdict = "SUSPICIOUS CONTENT"
            verdict_color = (255, 128, 0)  # Orange
        else:
            verdict = "LIKELY AUTHENTIC"
            verdict_color = (0, 128, 0)  # Green
        
        # Add verdict box
        pdf.set_fill_color(verdict_color[0], verdict_color[1], verdict_color[2])
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, verdict, 1, 1, 'C', True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)
        
        # Create summary table
        col_width = 95
        row_height = 8
        
        # Row 1
        pdf.cell(col_width, row_height, f"Deepfake Probability: {probability:.2f}", 1)
        pdf.cell(col_width, row_height, f"Confidence: {confidence:.2f}", 1)
        pdf.ln()
        
        # Row 2
        pdf.cell(col_width, row_height, f"Model: {model}", 1)
        pdf.cell(col_width, row_height, f"Processing Time: {processing_time:.2f}ms", 1)
        pdf.ln()
        
        # Row 3
        pdf.cell(col_width, row_height, f"Detection Type: {detection_type}", 1)
        
        # Add ensemble info if available
        ensemble = result.get("ensemble", False)
        indian_enhancement = result.get("indianEnhancement", False)
        
        if ensemble or indian_enhancement:
            enhancements = []
            if ensemble:
                enhancements.append("Ensemble")
            if indian_enhancement:
                enhancements.append("Indian Enhancement")
            
            pdf.cell(col_width, row_height, f"Enhancements: {', '.join(enhancements)}", 1)
        else:
            pdf.cell(col_width, row_height, "Enhancements: None", 1)
        
        pdf.ln(15)
    
    def _add_visualization(self, pdf: FPDF, image_path: str, result: Dict[str, Any], detection_id: str):
        """
        Add visualization to the PDF
        
        Args:
            pdf: PDF object
            image_path: Path to the input image
            result: Detection result dictionary
            detection_id: Unique ID for the detection
        """
        # Add section title
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Visual Analysis', 0, 1)
        
        # Generate visualization
        vis_path = os.path.join(self.visualizer.output_dir, f"{detection_id}_visualization.jpg")
        if not os.path.exists(vis_path):
            self.visualizer.visualize_detection(image_path, result, detection_id)
        
        # Generate heatmap
        heatmap_path = os.path.join(self.visualizer.output_dir, f"{detection_id}_heatmap.jpg")
        if not os.path.exists(heatmap_path):
            self.visualizer.create_heatmap(image_path, result, detection_id)
        
        # Add visualization to PDF if it exists
        if os.path.exists(vis_path):
            # Add visualization title
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Detection Visualization', 0, 1)
            
            # Add visualization image
            pdf.image(vis_path, x=10, w=190)
            pdf.ln(5)
        
        # Add heatmap to PDF if it exists
        if os.path.exists(heatmap_path):
            # Add heatmap title
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Probability Heatmap', 0, 1)
            
            # Add heatmap image
            pdf.image(heatmap_path, x=10, w=190)
            pdf.ln(10)
    
    def _add_detailed_analysis(self, pdf: FPDF, result: Dict[str, Any]):
        """
        Add detailed analysis to the PDF
        
        Args:
            pdf: PDF object
            result: Detection result dictionary
        """
        # Add section title
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Detailed Analysis', 0, 1)
        
        # Add analysis text
        pdf.set_font('Arial', '', 10)
        
        # Get probability and confidence
        probability = result.get("probability", 0)
        confidence = result.get("confidence", 0)
        
        # Add analysis based on probability
        if probability > 0.7:
            analysis = (
                "The analyzed content has a high probability of being a deepfake. "
                "The detection model has identified significant indicators of manipulation "
                "that are consistent with AI-generated or altered media. "
                f"With a confidence score of {confidence:.2f}, this assessment has a high degree of reliability."
            )
        elif probability > 0.4:
            analysis = (
                "The analyzed content shows some indicators of possible manipulation, "
                "but the evidence is not conclusive. This could be a sophisticated deepfake "
                "or it might contain elements that are commonly misidentified as manipulated. "
                f"The confidence score of {confidence:.2f} reflects this uncertainty."
            )
        else:
            analysis = (
                "The analyzed content appears to be authentic with a low probability of being a deepfake. "
                "The detection model found minimal indicators of manipulation. "
                f"With a confidence score of {confidence:.2f}, this assessment is relatively reliable, "
                "though sophisticated deepfakes can sometimes evade detection."
            )
        
        pdf.multi_cell(0, 5, analysis)
        pdf.ln(5)
        
        # Add region analysis if available
        regions = result.get("regions", [])
        if regions:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, f"Region Analysis ({len(regions)} regions detected)", 0, 1)
            
            pdf.set_font('Arial', '', 10)
            
            for i, region in enumerate(regions):
                region_prob = region.get("probability", 0)
                region_conf = region.get("confidence", 0)
                
                # Get skin tone if available
                skin_tone = region.get("skin_tone", {})
                tone_info = ""
                if skin_tone and skin_tone.get("success"):
                    indian_tone = skin_tone.get("indian_tone", {})
                    tone_name = indian_tone.get("name", "Unknown")
                    tone_value = indian_tone.get("value", 0)
                    tone_info = f" | Skin Tone: {tone_name} ({tone_value:.2f})"
                
                region_text = (
                    f"Region {i+1}: Probability {region_prob:.2f} | "
                    f"Confidence {region_conf:.2f}{tone_info}"
                )
                
                pdf.cell(0, 7, region_text, 0, 1)
            
            pdf.ln(5)
        
        # Add artifacts analysis if available
        artifacts = result.get("artifacts", {})
        if artifacts:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, "Artifacts Analysis", 0, 1)
            
            pdf.set_font('Arial', '', 10)
            
            # Create artifacts table
            col_width = 95
            row_height = 8
            
            # Add artifact metrics
            metrics = [
                ("Edge Density", artifacts.get("edge_density", 0)),
                ("Noise Score", artifacts.get("noise_score", 0)),
                ("Compression Score", artifacts.get("compression_score", 0)),
                ("Inconsistency Score", artifacts.get("inconsistency_score", 0)),
                ("Overall Artifacts Score", artifacts.get("score", 0))
            ]
            
            for i in range(0, len(metrics), 2):
                pdf.cell(col_width, row_height, f"{metrics[i][0]}: {metrics[i][1]:.2f}", 1)
                
                if i+1 < len(metrics):
                    pdf.cell(col_width, row_height, f"{metrics[i+1][0]}: {metrics[i+1][1]:.2f}", 1)
                else:
                    pdf.cell(col_width, row_height, "", 1)
                
                pdf.ln()
            
            pdf.ln(5)
    
    def _add_technical_details(self, pdf: FPDF, result: Dict[str, Any]):
        """
        Add technical details to the PDF
        
        Args:
            pdf: PDF object
            result: Detection result dictionary
        """
        # Add section title
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Technical Details', 0, 1)
        
        # Add technical details
        pdf.set_font('Arial', '', 10)
        
        # Get model details
        model = result.get("model", "Unknown")
        ensemble = result.get("ensemble", False)
        indian_enhancement = result.get("indianEnhancement", False)
        
        # Create model description
        if model == "efficientnet":
            model_desc = (
                "EfficientNet is a convolutional neural network architecture that uses a compound "
                "scaling method to balance network depth, width, and resolution. It is optimized "
                "for both accuracy and computational efficiency."
            )
        elif model == "xception":
            model_desc = (
                "Xception is a deep convolutional neural network architecture that relies on "
                "depthwise separable convolutions. It is particularly effective at identifying "
                "spatial patterns that are common in deepfake manipulations."
            )
        elif model == "indian_specialized":
            model_desc = (
                "Indian Specialized is a custom model trained specifically for detecting deepfakes "
                "in Indian faces. It accounts for the unique characteristics of Indian skin tones "
                "and facial features to improve detection accuracy."
            )
        else:
            model_desc = "Custom detection model optimized for deepfake detection."
        
        pdf.multi_cell(0, 5, f"Primary Model: {model} - {model_desc}")
        pdf.ln(5)
        
        # Add ensemble information if used
        if ensemble:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, "Ensemble Detection", 0, 1)
            
            pdf.set_font('Arial', '', 10)
            ensemble_desc = (
                "Ensemble detection combines multiple models to improve accuracy. "
                "Each model specializes in different aspects of deepfake detection, "
                "and their results are weighted and combined for a more reliable assessment."
            )
            pdf.multi_cell(0, 5, ensemble_desc)
            pdf.ln(5)
        
        # Add Indian enhancement information if used
        if indian_enhancement:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, "Indian Face Enhancement", 0, 1)
            
            pdf.set_font('Arial', '', 10)
            enhancement_desc = (
                "Indian Face Enhancement is a specialized processing pipeline that accounts "
                "for the unique characteristics of Indian skin tones and facial features. "
                "It improves detection accuracy for Indian faces by adjusting the analysis "
                "parameters based on detected skin tone and facial structure."
            )
            pdf.multi_cell(0, 5, enhancement_desc)
            pdf.ln(5)
    
    def _add_recommendations(self, pdf: FPDF, result: Dict[str, Any]):
        """
        Add recommendations to the PDF
        
        Args:
            pdf: PDF object
            result: Detection result dictionary
        """
        # Add section title
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Recommendations', 0, 1)
        
        # Add recommendations based on probability
        pdf.set_font('Arial', '', 10)
        
        probability = result.get("probability", 0)
        
        if probability > 0.7:
            recommendations = [
                "Treat this content as likely manipulated or AI-generated.",
                "Do not share or distribute this content without clear disclosure of its nature.",
                "If this content claims to represent a real person or event, seek verification from additional sources.",
                "Consider reporting this content if it is being presented as authentic.",
                "For further verification, consider submitting this content to multiple deepfake detection services."
            ]
        elif probability > 0.4:
            recommendations = [
                "Approach this content with caution as it shows some indicators of possible manipulation.",
                "Verify the source and context of this content before sharing or making decisions based on it.",
                "Look for corroborating evidence from trusted sources if this content makes significant claims.",
                "Consider submitting this content for additional analysis if important decisions depend on its authenticity.",
                "Be transparent about the uncertain nature of this content if you choose to share it."
            ]
        else:
            recommendations = [
                "This content appears to be authentic, but maintain healthy skepticism as no detection system is perfect.",
                "Consider the source and context of the content as additional verification.",
                "Be aware that sophisticated deepfakes can sometimes evade detection.",
                "If this content is particularly sensitive or important, consider additional verification methods.",
                "Remember that detection technology continues to evolve alongside deepfake creation technology."
            ]
        
        # Add recommendations as bullet points
        for recommendation in recommendations:
            pdf.cell(5, 5, "•", 0, 0)
            pdf.multi_cell(0, 5, recommendation)
        
        pdf.ln(5)
        
        # Add disclaimer
        pdf.set_font('Arial', 'I', 8)
        disclaimer = (
            "Disclaimer: This report is generated by an automated system and should be used as one of multiple "
            "factors in determining content authenticity. No deepfake detection system is 100% accurate, and "
            "results should be interpreted with appropriate caution. This report does not constitute legal "
            "evidence or professional authentication."
        )
        pdf.multi_cell(0, 4, disclaimer)