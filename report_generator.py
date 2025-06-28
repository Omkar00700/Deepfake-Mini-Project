"""
Report Generator
Generates detailed reports for deepfake detection results
"""

import os
import json
import logging
import time
import datetime
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import io
import base64
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, results_dir="results", reports_dir="reports"):
        """
        Initialize the Report Generator
        
        Args:
            results_dir: Directory containing detection results
            reports_dir: Directory to save reports
        """
        self.results_dir = os.path.join(os.path.dirname(__file__), results_dir)
        self.reports_dir = os.path.join(os.path.dirname(__file__), reports_dir)
        
        # Create directories if they don't exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        logger.info(f"Report Generator initialized")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Reports directory: {self.reports_dir}")
    
    def generate_report(self, detection_id: str) -> Dict[str, Any]:
        """
        Generate a report for a detection
        
        Args:
            detection_id: ID of the detection
            
        Returns:
            Report generation result
        """
        try:
            # Get result file path
            result_path = os.path.join(self.results_dir, f"{detection_id}.json")
            
            if not os.path.exists(result_path):
                return {
                    "success": False,
                    "error": f"Result not found: {detection_id}"
                }
            
            # Load result
            with open(result_path, 'r') as f:
                result = json.load(f)
            
            # Generate report
            if result["detectionType"] == "image":
                report_path = self._generate_image_report(result)
            elif result["detectionType"] == "video":
                report_path = self._generate_video_report(result)
            else:
                return {
                    "success": False,
                    "error": f"Unknown detection type: {result['detectionType']}"
                }
            
            return {
                "success": True,
                "report_path": report_path,
                "detection_id": detection_id
            }
        
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_image_report(self, result: Dict[str, Any]) -> str:
        """
        Generate a report for an image detection
        
        Args:
            result: Detection result
            
        Returns:
            Path to the generated report
        """
        # Create report file path
        report_path = os.path.join(self.reports_dir, f"{result['id']}_report.pdf")
        
        # Create PDF
        with PdfPages(report_path) as pdf:
            # Add summary page
            self._add_summary_page(pdf, result)
            
            # Add detection details page
            self._add_detection_details_page(pdf, result)
            
            # Add regions page
            self._add_regions_page(pdf, result)
            
            # Add technical details page
            self._add_technical_details_page(pdf, result)
        
        logger.info(f"Generated image report: {report_path}")
        
        return report_path
    
    def _generate_video_report(self, result: Dict[str, Any]) -> str:
        """
        Generate a report for a video detection
        
        Args:
            result: Detection result
            
        Returns:
            Path to the generated report
        """
        # Create report file path
        report_path = os.path.join(self.reports_dir, f"{result['id']}_report.pdf")
        
        # Create PDF
        with PdfPages(report_path) as pdf:
            # Add summary page
            self._add_summary_page(pdf, result)
            
            # Add detection details page
            self._add_detection_details_page(pdf, result)
            
            # Add frames page
            self._add_frames_page(pdf, result)
            
            # Add regions page
            self._add_regions_page(pdf, result)
            
            # Add technical details page
            self._add_technical_details_page(pdf, result)
        
        logger.info(f"Generated video report: {report_path}")
        
        return report_path
    
    def _add_summary_page(self, pdf: PdfPages, result: Dict[str, Any]):
        """
        Add summary page to report
        
        Args:
            pdf: PDF document
            result: Detection result
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(8.5, 11))
        
        # Remove axes
        ax.axis('off')
        
        # Add title
        ax.text(0.5, 0.95, "Deepfake Detection Report", fontsize=24, ha='center', va='top', fontweight='bold')
        
        # Add subtitle
        detection_type = result["detectionType"].capitalize()
        ax.text(0.5, 0.9, f"{detection_type} Analysis", fontsize=18, ha='center', va='top')
        
        # Add timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.text(0.5, 0.85, f"Generated on: {timestamp}", fontsize=12, ha='center', va='top')
        
        # Add separator
        ax.axhline(y=0.83, xmin=0.1, xmax=0.9, color='black', linestyle='-', linewidth=1)
        
        # Add summary
        ax.text(0.1, 0.78, "Summary:", fontsize=16, ha='left', va='top', fontweight='bold')
        
        # Add filename
        ax.text(0.1, 0.74, f"Filename: {result['filename']}", fontsize=12, ha='left', va='top')
        
        # Add detection ID
        ax.text(0.1, 0.71, f"Detection ID: {result['id']}", fontsize=12, ha='left', va='top')
        
        # Add verdict
        verdict = "DEEPFAKE" if result["probability"] > 0.5 else "REAL"
        verdict_color = "red" if result["probability"] > 0.5 else "green"
        ax.text(0.1, 0.67, f"Verdict: {verdict}", fontsize=16, ha='left', va='top', fontweight='bold', color=verdict_color)
        
        # Add probability
        ax.text(0.1, 0.63, f"Probability: {result['probability']:.2%}", fontsize=12, ha='left', va='top')
        
        # Add confidence
        ax.text(0.1, 0.6, f"Confidence: {result['confidence']:.2%}", fontsize=12, ha='left', va='top')
        
        # Add processing time
        ax.text(0.1, 0.57, f"Processing Time: {result['processingTime']:.2f} seconds", fontsize=12, ha='left', va='top')
        
        # Add model information
        ax.text(0.1, 0.53, f"Model: {result['model']}", fontsize=12, ha='left', va='top')
        ax.text(0.1, 0.5, f"Ensemble: {'Yes' if result['ensemble'] else 'No'}", fontsize=12, ha='left', va='top')
        ax.text(0.1, 0.47, f"Indian Enhancement: {'Yes' if result['indianEnhancement'] else 'No'}", fontsize=12, ha='left', va='top')
        
        # Add regions count
        regions_count = len(result.get("regions", []))
        ax.text(0.1, 0.43, f"Detected Faces: {regions_count}", fontsize=12, ha='left', va='top')
        
        # Add video-specific information if applicable
        if result["detectionType"] == "video":
            ax.text(0.1, 0.39, f"Duration: {result['duration']:.2f} seconds", fontsize=12, ha='left', va='top')
            ax.text(0.1, 0.36, f"FPS: {result['fps']:.2f}", fontsize=12, ha='left', va='top')
            ax.text(0.1, 0.33, f"Processed Frames: {result['frameCount']}", fontsize=12, ha='left', va='top')
        
        # Add probability gauge
        self._add_probability_gauge(ax, result["probability"], 0.6, 0.65, 0.3)
        
        # Add footer
        ax.text(0.5, 0.05, "Indian Deepfake Detector", fontsize=10, ha='center', va='bottom')
        ax.text(0.5, 0.03, "Confidential - For authorized use only", fontsize=8, ha='center', va='bottom')
        
        # Save page
        pdf.savefig(fig)
        plt.close(fig)
    
    def _add_detection_details_page(self, pdf: PdfPages, result: Dict[str, Any]):
        """
        Add detection details page to report
        
        Args:
            pdf: PDF document
            result: Detection result
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(8.5, 11))
        
        # Remove axes
        ax.axis('off')
        
        # Add title
        ax.text(0.5, 0.95, "Detection Details", fontsize=20, ha='center', va='top', fontweight='bold')
        
        # Add separator
        ax.axhline(y=0.93, xmin=0.1, xmax=0.9, color='black', linestyle='-', linewidth=1)
        
        # Add detection information
        ax.text(0.1, 0.9, "Detection Information:", fontsize=16, ha='left', va='top', fontweight='bold')
        
        # Add detection type
        ax.text(0.1, 0.86, f"Type: {result['detectionType'].capitalize()}", fontsize=12, ha='left', va='top')
        
        # Add detection time
        processing_time = result.get("processingTime", 0)
        ax.text(0.1, 0.83, f"Processing Time: {processing_time:.2f} seconds", fontsize=12, ha='left', va='top')
        
        # Add model information
        ax.text(0.1, 0.79, "Model Information:", fontsize=16, ha='left', va='top', fontweight='bold')
        
        # Add model name
        ax.text(0.1, 0.75, f"Primary Model: {result['model']}", fontsize=12, ha='left', va='top')
        
        # Add ensemble information
        ensemble_text = "Yes" if result.get("ensemble", False) else "No"
        ax.text(0.1, 0.72, f"Ensemble: {ensemble_text}", fontsize=12, ha='left', va='top')
        
        # Add Indian enhancement information
        enhancement_text = "Yes" if result.get("indianEnhancement", False) else "No"
        ax.text(0.1, 0.69, f"Indian Enhancement: {enhancement_text}", fontsize=12, ha='left', va='top')
        
        # Add detection results
        ax.text(0.1, 0.65, "Detection Results:", fontsize=16, ha='left', va='top', fontweight='bold')
        
        # Add probability
        ax.text(0.1, 0.61, f"Probability: {result['probability']:.2%}", fontsize=12, ha='left', va='top')
        
        # Add confidence
        ax.text(0.1, 0.58, f"Confidence: {result['confidence']:.2%}", fontsize=12, ha='left', va='top')
        
        # Add verdict
        verdict = "DEEPFAKE" if result["probability"] > 0.5 else "REAL"
        verdict_color = "red" if result["probability"] > 0.5 else "green"
        ax.text(0.1, 0.54, f"Verdict: {verdict}", fontsize=16, ha='left', va='top', fontweight='bold', color=verdict_color)
        
        # Add interpretation
        ax.text(0.1, 0.5, "Interpretation:", fontsize=16, ha='left', va='top', fontweight='bold')
        
        if result["probability"] > 0.8:
            interpretation = "High confidence that this is a deepfake."
        elif result["probability"] > 0.6:
            interpretation = "Moderate confidence that this is a deepfake."
        elif result["probability"] > 0.4:
            interpretation = "Uncertain, but leaning towards deepfake."
        elif result["probability"] > 0.2:
            interpretation = "Moderate confidence that this is real."
        else:
            interpretation = "High confidence that this is real."
        
        ax.text(0.1, 0.46, interpretation, fontsize=12, ha='left', va='top')
        
        # Add confidence explanation
        confidence_explanation = (
            "Confidence indicates how certain the model is about its prediction. "
            "Higher confidence means the model is more certain about the result."
        )
        ax.text(0.1, 0.42, confidence_explanation, fontsize=10, ha='left', va='top', style='italic')
        
        # Add probability distribution
        self._add_probability_distribution(ax, result["probability"], 0.1, 0.25, 0.8, 0.15)
        
        # Add footer
        ax.text(0.5, 0.05, "Indian Deepfake Detector", fontsize=10, ha='center', va='bottom')
        ax.text(0.5, 0.03, f"Detection ID: {result['id']}", fontsize=8, ha='center', va='bottom')
        
        # Save page
        pdf.savefig(fig)
        plt.close(fig)
    
    def _add_regions_page(self, pdf: PdfPages, result: Dict[str, Any]):
        """
        Add regions page to report
        
        Args:
            pdf: PDF document
            result: Detection result
        """
        # Get regions
        regions = result.get("regions", [])
        
        if not regions:
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8.5, 11))
        
        # Remove axes
        ax.axis('off')
        
        # Add title
        ax.text(0.5, 0.95, "Detected Faces", fontsize=20, ha='center', va='top', fontweight='bold')
        
        # Add separator
        ax.axhline(y=0.93, xmin=0.1, xmax=0.9, color='black', linestyle='-', linewidth=1)
        
        # Add regions count
        ax.text(0.1, 0.9, f"Total Faces: {len(regions)}", fontsize=16, ha='left', va='top')
        
        # Calculate max regions per page
        max_regions_per_page = 5
        
        # Calculate number of pages needed
        num_pages = (len(regions) + max_regions_per_page - 1) // max_regions_per_page
        
        # Add regions to pages
        for page in range(num_pages):
            if page > 0:
                # Create new page
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                ax.text(0.5, 0.95, f"Detected Faces (Page {page+1}/{num_pages})", fontsize=20, ha='center', va='top', fontweight='bold')
                ax.axhline(y=0.93, xmin=0.1, xmax=0.9, color='black', linestyle='-', linewidth=1)
            
            # Get regions for this page
            start_idx = page * max_regions_per_page
            end_idx = min(start_idx + max_regions_per_page, len(regions))
            page_regions = regions[start_idx:end_idx]
            
            # Add regions
            for i, region in enumerate(page_regions):
                # Calculate position
                y_pos = 0.85 - (i * 0.15)
                
                # Add region number
                region_idx = start_idx + i + 1
                ax.text(0.1, y_pos, f"Face {region_idx}:", fontsize=14, ha='left', va='top', fontweight='bold')
                
                # Add probability
                probability = region.get("probability", 0)
                verdict = "DEEPFAKE" if probability > 0.5 else "REAL"
                verdict_color = "red" if probability > 0.5 else "green"
                ax.text(0.25, y_pos, f"Verdict: {verdict}", fontsize=12, ha='left', va='top', color=verdict_color)
                
                # Add confidence
                confidence = region.get("confidence", 0)
                ax.text(0.45, y_pos, f"Confidence: {confidence:.2%}", fontsize=12, ha='left', va='top')
                
                # Add skin tone if available
                skin_tone = region.get("skin_tone", {})
                if skin_tone and skin_tone.get("success", False):
                    indian_tone = skin_tone.get("indian_tone", {})
                    if indian_tone:
                        tone_name = indian_tone.get("name", "Unknown")
                        ax.text(0.65, y_pos, f"Skin Tone: {tone_name}", fontsize=12, ha='left', va='top')
                
                # Add box information
                box = region.get("box", [0, 0, 0, 0])
                ax.text(0.1, y_pos - 0.04, f"Position: x={box[0]}, y={box[1]}, width={box[2]}, height={box[3]}", fontsize=10, ha='left', va='top')
                
                # Add mini probability gauge
                self._add_probability_gauge(ax, probability, 0.85, y_pos - 0.02, 0.1)
            
            # Add footer
            ax.text(0.5, 0.05, "Indian Deepfake Detector", fontsize=10, ha='center', va='bottom')
            ax.text(0.5, 0.03, f"Detection ID: {result['id']}", fontsize=8, ha='center', va='bottom')
            
            # Save page
            pdf.savefig(fig)
            plt.close(fig)
    
    def _add_frames_page(self, pdf: PdfPages, result: Dict[str, Any]):
        """
        Add frames page to report for video detection
        
        Args:
            pdf: PDF document
            result: Detection result
        """
        # Get frame results
        frame_results = result.get("frameResults", [])
        
        if not frame_results:
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8.5, 11))
        
        # Remove axes
        ax.axis('off')
        
        # Add title
        ax.text(0.5, 0.95, "Frame Analysis", fontsize=20, ha='center', va='top', fontweight='bold')
        
        # Add separator
        ax.axhline(y=0.93, xmin=0.1, xmax=0.9, color='black', linestyle='-', linewidth=1)
        
        # Add frames count
        ax.text(0.1, 0.9, f"Total Frames Analyzed: {len(frame_results)}", fontsize=16, ha='left', va='top')
        
        # Add video information
        ax.text(0.1, 0.86, f"Video Duration: {result.get('duration', 0):.2f} seconds", fontsize=12, ha='left', va='top')
        ax.text(0.1, 0.83, f"FPS: {result.get('fps', 0):.2f}", fontsize=12, ha='left', va='top')
        
        # Create frame probability data
        frame_numbers = []
        frame_probabilities = []
        frame_timestamps = []
        
        for frame in frame_results:
            frame_number = frame.get("frame", 0)
            frame_timestamp = frame.get("timestamp", 0)
            
            # Calculate average probability for this frame
            regions = frame.get("regions", [])
            if regions:
                avg_probability = sum(r.get("probability", 0) for r in regions) / len(regions)
            else:
                avg_probability = 0
            
            frame_numbers.append(frame_number)
            frame_probabilities.append(avg_probability)
            frame_timestamps.append(frame_timestamp)
        
        # Add frame probability plot
        ax_plot = fig.add_axes([0.1, 0.4, 0.8, 0.35])
        ax_plot.plot(frame_timestamps, frame_probabilities, 'b-', marker='o', markersize=4)
        ax_plot.set_xlabel('Time (seconds)')
        ax_plot.set_ylabel('Deepfake Probability')
        ax_plot.set_title('Deepfake Probability Over Time')
        ax_plot.grid(True, linestyle='--', alpha=0.7)
        ax_plot.set_ylim(0, 1)
        
        # Add threshold line
        ax_plot.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
        ax_plot.text(frame_timestamps[-1], 0.5, 'Threshold', ha='right', va='bottom', color='r')
        
        # Add frame summary
        ax.text(0.1, 0.35, "Frame Summary:", fontsize=16, ha='left', va='top', fontweight='bold')
        
        # Count frames above threshold
        frames_above_threshold = sum(1 for p in frame_probabilities if p > 0.5)
        percentage_above = frames_above_threshold / len(frame_probabilities) * 100 if frame_probabilities else 0
        
        ax.text(0.1, 0.31, f"Frames Classified as Deepfake: {frames_above_threshold} ({percentage_above:.1f}%)", 
                fontsize=12, ha='left', va='top')
        
        # Add interpretation
        ax.text(0.1, 0.27, "Interpretation:", fontsize=14, ha='left', va='top', fontweight='bold')
        
        if percentage_above > 80:
            interpretation = "Strong evidence of deepfake throughout the video."
        elif percentage_above > 50:
            interpretation = "Moderate evidence of deepfake in majority of the video."
        elif percentage_above > 20:
            interpretation = "Some evidence of deepfake in parts of the video."
        else:
            interpretation = "Little to no evidence of deepfake in the video."
        
        ax.text(0.1, 0.23, interpretation, fontsize=12, ha='left', va='top')
        
        # Add note about frame analysis
        note = (
            "Note: The graph shows the average deepfake probability for each analyzed frame. "
            "Values above 0.5 (red dashed line) indicate frames classified as deepfake."
        )
        ax.text(0.1, 0.18, note, fontsize=10, ha='left', va='top', style='italic')
        
        # Add footer
        ax.text(0.5, 0.05, "Indian Deepfake Detector", fontsize=10, ha='center', va='bottom')
        ax.text(0.5, 0.03, f"Detection ID: {result['id']}", fontsize=8, ha='center', va='bottom')
        
        # Save page
        pdf.savefig(fig)
        plt.close(fig)
    
    def _add_technical_details_page(self, pdf: PdfPages, result: Dict[str, Any]):
        """
        Add technical details page to report
        
        Args:
            pdf: PDF document
            result: Detection result
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(8.5, 11))
        
        # Remove axes
        ax.axis('off')
        
        # Add title
        ax.text(0.5, 0.95, "Technical Details", fontsize=20, ha='center', va='top', fontweight='bold')
        
        # Add separator
        ax.axhline(y=0.93, xmin=0.1, xmax=0.9, color='black', linestyle='-', linewidth=1)
        
        # Add detection information
        ax.text(0.1, 0.9, "Detection Information:", fontsize=16, ha='left', va='top', fontweight='bold')
        
        # Add detection ID
        ax.text(0.1, 0.86, f"Detection ID: {result['id']}", fontsize=12, ha='left', va='top')
        
        # Add timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.text(0.1, 0.83, f"Report Generated: {timestamp}", fontsize=12, ha='left', va='top')
        
        # Add file information
        ax.text(0.1, 0.79, "File Information:", fontsize=16, ha='left', va='top', fontweight='bold')
        
        # Add filename
        ax.text(0.1, 0.75, f"Filename: {result['filename']}", fontsize=12, ha='left', va='top')
        
        # Add detection type
        ax.text(0.1, 0.72, f"Type: {result['detectionType'].capitalize()}", fontsize=12, ha='left', va='top')
        
        # Add model information
        ax.text(0.1, 0.68, "Model Information:", fontsize=16, ha='left', va='top', fontweight='bold')
        
        # Add model name
        ax.text(0.1, 0.64, f"Primary Model: {result['model']}", fontsize=12, ha='left', va='top')
        
        # Add ensemble information
        ensemble_text = "Yes" if result.get("ensemble", False) else "No"
        ax.text(0.1, 0.61, f"Ensemble: {ensemble_text}", fontsize=12, ha='left', va='top')
        
        # Add Indian enhancement information
        enhancement_text = "Yes" if result.get("indianEnhancement", False) else "No"
        ax.text(0.1, 0.58, f"Indian Enhancement: {enhancement_text}", fontsize=12, ha='left', va='top')
        
        # Add processing information
        ax.text(0.1, 0.54, "Processing Information:", fontsize=16, ha='left', va='top', fontweight='bold')
        
        # Add processing time
        processing_time = result.get("processingTime", 0)
        ax.text(0.1, 0.5, f"Processing Time: {processing_time:.2f} seconds", fontsize=12, ha='left', va='top')
        
        # Add regions count
        regions_count = len(result.get("regions", []))
        ax.text(0.1, 0.47, f"Detected Faces: {regions_count}", fontsize=12, ha='left', va='top')
        
        # Add video-specific information if applicable
        if result["detectionType"] == "video":
            ax.text(0.1, 0.43, f"Duration: {result['duration']:.2f} seconds", fontsize=12, ha='left', va='top')
            ax.text(0.1, 0.4, f"FPS: {result['fps']:.2f}", fontsize=12, ha='left', va='top')
            ax.text(0.1, 0.37, f"Processed Frames: {result['frameCount']}", fontsize=12, ha='left', va='top')
        
        # Add detection methodology
        ax.text(0.1, 0.33, "Detection Methodology:", fontsize=16, ha='left', va='top', fontweight='bold')
        
        methodology_text = (
            "This detection uses deep learning models specialized for Indian faces. "
            "The system analyzes facial features, inconsistencies, and artifacts that "
            "are characteristic of deepfakes. For Indian faces, special enhancements "
            "are applied to account for different skin tones and facial features."
        )
        
        # Wrap text
        wrapped_text = self._wrap_text(methodology_text, 70)
        for i, line in enumerate(wrapped_text):
            ax.text(0.1, 0.29 - (i * 0.03), line, fontsize=10, ha='left', va='top')
        
        # Add disclaimer
        ax.text(0.1, 0.15, "Disclaimer:", fontsize=16, ha='left', va='top', fontweight='bold')
        
        disclaimer_text = (
            "This report is generated by an automated system and should be used for "
            "informational purposes only. While the system is designed to be accurate, "
            "it may not detect all deepfakes, especially sophisticated ones. The results "
            "should be verified by human experts for critical applications."
        )
        
        # Wrap text
        wrapped_disclaimer = self._wrap_text(disclaimer_text, 70)
        for i, line in enumerate(wrapped_disclaimer):
            ax.text(0.1, 0.11 - (i * 0.03), line, fontsize=10, ha='left', va='top', style='italic')
        
        # Add footer
        ax.text(0.5, 0.05, "Indian Deepfake Detector", fontsize=10, ha='center', va='bottom')
        ax.text(0.5, 0.03, "Confidential - For authorized use only", fontsize=8, ha='center', va='bottom')
        
        # Save page
        pdf.savefig(fig)
        plt.close(fig)
    
    def _add_probability_gauge(self, ax, probability, x, y, size):
        """
        Add a probability gauge to the figure
        
        Args:
            ax: Matplotlib axes
            probability: Probability value (0-1)
            x: X position (0-1)
            y: Y position (0-1)
            size: Size of the gauge (0-1)
        """
        # Create gauge
        gauge_ax = ax.figure.add_axes([x, y, size, size], projection='polar')
        
        # Configure gauge
        gauge_ax.set_theta_direction(-1)
        gauge_ax.set_theta_offset(np.pi / 2.0)
        gauge_ax.set_rlim(0, 1)
        gauge_ax.set_rticks([])
        gauge_ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
        gauge_ax.set_xticklabels(['0', '', '0.25', '', '0.5', '', '0.75', ''])
        gauge_ax.grid(True)
        
        # Add colored regions
        gauge_ax.bar(np.linspace(0, np.pi, 100), [1]*100, width=np.pi/100, color='green', alpha=0.3)
        gauge_ax.bar(np.linspace(np.pi, 2*np.pi, 100), [1]*100, width=np.pi/100, color='red', alpha=0.3)
        
        # Add needle
        needle_angle = 2 * np.pi * probability
        gauge_ax.plot([0, needle_angle], [0, 0.8], 'k-', linewidth=2)
        gauge_ax.plot([0], [0], 'ko', markersize=5)
        
        # Add probability text
        gauge_ax.text(0, -0.2, f"{probability:.2%}", ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Add labels
        gauge_ax.text(-np.pi/4, 1.2, "REAL", ha='center', va='center', fontsize=8, color='green')
        gauge_ax.text(5*np.pi/4, 1.2, "FAKE", ha='center', va='center', fontsize=8, color='red')
    
    def _add_probability_distribution(self, ax, probability, x, y, width, height):
        """
        Add a probability distribution to the figure
        
        Args:
            ax: Matplotlib axes
            probability: Probability value (0-1)
            x: X position (0-1)
            y: Y position (0-1)
            width: Width of the distribution (0-1)
            height: Height of the distribution (0-1)
        """
        # Create distribution axes
        dist_ax = ax.figure.add_axes([x, y, width, height])
        
        # Configure axes
        dist_ax.set_xlim(0, 1)
        dist_ax.set_ylim(0, 1)
        dist_ax.set_xlabel('Probability')
        dist_ax.set_ylabel('Confidence')
        dist_ax.set_title('Deepfake Probability Distribution')
        
        # Add colored regions
        dist_ax.axvspan(0, 0.5, color='green', alpha=0.2, label='Real')
        dist_ax.axvspan(0.5, 1, color='red', alpha=0.2, label='Fake')
        
        # Add threshold line
        dist_ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
        
        # Create distribution curve
        x_vals = np.linspace(0, 1, 1000)
        
        # Create a normal distribution centered at the probability
        sigma = 0.1
        y_vals = np.exp(-0.5 * ((x_vals - probability) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        y_vals = y_vals / np.max(y_vals)  # Normalize to [0, 1]
        
        # Plot distribution
        dist_ax.plot(x_vals, y_vals, 'b-', linewidth=2)
        
        # Add marker for the probability
        dist_ax.plot([probability], [0], 'ro', markersize=8)
        
        # Add legend
        dist_ax.legend(loc='upper right')
    
    def _wrap_text(self, text, width):
        """
        Wrap text to a specified width
        
        Args:
            text: Text to wrap
            width: Maximum width in characters
            
        Returns:
            List of wrapped lines
        """
        import textwrap
        return textwrap.wrap(text, width)
    
    def generate_html_report(self, detection_id: str) -> Dict[str, Any]:
        """
        Generate an HTML report for a detection
        
        Args:
            detection_id: ID of the detection
            
        Returns:
            Report generation result with HTML content
        """
        try:
            # Get result file path
            result_path = os.path.join(self.results_dir, f"{detection_id}.json")
            
            if not os.path.exists(result_path):
                return {
                    "success": False,
                    "error": f"Result not found: {detection_id}"
                }
            
            # Load result
            with open(result_path, 'r') as f:
                result = json.load(f)
            
            # Generate HTML report
            html_content = self._generate_html_content(result)
            
            # Create report file path
            report_path = os.path.join(self.reports_dir, f"{detection_id}_report.html")
            
            # Write HTML to file
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Generated HTML report: {report_path}")
            
            return {
                "success": True,
                "report_path": report_path,
                "detection_id": detection_id,
                "html_content": html_content
            }
        
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_html_content(self, result: Dict[str, Any]) -> str:
        """
        Generate HTML content for a report
        
        Args:
            result: Detection result
            
        Returns:
            HTML content
        """
        # Determine verdict
        verdict = "DEEPFAKE" if result["probability"] > 0.5 else "REAL"
        verdict_color = "red" if result["probability"] > 0.5 else "green"
        
        # Create HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Deepfake Detection Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1000px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 1px solid #eee;
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                }}
                .verdict {{
                    font-size: 24px;
                    font-weight: bold;
                    color: {verdict_color};
                    text-align: center;
                    padding: 10px;
                    margin: 20px 0;
                    border: 2px solid {verdict_color};
                    border-radius: 5px;
                }}
                .probability-bar {{
                    height: 30px;
                    background-color: #eee;
                    border-radius: 15px;
                    margin: 20px 0;
                    overflow: hidden;
                    position: relative;
                }}
                .probability-fill {{
                    height: 100%;
                    background: linear-gradient(to right, green, yellow, red);
                    width: {result["probability"] * 100}%;
                }}
                .probability-marker {{
                    position: absolute;
                    top: 0;
                    left: 50%;
                    height: 100%;
                    width: 2px;
                    background-color: #333;
                }}
                .probability-text {{
                    position: absolute;
                    top: 5px;
                    left: {result["probability"] * 100}%;
                    transform: translateX(-50%);
                    color: #000;
                    font-weight: bold;
                }}
                .info-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                }}
                .info-item {{
                    margin-bottom: 10px;
                }}
                .info-label {{
                    font-weight: bold;
                }}
                .regions-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                .regions-table th, .regions-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .regions-table th {{
                    background-color: #f2f2f2;
                }}
                .regions-table tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #eee;
                    font-size: 12px;
                    color: #777;
                }}
                .disclaimer {{
                    font-style: italic;
                    font-size: 12px;
                    color: #777;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Deepfake Detection Report</h1>
                <p>Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <div class="verdict">{verdict}</div>
                
                <div class="probability-bar">
                    <div class="probability-fill"></div>
                    <div class="probability-marker"></div>
                    <div class="probability-text">{result["probability"]:.2%}</div>
                </div>
                
                <div class="info-grid">
                    <div class="info-item">
                        <span class="info-label">Filename:</span> {result["filename"]}
                    </div>
                    <div class="info-item">
                        <span class="info-label">Detection ID:</span> {result["id"]}
                    </div>
                    <div class="info-item">
                        <span class="info-label">Probability:</span> {result["probability"]:.2%}
                    </div>
                    <div class="info-item">
                        <span class="info-label">Confidence:</span> {result["confidence"]:.2%}
                    </div>
                    <div class="info-item">
                        <span class="info-label">Processing Time:</span> {result["processingTime"]:.2f} seconds
                    </div>
                    <div class="info-item">
                        <span class="info-label">Model:</span> {result["model"]}
                    </div>
                    <div class="info-item">
                        <span class="info-label">Ensemble:</span> {"Yes" if result["ensemble"] else "No"}
                    </div>
                    <div class="info-item">
                        <span class="info-label">Indian Enhancement:</span> {"Yes" if result["indianEnhancement"] else "No"}
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Detection Details</h2>
                <p>
                    <strong>Interpretation:</strong> 
        """
        
        # Add interpretation
        if result["probability"] > 0.8:
            html += "High confidence that this is a deepfake."
        elif result["probability"] > 0.6:
            html += "Moderate confidence that this is a deepfake."
        elif result["probability"] > 0.4:
            html += "Uncertain, but leaning towards deepfake."
        elif result["probability"] > 0.2:
            html += "Moderate confidence that this is real."
        else:
            html += "High confidence that this is real."
        
        html += """
                </p>
                <p>
                    <strong>Confidence Explanation:</strong> 
                    Confidence indicates how certain the model is about its prediction. 
                    Higher confidence means the model is more certain about the result.
                </p>
            </div>
        """
        
        # Add regions section if available
        regions = result.get("regions", [])
        if regions:
            html += """
            <div class="section">
                <h2>Detected Faces</h2>
                <p>Total Faces: {}</p>
                
                <table class="regions-table">
                    <tr>
                        <th>Face #</th>
                        <th>Verdict</th>
                        <th>Probability</th>
                        <th>Confidence</th>
                        <th>Skin Tone</th>
                    </tr>
            """.format(len(regions))
            
            for i, region in enumerate(regions):
                probability = region.get("probability", 0)
                verdict = "DEEPFAKE" if probability > 0.5 else "REAL"
                verdict_color = "red" if probability > 0.5 else "green"
                
                # Get skin tone if available
                skin_tone = region.get("skin_tone", {})
                tone_name = "Unknown"
                if skin_tone and skin_tone.get("success", False):
                    indian_tone = skin_tone.get("indian_tone", {})
                    if indian_tone:
                        tone_name = indian_tone.get("name", "Unknown")
                
                html += f"""
                    <tr>
                        <td>{i+1}</td>
                        <td style="color: {verdict_color}; font-weight: bold;">{verdict}</td>
                        <td>{probability:.2%}</td>
                        <td>{region.get("confidence", 0):.2%}</td>
                        <td>{tone_name}</td>
                    </tr>
                """
            
            html += """
                </table>
            </div>
            """
        
        # Add video-specific section if applicable
        if result["detectionType"] == "video":
            html += f"""
            <div class="section">
                <h2>Video Analysis</h2>
                <div class="info-grid">
                    <div class="info-item">
                        <span class="info-label">Duration:</span> {result.get("duration", 0):.2f} seconds
                    </div>
                    <div class="info-item">
                        <span class="info-label">FPS:</span> {result.get("fps", 0):.2f}
                    </div>
                    <div class="info-item">
                        <span class="info-label">Processed Frames:</span> {result.get("frameCount", 0)}
                    </div>
                </div>
            </div>
            """
        
        # Add technical details
        html += """
            <div class="section">
                <h2>Technical Details</h2>
                <p>
                    <strong>Detection Methodology:</strong> 
                    This detection uses deep learning models specialized for Indian faces. 
                    The system analyzes facial features, inconsistencies, and artifacts that 
                    are characteristic of deepfakes. For Indian faces, special enhancements 
                    are applied to account for different skin tones and facial features.
                </p>
                
                <div class="disclaimer">
                    <strong>Disclaimer:</strong> 
                    This report is generated by an automated system and should be used for 
                    informational purposes only. While the system is designed to be accurate, 
                    it may not detect all deepfakes, especially sophisticated ones. The results 
                    should be verified by human experts for critical applications.
                </div>
            </div>
            
            <div class="footer">
                <p>Indian Deepfake Detector</p>
                <p>Confidential - For authorized use only</p>
                <p>Detection ID: {}</p>
            </div>
        </body>
        </html>
        """.format(result["id"])
        
        return html

# Test the report generator
if __name__ == "__main__":
    # Create report generator
    generator = ReportGenerator()
    
    # Get all result files
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if os.path.exists(results_dir):
        result_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
        
        if result_files:
            # Generate report for the first result
            detection_id = os.path.splitext(result_files[0])[0]
            result = generator.generate_report(detection_id)
            
            if result["success"]:
                print(f"Generated report: {result['report_path']}")
            else:
                print(f"Error generating report: {result.get('error', 'Unknown error')}")
        else:
            print("No result files found")
    else:
        print(f"Results directory not found: {results_dir}")