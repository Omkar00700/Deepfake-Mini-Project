"""
Enhanced PDF Report Generator for DeepDefend
Creates visually appealing and informative reports for deepfake detection results
"""

import os
import logging
import json
import time
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import base64
from io import BytesIO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import reportlab for PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    Image, PageBreak, ListFlowable, ListItem, Flowable, Frame, 
    NextPageTemplate, PageTemplate
)
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.widgets.markers import makeMarker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedPDFReportGenerator:
    """
    Enhanced PDF Report Generator for DeepDefend
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the PDF report generator
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize styles
        self.styles = getSampleStyleSheet()
        
        # Add custom styles
        self.styles.add(ParagraphStyle(
            name='Title',
            parent=self.styles['Heading1'],
            fontSize=24,
            leading=30,
            alignment=1,  # Center
            spaceAfter=20
        ))
        
        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Heading2'],
            fontSize=18,
            leading=22,
            alignment=1,  # Center
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            leading=20,
            spaceBefore=15,
            spaceAfter=10
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubsectionTitle',
            parent=self.styles['Heading3'],
            fontSize=14,
            leading=18,
            spaceBefore=10,
            spaceAfter=8
        ))
        
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=11,
            leading=14,
            spaceBefore=6,
            spaceAfter=6
        ))
        
        self.styles.add(ParagraphStyle(
            name='Caption',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=12,
            alignment=1,  # Center
            spaceBefore=4,
            spaceAfter=10,
            textColor=colors.gray
        ))
        
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            leading=10,
            alignment=1,  # Center
            textColor=colors.gray
        ))
        
        # Define colors
        self.colors = {
            'primary': colors.HexColor('#1a73e8'),
            'secondary': colors.HexColor('#4285f4'),
            'accent': colors.HexColor('#fbbc04'),
            'success': colors.HexColor('#34a853'),
            'warning': colors.HexColor('#fbbc04'),
            'danger': colors.HexColor('#ea4335'),
            'light': colors.HexColor('#f8f9fa'),
            'dark': colors.HexColor('#202124'),
            'gray': colors.HexColor('#5f6368')
        }
        
        logger.info(f"Initialized enhanced PDF report generator with output directory: {output_dir}")
    
    def generate_report(self, detection_result: Dict[str, Any], 
                       image_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive PDF report for a detection result
        
        Args:
            detection_result: Detection result dictionary
            image_path: Optional path to the original image
            
        Returns:
            Path to the generated PDF report
        """
        # Create a unique filename for the report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"deepfake_report_{timestamp}.pdf"
        output_path = os.path.join(self.output_dir, filename)
        
        # Create the document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Create story (content)
        story = []
        
        # Add header
        self._add_header(story, detection_result)
        
        # Add summary
        self._add_summary(story, detection_result)
        
        # Add image if provided
        if image_path and os.path.exists(image_path):
            self._add_image_analysis(story, image_path, detection_result)
        
        # Add detailed analysis
        self._add_detailed_analysis(story, detection_result)
        
        # Add skin tone analysis if available
        if self._has_skin_tone_data(detection_result):
            self._add_skin_tone_analysis(story, detection_result)
        
        # Add GAN detection results if available
        if self._has_gan_detection_data(detection_result):
            self._add_gan_detection_analysis(story, detection_result)
        
        # Add methodology section
        self._add_methodology(story)
        
        # Add disclaimer
        self._add_disclaimer(story)
        
        # Build the document
        doc.build(
            story,
            onFirstPage=self._add_first_page_header,
            onLaterPages=self._add_later_pages_header
        )
        
        logger.info(f"Generated enhanced PDF report: {output_path}")
        
        return output_path
    
    def _add_first_page_header(self, canvas, doc):
        """Add header to the first page"""
        canvas.saveState()
        
        # Add logo
        logo_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'assets', 'logo.png')
        if os.path.exists(logo_path):
            canvas.drawImage(logo_path, 72, letter[1] - 54, width=1.5*inch, height=0.5*inch, preserveAspectRatio=True)
        
        # Add report title
        canvas.setFont('Helvetica-Bold', 10)
        canvas.drawRightString(letter[0] - 72, letter[1] - 36, "DeepDefend Analysis Report")
        
        # Add date
        canvas.setFont('Helvetica', 8)
        canvas.drawRightString(letter[0] - 72, letter[1] - 48, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Add footer
        canvas.setFont('Helvetica', 8)
        canvas.drawCentredString(letter[0] / 2, 30, "DeepDefend - Advanced Deepfake Detection Platform")
        canvas.drawCentredString(letter[0] / 2, 20, "Page 1")
        
        canvas.restoreState()
    
    def _add_later_pages_header(self, canvas, doc):
        """Add header to later pages"""
        canvas.saveState()
        
        # Add mini logo
        logo_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'assets', 'logo.png')
        if os.path.exists(logo_path):
            canvas.drawImage(logo_path, 72, letter[1] - 36, width=1*inch, height=0.33*inch, preserveAspectRatio=True)
        
        # Add report title
        canvas.setFont('Helvetica', 8)
        canvas.drawRightString(letter[0] - 72, letter[1] - 30, "DeepDefend Analysis Report")
        
        # Add footer
        canvas.setFont('Helvetica', 8)
        canvas.drawCentredString(letter[0] / 2, 30, "DeepDefend - Advanced Deepfake Detection Platform")
        canvas.drawCentredString(letter[0] / 2, 20, f"Page {doc.page}")
        
        canvas.restoreState()
    
    def _add_header(self, story: List, detection_result: Dict[str, Any]):
        """
        Add report header
        
        Args:
            story: ReportLab story list
            detection_result: Detection result dictionary
        """
        # Add title
        story.append(Paragraph("Deepfake Detection Analysis", self.styles['Title']))
        
        # Add subtitle
        detection_type = detection_result.get("detectionType", "image")
        media_name = detection_result.get("imageName", "Unknown")
        story.append(Paragraph(f"{detection_type.capitalize()} Analysis: {media_name}", self.styles['Subtitle']))
        
        # Add spacer
        story.append(Spacer(1, 0.25*inch))
    
    def _add_summary(self, story: List, detection_result: Dict[str, Any]):
        """
        Add summary section
        
        Args:
            story: ReportLab story list
            detection_result: Detection result dictionary
        """
        # Add section title
        story.append(Paragraph("Executive Summary", self.styles['SectionTitle']))
        
        # Get key metrics
        probability = detection_result.get("probability", 0.0)
        confidence = detection_result.get("confidence", 0.0)
        
        # Determine result text and color
        if probability >= 0.7:
            result_text = "Likely Deepfake"
            result_color = self.colors['danger']
        elif probability >= 0.4:
            result_text = "Possible Deepfake"
            result_color = self.colors['warning']
        else:
            result_text = "Likely Authentic"
            result_color = self.colors['success']
        
        # Create summary table
        data = [
            ["Analysis Result", "Deepfake Probability", "Confidence"],
            [result_text, f"{probability:.1%}", f"{confidence:.1%}"]
        ]
        
        table = Table(data, colWidths=[2*inch, 2*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (2, 0), self.colors['primary']),
            ('TEXTCOLOR', (0, 0), (2, 0), colors.white),
            ('ALIGN', (0, 0), (2, 0), 'CENTER'),
            ('ALIGN', (0, 1), (2, 1), 'CENTER'),
            ('FONTNAME', (0, 0), (2, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (2, 1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (2, 0), 12),
            ('FONTSIZE', (0, 1), (2, 1), 14),
            ('BOTTOMPADDING', (0, 0), (2, 0), 8),
            ('BACKGROUND', (0, 1), (0, 1), result_color),
            ('TEXTCOLOR', (0, 1), (0, 1), colors.white),
            ('GRID', (0, 0), (2, 1), 1, colors.black),
            ('BOX', (0, 0), (2, 1), 1, colors.black),
        ]))
        
        story.append(table)
        story.append(Spacer(1, 0.2*inch))
        
        # Add summary text
        summary_text = f"""
        This report presents the analysis of the {detection_result.get('detectionType', 'image')} 
        "{detection_result.get('imageName', 'Unknown')}" for potential deepfake manipulation. 
        The analysis was performed using DeepDefend's advanced detection system with specialized 
        Indian face and skin tone analysis.
        """
        story.append(Paragraph(summary_text, self.styles['BodyText']))
        
        # Add key findings
        story.append(Paragraph("Key Findings:", self.styles['SubsectionTitle']))
        
        # Create findings based on probability
        findings = []
        
        if probability >= 0.7:
            findings.append("High probability of deepfake manipulation detected")
            if confidence >= 0.7:
                findings.append("High confidence in the detection result")
            else:
                findings.append(f"Moderate confidence ({confidence:.1%}) in the detection result")
        elif probability >= 0.4:
            findings.append("Moderate probability of deepfake manipulation detected")
            findings.append("Further analysis recommended")
        else:
            findings.append("Low probability of deepfake manipulation")
            if confidence >= 0.7:
                findings.append("High confidence that the content is authentic")
            else:
                findings.append(f"Moderate confidence ({confidence:.1%}) that the content is authentic")
        
        # Add processing time if available
        if "processingTime" in detection_result:
            processing_time = detection_result["processingTime"] / 1000.0  # Convert ms to seconds
            findings.append(f"Analysis completed in {processing_time:.2f} seconds")
        
        # Add findings as a list
        items = [ListItem(Paragraph(finding, self.styles['BodyText'])) for finding in findings]
        story.append(ListFlowable(items, bulletType='bullet', leftIndent=20))
        
        story.append(Spacer(1, 0.3*inch))
    
    def _add_image_analysis(self, story: List, image_path: str, detection_result: Dict[str, Any]):
        """
        Add image analysis section
        
        Args:
            story: ReportLab story list
            image_path: Path to the image
            detection_result: Detection result dictionary
        """
        # Add section title
        story.append(Paragraph("Visual Analysis", self.styles['SectionTitle']))
        
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                story.append(Paragraph("Error: Could not read image file.", self.styles['BodyText']))
                return
            
            # Convert to RGB (from BGR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create a figure with the original image
            plt.figure(figsize=(8, 6))
            plt.imshow(image_rgb)
            plt.axis('off')
            plt.title("Original Image")
            
            # Save to a BytesIO object
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Create ReportLab Image
            img_buffer.seek(0)
            img = Image(img_buffer)
            img.drawHeight = 3*inch
            img.drawWidth = 4*inch
            
            # Add image to story
            story.append(img)
            story.append(Paragraph("Original image analyzed for deepfake detection", self.styles['Caption']))
            
            # Add regions if available
            regions = detection_result.get("regions", [])
            if regions:
                # Create a copy of the image for drawing regions
                image_with_regions = image_rgb.copy()
                
                # Draw regions
                for region in regions:
                    if all(k in region for k in ["x", "y", "width", "height"]):
                        x, y, w, h = region["x"], region["y"], region["width"], region["height"]
                        
                        # Determine color based on probability
                        prob = region.get("probability", 0.0)
                        if prob >= 0.7:
                            color = (255, 0, 0)  # Red for high probability
                        elif prob >= 0.4:
                            color = (255, 165, 0)  # Orange for medium probability
                        else:
                            color = (0, 255, 0)  # Green for low probability
                        
                        # Draw rectangle
                        cv2.rectangle(image_with_regions, (x, y), (x+w, y+h), color, 2)
                        
                        # Add probability text
                        cv2.putText(image_with_regions, f"{prob:.2f}", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Create a figure with the regions
                plt.figure(figsize=(8, 6))
                plt.imshow(image_with_regions)
                plt.axis('off')
                plt.title("Detected Regions")
                
                # Save to a BytesIO object
                regions_buffer = BytesIO()
                plt.savefig(regions_buffer, format='png', dpi=150, bbox_inches='tight')
                plt.close()
                
                # Create ReportLab Image
                regions_buffer.seek(0)
                regions_img = Image(regions_buffer)
                regions_img.drawHeight = 3*inch
                regions_img.drawWidth = 4*inch
                
                # Add image to story
                story.append(regions_img)
                story.append(Paragraph("Detected regions with deepfake probability scores", self.styles['Caption']))
            
            # Add heatmap visualization if probability is high
            probability = detection_result.get("probability", 0.0)
            if probability >= 0.4:
                # Create a simple heatmap (this would be more sophisticated in a real implementation)
                heatmap = np.zeros_like(image_rgb)
                
                # Apply a gradient based on probability
                for region in regions:
                    if all(k in region for k in ["x", "y", "width", "height"]):
                        x, y, w, h = region["x"], region["y"], region["width"], region["height"]
                        prob = region.get("probability", 0.0)
                        
                        # Create a gradient from blue (low) to red (high)
                        r = int(min(255, prob * 2 * 255))
                        b = int(min(255, (1 - prob) * 2 * 255))
                        
                        # Apply to region
                        heatmap[y:y+h, x:x+w, 0] = r
                        heatmap[y:y+h, x:x+w, 2] = b
                
                # Blend with original image
                alpha = 0.3
                blended = cv2.addWeighted(image_rgb, 1 - alpha, heatmap, alpha, 0)
                
                # Create a figure with the heatmap
                plt.figure(figsize=(8, 6))
                plt.imshow(blended)
                plt.axis('off')
                plt.title("Manipulation Heatmap")
                
                # Save to a BytesIO object
                heatmap_buffer = BytesIO()
                plt.savefig(heatmap_buffer, format='png', dpi=150, bbox_inches='tight')
                plt.close()
                
                # Create ReportLab Image
                heatmap_buffer.seek(0)
                heatmap_img = Image(heatmap_buffer)
                heatmap_img.drawHeight = 3*inch
                heatmap_img.drawWidth = 4*inch
                
                # Add image to story
                story.append(heatmap_img)
                story.append(Paragraph("Heatmap visualization of potential manipulation areas", self.styles['Caption']))
        
        except Exception as e:
            logger.error(f"Error in image analysis: {str(e)}")
            story.append(Paragraph(f"Error processing image: {str(e)}", self.styles['BodyText']))
        
        story.append(Spacer(1, 0.3*inch))
    
    def _add_detailed_analysis(self, story: List, detection_result: Dict[str, Any]):
        """
        Add detailed analysis section
        
        Args:
            story: ReportLab story list
            detection_result: Detection result dictionary
        """
        # Add section title
        story.append(Paragraph("Detailed Analysis", self.styles['SectionTitle']))
        
        # Add probability gauge chart
        probability = detection_result.get("probability", 0.0)
        confidence = detection_result.get("confidence", 0.0)
        
        # Create probability gauge
        self._add_gauge_chart(story, "Deepfake Probability", probability)
        
        # Create confidence gauge
        self._add_gauge_chart(story, "Detection Confidence", confidence)
        
        # Add explanation text
        explanation_text = """
        <b>Deepfake Probability</b> indicates the likelihood that the content has been manipulated using deepfake technology.
        Higher values suggest a greater probability of manipulation.
        <br/><br/>
        <b>Detection Confidence</b> represents the system's confidence in its assessment. Higher confidence values indicate
        more reliable detection results.
        """
        story.append(Paragraph(explanation_text, self.styles['BodyText']))
        
        # Add detection factors
        story.append(Paragraph("Detection Factors:", self.styles['SubsectionTitle']))
        
        # Create factors based on probability
        factors = []
        
        if probability >= 0.7:
            factors.append("Strong indicators of digital manipulation detected")
            factors.append("Facial inconsistencies identified")
            factors.append("Unnatural visual patterns detected")
        elif probability >= 0.4:
            factors.append("Some indicators of digital manipulation detected")
            factors.append("Minor facial inconsistencies identified")
        else:
            factors.append("No significant indicators of manipulation detected")
            factors.append("Facial features appear consistent and natural")
        
        # Add model-specific factors
        model_name = detection_result.get("model", "enhanced_indian_detection")
        if "enhanced_indian" in model_name.lower():
            factors.append("Analysis optimized for Indian facial features and skin tones")
        
        # Add regions information if available
        regions = detection_result.get("regions", [])
        if regions:
            face_count = len(regions)
            factors.append(f"{face_count} {'faces' if face_count > 1 else 'face'} analyzed in the content")
            
            # Check for metadata in regions
            for region in regions:
                metadata = region.get("metadata", {})
                
                # Add skin tone information if available
                skin_tone = metadata.get("skin_tone", {})
                if skin_tone and skin_tone.get("success", False):
                    indian_tone = skin_tone.get("indian_tone", {})
                    if indian_tone:
                        tone_name = indian_tone.get("name", "Unknown")
                        factors.append(f"Detected skin tone: {tone_name}")
                        break
        
        # Add factors as a list
        items = [ListItem(Paragraph(factor, self.styles['BodyText'])) for factor in factors]
        story.append(ListFlowable(items, bulletType='bullet', leftIndent=20))
        
        story.append(Spacer(1, 0.3*inch))
    
    def _add_skin_tone_analysis(self, story: List, detection_result: Dict[str, Any]):
        """
        Add skin tone analysis section
        
        Args:
            story: ReportLab story list
            detection_result: Detection result dictionary
        """
        # Add section title
        story.append(Paragraph("Skin Tone Analysis", self.styles['SectionTitle']))
        
        # Extract skin tone data
        skin_tone_data = self._extract_skin_tone_data(detection_result)
        
        if not skin_tone_data:
            story.append(Paragraph("No skin tone data available.", self.styles['BodyText']))
            return
        
        # Add skin tone information
        indian_tone = skin_tone_data.get("indian_tone", {})
        fitzpatrick = skin_tone_data.get("fitzpatrick_type", {})
        
        if indian_tone:
            tone_name = indian_tone.get("name", "Unknown")
            tone_score = indian_tone.get("score", 0.0)
            
            story.append(Paragraph(f"Detected Indian Skin Tone: {tone_name}", self.styles['SubsectionTitle']))
            
            # Add explanation
            explanation = f"""
            The analysis detected a skin tone classified as <b>{tone_name}</b> with a darkness score of {tone_score:.2f}.
            This classification is based on specialized analysis optimized for Indian skin tones.
            """
            story.append(Paragraph(explanation, self.styles['BodyText']))
        
        if fitzpatrick:
            fitz_name = fitzpatrick.get("name", "Unknown")
            fitz_type = fitzpatrick.get("type", 0)
            
            story.append(Paragraph(f"Fitzpatrick Scale Classification: {fitz_name}", self.styles['SubsectionTitle']))
            
            # Add explanation
            explanation = f"""
            On the Fitzpatrick scale, the skin tone is classified as <b>{fitz_name}</b> (Type {fitz_type}).
            The Fitzpatrick scale is a numerical classification schema for human skin color, used worldwide in dermatological research.
            """
            story.append(Paragraph(explanation, self.styles['BodyText']))
        
        # Add skin metrics if available
        evenness_score = skin_tone_data.get("evenness_score", None)
        texture_score = skin_tone_data.get("texture_score", None)
        
        if evenness_score is not None or texture_score is not None:
            story.append(Paragraph("Skin Quality Metrics:", self.styles['SubsectionTitle']))
            
            # Create data for table
            data = [["Metric", "Score", "Interpretation"]]
            
            if evenness_score is not None:
                interpretation = "Natural" if evenness_score < 0.9 else "Unnaturally Even"
                data.append(["Skin Evenness", f"{evenness_score:.2f}", interpretation])
            
            if texture_score is not None:
                interpretation = "Natural" if texture_score > 0.2 else "Unnaturally Smooth"
                data.append(["Skin Texture", f"{texture_score:.2f}", interpretation])
            
            # Create table
            table = Table(data, colWidths=[1.5*inch, 1*inch, 2.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (2, 0), self.colors['secondary']),
                ('TEXTCOLOR', (0, 0), (2, 0), colors.white),
                ('ALIGN', (0, 0), (2, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (2, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (2, 0), 12),
                ('BOTTOMPADDING', (0, 0), (2, 0), 8),
                ('GRID', (0, 0), (2, len(data)-1), 1, colors.black),
                ('BOX', (0, 0), (2, len(data)-1), 1, colors.black),
            ]))
            
            story.append(table)
            
            # Add explanation
            explanation = """
            <b>Skin Evenness</b>: Measures how uniform the skin tone is across the face. Unnaturally even skin can indicate digital manipulation or filtering.
            <br/><br/>
            <b>Skin Texture</b>: Measures the presence of natural skin texture. Unnaturally smooth skin can indicate digital manipulation or filtering.
            """
            story.append(Paragraph(explanation, self.styles['BodyText']))
        
        # Add skin anomalies if available
        skin_anomalies = self._extract_skin_anomalies_data(detection_result)
        
        if skin_anomalies and skin_anomalies.get("success", False):
            anomalies = skin_anomalies.get("anomalies", [])
            
            if anomalies:
                story.append(Paragraph("Detected Skin Anomalies:", self.styles['SubsectionTitle']))
                
                # Create data for table
                data = [["Anomaly Type", "Description", "Severity"]]
                
                for anomaly in anomalies:
                    anomaly_type = anomaly.get("type", "Unknown")
                    description = anomaly.get("description", "Unknown anomaly")
                    severity = anomaly.get("severity", 0.0)
                    
                    # Format type for display
                    display_type = anomaly_type.replace("_", " ").title()
                    
                    data.append([display_type, description, f"{severity:.2f}"])
                
                # Create table
                table = Table(data, colWidths=[1.5*inch, 3*inch, 0.5*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (2, 0), self.colors['warning']),
                    ('TEXTCOLOR', (0, 0), (2, 0), colors.white),
                    ('ALIGN', (0, 0), (2, 0), 'CENTER'),
                    ('ALIGN', (2, 1), (2, len(data)-1), 'CENTER'),
                    ('FONTNAME', (0, 0), (2, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (2, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (2, 0), 8),
                    ('GRID', (0, 0), (2, len(data)-1), 1, colors.black),
                    ('BOX', (0, 0), (2, len(data)-1), 1, colors.black),
                ]))
                
                story.append(table)
                
                # Add explanation
                explanation = """
                Skin anomalies are irregularities in skin appearance that may indicate digital manipulation.
                Higher severity scores indicate stronger evidence of manipulation.
                """
                story.append(Paragraph(explanation, self.styles['BodyText']))
                
                # Add overall anomaly score
                anomaly_score = skin_anomalies.get("anomaly_score", 0.0)
                story.append(Paragraph(f"Overall Anomaly Score: {anomaly_score:.2f}", self.styles['BodyText']))
        
        story.append(Spacer(1, 0.3*inch))
    
    def _add_gan_detection_analysis(self, story: List, detection_result: Dict[str, Any]):
        """
        Add GAN detection analysis section
        
        Args:
            story: ReportLab story list
            detection_result: Detection result dictionary
        """
        # Add section title
        story.append(Paragraph("GAN Detection Analysis", self.styles['SectionTitle']))
        
        # Extract GAN detection data
        gan_data = self._extract_gan_detection_data(detection_result)
        
        if not gan_data:
            story.append(Paragraph("No GAN detection data available.", self.styles['BodyText']))
            return
        
        # Add GAN probability gauge
        gan_probability = gan_data.get("gan_probability", 0.0)
        self._add_gauge_chart(story, "GAN Generation Probability", gan_probability)
        
        # Add confidence
        confidence = gan_data.get("confidence", 0.0)
        story.append(Paragraph(f"Detection Confidence: {confidence:.2f}", self.styles['BodyText']))
        
        # Add method
        method = gan_data.get("method", "Unknown")
        story.append(Paragraph(f"Detection Method: {method}", self.styles['BodyText']))
        
        # Add explanation
        explanation = """
        GAN (Generative Adversarial Network) detection analyzes the image for artifacts and patterns
        characteristic of AI-generated content. High GAN probability indicates the image was likely
        created by an AI system rather than being a photograph of a real person.
        """
        story.append(Paragraph(explanation, self.styles['BodyText']))
        
        # Add scores if available
        scores = gan_data.get("scores", {})
        
        if scores:
            story.append(Paragraph("Detection Scores by Method:", self.styles['SubsectionTitle']))
            
            # Create data for table
            data = [["Detection Method", "Score"]]
            
            for method, score in scores.items():
                # Format method name for display
                display_method = method.replace("_", " ").title()
                
                data.append([display_method, f"{score:.2f}"])
            
            # Create table
            table = Table(data, colWidths=[3*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), self.colors['secondary']),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
                ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                ('ALIGN', (1, 1), (1, len(data)-1), 'CENTER'),
                ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (1, 0), 8),
                ('GRID', (0, 0), (1, len(data)-1), 1, colors.black),
                ('BOX', (0, 0), (1, len(data)-1), 1, colors.black),
            ]))
            
            story.append(table)
        
        # Add artifacts if available
        artifacts = gan_data.get("artifacts", [])
        
        if artifacts:
            story.append(Paragraph("Detected GAN Artifacts:", self.styles['SubsectionTitle']))
            
            # Create data for table
            data = [["Artifact Type", "Description", "Severity", "Location"]]
            
            for artifact in artifacts:
                artifact_type = artifact.get("type", "Unknown")
                description = artifact.get("description", "Unknown artifact")
                severity = artifact.get("severity", 0.0)
                location = artifact.get("location", "Unknown")
                
                # Format type for display
                display_type = artifact_type.replace("_", " ").title()
                
                data.append([display_type, description, f"{severity:.2f}", location])
            
            # Create table
            table = Table(data, colWidths=[1.2*inch, 2.3*inch, 0.5*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (3, 0), self.colors['warning']),
                ('TEXTCOLOR', (0, 0), (3, 0), colors.white),
                ('ALIGN', (0, 0), (3, 0), 'CENTER'),
                ('ALIGN', (2, 1), (2, len(data)-1), 'CENTER'),
                ('FONTNAME', (0, 0), (3, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (3, 0), 12),
                ('BOTTOMPADDING', (0, 0), (3, 0), 8),
                ('GRID', (0, 0), (3, len(data)-1), 1, colors.black),
                ('BOX', (0, 0), (3, len(data)-1), 1, colors.black),
            ]))
            
            story.append(table)
            
            # Add explanation
            explanation = """
            GAN artifacts are specific irregularities that are characteristic of AI-generated images.
            These can include unnatural eyes, hair, backgrounds, or unnaturally high symmetry.
            """
            story.append(Paragraph(explanation, self.styles['BodyText']))
        
        story.append(Spacer(1, 0.3*inch))
    
    def _add_methodology(self, story: List):
        """
        Add methodology section
        
        Args:
            story: ReportLab story list
        """
        # Add section title
        story.append(Paragraph("Methodology", self.styles['SectionTitle']))
        
        # Add methodology text
        methodology_text = """
        <b>DeepDefend's Analysis Methodology:</b>
        <br/><br/>
        This analysis was performed using DeepDefend's advanced deepfake detection system, which combines multiple specialized models:
        <br/><br/>
        1. <b>Ensemble Detection:</b> Multiple specialized models analyze the content, with results combined using weighted voting.
        <br/><br/>
        2. <b>Indian Face Specialization:</b> Models optimized for Indian facial features and skin tones provide more accurate results for Indian faces.
        <br/><br/>
        3. <b>Skin Tone Analysis:</b> Advanced skin tone analysis detects inconsistencies in skin appearance that may indicate manipulation.
        <br/><br/>
        4. <b>GAN Detection:</b> Specialized detection for GAN-generated images identifies artifacts characteristic of AI generation.
        <br/><br/>
        5. <b>Confidence Estimation:</b> Sophisticated algorithms estimate the reliability of detection results based on multiple factors.
        """
        story.append(Paragraph(methodology_text, self.styles['BodyText']))
        
        story.append(Spacer(1, 0.3*inch))
    
    def _add_disclaimer(self, story: List):
        """
        Add disclaimer section
        
        Args:
            story: ReportLab story list
        """
        # Add section title
        story.append(Paragraph("Disclaimer", self.styles['SectionTitle']))
        
        # Add disclaimer text
        disclaimer_text = """
        This report is provided for informational purposes only. While DeepDefend uses advanced technology to detect deepfakes,
        no detection system is 100% accurate. Results should be interpreted in context and may require human verification.
        DeepDefend is not responsible for decisions made based on this report.
        <br/><br/>
        © DeepDefend - All rights reserved.
        """
        story.append(Paragraph(disclaimer_text, self.styles['BodyText']))
    
    def _add_gauge_chart(self, story: List, title: str, value: float):
        """
        Add a gauge chart
        
        Args:
            story: ReportLab story list
            title: Chart title
            value: Value to display (0-1)
        """
        # Create a BytesIO object
        img_buffer = BytesIO()
        
        # Create figure
        plt.figure(figsize=(6, 3))
        
        # Create gauge chart
        ax = plt.subplot(111, polar=True)
        
        # Set the limits
        ax.set_ylim(0, 1)
        ax.set_xlim(-np.pi/2, np.pi/2)
        
        # Remove tick labels
        ax.set_yticklabels([])
        
        # Set custom x ticks
        ax.set_xticks(np.linspace(-np.pi/2, np.pi/2, 5))
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        
        # Add colored bands
        theta = np.linspace(-np.pi/2, np.pi/2, 100)
        
        # Green band (0-40%)
        r = np.ones_like(theta) * 0.9
        mask = (theta >= -np.pi/2) & (theta < -np.pi/2 + 0.4 * np.pi)
        ax.fill_between(theta[mask], 0, r[mask], color='green', alpha=0.3)
        
        # Yellow band (40-70%)
        mask = (theta >= -np.pi/2 + 0.4 * np.pi) & (theta < -np.pi/2 + 0.7 * np.pi)
        ax.fill_between(theta[mask], 0, r[mask], color='yellow', alpha=0.3)
        
        # Red band (70-100%)
        mask = (theta >= -np.pi/2 + 0.7 * np.pi) & (theta <= np.pi/2)
        ax.fill_between(theta[mask], 0, r[mask], color='red', alpha=0.3)
        
        # Add needle
        needle_theta = -np.pi/2 + value * np.pi
        ax.plot([0, needle_theta], [0, 0.8], 'k-', linewidth=2)
        
        # Add a center circle
        circle = plt.Circle((0, 0), 0.1, transform=ax.transData._b, color='darkgray', zorder=10)
        ax.add_artist(circle)
        
        # Add value text
        ax.text(0, -0.2, f"{value:.1%}", ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Add title
        plt.title(title, y=1.2, fontsize=14)
        
        # Save to BytesIO
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # Create ReportLab Image
        img_buffer.seek(0)
        img = Image(img_buffer)
        img.drawHeight = 1.5*inch
        img.drawWidth = 3*inch
        
        # Add to story
        story.append(img)
        story.append(Spacer(1, 0.1*inch))
    
    def _has_skin_tone_data(self, detection_result: Dict[str, Any]) -> bool:
        """
        Check if the detection result has skin tone data
        
        Args:
            detection_result: Detection result dictionary
            
        Returns:
            True if skin tone data is available, False otherwise
        """
        # Check regions for skin tone data
        regions = detection_result.get("regions", [])
        
        for region in regions:
            metadata = region.get("metadata", {})
            skin_tone = metadata.get("skin_tone", {})
            
            if skin_tone and skin_tone.get("success", False):
                return True
        
        return False
    
    def _extract_skin_tone_data(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract skin tone data from detection result
        
        Args:
            detection_result: Detection result dictionary
            
        Returns:
            Skin tone data dictionary
        """
        # Check regions for skin tone data
        regions = detection_result.get("regions", [])
        
        for region in regions:
            metadata = region.get("metadata", {})
            skin_tone = metadata.get("skin_tone", {})
            
            if skin_tone and skin_tone.get("success", False):
                return skin_tone
        
        return {}
    
    def _has_gan_detection_data(self, detection_result: Dict[str, Any]) -> bool:
        """
        Check if the detection result has GAN detection data
        
        Args:
            detection_result: Detection result dictionary
            
        Returns:
            True if GAN detection data is available, False otherwise
        """
        # Check regions for GAN detection data
        regions = detection_result.get("regions", [])
        
        for region in regions:
            metadata = region.get("metadata", {})
            gan_detection = metadata.get("gan_detection", {})
            
            if gan_detection and gan_detection.get("success", False):
                return True
        
        return False
    
    def _extract_gan_detection_data(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract GAN detection data from detection result
        
        Args:
            detection_result: Detection result dictionary
            
        Returns:
            GAN detection data dictionary
        """
        # Check regions for GAN detection data
        regions = detection_result.get("regions", [])
        
        for region in regions:
            metadata = region.get("metadata", {})
            gan_detection = metadata.get("gan_detection", {})
            
            if gan_detection and gan_detection.get("success", False):
                return gan_detection
        
        return {}
    
    def _extract_skin_anomalies_data(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract skin anomalies data from detection result
        
        Args:
            detection_result: Detection result dictionary
            
        Returns:
            Skin anomalies data dictionary
        """
        # Check regions for skin anomalies data
        regions = detection_result.get("regions", [])
        
        for region in regions:
            metadata = region.get("metadata", {})
            skin_anomalies = metadata.get("skin_anomalies", {})
            
            if skin_anomalies and skin_anomalies.get("success", False):
                return skin_anomalies
        
        return {}


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate enhanced PDF reports for deepfake detection results")
    parser.add_argument("--result", type=str, required=True, help="Path to detection result JSON file")
    parser.add_argument("--image", type=str, help="Path to original image (optional)")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Load detection result
    try:
        with open(args.result, 'r') as f:
            detection_result = json.load(f)
    except Exception as e:
        logger.error(f"Error loading detection result: {str(e)}")
        return
    
    # Create report generator
    report_generator = EnhancedPDFReportGenerator(output_dir=args.output_dir)
    
    # Generate report
    report_path = report_generator.generate_report(detection_result, args.image)
    
    print(f"Report generated: {report_path}")


if __name__ == "__main__":
    main()