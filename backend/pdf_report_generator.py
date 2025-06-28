"""
PDF Report Generator for DeepDefend
Generates detailed PDF reports for deepfake detection results
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
    Image, PageBreak, ListFlowable, ListItem, Flowable
)
from reportlab.pdfgen import canvas
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.charts.legends import Legend

# Configure logging
logger = logging.getLogger(__name__)

class SkinToneChart(Flowable):
    """Custom flowable for skin tone visualization"""
    
    def __init__(self, skin_tone_data, width=500, height=100):
        Flowable.__init__(self)
        self.skin_tone_data = skin_tone_data
        self.width = width
        self.height = height
    
    def draw(self):
        """Draw the skin tone chart"""
        # Create a color gradient
        canvas = self.canv
        
        # Draw title
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(0, self.height + 10, "Skin Tone Analysis")
        
        # Get skin tone data
        indian_tone = self.skin_tone_data.get("indian_tone", {})
        if not indian_tone:
            canvas.setFont("Helvetica", 9)
            canvas.drawString(0, self.height - 10, "No skin tone data available")
            return
        
        # Draw skin tone name
        tone_name = indian_tone.get("name", "Unknown")
        tone_score = indian_tone.get("score", 0.5)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(0, self.height - 15, f"Detected Skin Tone: {tone_name}")
        
        # Draw color gradient
        gradient_height = 30
        gradient_y = self.height - 50
        
        # Define colors for the gradient (fair to dark)
        colors_rgb = [
            (241, 194, 167),  # Fair
            (224, 172, 138),  # Wheatish
            (198, 134, 94),   # Medium
            (172, 112, 76),   # Dusky
            (141, 85, 54)     # Dark
        ]
        
        # Draw gradient segments
        segment_width = self.width / len(colors_rgb)
        for i, rgb in enumerate(colors_rgb):
            r, g, b = rgb
            canvas.setFillColorRGB(r/255, g/255, b/255)
            canvas.rect(
                i * segment_width, 
                gradient_y, 
                segment_width, 
                gradient_height, 
                fill=1, 
                stroke=0
            )
        
        # Draw labels
        labels = ["Fair", "Wheatish", "Medium", "Dusky", "Dark"]
        canvas.setFont("Helvetica", 8)
        canvas.setFillColorRGB(0, 0, 0)
        
        for i, label in enumerate(labels):
            canvas.drawString(
                i * segment_width + segment_width/2 - 15, 
                gradient_y - 15, 
                label
            )
        
        # Draw marker for detected tone
        marker_x = tone_score * self.width
        canvas.setFillColorRGB(0, 0, 0)
        canvas.setStrokeColorRGB(0, 0, 0)
        
        # Draw triangle marker
        canvas.line(marker_x, gradient_y + gradient_height + 5, 
                   marker_x - 5, gradient_y + gradient_height + 15)
        canvas.line(marker_x, gradient_y + gradient_height + 5, 
                   marker_x + 5, gradient_y + gradient_height + 15)
        canvas.line(marker_x - 5, gradient_y + gradient_height + 15, 
                   marker_x + 5, gradient_y + gradient_height + 15)

class PDFReportGenerator:
    """
    Generates detailed PDF reports for deepfake detection results
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the PDF report generator
        
        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize styles
        self.styles = getSampleStyleSheet()
        
        # Add custom styles
        self.styles.add(
            ParagraphStyle(
                name='CustomTitle',
                parent=self.styles['Title'],
                fontSize=24,
                spaceAfter=12
            )
        )
        
        self.styles.add(
            ParagraphStyle(
                name='SectionTitle',
                parent=self.styles['Heading2'],
                fontSize=14,
                spaceAfter=6
            )
        )
        
        self.styles.add(
            ParagraphStyle(
                name='SubsectionTitle',
                parent=self.styles['Heading3'],
                fontSize=12,
                spaceAfter=6
            )
        )
        
        self.styles.add(
            ParagraphStyle(
                name='BodyText',
                parent=self.styles['Normal'],
                fontSize=10,
                spaceAfter=6
            )
        )
        
        self.styles.add(
            ParagraphStyle(
                name='Caption',
                parent=self.styles['Normal'],
                fontSize=8,
                fontStyle='italic',
                alignment=1  # Center alignment
            )
        )
        
        logger.info(f"Initialized PDF report generator with output directory: {output_dir}")
    
    def generate_report(self, 
                       detection_result: Dict[str, Any], 
                       detection_id: str = None,
                       include_images: bool = True) -> str:
        """
        Generate a detailed PDF report for a detection result
        
        Args:
            detection_result: Detection result dictionary
            detection_id: Optional detection ID
            include_images: Whether to include images in the report
            
        Returns:
            Path to the generated PDF file
        """
        try:
            # Generate report ID if not provided
            if not detection_id:
                detection_id = f"report-{int(time.time())}"
            
            # Create output path
            output_path = os.path.join(self.output_dir, f"deepfake-report-{detection_id}.pdf")
            
            # Create PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Build report content
            content = []
            
            # Add title
            is_deepfake = detection_result.get("probability", 0) > 0.5
            title_text = "Deepfake Detected" if is_deepfake else "Authentic Media Detected"
            title = Paragraph(f"<font color={'red' if is_deepfake else 'green'}>DeepDefend Analysis Report: {title_text}</font>", self.styles["CustomTitle"])
            content.append(title)
            
            # Add timestamp
            timestamp = detection_result.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            content.append(Paragraph(f"Report generated on: {timestamp}", self.styles["Normal"]))
            content.append(Spacer(1, 0.2 * inch))
            
            # Add summary section
            content.extend(self._create_summary_section(detection_result))
            content.append(Spacer(1, 0.3 * inch))
            
            # Add detection details section
            content.extend(self._create_detection_details_section(detection_result))
            content.append(Spacer(1, 0.3 * inch))
            
            # Add face analysis section if we have face data
            if "regions" in detection_result and detection_result["regions"]:
                content.extend(self._create_face_analysis_section(detection_result, include_images))
                content.append(Spacer(1, 0.3 * inch))
            
            # Add skin tone analysis section if available
            if self._has_skin_tone_data(detection_result):
                content.extend(self._create_skin_tone_section(detection_result, include_images))
                content.append(Spacer(1, 0.3 * inch))
            
            # Add temporal analysis section for videos
            if detection_result.get("detectionType") == "video":
                content.extend(self._create_temporal_analysis_section(detection_result))
                content.append(Spacer(1, 0.3 * inch))
            
            # Add recommendations section
            content.extend(self._create_recommendations_section(detection_result))
            content.append(Spacer(1, 0.3 * inch))
            
            # Add technical details section
            content.extend(self._create_technical_details_section(detection_result))
            
            # Add footer
            content.append(Spacer(1, 0.5 * inch))
            content.append(Paragraph("This report was generated by DeepDefend AI Detection System.", self.styles["Caption"]))
            content.append(Paragraph("For more information, please visit our documentation.", self.styles["Caption"]))
            
            # Build the PDF
            doc.build(content)
            
            logger.info(f"Generated PDF report: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}", exc_info=True)
            raise
    
    def _create_summary_section(self, detection_result: Dict[str, Any]) -> List[Any]:
        """Create the summary section of the report"""
        content = []
        
        # Add section title
        content.append(Paragraph("Detection Summary", self.styles["SectionTitle"]))
        content.append(Spacer(1, 0.1 * inch))
        
        # Get key metrics
        probability = detection_result.get("probability", 0) * 100
        confidence = detection_result.get("confidence", 0) * 100
        detection_type = detection_result.get("detectionType", "Unknown")
        
        # Create summary table
        data = [
            ["Metric", "Value"],
            ["Detection Type", detection_type.capitalize()],
            ["Media Name", detection_result.get("imageName", "Unknown")],
            ["Deepfake Probability", f"{probability:.1f}%"],
            ["Confidence", f"{confidence:.1f}%"],
            ["Processing Time", f"{detection_result.get('processingTime', 0)} ms"],
            ["Model", detection_result.get("model", "Standard detection model")]
        ]
        
        # Add video-specific metrics
        if detection_type == "video":
            data.append(["Frames Analyzed", str(detection_result.get("frameCount", "N/A"))])
        
        # Create table
        table = Table(data, colWidths=[2*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        content.append(table)
        
        # Add probability visualization
        content.append(Spacer(1, 0.2 * inch))
        content.append(Paragraph("Deepfake Probability", self.styles["SubsectionTitle"]))
        
        # Create probability gauge
        gauge_image = self._create_probability_gauge(detection_result.get("probability", 0))
        if gauge_image:
            content.append(gauge_image)
            content.append(Paragraph("Probability gauge showing likelihood of deepfake", self.styles["Caption"]))
        
        return content
    
    def _create_detection_details_section(self, detection_result: Dict[str, Any]) -> List[Any]:
        """Create the detection details section of the report"""
        content = []
        
        # Add section title
        content.append(Paragraph("Detection Details", self.styles["SectionTitle"]))
        content.append(Spacer(1, 0.1 * inch))
        
        # Add detection explanation
        is_deepfake = detection_result.get("probability", 0) > 0.5
        if is_deepfake:
            explanation = (
                "The analyzed media has been classified as a <b>deepfake</b> with high probability. "
                "This means our AI system has detected characteristics commonly associated with "
                "artificially generated or manipulated content."
            )
        else:
            explanation = (
                "The analyzed media has been classified as <b>authentic</b> with high probability. "
                "Our AI system did not detect significant characteristics commonly associated with "
                "artificially generated or manipulated content."
            )
        
        content.append(Paragraph(explanation, self.styles["BodyText"]))
        content.append(Spacer(1, 0.1 * inch))
        
        # Add confidence explanation
        confidence = detection_result.get("confidence", 0)
        if confidence > 0.8:
            conf_explanation = (
                "The system has <b>high confidence</b> in this result, indicating that the "
                "detection features were very clear and consistent."
            )
        elif confidence > 0.6:
            conf_explanation = (
                "The system has <b>moderate confidence</b> in this result. While the detection "
                "is likely correct, there were some ambiguous features in the analysis."
            )
        else:
            conf_explanation = (
                "The system has <b>lower confidence</b> in this result. The media contains "
                "features that made classification challenging, and the result should be "
                "interpreted with caution."
            )
        
        content.append(Paragraph(conf_explanation, self.styles["BodyText"]))
        
        return content
    
    def _create_face_analysis_section(self, 
                                     detection_result: Dict[str, Any],
                                     include_images: bool = True) -> List[Any]:
        """Create the face analysis section of the report"""
        content = []
        
        # Add section title
        content.append(Paragraph("Face Analysis", self.styles["SectionTitle"]))
        content.append(Spacer(1, 0.1 * inch))
        
        # Get face regions
        regions = detection_result.get("regions", [])
        if not regions:
            content.append(Paragraph("No face regions detected in the media.", self.styles["BodyText"]))
            return content
        
        # Add face count
        content.append(Paragraph(f"Detected {len(regions)} face(s) in the media.", self.styles["BodyText"]))
        content.append(Spacer(1, 0.1 * inch))
        
        # Process each face
        for i, region in enumerate(regions):
            # Add face subsection
            content.append(Paragraph(f"Face {i+1}", self.styles["SubsectionTitle"]))
            
            # Create face data table
            face_prob = region.get("probability", 0) * 100
            face_conf = region.get("confidence", 0) * 100
            
            data = [
                ["Metric", "Value"],
                ["Deepfake Probability", f"{face_prob:.1f}%"],
                ["Confidence", f"{face_conf:.1f}%"]
            ]
            
            # Add bounding box if available
            if "x" in region and "y" in region and "width" in region and "height" in region:
                bbox_text = f"({region['x']}, {region['y']}, {region['width']}, {region['height']})"
                data.append(["Bounding Box", bbox_text])
            
            # Create table
            table = Table(data, colWidths=[2*inch, 3*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            content.append(table)
            content.append(Spacer(1, 0.1 * inch))
            
            # Add face image if available and requested
            if include_images and "image_data" in region:
                try:
                    face_image = self._decode_image_data(region["image_data"])
                    if face_image is not None:
                        img_width = 2 * inch
                        img_height = 2 * inch
                        content.append(Image(face_image, width=img_width, height=img_height))
                        content.append(Paragraph(f"Face {i+1} image", self.styles["Caption"]))
                        content.append(Spacer(1, 0.1 * inch))
                except Exception as e:
                    logger.error(f"Error adding face image: {str(e)}")
            
            # Add spacer between faces
            if i < len(regions) - 1:
                content.append(Spacer(1, 0.2 * inch))
        
        return content
    
    def _has_skin_tone_data(self, detection_result: Dict[str, Any]) -> bool:
        """Check if the detection result has skin tone data"""
        if "regions" not in detection_result:
            return False
        
        for region in detection_result.get("regions", []):
            if "metadata" in region and "skin_tone" in region["metadata"]:
                return True
            
            if "skin_tone" in region:
                return True
        
        return False
    
    def _create_skin_tone_section(self, 
                                 detection_result: Dict[str, Any],
                                 include_images: bool = True) -> List[Any]:
        """Create the skin tone analysis section of the report"""
        content = []
        
        # Add section title
        content.append(Paragraph("Skin Tone Analysis", self.styles["SectionTitle"]))
        content.append(Spacer(1, 0.1 * inch))
        
        # Add explanation
        content.append(Paragraph(
            "Skin tone analysis helps identify inconsistencies that may indicate deepfakes. "
            "Authentic images typically have consistent skin tones, while deepfakes may show "
            "unnatural variations or artifacts.",
            self.styles["BodyText"]
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        # Process each face
        regions = detection_result.get("regions", [])
        for i, region in enumerate(regions):
            # Get skin tone data
            skin_tone_data = None
            if "metadata" in region and "skin_tone" in region["metadata"]:
                skin_tone_data = region["metadata"]["skin_tone"]
            elif "skin_tone" in region:
                skin_tone_data = region["skin_tone"]
            
            if not skin_tone_data:
                continue
            
            # Add face subsection
            content.append(Paragraph(f"Face {i+1} Skin Analysis", self.styles["SubsectionTitle"]))
            
            # Add skin tone chart
            if skin_tone_data.get("success", False):
                content.append(SkinToneChart(skin_tone_data))
                content.append(Spacer(1, 0.1 * inch))
            
            # Create skin tone data table
            data = [["Metric", "Value"]]
            
            # Add Indian tone if available
            indian_tone = skin_tone_data.get("indian_tone", {})
            if indian_tone:
                data.append(["Indian Skin Tone", indian_tone.get("name", "Unknown")])
            
            # Add Fitzpatrick type if available
            fitzpatrick = skin_tone_data.get("fitzpatrick_type", {})
            if fitzpatrick:
                data.append(["Fitzpatrick Type", fitzpatrick.get("name", "Unknown")])
            
            # Add skin metrics if available
            if "evenness_score" in skin_tone_data:
                evenness = skin_tone_data["evenness_score"] * 100
                data.append(["Skin Evenness", f"{evenness:.1f}%"])
            
            if "texture_score" in skin_tone_data:
                texture = skin_tone_data["texture_score"] * 100
                data.append(["Texture Score", f"{texture:.1f}%"])
            
            # Create table
            if len(data) > 1:  # Only create table if we have data
                table = Table(data, colWidths=[2*inch, 3*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                    ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (1, 0), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                
                content.append(table)
                content.append(Spacer(1, 0.1 * inch))
            
            # Add skin anomalies if available
            skin_anomalies = None
            if "metadata" in region and "skin_anomalies" in region["metadata"]:
                skin_anomalies = region["metadata"]["skin_anomalies"]
            elif "skin_anomalies" in region:
                skin_anomalies = region["skin_anomalies"]
            
            if skin_anomalies and skin_anomalies.get("success", False):
                anomalies = skin_anomalies.get("anomalies", [])
                if anomalies:
                    content.append(Paragraph("Detected Skin Anomalies:", self.styles["BodyText"]))
                    
                    # Create list of anomalies
                    anomaly_items = []
                    for anomaly in anomalies:
                        description = anomaly.get("description", "Unknown anomaly")
                        severity = anomaly.get("severity", 0) * 100
                        anomaly_text = f"{description} (Severity: {severity:.1f}%)"
                        anomaly_items.append(ListItem(Paragraph(anomaly_text, self.styles["BodyText"])))
                    
                    if anomaly_items:
                        anomaly_list = ListFlowable(
                            anomaly_items,
                            bulletType='bullet',
                            leftIndent=20
                        )
                        content.append(anomaly_list)
                else:
                    content.append(Paragraph("No skin anomalies detected.", self.styles["BodyText"]))
            
            # Add spacer between faces
            if i < len(regions) - 1:
                content.append(Spacer(1, 0.2 * inch))
        
        return content
    
    def _create_temporal_analysis_section(self, detection_result: Dict[str, Any]) -> List[Any]:
        """Create the temporal analysis section of the report"""
        content = []
        
        # Add section title
        content.append(Paragraph("Temporal Analysis", self.styles["SectionTitle"]))
        content.append(Spacer(1, 0.1 * inch))
        
        # Check if we have temporal data
        has_temporal_data = False
        if "metadata" in detection_result and "temporal_analysis" in detection_result["metadata"]:
            temporal_data = detection_result["metadata"]["temporal_analysis"]
            has_temporal_data = True
        elif "temporal_analysis" in detection_result:
            temporal_data = detection_result["temporal_analysis"]
            has_temporal_data = True
        
        if not has_temporal_data:
            content.append(Paragraph("No temporal analysis data available.", self.styles["BodyText"]))
            return content
        
        # Add explanation
        content.append(Paragraph(
            "Temporal analysis examines consistency across video frames. Authentic videos typically "
            "show natural transitions between frames, while deepfakes may exhibit inconsistencies "
            "in motion, lighting, or facial features.",
            self.styles["BodyText"]
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        # Create temporal data table
        data = [["Metric", "Value"]]
        
        # Add frame count
        frame_count = detection_result.get("frameCount", 0)
        data.append(["Frames Analyzed", str(frame_count)])
        
        # Add consistency score if available
        if "consistency_score" in temporal_data:
            consistency = temporal_data["consistency_score"] * 100
            data.append(["Temporal Consistency", f"{consistency:.1f}%"])
        
        # Add other temporal metrics
        if "optical_flow_score" in temporal_data:
            flow_score = (1 - temporal_data["optical_flow_score"]) * 100  # Convert to consistency
            data.append(["Motion Consistency", f"{flow_score:.1f}%"])
        
        if "face_motion_score" in temporal_data:
            motion_score = (1 - temporal_data["face_motion_score"]) * 100  # Convert to consistency
            data.append(["Face Motion Consistency", f"{motion_score:.1f}%"])
        
        # Create table
        table = Table(data, colWidths=[2*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        content.append(table)
        content.append(Spacer(1, 0.1 * inch))
        
        # Add frame predictions chart if available
        if "frame_predictions" in temporal_data:
            frame_predictions = temporal_data["frame_predictions"]
            if frame_predictions:
                # Create frame predictions chart
                chart_image = self._create_frame_predictions_chart(frame_predictions)
                if chart_image:
                    content.append(chart_image)
                    content.append(Paragraph("Frame-by-frame deepfake probability", self.styles["Caption"]))
        
        return content
    
    def _create_recommendations_section(self, detection_result: Dict[str, Any]) -> List[Any]:
        """Create the recommendations section of the report"""
        content = []
        
        # Add section title
        content.append(Paragraph("Recommendations", self.styles["SectionTitle"]))
        content.append(Spacer(1, 0.1 * inch))
        
        # Generate recommendations based on detection result
        is_deepfake = detection_result.get("probability", 0) > 0.5
        confidence = detection_result.get("confidence", 0)
        
        recommendations = []
        
        if is_deepfake:
            # Recommendations for deepfakes
            recommendations.append(
                "This media has been identified as likely manipulated. Exercise caution before sharing."
            )
            
            if confidence > 0.8:
                recommendations.append(
                    "High confidence detection: Consider this media to be artificially generated or manipulated."
                )
            else:
                recommendations.append(
                    "Moderate confidence detection: While likely manipulated, verify with additional methods if critical."
                )
            
            recommendations.append(
                "Look for visual artifacts, unnatural skin tones, or inconsistent lighting that may indicate manipulation."
            )
            
            if detection_result.get("detectionType") == "video":
                recommendations.append(
                    "Check for temporal inconsistencies like unnatural motion or flickering between frames."
                )
        else:
            # Recommendations for authentic media
            recommendations.append(
                "This media has been identified as likely authentic."
            )
            
            if confidence > 0.8:
                recommendations.append(
                    "High confidence detection: This media appears to be genuine."
                )
            else:
                recommendations.append(
                    "Moderate confidence detection: While likely authentic, verify with additional methods if critical."
                )
            
            recommendations.append(
                "Even authentic media can be presented out of context. Verify the source and context."
            )
        
        # Add general recommendations
        recommendations.append(
            "For critical decisions, use multiple detection methods and human verification."
        )
        
        # Create list of recommendations
        recommendation_items = []
        for rec in recommendations:
            recommendation_items.append(ListItem(Paragraph(rec, self.styles["BodyText"])))
        
        recommendation_list = ListFlowable(
            recommendation_items,
            bulletType='bullet',
            leftIndent=20
        )
        
        content.append(recommendation_list)
        
        return content
    
    def _create_technical_details_section(self, detection_result: Dict[str, Any]) -> List[Any]:
        """Create the technical details section of the report"""
        content = []
        
        # Add section title
        content.append(Paragraph("Technical Details", self.styles["SectionTitle"]))
        content.append(Spacer(1, 0.1 * inch))
        
        # Add model information
        model_name = detection_result.get("model", "Standard detection model")
        content.append(Paragraph(f"<b>Detection Model:</b> {model_name}", self.styles["BodyText"]))
        
        # Add processing information
        processing_time = detection_result.get("processingTime", 0)
        content.append(Paragraph(f"<b>Processing Time:</b> {processing_time} ms", self.styles["BodyText"]))
        
        # Add detection type
        detection_type = detection_result.get("detectionType", "Unknown")
        content.append(Paragraph(f"<b>Detection Type:</b> {detection_type.capitalize()}", self.styles["BodyText"]))
        
        # Add additional technical details if available
        if "metadata" in detection_result and "technical_details" in detection_result["metadata"]:
            tech_details = detection_result["metadata"]["technical_details"]
            
            # Add model version if available
            if "model_version" in tech_details:
                content.append(Paragraph(f"<b>Model Version:</b> {tech_details['model_version']}", self.styles["BodyText"]))
            
            # Add detection parameters if available
            if "detection_parameters" in tech_details:
                content.append(Spacer(1, 0.1 * inch))
                content.append(Paragraph("Detection Parameters:", self.styles["BodyText"]))
                
                params = tech_details["detection_parameters"]
                param_items = []
                for key, value in params.items():
                    param_text = f"{key}: {value}"
                    param_items.append(ListItem(Paragraph(param_text, self.styles["BodyText"])))
                
                if param_items:
                    param_list = ListFlowable(
                        param_items,
                        bulletType='bullet',
                        leftIndent=20
                    )
                    content.append(param_list)
        
        return content
    
    def _create_probability_gauge(self, probability: float) -> Optional[Image]:
        """Create a probability gauge visualization"""
        try:
            # Create figure
            plt.figure(figsize=(5, 2.5))
            
            # Create gauge
            ax = plt.subplot(111)
            
            # Draw gauge background
            ax.add_patch(plt.Rectangle((0, 0), 1, 0.3, color='lightgray', alpha=0.5))
            
            # Draw gauge fill
            if probability <= 0.5:
                color = 'green'
            elif probability <= 0.7:
                color = 'orange'
            else:
                color = 'red'
            
            ax.add_patch(plt.Rectangle((0, 0), probability, 0.3, color=color, alpha=0.7))
            
            # Add labels
            ax.text(0, 0.4, "Authentic", fontsize=10, ha='left')
            ax.text(1, 0.4, "Deepfake", fontsize=10, ha='right')
            
            # Add probability text
            ax.text(probability, 0.15, f"{probability*100:.1f}%", 
                   fontsize=12, ha='center', va='center', 
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
            
            # Add marker
            ax.add_patch(plt.Rectangle((probability-0.01, 0), 0.02, 0.4, color='black'))
            
            # Set limits and remove axes
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 0.5)
            ax.axis('off')
            
            # Save to BytesIO
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            
            # Create Image object
            buf.seek(0)
            return Image(buf, width=4*inch, height=2*inch)
            
        except Exception as e:
            logger.error(f"Error creating probability gauge: {str(e)}")
            return None
    
    def _create_frame_predictions_chart(self, frame_predictions: List[Dict[str, Any]]) -> Optional[Image]:
        """Create a chart of frame-by-frame predictions"""
        try:
            # Extract frame indices and probabilities
            frames = []
            probs = []
            
            for pred in frame_predictions:
                frames.append(pred.get("frame", 0))
                probs.append(pred.get("probability", 0))
            
            if not frames:
                return None
            
            # Create figure
            plt.figure(figsize=(6, 3))
            
            # Create line chart
            plt.plot(frames, probs, 'r-', linewidth=2)
            
            # Add threshold line
            plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.7)
            
            # Add labels
            plt.xlabel("Frame")
            plt.ylabel("Deepfake Probability")
            plt.title("Frame-by-Frame Analysis")
            
            # Set limits
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            
            # Save to BytesIO
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            
            # Create Image object
            buf.seek(0)
            return Image(buf, width=5*inch, height=2.5*inch)
            
        except Exception as e:
            logger.error(f"Error creating frame predictions chart: {str(e)}")
            return None
    
    def _decode_image_data(self, image_data: str) -> Optional[BytesIO]:
        """Decode base64 image data"""
        try:
            # Check if it's a base64 string
            if isinstance(image_data, str) and image_data.startswith(('data:image', 'base64:')):
                # Extract the base64 part
                if ',' in image_data:
                    image_data = image_data.split(',', 1)[1]
                elif 'base64:' in image_data:
                    image_data = image_data.replace('base64:', '', 1)
                
                # Decode base64
                image_bytes = base64.b64decode(image_data)
                return BytesIO(image_bytes)
            
            # If it's already bytes, wrap in BytesIO
            elif isinstance(image_data, bytes):
                return BytesIO(image_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error decoding image data: {str(e)}")
            return None