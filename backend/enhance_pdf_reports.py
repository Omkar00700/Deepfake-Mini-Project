"""
Enhanced PDF Report Integration for DeepDefend
This script integrates enhanced PDF reports with Indian face-specific information
"""

import os
import sys
import logging
import argparse
import json
import time
from enhanced_pdf_report import EnhancedPDFReportGenerator
from pdf_report_generator import PDFReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def integrate_enhanced_reports():
    """
    Integrate enhanced PDF reports with Indian face-specific information
    
    This function modifies the app.py file to use the enhanced PDF report generator
    instead of the standard one.
    """
    try:
        # Path to app.py
        app_py_path = os.path.join(os.path.dirname(__file__), "app.py")
        
        # Read app.py
        with open(app_py_path, 'r') as f:
            app_py_content = f.read()
        
        # Check if already integrated
        if "from enhanced_pdf_report import EnhancedPDFReportGenerator" in app_py_content:
            logger.info("Enhanced PDF reports are already integrated")
            return {
                "success": True,
                "message": "Enhanced PDF reports are already integrated"
            }
        
        # Replace standard PDF generator with enhanced one
        app_py_content = app_py_content.replace(
            "from pdf_report_generator import PDFReportGenerator",
            "from enhanced_pdf_report import EnhancedPDFReportGenerator"
        )
        
        app_py_content = app_py_content.replace(
            "pdf_generator = PDFReportGenerator(output_dir=reports_dir)",
            "pdf_generator = EnhancedPDFReportGenerator(output_dir=reports_dir)"
        )
        
        # Write modified app.py
        with open(app_py_path, 'w') as f:
            f.write(app_py_content)
        
        logger.info("Successfully integrated enhanced PDF reports")
        
        return {
            "success": True,
            "message": "Successfully integrated enhanced PDF reports"
        }
        
    except Exception as e:
        logger.error(f"Error integrating enhanced PDF reports: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

def generate_sample_report(output_dir):
    """
    Generate a sample enhanced PDF report
    
    Args:
        output_dir: Directory to save the sample report
        
    Returns:
        Path to the generated report
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create sample detection result
        detection_result = {
            "detectionType": "image",
            "imageName": "sample_indian_face.jpg",
            "probability": 0.15,  # Low probability (likely authentic)
            "confidence": 0.85,
            "processingTime": 1250,  # ms
            "regions": [
                {
                    "bbox": [100, 150, 200, 200],
                    "probability": 0.15,
                    "confidence": 0.85,
                    "skin_tone": {
                        "success": True,
                        "indian_tone": {
                            "type": "wheatish",
                            "name": "Wheatish",
                            "score": 0.35
                        },
                        "fitzpatrick_type": {
                            "type": 3,
                            "name": "Type III (Medium)",
                            "score": 0.35
                        },
                        "dominant_color": {
                            "rgb": [224, 172, 138],
                            "hex": "#e0ac8a"
                        },
                        "evenness_score": 0.92,
                        "texture_score": 0.78
                    },
                    "skin_anomalies": {
                        "success": True,
                        "anomalies": [],
                        "anomaly_score": 0.0
                    }
                }
            ],
            "model": "indian_deepfake_detector",
            "modelVersion": "1.0.0",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Generate enhanced PDF report
        pdf_generator = EnhancedPDFReportGenerator(output_dir=output_dir)
        pdf_path = pdf_generator.generate_report(detection_result)
        
        logger.info(f"Generated sample enhanced PDF report: {pdf_path}")
        
        return pdf_path
        
    except Exception as e:
        logger.error(f"Error generating sample report: {str(e)}", exc_info=True)
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Integrate enhanced PDF reports")
    parser.add_argument("--integrate", action="store_true", help="Integrate enhanced PDF reports")
    parser.add_argument("--generate-sample", action="store_true", help="Generate a sample enhanced PDF report")
    parser.add_argument("--output-dir", type=str, default="enhanced_reports", help="Directory to save sample reports")
    
    args = parser.parse_args()
    
    if args.integrate:
        # Integrate enhanced PDF reports
        result = integrate_enhanced_reports()
        
        if result["success"]:
            logger.info(result["message"])
        else:
            logger.error(f"Integration failed: {result['error']}")
            sys.exit(1)
    
    if args.generate_sample:
        # Generate sample report
        pdf_path = generate_sample_report(args.output_dir)
        
        if pdf_path:
            logger.info(f"Sample report generated: {pdf_path}")
        else:
            logger.error("Failed to generate sample report")
            sys.exit(1)
    
    if not (args.integrate or args.generate_sample):
        parser.print_help()

if __name__ == "__main__":
    main()