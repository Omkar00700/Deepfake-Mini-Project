"""
Test script for enhanced deepfake detection
Tests the improved detection system with Indian face specialization
"""

import os
import sys
import argparse
import logging
import time
import json
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_image_detection(image_path: str, save_results: bool = True) -> Dict[str, Any]:
    """
    Test enhanced image detection
    
    Args:
        image_path: Path to test image
        save_results: Whether to save results to file
        
    Returns:
        Detection results
    """
    try:
        logger.info(f"Testing enhanced image detection on: {image_path}")
        
        # Import enhanced detection handler
        from enhanced_detection_handler import process_image_enhanced
        
        # Process image
        start_time = time.time()
        probability, confidence, regions = process_image_enhanced(image_path)
        processing_time = time.time() - start_time
        
        # Create result
        result = {
            "probability": float(probability),
            "confidence": float(confidence),
            "regions": regions,
            "imageName": os.path.basename(image_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "detectionType": "image",
            "model": "enhanced_indian_detection",
            "processingTime": int(processing_time * 1000)  # Convert to ms
        }
        
        # Print results
        logger.info(f"Detection results:")
        logger.info(f"  Probability: {probability:.4f}")
        logger.info(f"  Confidence: {confidence:.4f}")
        logger.info(f"  Processing time: {processing_time:.2f} seconds")
        logger.info(f"  Regions detected: {len(regions)}")
        
        # Save results if requested
        if save_results:
            output_dir = "test_results"
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(
                output_dir, 
                f"result_{os.path.splitext(os.path.basename(image_path))[0]}.json"
            )
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Results saved to: {output_path}")
            
            # Generate PDF report
            try:
                from pdf_report_generator import PDFReportGenerator
                
                pdf_generator = PDFReportGenerator(output_dir=output_dir)
                pdf_path = pdf_generator.generate_report(result)
                
                logger.info(f"PDF report generated: {pdf_path}")
            except Exception as e:
                logger.error(f"Error generating PDF report: {str(e)}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error testing image detection: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "success": False
        }

def test_video_detection(video_path: str, save_results: bool = True) -> Dict[str, Any]:
    """
    Test enhanced video detection
    
    Args:
        video_path: Path to test video
        save_results: Whether to save results to file
        
    Returns:
        Detection results
    """
    try:
        logger.info(f"Testing enhanced video detection on: {video_path}")
        
        # Import enhanced detection handler
        from enhanced_detection_handler import process_video_enhanced
        
        # Process video
        start_time = time.time()
        probability, confidence, frame_count, regions = process_video_enhanced(video_path)
        processing_time = time.time() - start_time
        
        # Create result
        result = {
            "probability": float(probability),
            "confidence": float(confidence),
            "frameCount": frame_count,
            "regions": regions,
            "imageName": os.path.basename(video_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "detectionType": "video",
            "model": "enhanced_indian_detection",
            "processingTime": int(processing_time * 1000)  # Convert to ms
        }
        
        # Print results
        logger.info(f"Detection results:")
        logger.info(f"  Probability: {probability:.4f}")
        logger.info(f"  Confidence: {confidence:.4f}")
        logger.info(f"  Processing time: {processing_time:.2f} seconds")
        logger.info(f"  Frames processed: {frame_count}")
        
        # Save results if requested
        if save_results:
            output_dir = "test_results"
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(
                output_dir, 
                f"result_{os.path.splitext(os.path.basename(video_path))[0]}.json"
            )
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Results saved to: {output_path}")
            
            # Generate PDF report
            try:
                from pdf_report_generator import PDFReportGenerator
                
                pdf_generator = PDFReportGenerator(output_dir=output_dir)
                pdf_path = pdf_generator.generate_report(result)
                
                logger.info(f"PDF report generated: {pdf_path}")
            except Exception as e:
                logger.error(f"Error generating PDF report: {str(e)}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error testing video detection: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "success": False
        }

def test_gan_detection(image_path: str) -> Dict[str, Any]:
    """
    Test GAN detection
    
    Args:
        image_path: Path to test image
        
    Returns:
        GAN detection results
    """
    try:
        logger.info(f"Testing GAN detection on: {image_path}")
        
        # Import GAN detector
        from gan_detector import detect_gan
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        # Detect GAN
        result = detect_gan(image)
        
        # Print results
        logger.info(f"GAN detection results:")
        logger.info(f"  GAN probability: {result.get('gan_probability', 0):.4f}")
        logger.info(f"  Confidence: {result.get('confidence', 0):.4f}")
        logger.info(f"  Method: {result.get('method', 'unknown')}")
        
        if 'scores' in result:
            logger.info(f"  Scores:")
            for name, score in result['scores'].items():
                logger.info(f"    {name}: {score:.4f}")
        
        if 'artifacts' in result:
            logger.info(f"  Artifacts detected: {len(result['artifacts'])}")
            for artifact in result['artifacts']:
                logger.info(f"    {artifact.get('description', 'Unknown artifact')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error testing GAN detection: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "success": False
        }

def test_skin_tone_analysis(image_path: str) -> Dict[str, Any]:
    """
    Test skin tone analysis
    
    Args:
        image_path: Path to test image
        
    Returns:
        Skin tone analysis results
    """
    try:
        logger.info(f"Testing skin tone analysis on: {image_path}")
        
        # Import skin tone analyzer
        from skin_tone_analyzer import SkinToneAnalyzer
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        # Create analyzer
        analyzer = SkinToneAnalyzer()
        
        # Analyze skin tone
        result = analyzer.analyze_skin_tone(image)
        
        # Print results
        logger.info(f"Skin tone analysis results:")
        
        if result.get('success', False):
            indian_tone = result.get('indian_tone', {})
            if indian_tone:
                logger.info(f"  Indian skin tone: {indian_tone.get('name', 'Unknown')}")
                logger.info(f"  Score: {indian_tone.get('score', 0):.4f}")
            
            fitzpatrick = result.get('fitzpatrick_type', {})
            if fitzpatrick:
                logger.info(f"  Fitzpatrick type: {fitzpatrick.get('name', 'Unknown')}")
            
            logger.info(f"  Evenness score: {result.get('evenness_score', 0):.4f}")
            logger.info(f"  Texture score: {result.get('texture_score', 0):.4f}")
        else:
            logger.warning(f"Skin tone analysis failed: {result.get('error', 'Unknown error')}")
        
        # Detect skin anomalies
        anomalies = analyzer.detect_skin_anomalies(image)
        
        logger.info(f"Skin anomaly detection results:")
        
        if anomalies.get('success', False):
            logger.info(f"  Anomaly score: {anomalies.get('anomaly_score', 0):.4f}")
            logger.info(f"  Anomalies detected: {len(anomalies.get('anomalies', []))}")
            
            for anomaly in anomalies.get('anomalies', []):
                logger.info(f"    {anomaly.get('description', 'Unknown anomaly')}: {anomaly.get('severity', 0):.4f}")
        else:
            logger.warning(f"Skin anomaly detection failed: {anomalies.get('error', 'Unknown error')}")
        
        return {
            "skin_tone": result,
            "anomalies": anomalies
        }
        
    except Exception as e:
        logger.error(f"Error testing skin tone analysis: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "success": False
        }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test enhanced deepfake detection")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--video", type=str, help="Path to test video")
    parser.add_argument("--gan", action="store_true", help="Test GAN detection")
    parser.add_argument("--skin", action="store_true", help="Test skin tone analysis")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    
    args = parser.parse_args()
    
    # Check if any arguments were provided
    if not (args.image or args.video or args.gan or args.skin or args.all):
        parser.print_help()
        return
    
    # Run tests
    if args.image or args.all:
        if args.image:
            image_path = args.image
        else:
            # Use default test image
            image_path = "test_images/test_image.jpg"
            if not os.path.exists(image_path):
                logger.error(f"Default test image not found: {image_path}")
                return
        
        test_image_detection(image_path, not args.no_save)
    
    if args.video or args.all:
        if args.video:
            video_path = args.video
        else:
            # Use default test video
            video_path = "test_videos/test_video.mp4"
            if not os.path.exists(video_path):
                logger.error(f"Default test video not found: {video_path}")
                return
        
        test_video_detection(video_path, not args.no_save)
    
    if args.gan or args.all:
        if args.image:
            image_path = args.image
        else:
            # Use default test image
            image_path = "test_images/test_image.jpg"
            if not os.path.exists(image_path):
                logger.error(f"Default test image not found: {image_path}")
                return
        
        test_gan_detection(image_path)
    
    if args.skin or args.all:
        if args.image:
            image_path = args.image
        else:
            # Use default test image
            image_path = "test_images/test_image.jpg"
            if not os.path.exists(image_path):
                logger.error(f"Default test image not found: {image_path}")
                return
        
        test_skin_tone_analysis(image_path)

if __name__ == "__main__":
    main()