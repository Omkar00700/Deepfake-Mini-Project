"""
Test script for the API endpoints
"""

import requests
import json
import os
import sys

def test_models_endpoint():
    """Test the models endpoint"""
    url = "http://localhost:5000/api/models"
    response = requests.get(url)
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("Models endpoint test passed!")
    else:
        print("Models endpoint test failed!")

def test_detect_endpoint():
    """Test the detect endpoint with an image"""
    url = "http://localhost:5000/api/detect"
    
    # Get test image path
    test_image_path = os.path.join(os.path.dirname(__file__), "test_faces", "test_face.jpg")
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return
    
    # Create form data
    files = {"file": open(test_image_path, "rb")}
    data = {
        "model": "efficientnet",
        "ensemble": "true",
        "indianEnhancement": "true"
    }
    
    # Send request
    response = requests.post(url, files=files, data=data)
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Detection ID: {result.get('detection_id')}")
        print(f"Probability: {result.get('result', {}).get('probability')}")
        print(f"Confidence: {result.get('result', {}).get('confidence')}")
        print("Detect endpoint test passed!")
    else:
        print(f"Response: {response.text}")
        print("Detect endpoint test failed!")

def main():
    """Main function"""
    print("Testing API endpoints...")
    
    # Test models endpoint
    print("\n=== Testing models endpoint ===")
    test_models_endpoint()
    
    # Test detect endpoint
    print("\n=== Testing detect endpoint ===")
    test_detect_endpoint()

if __name__ == "__main__":
    main()