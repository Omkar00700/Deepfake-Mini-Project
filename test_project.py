"""
Comprehensive Test Script for Deepfake Detector Project
"""

import os
import sys
import json
import requests
import time
import logging
import subprocess
import webbrowser
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
API_URL = "http://localhost:5000"
FRONTEND_URL = "http://localhost:8080"
TEST_TIMEOUT = 10  # seconds

def check_directory_structure():
    """Check if the required directories exist"""
    logger.info("Checking directory structure...")
    
    required_dirs = [
        "uploads",
        "results",
        "reports",
        "visualizations",
        "src",
        "public"
    ]
    
    for directory in required_dirs:
        if not os.path.isdir(directory):
            logger.warning(f"Directory '{directory}' does not exist. Creating it...")
            os.makedirs(directory, exist_ok=True)
    
    logger.info("Directory structure check completed.")

def check_api_server():
    """Check if the API server is running"""
    logger.info("Checking API server...")
    
    try:
        response = requests.get(f"{API_URL}/api/models", timeout=TEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                logger.info("API server is running and responding correctly.")
                return True
            else:
                logger.error(f"API server returned an error: {data.get('error')}")
                return False
        else:
            logger.error(f"API server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to the API server. Is it running?")
        return False
    except Exception as e:
        logger.error(f"Error checking API server: {str(e)}")
        return False

def test_api_endpoints():
    """Test all API endpoints"""
    logger.info("Testing API endpoints...")
    
    endpoints = [
        {"url": "/api/models", "method": "GET"},
        {"url": "/api/dashboard", "method": "GET"}
    ]
    
    all_passed = True
    
    for endpoint in endpoints:
        url = f"{API_URL}{endpoint['url']}"
        method = endpoint['method']
        
        logger.info(f"Testing {method} {url}...")
        
        try:
            if method == "GET":
                response = requests.get(url, timeout=TEST_TIMEOUT)
            else:
                logger.warning(f"Unsupported method: {method}")
                continue
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    logger.info(f"✅ {method} {url} - Success")
                else:
                    logger.error(f"❌ {method} {url} - API Error: {data.get('error')}")
                    all_passed = False
            else:
                logger.error(f"❌ {method} {url} - Status Code: {response.status_code}")
                all_passed = False
        except Exception as e:
            logger.error(f"❌ {method} {url} - Exception: {str(e)}")
            all_passed = False
    
    if all_passed:
        logger.info("All API endpoints tested successfully.")
    else:
        logger.error("Some API endpoint tests failed.")
    
    return all_passed

def start_minimal_api():
    """Start the minimal API server"""
    logger.info("Starting minimal API server...")
    
    try:
        # Check if the API server is already running
        if check_api_server():
            logger.info("API server is already running.")
            return True
        
        # Start the minimal API server
        process = subprocess.Popen(
            ["python", "minimal_api.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        
        # Wait for the server to start
        logger.info("Waiting for API server to start...")
        time.sleep(5)
        
        # Check if the server is running
        if check_api_server():
            logger.info("Minimal API server started successfully.")
            return True
        else:
            logger.error("Failed to start minimal API server.")
            return False
    except Exception as e:
        logger.error(f"Error starting minimal API server: {str(e)}")
        return False

def open_minimal_frontend():
    """Open the minimal frontend"""
    logger.info("Opening minimal frontend...")
    
    try:
        frontend_path = os.path.abspath("minimal_frontend/index.html")
        webbrowser.open(f"file://{frontend_path}")
        logger.info("Minimal frontend opened successfully.")
        return True
    except Exception as e:
        logger.error(f"Error opening minimal frontend: {str(e)}")
        return False

def run_tests():
    """Run all tests"""
    logger.info("Starting comprehensive project tests...")
    
    # Check directory structure
    check_directory_structure()
    
    # Start minimal API server
    if not start_minimal_api():
        logger.error("Failed to start minimal API server. Aborting tests.")
        return False
    
    # Test API endpoints
    if not test_api_endpoints():
        logger.error("API endpoint tests failed. Aborting tests.")
        return False
    
    # Open minimal frontend
    if not open_minimal_frontend():
        logger.warning("Failed to open minimal frontend. Continuing tests.")
    
    logger.info("All tests completed successfully.")
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)