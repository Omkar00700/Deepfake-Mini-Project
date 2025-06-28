"""
Setup and run script for DeepDefend
This script installs the required dependencies and runs the enhanced backend
"""

import os
import sys
import subprocess
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install the required dependencies"""
    logger.info("Installing dependencies...")
    
    try:
        # Install dependencies using pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {str(e)}")
        return False

def create_directories():
    """Create the required directories"""
    logger.info("Creating directories...")
    
    directories = [
        "models",
        "uploads",
        "results",
        "visualizations",
        "feedback",
        "data"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return True

def run_backend():
    """Run the enhanced backend"""
    logger.info("Starting enhanced backend...")
    
    try:
        # Run the enhanced backend
        subprocess.check_call([sys.executable, "run_enhanced_backend.py"])
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run enhanced backend: {str(e)}")
        return False

def run_frontend():
    """Run the frontend"""
    logger.info("Starting frontend...")
    
    try:
        # Run the frontend
        subprocess.check_call(["npm", "start"], cwd=".")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run frontend: {str(e)}")
        return False

def main():
    """Main function"""
    logger.info("Setting up and running DeepDefend...")
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies")
        return False
    
    # Create directories
    if not create_directories():
        logger.error("Failed to create directories")
        return False
    
    # Run backend
    if not run_backend():
        logger.error("Failed to run backend")
        return False
    
    return True

if __name__ == "__main__":
    main()
