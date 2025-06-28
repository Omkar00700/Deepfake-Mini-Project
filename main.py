"""
Main entry point for the Deepfake Detector
Runs the API server and provides a command-line interface
"""

import os
import sys
import argparse
import logging
import time
import threading
from flask import Flask
from api_endpoints import app as api_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('deepfake_detector.log')
    ]
)
logger = logging.getLogger(__name__)

def run_api_server(host='0.0.0.0', port=5000, debug=False):
    """
    Run the API server
    """
    logger.info(f"Starting API server on {host}:{port}")
    api_app.run(host=host, port=port, debug=debug)

def run_frontend_server():
    """
    Run the frontend server (if available)
    """
    try:
        # Check if npm is installed
        import subprocess
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Starting frontend server")
            
            # Change to project directory
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            
            # Start frontend server
            subprocess.Popen(['npm', 'run', 'dev'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logger.info("Frontend server started")
            return True
        else:
            logger.warning("npm not found, frontend server not started")
            return False
    
    except Exception as e:
        logger.error(f"Error starting frontend server: {str(e)}")
        return False

def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(description='Deepfake Detector')
    parser.add_argument('--api-only', action='store_true', help='Run API server only')
    parser.add_argument('--frontend-only', action='store_true', help='Run frontend server only')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='API server host')
    parser.add_argument('--port', type=int, default=5000, help='API server port')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    try:
        if args.frontend_only:
            # Run frontend server only
            success = run_frontend_server()
            
            if not success:
                logger.error("Failed to start frontend server")
                return 1
        elif args.api_only:
            # Run API server only
            run_api_server(host=args.host, port=args.port, debug=args.debug)
        else:
            # Run both servers
            # Start frontend server in a separate thread
            frontend_thread = threading.Thread(target=run_frontend_server)
            frontend_thread.daemon = True
            frontend_thread.start()
            
            # Run API server in main thread
            run_api_server(host=args.host, port=args.port, debug=args.debug)
    
    except KeyboardInterrupt:
        logger.info("Shutting down servers")
    
    except Exception as e:
        logger.error(f"Error running servers: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())