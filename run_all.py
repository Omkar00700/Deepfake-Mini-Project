"""
Run both the backend and frontend for DeepDefend
"""

import os
import sys
import subprocess
import time
import threading

def run_backend():
    """Run the backend server"""
    print("Starting backend server...")
    subprocess.run([sys.executable, "simple_backend.py"], check=True)

def run_frontend():
    """Run the frontend server"""
    print("Starting frontend server...")
    subprocess.run(["npm", "run", "dev"], check=True)

def main():
    """Run both servers"""
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=run_backend)
    backend_thread.daemon = True
    backend_thread.start()
    
    # Wait for backend to start
    print("Waiting for backend to start...")
    time.sleep(3)
    
    # Start frontend
    try:
        run_frontend()
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(f"Error running frontend: {str(e)}")

if __name__ == "__main__":
    print("Starting DeepDefend...")
    main()
