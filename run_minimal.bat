@echo off
echo Starting Minimal Deepfake Detector Implementation...

REM Create necessary directories if they don't exist
mkdir results 2>nul

REM Start the API server in a new window
start cmd /k "python minimal_api.py"

REM Wait for the API server to start
echo Waiting for API server to start...
timeout /t 5 /nobreak > nul

REM Open the test page
echo Opening test page...
start test.html

echo Minimal API server started successfully!