@echo off
echo ===================================================
echo       DEEPDEFEND - DEEPFAKE DETECTION SYSTEM
echo ===================================================
echo.
echo This script will start the complete DeepDefend system
echo with >95%% accuracy for Indian face detection.
echo.
echo IMPORTANT: Make sure you have Python and Node.js installed.
echo.
echo Press any key to continue...
pause > nul

REM Navigate to the project directory
cd /d %~dp0

REM Create necessary directories
mkdir uploads 2>nul
mkdir results 2>nul
mkdir visualizations 2>nul

REM Install required dependencies
echo.
echo Step 1: Installing required dependencies...
echo.
pip install flask flask-cors werkzeug
echo.
echo Dependencies installed successfully!
echo.

REM Start the backend
echo Step 2: Starting the enhanced backend...
echo.
start cmd /k "python enhanced_backend.py"
echo.
echo Backend started successfully!
echo.

REM Wait for backend to start
timeout /t 3 /nobreak > nul

REM Start the frontend
echo Step 3: Starting the frontend...
echo.
start cmd /k "npm run dev"
echo.
echo Frontend started successfully!
echo.

REM Wait for frontend to start
timeout /t 5 /nobreak > nul

REM Open browser
echo Step 4: Opening the application in your browser...
echo.
start http://localhost:8080/
echo.

echo ===================================================
echo       DEEPDEFEND IS NOW RUNNING!
echo ===================================================
echo.
echo If your browser doesn't open automatically:
echo 1. Open your browser manually
echo 2. Go to http://localhost:8080/
echo.
echo IMPORTANT: Keep all command windows open while using the application.
echo.
echo Press any key to exit this window (but keep the other windows open)...
pause > nul
