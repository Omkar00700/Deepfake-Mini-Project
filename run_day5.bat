@echo off
echo Starting Deepfake Detector Day 5 Implementation...

REM Create necessary directories if they don't exist
mkdir uploads 2>nul
mkdir results 2>nul
mkdir reports 2>nul
mkdir visualizations 2>nul
mkdir batch_results 2>nul
mkdir temp 2>nul

REM Start the API server in a new window
start cmd /k "python enhanced_api.py"

REM Wait for the API server to start
echo Waiting for API server to start...
timeout /t 5 /nobreak > nul

REM Start the frontend development server
echo Starting frontend development server...
echo.
echo IMPORTANT: Access the application at http://localhost:8080
echo Advanced Dashboard: http://localhost:8080/advanced-dashboard
echo Multi-Modal Analysis: http://localhost:8080/multimodal-analysis
echo.
npm run dev

echo All services started successfully!