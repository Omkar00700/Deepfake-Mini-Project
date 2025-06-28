@echo off
echo Starting Deepfake Detector Day 4 Implementation...

REM Create necessary directories if they don't exist
mkdir uploads 2>nul
mkdir results 2>nul
mkdir reports 2>nul
mkdir visualizations 2>nul

REM Start the API server in a new window
start cmd /k "python simple_api.py"

REM Wait for the API server to start
echo Waiting for API server to start...
timeout /t 5 /nobreak > nul

REM Start the frontend development server
echo Starting frontend development server...
echo.
echo IMPORTANT: Access the application at http://localhost:8080
echo Simple Dashboard page: http://localhost:8080/dashboard
echo Full Dashboard page: http://localhost:8080/full-dashboard
echo.
echo If you see a white screen, try the test.html file first to check if the API is working
echo.
npm run dev

echo All services started successfully!