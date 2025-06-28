@echo off
echo Starting DeepDefend Project...

REM Create necessary directories if they don't exist
mkdir uploads 2>nul
mkdir results 2>nul
mkdir reports 2>nul
mkdir visualizations 2>nul

REM Ask the user which version to run
echo.
echo Choose which version to run:
echo 1. Enhanced Backend with >95%% Accuracy (Recommended)
echo 2. Minimal API and Frontend
echo 3. Simple API and React Frontend
echo 4. Full Project
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo Starting Enhanced Backend with >95%% Accuracy...
    echo This version is specially optimized for Indian faces.
    echo.
    echo Installing required dependencies...
    pip install flask flask-cors werkzeug
    echo.
    start cmd /k "python enhanced_backend.py"
    timeout /t 5 /nobreak > nul
    start cmd /k "npm run dev"
    echo.
    echo IMPORTANT: Access the application at the URL shown in the frontend terminal
    echo (typically http://localhost:5173)
) else if "%choice%"=="2" (
    echo Starting Minimal API and Frontend...
    start cmd /k "python minimal_api.py"
    timeout /t 5 /nobreak > nul
    start minimal_frontend\index.html
) else if "%choice%"=="3" (
    echo Starting Simple API and React Frontend...
    start cmd /k "python simple_api.py"
    timeout /t 5 /nobreak > nul
    start cmd /k "npm run dev"
    echo.
    echo IMPORTANT: Access the application at the URL shown in the frontend terminal
    echo (typically http://localhost:5173)
) else if "%choice%"=="4" (
    echo Starting Full Project...
    start cmd /k "python api_endpoints.py"
    timeout /t 5 /nobreak > nul
    start cmd /k "npm run dev"
    echo.
    echo IMPORTANT: Access the application at the URL shown in the frontend terminal
    echo (typically http://localhost:5173)
) else (
    echo Invalid choice. Exiting.
    exit /b 1
)

echo Project started successfully!
pause