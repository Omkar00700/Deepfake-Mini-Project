@echo off
echo Starting DeepDefend Project...
echo.

REM Navigate to the project directory
cd /d %~dp0

REM Install required dependencies
echo Installing required dependencies...
pip install flask flask-cors werkzeug
echo.

REM Start the backend
echo Starting the enhanced backend...
start cmd /k "python enhanced_backend.py"
echo.

REM Wait for backend to start
timeout /t 5 /nobreak > nul

REM Start the frontend
echo Starting the frontend...
start cmd /k "npm run dev"
echo.

echo DeepDefend is now running!
echo.
echo IMPORTANT: Access the application at the URL shown in the frontend terminal
echo (typically http://localhost:5173)
echo.
echo Press any key to exit this window (but keep the other windows open)...
pause
