@echo off
echo === Validating Deepfake Detector Project ===

echo.
echo === 1. Installing dependencies ===
pip install -r requirements.txt

echo.
echo === 2. Downloading required models ===
cd backend\models
if not exist shape_predictor_68_face_landmarks.dat (
    echo Downloading dlib face landmark model...
    curl -L "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2" -o shape_predictor_68_face_landmarks.dat.bz2
    echo Extracting...
    python -c "import bz2; open('shape_predictor_68_face_landmarks.dat', 'wb').write(bz2.BZ2File('shape_predictor_68_face_landmarks.dat.bz2').read())"
    del shape_predictor_68_face_landmarks.dat.bz2
)
cd ..\..

echo.
echo === 3. Running backend tests ===
cd backend
python -m pytest tests/

echo.
echo === 4. Running validation ===
python validation.py

echo.
echo === 5. Testing frontend ===
cd ..\frontend
npm test

echo.
echo === 6. Starting services ===
echo Starting backend server...
start cmd /k "cd ..\backend && python app.py"
echo Starting frontend...
start cmd /k "npm run dev"

echo.
echo === Validation complete ===
echo Backend server running at http://localhost:5000
echo Frontend running at http://localhost:8080