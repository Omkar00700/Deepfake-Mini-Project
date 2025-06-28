@echo off
echo DeepDefend Indian Face Enhancement
echo ================================
echo.

REM Set paths
set DATA_DIR=data\indian_faces
set TEST_RESULTS_DIR=test_results
set FINE_TUNED_DIR=fine_tuned_models
set VALIDATION_DIR=validation_results
set REPORTS_DIR=enhanced_reports

REM Create directories
mkdir %DATA_DIR%\train\real %DATA_DIR%\train\fake %DATA_DIR%\val\real %DATA_DIR%\val\fake %DATA_DIR%\test\real %DATA_DIR%\test\fake %TEST_RESULTS_DIR% %FINE_TUNED_DIR% %VALIDATION_DIR% %REPORTS_DIR%

echo.
echo Step 1: Collect Indian Face Data
echo -------------------------------
echo.
echo Please provide the path to your real Indian face images:
set /p REAL_DIR="Real faces directory: "
echo.
echo Please provide the path to your fake (deepfake) Indian face images:
set /p FAKE_DIR="Fake faces directory: "
echo.

python backend\collect_indian_faces.py --real-dir "%REAL_DIR%" --fake-dir "%FAKE_DIR%" --organize --analyze

echo.
echo Step 2: Run Initial Tests
echo ------------------------
echo.

python backend\test_indian_faces.py --dataset-dir %DATA_DIR%\test --output-dir %TEST_RESULTS_DIR%

echo.
echo Step 3: Fine-tune Models
echo ----------------------
echo.

python backend\run_indian_finetuning.py --dataset-dir %DATA_DIR% --model-type efficientnet --epochs 10 --fine-tune-epochs 5

echo.
echo Step 4: Validate Improvements
echo ---------------------------
echo.

REM Get original model path
echo Please provide the path to your original model:
set /p ORIGINAL_MODEL="Original model path: "
echo.

python backend\validate_improvements.py --original-model "%ORIGINAL_MODEL%" --fine-tuned-model %FINE_TUNED_DIR%\indian_deepfake_detector.h5 --test-dataset %DATA_DIR%\test --output-dir %VALIDATION_DIR%

echo.
echo Step 5: Deploy Enhanced Reports
echo ----------------------------
echo.

python backend\enhance_pdf_reports.py --integrate --generate-sample --output-dir %REPORTS_DIR%

echo.
echo Enhancement process completed!
echo.
echo Please check the following directories for results:
echo - Test results: %TEST_RESULTS_DIR%
echo - Fine-tuned model: %FINE_TUNED_DIR%
echo - Validation results: %VALIDATION_DIR%
echo - Enhanced reports: %REPORTS_DIR%
echo.
echo For more information, see INDIAN_FACES_README.md
echo.

pause