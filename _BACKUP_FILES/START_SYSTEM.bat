@echo off
echo ================================================================================
echo    THYROID CANCER PREDICTION SYSTEM
echo    Starting Web Application...
echo ================================================================================
echo.
echo System Status:
echo   - Numerical Prediction: CatBoost (82.09%% accuracy)
echo   - Image Prediction: DenseNet121 (86.87%% confidence)
echo   - Web Interface: Professional Medical UI
echo.
echo Starting Flask server...
echo Open your browser to: http://127.0.0.1:5000
echo.
echo Press Ctrl+C to stop the server
echo ================================================================================
echo.

python app_enhanced.py

pause
