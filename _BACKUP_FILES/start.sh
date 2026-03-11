#!/bin/bash

echo "========================================"
echo "Thyroid Cancer Prediction System"
echo "========================================"
echo ""

echo "[1/3] Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi
echo ""

echo "[2/3] Training models (this may take 2-5 minutes)..."
python train_models.py
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to train models"
    exit 1
fi
echo ""

echo "[3/3] Starting web application..."
echo ""
echo "========================================"
echo "Application will be available at:"
echo "http://localhost:5000"
echo "========================================"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
