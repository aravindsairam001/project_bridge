#!/bin/bash

# Script to run the Bridge Defect Detection Streamlit App

echo "Starting Bridge Defect Detection App..."
echo "Make sure you have trained the model and have 'dacl10k_ninja.pth' in the current directory"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit is not installed. Installing dependencies..."
    pip install -r streamlit_requirements.txt
fi

# Check if model file exists
if [ ! -f "dacl10k_ninja.pth" ]; then
    echo "Warning: Model file 'dacl10k_ninja.pth' not found!"
    echo "Please train the model first using: python train_dacl10k.py"
    echo ""
fi

echo "Starting Streamlit app..."
echo "The app will open in your browser automatically."
echo "If it doesn't open, navigate to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the app"
echo ""

streamlit run streamlit_app.py
