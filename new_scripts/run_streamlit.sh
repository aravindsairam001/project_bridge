#!/bin/bash

# Script to run Streamlit with warning suppression
# Usage: ./run_streamlit.sh

# Suppress PyTorch warnings
export PYTHONWARNINGS="ignore::UserWarning"

# Suppress matplotlib warnings  
export MPLBACKEND="Agg"

# Run Streamlit on network interface
echo "🚀 Starting Streamlit Bridge Defect Detection App..."
echo "📡 Access from other devices at: http://$(hostname -I | awk '{print $1}'):8501"

streamlit run bridge_app.py --server.address=0.0.0.0 --server.port=8501
