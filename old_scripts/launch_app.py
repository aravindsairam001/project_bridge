#!/usr/bin/env python3
"""
Bridge Defect Detection App Launcher
Run this script to start the Streamlit web application.
"""

import os
import sys
import subprocess
import importlib.util

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'torch', 'torchvision', 'segmentation_models_pytorch',
        'opencv-python', 'PIL', 'numpy', 'pandas', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        # Handle package name variations
        if package == 'opencv-python':
            package_import = 'cv2'
        elif package == 'PIL':
            package_import = 'PIL'
        elif package == 'segmentation_models_pytorch':
            package_import = 'segmentation_models_pytorch'
        else:
            package_import = package
            
        spec = importlib.util.find_spec(package_import)
        if spec is None:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("Installing missing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "streamlit_requirements.txt"
        ])
        print("✓ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        return False

def check_model_file():
    """Check if the model file exists"""
    model_files = ["dacl10k_ninja.pth", "dacl10k_deeplabv3plus.pth", "dacl10k15_deeplabv3plus.pth"]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✓ Found model file: {model_file}")
            return model_file
    
    print("⚠️  No model file found!")
    print("   Please train the model first using: python train_dacl10k.py")
    print("   Or ensure the model file is in the current directory.")
    return None

def main():
    print("🌉 Bridge Defect Detection App Launcher")
    print("=" * 50)
    
    # Check dependencies
    print("Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        if input("Install missing dependencies? (y/n): ").lower().startswith('y'):
            if not install_dependencies():
                return
        else:
            print("Cannot run app without required dependencies.")
            return
    else:
        print("✓ All dependencies are installed!")
    
    # Check model file
    print("\nChecking for model file...")
    model_file = check_model_file()
    
    if not model_file:
        if input("Continue anyway? (y/n): ").lower().startswith('y'):
            print("Note: You'll need to specify the correct model path in the app.")
        else:
            return
    
    # Start the app
    print("\n🚀 Starting Streamlit app...")
    print("The app will open in your browser automatically.")
    print("If it doesn't open, navigate to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        subprocess.run(["streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n\n👋 App stopped. Thank you for using Bridge Defect Detection!")
    except FileNotFoundError:
        print("Error: Streamlit not found. Please install it using:")
        print("pip install streamlit")

if __name__ == "__main__":
    main()
