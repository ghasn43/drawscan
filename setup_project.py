#!/usr/bin/env python3
"""
Setup script for DXF BOQ Extractor
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_project():
    print("Setting up DXF BOQ Extractor Project...")
    print("="*60)
    
    # Create folder structure
    folders = ['input', 'output']
    files = ['dxf_processor.py', 'boq_generator.py', 'main_app.py', 'requirements.txt', 'run.bat']
    
    for folder in folders:
        Path(folder).mkdir(exist_ok=True)
        print(f"Created folder: {folder}")
    
    # Check if files already exist
    existing_files = [f for f in files if Path(f).exists()]
    if existing_files:
        print(f"\n⚠️  Some files already exist: {', '.join(existing_files)}")
        response = input("Overwrite? (y/n): ").lower()
        if response != 'y':
            print("Setup cancelled.")
            return
    
    print("\n✅ Project structure created successfully!")
    print("\nNext steps:")
    print("1. Place your DXF files in the 'input' folder")
    print("2. Run: python main_app.py")
    print("3. Or double-click 'run.bat' (Windows)")
    print("\nFor help: python main_app.py --help")
    
    # Check Python packages
    print("\nChecking required packages...")
    required_packages = ['ezdxf', 'pandas', 'openpyxl', 'numpy']
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - Not installed")
            install = input(f"Install {package}? (y/n): ").lower()
            if install == 'y':
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"  ✓ Installed {package}")
    
    print("\n" + "="*60)
    print("Setup complete! Ready to extract BOQ from DXF files.")
    print("="*60)

if __name__ == "__main__":
    setup_project()