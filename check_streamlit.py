#!/usr/bin/env python3
"""
Check if all dependencies are installed for Streamlit app
"""

import sys
import subprocess

required_packages = [
    "streamlit",
    "pandas",
    "openpyxl",
    "numpy",
    "ezdxf"
]

missing_packages = []

for package in required_packages:
    try:
        __import__(package)
        print(f"✅ {package}")
    except ImportError:
        missing_packages.append(package)
        print(f"❌ {package}")

if missing_packages:
    print(f"\nMissing packages: {', '.join(missing_packages)}")
    install = input("Install missing packages? (y/n): ")
    if install.lower() == 'y':
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
    
    print("\nNow run: streamlit run streamlit_app.py")
else:
    print("\n✅ All packages installed!")
    print("Run: streamlit run streamlit_app.py")