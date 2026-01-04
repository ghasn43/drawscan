@echo off
echo ====================================
echo DXF BOQ Extractor - Professional Edition
echo ====================================
echo.

REM Check for virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Installing dependencies...
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

REM Create necessary folders
if not exist "input" mkdir input
if not exist "output" mkdir output

echo.
echo Place your DXF files in the 'input' folder
echo.

REM Run the application
python main_app.py

pause