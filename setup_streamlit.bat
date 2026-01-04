@echo off
echo ============================================
echo DXF BOQ Extractor - Streamlit Setup
echo ============================================
echo.

REM Check Python version
python --version
echo.

echo Installing required packages...
echo.

REM Install packages
pip install streamlit pandas openpyxl numpy ezdxf plotly

echo.
echo ============================================
echo Installation Complete!
echo ============================================
echo.
echo To run the web application:
echo   streamlit run streamlit_app.py
echo.
echo The app will open in your web browser at:
echo   http://localhost:8501
echo.
pause