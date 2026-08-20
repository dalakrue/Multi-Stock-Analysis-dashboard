@echo off
setlocal
cd /d "%~dp0"
python -m streamlit run app.py
if errorlevel 1 (
  echo.
  echo Could not start Streamlit. Install dependencies with:
  echo   python -m pip install -r requirements.txt
  pause
)
