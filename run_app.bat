@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run START_HERE.bat first to install the application.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1
if errorlevel 1 pause

