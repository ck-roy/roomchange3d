@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Preparing RoomChange3D. This may take several minutes and downloads PyTorch/VGGT code.
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
  if errorlevel 1 (
    echo Setup failed. Read setup.log for details.
    pause
    exit /b 1
  )
)
call "%~dp0run_app.bat"

