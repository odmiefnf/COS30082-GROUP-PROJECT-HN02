@echo off
setlocal
set OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0
cd /d "%~dp0"
if exist ".venv-gpu\Scripts\python.exe" (
  ".venv-gpu\Scripts\python.exe" -m attendance_app.web --open %*
  exit /b %errorlevel%
)
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m attendance_app.web --open %*
  exit /b %errorlevel%
)
if exist "..\lab4\.venv\Scripts\python.exe" (
  "..\lab4\.venv\Scripts\python.exe" -m attendance_app.web --open %*
  exit /b %errorlevel%
)
echo No Python environment found. See docs\06_user_interface.md.
pause
exit /b 2
