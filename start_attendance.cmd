@echo off
setlocal
set OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0
cd /d "%~dp0"
if exist ".venv-gpu\Scripts\pythonw.exe" (
  start "" ".venv-gpu\Scripts\pythonw.exe" -m attendance_app.gui
  exit /b 0
)
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" -m attendance_app.gui
  exit /b 0
)
if exist "..\lab4\.venv\Scripts\pythonw.exe" (
  start "" "..\lab4\.venv\Scripts\pythonw.exe" -m attendance_app.gui
  exit /b 0
)
echo No Python environment found. See docs\06_user_interface.md.
pause
exit /b 2
