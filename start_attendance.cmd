@echo off
setlocal
set "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0"
cd /d "%~dp0"

set "PY="
if exist ".venv-gpu\Scripts\python.exe" set "PY=.venv-gpu\Scripts\python.exe"
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if defined PY goto :check_deps

echo [Setup] No virtual environment found. Creating .venv...
call :find_python
if not defined SYS_PY (
  echo.
  echo [Error] Python 3.10+ was not found on this computer.
  echo Install Python 3.11 or 3.12 from https://www.python.org/downloads/
  echo and check "Add python.exe to PATH", then run start_attendance.cmd again.
  pause
  exit /b 2
)

echo [Setup] Using %SYS_PY% %SYS_PY_ARGS%
"%SYS_PY%" %SYS_PY_ARGS% -m venv .venv
if errorlevel 1 (
  echo [Error] Failed to create .venv using "%SYS_PY%" %SYS_PY_ARGS%.
  pause
  exit /b 2
)
set "PY=.venv\Scripts\python.exe"

:check_deps
"%PY%" -c "import torch, torchvision, cv2, PIL, numpy" >nul 2>&1
if not errorlevel 1 goto :run_app

echo [Setup] Installing required packages from requirements-webcam.txt...
echo This only runs once and may take a few minutes.
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements-webcam.txt
if errorlevel 1 (
  echo.
  echo [Error] Package installation failed. Check your internet connection or Python version ^(3.10-3.13^).
  pause
  exit /b 1
)

:run_app
"%PY%" -m attendance_app.web --open %*
set "APP_ERR=%errorlevel%"
if "%APP_ERR%"=="0" exit /b 0
echo.
echo [Error] App exited with code %APP_ERR%.
pause
exit /b %APP_ERR%

:find_python
set "SYS_PY="
set "SYS_PY_ARGS="
for %%V in (-3.11 -3.12 -3.13 -3.10 -3) do (
  py %%V -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
  if not errorlevel 1 (
    set "SYS_PY=py"
    set "SYS_PY_ARGS=%%V"
    exit /b 0
  )
)
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 (
  set "SYS_PY=python"
  exit /b 0
)
for %%P in (
  "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
  "%ProgramFiles%\Python311\python.exe"
  "%ProgramFiles%\Python312\python.exe"
  "%ProgramFiles%\Python313\python.exe"
) do (
  if exist "%%~P" (
    "%%~P" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
    if not errorlevel 1 (
      set "SYS_PY=%%~P"
      exit /b 0
    )
  )
)
exit /b 1
