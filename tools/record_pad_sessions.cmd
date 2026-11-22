@echo off
rem Guided webcam PAD recording at the app's 1.4x crop. Images stay in data\pad_webcam (git-ignored).
rem Order matters: per type, the last session becomes test and the one before it validation.
setlocal
set OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0
cd /d "%~dp0.."
set PY=.venv-gpu\Scripts\python.exe
if not exist %PY% set PY=.venv\Scripts\python.exe
echo.
echo  Stop the camera in the attendance app first (Stop button), then press a key.
echo  Each session: 5 s to get ready, then 20 s recording. Press Q in the window to stop early.
pause

call :rec live-person "glasses, room light on" "1/12 REAL FACE - wear glasses, room light ON, move your head slowly"
call :rec live-person "glasses, dark room" "2/12 REAL FACE - wear glasses, room light OFF (dim, like the failed test)"
call :rec live-person "no glasses, room light on" "3/12 REAL FACE - glasses OFF, room light ON"
call :rec live-person "no glasses, dark room" "4/12 REAL FACE - glasses OFF, room light OFF"
call :rec phone-still-photo "photo, light on, normal brightness" "5/12 PHONE PHOTO of your face - room light ON, normal screen brightness"
call :rec phone-still-photo "photo, dark room, max brightness" "6/12 PHONE PHOTO - room light OFF, screen brightness MAX (the attack that passed)"
call :rec phone-still-photo "photo, light on, max brightness, tilted" "7/12 PHONE PHOTO - light ON, brightness MAX, tilt and move the phone"
call :rec phone-still-photo "photo, dark room, max brightness, close" "8/12 PHONE PHOTO - light OFF, brightness MAX, hold the phone close so the face fills the screen"
call :rec recorded-video "video, light on" "9/12 PHONE VIDEO of you talking/moving - room light ON"
call :rec recorded-video "video, dark room" "10/12 PHONE VIDEO - room light OFF, brightness MAX"
call :rec recorded-video "video, light on, moving" "11/12 PHONE VIDEO - light ON, move the phone slowly"
call :rec recorded-video "video, dark room, close" "12/12 PHONE VIDEO - light OFF, brightness MAX, phone close to the camera"

echo.
echo  Done. Tell Claude the recording is done to build the splits and retrain.
pause
exit /b 0

:rec
echo.
echo  ============================================================
echo   %~3
echo  ============================================================
pause
%PY% -m tools.capture_pad_data capture --presentation %1 --note %2 --crop-scale 1.4
exit /b 0
