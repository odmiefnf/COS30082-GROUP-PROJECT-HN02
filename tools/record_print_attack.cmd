@echo off
rem Printed-photo attack test for the app's liveness model (brief: "printed images or screen photos").
rem Crops go to data\pad_webcam_print (git-ignored, personal data) and are NOT used for training.
rem The last step scores them with the app model and its validation threshold.
setlocal
set OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0
cd /d "%~dp0.."
set PY=.venv-gpu\Scripts\python.exe
if not exist %PY% set PY=.venv\Scripts\python.exe
set ROOT=data\pad_webcam_print
echo.
echo  You need: one colour print of a person's face (A4 or A5, face about the size of a real face),
echo  printed only with that person's consent. Stop the camera in the attendance app first.
echo  Each session: 5 s to get ready, then 20 s recording. Press Q in the window to stop early.
pause

call :rec live-person "print test day, real face, room light on" "1/6 REAL FACE - room light ON, move your head slowly"
call :rec live-person "print test day, real face, dim light" "2/6 REAL FACE - room light dim or OFF"
call :rec printed-photo "print, light on, flat" "3/6 PRINTED PHOTO - light ON, hold the paper flat in front of the camera, real face out of view"
call :rec printed-photo "print, light on, curved and moving" "4/6 PRINTED PHOTO - light ON, bend the paper slightly and move it slowly"
call :rec printed-photo "print, dim light" "5/6 PRINTED PHOTO - room light dim or OFF"
call :rec printed-photo "print, close and tilted" "6/6 PRINTED PHOTO - hold it close so the face fills the frame, tilt it a little"

echo.
echo  Scoring the crops with the app liveness model...
%PY% -m anti_spoofing.evaluate --checkpoint results\liveness_mobilenetv3_webcam_20261120\best_model.pt --manifest %ROOT%\manifest.csv --output-dir results\liveness_mobilenetv3_webcam_20261120_on_print
echo.
echo  Done. Tell Claude the print test is recorded so the result can be checked and written up.
pause
exit /b 0

:rec
echo.
echo  ============================================================
echo   %~3
echo  ============================================================
pause
%PY% -m tools.capture_pad_data --root %ROOT% capture --presentation %1 --note %2 --crop-scale 1.4
exit /b 0
