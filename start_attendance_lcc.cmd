@echo off
rem Same app with the LCC-only liveness model (threshold 0.0408). It passes genuine faces in more rooms,
rem but at home it let about half of phone photos and videos through, so use it for demos and recording only.
rem start_attendance.cmd keeps the webcam-trained model, the one that blocks phone attacks.
call "%~dp0start_attendance.cmd" --liveness-checkpoint results/liveness_mobilenetv3_20261120/best_model.pt %*
