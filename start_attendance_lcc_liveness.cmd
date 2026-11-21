@echo off
rem Temporary: LCC-only MobileNetV3 liveness. It passes genuine faces in rooms the webcam model never saw,
rem but phone photos and videos can pass (held-out webcam test: 29/52 photo, 67/116 video crops accepted).
call "%~dp0start_attendance.cmd" --liveness-checkpoint results/liveness_mobilenetv3_20261120/best_model.pt %*
