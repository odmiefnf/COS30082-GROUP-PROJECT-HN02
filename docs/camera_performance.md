# Webcam performance and stable layout — 8 November 2026

Update on 8 November 2026: the default window is now up to 1440 by 960,
clamped to the available screen. Compact header/toolbar spacing gives more
space to the same stable camera layout. The current outline is yellow until
identity is known and PAD passes, then green. No outline is retained from an
expired or incompatible observation. The geometry measurements below describe
the current layout; software evidence is in
`results/readiness_20261108`.

The reported whole-video shrinking/enlarging was reproduced as a Tk layout
problem: recognition messages changed the requested column width, guidance
wrapping changed its height, and a long recognized name enlarged its result
card. The video was fitted to the changing canvas, which made it appear to zoom.

The corrected layout gives the video column a stable preferred size while still
allowing the window to resize. Status, guidance and result values have reserved
two-line slots. The camera card uses a grid with only its video row expanding;
instructions remain visible at the minimum window size. The full identity value
is available by clicking the **IDENTITY · details** value or focusing it and
pressing Enter.

Two real-Tk regression tests cycle 26 status/result cases, including an
80-character name and changing FPS labels, then resize the window and repeat.
At a fixed window size the canvas position and size remain identical:

| Window | Stable canvas |
| --- | --- |
| 1200 × 790 | 734 × 290 |
| 1100 × 700 | 665 × 200 |
| 1440 × 960 | 902 × 441 |

The full current suite passed **202 tests** in 71.609 seconds. Evidence:
`results/readiness_20261108/full_tests.log` and
`results/readiness_20261108/layout_geometry.json`. No webcam, models or
profile database were used for these layout checks. Detector boxes, RGB crops,
recognition and liveness thresholds were not changed by this fix.

The subsequent single-view update removes the analyzed inset and puts the
face-position guide directly on the live preview. Identity, PAD, expression,
registration, appearances, manual review, attendance and history retain their
existing processing paths. Focused synthetic-image checks verify one
video image, a recent recognition outline, no stale/mismatched outline and clearing
after face loss. The focused single-view/layout/scheduling/capture test log is
`results/readiness_20261108/full_tests.log`.

## Earlier timing investigation

The earlier GUI waited for camera read, both Haar detectors, PAD, expression,
identity and an attendance query before displaying each image. Tk checked its
queue every 45 ms, and the video queue retained older images when busy. A slow
analysis therefore caused a visible pause in the preview.

## Changes

- A dedicated capture thread owns the camera. It publishes the newest RGB frame
  independently of model inference, with a sequence number and capture time.
- Preview and analysis each have one replaceable slot. Control events have their
  own ordered queue. Slow processing drops obsolete frames instead of building
  a backlog; each selected frame is analyzed once.
- Tk checks for fresh updates every 15 ms and reuses its canvas image item.
  Preview and analysis FPS are shown separately.
- One live camera preview shows the current video with a yellow or green
  face outline. There is no inset or second video. An outline is displayed only for
  a recent analysis (at most 200 ms old) from the same stream and resolution,
  and cannot come from a newer frame than the preview. It is a positioning
  guide; identity, liveness and expression results remain in their result cards.
- Capture time controls observation freshness. A frame older than one second
  cannot authorize capture actions; an additional check after inference blocks
  enrollment and attendance writes if the observation aged during processing.
- Stop/restart clears pending images and changes the stream identifier, so an
  old run cannot update the restarted camera's results.

The detector parameters, original RGB crops, model weights, calibrated thresholds,
three-observation attendance rule and stored profiles remain unchanged. CPU stays
the default: the device benchmark below does not justify switching these small
CNNs to the GPU. This performance change does not resolve the phone-photo PAD
failure recorded in the Phase 5 evaluation.

## Measurements and scope

Evidence is stored in `results/app_performance_20261106`. No camera images or
videos were saved. All live performance checks used a disposable SQLite gallery
copy; the production gallery was opened read-only for backup. These short runs
measure scheduling and speed, not identification accuracy or spoof resistance.

In the original synchronous pipeline, an eight-second CPU run processed 221
frames (27.53 FPS), including 134 crops that reached the CNNs. Haar localization
averaged 22.15 ms and the complete service analysis averaged 5.82 ms for those
crops. One initial detector call took 306.74 ms. SQLite queries averaged 0.25 ms.
This profiler excludes Tk rendering, so 27.53 FPS is not the old display FPS.

A warmed benchmark of all three CNNs on the same synthetic RGB crop, 100
iterations per device with four Torch threads, averaged 3.94 ms on CPU and
4.37 ms on the RTX 3060 Laptop GPU. It measures model timing only. A separate
CUDA/OpenCV-four-thread webcam run had no model-scored crops; its 22.81 FPS must
not be treated as a controlled CPU/GPU comparison or evidence about accuracy.

An initial latest-frame run processed 217 unique frames (27.08 FPS), with mean
observation age 1.52 ms at analysis start and 27.66 ms at completion. No sequence
was analyzed twice. Its capture pacing was subsequently changed to fixed
deadlines to avoid accumulating Windows wait rounding; the actual GUI check is
the evidence for the capture implementation.

The earlier 30-second actual GUI run completed at 29.07 preview FPS and 20.70
analysis FPS, with no duplicate sequences or reported errors. The camera and
worker closed automatically, and the production database SHA256 was unchanged.
Those timings preceded the layout correction above; FPS alone did not diagnose
the whole-video size jitter. See `gui_latest_cpu.json` in the performance evidence
directory.

## Use

Close any older app window and reopen `start_attendance.cmd`, then select
**Start camera**. The upper-right camera label reports **Preview … FPS ·
Analysis … FPS**. The analysis value can be lower while the main preview
continues updating. Diagnostics also shows analysis duration and observation age.

For repeatable developer checks using a temporary gallery:

```powershell
.\.venv-gpu\Scripts\python.exe -m tools.check_app_performance --seconds 30 --output results/app_performance_20261106/gui_check_new.json
.\.venv-gpu\Scripts\python.exe -m tools.profile_camera_performance --mode latest --seconds 8 --output results/app_performance_20261106/pipeline_check_new.json
```

Choose a new output filename for each run. The GUI check starts the camera and
closes automatically. Camera drivers can still block open/read/release; the app
cannot promise a driver-level shutdown timeout. Physical low-light exposure,
camera frame rate and system load may also limit smoothness.
