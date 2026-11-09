"""A selected-camera probe: no models, database, attendance or image persistence."""
from __future__ import annotations

import numpy as np


CAMERA_HELP = "Close other camera apps, check the selected index and Windows camera access, then try again."


def camera_index(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Choose a camera index from 0 to 99.") from exc
    if isinstance(value, bool) or str(value).strip() != str(parsed) or not 0 <= parsed <= 99:
        raise ValueError("Choose a camera index from 0 to 99.")
    return parsed


def probe_camera(index):
    """Read one frame and always release; return metadata, never the frame.

    Run off the Tk thread. A camera driver's open/read calls may block; no
    promise of a driver-level timeout is made by this software check.
    """
    camera = None
    try:
        import cv2
        index = camera_index(index)
        camera = cv2.VideoCapture(index)
        if not camera.isOpened():
            raise RuntimeError("The selected camera could not be opened.")
        success, frame = camera.read()
        if (not success or not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) == 0):
            raise RuntimeError("The selected camera did not return a usable color frame.")
        result = {"ok": True, "index": index, "width": int(frame.shape[1]), "height": int(frame.shape[0]),
                "message": f"Camera {index} returned a frame. Device released; no attendance recorded."}
    except Exception as exc:
        result = {"ok": False, "index": index, "message": f"{exc} {CAMERA_HELP}"}
    finally:
        if camera is not None:
            try:
                camera.release()
            except Exception as exc:
                result = {"ok": False, "index": index,
                          "message": f"Camera release failed: {exc}. Close the app before retrying."}
    return result
