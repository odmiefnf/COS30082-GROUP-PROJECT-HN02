"""Fresh, bounded webcam capture, independent of model inference.

Only the capture thread touches OpenCV's camera handle. Frames live in one
replaceable in-memory slot; a slow consumer never builds a video backlog.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import queue
import threading
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraSample:
    sequence: int
    captured_at: float
    rgb: np.ndarray


def put_latest(slot, value):
    """Replace an old preview/analysis value in a bounded video-only queue.

    Control events belong in their own ordered queue and must not use this
    helper. This also tolerates the consumer draining the slot concurrently.
    """
    if slot.maxsize != 1:
        raise ValueError("Latest-frame queues must have maxsize=1.")
    while True:
        try:
            slot.put_nowait(value)
            return
        except queue.Full:
            try:
                slot.get_nowait()
            except queue.Empty:
                pass


class LatestCamera:
    """Capture at most ``fps`` frames/second and expose only the newest frame.

    ``on_preview`` runs on the capture thread and must return promptly, for
    example by replacing a preview slot. It must never call Tk directly.
    Opening and closing wait for the driver; OpenCV cannot guarantee a timeout
    if a device driver blocks inside open/read/release.
    """

    def __init__(self, index, on_preview=None, width=960, height=720, fps=30):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= 99:
            raise ValueError("Camera index must be an integer from 0 to 99.")
        for name, value in (("width", width), ("height", height)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Camera {name} must be a positive integer.")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Camera FPS must be positive and finite.")
        if on_preview is not None and not callable(on_preview):
            raise ValueError("Preview callback must be callable.")
        self.index, self.width, self.height, self.fps = index, width, height, float(fps)
        self.on_preview = on_preview
        self._condition = threading.Condition()
        self._startup = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._latest = None
        self._error = None
        self._startup_error = None
        self._release_error = None
        self._opened = False
        self._closed = False
        self._metadata = {"width": None, "height": None, "fps": None,
                          "requested_width": width, "requested_height": height,
                          "requested_fps": self.fps}

    @property
    def metadata(self):
        with self._condition:
            return dict(self._metadata)

    @property
    def error(self):
        with self._condition:
            return self._error

    def open(self):
        with self._condition:
            if self._closed:
                raise RuntimeError("This camera stream has been closed.")
            if self._thread is None:
                self._thread = threading.Thread(target=self._capture,
                                                name=f"camera-{self.index}", daemon=True)
                self._thread.start()
        self._startup.wait()
        if self._startup_error is not None:
            # An opening/property failure must finish releasing before the
            # caller can retry with another stream.
            self._thread.join()
            raise RuntimeError(self._startup_error)
        return self

    def isOpened(self):
        with self._condition:
            return self._opened and not self._closed

    def read_latest(self, after_sequence=0, timeout=.05):
        """Return a newer observation once, ``None`` on timeout/explicit close.

        The caller keeps the last processed sequence number. A final unread
        sample remains available if a later read fails; the following request
        raises the error. Repeated polling never invents a new observation.
        """
        if (isinstance(after_sequence, bool) or not isinstance(after_sequence, int)
                or after_sequence < 0):
            raise ValueError("Frame sequence must be a nonnegative integer.")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout < 0):
            raise ValueError("Read timeout must be nonnegative and finite.")
        deadline = time.monotonic() + timeout
        with self._condition:
            if self._thread is None and not self._closed:
                raise RuntimeError("Open the camera before reading frames.")
            while True:
                if self._closed:
                    return None
                if self._latest is not None and self._latest.sequence > after_sequence:
                    return self._latest
                if self._error is not None:
                    raise RuntimeError(self._error)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def close(self):
        with self._condition:
            already_closed = self._closed
            self._closed = True
            self._latest = None
            self._stop.set()
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        if not already_closed and self._release_error is not None:
            raise RuntimeError(self._release_error)

    release = close

    def _capture(self):
        camera = None
        stage = "Camera open"
        try:
            camera = cv2.VideoCapture(self.index)
            if not camera.isOpened():
                raise RuntimeError(f"Cannot open camera {self.index}.")
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            camera.set(cv2.CAP_PROP_FPS, self.fps)
            # Some backends ignore this property. Latest-frame capture still
            # drains the device while inference runs on its separate worker.
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            try:
                driver_fps = float(camera.get(cv2.CAP_PROP_FPS))
            except (TypeError, ValueError, cv2.error):
                driver_fps = None
            with self._condition:
                self._metadata["fps"] = (driver_fps if driver_fps is not None
                                          and math.isfinite(driver_fps) and driver_fps > 0 else None)
                self._opened = True
            self._startup.set()
            sequence = 0
            stage = "Camera read"
            next_capture = time.monotonic()
            while not self._stop.is_set():
                success, bgr = camera.read()
                captured_at = time.monotonic()
                if self._stop.is_set():
                    break
                if (not success or not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8
                        or bgr.ndim != 3 or bgr.shape[2] != 3 or min(bgr.shape[:2]) == 0):
                    raise RuntimeError("The camera stopped returning valid color frames.")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb.setflags(write=False)
                sequence += 1
                sample = CameraSample(sequence, captured_at, rgb)
                with self._condition:
                    if self._closed:
                        break
                    self._latest = sample
                    self._metadata.update(width=rgb.shape[1], height=rgb.shape[0])
                    self._condition.notify_all()
                if self.on_preview is not None:
                    self.on_preview(sample)
                # Fixed deadlines avoid adding Windows wait rounding to every
                # frame period. A long driver stall does not trigger a burst.
                next_capture += 1 / self.fps
                now = time.monotonic()
                if now-next_capture > 1 / self.fps:
                    next_capture = now
                self._stop.wait(max(0., next_capture-now))
        except Exception as exc:
            message = f"{stage}: {exc}"
            with self._condition:
                self._error = message
                if not self._startup.is_set():
                    self._startup_error = message
                self._condition.notify_all()
        finally:
            if camera is not None:
                try:
                    camera.release()
                except Exception as exc:
                    with self._condition:
                        self._release_error = f"Camera release: {exc}"
                        self._error = self._error or self._release_error
            with self._condition:
                self._opened = False
                self._condition.notify_all()
            self._startup.set()
