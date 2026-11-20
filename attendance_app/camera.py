"""Camera-frame handling, face localization and a shared RGB crop."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from anti_spoofing.data import crop_liveness_image

# OpenCV Zoo YuNet 2023mar, SHA-256 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4.
YUNET_MODEL = Path(__file__).with_name("models")/"face_detection_yunet_2023mar.onnx"
YUNET_SCORE = .6
# Haar box side / YuNet longest side, median of 751 matched LCC-FASD development
# frames (IQR .873-.951, centres within 2%). Keeps crops the CNNs were calibrated on.
YUNET_TO_HAAR = .91


class FaceLocator:
    """Localize current-frame faces, never identities.

    ``yunet`` is a small pretrained CNN detector used only for localization; its
    boxes are mapped to the square Haar geometry the downstream CNNs were
    calibrated with. ``haar`` keeps the two classical cascades. Contrast
    normalization is only for localization: the CNNs still receive the
    unmodified RGB pixels inside the selected box.
    """
    def __init__(self, mode="yunet", roi=None):
        import cv2
        if mode not in ("yunet", "haar", "roi"):
            raise ValueError("Face locator must be yunet, haar or roi.")
        self.mode, self.roi = mode, roi
        self.cascade = None
        self.alternate = None
        self.contrast = None
        self.net = None
        if mode == "yunet":
            if not YUNET_MODEL.is_file():
                raise RuntimeError(f"YuNet face detector is missing: {YUNET_MODEL}")
            self.net = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (320, 320), YUNET_SCORE)
            # Warm up at load time so the first camera frame is not delayed (one-off initialisation).
            self.net.setInputSize((64, 64))
            self.net.detect(np.zeros((64, 64, 3), dtype=np.uint8))
        if mode == "haar":
            self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.alternate = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
            self.contrast = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8))
            if self.cascade.empty() or self.alternate.empty():
                raise RuntimeError("OpenCV Haar face localization data is missing.")
            warmup = np.zeros((32, 32), dtype=np.uint8)
            self.cascade.detectMultiScale(warmup, scaleFactor=1.08, minNeighbors=4, minSize=(24, 24))
            self.alternate.detectMultiScale(warmup, scaleFactor=1.08, minNeighbors=4, minSize=(24, 24))

    @staticmethod
    def _deduplicate(boxes):
        """Suppress overlapping detections of one face; retain distinct people.

        Keep the first actual detection instead of averaging boxes or carrying a
        box from a previous frame. Default-cascade boxes have priority so a
        second successful cascade does not needlessly change an existing crop.
        A box mostly inside another (e.g. an inner box around glasses) is the
        same face, not a second person, so it no longer reports multiple faces.
        """
        kept = []
        for box in boxes:
            x, y, width, height = box
            if width <= 0 or height <= 0:
                continue
            duplicate = False
            for a, b, w, h in kept:
                overlap = max(0, min(x+width, a+w)-max(x, a)) * max(0, min(y+height, b+h)-max(y, b))
                union = width*height + w*h - overlap
                smaller = min(width*height, w*h)
                if overlap / union >= .55 or overlap / smaller >= .75:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(box)
        return kept

    def detect(self, rgb):
        import cv2
        height, width = rgb.shape[:2]
        if self.mode == "roi":
            side = int(min(height, width) * .65)
            return [tuple(self.roi or ((width-side)//2, (height-side)//2, side, side))]
        scale = min(1., 800 / width)
        small = cv2.resize(rgb, (round(width * scale), round(height * scale))) if scale < 1 else rgb
        scale_x, scale_y = width / small.shape[1], height / small.shape[0]
        if self.mode == "yunet":
            return self._detect_yunet(small, scale_x, scale_y)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        # Keep the same 80-pixel original-frame quality requirement, accounting
        # for the detector resize rather than imposing an extra 90-pixel limit.
        minimum = (max(24, round(80 / scale_x)), max(24, round(80 / scale_y)))
        options = dict(scaleFactor=1.08, minNeighbors=4, minSize=minimum)
        candidates = []
        for cascade, image in ((self.cascade, cv2.equalizeHist(gray)),
                               (self.alternate, self.contrast.apply(gray))):
            candidates.extend(cascade.detectMultiScale(image, **options))
        boxes = self._deduplicate([tuple(int(v) for v in box) for box in candidates])
        return [(round(x*scale_x), round(y*scale_y), round(w*scale_x), round(h*scale_y))
                for x, y, w, h in boxes]

    def _detect_yunet(self, small, scale_x, scale_y):
        import cv2
        self.net.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.net.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        boxes = []
        for x, y, w, h in ([] if faces is None else faces[:, :4]):
            cx, cy = (x + w/2) * scale_x, (y + h/2) * scale_y
            side = round(YUNET_TO_HAAR * max(w*scale_x, h*scale_y))
            if side >= 80:  # same native-frame minimum as the Haar path
                boxes.append((round(cx - side/2), round(cy - side/2), side, side))
        return self._deduplicate(boxes)  # YuNet's NMS can keep a small box nested in a larger one


class FrameProcessor:
    def __init__(self, locator, service, *, pad_context=False, pad_scale=None):
        self.locator, self.service = locator, service
        self.pad_context = pad_context
        self.pad_scale = pad_scale

    def _state(self, result):
        result.update(getattr(self.service, "frame_state", lambda: {})())
        return result

    def process(self, rgb, *, observed_at=None):
        import cv2
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
                or rgb.shape[2] != 3 or min(rgb.shape[:2]) == 0):
            raise ValueError("Camera frame must be HWC uint8 RGB.")
        boxes = self.locator.detect(rgb)
        result = {"boxes": [list(box) for box in boxes], "identity": None,
                  "liveness": None, "expression": None, "attendance_recorded": False}
        if len(boxes) != 1:
            self.service.reset_tracking("Face lost or multiple faces detected. Cancel and try again.")
            result["status"] = "no_face" if not boxes else "multiple_faces"
            return self._state(result)
        x, y, width, height = boxes[0]
        frame_h, frame_w = rgb.shape[:2]
        if width < 80 or height < 80 or min(x, y) < 0 or x + width > frame_w or y + height > frame_h:
            self.service.reset_tracking("The face is too small or outside the frame. Cancel and try again.")
            return self._state({**result, "status": "face_too_small_or_outside"})
        crop = crop_liveness_image(rgb, bbox=(x, y, width, height))
        gray = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        result.update(sharpness=sharpness, brightness=brightness,
                      quality_reasons=(["blur"] if sharpness < 20 else [])
                      + (["dark"] if brightness < 20 else ["bright"] if brightness > 235 else []))
        if sharpness < 20 or not 20 <= brightness <= 235:
            self.service.reset_tracking("Image quality was lost. Cancel and capture a clear face.")
            return self._state({**result, "status": "image_quality"})
        # A preview refresh is never a new observation. Async capture supplies
        # the monotonic timestamp of this unique frame, not inference completion.
        options = {}
        if self.pad_scale is not None:
            options["context_crop"] = crop_liveness_image(rgb, (x, y, width, height), crop_scale=self.pad_scale)
        elif self.pad_context:
            options["context_crop"] = crop_liveness_image(rgb, (x, y, width, height), context=True)
        if observed_at is not None:
            options.update(now=observed_at, max_observation_age=1.)
        result.update(self.service.analyze(crop, **options))
        return result
