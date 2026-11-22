"""Operator-reviewed import of three reference photos, never attendance/training."""
from __future__ import annotations

import hashlib
from pathlib import Path
import secrets
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

from anti_spoofing.data import crop_liveness_image
from .service import uses_pad_context
from .store import clean_text


class ReferenceImporter:
    def __init__(self, store, recognizer, liveness, locator):
        self.store, self.recognizer, self.liveness, self.locator = store, recognizer, liveness, locator
        self.pad_context = uses_pad_context(liveness)
        self.pending = None
        self._check_encoder()

    def _check_encoder(self):
        metadata = dict(self.store.connection.execute("SELECT key,value FROM metadata"))
        if (metadata.get("encoder_sha256") != self.recognizer.fingerprint
                or int(metadata.get("dimension", 0)) != self.recognizer.dimension):
            raise ValueError("Reference photos must use the profile database's exact encoder.")

    def cancel(self, operation_id=None):
        if operation_id is None or self.pending is not None and self.pending["operation_id"] == operation_id:
            self.pending = None

    def _validate_target(self, person_id, name, label, profiles):
        if person_id is None:
            name = clean_text(name)
            if any(p["name"].casefold() == name.casefold() for p in profiles):
                raise ValueError("This name already exists. Select its profile and add an appearance instead.")
            return name, None
        person = next((p for p in profiles if p["id"] == person_id), None)
        if person is None:
            raise ValueError("The selected profile no longer exists.")
        label = clean_text(label, "Appearance label")
        if len(person["appearances"]) >= 4:
            raise ValueError("This person already has four appearances, including the original.")
        if label.casefold() == "original" or any(a["label"].casefold() == label.casefold() for a in person["appearances"]):
            raise ValueError("Use a different appearance label; this label already exists.")
        return person["name"], label

    def _read_and_check(self, path):
        if getattr(self.locator, "mode", None) == "roi":
            raise ValueError("Photo import requires automatic face checks; ROI mode is for camera enrollment.")
        path = Path(path).expanduser().resolve()
        if not path.is_file() or path.suffix.casefold() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            raise ValueError("Choose a readable JPG, PNG, BMP or WebP photo.")
        if path.stat().st_size > 50*1024*1024:
            raise ValueError("Photo exceeds the 50 MB limit.")
        with Image.open(path) as source:
            if source.width*source.height > 50_000_000:
                raise ValueError("Photo exceeds the 50 megapixel limit.")
            image = ImageOps.exif_transpose(source).convert("RGB")
            digest = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
            image.thumbnail((1600, 1600), Image.Resampling.BILINEAR)
            rgb = np.asarray(image)
        boxes = self.locator.detect(rgb)
        if len(boxes) != 1:
            raise ValueError("Expected one face; found none or multiple faces.")
        x, y, width, height = boxes[0]
        if min(width, height) < 80 or min(x, y) < 0 or x+width > rgb.shape[1] or y+height > rgb.shape[0]:
            raise ValueError("The face is too small or outside the image.")
        crop = Image.fromarray(rgb[y:y+height, x:x+width].copy())
        gray = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 20 or not 20 <= gray.mean() <= 235:
            raise ValueError("The face is too blurry, dark or bright.")
        pad_crop = crop_liveness_image(rgb, (x, y, width, height), context=True) if self.pad_context else crop
        live = self.liveness.predict_crop(pad_crop)
        if live.get("is_live") is not True:
            raise ValueError("PAD model check not passed. This does not establish live presence from a photo.")
        return self.recognizer.embed(crop), digest

    def _check_matches(self, embeddings, profiles, person_id):
        for embedding in embeddings:
            match = self.recognizer.match(embedding, profiles)
            if (match["status"] == "ambiguous" or match["status"] == "known"
                    and (person_id is None or match["person_id"] != person_id)):
                raise ValueError("A photo matches an existing/different profile or is ambiguous. Verify the selected person.")
        for index, embedding in enumerate(embeddings):
            if any(float(np.dot(embedding, other)) < self.recognizer.threshold for other in embeddings[:index]):
                raise ValueError("The three photos are inconsistent. Use the same person and appearance in all three.")

    def prepare(self, *, operation_id, files, person_id=None, name="", label="", now=None):
        self.pending = None
        self._check_encoder()
        if not isinstance(operation_id, str) or not 16 <= len(operation_id) <= 100:
            raise ValueError("Invalid import operation ID.")
        if not isinstance(files, (list, tuple)) or len(files) != 3:
            raise ValueError("Choose exactly three distinct photos for one appearance.")
        profiles = self.store.profiles()
        name, label = self._validate_target(person_id, name, label, profiles)
        items, vectors, digests = [], [], set()
        for file in files:
            item = {"file": Path(file).name, "passed": False}
            try:
                vector, digest = self._read_and_check(file)
                if digest in digests:
                    raise ValueError("Duplicate photo pixels. Choose three different photographs.")
                digests.add(digest)
                vectors.append(vector.copy())
                item.update(passed=True, message="Face, quality and PAD checks passed")
            except (OSError, ValueError, RuntimeError, cv2.error, Image.DecompressionBombError) as exc:
                item["message"] = str(exc)
            items.append(item)
        valid = all(item["passed"] for item in items)
        message = "All three photos passed. Review the selected person, then confirm import."
        if valid:
            try:
                self._check_matches(vectors, profiles, person_id)
            except ValueError as exc:
                valid, message = False, str(exc)
        else:
            message = "No samples saved. Replace the rejected photos and check again."
        now = time.monotonic() if now is None else float(now)
        if not np.isfinite(now):
            raise ValueError("Import clock must be finite.")
        token = secrets.token_urlsafe(24) if valid else None
        if valid:
            self.pending = {"token": token, "operation_id": operation_id, "person_id": person_id,
                            "name": name, "label": label, "embeddings": vectors, "expires_at": now+120.}
            self.pending["opened_at"] = now
        return {"operation_id": operation_id, "valid": valid, "token": token, "items": items,
                "message": message, "expires_at": now+120. if valid else None}

    def commit(self, *, operation_id, token, confirmed, now=None):
        if confirmed is not True:
            raise ValueError("Verify the person and permission to use all three photos before importing.")
        pending = self.pending
        if (pending is None or pending["operation_id"] != operation_id
                or not secrets.compare_digest(str(token), pending["token"])):
            raise ValueError("Check the photos again; the import request is no longer valid.")
        now = time.monotonic() if now is None else float(now)
        if not np.isfinite(now) or not pending["opened_at"] <= now < pending["expires_at"]:
            self.pending = None
            raise ValueError("The checked photos have expired. Check them again.")
        self._check_encoder()
        profiles = self.store.profiles()
        name, label = self._validate_target(pending["person_id"], pending["name"], pending["label"], profiles)
        if name != pending["name"]:
            raise ValueError("The selected profile was renamed. Check the photos again.")
        self._check_matches(pending["embeddings"], profiles, pending["person_id"])
        if pending["person_id"] is None:
            person_id = self.store.enroll(name, pending["embeddings"])
            appearance_id = "original:"+person_id
        else:
            person_id = pending["person_id"]
            appearance_id = self.store.add_appearance(person_id, label, pending["embeddings"])
        self.pending = None
        return {"person_id": person_id, "appearance_id": appearance_id, "name": name}
