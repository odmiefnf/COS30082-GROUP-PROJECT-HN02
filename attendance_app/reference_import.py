"""Import of 3-10 reference photos straight into a profile, never attendance/training.

By the operator's choice there are no photo checks (liveness, quality, consistency or
match against other profiles): a wrong profile is found in use and deleted. The face is
still located, because the identity model needs a face crop.
"""
from __future__ import annotations

from pathlib import Path
import secrets
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

from .store import clean_text

MIN_PHOTOS, MAX_PHOTOS = 3, 10  # 3 = the store's minimum per appearance; 10 = one camera registration


class ReferenceImporter:
    def __init__(self, store, recognizer, locator):
        self.store, self.recognizer, self.locator = store, recognizer, locator
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

    def _read_face(self, path):
        if getattr(self.locator, "mode", None) == "roi":
            raise ValueError("Photo import needs automatic face detection; ROI mode is for camera enrollment.")
        path = Path(path).expanduser().resolve()
        if not path.is_file() or path.suffix.casefold() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            raise ValueError("Choose a readable JPG, PNG, BMP or WebP photo.")
        if path.stat().st_size > 50*1024*1024:
            raise ValueError("Photo exceeds the 50 MB limit.")
        with Image.open(path) as source:
            if source.width*source.height > 50_000_000:
                raise ValueError("Photo exceeds the 50 megapixel limit.")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((1600, 1600), Image.Resampling.BILINEAR)
            rgb = np.asarray(image)
        boxes = self.locator.detect(rgb)
        if not boxes:
            raise ValueError("No face found; photo skipped.")
        # Largest face, like the camera's main face; clipped so faces at the photo edge still work.
        x, y, width, height = max(boxes, key=lambda box: box[2]*box[3])
        left, top = max(int(x), 0), max(int(y), 0)
        right, bottom = min(int(x+width), rgb.shape[1]), min(int(y+height), rgb.shape[0])
        if right <= left or bottom <= top:
            raise ValueError("No face found; photo skipped.")
        return self.recognizer.embed(Image.fromarray(rgb[top:bottom, left:right].copy()))

    def prepare(self, *, operation_id, files, person_id=None, name="", label="", now=None):
        self.pending = None
        self._check_encoder()
        if not isinstance(operation_id, str) or not 16 <= len(operation_id) <= 100:
            raise ValueError("Invalid import operation ID.")
        if not isinstance(files, (list, tuple)) or not MIN_PHOTOS <= len(files) <= MAX_PHOTOS:
            raise ValueError(f"Choose {MIN_PHOTOS} to {MAX_PHOTOS} distinct photos for one appearance.")
        profiles = self.store.profiles()
        name, label = self._validate_target(person_id, name, label, profiles)
        items, vectors = [], []
        for file in files:
            item = {"file": Path(file).name, "passed": False}
            try:
                vectors.append(self._read_face(file).copy())
                item.update(passed=True, message="Face found")
            except (OSError, ValueError, RuntimeError, cv2.error, Image.DecompressionBombError) as exc:
                item["message"] = str(exc)
            items.append(item)
        skipped = len(items)-len(vectors)
        valid = len(vectors) >= MIN_PHOTOS
        message = (f"{len(vectors)} photos ready" + (f"; {skipped} skipped." if skipped else ".") if valid else
                   f"Only {len(vectors)} photos had a face; at least {MIN_PHOTOS} are needed. Nothing was saved.")
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
            raise ValueError("Verify the person and permission to use all photos before importing.")
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
        if pending["person_id"] is None:
            person_id = self.store.enroll(name, pending["embeddings"])
            appearance_id = "original:"+person_id
        else:
            person_id = pending["person_id"]
            appearance_id = self.store.add_appearance(person_id, label, pending["embeddings"])
        self.pending = None
        return {"person_id": person_id, "appearance_id": appearance_id, "name": name, "samples": len(pending["embeddings"])}
