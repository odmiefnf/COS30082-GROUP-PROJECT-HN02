"""One embedding per probe and open-set matching against enrolled centroids."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from face_verification.data import preprocess_face_image
from face_verification.evaluate import load_encoder
from face_verification.image_cache import sha256
from .store import unit_vector
from .templates import profile_similarity


class FaceRecognizer:
    def __init__(self, configuration, device="cpu"):
        configuration = Path(configuration).resolve()
        self.settings = json.loads(configuration.read_text(encoding="utf-8"))
        self.checkpoint = Path(self.settings["checkpoint"])
        if not self.checkpoint.is_absolute():
            self.checkpoint = (configuration.parent / self.checkpoint).resolve()
        self.fingerprint = sha256(self.checkpoint)
        if self.fingerprint != self.settings["checkpoint_sha256"]:
            raise ValueError("Identity checkpoint does not match the calibration fingerprint.")
        if self.settings.get("metric") != "cosine" or self.settings.get("threshold_source") != "verification_validation":
            raise ValueError("A cosine threshold calibrated on verification validation is required.")
        self.threshold = float(self.settings["threshold"])
        self.margin = float(self.settings.get("ambiguity_margin", .05))
        if not np.isfinite([self.threshold, self.margin]).all() or not -1 <= self.threshold <= 1 or not 0 <= self.margin <= 2:
            raise ValueError("Invalid identity threshold or ambiguity margin.")
        if device not in ("auto", "cpu", "cuda") or device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Invalid or unavailable identity device.")
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
        self.encoder, self.config = load_encoder(self.checkpoint, self.device)
        if self.config.get("smoke_test") is not False or self.config.get("training", {}).get("initialization") != "random_from_scratch":
            raise ValueError("The attendance app requires a completed from-scratch training checkpoint.")
        self.dimension = self.config["embedding_dim"]

    def embed(self, crop):
        images = preprocess_face_image(crop, self.config["image_size"]).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            value = self.encoder(images)[0].cpu().numpy()
        return unit_vector(value, self.dimension)

    def match(self, embedding, profiles):
        if not profiles:
            return {"status": "unknown", "person_id": None, "name": None, "cosine": None, "gap": None}
        query = unit_vector(embedding, self.dimension)
        # Compare appearance templates within a person first. The ambiguity
        # runner-up must be a different person, never their second appearance.
        scores = np.asarray([profile_similarity(query, p, self.dimension) for p in profiles])
        order = np.argsort(-scores, kind="stable")
        best = int(order[0]); value = float(scores[best])
        gap = float(value - scores[order[1]]) if len(order) > 1 else None
        status = "unknown" if value < self.threshold else "ambiguous" if gap is not None and gap < self.margin else "known"
        return {"status": status, "person_id": profiles[best]["id"] if status == "known" else None,
                "name": profiles[best]["name"] if status == "known" else None, "cosine": value, "gap": gap}
