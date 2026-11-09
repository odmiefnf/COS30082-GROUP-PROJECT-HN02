"""Similarity to one person's separately enrolled appearance centroids."""
from __future__ import annotations

import numpy as np

from .store import unit_vector


def profile_similarity(embedding, profile, dimension):
    query = unit_vector(embedding, dimension)
    appearances = profile.get("appearances")
    values = [a["embedding"] for a in appearances] if appearances else [profile["embedding"]]
    return float(np.clip(np.max(np.stack([unit_vector(v, dimension) for v in values]) @ query), -1., 1.))
