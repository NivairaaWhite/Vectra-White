"""Feature-extractor interface.

A *backend* turns images into fixed-length raw feature vectors. It is frozen: it never trains.
All learning happens in the small head on top of it (see ``embedder.py`` / ``trainer.py``),
which is what makes training fast and makes backends interchangeable.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from PIL import Image


class Backend(ABC):
    name: str = "base"
    feat_dim: int = 0

    @abstractmethod
    def extract(self, images: Sequence[Image.Image]) -> np.ndarray:
        """RGB PIL images -> float32 array [N, feat_dim]. Must be deterministic."""

    @abstractmethod
    def identity(self) -> dict:
        """Everything that changes the numbers this backend produces (name, version, weights
        hash, preprocessing...). Hashed into :attr:`backend_id`."""

    @property
    def backend_id(self) -> str:
        blob = repr(sorted(self.identity().items())).encode()
        return f"{self.name}-{hashlib.sha1(blob).hexdigest()[:10]}"

    def describe(self) -> str:
        return f"{self.name} ({self.feat_dim}-d)"
