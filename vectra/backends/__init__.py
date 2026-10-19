"""Backend factory."""

from __future__ import annotations

import os
from pathlib import Path

from ..config import Settings
from .base import Backend


def create_backend(settings: Settings, home: Path) -> Backend:
    if settings.backend == "handcrafted":
        from .handcrafted import HandcraftedBackend
        return HandcraftedBackend()
    from .resnet import ResNetBackend
    return ResNetBackend(
        settings.backend,
        weights_path=settings.weights_path or os.environ.get("VECTRA_WEIGHTS") or None,
        home=home,
        image_size=settings.image_size,
        allow_random=os.environ.get("VECTRA_ALLOW_RANDOM_WEIGHTS") == "1",  # test hook only
    )


__all__ = ["Backend", "create_backend"]
