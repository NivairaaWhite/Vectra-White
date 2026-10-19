"""Vectra: local, trainable reverse image search."""

from .config import VERSION as __version__
from .errors import VectraError
from .library import Library, Match

__all__ = ["Library", "Match", "VectraError", "__version__"]
