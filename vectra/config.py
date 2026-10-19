"""Library settings.

One small JSON file (``settings.json``) lives next to the data, so a library directory is
self-describing and can be moved, zipped or backed up as a unit.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

from .errors import LibraryNotFound, VectraError

VERSION = "2.0.0"
SETTINGS_FILE = "settings.json"
BACKENDS = ("handcrafted", "resnet18", "mobilenet_v3_small")
INDEX_KINDS = ("auto", "flat", "hnsw", "ivf")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
THUMB_SIZES = (160, 320, 640)
DEFAULT_HOME = "vectra_data"


def resolve_home(arg: Optional[str] = None) -> Path:
    """--home flag > $VECTRA_HOME > ./vectra_data"""
    raw = arg or os.environ.get("VECTRA_HOME") or DEFAULT_HOME
    return Path(raw).expanduser().resolve()


@dataclass
class Settings:
    backend: str = "handcrafted"      # which feature extractor this library is pinned to
    embed_dim: int = 256              # target size of the learned embedding
    image_size: int = 224             # input size for CNN backends
    index_kind: str = "auto"          # auto | flat | hnsw | ivf
    auto_hnsw_threshold: int = 200_000  # "auto" stays exact (flat) up to this size, then uses HNSW if faiss is installed
    copy_images: bool = True          # keep a content-addressed copy inside the library
    weights_path: Optional[str] = None  # explicit CNN weights file (optional)
    active_model: Optional[str] = None  # id of the trained head in models/, None = raw features
    version: str = VERSION

    def validate(self) -> "Settings":
        if self.backend not in BACKENDS:
            raise VectraError(f"unknown backend {self.backend!r}; choose one of {', '.join(BACKENDS)}")
        if self.index_kind not in INDEX_KINDS:
            raise VectraError(f"unknown index kind {self.index_kind!r}; choose one of {', '.join(INDEX_KINDS)}")
        if not (8 <= int(self.embed_dim) <= 2048):
            raise VectraError("embed_dim must be between 8 and 2048")
        return self

    # --- persistence -------------------------------------------------------------------
    @classmethod
    def load(cls, home: Path) -> "Settings":
        path = Path(home) / SETTINGS_FILE
        if not path.exists():
            raise LibraryNotFound(
                f"No Vectra library at {home}. Create one with:  vectra --home {home} init"
            )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise VectraError(f"cannot read {path}: {e}") from e
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known}).validate()

    def save(self, home: Path) -> None:
        home = Path(home)
        home.mkdir(parents=True, exist_ok=True)
        tmp = home / (SETTINGS_FILE + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, home / SETTINGS_FILE)  # atomic on every platform we care about
