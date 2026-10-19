"""Embedder = frozen backend + small learned linear head.

    raw features --(centre, project)--> head --> L2-normalise --> embedding

The head is a single matrix, so it is cheap to fit, easy to inspect, and stored as plain
``.npz`` (no pickle). Three kinds:

  raw      no head: just L2-normalise the backend features.            (always available)
  pca      centred + partially-whitened PCA, fitted on your images.    (no gradient training)
  trained  the PCA head refined with contrastive learning on augmented views.

The original code initialised a *random* projection head on every start and then indexed with
one instance and searched with another -- the two never agreed. Here every head is either
absent or derived from data, and the model id is a hash of its numbers, so an index can never
be paired with the wrong model (see ``library.py``).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from PIL import Image

from .backends.base import Backend
from .errors import ModelMismatch, VectraError


@dataclass
class Head:
    kind: str = "raw"                    # raw | pca | trained
    mu: Optional[np.ndarray] = None      # [F]
    W: Optional[np.ndarray] = None       # [F, D]
    info: Dict = field(default_factory=dict)


def fit_pca(feats: np.ndarray, dim: int = 256, whiten: float = 0.5, floor: float = 1e-3,
            chunk: int = 20000) -> Head:
    """PCA with partial whitening (power ``whiten``: 0 = rotation only, 1 = full whitening).

    ``floor`` regularises tiny eigenvalues (relative to the largest) so a handful of noisy
    directions cannot dominate; with few images it matters a lot.
    """
    feats = np.asarray(feats, np.float32)
    n, f = feats.shape
    if n < 3:
        raise VectraError("need at least 3 images to fit a PCA head")
    mu = feats.mean(0, dtype=np.float64)
    cov = np.zeros((f, f), np.float64)
    for s in range(0, n, chunk):
        x = feats[s:s + chunk].astype(np.float64) - mu
        cov += x.T @ x
    cov /= max(1, n - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = np.clip(evals[order], 0.0, None), evecs[:, order]
    keep = int(min(dim, n - 1, f, int((evals > 1e-10 * max(evals[0], 1e-30)).sum())))
    keep = max(keep, 1)
    scale = (evals[:keep] + floor * evals[0] + 1e-30) ** (-whiten / 2.0)
    W = (evecs[:, :keep] * scale).astype(np.float32)
    return Head("pca", mu.astype(np.float32), W,
                {"whiten": whiten, "floor": floor, "n_fit": int(n), "dim": keep})


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


class Embedder:
    def __init__(self, backend: Backend, head: Optional[Head] = None):
        self.backend = backend
        self.head = head or Head("raw")
        if self.head.W is not None and self.head.W.shape[0] != backend.feat_dim:
            raise ModelMismatch(
                f"head expects {self.head.W.shape[0]}-d features but backend "
                f"{backend.name!r} produces {backend.feat_dim}-d"
            )

    # --- identity ------------------------------------------------------------------------
    @property
    def kind(self) -> str:
        return self.head.kind

    @property
    def dim(self) -> int:
        return int(self.head.W.shape[1]) if self.head.W is not None else int(self.backend.feat_dim)

    @property
    def model_id(self) -> str:
        if self.head.kind == "raw":
            return "raw"
        h = hashlib.sha1(self.backend.backend_id.encode())
        for arr in (self.head.mu, self.head.W):
            if arr is not None:
                h.update(np.ascontiguousarray(arr).tobytes())
        return f"{self.head.kind}-{h.hexdigest()[:10]}"

    # --- embedding -----------------------------------------------------------------------
    def embed_features(self, feats: np.ndarray) -> np.ndarray:
        x = np.asarray(feats, np.float32)
        if x.ndim == 1:
            x = x[None]
        if self.head.mu is not None:
            x = x - self.head.mu
        if self.head.W is not None:
            x = x @ self.head.W
        return l2_normalize(x)

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        return self.embed_features(self.backend.extract(images))

    # --- persistence ---------------------------------------------------------------------
    def save(self, models_dir: Path, extra: Optional[dict] = None) -> Path:
        if self.head.kind == "raw":
            raise VectraError("the raw model has nothing to save")
        d = Path(models_dir) / self.model_id
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "head.npz", mu=self.head.mu, W=self.head.W)
        meta = {
            "model_id": self.model_id, "kind": self.head.kind,
            "backend": self.backend.name, "backend_id": self.backend.backend_id,
            "feat_dim": int(self.backend.feat_dim), "dim": self.dim,
            "created": time.time(), "info": self.head.info, **(extra or {}),
        }
        (d / "model.json").write_text(json.dumps(meta, indent=2, default=float) + "\n")
        return d

    @classmethod
    def load(cls, model_dir: Path, backend: Backend) -> "Embedder":
        model_dir = Path(model_dir)
        try:
            meta = json.loads((model_dir / "model.json").read_text())
            with np.load(model_dir / "head.npz", allow_pickle=False) as z:
                mu, W = z["mu"], z["W"]
        except (OSError, ValueError, KeyError) as e:
            raise VectraError(f"cannot load model from {model_dir}: {e}") from e
        if meta.get("backend_id") != backend.backend_id:
            raise ModelMismatch(
                f"model {meta.get('model_id')} was trained for feature extractor "
                f"{meta.get('backend_id')}, but this library uses {backend.backend_id}.\n"
                f"Re-train with `vectra train`, or switch back with `vectra use raw`."
            )
        emb = cls(backend, Head(meta["kind"], mu, W, meta.get("info", {})))
        if emb.model_id != meta["model_id"]:
            raise VectraError(f"model files in {model_dir} are corrupted (id mismatch)")
        return emb


def list_models(models_dir: Path):
    out = []
    for p in sorted(Path(models_dir).glob("*/model.json")):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return out
