"""Zero-download feature extractor (NumPy + Pillow only).

Works on a machine that has never seen the internet, needs no torch, and is strong for the
classic reverse-image-search job: *finding the same picture after it was resized,
recompressed, recoloured, watermarked or lightly cropped*. It is weaker than a CNN at
"same kind of object, different photo" -- use ``resnet18`` when you need that.

The descriptor is a concatenation of independently L2-normalised blocks:

  layout   8x8 YCbCr thumbnail (mean-subtracted)            where the big colour masses are
  mean     global mean colour                                 overall tint
  colour   RGB-cube histograms on a 1x1 and 2x2 grid        colour distribution, coarse layout
  grad     gradient-orientation pyramid 1x1/2x2/4x4/8x8      edges & texture (HOG-like)
  dct      low-frequency DCT of the grayscale image         structure, invariant to brightness

Coarse grid levels tolerate crops/shifts; fine levels discriminate. The learned head on top
(see embedder.py) decides how much to trust each block for *your* data.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from PIL import Image

from ..imageio import RESAMPLE_BOX, RESAMPLE_LANCZOS, _dct_matrix
from .base import Backend

S = 64            # analysis resolution
CELLS = 8         # finest orientation grid (S / CELLS = 8 px per cell)
NBINS = 9         # orientation bins (unsigned gradients)
VERSION = 2

_YY, _XX = np.indices((S, S))


def _grid_cells(g: int) -> np.ndarray:
    cs = S // g
    return (_YY // cs) * g + (_XX // cs)


_CELL_IDX = {g: _grid_cells(g) for g in (1, 2, 4, 8)}


def _hellinger(cells: np.ndarray) -> np.ndarray:
    """[n_cells, bins] non-negative -> flat unit-norm vector (per-cell L1 + sqrt, with a
    relative floor so flat, textureless cells contribute ~0 instead of amplified noise)."""
    l1 = cells.sum(1, keepdims=True)
    tau = 0.1 * float(l1.mean()) + 1e-8
    v = np.sqrt(cells / (l1 + tau)).reshape(-1)
    return v / np.sqrt(len(cells))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


class HandcraftedBackend(Backend):
    name = "handcrafted"

    # relative weight of each block in the concatenated descriptor
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "layout": 1.0, "mean": 0.5, "colour": 1.0, "grad": 1.0, "dct": 0.8,
    }

    def __init__(self, weights: Dict[str, float] | None = None):
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.feat_dim = int(self._compute(Image.new("RGB", (S, S), (128, 128, 128))).shape[0])

    def identity(self) -> dict:
        return {"v": VERSION, "S": S, "bins": NBINS, "weights": tuple(sorted(self.weights.items()))}

    def describe(self) -> str:
        return f"handcrafted v{VERSION} ({self.feat_dim}-d, no download)"

    # ------------------------------------------------------------------------------------
    def extract(self, images: Sequence[Image.Image]) -> np.ndarray:
        if len(images) == 0:
            return np.zeros((0, self.feat_dim), np.float32)
        return np.stack([self._compute(im) for im in images]).astype(np.float32)

    def _compute(self, img: Image.Image) -> np.ndarray:
        small = img.convert("RGB").resize((S, S), RESAMPLE_LANCZOS)
        rgb8 = np.asarray(small, dtype=np.uint8)
        rgb = rgb8.astype(np.float32) / 255.0
        gray = rgb @ np.array([0.299, 0.587, 0.114], np.float32)

        blocks: Dict[str, np.ndarray] = {}

        # layout + mean colour --------------------------------------------------------
        ycc = np.asarray(small.convert("YCbCr").resize((8, 8), RESAMPLE_BOX), np.float32) / 255.0
        mean_col = ycc.reshape(-1, 3).mean(0)
        blocks["layout"] = _unit((ycc - mean_col).reshape(-1))
        blocks["mean"] = mean_col / (np.linalg.norm(mean_col) + 1e-8)

        # colour histograms on a pyramid ------------------------------------------------
        q = (rgb8 >> 6).astype(np.int32)                     # 4 levels per channel
        code = q[..., 0] * 16 + q[..., 1] * 4 + q[..., 2]    # 0..63
        parts = []
        for g in (1, 2):
            h = np.bincount((_CELL_IDX[g] * 64 + code).ravel(), minlength=g * g * 64)
            parts.append(_unit(_hellinger(h.reshape(g * g, 64).astype(np.float32))))
        blocks["colour"] = _unit(np.concatenate(parts))

        # gradient-orientation pyramid ---------------------------------------------------
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        mag = np.hypot(gx, gy)
        ang = np.arctan2(gy, gx) % np.pi
        pos = ang / np.pi * NBINS - 0.5
        b0 = np.floor(pos).astype(np.int32)
        frac = pos - b0
        b0m, b1m = b0 % NBINS, (b0 + 1) % NBINS
        fine = np.empty((CELLS, CELLS, NBINS), np.float32)
        cs = S // CELLS
        for b in range(NBINS):
            w = mag * ((b0m == b) * (1.0 - frac) + (b1m == b) * frac)
            fine[:, :, b] = w.reshape(CELLS, cs, CELLS, cs).sum((1, 3))
        parts = []
        for g in (1, 2, 4, 8):
            k = CELLS // g
            c = fine.reshape(g, k, g, k, NBINS).sum((1, 3)).reshape(g * g, NBINS)
            parts.append(_unit(_hellinger(c)))
        blocks["grad"] = _unit(np.concatenate(parts))

        # low-frequency DCT of a 32x32 grayscale ------------------------------------------
        g32 = gray.reshape(32, 2, 32, 2).mean((1, 3))
        m = _dct_matrix(32)
        low = (m @ g32 @ m.T)[:8, :8].reshape(-1)[1:]        # drop DC -> brightness invariant
        blocks["dct"] = _unit(low)

        return _unit(np.concatenate([self.weights[k] * blocks[k] for k in
                                     ("layout", "mean", "colour", "grad", "dct")]))
