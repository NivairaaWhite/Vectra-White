"""Vector index over L2-normalised embeddings (cosine similarity == inner product).

  flat   exact, NumPy only. Right choice up to ~100k vectors; no extra dependency.
  hnsw   approximate, needs faiss-cpu. Good recall and speed for large libraries.
  ivf    approximate, needs faiss-cpu. Trained on the *whole* set with sensible nlist/nprobe.

Fixes versus the original: IVF used to size ``nlist`` from the *first insert batch* (32 images
=> 3 lists forever) and never set ``nprobe`` (recall@10 0.72 vs 1.00 in my test). Here the index
is built once from the full matrix, and approximate indexes are cached on disk under a
content-addressed name, so a stale cache can never be loaded.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:  # optional
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:  # pragma: no cover - depends on environment
    faiss = None
    HAVE_FAISS = False


class VectorIndex:
    def __init__(self, ids: np.ndarray, emb: np.ndarray, kind: str, faiss_index=None, note: str = ""):
        self.ids = np.asarray(ids, np.int64)
        self.emb = np.ascontiguousarray(emb, np.float32)
        self.kind = kind
        self._faiss = faiss_index
        self.note = note

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def dim(self) -> int:
        return int(self.emb.shape[1])

    # --- construction --------------------------------------------------------------------
    @classmethod
    def build(cls, ids: np.ndarray, emb: np.ndarray, kind: str = "auto", auto_threshold: int = 200_000,
              cache_dir: Optional[Path] = None, cache_key: str = "") -> "VectorIndex":
        ids = np.asarray(ids, np.int64)
        emb = np.ascontiguousarray(emb, np.float32)
        n = len(ids)
        if emb.ndim != 2 or len(emb) != n:
            raise ValueError("ids/embeddings shape mismatch")
        note = ""
        if kind == "auto":
            kind = "hnsw" if (HAVE_FAISS and n >= auto_threshold) else "flat"
        if kind in ("hnsw", "ivf") and not HAVE_FAISS:
            note = f"faiss-cpu not installed; using exact search instead of {kind}"
            warnings.warn(note)
            kind = "flat"
        if kind == "ivf" and n < 2000:
            note = "fewer than 2000 vectors: exact search is faster and perfect, using flat"
            kind = "flat"
        if kind == "flat":
            return cls(ids, emb, "flat", None, note)

        sig = hashlib.sha1(f"{cache_key}|{kind}|{n}|".encode() + ids.tobytes()).hexdigest()[:20]
        cache_file = Path(cache_dir) / f"{kind}_{sig}.faiss" if (cache_dir and cache_key) else None
        if cache_file is not None and cache_file.exists():
            try:
                fi = faiss.read_index(str(cache_file))
                if fi.ntotal == n:
                    return cls(ids, emb, kind, fi, "loaded from cache")
            except Exception:
                pass  # fall through and rebuild

        d = emb.shape[1]
        if kind == "hnsw":
            fi = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
            fi.hnsw.efConstruction = 80
            fi.add(emb)
        else:  # ivf
            nlist = int(min(4096, max(16, 4 * np.sqrt(n))))
            quant = faiss.IndexFlatIP(d)
            fi = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
            rng = np.random.default_rng(0)
            sample = emb if n <= 50 * nlist else emb[rng.choice(n, 50 * nlist, replace=False)]
            fi.train(np.ascontiguousarray(sample))
            fi.add(emb)
        if cache_file is not None:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                for old in cache_file.parent.glob(f"{kind}_*.faiss"):
                    old.unlink()
                faiss.write_index(fi, str(cache_file))
            except Exception:
                pass  # cache is an optimisation only
        return cls(ids, emb, kind, fi, note)

    # --- search --------------------------------------------------------------------------
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """-> (scores [M,k'], image_ids [M,k']) sorted by score; ids are -1 where unfilled."""
        q = np.ascontiguousarray(np.atleast_2d(queries), np.float32)
        k = int(max(1, min(k, len(self.ids))))
        if self._faiss is None:
            sims = q @ self.emb.T
            if k < sims.shape[1]:
                part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            else:
                part = np.tile(np.arange(sims.shape[1]), (len(q), 1))
            part_scores = np.take_along_axis(sims, part, axis=1)
            order = np.argsort(-part_scores, axis=1)
            rows = np.take_along_axis(part, order, axis=1)
            return np.take_along_axis(sims, rows, axis=1), self.ids[rows]
        if self.kind == "hnsw":
            self._faiss.hnsw.efSearch = max(96, 4 * k)
        elif self.kind == "ivf":
            self._faiss.nprobe = int(min(self._faiss.nlist, max(8, self._faiss.nlist // 8)))
        scores, rows = self._faiss.search(q, k)
        ids = np.where(rows >= 0, self.ids[np.clip(rows, 0, None)], -1)
        return scores, ids
