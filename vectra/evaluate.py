"""Evaluation: how well does *this* library find an image after it has been modified?

For a random sample of your own images, apply each named edit (resize, JPEG, flip, crop,
watermark, ...), search the whole library with the edited copy, and check whether the original
comes back first. Reported per edit, so you can see exactly which modifications your current
backend/head survives -- the number a reverse-image-search tool actually lives or dies by.

The edits here are applied one at a time with fixed strengths and are *not* the random mixture
used for training (``augment.random_view``), apart from the final "mixed" row.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .augment import EVAL_TRANSFORMS
from .errors import ImageError, VectraError
from .imageio import load


@dataclass
class EvalReport:
    n_queries: int = 0
    gallery: int = 0
    k: int = 10
    model: str = ""
    backend: str = ""
    per_transform: Dict[str, Dict[str, float]] = field(default_factory=dict)
    mean_r1: float = 0.0                 # mean recall@1 over all edits except "identity"
    score_stats: Dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    def table(self) -> str:
        lines = [f"{'edit':<13}{'recall@1':>9}{'recall@5':>9}{'recall@10':>10}{'MRR':>8}"]
        for name, m in self.per_transform.items():
            lines.append(f"{name:<13}{m['r1']:>9.3f}{m['r5']:>9.3f}{m['r10']:>10.3f}{m['mrr']:>8.3f}")
        lines.append(f"{'mean (edits)':<13}{self.mean_r1:>9.3f}")
        return "\n".join(lines)


def run_eval(lib, n_queries: int = 200, seed: int = 0, k: int = 10,
             transforms: Optional[Dict[str, Callable]] = None,
             progress: Optional[Callable[[str], None]] = None) -> EvalReport:
    t0 = time.time()
    transforms = transforms or EVAL_TRANSFORMS
    lib.ensure_features()
    index = lib.get_index()
    rep = EvalReport(k=k, gallery=len(index), model=lib.embedder.model_id, backend=lib.backend.name)
    if len(index) < 5:
        raise VectraError("need at least 5 images in the library to evaluate")

    rng = random.Random(seed)
    pool = [int(i) for i in index.ids]
    ids = rng.sample(pool, min(n_queries, len(pool)))
    recs = lib.store.get_images(ids)
    originals = {}
    for i in ids:
        try:
            originals[i] = load(lib.image_path(recs[i]))
        except ImageError:
            continue
    ids = [i for i in ids if i in originals]
    rep.n_queries = len(ids)
    if len(ids) < 5:
        raise VectraError("could not read enough library images to evaluate")
    if len(index) < 50:
        rep.notes.append(f"small library ({len(index)} images): recall will look optimistic")

    pos, neg = [], []
    firsts = []
    for name, fn in transforms.items():
        if progress:
            progress(f"  evaluating {name} ...")
        queries = [fn(originals[i], random.Random(f"{seed}-{i}-{name}")) for i in ids]
        emb = lib.embedder.embed_features(lib.extract(queries))
        scores, found = index.search(emb, k)
        ranks = []
        for row, iid in enumerate(ids):
            hits = np.nonzero(found[row] == iid)[0]
            ranks.append(int(hits[0]) + 1 if len(hits) else None)
            for col in range(found.shape[1]):
                if found[row, col] < 0:
                    continue
                (pos if found[row, col] == iid else neg).append(float(scores[row, col]))
        r = np.array([x if x is not None else np.inf for x in ranks], dtype=float)
        rep.per_transform[name] = {
            "r1": float((r <= 1).mean()), "r5": float((r <= 5).mean()),
            "r10": float((r <= 10).mean()), "mrr": float(np.where(np.isinf(r), 0.0, 1.0 / r).mean()),
        }
        if name != "identity":
            firsts.append(rep.per_transform[name]["r1"])
    rep.mean_r1 = float(np.mean(firsts)) if firsts else 0.0
    if pos and neg:
        rep.score_stats = {
            "true_match_p05": float(np.percentile(pos, 5)),
            "true_match_median": float(np.median(pos)),
            "other_p95": float(np.percentile(neg, 95)),
            "other_p99": float(np.percentile(neg, 99)),
        }
        gap = rep.score_stats["true_match_p05"] - rep.score_stats["other_p99"]
        rep.notes.append(
            "score threshold: " + (
                f"scores above ~{rep.score_stats['other_p99']:.2f} were (almost) always true matches in this test"
                if gap > 0 else
                "true-match and other scores overlap here, so no single score threshold separates them cleanly"
            )
        )
    rep.seconds = time.time() - t0
    return rep
