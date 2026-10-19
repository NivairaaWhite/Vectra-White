"""Training: fit the head that sits on top of the frozen backbone.

What the original did, and why it learned nothing:
  * InfoNCE over a batch of 4 (3 negatives; chance loss = ln 4 = 1.39, which is where its loss sat),
    ~3 gradient steps per epoch, on random coloured squares, with colour-destroying augmentation;
  * the "frozen" backbone was still in train mode, so BatchNorm statistics drifted;
  * it saved the best-loss epoch but returned the last one (index and search used different weights).

What this does:
  1. Backbone features are computed ONCE for each image and for several augmented views of it,
     then cached. Training touches only these small vectors, so a batch of 256-1024 (hundreds
     of negatives) costs milliseconds even on a laptop CPU, and the CNN never changes.
  2. Candidate heads are scored on held-out images: *query = a randomly edited copy of a held-out
     image; success = retrieving the clean original from the whole gallery*. Candidates are
     raw features, and PCA-whitening at several strengths.
  3. The best PCA head initialises a linear head that is refined with symmetric InfoNCE.
  4. The refined head is kept only if it beats the best baseline on held-out data (``force`` to
     override). Training can therefore never silently make your search worse.
  5. Whatever is chosen is exactly what gets saved and activated.
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .augment import random_view
from .embedder import Embedder, Head, fit_pca, l2_normalize
from .errors import ImageError, TrainingError
from .imageio import load

Log = Callable[[str], None]


@dataclass
class TrainReport:
    n_images: int = 0
    n_train: int = 0
    n_val: int = 0
    views: int = 0
    baselines: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_baseline: str = ""
    trained: Optional[Dict[str, float]] = None
    trained_steps: int = 0
    chosen: str = "raw"                # raw | pca | trained
    model_id: str = "raw"
    accepted_trained: bool = False
    contrastive_skipped: str = ""
    copy_threshold: Optional[float] = None   # score above which a match is called a "likely copy"
    copy_recall: Optional[float] = None      # share of held-out edited copies flagged by score OR margin
    calibration: Dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [f"images used: {self.n_images} ({self.n_train} train / {self.n_val} held out), "
                 f"{self.views} edited views each",
                 "held-out retrieval of the original from an edited copy:"]
        for name, m in self.baselines.items():
            lines.append(f"  {name:<14} recall@1 {m['r1']:.3f}   recall@5 {m['r5']:.3f}   MRR {m['mrr']:.3f}")
        if self.trained:
            lines.append(f"  {'contrastive':<14} recall@1 {self.trained['r1']:.3f}   recall@5 "
                         f"{self.trained['r5']:.3f}   MRR {self.trained['mrr']:.3f}   (best of {self.trained_steps} steps)")
        elif self.contrastive_skipped:
            lines.append(f"  contrastive stage skipped: {self.contrastive_skipped}")
        verdict = {"raw": "raw features (no head) were best -> nothing to save",
                   "pca": f"kept PCA head {self.model_id}",
                   "trained": f"kept contrastive head {self.model_id}"}[self.chosen]
        lines.append(f"result: {verdict}   [{self.seconds:.1f}s]")
        if self.copy_threshold is not None:
            c = self.calibration
            lines.append(f"'likely copy' rule (about 1% false alarms): score >= {c['copy_threshold']:.2f}, "
                         f"or top match ahead of the runner-up by >= {c['copy_margin']:.2f}")
            lines.append(f"  catches {c['copy_recall_score']:.0%} of held-out edited copies by score alone, "
                         f"{c['copy_recall']:.0%} with the margin test")
        return "\n".join(lines)


# --- metric -------------------------------------------------------------------------------
def retrieval_metrics(q_emb: np.ndarray, g_emb: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """recall@1/5 and MRR of finding ``targets[i]`` (a row of the gallery) for query i."""
    sims = q_emb @ g_emb.T
    true = sims[np.arange(len(targets)), targets][:, None]
    rank = 1 + (sims > true).sum(1)           # ties resolved optimistically for the true item only
    return {"r1": float((rank == 1).mean()), "r5": float((rank <= 5).mean()), "mrr": float((1.0 / rank).mean())}


def _embed(feats: np.ndarray, mu: Optional[np.ndarray], W: Optional[np.ndarray]) -> np.ndarray:
    x = feats if mu is None else feats - mu
    if W is not None:
        x = x @ W
    return l2_normalize(x)


# --- view cache ---------------------------------------------------------------------------
def build_views(lib, ids: List[int], n_views: int, seed: int, workers: int, log: Log) -> Tuple[np.ndarray, np.ndarray]:
    """Backbone features of ``n_views`` random edits of each image -> [n, n_views, F]."""
    recs = lib.store.get_images(ids)
    f = lib.backend.feat_dim
    out = np.zeros((len(ids), n_views, f), np.float32)
    ok = np.ones(len(ids), bool)

    def make(job: Tuple[int, int]):
        row, iid = job
        try:
            img = load(lib.image_path(recs[iid]))
        except ImageError:
            return row, None
        return row, [random_view(img, random.Random(f"{seed}-{iid}-{v}")) for v in range(n_views)]

    jobs = list(enumerate(ids))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        pending: List[Tuple[int, List[Image.Image]]] = []
        n_pend = 0
        done = 0

        def flush():
            nonlocal pending, n_pend
            if not pending:
                return
            flat = [im for _, views in pending for im in views]
            feats = lib.extract(flat).reshape(len(pending), n_views, f)
            for (row, _), fe in zip(pending, feats):
                out[row] = fe
            pending, n_pend = [], 0

        for row, views in ex.map(make, jobs):
            done += 1
            if views is None:
                ok[row] = False
            else:
                pending.append((row, views))
                n_pend += 1
                if n_pend >= 16:
                    flush()
            if done % 200 == 0:
                log(f"  edited views: {done}/{len(ids)} images ({time.time() - t0:.0f}s)")
        flush()
    if not ok.all():
        log(f"  skipped {int((~ok).sum())} unreadable images")
    return out, ok


# --- contrastive stage --------------------------------------------------------------------
def _train_linear_head(X_all: np.ndarray, mu: np.ndarray, W0: np.ndarray,
                       val_fn: Callable[[np.ndarray], Dict[str, float]], *, steps: int, batch: int,
                       lr: float, temperature: float, reg: float, seed: int, eval_every: int,
                       patience: int, log: Log) -> Tuple[np.ndarray, Dict[str, float], int]:
    """Refine W (starting at ``W0``) with symmetric InfoNCE over cached views.

    X_all [n, V+1, F]: view 0 is the clean image, 1..V are edits. Two different views of the same
    image are positives; every other image in the batch is a negative.
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.from_numpy(X_all).to(dev)
    mu_t = torch.from_numpy(mu).to(dev)
    W0_t = torch.from_numpy(W0).to(dev)
    A = torch.zeros_like(W0_t, requires_grad=True)
    w_scale = float(W0_t.pow(2).mean().sqrt()) + 1e-12
    opt = torch.optim.Adam([A], lr=lr * w_scale)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    n, nv = X.shape[0], X.shape[1]
    b = min(batch, n)

    best_m, best_W, best_step, since = None, W0.copy(), 0, 0
    for step in range(1, steps + 1):
        idx = torch.randperm(n, device=dev)[:b]
        vi = torch.randint(0, nv, (b,), device=dev)
        vj = (vi + torch.randint(1, nv, (b,), device=dev)) % nv     # a different view of the same image
        W = W0_t + A
        z1 = F.normalize((X[idx, vi] - mu_t) @ W, dim=1)
        z2 = F.normalize((X[idx, vj] - mu_t) @ W, dim=1)
        logits = z1 @ z2.T / temperature
        labels = torch.arange(b, device=dev)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        loss = loss + reg * A.pow(2).mean() / (W0_t.pow(2).mean() + 1e-12)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == steps:
            W_np = (W0_t + A).detach().cpu().numpy().astype(np.float32)
            m = val_fn(W_np)
            if best_m is None or m["mrr"] > best_m["mrr"] + 1e-4:
                best_m, best_W, best_step, since = m, W_np, step, 0
            else:
                since += 1
            if step % (eval_every * 4) == 0:
                log(f"  step {step:>4}  loss {loss.item():.3f}  held-out MRR {m['mrr']:.3f}  (best {best_m['mrr']:.3f})")
            if since >= patience:
                log(f"  early stop at step {step} (no held-out improvement for {patience} checks)")
                break
    return best_W, best_m or {"r1": 0.0, "r5": 0.0, "mrr": 0.0}, best_step


# --- public entry point --------------------------------------------------------------------
def train_model(lib, *, steps: int = 400, batch: int = 512, views: int = 6, temperature: float = 0.1,
                lr: float = 0.02, reg: float = 0.05, max_images: int = 4000, val_frac: float = 0.2,
                seed: int = 0, min_images_contrastive: int = 24, dim: Optional[int] = None,
                whiten_grid: Tuple[float, ...] = (0.0, 0.5, 1.0), force: bool = False,
                workers: Optional[int] = None, log: Log = print) -> TrainReport:
    import os
    t0 = time.time()
    rng = random.Random(seed)
    workers = workers or min(8, os.cpu_count() or 2)
    dim = int(dim or lib.settings.embed_dim)
    rep = TrainReport(views=views)

    fr = lib.ensure_features()
    if fr.failed:
        log(f"  note: {len(fr.failed)} images could not be read and are ignored")
    bid = lib.backend.backend_id
    ids, feats = lib.store.load_features(bid)
    if len(ids) < 8:
        raise TrainingError(f"need at least 8 readable images to train (have {len(ids)}). "
                            f"Ingest more with `vectra ingest <folder>` or try `vectra demo`.")

    # drop images whose perceptual hash is identical: they would be each other's false negatives
    recs = lib.store.get_images(list(ids))
    seen, keep = set(), []
    for row, iid in enumerate(ids):
        h = recs[int(iid)]["phash"]
        if h not in seen:
            seen.add(h)
            keep.append(row)
    if len(keep) < len(ids):
        log(f"  ignoring {len(ids) - len(keep)} visual duplicates for training")
    ids, feats = ids[keep], feats[keep]
    if len(ids) > max_images:
        sel = sorted(rng.sample(range(len(ids)), max_images))
        ids, feats = ids[sel], feats[sel]
    n = len(ids)
    if n < 8:
        raise TrainingError("fewer than 8 distinct images after removing duplicates")

    log(f"Building {views} edited views for {n} images ...")
    views_feats, ok = build_views(lib, [int(i) for i in ids], views, seed, workers, log)
    ids, feats, views_feats = ids[ok], feats[ok], views_feats[ok]
    n = len(ids)
    if n < 8:
        raise TrainingError("fewer than 8 images could be read")

    order = list(range(n))
    rng.shuffle(order)
    n_val = max(4, int(round(val_frac * n)))
    n_val = min(n_val, n - 4)
    val_idx = np.array(sorted(order[:n_val]))
    train_idx = np.array(sorted(order[n_val:]))
    rep.n_images, rep.n_train, rep.n_val = n, len(train_idx), len(val_idx)

    q_feats = views_feats[val_idx].reshape(-1, views_feats.shape[2])
    q_targets = np.repeat(val_idx, views)

    def score(mu: Optional[np.ndarray], W: Optional[np.ndarray]) -> Dict[str, float]:
        g = _embed(feats, mu, W)
        q = _embed(q_feats, mu, W)
        return retrieval_metrics(q, g, q_targets)

    def calibrate(mu: Optional[np.ndarray], W: Optional[np.ndarray]) -> Dict[str, float]:
        """Decide when a match is a *copy* rather than a lookalike, at about 1% false alarms.

        Two complementary tests, both measured on held-out edited copies:
          score   the match scores above what the best wrong match reaches for 99% of queries;
          margin  the top match is ahead of the runner-up by more than the 99th-percentile gap
                  seen when the true image is NOT in the gallery (the spirit of Lowe's ratio test).
        """
        sims = _embed(q_feats, mu, W) @ _embed(feats, mu, W).T
        rows = np.arange(len(q_targets))
        pos = sims[rows, q_targets].copy()
        sims[rows, q_targets] = -np.inf                 # now: "the original is not in the library"
        top2 = np.sort(sims, axis=1)[:, -2:]
        best_neg, second_neg = top2[:, 1], top2[:, 0]
        thr = float(np.clip(np.quantile(best_neg, 0.99) + 0.01, 0.0, 0.99))
        mthr = float(np.clip(np.quantile(best_neg - second_neg, 0.99) + 0.01, 0.0, 1.0))
        by_score = pos >= thr
        by_margin = (pos - best_neg) >= mthr
        return {"copy_threshold": round(thr, 4), "copy_margin": round(mthr, 4),
                "copy_recall_score": float(by_score.mean()),
                "copy_recall_margin": float(by_margin.mean()),
                "copy_recall": float((by_score | by_margin).mean())}

    # 1) baselines ------------------------------------------------------------------------
    cands: Dict[str, Tuple[Optional[Head], Dict[str, float]]] = {}
    cands["raw"] = (None, score(None, None))
    pca_heads: Dict[float, Head] = {}
    for w in whiten_grid:
        try:
            h = fit_pca(feats[train_idx], dim=dim, whiten=w)
        except Exception as e:  # pragma: no cover - degenerate data
            log(f"  PCA(whiten={w}) failed: {e}")
            continue
        pca_heads[w] = h
        cands[f"pca w={w:g}"] = (h, score(h.mu, h.W))
    rep.baselines = {k: v[1] for k, v in cands.items()}
    best_name = max(cands, key=lambda k: (cands[k][1]["mrr"], cands[k][1]["r1"]))
    rep.best_baseline = best_name
    best_head, best_metrics = cands[best_name]
    log(f"Best baseline on held-out images: {best_name} (MRR {best_metrics['mrr']:.3f})")

    chosen_head, chosen_kind = best_head, ("raw" if best_head is None else "pca")

    # 2) contrastive refinement --------------------------------------------------------------
    if n < min_images_contrastive:
        rep.contrastive_skipped = f"only {n} images (need {min_images_contrastive}); a head trained on so few would just memorise them"
    else:
        try:
            import torch  # noqa: F401
        except ImportError:
            rep.contrastive_skipped = "PyTorch not installed (pip install torch); PCA head is still used"
        else:
            init = pca_heads.get(0.5) or next(iter(pca_heads.values()))
            if best_head is not None:
                init = best_head
            X_all = np.concatenate([feats[:, None, :], views_feats], axis=1)[train_idx]
            log(f"Contrastive refinement: batch {min(batch, len(train_idx))}, up to {steps} steps ...")
            W, m, step = _train_linear_head(
                X_all, init.mu, init.W, lambda Wn: score(init.mu, Wn),
                steps=steps, batch=batch, lr=lr, temperature=temperature, reg=reg, seed=seed,
                eval_every=max(5, steps // 40), patience=8, log=log)
            rep.trained, rep.trained_steps = m, step
            if (m["mrr"] > best_metrics["mrr"] + 0.002) or force:
                chosen_head = Head("trained", init.mu, W, {**init.info, "steps": step})
                chosen_kind, rep.accepted_trained = "trained", True
            else:
                log("  contrastive head did not beat the baseline on held-out images; keeping the baseline")

    # 3) persist exactly what was validated --------------------------------------------------
    rep.chosen = chosen_kind
    if chosen_head is None:
        lib.set_active_model(None)
        rep.model_id = "raw"
    else:
        cal = calibrate(chosen_head.mu, chosen_head.W)
        rep.copy_threshold, rep.copy_recall = cal["copy_threshold"], cal["copy_recall"]
        rep.calibration = cal
        chosen_head.info.update(cal)
        chosen_head.info.update({
            "n_images": n, "views": views, "seed": seed,
            "heldout": (rep.trained if chosen_kind == "trained" else best_metrics),
            "baseline": best_name,
        })
        emb = Embedder(lib.backend, chosen_head)
        emb.save(lib.models_dir)
        lib.set_active_model(emb)
        rep.model_id = emb.model_id
    rep.seconds = time.time() - t0
    return rep
