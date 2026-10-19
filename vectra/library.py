"""The Library: one directory holding images, a SQLite database, trained heads and caches.

    vectra_data/
      settings.json     backend, embedding size, index type, active model
      library.db        image metadata + cached backbone features (SQLite)
      images/ab/abcd..  content-addressed copies of ingested images
      models/<id>/      trained heads (model.json + head.npz, no pickle)
      weights/          optional CNN weights
      thumbs/ index/    caches, safe to delete

Invariant that fixes the original's central bug: a query is embedded by ``self.embedder`` and the
index is built from the *same* embedder applied to the cached features. There is no persisted
embedding file that could belong to a different model.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageOps

from .backends import create_backend
from .backends.base import Backend
from .config import IMAGE_SUFFIXES, THUMB_SIZES, Settings
from .embedder import Embedder, list_models
from .errors import ImageError, LibraryNotFound, VectraError
from .imageio import decode, hamming, load, make_thumbnail, phash, prepare, read_file, sha256
from .index import HAVE_FAISS, VectorIndex
from .store import Store

log = logging.getLogger("vectra")
Progress = Optional[Callable[[int, int], None]]
NEAR_DUP_HAMMING = 6          # pHash distance (of 64 bits) at or below which we call it a near-duplicate
_MIME = {".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
         ".bmp": "image/bmp", ".gif": "image/gif", ".tif": "image/tiff"}


@dataclass
class AddResult:
    id: int
    created: bool
    sha256: str
    reason: str = ""            # "", "duplicate", "too small", "invalid: ..."


@dataclass
class IngestReport:
    scanned: int = 0
    added: int = 0
    duplicates: int = 0
    too_small: int = 0
    invalid: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class FeatureReport:
    done: int = 0
    failed: List[Tuple[int, str]] = field(default_factory=list)


@dataclass
class Match:
    id: int
    score: float
    kind: str                   # exact | near-duplicate | likely-copy | similar
    path: str
    sha256: str
    width: int
    height: int
    phash_distance: Optional[int]
    source: Optional[str]
    source_url: Optional[str]
    page_url: Optional[str]
    alt: Optional[str]
    title: Optional[str]

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["score"] = round(float(self.score), 5)
        return d


class Library:
    # ---------------------------------------------------------------------------------------
    # construction
    # ---------------------------------------------------------------------------------------
    def __init__(self, home: Union[str, Path], settings: Settings):
        self.home = Path(home)
        self.settings = settings
        for sub in ("images", "models", "weights", "thumbs", "index"):
            (self.home / sub).mkdir(parents=True, exist_ok=True)
        self.store = Store(self.home / "library.db")
        self._backend: Optional[Backend] = None
        self._embedder: Optional[Embedder] = None
        self._index: Optional[VectorIndex] = None
        self._index_stamp: Optional[Tuple] = None
        self._lock = threading.RLock()          # lazy init, index build
        self._model_lock = threading.Lock()     # serialise backend inference (torch / shared buffers)

    @classmethod
    def create(cls, home: Union[str, Path], **overrides) -> "Library":
        home = Path(home)
        settings = Settings(**{k: v for k, v in overrides.items() if v is not None}).validate()
        settings.save(home)
        return cls(home, settings)

    @classmethod
    def open(cls, home: Union[str, Path]) -> "Library":
        home = Path(home)
        return cls(home, Settings.load(home))

    @classmethod
    def exists(cls, home: Union[str, Path]) -> bool:
        return (Path(home) / "settings.json").exists()

    @classmethod
    def open_or_create(cls, home: Union[str, Path], **overrides) -> "Library":
        return cls.open(home) if cls.exists(home) else cls.create(home, **overrides)

    def close(self) -> None:
        self.store.close()

    # ---------------------------------------------------------------------------------------
    # lazily-built model parts
    # ---------------------------------------------------------------------------------------
    @property
    def backend(self) -> Backend:
        with self._lock:
            if self._backend is None:
                self._backend = create_backend(self.settings, self.home)
            return self._backend

    @property
    def embedder(self) -> Embedder:
        with self._lock:
            if self._embedder is None:
                active = self.settings.active_model
                if active:
                    self._embedder = Embedder.load(self.home / "models" / active, self.backend)
                else:
                    self._embedder = Embedder(self.backend)
            return self._embedder

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    def set_active_model(self, embedder: Optional[Embedder]) -> None:
        """Switch the active head (None = raw features). Saves settings, drops the index cache."""
        with self._lock:
            if embedder is None or embedder.kind == "raw":
                self.settings.active_model = None
                self._embedder = None
            else:
                self.settings.active_model = embedder.model_id
                self._embedder = embedder
            self.settings.save(self.home)
            self._invalidate_index()

    def use_model(self, model_id: str) -> Embedder:
        if model_id in ("raw", "none", ""):
            self.set_active_model(None)
            return self.embedder
        path = self.models_dir / model_id
        if not path.is_dir():
            known = ", ".join(m["model_id"] for m in list_models(self.models_dir)) or "(none trained yet)"
            raise VectraError(f"no such model {model_id!r}. Available: raw, {known}")
        emb = Embedder.load(path, self.backend)
        self.set_active_model(emb)
        return emb

    def _invalidate_index(self) -> None:
        self._index = None
        self._index_stamp = None

    # ---------------------------------------------------------------------------------------
    # paths & raw access
    # ---------------------------------------------------------------------------------------
    def image_path(self, rec: Dict) -> Path:
        return (self.home / "images" / rec["path"]) if rec["managed"] else Path(rec["path"])

    def image_bytes(self, image_id: int) -> Optional[Tuple[bytes, str]]:
        rec = self.store.get_image(image_id)
        if rec is None:
            return None
        try:
            return self.image_path(rec).read_bytes(), _MIME.get(rec["ext"], "application/octet-stream")
        except OSError:
            return None

    def thumbnail(self, image_id: int, size: int = 320) -> Optional[bytes]:
        if size not in THUMB_SIZES:
            raise VectraError(f"thumbnail size must be one of {THUMB_SIZES}")
        cache = self.home / "thumbs" / f"{int(image_id)}_{size}.jpg"
        try:
            return cache.read_bytes()
        except OSError:
            pass
        rec = self.store.get_image(image_id)
        if rec is None:
            return None
        try:
            data = make_thumbnail(load(self.image_path(rec), max_side=1280), size)
        except ImageError:
            return None
        try:
            tmp = cache.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, cache)
        except OSError:
            pass
        return data

    # ---------------------------------------------------------------------------------------
    # ingest
    # ---------------------------------------------------------------------------------------
    def _prepare(self, data: bytes, min_side: int = 0):
        """CPU-heavy, side-effect-free half of an add: -> (sha256, Prepared | None, reason)."""
        digest = sha256(data)
        try:
            prep = prepare(data)
        except ImageError as e:
            return digest, None, f"invalid: {e}"
        if min_side and min(prep.width, prep.height) < min_side:
            return digest, None, "too small"
        return digest, prep, ""

    def _commit(self, data: bytes, digest: str, prep, *, source: str, source_url: Optional[str],
                page_url: Optional[str], alt: Optional[str], title: Optional[str],
                original_path: Optional[Union[str, Path]], copy: Optional[bool]) -> AddResult:
        """Write the file (content-addressed) and insert the row. Idempotent."""
        copy = self.settings.copy_images if copy is None else copy
        if copy or original_path is None:
            rel = f"{digest[:2]}/{digest}{prep.ext}"
            dest = self.home / "images" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                tmp = dest.with_name(dest.name + f".{threading.get_ident()}.tmp")
                tmp.write_bytes(data)
                os.replace(tmp, dest)
            path, managed = rel, 1
        else:
            path, managed = str(Path(original_path).resolve()), 0
        image_id, created = self.store.add_image({
            "sha256": digest, "path": path, "managed": managed, "ext": prep.ext,
            "width": prep.width, "height": prep.height, "nbytes": prep.nbytes, "phash": prep.phash,
            "source": source, "source_url": source_url, "page_url": page_url,
            "alt": alt, "title": title,
        })
        if created:
            self._invalidate_index()
        return AddResult(image_id, created, digest, "" if created else "duplicate")

    def add_image_bytes(self, data: bytes, *, source: str = "upload", source_url: Optional[str] = None,
                        page_url: Optional[str] = None, alt: Optional[str] = None,
                        title: Optional[str] = None, original_path: Optional[Union[str, Path]] = None,
                        copy: Optional[bool] = None, min_side: int = 0) -> AddResult:
        """Validate and store one image. Idempotent: the same bytes are never stored twice."""
        digest = sha256(data)
        existing = self.store.find_by_sha(digest)
        if existing is not None:
            return AddResult(existing, False, digest, "duplicate")
        digest, prep, reason = self._prepare(data, min_side)
        if prep is None:
            return AddResult(-1, False, digest, reason)
        return self._commit(data, digest, prep, source=source, source_url=source_url, page_url=page_url,
                            alt=alt, title=title, original_path=original_path, copy=copy)

    def ingest_paths(self, paths: Iterable[Union[str, Path]], *, copy: Optional[bool] = None,
                     source: str = "local", progress: Progress = None, workers: int = 4) -> IngestReport:
        """Add images from files and/or folders (recursive). Safe to re-run: duplicates are skipped.

        Decoding/hashing runs in a thread pool; rows are committed in file order, so image ids
        are deterministic for a given folder."""
        report = IngestReport()
        files: List[Path] = []
        for raw in paths:
            p = Path(raw).expanduser()
            if p.is_dir():
                files += sorted(q for q in p.rglob("*") if q.is_file() and q.suffix.lower() in IMAGE_SUFFIXES)
            elif p.is_file():
                files.append(p)
            else:
                report.invalid.append((str(p), "not found"))
        report.scanned = len(files)

        def prep_one(path: Path):
            try:
                data = read_file(path)
            except ImageError as e:
                return path, None, "", None, f"invalid: {e}"
            digest = sha256(data)
            if self.store.find_by_sha(digest) is not None:
                return path, data, digest, None, "duplicate"
            digest, prep, reason = self._prepare(data)
            return path, data, digest, prep, reason

        done = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for path, data, digest, prep, reason in ex.map(prep_one, files):
                done += 1
                if prep is not None:
                    res = self._commit(
                        data, digest, prep, source=source, source_url=path.resolve().as_uri(),
                        page_url=None, alt=None, title=path.name, original_path=path, copy=copy)
                    reason = "" if res.created else "duplicate"
                if reason == "":
                    report.added += 1
                elif reason == "duplicate":
                    report.duplicates += 1
                elif reason == "too small":
                    report.too_small += 1
                else:
                    report.invalid.append((str(path), reason))
                if progress:
                    progress(done, len(files))
        return report

    def remove(self, ids: Sequence[int]) -> int:
        rows = self.store.delete_images(ids)
        for r in rows:
            if r["managed"]:
                try:
                    self.image_path(r).unlink()
                except OSError:
                    pass
            for t in (self.home / "thumbs").glob(f"{r['id']}_*.jpg"):
                try:
                    t.unlink()
                except OSError:
                    pass
        if rows:
            self._invalidate_index()
        return len(rows)

    # ---------------------------------------------------------------------------------------
    # features
    # ---------------------------------------------------------------------------------------
    def extract(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Backbone features, serialised so concurrent callers (web server) are safe."""
        with self._model_lock:
            return self.backend.extract(images)

    def ensure_features(self, progress: Progress = None, batch_size: int = 64,
                        workers: Optional[int] = None) -> FeatureReport:
        """Compute backbone features for every image that lacks them. Incremental."""
        report = FeatureReport()
        backend = self.backend
        bid = backend.backend_id
        missing = self.store.ids_missing_features(bid)
        if not missing:
            return report
        workers = workers or min(8, (os.cpu_count() or 2))

        def read(rec: Dict):
            try:
                return rec["id"], load(self.image_path(rec))
            except ImageError as e:
                return rec["id"], str(e)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for s in range(0, len(missing), batch_size):
                chunk = missing[s:s + batch_size]
                recs = self.store.get_images(chunk)
                loaded = list(ex.map(read, [recs[i] for i in chunk if i in recs]))
                ok = [(i, im) for i, im in loaded if isinstance(im, Image.Image)]
                for i, msg in loaded:
                    if not isinstance(msg, Image.Image):
                        report.failed.append((i, msg))
                if ok:
                    feats = self.extract([im for _, im in ok])
                    self.store.put_features(bid, [i for i, _ in ok], feats)
                    report.done += len(ok)
                if progress:
                    progress(min(s + batch_size, len(missing)), len(missing))
        self._invalidate_index()
        return report

    # ---------------------------------------------------------------------------------------
    # index & search
    # ---------------------------------------------------------------------------------------
    def get_index(self) -> VectorIndex:
        with self._lock:
            bid = self.backend.backend_id
            stamp = (self.embedder.model_id, bid, self.store.feature_stamp(bid))
            if self._index is not None and self._index_stamp == stamp:
                return self._index
            ids, feats = self.store.load_features(bid)
            if len(ids) == 0:
                raise VectraError("The library has no embedded images yet. Add some with:  vectra ingest <folder>")
            emb = self.embedder.embed_features(feats)
            key = f"{self.embedder.model_id}|{bid}"
            self._index = VectorIndex.build(
                ids, emb, self.settings.index_kind, self.settings.auto_hnsw_threshold,
                cache_dir=self.home / "index", cache_key=key,
            )
            self._index_stamp = stamp
            return self._index

    def _resolve_query(self, query) -> Tuple[Image.Image, Optional[str], Optional[int]]:
        """-> (image, sha256-of-source-bytes or None, library id if the query *is* a library image)"""
        if isinstance(query, Image.Image):
            return load(query), None, None
        if isinstance(query, (int, np.integer)):
            rec = self.store.get_image(int(query))
            if rec is None:
                raise VectraError(f"no image with id {int(query)}")
            return load(self.image_path(rec)), rec["sha256"], int(query)
        data = bytes(query) if isinstance(query, (bytes, bytearray, memoryview)) else read_file(query)
        return decode(data), sha256(data), None

    def search(self, query, k: int = 10, *, flip: bool = False, min_score: Optional[float] = None,
               exclude_exact: bool = False, exclude_ids: Sequence[int] = ()) -> List[Match]:
        """Top-``k`` most similar library images.

        query         path / bytes / PIL image / library image id
        flip          also try the mirrored query and keep the better score per image
        exclude_exact drop the stored copy of the query itself (bit-identical bytes, or its own id)
        """
        img, qsha, qid = self._resolve_query(query)
        variants = [img] + ([ImageOps.mirror(img)] if flip else [])
        emb = self.embedder.embed_features(self.extract(variants))
        index = self.get_index()

        excluded = set(int(i) for i in exclude_ids)
        if qid is not None and exclude_exact:
            excluded.add(qid)
        fetch = int(k) + len(excluded) + 4
        scores, ids = index.search(emb, fetch)
        best: Dict[int, float] = {}
        for srow, irow in zip(scores, ids):
            for s, i in zip(srow, irow):
                if i < 0:
                    continue
                if s > best.get(int(i), -2.0):
                    best[int(i)] = float(s)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        rows = self.store.get_images([i for i, _ in ranked])
        qhash = phash(img)
        copy_thr = self.embedder.head.info.get("copy_threshold") if self.embedder.head else None
        out: List[Match] = []
        for i, s in ranked:
            r = rows.get(i)
            if r is None or i in excluded:
                continue
            if exclude_exact and qsha is not None and r["sha256"] == qsha:
                continue
            if min_score is not None and s < min_score:
                continue
            dist = hamming(qhash, int(r["phash"], 16))
            if qsha is not None and r["sha256"] == qsha:
                kind = "exact"
            elif dist <= NEAR_DUP_HAMMING:
                kind = "near-duplicate"
            elif copy_thr is not None and s >= copy_thr:
                kind = "likely-copy"      # an edited copy: calibrated on held-out edits during `train`
            else:
                kind = "similar"
            out.append(Match(
                id=i, score=s, kind=kind, path=str(self.image_path(r)), sha256=r["sha256"],
                width=r["width"], height=r["height"], phash_distance=dist, source=r["source"],
                source_url=r["source_url"], page_url=r["page_url"], alt=r["alt"], title=r["title"],
            ))
            if len(out) >= k:
                break
        # margin test: a top match that clearly outdistances the runner-up is a copy even when its raw
        # score is modest (heavy crop / watermark). Threshold calibrated on held-out edits in `train`.
        margin = self.embedder.head.info.get("copy_margin") if self.embedder.head else None
        if margin is not None and len(out) >= 2 and out[0].kind == "similar" \
                and out[0].score - out[1].score >= margin:
            out[0].kind = "likely-copy"
        return out

    # ---------------------------------------------------------------------------------------
    # training / evaluation (implemented in their own modules)
    # ---------------------------------------------------------------------------------------
    def train(self, **kwargs):
        from .trainer import train_model
        return train_model(self, **kwargs)

    def evaluate(self, **kwargs):
        from .evaluate import run_eval
        return run_eval(self, **kwargs)

    # ---------------------------------------------------------------------------------------
    def stats(self) -> Dict:
        t = self.store.totals()
        info: Dict = {
            "home": str(self.home), "images": t["images"], "bytes": t["bytes"], "sources": t["sources"],
            "backend": self.settings.backend, "embed_dim_setting": self.settings.embed_dim,
            "index_kind_setting": self.settings.index_kind, "faiss": HAVE_FAISS,
            "active_model": self.settings.active_model or "raw", "version": self.settings.version,
        }
        try:
            bid = self.backend.backend_id
            info["backend_id"] = bid
            info["embedded"] = self.store.feature_count(bid)
            info["pending"] = t["images"] - info["embedded"]
            info["dim"] = self.embedder.dim
            info["model_kind"] = self.embedder.kind
        except VectraError as e:
            info["backend_error"] = str(e)
        return info

