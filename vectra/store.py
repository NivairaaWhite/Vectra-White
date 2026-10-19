"""SQLite storage: image metadata + cached backbone features.

Why SQLite instead of the original FAISS-file + pickle pair:
  * one atomic, inspectable file (``sqlite3 library.db``) instead of two files that can drift;
  * no pickle -> opening someone else's library cannot execute code;
  * incremental by construction: adding images never rebuilds anything that already exists;
  * features are cached per backend, so retraining the (cheap) head never re-runs the
    (expensive) CNN over every image.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images(
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  sha256     TEXT NOT NULL UNIQUE,
  path       TEXT NOT NULL,              -- relative to <home>/images when managed, else absolute
  managed    INTEGER NOT NULL DEFAULT 1,
  ext        TEXT NOT NULL,
  width      INTEGER NOT NULL,
  height     INTEGER NOT NULL,
  nbytes     INTEGER NOT NULL,
  phash      TEXT NOT NULL,              -- 16 hex chars (64-bit DCT hash)
  source     TEXT,                       -- local | crawl | upload | demo
  source_url TEXT,
  page_url   TEXT,
  alt        TEXT,
  title      TEXT,
  added_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS features(
  image_id   INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  backend_id TEXT NOT NULL,
  dim        INTEGER NOT NULL,
  vec        BLOB NOT NULL,              -- float16
  PRIMARY KEY(image_id, backend_id)
);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""

IMAGE_COLUMNS = (
    "sha256", "path", "managed", "ext", "width", "height", "nbytes", "phash",
    "source", "source_url", "page_url", "alt", "title", "added_at",
)


class Store:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        with self._conn() as c:
            c.executescript(_SCHEMA)
            c.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # --- connections (one per thread; sqlite3 objects must not cross threads) ----------
    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(str(self.db_path), timeout=30)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA foreign_keys=ON")
            self._local.conn = c
        return c

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None

    # --- images ------------------------------------------------------------------------
    def add_image(self, rec: Dict) -> Tuple[int, bool]:
        """Insert unless this exact content (sha256) is already stored. -> (id, created)"""
        row = {k: rec.get(k) for k in IMAGE_COLUMNS}
        row["added_at"] = row["added_at"] or time.time()
        row["managed"] = int(bool(row["managed"] if row["managed"] is not None else 1))
        cols = ", ".join(IMAGE_COLUMNS)
        marks = ", ".join("?" for _ in IMAGE_COLUMNS)
        with self._write_lock:
            c = self._conn()
            cur = c.execute(
                f"INSERT OR IGNORE INTO images({cols}) VALUES({marks})",
                [row[k] for k in IMAGE_COLUMNS],
            )
            created = cur.rowcount == 1
            c.commit()
            if created:
                return int(cur.lastrowid), True
            existing = c.execute("SELECT id FROM images WHERE sha256=?", (row["sha256"],)).fetchone()
            return int(existing["id"]), False

    def find_by_sha(self, sha: str) -> Optional[int]:
        r = self._conn().execute("SELECT id FROM images WHERE sha256=?", (sha,)).fetchone()
        return int(r["id"]) if r else None

    def get_image(self, image_id: int) -> Optional[Dict]:
        r = self._conn().execute("SELECT * FROM images WHERE id=?", (int(image_id),)).fetchone()
        return dict(r) if r else None

    def get_images(self, ids: Sequence[int]) -> Dict[int, Dict]:
        out: Dict[int, Dict] = {}
        ids = [int(i) for i in ids]
        for s in range(0, len(ids), 500):  # stay under SQLite's variable limit
            chunk = ids[s:s + 500]
            q = ",".join("?" for _ in chunk)
            for r in self._conn().execute(f"SELECT * FROM images WHERE id IN ({q})", chunk):
                out[int(r["id"])] = dict(r)
        return out

    def count(self) -> int:
        return int(self._conn().execute("SELECT COUNT(*) FROM images").fetchone()[0])

    def all_ids(self) -> List[int]:
        return [int(r[0]) for r in self._conn().execute("SELECT id FROM images ORDER BY id")]

    def delete_images(self, ids: Iterable[int]) -> List[Dict]:
        """Delete rows (features cascade). Returns the deleted rows so callers can unlink files."""
        ids = [int(i) for i in ids]
        rows = list(self.get_images(ids).values())
        with self._write_lock:
            c = self._conn()
            for s in range(0, len(ids), 500):
                chunk = ids[s:s + 500]
                c.execute(f"DELETE FROM images WHERE id IN ({','.join('?' for _ in chunk)})", chunk)
            c.commit()
        return rows

    def totals(self) -> Dict:
        r = self._conn().execute(
            "SELECT COUNT(*) n, COALESCE(SUM(nbytes),0) b, MIN(added_at) t0, MAX(added_at) t1 FROM images"
        ).fetchone()
        sources = {
            (x["source"] or "?"): int(x["n"])
            for x in self._conn().execute("SELECT source, COUNT(*) n FROM images GROUP BY source")
        }
        return {"images": int(r["n"]), "bytes": int(r["b"]), "sources": sources}

    # --- features ----------------------------------------------------------------------
    def ids_missing_features(self, backend_id: str) -> List[int]:
        q = (
            "SELECT i.id FROM images i LEFT JOIN features f "
            "ON f.image_id = i.id AND f.backend_id = ? WHERE f.image_id IS NULL ORDER BY i.id"
        )
        return [int(r[0]) for r in self._conn().execute(q, (backend_id,))]

    def put_features(self, backend_id: str, ids: Sequence[int], feats: np.ndarray) -> None:
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim != 2 or len(ids) != len(feats):
            raise ValueError("ids/features shape mismatch")
        rows = [
            (int(i), backend_id, int(feats.shape[1]), feats[j].astype(np.float16).tobytes())
            for j, i in enumerate(ids)
        ]
        with self._write_lock:
            c = self._conn()
            c.executemany(
                "INSERT OR REPLACE INTO features(image_id, backend_id, dim, vec) VALUES(?,?,?,?)", rows
            )
            c.commit()

    def load_features(self, backend_id: str, ids: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """-> (ids int64 [N], features float32 [N, F]) ordered by image id."""
        if ids is None:
            cur = self._conn().execute(
                "SELECT image_id, dim, vec FROM features WHERE backend_id=? ORDER BY image_id", (backend_id,)
            )
            rows = cur.fetchall()
        else:
            ids = sorted(int(i) for i in ids)
            rows = []
            for s in range(0, len(ids), 500):
                chunk = ids[s:s + 500]
                q = ",".join("?" for _ in chunk)
                rows.extend(self._conn().execute(
                    f"SELECT image_id, dim, vec FROM features WHERE backend_id=? AND image_id IN ({q})",
                    [backend_id, *chunk],
                ))
            rows.sort(key=lambda r: r["image_id"])
        if not rows:
            return np.zeros(0, np.int64), np.zeros((0, 0), np.float32)
        dim = int(rows[0]["dim"])
        out = np.empty((len(rows), dim), np.float32)
        out_ids = np.empty(len(rows), np.int64)
        for j, r in enumerate(rows):
            if int(r["dim"]) != dim:
                raise ValueError("inconsistent feature dimensions for one backend id")
            out[j] = np.frombuffer(r["vec"], dtype=np.float16)
            out_ids[j] = r["image_id"]
        return out_ids, out

    def feature_stamp(self, backend_id: str) -> Tuple[int, int]:
        """(row count, max image id): a cheap fingerprint that changes when features are added/removed."""
        r = self._conn().execute(
            "SELECT COUNT(*), COALESCE(MAX(image_id), 0) FROM features WHERE backend_id=?", (backend_id,)
        ).fetchone()
        return int(r[0]), int(r[1])

    def feature_count(self, backend_id: str) -> int:
        return int(self._conn().execute(
            "SELECT COUNT(*) FROM features WHERE backend_id=?", (backend_id,)
        ).fetchone()[0])

    def drop_features(self, keep_backend_id: Optional[str] = None) -> int:
        with self._write_lock:
            c = self._conn()
            if keep_backend_id:
                cur = c.execute("DELETE FROM features WHERE backend_id != ?", (keep_backend_id,))
            else:
                cur = c.execute("DELETE FROM features")
            c.commit()
            return cur.rowcount

    # --- meta --------------------------------------------------------------------------
    def meta_get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        r = self._conn().execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r["value"] if r else default

    def meta_set(self, key: str, value: str) -> None:
        with self._write_lock:
            c = self._conn()
            c.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?,?)", (key, value))
            c.commit()
