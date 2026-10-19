import threading

import numpy as np
import pytest

from vectra.index import HAVE_FAISS, VectorIndex
from vectra.store import Store


def rec(i, **kw):
    base = dict(sha256=f"{i:064x}", path=f"{i:02x}/x.jpg", managed=1, ext=".jpg", width=10, height=10,
                nbytes=100, phash="0" * 16, source="test")
    base.update(kw)
    return base


# ---------------------------------------------------------------- store
def test_add_image_is_idempotent(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    a, created = s.add_image(rec(1))
    b, created2 = s.add_image(rec(1))
    assert created and not created2 and a == b and s.count() == 1


def test_features_roundtrip_and_missing(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    ids = [s.add_image(rec(i))[0] for i in range(5)]
    assert s.ids_missing_features("bk") == ids
    feats = np.random.default_rng(0).normal(size=(3, 7)).astype(np.float32)
    s.put_features("bk", ids[:3], feats)
    assert s.ids_missing_features("bk") == ids[3:]
    got_ids, got = s.load_features("bk")
    assert list(got_ids) == ids[:3] and np.allclose(got, feats, atol=2e-3)   # float16 storage
    assert s.ids_missing_features("other-backend") == ids                   # features are per backend


def test_delete_cascades_to_features_and_changes_stamp(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    ids = [s.add_image(rec(i))[0] for i in range(3)]
    s.put_features("bk", ids, np.ones((3, 4), np.float32))
    before = s.feature_stamp("bk")
    rows = s.delete_images([ids[0]])
    assert rows[0]["id"] == ids[0] and s.feature_count("bk") == 2
    assert s.feature_stamp("bk") != before


def test_concurrent_inserts(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    def work(base):
        for i in range(25):
            s.add_image(rec(base + i))
    ts = [threading.Thread(target=work, args=(b * 100,)) for b in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert s.count() == 100


def test_no_pickle_anywhere(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    s.add_image(rec(1))
    import vectra.store, vectra.library, vectra.embedder
    for mod in (vectra.store, vectra.library, vectra.embedder):
        src = open(mod.__file__).read()
        assert "import pickle" not in src and "pickle.load" not in src


# ---------------------------------------------------------------- index
def _clustered(n, d=64, clusters=30, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(clusters, d))
    x = (c[rng.integers(0, clusters, n)] + 0.7 * rng.normal(size=(n, d))).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_flat_index_matches_brute_force_and_maps_ids():
    x = _clustered(500)
    ids = np.arange(1000, 1500)
    idx = VectorIndex.build(ids, x, "flat")
    q = x[:20] + 0.01
    scores, got = idx.search(q, 10)
    brute = np.argsort(-(q @ x.T), axis=1)[:, :10]
    assert np.array_equal(got, ids[brute])
    assert np.all(np.diff(scores, axis=1) <= 1e-6)             # sorted descending


def test_k_larger_than_index():
    x = _clustered(5)
    scores, got = VectorIndex.build(np.arange(5), x, "flat").search(x[:1], 50)
    assert got.shape == (1, 5)


@pytest.mark.skipif(not HAVE_FAISS, reason="faiss not installed")
@pytest.mark.parametrize("kind", ["hnsw", "ivf"])
def test_approximate_indexes_have_high_recall(kind, tmp_path):
    """Regression: the original IVF index sized nlist from the first insert batch (3 lists) and
    never set nprobe, giving recall@10 of ~0.72. These are built once from all data."""
    x = _clustered(6000, d=64, seed=1)
    ids = np.arange(len(x))
    q = x[np.random.default_rng(2).integers(0, len(x), 200)] + 0.02 * np.random.default_rng(3).normal(size=(200, 64)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    truth = np.argsort(-(q @ x.T), axis=1)[:, :10]
    idx = VectorIndex.build(ids, x, kind, cache_dir=tmp_path, cache_key="k1")
    assert idx.kind == kind
    _, got = idx.search(q.astype(np.float32), 10)
    recall = np.mean([len(set(got[i]) & set(truth[i])) / 10 for i in range(len(q))])
    assert recall >= 0.93, recall


@pytest.mark.skipif(not HAVE_FAISS, reason="faiss not installed")
def test_index_cache_is_content_addressed(tmp_path):
    x = _clustered(2500, d=32)
    ids = np.arange(len(x))
    first = VectorIndex.build(ids, x, "hnsw", cache_dir=tmp_path, cache_key="m1")
    assert list(tmp_path.glob("hnsw_*.faiss")) and first.note != "loaded from cache"
    again = VectorIndex.build(ids, x, "hnsw", cache_dir=tmp_path, cache_key="m1")
    assert again.note == "loaded from cache"
    other_ids = ids[:-1]                                         # different content => different file
    third = VectorIndex.build(other_ids, x[:-1], "hnsw", cache_dir=tmp_path, cache_key="m1")
    assert third.note != "loaded from cache" and len(third) == len(x) - 1
    assert len(list(tmp_path.glob("hnsw_*.faiss"))) == 1       # stale cache replaced, not accumulated


def test_small_ivf_falls_back_to_exact():
    x = _clustered(300)
    idx = VectorIndex.build(np.arange(300), x, "ivf")
    assert idx.kind == "flat"
