import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vectra import Library
from vectra.augment import crop_center, flip, jpeg, rescale, watermark
from vectra.demo import make_demo_images
from vectra.embedder import Embedder, fit_pca
from vectra.errors import LibraryNotFound, ModelMismatch, VectraError
from vectra.imageio import sha256


def sha_of(path):
    return sha256(Path(path).read_bytes())


def test_missing_library_gives_actionable_error(tmp_path):
    with pytest.raises(LibraryNotFound) as e:
        Library.open(tmp_path / "nope")
    assert "vectra" in str(e.value) and "init" in str(e.value)


def test_ingest_is_idempotent_and_ids_follow_file_order(tmp_path, demo_dir):
    lib = Library.create(tmp_path / "lib")
    r1 = lib.ingest_paths([demo_dir])
    assert (r1.scanned, r1.added, r1.duplicates) == (60, 60, 0)
    r2 = lib.ingest_paths([demo_dir])
    assert (r2.added, r2.duplicates) == (0, 60)
    first = lib.store.get_image(1)
    assert first["title"] == "demo_0001.jpg"                      # deterministic, not thread-timing dependent
    assert lib.store.count() == 60


def test_unreadable_and_non_images_are_reported_not_fatal(tmp_path, demo_dir):
    junk = tmp_path / "junk"
    junk.mkdir()
    (junk / "fake.jpg").write_bytes(b"not an image")
    (junk / "notes.txt").write_text("ignored: wrong extension")
    lib = Library.create(tmp_path / "lib")
    rep = lib.ingest_paths([junk, tmp_path / "missing_folder", demo_dir / "demo_0001.jpg"])
    assert rep.added == 1 and len(rep.invalid) == 2


def test_search_finds_heavily_edited_copy_and_labels_exact(lib, demo_dir):
    src = demo_dir / "demo_0007.jpg"
    original = Image.open(src).convert("RGB")
    edited = jpeg(watermark(crop_center(rescale(original, 0.6), 0.8), random.Random(2), "SAMPLE"), 35)
    top = lib.search(edited, k=5)[0]
    assert top.sha256 == sha_of(src)
    exact = lib.search(src, k=3)[0]
    assert exact.kind == "exact" and exact.score == pytest.approx(1.0, abs=1e-3) and exact.sha256 == sha_of(src)


def test_exclude_exact_and_search_by_id(lib, demo_dir):
    src = demo_dir / "demo_0011.jpg"
    own = sha_of(src)
    assert all(m.sha256 != own for m in lib.search(src, k=5, exclude_exact=True))
    iid = lib.store.find_by_sha(own)
    res = lib.search(iid, k=5, exclude_exact=True)
    assert all(m.id != iid for m in res)
    assert lib.search(iid, k=1)[0].id == iid


def test_flip_option_finds_mirrored_copy_with_higher_score(lib, demo_dir):
    src = demo_dir / "demo_0021.jpg"
    mirrored = flip(Image.open(src).convert("RGB"))
    plain = lib.search(mirrored, k=1)[0]
    both = lib.search(mirrored, k=1, flip=True)[0]
    assert both.sha256 == sha_of(src) and both.score > plain.score
    assert both.score > 0.95


def test_incremental_ingest_keeps_old_images_in_index(tmp_path):
    """Regression: the original rebuilt the index from only the newest batch, silently dropping
    everything ingested earlier (5 new images => 5 vectors, 49 images on disk)."""
    a = make_demo_images(tmp_path / "a", n=30, seed=1)
    b = make_demo_images(tmp_path / "b", n=20, seed=2)
    lib = Library.create(tmp_path / "lib")
    lib.ingest_paths([tmp_path / "a"]); lib.ensure_features()
    assert len(lib.get_index()) == 30
    lib.ingest_paths([tmp_path / "b"]); lib.ensure_features()
    assert len(lib.get_index()) == 50
    assert lib.search(a[3], k=1)[0].sha256 == sha_of(a[3])        # old image still found
    assert lib.search(b[3], k=1)[0].sha256 == sha_of(b[3])        # new image found


def test_reopened_library_agrees_with_the_one_that_built_it(lib, demo_dir):
    """Regression for the original's central bug: index built by one embedder instance, searched by
    another with a different random projection head => garbage (top-1 self-match 1/12)."""
    q = jpeg(rescale(Image.open(demo_dir / "demo_0003.jpg").convert("RGB"), 0.5), 40)
    before = [(m.id, round(m.score, 5)) for m in lib.search(q, k=5)]
    home = lib.home
    lib.close()
    fresh = Library.open(home)
    after = [(m.id, round(m.score, 5)) for m in fresh.search(q, k=5)]
    assert before == after
    assert fresh.search(demo_dir / "demo_0003.jpg", k=1)[0].sha256 == sha_of(demo_dir / "demo_0003.jpg")


def test_model_id_is_a_hash_of_its_numbers_and_save_load_roundtrips(lib):
    ids, feats = lib.store.load_features(lib.backend.backend_id)
    head = fit_pca(feats, dim=32)
    e1 = Embedder(lib.backend, head)
    assert e1.model_id == Embedder(lib.backend, fit_pca(feats, dim=32)).model_id      # deterministic
    path = e1.save(lib.models_dir)
    e2 = Embedder.load(path, lib.backend)
    assert e2.model_id == e1.model_id
    assert np.allclose(e1.embed_features(feats), e2.embed_features(feats))
    assert np.allclose(np.linalg.norm(e2.embed_features(feats), axis=1), 1.0, atol=1e-5)


def test_model_for_another_backend_is_refused(lib, tmp_path):
    ids, feats = lib.store.load_features(lib.backend.backend_id)
    path = Embedder(lib.backend, fit_pca(feats, dim=16)).save(lib.models_dir)
    class Other:
        feat_dim = lib.backend.feat_dim
        name = "other"
        backend_id = "other-123"
    with pytest.raises(ModelMismatch):
        Embedder.load(path, Other())


def test_switching_models_keeps_search_self_consistent(lib, demo_dir):
    ids, feats = lib.store.load_features(lib.backend.backend_id)
    emb = Embedder(lib.backend, fit_pca(feats, dim=48))
    emb.save(lib.models_dir)
    target = sha_of(demo_dir / "demo_0031.jpg")
    q = jpeg(rescale(Image.open(demo_dir / "demo_0031.jpg").convert("RGB"), 0.5), 40)
    for choose in (emb.model_id, "raw", emb.model_id):
        lib.use_model(choose)
        assert lib.search(q, k=1)[0].sha256 == target
    with pytest.raises(VectraError):
        lib.use_model("no-such-model")


def test_remove_deletes_rows_files_thumbs_and_updates_index(lib, demo_dir):
    iid = lib.store.find_by_sha(sha_of(demo_dir / "demo_0006.jpg"))
    rec = lib.store.get_image(iid)
    assert lib.thumbnail(iid, 160)
    path = lib.image_path(rec)
    assert path.exists() and len(lib.get_index()) == 60
    assert lib.remove([iid]) == 1
    assert not path.exists() and not list((lib.home / "thumbs").glob(f"{iid}_*"))
    assert len(lib.get_index()) == 59
    assert all(m.id != iid for m in lib.search(demo_dir / "demo_0006.jpg", k=10))


def test_no_copy_mode_references_files_in_place(tmp_path, demo_dir):
    lib = Library.create(tmp_path / "lib", copy_images=False)
    lib.ingest_paths([demo_dir]); lib.ensure_features()
    rec = lib.store.get_image(1)
    assert rec["managed"] == 0 and Path(rec["path"]).is_absolute()
    assert not any((lib.home / "images").rglob("*.jpg"))
    assert lib.search(demo_dir / "demo_0001.jpg", k=1)[0].sha256 == sha_of(demo_dir / "demo_0001.jpg")
    lib.remove([1])
    assert (demo_dir / "demo_0001.jpg").exists()                  # never deletes the user's own files


def test_missing_external_file_is_reported_not_crash(tmp_path):
    src = make_demo_images(tmp_path / "src", n=6, seed=3)
    lib = Library.create(tmp_path / "lib", copy_images=False)
    lib.ingest_paths([tmp_path / "src"])
    src[0].unlink()
    rep = lib.ensure_features()
    assert rep.done == 5 and len(rep.failed) == 1


def test_stats_and_empty_search_message(tmp_path):
    lib = Library.create(tmp_path / "lib")
    assert lib.stats()["images"] == 0
    with pytest.raises(VectraError) as e:
        lib.search(Image.new("RGB", (32, 32)), k=3)
    assert "no embedded images" in str(e.value)


def test_min_score_filters(lib, demo_dir):
    res = lib.search(demo_dir / "demo_0002.jpg", k=20, min_score=0.99)
    assert len(res) >= 1 and all(m.score >= 0.99 for m in res)


def test_upload_bytes_are_stored_under_their_own_hash(lib):
    data = Path(lib.image_path(lib.store.get_image(1))).read_bytes()
    r = lib.add_image_bytes(data, source="upload")
    assert not r.created and r.reason == "duplicate" and r.id == 1
    novel = Image.new("RGB", (80, 80), (1, 2, 3))
    import io
    buf = io.BytesIO(); novel.save(buf, "PNG")
    r2 = lib.add_image_bytes(buf.getvalue(), source="upload")
    assert r2.created and (lib.home / "images" / r2.sha256[:2] / f"{r2.sha256}.png").exists()
    tiny = io.BytesIO(); Image.new("RGB", (10, 10)).save(tiny, "PNG")
    assert lib.add_image_bytes(tiny.getvalue(), min_side=64).reason == "too small"
