import numpy as np
import pytest

from vectra import Library
from vectra.demo import make_demo_images
from vectra.errors import TrainingError
from vectra.trainer import retrieval_metrics

torch = pytest.importorskip("torch")


def test_retrieval_metrics_hand_computed():
    g = np.eye(4, dtype=np.float32)
    q = np.array([[1, 0, 0, 0], [0.1, 0.9, 0, 0], [0.6, 0, 0.5, 0]], np.float32)
    m = retrieval_metrics(q, g, np.array([0, 2, 2]))
    # query 0: rank 1; query 1 (true=2): sims [.1,.9,0,0] -> rank 3; query 2 (true=2): rank 2
    assert m["r1"] == pytest.approx(1 / 3) and m["r5"] == pytest.approx(1.0)
    assert m["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 2) / 3)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("train")
    make_demo_images(root / "imgs", n=90, seed=11)
    lib = Library.create(root / "lib")
    lib.ingest_paths([root / "imgs"])
    rep = lib.train(steps=150, views=4, log=lambda m: None)
    return lib, rep, root


def test_training_runs_selects_on_held_out_data_and_persists(trained):
    lib, rep, _ = trained
    assert rep.n_val >= 4 and rep.n_train + rep.n_val == rep.n_images == 90
    assert "raw" in rep.baselines and any(k.startswith("pca") for k in rep.baselines)
    if rep.chosen != "raw":
        assert rep.model_id == lib.settings.active_model == lib.embedder.model_id
        assert (lib.models_dir / rep.model_id / "head.npz").exists()
        # the model that was validated is exactly the model that is saved and now active
        reloaded = Library.open(lib.home)
        assert reloaded.embedder.model_id == rep.model_id
    best_base = max(m["mrr"] for m in rep.baselines.values())
    chosen_mrr = rep.trained["mrr"] if rep.accepted_trained else best_base
    assert chosen_mrr >= best_base - 1e-9                          # never worse than the best baseline


def test_trained_head_is_not_worse_than_raw_on_unseen_images(trained):
    """The point of training: generalise to images the head has never seen."""
    lib, rep, root = trained
    make_demo_images(root / "unseen", n=60, seed=99)
    other = Library.create(root / "other")
    other.ingest_paths([root / "unseen"]); other.ensure_features()
    raw = other.evaluate(n_queries=60, seed=1)
    import shutil
    if rep.chosen != "raw":
        shutil.copytree(lib.models_dir / rep.model_id, other.models_dir / rep.model_id)
        other.use_model(rep.model_id)
        trained_eval = other.evaluate(n_queries=60, seed=1)
        assert trained_eval.mean_r1 >= raw.mean_r1 - 0.02
    assert raw.per_transform["identity"]["r1"] == 1.0


def test_calibration_present_and_used(trained):
    lib, rep, _ = trained
    if rep.chosen == "raw":
        pytest.skip("raw chosen")
    info = lib.embedder.head.info
    assert 0 < info["copy_threshold"] <= 0.99 and 0 < info["copy_margin"] <= 1.0
    assert 0 <= info["copy_recall"] <= 1.0


def test_same_seed_gives_same_model(tmp_path):
    make_demo_images(tmp_path / "i", n=40, seed=5)
    ids = []
    for n in range(2):
        lib = Library.create(tmp_path / f"l{n}")
        lib.ingest_paths([tmp_path / "i"])
        ids.append(lib.train(steps=60, views=3, seed=4, log=lambda m: None).model_id)
    assert ids[0] == ids[1]


def test_too_few_images_is_a_clear_error(tmp_path):
    make_demo_images(tmp_path / "i", n=5, seed=1)
    lib = Library.create(tmp_path / "lib")
    lib.ingest_paths([tmp_path / "i"])
    with pytest.raises(TrainingError) as e:
        lib.train(log=lambda m: None)
    assert "at least 8" in str(e.value)


def test_eval_report_shape_and_notes(lib):
    rep = lib.evaluate(n_queries=40, seed=0)
    assert rep.n_queries == 40 and rep.gallery == 60
    assert {"identity", "flip", "crop_50%", "mixed"} <= set(rep.per_transform)
    assert rep.per_transform["identity"]["r1"] == 1.0
    assert not any("small library" in n for n in rep.notes)       # 60 images: no caveat
    assert "recall@1" in rep.table()


def test_eval_warns_that_tiny_libraries_look_optimistic(tmp_path):
    make_demo_images(tmp_path / "i", n=12, seed=2)
    lib = Library.create(tmp_path / "lib")
    lib.ingest_paths([tmp_path / "i"])
    assert any("small library" in n for n in lib.evaluate(n_queries=10).notes)
