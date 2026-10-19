import json
import random

import pytest
from PIL import Image

from vectra import Library
from vectra.augment import jpeg, rescale
from vectra.cli import main
from vectra.demo import make_demo_images
from vectra.errors import BackendUnavailable, WeightsNotFound


# ---------------------------------------------------------------- CLI
def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_full_workflow(tmp_path, capsys):
    home = str(tmp_path / "lib")
    imgs = make_demo_images(tmp_path / "imgs", n=40, seed=21)
    assert run(capsys, "--home", home, "init")[0] == 0
    code, out, _ = run(capsys, "--home", home, "ingest", str(tmp_path / "imgs"))
    assert code == 0 and "40 added" in out
    code, out, _ = run(capsys, "--home", home, "ingest", str(tmp_path / "imgs"))
    assert "0 added" in out and "40 already" in out
    q = tmp_path / "q.jpg"
    jpeg(rescale(Image.open(imgs[4]).convert("RGB"), 0.5), 35).save(q)
    code, out, _ = run(capsys, "--home", home, "search", str(q), "-k", "3", "--json")
    res = json.loads(out)
    assert code == 0 and len(res) == 3 and res[0]["title"] == imgs[4].name and res[0]["score"] > res[1]["score"]
    code, out, _ = run(capsys, "--home", home, "info")
    assert "images           40" in out and "handcrafted" in out
    code, out, _ = run(capsys, "--home", home, "eval", "--queries", "20")
    assert code == 0 and "recall@1" in out
    code, out, _ = run(capsys, "--home", home, "models")
    assert "raw" in out and "*" in out
    assert run(capsys, "--home", home, "remove", "1", "2")[0] == 0
    assert "38" in run(capsys, "--home", home, "info")[1]
    assert run(capsys, "--home", home, "use", "raw")[0] == 0


def test_cli_train_then_use(tmp_path, capsys):
    pytest.importorskip("torch")
    home = str(tmp_path / "lib")
    code, out, _ = run(capsys, "--home", home, "build", "--demo", "50")
    assert code == 0 and "Ready: 50 images" in out
    code, out, _ = run(capsys, "--home", home, "models")
    assert out.count("\n") >= 2


def test_cli_errors_are_one_line_not_tracebacks(tmp_path, capsys):
    code, out, err = run(capsys, "--home", str(tmp_path / "nothing"), "info")
    assert code == 1 and err.startswith("error: No Vectra library") and "Traceback" not in err
    home = str(tmp_path / "lib")
    run(capsys, "--home", home, "init")
    code, out, err = run(capsys, "--home", home, "search", str(tmp_path / "nope.jpg"))
    assert code == 1 and err.startswith("error:")
    code, out, err = run(capsys, "--home", home, "train")
    assert code == 1 and "Cannot train" in out


def test_cli_no_command_prints_help(capsys):
    code, out, _ = run(capsys)
    assert code == 2 and "usage" in out.lower()


def test_cli_demo_command(tmp_path, capsys):
    code, out, _ = run(capsys, "demo", str(tmp_path / "d"), "--n", "7")
    assert code == 0 and len(list((tmp_path / "d").iterdir())) == 7


def test_init_never_overwrites(tmp_path, capsys):
    home = str(tmp_path / "lib")
    run(capsys, "--home", home, "init", "--backend", "handcrafted", "--dim", "64")
    code, out, _ = run(capsys, "--home", home, "init", "--dim", "128")
    assert "already exists" in out and Library.open(home).settings.embed_dim == 64


# ---------------------------------------------------------------- ResNet backend (needs torch)
torch = pytest.importorskip("torch")
from vectra.backends.resnet import ResNetBackend, find_weights  # noqa: E402


def test_missing_weights_message_tells_you_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "empty_torch_cache"))
    with pytest.raises(WeightsNotFound) as e:
        find_weights("resnet18", tmp_path, None)
    msg = str(e.value)
    assert "fetch-weights" in msg and "handcrafted" in msg
    with pytest.raises(WeightsNotFound):
        find_weights("resnet18", tmp_path, str(tmp_path / "nope.pth"))


def test_cnn_backend_is_frozen_eval_and_deterministic(monkeypatch):
    monkeypatch.setenv("VECTRA_ALLOW_RANDOM_WEIGHTS", "1")
    a = ResNetBackend("resnet18", allow_random=True)
    b = ResNetBackend("resnet18", allow_random=True)
    imgs = [Image.new("RGB", (50, 70), (i * 40, 90, 200 - i * 30)) for i in range(5)]
    fa, fb = a.extract(imgs), b.extract(imgs)
    assert fa.shape == (5, 512) and (fa == fb).all() and a.backend_id == b.backend_id
    assert not a.trunk.training                                   # BatchNorm can never drift
    assert not any(p.requires_grad for p in a.trunk.parameters())
    # extraction itself leaves BN statistics untouched
    bn = a.trunk[1]
    before = bn.running_mean.clone()
    a.extract(imgs)
    assert torch.equal(before, bn.running_mean)


def test_weights_file_is_hashed_into_backend_id(tmp_path):
    from torchvision import models
    paths = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        p = tmp_path / f"w{seed}.pth"
        torch.save(models.resnet18(weights=None).state_dict(), p)
        paths.append(str(p))
    a = ResNetBackend("resnet18", weights_path=paths[0])
    b = ResNetBackend("resnet18", weights_path=paths[1])
    a2 = ResNetBackend("resnet18", weights_path=paths[0])
    assert a.backend_id != b.backend_id and a.backend_id == a2.backend_id


def test_library_pipeline_with_cnn_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTRA_ALLOW_RANDOM_WEIGHTS", "1")
    imgs = make_demo_images(tmp_path / "imgs", n=24, seed=8)
    lib = Library.create(tmp_path / "lib", backend="resnet18")
    lib.ingest_paths([tmp_path / "imgs"])
    assert lib.ensure_features().done == 24
    q = jpeg(rescale(Image.open(imgs[2]).convert("RGB"), 0.6), 50)
    assert lib.search(q, k=1)[0].title == imgs[2].name
    rep = lib.train(steps=40, views=3, log=lambda m: None)
    assert rep.n_images == 24
    assert lib.search(imgs[5], k=1)[0].title == imgs[5].name
