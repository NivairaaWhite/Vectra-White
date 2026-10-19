"""CNN feature extractor (torchvision ResNet-18 / MobileNetV3-small), frozen, eval-mode only.

Differences from the original implementation, each fixing a concrete problem:
  * Pretrained weights are located *explicitly* (``--weights``, ``<library>/weights/``, the
    torch cache) and fetched only by ``vectra fetch-weights``. A missing file raises a clear
    error instead of the original's crash-on-first-use inside ``torchvision``'s downloader.
  * The backbone is never in train mode, so BatchNorm statistics cannot drift.
  * Features are GeM-pooled (p=3) from the last conv map, which retrieves better than the
    global-average pool the classifier uses; the full image is used (squashed to a square), no
    centre crop that throws away the borders.
  * Normalisation constants are defined here once (ImageNet), not twice with different values.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from ..errors import BackendUnavailable, WeightsNotFound
from ..imageio import RESAMPLE_BICUBIC
from .base import Backend

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
GEM_P = 3.0

_SPECS = {
    "resnet18": {"enum": "ResNet18_Weights", "feat": 512},
    "mobilenet_v3_small": {"enum": "MobileNet_V3_Small_Weights", "feat": 576},
}


def _require_torch():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as e:
        raise BackendUnavailable(
            "The CNN backends need PyTorch.  Install it with:  pip install torch torchvision\n"
            "(or use the built-in 'handcrafted' backend, which needs neither)."
        ) from e


def _weights_url(name: str) -> str:
    from torchvision import models
    return getattr(models, _SPECS[name]["enum"]).IMAGENET1K_V1.url


def find_weights(name: str, home: Optional[Path] = None, explicit: Optional[str] = None) -> Path:
    """Locate pretrained weights without touching the network."""
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise WeightsNotFound(f"weights file not found: {p}")
        return p
    candidates: List[Path] = []
    if home is not None:
        wdir = Path(home) / "weights"
        if wdir.is_dir():
            candidates += sorted(wdir.glob(f"{name}*.pth"))
    try:
        import torch
        fname = os.path.basename(_weights_url(name))
        candidates.append(Path(torch.hub.get_dir()) / "checkpoints" / fname)
    except Exception:
        pass
    for c in candidates:
        if c.is_file():
            return c
    raise WeightsNotFound(
        f"No pretrained weights for {name!r} found locally.\n"
        f"  - one-time download (needs internet):  vectra fetch-weights\n"
        f"  - or point at a file you already have:  vectra init --weights /path/to/{name}.pth\n"
        f"  - or use the built-in backend that needs no weights:  --backend handcrafted"
    )


def fetch_weights(name: str, home: Path) -> Path:
    """Explicit, user-initiated download of torchvision's ImageNet weights into the library."""
    _require_torch()
    import torch
    url = _weights_url(name)
    dest_dir = Path(home) / "weights"
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        torch.hub.load_state_dict_from_url(url, model_dir=str(dest_dir), progress=True,
                                           check_hash=True, weights_only=True)
    except TypeError:  # older torch without the weights_only kwarg
        torch.hub.load_state_dict_from_url(url, model_dir=str(dest_dir), progress=True, check_hash=True)
    return dest_dir / os.path.basename(url)


class ResNetBackend(Backend):
    def __init__(self, name: str = "resnet18", *, weights_path: Optional[str] = None,
                 home: Optional[Path] = None, image_size: int = 224,
                 device: Optional[str] = None, allow_random: bool = False, batch_size: int = 64):
        if name not in _SPECS:
            raise ValueError(f"unknown CNN backbone {name!r}")
        _require_torch()
        import torch
        import torch.nn as nn
        from torchvision import models

        self.name = name
        self.image_size = int(image_size)
        self.batch_size = batch_size
        self.feat_dim = _SPECS[name]["feat"]
        self._torch = torch

        model = getattr(models, name)(weights=None)
        if allow_random:  # tests only (VECTRA_ALLOW_RANDOM_WEIGHTS=1): never silently used otherwise
            with torch.random.fork_rng():
                torch.manual_seed(1234)
                model = getattr(models, name)(weights=None)
            self._weights_tag = "random-seed1234"
        else:
            path = find_weights(name, home, weights_path)
            state = torch.load(str(path), map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            self._weights_tag = h.hexdigest()[:16]

        if name == "resnet18":
            trunk = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                                  model.layer1, model.layer2, model.layer3, model.layer4)
        else:
            trunk = model.features
        trunk.eval()
        for p in trunk.parameters():
            p.requires_grad = False
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.trunk = trunk.to(self.device)

    def identity(self) -> dict:
        return {"arch": self.name, "weights": self._weights_tag, "size": self.image_size,
                "pool": f"gem{GEM_P}", "norm": "imagenet", "v": 1}

    def describe(self) -> str:
        return f"{self.name} ({self.feat_dim}-d, weights {self._weights_tag}, {self.device})"

    def _prep(self, img: Image.Image) -> np.ndarray:
        s = self.image_size
        a = np.asarray(img.convert("RGB").resize((s, s), RESAMPLE_BICUBIC), np.float32) / 255.0
        return ((a - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)

    def extract(self, images: Sequence[Image.Image]) -> np.ndarray:
        torch = self._torch
        if len(images) == 0:
            return np.zeros((0, self.feat_dim), np.float32)
        out = []
        with torch.inference_mode():
            for s in range(0, len(images), self.batch_size):
                batch = np.stack([self._prep(im) for im in images[s:s + self.batch_size]])
                x = torch.from_numpy(batch).to(self.device)
                fmap = self.trunk(x).clamp(min=1e-6)
                feats = fmap.pow(GEM_P).mean(dim=(2, 3)).pow(1.0 / GEM_P)
                out.append(feats.float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)
