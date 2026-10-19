"""Procedural demo images.

The original project's demo set was random-coloured squares -- so uniform that contrastive
training on it taught the model nothing (and colour-jitter augmentation even destroyed the
only signal). These are structured scenes (gradients, shapes, textures, text) with enough
variety that retrieval and training behave like they do on real photos. Deterministic per seed.
"""

from __future__ import annotations

import colorsys
import math
import random
from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

_SIZES = [(320, 240), (256, 256), (240, 320), (384, 216), (300, 300), (352, 224)]
_WORDS = ["harbor", "ember", "atlas", "monsoon", "quartz", "lumen", "tundra", "saffron",
          "delta", "orbit", "velvet", "cobalt", "mosaic", "zephyr", "nimbus", "ivory"]


def _color(rng: random.Random, s=(0.3, 0.95), v=(0.35, 1.0)):
    r, g, b = colorsys.hsv_to_rgb(rng.random(), rng.uniform(*s), rng.uniform(*v))
    return int(r * 255), int(g * 255), int(b * 255)


def make_scene(seed: int) -> Image.Image:
    rng = random.Random(seed * 7919 + 13)
    w, h = _SIZES[seed % len(_SIZES)]
    # background gradient (vertical or diagonal)
    c1, c2 = _color(rng), _color(rng)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    if rng.random() < 0.5:
        t = np.broadcast_to(t, (h, w))
    else:
        t = (t + np.linspace(0, 1, w, dtype=np.float32)[None, :]) / 2
    bg = (np.array(c1, np.float32) * (1 - t[..., None]) + np.array(c2, np.float32) * t[..., None])
    img = Image.fromarray(bg.clip(0, 255).astype(np.uint8), "RGB")
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    for _ in range(rng.randint(4, 10)):
        kind = rng.choice(["ellipse", "rect", "poly", "line", "ring"])
        col = _color(rng) + (rng.randint(120, 255),)
        x0, y0 = rng.randint(-10, w - 20), rng.randint(-10, h - 20)
        x1, y1 = x0 + rng.randint(20, w // 2), y0 + rng.randint(20, h // 2)
        if kind == "ellipse":
            d.ellipse([x0, y0, x1, y1], fill=col)
        elif kind == "rect":
            d.rectangle([x0, y0, x1, y1], fill=col)
        elif kind == "poly":
            pts = [(rng.randint(x0, x1), rng.randint(y0, y1)) for _ in range(rng.randint(3, 6))]
            d.polygon(pts, fill=col)
        elif kind == "ring":
            d.ellipse([x0, y0, x1, y1], outline=col, width=rng.randint(2, 7))
        else:
            d.line([x0, y0, x1, y1], fill=col, width=rng.randint(2, 9))

    # textured patch: stripes or checker
    px, py = rng.randint(0, w // 2), rng.randint(0, h // 2)
    pw, ph = rng.randint(w // 5, w // 2), rng.randint(h // 5, h // 2)
    tex_col = _color(rng) + (150,)
    if rng.random() < 0.5:
        step = rng.randint(4, 12)
        for k in range(0, pw + ph, step):
            d.line([px + k, py, px, py + k], fill=tex_col, width=2)
    else:
        step = rng.randint(6, 16)
        for i in range(0, pw, step):
            for j in range(0, ph, step):
                if ((i + j) // step) % 2 == 0:
                    d.rectangle([px + i, py + j, px + i + step - 1, py + j + step - 1], fill=tex_col)

    img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")

    # a word of text, so scenes differ in fine structure too
    try:
        font = ImageFont.load_default(size=rng.randint(14, 30))
    except TypeError:  # old Pillow: fixed-size bitmap font
        font = ImageFont.load_default()
    ImageDraw.Draw(img).text((rng.randint(5, w // 2), rng.randint(5, h - 35)),
                             rng.choice(_WORDS), fill=_color(rng, (0, 0.4), (0.8, 1)), font=font)

    img = img.filter(ImageFilter.GaussianBlur(0.5))
    arr = np.asarray(img, np.float32) + np.random.default_rng(seed).normal(0, 2.0, (h, w, 3))
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB")


def make_demo_images(out_dir: Union[str, Path], n: int = 60, seed: int = 0) -> List[Path]:
    """Write ``n`` distinct scenes (mostly JPEG, some PNG) and return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        img = make_scene(seed * 100_003 + i)
        p = out / (f"demo_{i + 1:04d}" + (".png" if i % 5 == 4 else ".jpg"))
        if p.suffix == ".png":
            img.save(p)
        else:
            img.save(p, quality=92)
        paths.append(p)
    return paths
