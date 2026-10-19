"""Image transformations for training and evaluation (Pillow + NumPy only).

Two families, deliberately different:
  * ``random_view``  -- the *training* augmentation: a random composition of mild/moderate edits.
  * ``EVAL_TRANSFORMS`` -- named, individually applied edits (plus a "mixed" one) so the
    evaluation can tell you *which* kinds of modification your library survives.

What the original did instead was RandomResizedCrop + heavy ColorJitter + 20 % grayscale, which
teaches a model to ignore colour -- precisely the cue that matters most for finding copies.
"""

from __future__ import annotations

import io
import math
import random
from typing import Callable, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from .imageio import RESAMPLE_BICUBIC, RESAMPLE_BILINEAR


# --- primitive edits ---------------------------------------------------------------------
def crop_random(img: Image.Image, rng: random.Random, area=(0.5, 1.0), ratio=(0.75, 1.333)) -> Image.Image:
    w, h = img.size
    for _ in range(10):
        a = rng.uniform(*area) * w * h
        r = math.exp(rng.uniform(math.log(ratio[0]), math.log(ratio[1])))
        cw, ch = int(round(math.sqrt(a * r))), int(round(math.sqrt(a / r)))
        if 8 <= cw <= w and 8 <= ch <= h:
            x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
            return img.crop((x, y, x + cw, y + ch))
    return img


def crop_center(img: Image.Image, area: float) -> Image.Image:
    w, h = img.size
    s = math.sqrt(area)
    cw, ch = max(8, int(w * s)), max(8, int(h * s))
    x, y = (w - cw) // 2, (h - ch) // 2
    return img.crop((x, y, x + cw, y + ch))


def flip(img: Image.Image) -> Image.Image:
    return ImageOps.mirror(img)


def rotate(img: Image.Image, degrees: float) -> Image.Image:
    fill = tuple(int(v) for v in np.asarray(img.resize((1, 1), RESAMPLE_BILINEAR)).reshape(-1)[:3])
    return img.rotate(degrees, resample=RESAMPLE_BICUBIC, expand=False, fillcolor=fill)


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out.convert("RGB")


def blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius))


def noise(img: Image.Image, sigma: float, seed: int = 0) -> Image.Image:
    arr = np.asarray(img, np.float32)
    arr = arr + np.random.default_rng(seed).normal(0, sigma, arr.shape)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB")


def rescale(img: Image.Image, factor: float) -> Image.Image:
    """Downscale then upscale back: what 'a smaller copy that was resized up again' looks like."""
    w, h = img.size
    small = img.resize((max(8, int(w * factor)), max(8, int(h * factor))), RESAMPLE_BICUBIC)
    return small.resize((w, h), RESAMPLE_BILINEAR)


def color_shift(img: Image.Image, brightness=1.0, contrast=1.0, saturation=1.0, hue=0.0) -> Image.Image:
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation)
    if hue:
        hsv = np.asarray(img.convert("HSV")).copy()
        hsv[..., 0] = (hsv[..., 0].astype(np.int32) + int(hue * 255)) % 256
        img = Image.fromarray(hsv, "HSV").convert("RGB")
    return img


def watermark(img: Image.Image, rng: random.Random, text: str = "SAMPLE") -> Image.Image:
    w, h = img.size
    base = img.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    try:
        font = ImageFont.load_default(size=max(12, h // 9))
    except TypeError:
        font = ImageFont.load_default()
    x, y = rng.randint(0, max(1, w // 2)), rng.randint(0, max(1, h - h // 5))
    d.rectangle([x - 4, y - 2, x + w // 3, y + h // 8], fill=(0, 0, 0, 70))
    d.text((x, y), text, fill=(255, 255, 255, 190), font=font)
    return Image.alpha_composite(base, layer).convert("RGB")


def border(img: Image.Image, frac: float, color=(20, 20, 20)) -> Image.Image:
    px = max(2, int(min(img.size) * frac))
    return ImageOps.expand(img, border=px, fill=color)


# --- training augmentation ---------------------------------------------------------------
def random_view(img: Image.Image, rng: random.Random, flip_p: float = 0.25) -> Image.Image:
    """One random, plausible 'copy of the same picture' (never destroys colour identity)."""
    v = img
    if rng.random() < 0.8:
        v = crop_random(v, rng, area=(0.55, 1.0))
    if rng.random() < flip_p:
        v = flip(v)
    if rng.random() < 0.3:
        v = rotate(v, rng.uniform(-10, 10))
    if rng.random() < 0.7:
        v = color_shift(v, rng.uniform(0.75, 1.25), rng.uniform(0.75, 1.25),
                        rng.uniform(0.7, 1.3), rng.uniform(-0.03, 0.03))
    if rng.random() < 0.3:
        v = rescale(v, rng.uniform(0.35, 0.8))
    if rng.random() < 0.25:
        v = blur(v, rng.uniform(0.3, 1.4))
    if rng.random() < 0.2:
        v = noise(v, rng.uniform(2, 9), rng.randint(0, 2**31 - 1))
    if rng.random() < 0.15:
        v = watermark(v, rng)
    if rng.random() < 0.1:
        v = border(v, rng.uniform(0.03, 0.1))
    if rng.random() < 0.5:
        v = jpeg(v, rng.randint(25, 90))
    return v


# --- evaluation transforms ---------------------------------------------------------------
EVAL_TRANSFORMS: Dict[str, Callable[[Image.Image, random.Random], Image.Image]] = {
    "identity":    lambda im, r: im,
    "resize_40%":  lambda im, r: rescale(im, 0.4),
    "jpeg_q20":    lambda im, r: jpeg(im, 20),
    "color_shift": lambda im, r: color_shift(im, 1.25, 0.8, 1.3, 0.04),
    "blur":        lambda im, r: blur(im, 1.5),
    "noise":       lambda im, r: noise(im, 12, r.randint(0, 2**31 - 1)),
    "watermark":   lambda im, r: watermark(im, r),
    "border":      lambda im, r: border(im, 0.08),
    "flip":        lambda im, r: flip(im),
    "rotate_10":   lambda im, r: rotate(im, 10 if r.random() < 0.5 else -10),
    "crop_80%":    lambda im, r: crop_center(im, 0.8),
    "crop_50%":    lambda im, r: crop_center(im, 0.5),
    "mixed":       lambda im, r: random_view(im, r, flip_p=0.3),
}
