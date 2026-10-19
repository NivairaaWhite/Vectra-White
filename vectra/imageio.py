"""Safe image decoding plus the tiny helpers every other module needs.

All images enter the system through :func:`decode`, which guarantees: RGB, upright (EXIF
orientation applied), transparency flattened onto white (a naive ``convert("RGB")`` turns
transparent pixels black), and a hard cap on pixel count so a decompression bomb cannot eat
the machine.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import ImageError

try:  # Pillow >= 9.1
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
    RESAMPLE_BOX = Image.Resampling.BOX
except AttributeError:  # pragma: no cover - very old Pillow
    RESAMPLE_LANCZOS, RESAMPLE_BICUBIC = Image.LANCZOS, Image.BICUBIC
    RESAMPLE_BILINEAR, RESAMPLE_BOX = Image.BILINEAR, Image.BOX

MAX_PIXELS = 100_000_000          # refuse anything bigger (decompression-bomb guard)
MAX_FILE_BYTES = 200 * 1024 * 1024
DECODE_HINT = 512                 # JPEGs far larger than this are decoded at reduced scale
ALLOWED_FORMATS = {
    "JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp",
    "BMP": ".bmp", "GIF": ".gif", "TIFF": ".tif",
}

ImageSource = Union[bytes, str, Path, Image.Image]


def to_rgb(img: Image.Image) -> Image.Image:
    """Convert any PIL image to plain RGB, flattening transparency onto white."""
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    if img.mode.startswith("I") or img.mode == "F":  # 16/32-bit grayscale
        arr = np.asarray(img).astype(np.float32)
        hi = float(arr.max()) if arr.size else 0.0
        if hi > 0:
            arr = arr * (255.0 / hi)
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    return img.convert("RGB")


def decode(data: bytes, max_side: Optional[int] = DECODE_HINT) -> Image.Image:
    """bytes -> upright RGB PIL image. Raises ImageError on anything unusable."""
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        if fmt not in ALLOWED_FORMATS:
            raise ImageError(f"unsupported image format: {fmt}")
        w, h = img.size
        if w < 1 or h < 1 or w * h > MAX_PIXELS:
            raise ImageError(f"image dimensions not accepted: {w}x{h}")
        if fmt == "JPEG" and max_side:
            img.draft("RGB", (max_side, max_side))  # cheap DCT-domain downscale for huge JPEGs
        img.load()
    except ImageError:
        raise
    except UnidentifiedImageError as e:
        raise ImageError("this is not an image file Vectra can read") from e
    except Image.DecompressionBombError as e:
        raise ImageError(f"image is too large (over {MAX_PIXELS // 1_000_000} megapixels)") from e
    except Exception as e:  # PIL raises many unrelated types for corrupt files
        raise ImageError("the image is corrupt or truncated and could not be decoded") from e
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:  # malformed EXIF should not make the image unusable
        pass
    return to_rgb(img)


def read_file(path: Union[str, Path], max_bytes: int = MAX_FILE_BYTES) -> bytes:
    p = Path(path)
    try:
        size = p.stat().st_size
        if size > max_bytes:
            raise ImageError(f"{p.name}: file too large ({size / 1e6:.0f} MB)")
        return p.read_bytes()
    except OSError as e:
        raise ImageError(f"cannot read {p}: {e}") from e


def load(src: ImageSource, max_side: Optional[int] = DECODE_HINT) -> Image.Image:
    """Accept bytes / path / PIL image and return an upright RGB PIL image."""
    if isinstance(src, Image.Image):
        return to_rgb(src)
    if isinstance(src, (bytes, bytearray, memoryview)):
        return decode(bytes(src), max_side)
    return decode(read_file(src), max_side)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sniff_ext(data: bytes) -> str:
    """File extension from the *content* (never trust a filename or URL)."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            return ALLOWED_FORMATS.get(im.format or "", ".img")
    except Exception:
        return ".img"


# --- perceptual hash --------------------------------------------------------------------
_DCT32: Optional[np.ndarray] = None


def _dct_matrix(n: int = 32) -> np.ndarray:
    global _DCT32
    if _DCT32 is None or _DCT32.shape[0] != n:
        k = np.arange(n)[:, None]
        i = np.arange(n)[None, :]
        m = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
        m[0, :] *= 1 / np.sqrt(2)
        _DCT32 = (m * np.sqrt(2.0 / n)).astype(np.float32)
    return _DCT32


def phash(img: Image.Image) -> int:
    """64-bit DCT perceptual hash. Survives resizing, recompression, mild colour changes."""
    g = img.convert("L").resize((32, 32), RESAMPLE_LANCZOS)
    a = np.asarray(g, dtype=np.float32)
    m = _dct_matrix(32)
    low = (m @ a @ m.T)[:8, :8].reshape(-1)
    bits = low > np.median(low[1:])  # median excludes the DC term
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def phash_hex(value: int) -> str:
    return f"{value:016x}"


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def make_thumbnail(img: Image.Image, size: int) -> bytes:
    t = img.copy()
    t.thumbnail((size, size), RESAMPLE_LANCZOS)
    out = io.BytesIO()
    t.save(out, format="JPEG", quality=82, optimize=True)
    return out.getvalue()


@dataclass
class Prepared:
    """Everything we record about an image at ingest time."""
    sha256: str
    ext: str
    width: int
    height: int
    nbytes: int
    phash: str


def prepare(data: bytes) -> Prepared:
    img = decode(data)
    w, h = img.size
    # report true dimensions even if we decoded a reduced-scale JPEG
    try:
        with Image.open(io.BytesIO(data)) as probe:
            w, h = probe.size
            exif = probe.getexif()
            if exif.get(0x0112, 1) in (5, 6, 7, 8):
                w, h = h, w
    except Exception:
        pass
    return Prepared(sha256(data), sniff_ext(data), w, h, len(data), phash_hex(phash(img)))
