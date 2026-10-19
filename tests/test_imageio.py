import io
import random

import numpy as np
import pytest
from PIL import Image

from vectra import imageio
from vectra.augment import crop_center, jpeg, rescale
from vectra.demo import make_scene
from vectra.errors import ImageError


def _bytes(img, fmt="PNG", **kw):
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def test_transparent_png_is_flattened_onto_white_not_black():
    rgba = Image.new("RGBA", (20, 20), (255, 0, 0, 0))          # fully transparent red
    out = imageio.decode(_bytes(rgba))
    assert out.mode == "RGB" and out.getpixel((5, 5)) == (255, 255, 255)


def test_palette_image_with_transparency():
    p = Image.new("P", (10, 10), 0)
    p.putpalette([0, 0, 0] * 256)
    out = imageio.decode(_bytes(p, "PNG", transparency=0))
    assert out.getpixel((1, 1)) == (255, 255, 255)


def test_exif_orientation_is_applied():
    img = Image.new("RGB", (40, 20), (10, 200, 30))
    exif = Image.Exif()
    exif[0x0112] = 6                                             # rotate 90 CW when displayed
    out = imageio.decode(_bytes(img, "JPEG", exif=exif))
    assert out.size == (20, 40)
    prep = imageio.prepare(_bytes(img, "JPEG", exif=exif))
    assert (prep.width, prep.height) == (20, 40)                 # reported size matches what you see


def test_sixteen_bit_grayscale():
    arr = (np.linspace(0, 65535, 100 * 100).reshape(100, 100)).astype(np.uint16)
    out = imageio.decode(_bytes(Image.fromarray(arr)))
    assert out.mode == "RGB" and max(out.getpixel((99, 99))) > 200


def test_garbage_and_unsupported_raise_clean_errors():
    with pytest.raises(ImageError) as e:
        imageio.decode(b"definitely not an image")
    assert "0x" not in str(e.value) and "BytesIO" not in str(e.value)   # no Python object reprs leaking
    with pytest.raises(ImageError):
        imageio.decode(_bytes(Image.new("RGB", (8, 8)), "PNG")[:30])    # truncated


def test_pixel_cap_blocks_decompression_bombs(monkeypatch):
    monkeypatch.setattr(imageio, "MAX_PIXELS", 100)
    with pytest.raises(ImageError):
        imageio.decode(_bytes(Image.new("RGB", (50, 50))))


def test_phash_survives_edits_but_separates_images():
    a, b = make_scene(1), make_scene(2)
    ha = imageio.phash(a)
    assert imageio.hamming(ha, imageio.phash(jpeg(a, 40))) <= 6
    assert imageio.hamming(ha, imageio.phash(rescale(a, 0.5))) <= 8
    assert imageio.hamming(ha, imageio.phash(b)) > 12


def test_sniff_ext_uses_content_not_filename():
    assert imageio.sniff_ext(_bytes(Image.new("RGB", (4, 4)), "JPEG")) == ".jpg"
    assert imageio.sniff_ext(_bytes(Image.new("RGB", (4, 4)), "PNG")) == ".png"
    assert imageio.sniff_ext(b"junk") == ".img"


def test_thumbnail_is_jpeg_and_bounded():
    t = Image.open(io.BytesIO(imageio.make_thumbnail(make_scene(3), 160)))
    assert t.format == "JPEG" and max(t.size) <= 160


def test_decompression_bomb_gets_an_honest_message():
    import io
    from PIL import Image
    from vectra.errors import ImageError
    from vectra.imageio import decode
    buf = io.BytesIO()
    Image.new("1", (20000, 20000)).save(buf, "PNG")        # 400 megapixels, ~50 KB on disk
    with pytest.raises(ImageError) as e:
        decode(buf.getvalue())
    assert "too large" in str(e.value) and "corrupt" not in str(e.value)
