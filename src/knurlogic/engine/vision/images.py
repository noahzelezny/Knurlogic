"""Request bytes -> a bounded RGB image, and the hash that names it.

The hash is of PIXELS, not of the base64 a client sent: the same picture
re-encoded (a client that re-compresses PNGs, a different base64 wrapping)
still hits the store and the prompt cache. exo's precedent (vision.py:724,
749-754) hashed `tobytes()` alone; here mode and size go in too, so a
100x1 and a 1x100 image of the same bytes cannot share a name.

Normalisation matches mlx-vlm 0.6.17's `utils.load_image` exactly --
`ImageOps.exif_transpose` then `.convert("RGB")` -- because the identity
gates (G1-G4) compare against goldens made through that function: the same
file must reach the processor as the same pixels on both sides.

CLAMPS, BEFORE HASHING (design D6, critique issue 11). An image arrives from
an HTTP client on a shared host:
  * MAX_BYTES of encoded input, checked before decoding anything;
  * BOMB_PIXELS: PIL's own decompression-bomb limit (Image.MAX_IMAGE_PIXELS
    default, 89,478,485 px), checked from the header before pixels are
    decoded; over it is refused (ImageRejected), not shrunk -- shrinking
    would first have to decode it;
  * MAX_DECODE_PIXELS: anything larger but legal is downscaled (aspect kept,
    BICUBIC, deterministic) before hashing, so the hash names what the
    processor actually sees. It is a safety bound, not a quality knob: each
    family's processor applies its own max_pixels after this (VisionSpec).

No URLs are fetched: a server that fetches client-supplied URLs is an SSRF
hole, and reading a local path named by a remote client is worse. Paths are
allowed only when the caller says the request is local (allow_paths).

PIL is imported lazily: this module is imported by the serve path's front
door, and a text-only server should not pay for PIL (not measured; the
same rule as every lazy import in engine/).
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import warnings
from pathlib import Path
from typing import Any, Optional, Tuple, Union

from . import ImageRejected

#: Encoded input bytes accepted per image (after base64 decode).
MAX_BYTES = 32 * 1024 * 1024
#: PIL's decompression-bomb limit (its Image.MAX_IMAGE_PIXELS default).
#: Recorded rather than read so a process that raised PIL's global cannot
#: raise ours.
BOMB_PIXELS = 89_478_485
#: Larger images are downscaled to at most this many pixels before hashing.
MAX_DECODE_PIXELS = 4096 * 4096


def _bytes_of(src: Union[str, bytes], allow_paths: bool) -> bytes:
    if isinstance(src, (bytes, bytearray)):
        data = bytes(src)
    elif isinstance(src, str):
        s = src.strip()
        if s.startswith("data:"):
            head, sep, body = s.partition(",")
            if not sep or ";base64" not in head:
                raise ImageRejected("data URL is not base64")
            s = body
        elif s.startswith(("http://", "https://")):
            raise ImageRejected("image URLs are not fetched by the server; "
                                "send the image as base64 or a data URL")
        elif allow_paths and len(s) < 4096 and Path(s).expanduser().is_file():
            p = Path(s).expanduser()
            if p.stat().st_size > MAX_BYTES:
                raise ImageRejected(f"image file over {MAX_BYTES} bytes")
            return p.read_bytes()
        if len(s) > MAX_BYTES * 4 // 3 + 4:
            raise ImageRejected(f"image over {MAX_BYTES} bytes")
        try:
            data = base64.b64decode(s, validate=False)
        except (binascii.Error, ValueError) as e:
            raise ImageRejected(f"image is not valid base64: {e}") from None
    else:
        raise ImageRejected(f"unsupported image source {type(src).__name__}")
    if len(data) > MAX_BYTES:
        raise ImageRejected(f"image over {MAX_BYTES} bytes")
    if not data:
        raise ImageRejected("empty image")
    return data


def decode(src: Union[str, bytes], *, allow_paths: bool = False) -> Any:
    """base64 / data URL / bytes (/ a local path if allow_paths) -> a PIL
    RGB image, clamped. Raises ImageRejected for anything unusable."""
    from PIL import Image, ImageOps
    data = _bytes_of(src, allow_paths)
    try:
        with warnings.catch_warnings():
            # PIL warns (not raises) between 1x and 2x its limit; ours is a
            # hard limit at 1x, checked below from the header.
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data))
            w, h = img.size
            if w * h > BOMB_PIXELS:
                raise ImageRejected(f"image is {w}x{h} = {w * h} px, over the "
                                    f"decompression-bomb limit {BOMB_PIXELS}")
            if getattr(img, "n_frames", 1) > 1:
                img.seek(0)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
    except ImageRejected:
        raise
    except Exception as e:  # PIL raises a zoo: UnidentifiedImageError, OSError...
        raise ImageRejected(f"image could not be decoded: {e}") from None
    return clamp(img)


def clamp(img: Any, max_pixels: Optional[int] = None) -> Any:
    """Downscale to at most max_pixels (default MAX_DECODE_PIXELS, read at
    call time), aspect kept; identity when under."""
    from PIL import Image
    max_pixels = MAX_DECODE_PIXELS if max_pixels is None else max_pixels
    w, h = img.size
    if w * h <= max_pixels:
        return img
    s = (max_pixels / (w * h)) ** 0.5
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    return img.resize((nw, nh), Image.BICUBIC)


def pixel_sha(img: Any) -> str:
    """sha256 hex over mode, size and the raw pixels -- the image's name in
    the store, the sentinel and ImageRef.sha."""
    h = hashlib.sha256()
    h.update(f"{img.mode}:{img.size[0]}x{img.size[1]}:".encode())
    h.update(img.tobytes())
    return h.hexdigest()


def load(src: Union[str, bytes], *,
         allow_paths: bool = False) -> Tuple[Any, str]:
    """decode() and pixel_sha() in one: (image, sha)."""
    img = decode(src, allow_paths=allow_paths)
    return img, pixel_sha(img)
