"""Request bytes -> a bounded RGB image, and the hash that names it.

The hash is of PIXELS plus mode and size, not of the base64 a client sent,
so the same picture re-encoded still hits the store and the prompt cache.
Normalisation matches mlx-vlm 0.6.17's `utils.load_image` exactly
(`ImageOps.exif_transpose` then `.convert("RGB")`), because the identity
gates (G1-G4) compare against goldens made through that function.

Clamps, before hashing: MAX_BYTES of encoded input; BOMB_PIXELS (PIL's
decompression-bomb limit, read from the header) is refused, not shrunk;
MAX_DECODE_PIXELS downscales anything larger but legal. No URLs are
fetched (SSRF); local paths only when the caller says the request is local
(allow_paths). PIL is imported lazily. Design: docs/design/vision.md.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import warnings
from pathlib import Path
from typing import Any

from . import ImageRejected, ImageTooLarge

#: Encoded input bytes accepted per image (after base64 decode).
MAX_BYTES = 32 * 1024 * 1024
#: PIL's decompression-bomb limit (its Image.MAX_IMAGE_PIXELS default).
#: Recorded rather than read so a process that raised PIL's global cannot
#: raise ours.
BOMB_PIXELS = 89_478_485
#: Larger images are downscaled to at most this many pixels before hashing.
MAX_DECODE_PIXELS = 4096 * 4096


def _bytes_of(src: str | bytes, allow_paths: bool) -> bytes:
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
                raise ImageTooLarge(f"an image file is over the maximum of "
                                    f"{MAX_BYTES} bytes")
            return p.read_bytes()
        if len(s) > MAX_BYTES * 4 // 3 + 4:
            raise ImageTooLarge(f"an image is over the maximum of "
                                f"{MAX_BYTES} bytes")
        try:
            data = base64.b64decode(s, validate=False)
        except (binascii.Error, ValueError) as e:
            raise ImageRejected(f"image is not valid base64: {e}") from None
    else:
        raise ImageRejected(f"unsupported image source {type(src).__name__}")
    if len(data) > MAX_BYTES:
        raise ImageTooLarge(f"an image is {len(data)} bytes; the maximum is "
                            f"{MAX_BYTES} bytes")
    if not data:
        raise ImageRejected("empty image")
    return data


#: A JPEG decodes at 1/8 scale directly (its DCT), never unpacked whole,
#: so its limit is the reduced decode's: 64x the pixels.
JPEG_SCALE = 8


def _limit(img) -> int:
    """The largest image of this format decoded safely: formats that must
    be unpacked whole before shrinking (PNG and most others) stop at
    BOMB_PIXELS; a JPEG is decoded already reduced, so it may be
    JPEG_SCALE^2 times larger."""
    return BOMB_PIXELS * (JPEG_SCALE ** 2 if img.format == "JPEG" else 1)


def _bomb_message(size, limit: int = BOMB_PIXELS) -> str:
    edge = int(limit ** 0.5)
    got = (f"an image is {size[0]}x{size[1]} = {size[0] * size[1]} pixels"
           if size else "an image is far over the pixel limit")
    if limit == BOMB_PIXELS:
        why = ("since it must be unpacked whole before it can be shrunk. "
               "A JPEG may be 64x larger: it decodes already reduced.")
    else:
        why = "even decoded at 1/8 scale, as a JPEG is, that is too large."
    return (f"{got}; the maximum for its format is {limit} pixels (about "
            f"{edge}x{edge}), {why} Images under it are downscaled to what "
            f"the model takes, not refused.")


def decode(src: str | bytes, *, allow_paths: bool = False) -> Any:
    """base64 / data URL / bytes (/ a local path if allow_paths) -> a PIL
    RGB image, clamped. Raises ImageRejected for anything unusable."""
    from PIL import Image, ImageOps
    data = _bytes_of(src, allow_paths)
    try:
        with warnings.catch_warnings():
            # PIL's own bomb check refuses at open, before the format is
            # known; ours below is format-aware. PIL's is raised to the
            # largest limit ours allows (a JPEG's), not switched off: it is
            # process-wide, and anything else opening an image keeps a bound.
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            Image.MAX_IMAGE_PIXELS = BOMB_PIXELS * JPEG_SCALE ** 2
            img = Image.open(io.BytesIO(data))
            w, h = img.size
            if w * h > _limit(img):
                raise ImageTooLarge(_bomb_message((w, h), _limit(img)))
            if getattr(img, "n_frames", 1) > 1:
                img.seek(0)
            if w * h > MAX_DECODE_PIXELS:
                # decode at reduced size where the format can (JPEG's DCT
                # scaling): a legal 80 Mpx photo need not be unpacked whole
                # before clamp() downscales it
                s = (MAX_DECODE_PIXELS / (w * h)) ** 0.5
                img.draft("RGB", (max(1, int(w * s)), max(1, int(h * s))))
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
    except ImageRejected:
        raise
    # PIL raises a zoo: UnidentifiedImageError, OSError...  # decoding untrusted bytes
    # raises a zoo; any failure is a rejection
    except Exception as e:
        raise ImageRejected(f"image could not be decoded: {e}") from None
    return clamp(img)


def clamp(img: Any, max_pixels: int | None = None) -> Any:
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


def load(src: str | bytes, *,
         allow_paths: bool = False) -> tuple[Any, str]:
    """decode() and pixel_sha() in one: (image, sha)."""
    img = decode(src, allow_paths=allow_paths)
    return img, pixel_sha(img)
