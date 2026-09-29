"""Build the shared vision goldens in the reference interpreter
(mlx-vlm 0.6.17):

    $KNURLOGIC_VLM_PYTHON tests/goldens/build_p0.py

p0_masked_scatter.npz  mlx-vlm's own gemma4 `masked_scatter` on seeded
                       inputs -- holds engine/vision/scatter.py's vendored
                       copy to the reference.
p0_load_image.npz      a PNG carrying an EXIF orientation (6, rotate 90),
                       and the pixels mlx-vlm's `utils.load_image` makes of
                       it -- holds images.decode's normalisation to the
                       reference the identity gates are made through.

Tiny, float32, seed 0; no model files are read.
"""
import base64
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fixtures_vision as fv  # noqa: E402

import mlx.core as mx  # noqa: E402
from mlx_vlm.models.gemma4.gemma4 import masked_scatter  # noqa: E402
from mlx_vlm.utils import load_image  # noqa: E402
from PIL import Image  # noqa: E402


def scatter():
    mx.random.seed(0)
    L, D, n = 11, 8, 4
    embeds = mx.random.normal((1, L, D)).astype(mx.float32)
    mask_rows = np.zeros(L, dtype=bool)
    mask_rows[[2, 3, 7, 8]] = True
    mask = mx.broadcast_to(mx.array(mask_rows)[None, :, None], (1, L, D))
    source = mx.random.normal((n, D)).astype(mx.float32)
    out = masked_scatter(embeds, mask, source)
    mx.eval(out)
    fv.save_golden("p0_masked_scatter", {
        "embeds": np.array(embeds), "mask_rows": mask_rows,
        "source": np.array(source), "out": np.array(out)},
        {"what": "mlx_vlm.models.gemma4.gemma4.masked_scatter", "seed": 0})


def exif_png():
    img = fv.tiny_image(20, 12, seed=3)
    exif = Image.Exif()
    exif[0x0112] = 6                      # Orientation: rotate 90 CW
    buf = io.BytesIO()
    img.save(buf, format="PNG", exif=exif.tobytes())
    raw = buf.getvalue()
    url = "data:image/png;base64," + base64.b64encode(raw).decode()
    ref = load_image(url)
    fv.save_golden("p0_load_image", {
        "png": np.frombuffer(raw, dtype=np.uint8),
        "pixels": np.asarray(ref), "size": np.array(ref.size)},
        {"what": "mlx_vlm.utils.load_image on an EXIF-6 PNG data URL",
         "seed": 3})


if __name__ == "__main__":
    scatter()
    exif_png()
    print("ok")
