"""Deterministic orientation check for flat objects (replay-safe: pure functions of PNG bytes).

Round-0 finding: a flat object photographed from above (pizza, board, compass, flatbread) is
usually modelled lying in the XZ plane, so the organizer judge's front views (phi 0 / 15 / -30)
see it edge-on as a sliver while the photo shows its face. 16 of our 128 round-0 front views
were slivers versus 2 for the best competitor. The fix compares the candidate's front-view
silhouette with the photo's silhouette and, when the photo is face-on but the render is
edge-on, tilts the model toward +Z in the normalisation wrapper (tilt chosen so the rendered
silhouette aspect matches the photo's).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

PHOTO_FACE_ON_MIN_ASPECT = 0.45  # photo silhouette height / width above this = the face is visible
RENDER_EDGE_ON_MAX_ASPECT = 0.30  # render silhouette below this = seen edge-on
TILTS = (15.0, 30.0, 45.0, 60.0, 75.0, 90.0)  # degrees about X; +90 maps +Y (face up) to +Z (facing the camera)


def _mask(png: bytes, *, white_background: bool) -> np.ndarray:
    image = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(np.int16)
    if white_background:
        return image.min(axis=2) < 235
    border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
    background = np.median(border, axis=0)
    return np.abs(image - background).max(axis=2) > 24


def silhouette_aspect(png: bytes, *, white_background: bool) -> float:
    """Height / width of the foreground bounding box (0.0 for an empty image).

    Rows and columns count only when they hold more than 0.5 % foreground pixels, which
    ignores specks and soft shadows.
    """
    mask = _mask(png, white_background=white_background)
    rows = np.where(mask.sum(axis=1) > 0.005 * mask.shape[1])[0]
    cols = np.where(mask.sum(axis=0) > 0.005 * mask.shape[0])[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0.0
    height = int(rows[-1] - rows[0] + 1)
    width = int(cols[-1] - cols[0] + 1)
    return height / width


def needs_upright(photo: bytes, front: bytes) -> bool:
    """True when the photo shows the object's face but the front render shows an edge.

    An undecodable image never triggers a tilt (deterministic skip, never an error).
    """
    try:
        return (
            silhouette_aspect(photo, white_background=False) >= PHOTO_FACE_ON_MIN_ASPECT
            and silhouette_aspect(front, white_background=True) < RENDER_EDGE_ON_MAX_ASPECT
        )
    except (OSError, ValueError):
        return False


def best_tilt(photo: bytes, fronts: dict[float, bytes]) -> float:
    """The tilt whose front render matches the photo's silhouette aspect best (ties: smaller tilt)."""
    target = silhouette_aspect(photo, white_background=False)
    scored = sorted(
        (abs(silhouette_aspect(png, white_background=True) - target), tilt) for tilt, png in fronts.items()
    )
    return scored[0][1]
