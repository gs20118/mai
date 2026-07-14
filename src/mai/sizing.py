"""Turn a detection box into a physical size, and a crater into big / medium / small.

Crater size is worth 15 of the 25 crater points, and it is not read off the network --
it is measured. We have a metric homography, so a box in pixels becomes a box in
centimetres, and the three crater sizes are far enough apart in area to separate
cleanly.

THE CORRECTION THAT MATTERS. These objects sit ON the runway, so a nadir view sees the
base AND the top, displaced by `height x distance_from_nadir / altitude`. The bounding
box spans both, so it is systematically LARGER than the object -- and larger by an
amount that grows toward the arena edges. A crater is only 1.6-3cm tall, which sounds
negligible until you notice that medium and big craters differ by just 20mm in width:
an uncorrected medium at the arena edge can inflate past the big threshold. So we
subtract the expected lean before classifying.

The nadir camera's own pose cannot be recovered from its homography (a fronto-parallel
view makes the focal length unidentifiable -- see pose.estimate, which now says so
rather than inventing one). We do not need it: the point directly beneath the camera is
by definition the one that projects to the image centre, and the altitude is simply
focal / (px per cm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arena import Arena
from .homography import Homography

# Geometric means of the three crater areas (101 / 239 / 358 cm^2). Measured areas from
# the labelled data are 100 / 248 / 409, and these two thresholds split the 146 boxes
# 50 / 48 / 48 -- almost exactly the ~24-25 of each size x 2 frames we would expect.
CRATER_AREA_THRESHOLDS_CM2 = (155.0, 293.0)

CRATER_HEIGHT_CM = {"small": 1.6, "medium": 2.3, "big": 3.0}


@dataclass(frozen=True)
class NadirGeometry:
    """Where the camera is looking straight down, and from how high. That is all lean needs."""

    nadir_cm: np.ndarray  # arena (x, y) directly beneath the camera
    altitude_cm: float

    @classmethod
    def from_homography(
        cls, solved: Homography, image_size: tuple[int, int], focal_px: float
    ) -> "NadirGeometry":
        centre = np.array([[image_size[0] / 2.0, image_size[1] / 2.0]])
        nadir = solved.to_world(centre)[0]
        px_per_cm = solved.source_px_per_cm(nadir)
        if px_per_cm <= 0:
            raise ValueError("degenerate homography: no scale at the image centre")
        return cls(nadir_cm=nadir, altitude_cm=float(focal_px / px_per_cm))

    def lean_cm(self, point_cm, height_cm: float) -> np.ndarray:
        """Base -> top offset, in arena cm, for an object of this height at this point.

        Straight similar triangles: the top of the object is `height` closer to the
        camera, so it projects `height/altitude` further out along the ray from the
        nadir. Zero at the nadir, largest at the arena corners.
        """
        offset = np.asarray(point_cm, dtype=np.float64) - self.nadir_cm
        return offset * (height_cm / self.altitude_cm)


def footprint_cm(
    box_w_cm: float,
    box_h_cm: float,
    point_cm,
    geometry: NadirGeometry,
    height_cm: float,
) -> tuple[float, float]:
    """The object's true ground footprint, with the lean taken back out of the box."""
    lean = np.abs(geometry.lean_cm(point_cm, height_cm))
    return (
        max(box_w_cm - float(lean[0]), 0.1),
        max(box_h_cm - float(lean[1]), 0.1),
    )


def crater_size(
    box_w_cm: float,
    box_h_cm: float,
    point_cm,
    geometry: NadirGeometry | None = None,
) -> str:
    """big / medium / small, from the lean-corrected footprint area.

    The correction needs the crater's height, but the height depends on its size, which
    is what we are trying to find. So iterate: classify uncorrected, correct using that
    class's height, reclassify. It converges immediately -- the correction is at most
    1.5cm on a 10-20cm object, so it only ever moves a crater that was already sitting
    on a threshold, which is precisely the case we are trying to get right.
    """
    size = _classify(box_w_cm * box_h_cm)
    if geometry is None:
        return size

    for _ in range(3):
        width, depth = footprint_cm(
            box_w_cm, box_h_cm, point_cm, geometry, CRATER_HEIGHT_CM[size]
        )
        refined = _classify(width * depth)
        if refined == size:
            return refined
        size = refined
    return size


def _classify(area_cm2: float) -> str:
    small, medium = CRATER_AREA_THRESHOLDS_CM2
    if area_cm2 < small:
        return "small"
    if area_cm2 < medium:
        return "medium"
    return "big"


def box_to_arena(
    box_xyxy: tuple[float, float, float, float],
    zone_id: str,
    arena: Arena,
    px_per_cm: float,
    pad_cm: float = 5.0,
) -> tuple[np.ndarray, float, float]:
    """A detection box in ZONE-CROP pixels -> (base point in arena cm, width cm, depth cm).

    The crop was rectified from the zone's padded rectangle, so crop pixels map to arena
    centimetres by a pure scale and offset -- no homography needed here.

    The returned point is the box's BOTTOM-CENTRE, not its centroid: a tall object leans
    away from the nadir, so its centroid sits off its footprint and can land in the
    neighbouring zone. This mirrors homography.base_point, for the same reason.
    """
    zone = arena.zone(zone_id)
    x1, y1, x2, y2 = box_xyxy

    origin = np.array([zone.x - pad_cm, zone.y - pad_cm], dtype=np.float64)
    base_px = np.array([(x1 + x2) / 2.0, y2], dtype=np.float64)
    base_cm = origin + base_px / px_per_cm

    return base_cm, (x2 - x1) / px_per_cm, (y2 - y1) / px_per_cm
