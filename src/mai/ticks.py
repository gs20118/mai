"""The arena's printed boundary ticks, as fiducials in their own right.

The board prints its own ground truth: small yellow ticks on the outer border at
every sector boundary, at exactly known arena coordinates (x = 160, 340 for the
facility columns; y = 80, 160, 240, 320 for the five bands).

A tick constrains ONE coordinate, not two. A tick on the left border marking the
y=80 band boundary tells us that whatever image point it sits at must map to
Y = 80; its X is merely "somewhere on the border" and is not known. That is a
one-dimensional constraint, and homography.solve_constrained consumes it directly.

This is what rescues the oblique view. With one corner marker buried behind a
building, a three-marker fit is accurate near those three markers and drifts badly
in the unconstrained far corner -- by ~13cm on this footage, a quarter of a runway
zone -- while reporting a reprojection error of only ~1cm, because it is only
checking itself where it already has evidence. The ticks put evidence where the
markers cannot.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .arena import Arena
from .homography import AxisConstraint

# Printed sector boundaries, per TASK.md 7.4 and confirmed against the board.
BOUNDARY_X_CM = [160.0, 340.0]  # FA-01 | FA-02 | FA-03 column splits
BOUNDARY_Y_CM = [80.0, 160.0, 240.0, 320.0]  # FA | TW-A | RW | TW-B | FA

# The ticks are a saturated yellow-green. Thresholds are deliberately loose: in an
# oblique view the far ticks are dim and washed out, and the tight range that works
# on nadir footage finds nothing at all there.
TICK_HSV_LO = (20, 55, 90)
TICK_HSV_HI = (50, 255, 255)
TICK_MIN_AREA_PX = 40

# How far outside/inside the arena edge a tick may sit and still be attributed to it.
# The ticks straddle the border, and the seed homography that sorts them is itself
# imperfect, so this has to be generous.
EDGE_BAND_CM = 16.0


@dataclass(frozen=True)
class Tick:
    image_xy: np.ndarray  # where it is, in source-image pixels
    edge: str  # top | bottom | left | right
    axis: int  # 0 if it fixes X, 1 if it fixes Y
    value_cm: float  # the coordinate it fixes
    area_px: int


def find(
    image: np.ndarray,
    seed: np.ndarray,
    arena: Arena,
    px_per_cm: float = 8.0,
    margin_cm: float = 30.0,
) -> list[Tick]:
    """Locate the border ticks and label each with the coordinate it fixes.

    `seed` is a rough image -> arena cm homography, used only to bring the border
    into view and to decide which edge each tick belongs to. Tick positions are
    mapped back to the SOURCE image, so the final fit never inherits the seed's
    error -- which matters, because the seed is exactly the fit we are trying to
    improve on.
    """
    to_canvas = (
        np.array(
            [
                [px_per_cm, 0, px_per_cm * margin_cm],
                [0, px_per_cm, px_per_cm * margin_cm],
                [0, 0, 1],
            ]
        )
        @ seed
    )
    size = (
        int((arena.width_cm + 2 * margin_cm) * px_per_cm),
        int((arena.height_cm + 2 * margin_cm) * px_per_cm),
    )
    canvas = cv2.warpPerspective(image, to_canvas, size)
    mask = cv2.inRange(cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV), TICK_HSV_LO, TICK_HSV_HI)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    back = np.linalg.inv(to_canvas)
    found: dict[tuple[str, float], Tick] = {}

    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < TICK_MIN_AREA_PX:
            continue
        canvas_x, canvas_y = centroids[index]
        arena_x = canvas_x / px_per_cm - margin_cm
        arena_y = canvas_y / px_per_cm - margin_cm

        edge = _which_edge(arena_x, arena_y, arena)
        if edge is None:
            continue

        # A top/bottom tick marks a COLUMN split, so it fixes X; the free coordinate
        # is the one running across the border. Left/right ticks fix Y.
        if edge in ("top", "bottom"):
            axis, along, candidates = 0, arena_x, BOUNDARY_X_CM
        else:
            axis, along, candidates = 1, arena_y, BOUNDARY_Y_CM

        value = min(candidates, key=lambda c: abs(c - along))
        if abs(value - along) > 20.0:
            continue  # too far from any known boundary to be one

        source = cv2.perspectiveTransform(
            np.float32([[[canvas_x, canvas_y]]]), back
        ).ravel()
        tick = Tick(
            image_xy=source.astype(np.float64),
            edge=edge,
            axis=axis,
            value_cm=value,
            area_px=area,
        )
        # One tick per (edge, boundary). Keep the biggest blob if glare splits one.
        key = (edge, value)
        if key not in found or area > found[key].area_px:
            found[key] = tick

    return sorted(found.values(), key=lambda t: (t.edge, t.value_cm))


def _which_edge(x_cm: float, y_cm: float, arena: Arena) -> str | None:
    on_top = abs(y_cm) <= EDGE_BAND_CM
    on_bottom = abs(y_cm - arena.height_cm) <= EDGE_BAND_CM
    on_left = abs(x_cm) <= EDGE_BAND_CM
    on_right = abs(x_cm - arena.width_cm) <= EDGE_BAND_CM

    # A corner would satisfy two of these; ignore it, we cannot say which boundary
    # it belongs to.
    if sum([on_top, on_bottom, on_left, on_right]) != 1:
        return None
    if on_top:
        return "top"
    if on_bottom:
        return "bottom"
    if on_left:
        return "left"
    return "right"


def constraints(ticks: list[Tick]) -> list[AxisConstraint]:
    return [
        AxisConstraint(image_xy=tick.image_xy, axis=tick.axis, value_cm=tick.value_cm)
        for tick in ticks
    ]
