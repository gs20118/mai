"""Overlay rendering. Everything here exists to be looked at by a human.

The single most valuable check in this project is `draw_zone_grid_on_source`:
projecting the arena grid back onto the raw drone frame. If the drawn grid lands
exactly on the real runway and taxiway edges, then the marker map, the distortion
profile, the homography, and the band layout are all simultaneously correct. If
it doesn't, one of them is wrong and it is immediately obvious which way.
"""

from __future__ import annotations

import cv2
import numpy as np

from .arena import Arena
from .aruco import Detection
from .homography import Homography

_BAND_COLORS = {
    "facility": (255, 170, 60),
    "taxiway_a": (90, 220, 90),
    "taxiway_b": (90, 220, 90),
    "runway": (70, 120, 255),
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _scaled(image: np.ndarray, base: float = 1600.0) -> tuple[float, int]:
    """Line thickness and font scale that look the same on a 1080p and a 4K frame."""
    longest = max(image.shape[:2])
    factor = max(longest / base, 0.5)
    return 0.55 * factor, max(int(round(2 * factor)), 1)


def draw_markers(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    overlay = image.copy()
    font_scale, thickness = _scaled(image)

    for detection in detections:
        corners = detection.corners.astype(np.int32)
        cv2.polylines(overlay, [corners], True, (0, 255, 255), thickness, cv2.LINE_AA)
        # Mark corner 0 so a rotated marker map shows up as a visibly wrong corner.
        cv2.circle(overlay, tuple(corners[0]), thickness * 3, (0, 0, 255), -1)
        center = detection.center.astype(int)
        cv2.putText(
            overlay,
            f"id={detection.id}",
            (center[0] + 10, center[1] - 10),
            _FONT,
            font_scale * 1.4,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return overlay


def draw_zone_grid_on_source(
    image: np.ndarray, arena: Arena, homography: Homography
) -> np.ndarray:
    """Project the arena zone grid back onto a (undistorted) source frame."""
    overlay = image.copy()
    font_scale, thickness = _scaled(image)

    for zone in arena.zones:
        quad = homography.to_image(zone.polygon()).astype(np.int32)
        color = _BAND_COLORS.get(zone.band, (255, 255, 255))
        cv2.polylines(overlay, [quad], True, color, thickness, cv2.LINE_AA)
        center = homography.to_image(np.array([zone.center]))[0].astype(int)
        _label(overlay, zone.id, center, color, font_scale, thickness)

    return overlay


def draw_zone_grid_on_topview(
    topview: np.ndarray, arena: Arena, px_per_cm: float
) -> np.ndarray:
    overlay = topview.copy()
    font_scale, thickness = _scaled(topview)

    for zone in arena.zones:
        quad = (zone.polygon() * px_per_cm).astype(np.int32)
        color = _BAND_COLORS.get(zone.band, (255, 255, 255))
        cv2.polylines(overlay, [quad], True, color, thickness, cv2.LINE_AA)
        center = (np.array(zone.center) * px_per_cm).astype(int)
        _label(overlay, zone.id, center, color, font_scale, thickness)

    return overlay


def _label(
    image: np.ndarray,
    text: str,
    center: np.ndarray,
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    (text_w, text_h), _ = cv2.getTextSize(text, _FONT, font_scale, thickness)
    origin = (int(center[0] - text_w / 2), int(center[1] + text_h / 2))
    # Black halo so labels stay legible over both dark asphalt and bright markings.
    cv2.putText(image, text, origin, _FONT, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, _FONT, font_scale, color, thickness, cv2.LINE_AA)


def contact_sheet(
    crops: dict[str, np.ndarray],
    order: list[str],
    columns: int = 5,
    cell_px: int = 420,
    captions: dict[str, str] | None = None,
) -> np.ndarray:
    """Tile every zone crop into one image, letterboxed, labelled, in a fixed order.

    This is the artifact that decides whether we train models: it is where you see
    whether a 28mm cluster munition is actually visible or just a smudge.
    """
    columns = max(columns, 1)
    rows = (len(order) + columns - 1) // columns
    header = 34
    sheet = np.full(
        (rows * (cell_px + header), columns * cell_px, 3), 30, dtype=np.uint8
    )

    for position, zone_id in enumerate(order):
        row, column = divmod(position, columns)
        y0 = row * (cell_px + header)
        x0 = column * cell_px

        crop = crops.get(zone_id)
        if crop is not None and crop.size:
            height, width = crop.shape[:2]
            scale = min(cell_px / width, cell_px / height)
            resized = cv2.resize(
                crop, (max(int(width * scale), 1), max(int(height * scale), 1))
            )
            pad_y = y0 + header + (cell_px - resized.shape[0]) // 2
            pad_x = x0 + (cell_px - resized.shape[1]) // 2
            sheet[pad_y : pad_y + resized.shape[0], pad_x : pad_x + resized.shape[1]] = resized

        caption = zone_id if captions is None else f"{zone_id}  {captions.get(zone_id, '')}"
        cv2.putText(
            sheet, caption, (x0 + 8, y0 + 24), _FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.rectangle(
            sheet, (x0, y0), (x0 + cell_px - 1, y0 + header + cell_px - 1), (80, 80, 80), 1
        )

    return sheet
