"""Per-zone rectified crops, and the resolution report that decides the project.

Crops are warped DIRECTLY from the undistorted source frame at each zone's own
native resolution, rather than being cut out of a downscaled top-view canvas.
That matters more than it sounds: the arena is 500cm across and a 4K frame is
3840px, so the source carries roughly 7.7 px/cm. Warping the whole arena to a
2 px/cm canvas first (what legacy/aruco_homography.py did) throws away ~4x of
linear resolution, and a 28mm cluster munition collapses from ~21px to ~5px.
Cropping straight from the source keeps every pixel the sensor actually captured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from .arena import Arena, Zone
from .homography import Homography


@dataclass
class ZoneCrop:
    zone_id: str
    band: str
    px_per_cm: float  # effective SOURCE resolution sampled for this zone
    mm_per_px: float
    size_px: tuple[int, int]
    visible: bool  # does the zone actually fall inside the frame?
    coverage: float  # fraction of the zone's corners inside the frame


def crop_zone(
    image: np.ndarray,
    arena_zone: Zone,
    homography: Homography,
    pad_cm: float = 5.0,
    px_per_cm: float | None = None,
    max_px_per_cm: float = 20.0,
) -> tuple[np.ndarray, ZoneCrop]:
    """Rectify one zone out of an undistorted source frame.

    `px_per_cm` defaults to the zone's own native source resolution, so the warp
    neither invents detail nor discards it. `pad_cm` keeps objects that straddle a
    zone boundary from being clipped — craters are up to 20cm across and a zone is
    only 50cm wide, so this is a real risk.
    """
    height, width = image.shape[:2]

    if px_per_cm is None:
        px_per_cm = homography.source_px_per_cm(np.array(arena_zone.center))
    px_per_cm = float(np.clip(px_per_cm, 1.0, max_px_per_cm))

    padded = arena_zone.polygon(pad_cm)
    origin = padded[0]
    span_cm = padded[2] - padded[0]

    # world cm -> zone-local px, then compose with image px -> world cm.
    to_local = np.array(
        [
            [px_per_cm, 0.0, -px_per_cm * origin[0]],
            [0.0, px_per_cm, -px_per_cm * origin[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    matrix = to_local @ homography.matrix

    out_w = max(int(round(span_cm[0] * px_per_cm)), 1)
    out_h = max(int(round(span_cm[1] * px_per_cm)), 1)
    crop = cv2.warpPerspective(image, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR)

    # A zone the drone never framed produces a black rectangle, which would look
    # like "nothing here" rather than "not seen". Track it explicitly.
    source_quad = homography.to_image(arena_zone.polygon())
    inside = [bool(0 <= x < width and 0 <= y < height) for x, y in source_quad]
    coverage = sum(inside) / len(inside)

    return crop, ZoneCrop(
        zone_id=arena_zone.id,
        band=arena_zone.band,
        px_per_cm=round(px_per_cm, 3),
        mm_per_px=round(10.0 / px_per_cm, 3),
        size_px=(out_w, out_h),
        visible=coverage > 0.0,
        coverage=round(coverage, 2),
    )


def crop_all(
    image: np.ndarray,
    arena: Arena,
    homography: Homography,
    output_dir: Path,
    pad_cm: float = 5.0,
) -> tuple[dict[str, np.ndarray], list[ZoneCrop]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    crops: dict[str, np.ndarray] = {}
    records: list[ZoneCrop] = []

    for zone in arena.zones:
        crop, record = crop_zone(image, zone, homography, pad_cm=pad_cm)
        crops[zone.id] = crop
        records.append(record)
        cv2.imwrite(str(output_dir / f"{zone.id}.jpg"), crop)

    return crops, records


def gsd_report(
    arena: Arena, records: list[ZoneCrop], output_path: Path | None = None
) -> dict:
    """Per zone: how many pixels each real target would occupy.

    This turns "can we detect a cluster munition?" from a guess into a number,
    before we spend a day training anything. Targets under roughly 10px are not
    reliably detectable by a YOLO-class model; 20px+ is comfortable.
    """
    per_zone = []
    for record in records:
        targets = {
            name: round(spec["w_mm"] / record.mm_per_px, 1)
            for name, spec in arena.targets.items()
        }
        per_zone.append({**asdict(record), "target_px": targets})

    visible = [record for record in records if record.visible]
    mm_per_px_values = [record.mm_per_px for record in visible]

    summary = {}
    if mm_per_px_values:
        worst = max(mm_per_px_values)
        best = min(mm_per_px_values)
        summary = {
            "zones_visible": len(visible),
            "zones_total": len(records),
            "mm_per_px_best": round(best, 3),
            "mm_per_px_worst": round(worst, 3),
            # The smallest target in the worst-resolved zone is the binding
            # constraint on the whole UXO mission.
            "smallest_target_px_worst_zone": {
                name: round(spec["w_mm"] / worst, 1)
                for name, spec in arena.targets.items()
            },
        }

    report = {"summary": summary, "zones": per_zone}
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
    return report
