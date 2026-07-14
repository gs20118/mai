"""Measure the ArUco markers' true arena positions, using the arena's printed ticks.

    python -m mai.cli.calibrate_arena --source data/raw --write

The arena prints its own ground truth: yellow tick marks on the outer border at
every sector boundary, plus the sector names (FA / TW-A / RW / TW-B / FA). Those
boundaries are at exactly known arena coordinates -- x = 160, 340 for the facility
columns, y = 80, 160, 240, 320 for the five bands -- so they can calibrate the
marker map with no tape measure and no guessing.

WHY THIS IS NOT OPTIONAL. The obvious assumption -- that the four markers sit ON the
arena corners -- is wrong, and wrong in a way that looks fine. Under it, the printed
bands come out at 72.9 / 157.8 / 242.5 / 327.5cm instead of 80 / 160 / 240 / 320,
and they stop being uniform. The markers are actually inset about 11cm, i.e. half
their own width: each is placed with its OUTER corner flush to the arena corner.
Left uncorrected, every zone assignment would be skewed by up to ~11cm -- a fifth of
a 50cm runway zone, and enough to push a crater into the neighbouring one.

Method: each tick pair spans the arena (top+bottom ticks share an x; left+right
ticks share a y), so each pair defines a line of known arena coordinate. Intersect
the 2 vertical lines with the 4 horizontal ones to get 8 points whose arena
coordinates are exactly known, solve the image -> arena homography from those, then
simply read the marker centres off it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from mai import aruco, frames
from mai.arena import DEFAULT_ARENA_CONFIG, Arena

# Where the printed sector boundaries are, per TASK.md 7.4.
BOUNDARY_X_CM = [160.0, 340.0]  # FA-01|FA-02|FA-03 column splits
BOUNDARY_Y_CM = [80.0, 160.0, 240.0, 320.0]  # FA | TW-A | RW | TW-B | FA

# The ticks are a saturated yellow-green, unlike anything else on the board.
TICK_HSV_LO = (25, 120, 140)
TICK_HSV_HI = (45, 255, 255)
TICK_MIN_AREA_PX = 60


def find_ticks(image, seed_homography, px_per_cm=12.0, margin_cm=30.0):
    """Locate the border ticks and return them in ORIGINAL image pixels.

    A rough homography is enough to find them: we only use it to warp the border
    into view and to sort ticks by edge. Their positions are then mapped back to the
    source image, so the final fit never inherits the seed's error.
    """
    to_canvas = (
        np.array(
            [[px_per_cm, 0, px_per_cm * margin_cm], [0, px_per_cm, px_per_cm * margin_cm], [0, 0, 1]]
        )
        @ seed_homography
    )
    size = (
        int((500 + 2 * margin_cm) * px_per_cm),
        int((400 + 2 * margin_cm) * px_per_cm),
    )
    canvas = cv2.warpPerspective(image, to_canvas, size)

    mask = cv2.inRange(cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV), TICK_HSV_LO, TICK_HSV_HI)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    back = np.linalg.inv(to_canvas)
    edges: dict[str, list] = {"top": [], "bottom": [], "left": [], "right": []}
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < TICK_MIN_AREA_PX:
            continue
        canvas_x, canvas_y = centroids[index]
        arena_x = canvas_x / px_per_cm - margin_cm
        arena_y = canvas_y / px_per_cm - margin_cm

        source = cv2.perspectiveTransform(
            np.float32([[[canvas_x, canvas_y]]]), back
        ).ravel()

        if -margin_cm <= arena_y <= -2:
            edges["top"].append((arena_x, source))
        elif 402 <= arena_y <= 400 + margin_cm:
            edges["bottom"].append((arena_x, source))
        elif -margin_cm <= arena_x <= -2:
            edges["left"].append((arena_y, source))
        elif 502 <= arena_x <= 500 + margin_cm:
            edges["right"].append((arena_y, source))

    for key in edges:
        edges[key].sort(key=lambda item: item[0])
    return edges


def _line_through(point_a, point_b):
    """Homogeneous line through two image points."""
    return np.cross([*point_a, 1.0], [*point_b, 1.0])


def _intersect(line_a, line_b):
    point = np.cross(line_a, line_b)
    if abs(point[2]) < 1e-9:
        return None
    return point[:2] / point[2]


def solve_from_ticks(edges) -> tuple[np.ndarray, float]:
    """Homography image px -> arena cm, from the printed boundaries alone."""
    if len(edges["top"]) != 2 or len(edges["bottom"]) != 2:
        raise RuntimeError(
            f"expected 2 ticks on the top and bottom edges, found "
            f"{len(edges['top'])} and {len(edges['bottom'])}"
        )
    if len(edges["left"]) != 4 or len(edges["right"]) != 4:
        raise RuntimeError(
            f"expected 4 ticks on the left and right edges, found "
            f"{len(edges['left'])} and {len(edges['right'])}"
        )

    # A top tick and the bottom tick below it both sit on the same arena column, so
    # the image line through them IS that column. Same for left/right and rows.
    verticals = [
        (BOUNDARY_X_CM[i], _line_through(edges["top"][i][1], edges["bottom"][i][1]))
        for i in range(2)
    ]
    horizontals = [
        (BOUNDARY_Y_CM[i], _line_through(edges["left"][i][1], edges["right"][i][1]))
        for i in range(4)
    ]

    image_points, arena_points = [], []
    for x_cm, vertical in verticals:
        for y_cm, horizontal in horizontals:
            crossing = _intersect(vertical, horizontal)
            if crossing is None:
                continue
            image_points.append(crossing)
            arena_points.append([x_cm, y_cm])

    image_points = np.float32(image_points)
    arena_points = np.float32(arena_points)
    homography, _ = cv2.findHomography(image_points, arena_points, 0)

    projected = cv2.perspectiveTransform(
        image_points.reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((projected - arena_points) ** 2, axis=1))))
    return homography, rms


def measure(image, detector) -> dict | None:
    detections = {d.id: d for d in aruco.detect(image, detector)}
    if len(detections) != 4:
        return None

    # Seed: assume the markers are on the arena corners. Only good enough to find
    # the ticks; the real fit below throws it away.
    corner_ids = _corner_ids(detections)
    seed, _ = cv2.findHomography(
        np.float32([detections[i].center for i in corner_ids]),
        np.float32([[0, 0], [500, 0], [500, 400], [0, 400]]),
    )

    edges = find_ticks(image, seed)
    homography, rms = solve_from_ticks(edges)

    result = {"rms_cm": rms, "markers": {}}
    for marker_id, detection in detections.items():
        corners = cv2.perspectiveTransform(
            detection.corners.reshape(-1, 1, 2).astype(np.float32), homography
        ).reshape(4, 2)
        sides = [
            float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])) for i in range(4)
        ]
        result["markers"][marker_id] = {
            "center": corners.mean(axis=0),
            "size_cm": float(np.mean(sides)),
        }
    return result


def _corner_ids(detections) -> list[int]:
    """IDs ordered TL, TR, BR, BL by their image position."""
    ids = list(detections)
    centers = np.array([detections[i].center for i in ids])
    middle = centers.mean(axis=0)
    ordered = {}
    for marker_id, center in zip(ids, centers):
        key = (center[1] > middle[1], center[0] > middle[0])  # (is_bottom, is_right)
        ordered[key] = marker_id
    return [
        ordered[(False, False)],
        ordered[(False, True)],
        ordered[(True, True)],
        ordered[(True, False)],
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Video, image, or directory.")
    parser.add_argument("--sample-sec", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--dictionary", default=None, help="Default: whatever arena.yaml says.")
    parser.add_argument("--arena", default=str(DEFAULT_ARENA_CONFIG))
    parser.add_argument("--write", action="store_true", help="Update arena.yaml in place.")
    args = parser.parse_args()

    arena = Arena.from_yaml(args.arena)
    detector = aruco.build_detector(args.dictionary or arena.dictionary)

    sources = (
        sorted(Path(args.source).glob("*.mp4"))
        if Path(args.source).is_dir()
        else [Path(args.source)]
    )

    samples: dict[int, list] = {}
    sizes: list[float] = []
    rms_values: list[float] = []
    used = 0

    for source in sources:
        taken = 0
        for frame in frames.iter_frames(source, sample_sec=args.sample_sec):
            if taken >= args.max_frames:
                break
            try:
                result = measure(frame.image, detector)
            except RuntimeError:
                continue
            if result is None:
                continue
            for marker_id, spec in result["markers"].items():
                samples.setdefault(marker_id, []).append(spec["center"])
                sizes.append(spec["size_cm"])
            rms_values.append(result["rms_cm"])
            taken += 1
            used += 1

    if not samples:
        print("Could not calibrate: no frame gave 4 markers AND all 8 border ticks.")
        return 1

    print(f"calibrated from {used} frames across {len(sources)} source(s)")
    print(f"  tick-fit reprojection RMS: {np.mean(rms_values):.3f} cm\n")

    marker_size = float(np.median(sizes))
    print(f"  marker_size_cm: {marker_size:.1f}\n")
    print("  measured marker centres (arena cm):")

    resolved = {}
    for marker_id in sorted(samples):
        points = np.array(samples[marker_id])
        center = points.mean(axis=0)
        spread = points.std(axis=0)
        resolved[marker_id] = center
        print(
            f"    id {marker_id}: ({center[0]:7.2f}, {center[1]:7.2f})  "
            f"+/- ({spread[0]:.2f}, {spread[1]:.2f})"
        )

    # A marker placed with its outer corner flush to the arena corner has its centre
    # inset by exactly half its width. Reporting this makes the geometry legible, and
    # a large disagreement would mean something is off.
    insets = [
        min(center[0], 500 - center[0], center[1], 400 - center[1])
        for center in resolved.values()
    ]
    print(
        f"\n  inset from arena edge: {np.mean(insets):.2f} cm "
        f"(half the marker width is {marker_size / 2:.2f} cm)"
    )

    if not args.write:
        print("\n(dry run -- pass --write to update arena.yaml)")
        return 0

    path = Path(args.arena)
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    raw["aruco"]["dictionary"] = args.dictionary or arena.dictionary
    raw["aruco"]["marker_size_cm"] = round(marker_size, 1)
    raw["aruco"]["markers"] = {
        int(marker_id): {
            "center": [round(float(center[0]), 2), round(float(center[1]), 2)],
            "rotation_deg": 0,
        }
        for marker_id, center in sorted(resolved.items())
    }
    header = "".join(
        line
        for line in text.splitlines(keepends=True)[
            : next(
                (
                    index
                    for index, line in enumerate(text.splitlines())
                    if line.strip() and not line.lstrip().startswith("#")
                ),
                0,
            )
        ]
    )
    path.write_text(
        header + yaml.safe_dump(raw, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"\nwrote measured marker map to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
