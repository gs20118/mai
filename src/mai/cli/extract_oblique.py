"""Register and crop the 45-degree oblique view, where markers are occluded.

    python -m mai.cli.extract_oblique --source data/raw/45_degree_side.mp4

TWO PROBLEMS, TWO ANSWERS.

1. A BUILDING BURIES A MARKER. Only 2-3 of the 4 corner markers are ever visible, and
   the one behind the radar tower is never seen. Markers alone then place the far
   side of the arena wrong by 4-26cm -- up to half a runway zone -- while reporting a
   healthy ~1cm reprojection error, because they only check themselves where they
   already have evidence. The arena's printed border ticks put evidence where the
   markers cannot: each fixes one coordinate, which is a linear constraint that drops
   straight into the DLT. With them, every frame registers to under 0.55cm.

2. A HOMOGRAPHY ONLY RECTIFIES THE GROUND. At 44 degrees elevation the top of a 50cm
   building lands a full 50cm from its base -- an entire runway zone away -- so the
   rectified view smears every building into a streak. That is fatal for the very
   thing an oblique view is FOR: seeing building facades, which is where damage
   actually shows. A nadir shot sees roofs.

So this emits both:

    topview/     the requested rectified projection, matching the nadir top view.
                 Trustworthy for FLAT things -- craters (3cm lean) and cluster
                 munitions (2cm). Buildings are smeared; that is not a bug.

    zones_flat/  ground-plane zone crops from that rectification.

    zones_oblique/ crops in the ORIGINAL frame, of each zone's footprint extruded to
                 the height of what stands on it. Facades intact, unsmeared. This is
                 what the 45-degree view is worth labelling for: facility damage
                 state, and telling a 115mm missile from a 93mm shell by its side.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from mai import aruco, frames, homography, pose as posemod, ticks, topview, viz, zones
from mai.arena import Arena
from mai.undistort import CameraProfile, Undistorter

# How far above the board each class of zone reaches. Facility zones hold buildings
# (TASK.md 8.2: 30-50cm tall); airfield zones hold UXO (up to 11.5cm).
ZONE_HEIGHT_CM = {"facility": 55.0, "runway": 14.0, "taxiway_a": 14.0, "taxiway_b": 14.0}


def register(image, arena, undistorter, detector):
    """Homography from markers PLUS printed ticks, and the camera pose behind it."""
    undistorted = undistorter(image)
    detections = aruco.detect(undistorted, detector, keep_ids=set(arena.markers))
    if len(detections) < 2:
        raise homography.HomographyError(
            f"only {len(detections)} marker(s); need >=2 even with ticks"
        )

    image_points, world_points = aruco.correspondences(detections, arena)
    seed, _ = cv2.findHomography(
        image_points.reshape(-1, 1, 2), world_points.reshape(-1, 1, 2)
    )
    if seed is None:
        raise homography.HomographyError("seed homography failed")

    found = ticks.find(undistorted, seed, arena)
    constraints = ticks.constraints(found)
    marker_ids = [d.id for d in detections]

    solved = homography.solve_constrained(
        image_points, world_points, constraints, marker_ids
    )
    residuals = homography.axis_residuals(solved, constraints)
    return undistorted, detections, found, solved, residuals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/raw/45_degree_side.mp4")
    parser.add_argument("--output", default="outputs/dataset_oblique")
    parser.add_argument("--arena", default=None)
    parser.add_argument("--profile", default="video_4k")
    parser.add_argument("--per-video", type=int, default=3)
    parser.add_argument("--sample-sec", type=float, default=0.3)
    parser.add_argument(
        "--max-tick-error-cm",
        type=float,
        default=1.5,
        help="Reject a frame whose printed-tick residual exceeds this. This is the "
        "honest gate: marker reprojection error only checks the fit where the "
        "markers are, which on this view is exactly where it is already right.",
    )
    parser.add_argument("--pad-cm", type=float, default=5.0)
    args = parser.parse_args()

    arena = Arena.from_yaml(args.arena) if args.arena else Arena.from_yaml()
    undistorter = Undistorter(CameraProfile.load(args.profile))
    detector = aruco.build_detector(arena.dictionary)

    output = Path(args.output)
    for sub in ("topview", "zones_flat", "zones_oblique", "qa"):
        (output / sub).mkdir(parents=True, exist_ok=True)

    source = Path(args.source)
    candidates = []
    for frame in frames.iter_frames(source, sample_sec=args.sample_sec):
        try:
            undistorted, detections, found, solved, residuals = register(
                frame.image, arena, undistorter, detector
            )
        except homography.HomographyError as error:
            print(f"  f{frame.index:05d}: rejected ({error})")
            continue

        worst = float(np.abs(residuals).max()) if len(residuals) else np.inf
        if worst > args.max_tick_error_cm:
            print(f"  f{frame.index:05d}: rejected (tick error {worst:.2f}cm)")
            continue
        candidates.append(
            (frame, undistorted, detections, found, solved, worst,
             frames.sharpness(undistorted))
        )

    if not candidates:
        print("No frame registered.")
        return 1

    candidates.sort(key=lambda c: -c[6])  # sharpest first
    keep = candidates[: args.per_video]
    print(f"\n{len(candidates)} frames registered, keeping {len(keep)}\n")

    manifest = []
    for frame, undistorted, detections, found, solved, worst, sharp in keep:
        stem = f"{source.stem}_f{frame.index:05d}"
        size = (undistorted.shape[1], undistorted.shape[0])
        pose = posemod.estimate(solved.inverse, size)

        print(
            f"{stem}: markers {sorted(d.id for d in detections)} + {len(found)} ticks | "
            f"tick err {worst:.2f}cm | elevation {pose.elevation_deg:.0f}deg | "
            f"a 50cm building leans {pose.lean_cm(50):.0f}cm on the ground"
        )

        # --- 1. The requested rectified projection, matching the nadir top view. ---
        px_per_cm = topview.native_px_per_cm(arena, solved)
        warped = topview.warp(undistorted, arena, solved, px_per_cm)
        cv2.imwrite(str(output / "topview" / f"{stem}.jpg"), warped)
        cv2.imwrite(
            str(output / "qa" / f"{stem}_grid.jpg"),
            viz.draw_zone_grid_on_source(undistorted, arena, solved),
        )

        # --- 2. Ground-plane zone crops (flat objects only). ---
        flat_dir = output / "zones_flat"
        for zone in arena.zones:
            crop, _ = zones.crop_zone(undistorted, zone, solved, pad_cm=args.pad_cm)
            cv2.imwrite(str(flat_dir / f"{stem}_{zone.id}.jpg"), crop)

        # --- 3. Oblique crops: the footprint extruded to what stands on it. ---
        oblique_dir = output / "zones_oblique"
        overlay = undistorted.copy()
        for zone in arena.zones:
            height = ZONE_HEIGHT_CM[zone.band]
            quad_cm = zone.polygon(args.pad_cm)
            quad_px = solved.to_image(quad_cm)
            left, top, right, bottom = posemod.zone_box_image_bounds(
                pose, quad_px, quad_cm, height, size, pad_px=25
            )
            if right - left < 20 or bottom - top < 20:
                continue
            cv2.imwrite(
                str(oblique_dir / f"{stem}_{zone.id}.jpg"),
                undistorted[top:bottom, left:right],
            )
            cv2.rectangle(overlay, (left, top), (right, bottom), (0, 220, 255), 2)
            cv2.putText(
                overlay, zone.id, (left + 6, top + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA,
            )
            cv2.putText(
                overlay, zone.id, (left + 6, top + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 1, cv2.LINE_AA,
            )
        cv2.imwrite(str(output / "qa" / f"{stem}_oblique_boxes.jpg"), overlay)

        manifest.append(
            {
                "video": source.name,
                "frame_index": frame.index,
                "stem": stem,
                "marker_ids": sorted(d.id for d in detections),
                "tick_count": len(found),
                "max_tick_error_cm": round(worst, 3),
                "marker_reprojection_rms_cm": round(solved.rms_cm, 3),
                "homography_image_to_arena_cm": solved.matrix.tolist(),
                "camera": {
                    "focal_px": round(pose.focal_px, 1),
                    "elevation_deg": round(pose.elevation_deg, 2),
                    "center_cm": [round(v, 1) for v in pose.center_cm],
                },
                "ground_lean_cm": {
                    "building_50cm": round(pose.lean_cm(50), 1),
                    "missile_11_5cm": round(pose.lean_cm(11.5), 1),
                    "crater_3cm": round(pose.lean_cm(3), 1),
                },
                "topview_px_per_cm": round(px_per_cm, 3),
            }
        )

    with (output / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print(f"\n-> {output}")
    print("   topview/        rectified projection (flat objects only)")
    print("   zones_flat/     ground-plane zone crops")
    print("   zones_oblique/  crops with height -- facades intact, LABEL THESE")
    print("   qa/             check *_grid.jpg and *_oblique_boxes.jpg first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
