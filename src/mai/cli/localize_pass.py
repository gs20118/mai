"""Register markerless high-resolution frames against the ArUco-anchored top view.

    python -m mai.cli.localize_pass --run outputs/topview_... --frames data/frames/strip
    python -m mai.cli.localize_pass --run outputs/demo --synthetic

The strip pass cannot see the corner markers, and that is forced rather than
chosen: resolution comes from covering less arena, and any frame containing all
four markers must span the full arena depth. So it localises against the top view
by feature matching.

Frames that cannot be placed SAFELY are refused, not guessed. A periodic runway can
yield a homography shifted by exactly one zone that looks perfectly healthy, and
that answer scores zero while looking like success. When a frame is refused, fall
back to the marker-registered top view for that region.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from mai import frames, localize, viz, zones
from mai.arena import Arena
from mai.homography import Homography
from mai.localize import Gates, LocalizationError, Reference


def synthetic_strip(arena: Arena, run_dir: Path):
    """A left-half airfield pass, for exercising this without a drone."""
    from mai.cli.build_topview import DEMO_OBJECTS
    from mai.synthetic import (
        DEMO_CANVAS_PX_PER_CM,
        CameraPose,
        capture,
        fit_focal_for_coverage,
        render_topdown,
    )
    from mai.undistort import CameraProfile, Undistorter

    # Same seed and scale as build_topview's synthetic frame, so both are photographs
    # of the SAME physical arena. A different seed would draw a different random
    # surface, leaving the two views with no shared detail to match on -- which is
    # not a property real footage has.
    canvas = render_topdown(
        arena, DEMO_OBJECTS, px_per_cm=DEMO_CANVAS_PX_PER_CM, texture="speckle", seed=0
    )
    image_size = (4032, 3024)  # a 4:3 still, not 16:9 video
    passes = []
    for name, center_x in (("strip_left", 125.0), ("strip_right", 375.0)):
        pose = CameraPose(center_x, 200.0, 130.0, pitch_deg=1.0, yaw_deg=-2.0)
        focal = fit_focal_for_coverage((250.0, 240.0), pose, image_size)
        scene = capture(
            arena,
            pose=pose,
            objects=DEMO_OBJECTS,
            image_size=image_size,
            focal_px=focal,
            distortion=(-0.05, 0.01, 0, 0, 0),
            render_px_per_cm=DEMO_CANVAS_PX_PER_CM,
            topdown=canvas,
        )
        profile = CameraProfile(
            name="synthetic",
            image_size=image_size,
            fx=focal, fy=focal,
            cx=image_size[0] / 2, cy=image_size[1] / 2,
            k1=-0.05, k2=0.01, alpha=1.0,
        )
        passes.append((name, Undistorter(profile)(scene.image)))
    return passes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Top-view run dir (the map).")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frames", help="Directory of markerless high-res frames.")
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--arena", default=None)
    parser.add_argument("--profile", default="photo_4k")
    parser.add_argument("--pad-cm", type=float, default=5.0)
    parser.add_argument("--min-inliers", type=int, default=Gates.min_inliers)
    parser.add_argument("--max-rms-cm", type=float, default=Gates.max_rms_cm)
    args = parser.parse_args()

    run_dir = Path(args.run)
    arena = Arena.from_yaml(args.arena) if args.arena else Arena.from_yaml()

    with (run_dir / "metadata.json").open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    topview_image = cv2.imread(str(run_dir / "topview.jpg"))
    if topview_image is None:
        raise FileNotFoundError(run_dir / "topview.jpg")

    reference = Reference(topview_image, metadata["topview_px_per_cm"], arena)
    print(
        f"map: {run_dir.name}  {reference.px_per_cm:.2f} px/cm  "
        f"({10 / reference.px_per_cm:.2f} mm/px)  {reference.feature_count} features"
    )
    if reference.feature_count < 200:
        print(
            "  WARNING: very few features on the top view. If the arena surface is this\n"
            "  smooth in reality, markerless localisation will not work and the strip\n"
            "  pass cannot be used. Check this before relying on the flight plan."
        )

    if args.synthetic:
        candidates = synthetic_strip(arena, run_dir)
    else:
        from mai.undistort import CameraProfile, Undistorter

        undistorter = Undistorter(CameraProfile.load(args.profile))
        candidates = [
            (Path(frame.source).stem, undistorter(frame.image))
            for frame in frames.iter_frames(args.frames)
        ]

    gates = Gates(min_inliers=args.min_inliers, max_rms_cm=args.max_rms_cm)
    output_dir = run_dir / "passes"
    output_dir.mkdir(parents=True, exist_ok=True)

    placed, refused = [], []
    for name, image in candidates:
        print(f"\n{name}: ", end="")
        try:
            result = localize.localize(image, reference, gates=gates)
        except LocalizationError as error:
            print(f"REFUSED\n  {error}")
            refused.append({"frame": name, "reason": str(error)})
            continue

        print(
            f"placed  {result.inliers}/{result.matches} inliers  "
            f"RMS {result.homography.rms_cm:.2f}cm  "
            f"{result.px_per_cm:.1f} px/cm ({10 / result.px_per_cm:.2f} mm/px)  "
            f"rot {result.rotation_deg:+.0f}deg  alias {result.alias_ratio:.2f}"
        )
        gain = result.px_per_cm / reference.px_per_cm
        cluster_px = 28.0 / (10.0 / result.px_per_cm)
        print(f"  {gain:.1f}x the top view -> cluster munition is now {cluster_px:.0f}px")
        print(f"  covers {len(result.zones)} zones: {', '.join(result.zones)}")

        pass_dir = output_dir / name
        covered = [zone for zone in arena.zones if zone.id in result.zones]
        crops = {}
        for zone in covered:
            crop, _ = zones.crop_zone(
                image, zone, result.homography, pad_cm=args.pad_cm
            )
            crops[zone.id] = crop
            (pass_dir / "zones").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(pass_dir / "zones" / f"{zone.id}.jpg"), crop)

        cv2.imwrite(
            str(pass_dir / "grid_on_source.jpg"),
            viz.draw_zone_grid_on_source(image, arena, result.homography),
        )
        cv2.imwrite(
            str(pass_dir / "contact_sheet.jpg"),
            viz.contact_sheet(
                crops,
                [zone.id for zone in covered],
                columns=5,
                captions={zone.id: f"{10 / result.px_per_cm:.2f}mm/px" for zone in covered},
            ),
        )
        placed.append(
            {
                "frame": name,
                "homography_image_to_arena_cm": result.homography.matrix.tolist(),
                "inliers": result.inliers,
                "matches": result.matches,
                "rms_cm": round(result.homography.rms_cm, 4),
                "px_per_cm": round(result.px_per_cm, 3),
                "mm_per_px": round(10 / result.px_per_cm, 3),
                "rotation_deg": round(result.rotation_deg, 2),
                "alias_ratio": round(result.alias_ratio, 3),
                "zones": result.zones,
            }
        )

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump({"placed": placed, "refused": refused}, file, indent=2)

    print(f"\n{len(placed)} placed, {len(refused)} refused -> {output_dir}")
    if refused:
        print("Refused frames fall back to the top view; they are not guessed.")
    covered = sorted({zone for entry in placed for zone in entry["zones"]})
    airfield = {zone.id for zone in arena.airfield_zones}
    missing = sorted(airfield - set(covered))
    if missing:
        print(f"\nAirfield zones with NO high-res coverage: {missing}")
        print("  These fall back to the top view, where a cluster munition is ~12px.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
