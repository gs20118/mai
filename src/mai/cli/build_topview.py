"""Frame -> undistort -> ArUco -> homography -> top-view.

    python -m mai.cli.build_topview --source data/raw/hover.mp4 --profile video_4k
    python -m mai.cli.build_topview --synthetic          # no drone required

Writes a run directory whose most important artifact is grid_on_source.jpg: the
arena grid projected back onto the drone's own frame. If those lines land on the
real runway and taxiway edges, then the marker map, the distortion profile, the
homography and the band layout are all correct at once. If they don't, one of them
is wrong and it is usually obvious which.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from mai import topview, viz
from mai.arena import Arena
from mai.frames import Frame
from mai.synthetic import CameraPose, PlacedObject, capture, fit_focal_px
from mai.undistort import CameraProfile, Undistorter

# A demo layout so `--synthetic` exercises the same code path real footage will.
DEMO_OBJECTS = [
    PlacedObject("crater_big", 175.0, 200.0),  # RW-04
    PlacedObject("crater_medium", 275.0, 200.0),  # RW-06
    PlacedObject("crater_small", 475.0, 200.0),  # RW-10
    PlacedObject("crater_medium", 150.0, 120.0),  # TW-A2
    PlacedObject("crater_small", 350.0, 280.0),  # TW-B4
    PlacedObject("uxo_misile", 25.0, 200.0),  # RW-01
    PlacedObject("uxo_dumb", 225.0, 200.0),  # RW-05
    PlacedObject("uxo_cluster", 425.0, 200.0),  # RW-09
    PlacedObject("uxo_cluster", 50.0, 120.0),  # TW-A1
    PlacedObject("uxo_dumb", 450.0, 120.0),  # TW-A5
    PlacedObject("uxo_misile", 150.0, 280.0),  # TW-B2
]


def synthetic_frame(arena: Arena, height_cm: float) -> tuple[Frame, CameraProfile]:
    pose = CameraPose(
        x_cm=arena.width_cm / 2,
        y_cm=arena.height_cm / 2,
        height_cm=height_cm,
        pitch_deg=2.0,
        yaw_deg=4.0,
    )
    image_size = (3840, 2160)
    focal = fit_focal_px(arena, pose, image_size)
    distortion = (-0.12, 0.02, 0.0, 0.0, 0.0)

    scene = capture(
        arena,
        pose=pose,
        objects=DEMO_OBJECTS,
        image_size=image_size,
        focal_px=focal,
        distortion=distortion,
    )
    profile = CameraProfile(
        name="synthetic",
        image_size=image_size,
        fx=focal,
        fy=focal,
        cx=image_size[0] / 2,
        cy=image_size[1] / 2,
        k1=distortion[0],
        k2=distortion[1],
        p1=distortion[2],
        p2=distortion[3],
        k3=distortion[4],
        alpha=1.0,
    )
    return Frame(image=scene.image, index=0, source="synthetic"), profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", help="Video file, image, or directory of images.")
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="Render a fake arena instead. Exercises the full pipeline with no drone.",
    )
    parser.add_argument("--arena", default=None, help="Path to arena.yaml.")
    parser.add_argument("--profile", default="video_4k", help="Camera profile name.")
    parser.add_argument("--camera", default=None, help="Path to camera.yaml.")
    parser.add_argument("--output", default=None, help="Run directory.")
    parser.add_argument("--sample-sec", type=float, default=0.5)
    parser.add_argument(
        "--max-rms-cm",
        type=float,
        default=1.0,
        help="Reject frames whose marker reprojection error exceeds this.",
    )
    parser.add_argument("--px-per-cm", type=float, default=None, help="Top-view scale.")
    parser.add_argument("--synthetic-height-cm", type=float, default=300.0)
    args = parser.parse_args()

    arena = Arena.from_yaml(args.arena) if args.arena else Arena.from_yaml()
    print(arena)

    run_dir = Path(
        args.output
        or f"outputs/topview_{datetime.now():%Y%m%d_%H%M%S}"
        + ("_synthetic" if args.synthetic else "")
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        frame, profile = synthetic_frame(arena, args.synthetic_height_cm)
        undistorter = Undistorter(profile)
        detector_arena = arena
        from mai import aruco as aruco_module

        best = topview.register(
            frame, detector_arena, undistorter, aruco_module.build_detector(arena.dictionary)
        )
        rejections = []
    else:
        profile = (
            CameraProfile.load(args.profile, args.camera)
            if args.camera
            else CameraProfile.load(args.profile)
        )
        undistorter = Undistorter(profile)
        best, rejections = topview.select_best(
            args.source,
            arena,
            undistorter,
            sample_sec=args.sample_sec,
            max_rms_cm=args.max_rms_cm,
        )

    if best is None:
        print(f"\nNo frame could be registered. {len(rejections)} rejected:")
        for rejection in rejections[:10]:
            print(f"  frame {rejection.frame_index}: {rejection.reason}")
        print(
            "\nIf markers were never seen, the drone is too high, too fast, or the\n"
            "dictionary in arena.yaml is wrong -- run `python -m mai.cli.scan_aruco`.\n"
            "If they were seen but reprojected badly, the marker map is wrong (wrong\n"
            "centres, or a marker taped down rotated) or the camera profile is off."
        )
        return 1

    solved = best.homography
    px_per_cm = args.px_per_cm or topview.native_px_per_cm(arena, solved)
    warped = topview.warp(best.undistorted, arena, solved, px_per_cm)

    cv2.imwrite(str(run_dir / "selected_frame.jpg"), best.frame.image)
    cv2.imwrite(str(run_dir / "undistorted.jpg"), best.undistorted)
    cv2.imwrite(
        str(run_dir / "markers_overlay.jpg"),
        viz.draw_markers(best.undistorted, best.detections),
    )
    cv2.imwrite(
        str(run_dir / "grid_on_source.jpg"),
        viz.draw_zone_grid_on_source(best.undistorted, arena, solved),
    )
    cv2.imwrite(str(run_dir / "topview.jpg"), warped)
    cv2.imwrite(
        str(run_dir / "topview_grid.jpg"),
        viz.draw_zone_grid_on_topview(warped, arena, px_per_cm),
    )

    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": best.frame.source,
        "frame_index": best.frame.index,
        "frame_time_sec": best.frame.time_sec,
        "sharpness": round(best.sharpness, 1),
        "camera_profile": profile.name,
        "camera": {
            "fx": profile.fx, "fy": profile.fy, "cx": profile.cx, "cy": profile.cy,
            "k1": profile.k1, "k2": profile.k2, "p1": profile.p1,
            "p2": profile.p2, "k3": profile.k3, "alpha": profile.alpha,
        },
        "marker_ids": solved.marker_ids,
        "homography_image_to_arena_cm": solved.matrix.tolist(),
        "reprojection_rms_cm": round(solved.rms_cm, 4),
        "reprojection_max_cm": round(solved.max_error_cm, 4),
        "inliers": f"{solved.inliers}/{solved.total}",
        "topview_px_per_cm": round(px_per_cm, 3),
        "topview_mm_per_px": round(10.0 / px_per_cm, 3),
        "undistorted_size": list(best.undistorted.shape[1::-1]),
        "rejected_frames": len(rejections),
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"\nregistered on markers {solved.marker_ids}")
    print(
        f"  reprojection RMS {solved.rms_cm:.3f}cm  (max {solved.max_error_cm:.3f}cm, "
        f"{solved.inliers}/{solved.total} inliers)"
    )
    print(f"  top-view scale   {px_per_cm:.2f} px/cm = {10 / px_per_cm:.2f} mm/px")
    print(f"\n  -> {run_dir}")
    print(f"  LOOK AT {run_dir / 'grid_on_source.jpg'} FIRST.")
    print("  The drawn grid must land on the real runway and taxiway edges.")
    print(f"\nNext: python -m mai.cli.crop_zones --run {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
