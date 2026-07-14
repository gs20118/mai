"""Turn raw drone videos into rectified, per-zone crops ready for labelling.

    python -m mai.cli.extract_dataset --source data/raw --output outputs/dataset

For each video: sample frames, keep the sharp ones, register each on the four
corner ArUco markers, gate on reprojection error, then emit

    <output>/topview/<video>_f<frame>.jpg          whole arena, rectified
    <output>/zones/<video>_f<frame>_<ZONE>.jpg     one crop per zone, native res
    <output>/qa/<video>_f<frame>_grid.jpg          zone grid drawn on the source
    <output>/manifest.json                         provenance + homography + GSD

Zone crops come straight from the source frame at each zone's own resolution, so
nothing is resampled away -- a 28mm cluster munition has no pixels to spare.

Undistortion is a pass-through whenever the camera profile carries zero distortion
coefficients, which is the case here: the drone's footage is already lens-corrected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from mai import aruco, frames, homography, topview, viz, zones
from mai.arena import Arena
from mai.undistort import CameraProfile, Undistorter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Video, or a directory of videos.")
    parser.add_argument("--output", default="outputs/dataset")
    parser.add_argument("--arena", default=None)
    parser.add_argument("--profile", default="video_4k")
    parser.add_argument(
        "--per-video",
        type=int,
        default=2,
        help="Frames to keep per video. The drone hovers, so extra frames are near "
        "duplicates -- useful as augmentation, not as new scenes.",
    )
    parser.add_argument("--sample-sec", type=float, default=0.5)
    parser.add_argument("--max-rms-cm", type=float, default=1.0)
    parser.add_argument("--pad-cm", type=float, default=5.0)
    parser.add_argument(
        "--zones",
        default="all",
        help="'all', 'airfield' (RW+TW only), or a comma-separated list of zone ids.",
    )
    args = parser.parse_args()

    arena = Arena.from_yaml(args.arena) if args.arena else Arena.from_yaml()
    profile = CameraProfile.load(args.profile)
    undistorter = Undistorter(profile)
    detector = aruco.build_detector(arena.dictionary)

    print(arena)
    print(f"markers: {sorted(arena.markers)}  size {arena.marker_size_cm}cm")
    if not profile.has_distortion:
        print(f"camera profile '{profile.name}': no distortion, undistortion is a no-op")

    if args.zones == "all":
        wanted = arena.zones
    elif args.zones == "airfield":
        wanted = arena.airfield_zones
    else:
        wanted = [arena.zone(z.strip()) for z in args.zones.split(",")]
    print(f"cropping {len(wanted)} zones per frame\n")

    source_path = Path(args.source)
    sources = (
        sorted(p for p in source_path.iterdir() if p.suffix.lower() in {".mp4", ".mov"})
        if source_path.is_dir()
        else [source_path]
    )

    output = Path(args.output)
    for sub in ("topview", "zones", "qa"):
        (output / sub).mkdir(parents=True, exist_ok=True)

    manifest = []
    total_crops = 0

    for source in sources:
        candidates = []
        rejected = 0
        for frame in frames.iter_frames(source, sample_sec=args.sample_sec):
            try:
                registration = topview.register(frame, arena, undistorter, detector)
            except homography.HomographyError:
                rejected += 1
                continue
            if registration.homography.rms_cm > args.max_rms_cm:
                rejected += 1
                continue
            candidates.append(registration)

        if not candidates:
            print(f"{source.name}: NO usable frames ({rejected} rejected)")
            continue

        # Sharpest first: motion blur destroys a 12px target far more surely than a
        # tenth of a millimetre of reprojection error does.
        candidates.sort(key=lambda r: -r.sharpness)
        keep = candidates[: args.per_video]

        rms = np.mean([r.homography.rms_cm for r in candidates])
        print(
            f"{source.name}: {len(candidates)} registered ({rejected} rejected), "
            f"mean RMS {rms:.3f}cm -> keeping {len(keep)}"
        )

        for registration in keep:
            stem = f"{source.stem}_f{registration.frame.index:05d}"
            solved = registration.homography
            image = registration.undistorted

            px_per_cm = topview.native_px_per_cm(arena, solved)
            warped = topview.warp(image, arena, solved, px_per_cm)
            cv2.imwrite(str(output / "topview" / f"{stem}.jpg"), warped)
            cv2.imwrite(
                str(output / "qa" / f"{stem}_grid.jpg"),
                viz.draw_zone_grid_on_source(image, arena, solved),
            )

            records = []
            for zone in wanted:
                crop, record = zones.crop_zone(
                    image, zone, solved, pad_cm=args.pad_cm
                )
                cv2.imwrite(str(output / "zones" / f"{stem}_{zone.id}.jpg"), crop)
                records.append(record)
                total_crops += 1

            manifest.append(
                {
                    "video": source.name,
                    "frame_index": registration.frame.index,
                    "time_sec": registration.frame.time_sec,
                    "stem": stem,
                    "sharpness": round(registration.sharpness, 1),
                    "marker_ids": solved.marker_ids,
                    "reprojection_rms_cm": round(solved.rms_cm, 4),
                    "homography_image_to_arena_cm": solved.matrix.tolist(),
                    "topview_px_per_cm": round(px_per_cm, 3),
                    "mm_per_px": round(10.0 / px_per_cm, 3),
                    "zones": {r.zone_id: {"mm_per_px": r.mm_per_px, "size_px": r.size_px} for r in records},
                }
            )
            print(
                f"    f{registration.frame.index:05d}  RMS {solved.rms_cm:.3f}cm  "
                f"{10 / px_per_cm:.2f} mm/px  sharpness {registration.sharpness:.0f}"
            )

    with (output / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    if not manifest:
        print("\nNothing extracted.")
        return 1

    mm_per_px = np.array([entry["mm_per_px"] for entry in manifest])
    print(f"\n{len(manifest)} frames, {total_crops} zone crops -> {output}")
    print(
        f"ground sample distance: {mm_per_px.min():.2f} .. {mm_per_px.max():.2f} mm/px"
    )
    print("\nTarget size at the median GSD:")
    median = float(np.median(mm_per_px))
    for name, spec in arena.targets.items():
        pixels = spec["w_mm"] / median
        verdict = "OK" if pixels >= 20 else ("MARGINAL" if pixels >= 10 else "TOO SMALL")
        print(f"  {name:15s} {pixels:6.1f}px   {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
