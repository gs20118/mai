"""One-time discovery: which ArUco dictionary did the organizers use, and which IDs?

    python -m mai.cli.scan_aruco --source data/raw/practice.mp4

Brute-forces every candidate dictionary across the footage and reports which ones
decode, how often, and how large the markers appear. Run it once on practice
footage, paste the resulting snippet into configs/arena.yaml, and never run it
again -- it is far too slow for the mission path.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np

from mai import aruco, frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Video, image, or directory.")
    parser.add_argument("--sample-sec", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=30)
    args = parser.parse_args()

    hits: dict[str, Counter] = defaultdict(Counter)
    sizes: dict[str, list[float]] = defaultdict(list)
    scanned = 0

    for frame in frames.iter_frames(args.source, sample_sec=args.sample_sec):
        if scanned >= args.max_frames:
            break
        scanned += 1
        for name in aruco.CANDIDATE_DICTIONARIES:
            try:
                detections = aruco.detect(frame.image, aruco.build_detector(name))
            except AttributeError:
                continue  # dictionary absent from this OpenCV build
            for detection in detections:
                hits[name][detection.id] += 1
                sizes[name].append(float(np.sqrt(detection.area_px)))

    if not hits:
        print(f"No ArUco markers found in {scanned} frames of {args.source}.")
        print("Either the arena has none, or the drone is too high / the frames too blurred.")
        return 1

    print(f"scanned {scanned} frames\n")
    ranked = sorted(hits, key=lambda name: -sum(hits[name].values()))
    for name in ranked:
        counts = hits[name]
        median_side = float(np.median(sizes[name]))
        print(f"{name}")
        print(f"  ids seen     : {sorted(counts)}")
        print(f"  frames w/ id : {dict(sorted(counts.items()))} of {scanned}")
        print(f"  median size  : {median_side:.0f}px per side")
        if median_side < 30:
            print("  WARNING: markers this small decode unreliably. Fly lower, or")
            print("           expect frames to be rejected for missing markers.")
        print()

    best = ranked[0]
    print("-" * 68)
    print(f"Most likely: {best}. Paste into configs/arena.yaml, then MEASURE the")
    print("marker centres in arena cm (origin = top-left corner, +x right, +y down):\n")
    print("aruco:")
    print(f"  dictionary: {best}")
    print("  marker_size_cm: ??.?   # measure the black square's edge")
    print("  markers:")
    for marker_id in sorted(hits[best]):
        print(f"    {marker_id}: {{ center: [??.?, ??.?], rotation_deg: 0 }}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
