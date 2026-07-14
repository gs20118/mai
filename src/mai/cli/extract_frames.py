"""Pull sharp frames out of drone footage.

    python -m mai.cli.extract_frames --source data/raw/hover.mp4 --output data/frames/hover

Motion blur is the enemy: a 28mm cluster munition is only ~12 pixels across at
whole-arena altitude, and a blurred 12px blob is nothing at all. So frames are
scored by variance-of-Laplacian and the soft ones are dropped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from mai import frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval-sec", type=float, default=0.5)
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=None,
        help="Laplacian variance floor. Default: keep the sharpest --keep-frac.",
    )
    parser.add_argument("--keep-frac", type=float, default=0.6)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for frame in frames.iter_frames(args.source, sample_sec=args.interval_sec):
        candidates.append((frames.sharpness(frame.image), frame))

    if not candidates:
        print(f"no frames read from {args.source}")
        return 1

    scores = np.array([score for score, _ in candidates])
    threshold = (
        args.min_sharpness
        if args.min_sharpness is not None
        else float(np.quantile(scores, 1.0 - args.keep_frac))
    )

    kept = 0
    for score, frame in candidates:
        if score < threshold:
            continue
        cv2.imwrite(str(output_dir / f"{frame.label}.jpg"), frame.image)
        kept += 1
        if args.max_frames and kept >= args.max_frames:
            break

    print(f"read {len(candidates)} frames, kept {kept} above sharpness {threshold:.0f}")
    print(f"  sharpness range: {scores.min():.0f} .. {scores.max():.0f}")
    print(f"  -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
