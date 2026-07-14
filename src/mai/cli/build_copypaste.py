"""Grow the UXO training set by pasting real objects into real backgrounds.

    python -m mai.cli.build_copypaste --task uxo --per-image 3

Only the TRAIN split is augmented -- pasting into val would be marking our own homework.

Placement is restricted to zones where the patch's existing lean is still physically
correct: its own zone in another scene, or the zone mirrored through the drone's nadir
(with the patch mirrored to match). See `mai/copypaste.py` for why that restriction
exists; the short version is that lean encodes height, height is what separates the
classes at 18-30px, and pasting a left-leaning missile onto a right-leaning zone's
background teaches the model that lean means nothing.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from mai import copypaste
from mai.arena import Arena

CROP_RE = re.compile(r"^(?P<video>top_center_\d+)_f\d+_(?P<zone>[A-Z]{2}-[A-Z]?\d+)$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="uxo", choices=["uxo", "crater"])
    parser.add_argument("--dataset", default=None, help="Defaults to data/yolo/<task>")
    parser.add_argument("--output", default=None, help="Defaults to data/yolo/<task>_cp")
    parser.add_argument(
        "--per-image",
        type=int,
        default=3,
        help="Synthetic variants generated per real TRAIN crop.",
    )
    parser.add_argument("--max-objects", type=int, default=2, help="Extra objects per crop.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    arena = Arena.from_yaml()
    dataset = Path(args.dataset or f"data/yolo/{args.task}")
    output = Path(args.output or f"data/yolo/{args.task}_cp")
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    if output.exists():
        shutil.rmtree(output)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    # val is copied through untouched. Synthetic data in a validation set would make the
    # metric a measure of our own augmentation, not of the arena.
    for split in ("train", "val"):
        for image in (dataset / "images" / split).glob("*.jpg"):
            shutil.copy2(image, output / "images" / split / image.name)
            label = dataset / "labels" / split / f"{image.stem}.txt"
            shutil.copy2(label, output / "labels" / split / f"{image.stem}.txt")

    # --- object bank, cut from the TRAIN split only ---
    bank: list[copypaste.Patch] = []
    for image_path in sorted((dataset / "images" / "train").glob("*.jpg")):
        match = CROP_RE.match(image_path.stem)
        if not match:
            continue
        label = dataset / "labels" / "train" / f"{image_path.stem}.txt"
        text = label.read_text().strip()
        if not text:
            continue
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        for line in text.split("\n"):
            cls, cx, cy, bw, bh = map(float, line.split())
            box = (
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            )
            patch = copypaste.cut(
                image, box, int(cls), match.group("zone"), image_path.stem
            )
            if patch is not None:
                bank.append(patch)

    by_class = Counter(patch.cls for patch in bank)
    print(f"object bank: {len(bank)} patches  {dict(sorted(by_class.items()))}")

    # Which zones each patch may legally land in.
    placer = copypaste.Placer(arena)
    by_zone: dict[str, list[copypaste.Patch]] = {}
    for patch in bank:
        for zone_id, mirror in placer.targets(patch.zone_id):
            by_zone.setdefault(zone_id, []).append((patch, mirror))
    reach = np.mean([len(placer.targets(p.zone_id)) for p in bank])
    print(f"average legal destinations per patch: {reach:.1f} zone(s) x scenes\n")

    # --- generate ---
    made = 0
    added = Counter()
    for image_path in sorted((dataset / "images" / "train").glob("*.jpg")):
        match = CROP_RE.match(image_path.stem)
        if not match:
            continue
        zone_id = match.group("zone")
        candidates = by_zone.get(zone_id, [])
        if not candidates:
            continue

        background = cv2.imread(str(image_path))
        existing = (dataset / "labels" / "train" / f"{image_path.stem}.txt").read_text().strip()

        for variant in range(args.per_image):
            canvas = background.copy()
            boxes = []
            if existing:
                for line in existing.split("\n"):
                    cls, cx, cy, bw, bh = map(float, line.split())
                    h, w = canvas.shape[:2]
                    boxes.append(
                        (
                            int(cls),
                            (
                                (cx - bw / 2) * w, (cy - bh / 2) * h,
                                (cx + bw / 2) * w, (cy + bh / 2) * h,
                            ),
                        )
                    )

            wanted = rng.randint(1, args.max_objects)
            placed = 0
            for _ in range(wanted * 6):
                if placed >= wanted:
                    break
                patch, mirror = rng.choice(candidates)
                # Do not paste an object back into the crop it was cut from.
                if patch.source == image_path.stem:
                    continue
                result = placer.paste(canvas, patch, mirror, rng)
                if result is None:
                    continue
                candidate, box = result
                if any(copypaste.boxes_overlap(box, b) for _, b in boxes):
                    continue
                canvas = candidate
                boxes.append((patch.cls, box))
                added[patch.cls] += 1
                placed += 1

            if placed == 0:
                continue

            stem = f"{image_path.stem}_cp{variant}"
            cv2.imwrite(str(output / "images" / "train" / f"{stem}.jpg"), canvas)
            lines = [
                f"{cls} " + " ".join(f"{v:.6f}" for v in copypaste.to_yolo(box, canvas.shape))
                for cls, box in boxes
            ]
            (output / "labels" / "train" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            made += 1

    shutil.copy2(dataset / "data.yaml", output / "data.yaml")
    text = (output / "data.yaml").read_text().replace(
        str((dataset).resolve()), str(output.resolve())
    )
    (output / "data.yaml").write_text(text)

    real = len(list((dataset / "images" / "train").glob("*.jpg")))
    print(f"generated {made} synthetic crops (train now {real + made}, was {real})")
    print(f"objects pasted by class: {dict(sorted(added.items()))}")
    print(f"  -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
