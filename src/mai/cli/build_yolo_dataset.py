"""Assemble the labelled zone crops into YOLO datasets.

    python -m mai.cli.build_yolo_dataset --output data/yolo

Builds two datasets from the same crops -- `crater` (1 class) and `uxo` (3 classes) --
because the two problems are not equally hard and should not share a model: craters are
50-90px and high contrast, UXO bottom out at an 18px ball.

THE SPLIT IS BY VIDEO, AND THAT IS NOT A DETAIL. There are only 8 scenes, and each one
contributed 2 frames of a hovering drone -- so the two frames of a video are the same
arena, pixels apart. Split those at random and the *same crater* lands in both train and
val. Validation would come back near-perfect and mean absolutely nothing. Whole videos
go to one side or the other, and `tests/test_dataset.py` asserts it.

NEGATIVES ARE INCLUDED ON PURPOSE. Only the crops containing an object have a label
file; the other ~175 have none. An unlabelled image is not the same as an absent one --
YOLO treats a listed image with an empty label file as pure background, and on a runway
covered in scorch marks, tyre streaks and paint, background crops are the cheapest
false-positive suppression available. So every airfield crop is listed, and the empty
ones get an explicit empty .txt.
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter
from pathlib import Path

import yaml

CLASSES = {
    "crater": ["crater"],
    # Order fixed by the label files. TASK.md 8.3 misdescribes the last two props
    # (dumb is a small ball, cluster is a long object); the names below are the
    # dashboard's own codes, which is what ultimately has to be emitted.
    "uxo": ["misile", "dumb", "cluster"],
}
LABEL_DIRS = {
    "crater": "zones_labels_crater/train",
    "uxo": "zones_labels_UXO/train",
}

CROP_RE = re.compile(r"^(?P<video>top_center_\d+)_f\d+_(?P<zone>[A-Z]{2}-[A-Z]?\d+)$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", default="outputs/dataset/zones")
    parser.add_argument("--labels", default="outputs/dataset/labels")
    parser.add_argument("--output", default="data/yolo")
    parser.add_argument(
        "--val-videos",
        default="top_center_7,top_center_8",
        help="Whole videos held out for validation. Never split frames.",
    )
    args = parser.parse_args()

    crops = Path(args.crops)
    labels = Path(args.labels)
    output = Path(args.output)
    val_videos = {v.strip() for v in args.val_videos.split(",")}

    # Airfield only: craters and UXO are scored in RW/TW, never in the facility zones.
    airfield = []
    for image in sorted(crops.glob("*.jpg")):
        match = CROP_RE.match(image.stem)
        if match and not match.group("zone").startswith("FA"):
            airfield.append((image, match.group("video")))

    videos = sorted({video for _, video in airfield})
    unknown = val_videos - set(videos)
    if unknown:
        raise SystemExit(f"--val-videos names videos that do not exist: {sorted(unknown)}")

    print(f"{len(airfield)} airfield crops from {len(videos)} videos")
    print(f"  train: {sorted(set(videos) - val_videos)}")
    print(f"  val  : {sorted(val_videos)}\n")

    for task, names in CLASSES.items():
        label_dir = labels / LABEL_DIRS[task]
        root = output / task
        if root.exists():
            shutil.rmtree(root)

        counts = {"train": Counter(), "val": Counter()}
        images = {"train": 0, "val": 0}
        empties = {"train": 0, "val": 0}

        for image, video in airfield:
            split = "val" if video in val_videos else "train"
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)

            shutil.copy2(image, root / "images" / split / image.name)

            source = label_dir / f"{image.stem}.txt"
            text = source.read_text().strip() if source.exists() else ""
            # An explicit empty file, not a missing one: this crop IS background, and we
            # want the model told so.
            (root / "labels" / split / f"{image.stem}.txt").write_text(
                text + "\n" if text else ""
            )

            images[split] += 1
            if not text:
                empties[split] += 1
            else:
                for line in text.split("\n"):
                    counts[split][int(line.split()[0])] += 1

        with (root / "data.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                {
                    "path": str(root.resolve()),
                    "train": "images/train",
                    "val": "images/val",
                    "names": {index: name for index, name in enumerate(names)},
                },
                file,
                sort_keys=False,
            )

        print(f"{task}:")
        for split in ("train", "val"):
            per_class = ", ".join(
                f"{names[index]} {counts[split][index]}" for index in range(len(names))
            )
            print(
                f"  {split:5s} {images[split]:3d} images "
                f"({empties[split]:3d} background) | {per_class}"
            )
        print(f"  -> {root}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
