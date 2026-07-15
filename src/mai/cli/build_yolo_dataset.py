"""Assemble the labelled zone crops into YOLO datasets.

    python -m mai.cli.build_yolo_dataset --output data/yolo

Builds two datasets from the same crops -- `crater` (1 class) and `uxo` (3 classes) --
because the two problems are not equally hard and should not share a model: craters are
50-90px and high contrast, UXO bottom out at an 18px ball.

Pulls from every labelled source (`zones`, `zones_2`, ...). They are the same arena at
slightly different altitudes -- 4.92 vs 5.25 px/cm -- which is variety we want, not noise.

ONE SCENE PER VIDEO. The drone HOVERS, so two frames of the same video are the same arena
a few pixels apart: the same craters, the same UXO, the same shadows. Keeping both does
not double the information, it doubles the COUNT -- which flatters every metric and
invites leakage. So each video contributes exactly one frame (the sharpest, where we can
tell), and the dataset size then means what it says.

THE SPLIT IS BY VIDEO, and that is not a detail either. Whole videos go to one side or
the other; `tests/test_sizing.py` asserts it. Split at random and the same object can
land on both sides, and validation comes back near-perfect and meaningless.

NEGATIVES ARE INCLUDED ON PURPOSE. Only crops containing an object have a label file. An
unlabelled image is not the same as an absent one -- YOLO treats a listed image with an
empty label file as pure background, and on a runway covered in scorch marks, tyre
streaks and paint, background crops are the cheapest false-positive suppression there is.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

# Every labelled source, and the label dir for each task -- or None if that task was not
# labelled for this source. None is NOT the same as "empty": a source with crater=None
# has crops we simply never inspected for craters, so they must not enter the crater
# dataset AT ALL. Feeding them in as empty (background) would assert "no crater here" on
# crops that might well contain one -- teaching the crater model false negatives.
# zones_3 is UXO-only, and its labels sit one directory deeper than the others.
SOURCES = [
    {"crops": "zones", "crater": "zones_labels_crater/train", "uxo": "zones_labels_UXO/train"},
    {"crops": "zones_2", "crater": "zones_2_labels_crater/train", "uxo": "zones_2_labels_UXO/train"},
    {"crops": "zones_3", "crater": None, "uxo": "zones_3_labels_UXO/labels/train/zones"},
]

CLASSES = {
    "crater": ["crater"],
    # Order fixed by the label files, and verified to be consistent across BOTH sources
    # by the lean test: "dumb" shows near-zero lean in each (0.4cm), which only the flat
    # ball can do. The names are the dashboard's own codes, misspellings included.
    "uxo": ["misile", "dumb", "cluster"],
}

CROP_RE = re.compile(
    r"^(?P<stem>(?P<video>top_center_\d+)_f\d+)_(?P<zone>[A-Z]{2}-[A-Z]?\d+)$"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/dataset")
    parser.add_argument("--output", default="data/yolo")
    parser.add_argument(
        "--val-scenes",
        default="top_center_9,top_center_13",
        help="Held out, named by ANY video in the scene. The whole scene goes with it.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    val_seeds = {v.strip() for v in args.val_scenes.split(",") if v.strip()}

    sharpness = {}
    manifest = root / "manifest.json"
    if manifest.exists():
        sharpness = {e["stem"]: e.get("sharpness", 0.0) for e in json.load(open(manifest))}

    # --- gather every airfield crop, grouped by video then scene ---
    scenes: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    label_dirs: dict[str, dict[str, Path | None]] = {}

    for source in SOURCES:
        crops_name = source["crops"]
        crops = root / crops_name
        if not crops.exists():
            print(f"skipping {crops_name}: not found")
            continue
        label_dirs[crops_name] = {
            task: (root / "labels" / source[task]) if source[task] else None
            for task in CLASSES
        }
        for image in sorted(crops.glob("*.jpg")):
            match = CROP_RE.match(image.stem)
            if not match or match.group("zone").startswith("FA"):
                continue  # craters and UXO are only ever scored in RW/TW
            scenes[match.group("video")][match.group("stem")].append(
                (image, crops_name)
            )

    # --- one frame per video: the drone hovers, so its frames are near-duplicates ---
    chosen = {
        video: max(by_stem, key=lambda s: sharpness.get(s, 0.0))
        for video, by_stem in scenes.items()
    }
    videos = sorted(chosen)

    # A DIFFERENT VIDEO IS NOT A DIFFERENT SCENE, and this dataset is a trap.
    #
    # 13 videos contain only 6 distinct object layouts: videos 4/5/6/7/8 are ONE scene,
    # 10/12/14 are one, 2/3 are one. The drone was simply re-flown over an unchanged
    # arena. Split those by VIDEO and the same craters land in train and val, and
    # validation comes back near-perfect while measuring nothing at all -- which is
    # exactly what happened to an earlier run of this pipeline, which reported recall
    # 1.000 and mAP50 0.995 on a "held-out" pair that shared its scene with three
    # training videos.
    #
    # So fingerprint each video by WHAT object is in WHICH zone, group videos that agree,
    # and split by GROUP. Videos of the same scene are kept -- they are genuine viewpoint
    # variety, different drone position and lean -- but they never straddle the split.
    fingerprints: dict[str, frozenset] = {}
    for video in videos:
        marks = set()
        for image, crops_name in scenes[video][chosen[video]]:
            zone = CROP_RE.match(image.stem).group("zone")
            for task in ("crater", "uxo"):
                task_dir = label_dirs[crops_name][task]
                if task_dir is None:
                    continue
                path = task_dir / f"{image.stem}.txt"
                if not path.exists():
                    continue
                for line in path.read_text().strip().split("\n"):
                    if line:
                        marks.add((task, zone, line.split()[0]))
        fingerprints[video] = frozenset(marks)

    groups: dict[frozenset, list[str]] = defaultdict(list)
    for video in videos:
        groups[fingerprints[video]].append(video)

    val_videos: set[str] = set()
    for members in groups.values():
        if val_seeds & set(members):
            val_videos |= set(members)
    if not val_videos:
        raise SystemExit(f"--val-scenes {sorted(val_seeds)} matched no video")

    print(f"{len(videos)} videos, but only {len(groups)} DISTINCT SCENES:\n")
    for index, (_, members) in enumerate(
        sorted(groups.items(), key=lambda kv: min(int(v.split("_")[-1]) for v in kv[1])), 1
    ):
        side = "VAL " if set(members) & val_videos else "train"
        order = sorted(members, key=lambda v: int(v.split("_")[-1]))
        print(f"  {side} scene {index}: {', '.join(order)}")
    n_val = sum(1 for _, m in groups.items() if set(m) & val_videos)
    print(f"\n  {len(groups) - n_val} scenes train / {n_val} scenes val "
          f"(the videos within a scene are viewpoint variety, and never straddle)\n")

    for task, names in CLASSES.items():
        task_root = output / task
        if task_root.exists():
            shutil.rmtree(task_root)

        counts = {"train": Counter(), "val": Counter()}
        images = {"train": 0, "val": 0}
        empties = {"train": 0, "val": 0}

        for video in videos:
            split = "val" if video in val_videos else "train"
            (task_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (task_root / "labels" / split).mkdir(parents=True, exist_ok=True)

            for image, crops_name in scenes[video][chosen[video]]:
                task_dir = label_dirs[crops_name][task]
                # This source was never labelled for this task. Its crops are not
                # "background" -- they are UNKNOWN, so they cannot be a training signal
                # for this task and are dropped from it entirely.
                if task_dir is None:
                    continue

                shutil.copy2(image, task_root / "images" / split / image.name)

                source = task_dir / f"{image.stem}.txt"
                text = source.read_text().strip() if source.exists() else ""
                # An EXPLICIT empty file, not a missing one: this crop IS background and
                # we want the model told so.
                (task_root / "labels" / split / f"{image.stem}.txt").write_text(
                    text + "\n" if text else ""
                )

                images[split] += 1
                if not text:
                    empties[split] += 1
                else:
                    for line in text.split("\n"):
                        counts[split][int(line.split()[0])] += 1

        with (task_root / "data.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                {
                    "path": str(task_root.resolve()),
                    "train": "images/train",
                    "val": "images/val",
                    "names": dict(enumerate(names)),
                },
                file,
                sort_keys=False,
            )

        print(f"{task}:")
        for split in ("train", "val"):
            per_class = ", ".join(
                f"{names[i]} {counts[split][i]}" for i in range(len(names))
            )
            print(
                f"  {split:5s} {images[split]:3d} crops "
                f"({empties[split]:3d} background) | {per_class}"
            )
        print(f"  -> {task_root}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
