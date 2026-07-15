"""Train the crater and UXO detectors.

    python -m mai.cli.train_detector --task crater
    python -m mai.cli.train_detector --task uxo

Two models, deliberately unequal, because the two problems are unequal:

  CRATER -- yolo11n @ 512.  One class, targets 50-90px, black on grey asphalt. Nano is
            ample (6.5 GFLOPs) and there is no reason to pay more.
  UXO    -- yolo11s @ 640.  Three classes, the smallest an 18px ball, and a genuinely
            hard missile-vs-cluster distinction. Small (21.5 GFLOPs) earns its keep.

Sizing them separately means the pair costs roughly n + s rather than 2 x s, which is
most of the reason we can afford two models at all.

MOSAIC IS OFF FOR UXO. It is Ultralytics' strongest default augmentation and it works by
tiling four images down into one -- which shrinks every object. That is the last thing an
18px ball needs. Craters are big enough not to care, so they keep it.

The dataset is small (240 train crops from 6 videos), so we run many epochs with early
stopping rather than few. Colour and brightness augmentation is pushed hard: the rules
warn that venue lighting will differ on the day, and lighting is the one shift we know is
coming.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PRESETS = {
    "crater": {
        "model": "yolo11n.pt",
        "imgsz": 512,
        "epochs": 120,
        "batch": 32,
        "mosaic": 0.5,
    },
    "uxo": {
        "model": "yolo11s.pt",
        "imgsz": 640,
        "epochs": 150,
        "batch": 16,
        "mosaic": 0.0,  # see module docstring: it shrinks small objects
    },
}

# Lighting augmentation, deliberately GENTLE. The venue lighting is close to the training
# lighting (same hall, same lamps), so we do not need the heavy shifts that would harden
# the model against a different room -- and heavy shifts have a real cost here: pushing
# brightness down 50% turns a dark prop on dark asphalt into a featureless blob, throwing
# away the very detail an 18px object cannot spare. TASK.md 10.4 warns lighting *may*
# change, so we keep a little (not zero), but far less than the defaults.
HSV_S = 0.4  # was 0.7
HSV_V = 0.3  # was 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(PRESETS), required=True)
    parser.add_argument("--data", default=None, help="Defaults to data/yolo/<task>/data.yaml")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument(
        "--patience",
        type=int,
        default=40,
        help="Early-stop after this many epochs with no val improvement. Lower is faster.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    from ultralytics import YOLO

    preset = PRESETS[args.task]
    data = Path(args.data or f"data/yolo/{args.task}/data.yaml").resolve()
    if not data.exists():
        raise SystemExit(f"{data} not found -- run `python -m mai.cli.build_yolo_dataset` first")

    model = YOLO(args.model or preset["model"])
    model.train(
        data=str(data),
        epochs=args.epochs or preset["epochs"],
        imgsz=args.imgsz or preset["imgsz"],
        batch=args.batch or preset["batch"],
        device=args.device,
        project=args.project,
        name=args.name or args.task,
        exist_ok=True,
        patience=args.patience,
        # --- speed. The dataset is small enough to live in RAM, which removes disk I/O as
        #     the per-epoch bottleneck; AMP runs the forward/backward in FP16 on the 4070's
        #     tensor cores; persistent workers avoid re-spawning the dataloader each epoch.
        cache="ram",
        amp=True,
        workers=8,
        # Small dataset: let it look at the data many times, and stop when it stalls.
        close_mosaic=10,
        mosaic=preset["mosaic"],
        # Gentle lighting jitter: the venue is close to training, and heavy shifts destroy
        # detail on small dark objects. See HSV_S / HSV_V above.
        hsv_h=0.015,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        # The arena has no canonical "up" from above, and objects lean in every
        # direction depending on where they sit, so flips are free and safe variety.
        fliplr=0.5,
        flipud=0.5,
        degrees=10.0,
        translate=0.15,
        scale=0.3,
        # Never distort the aspect ratio: object SHAPE is what separates a round ball
        # from a long cluster, and it is also what the size classifier measures.
        shear=0.0,
        perspective=0.0,
        verbose=True,
    )
    print(f"\nweights -> {Path(args.project) / (args.name or args.task) / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
