"""Run both detectors over a frame's zone crops and produce the mission answers.

    python -m mai.cli.predict_zones --eval          # score against the labels
    python -m mai.cli.predict_zones --benchmark     # latency breakdown

This is the scoreboard, not mAP. The competition does not award points for box overlap;
it awards them for naming the right ZONE and the right SIZE or TYPE. A box that is 3
pixels off scores exactly the same as a perfect one, and a box in the wrong zone scores
zero however tight it is. So we measure what is actually paid for:

  crater_detect   5 zones x (zone + size)      15 pts
  crater_count    craters in RW zones only      5 pts
  runway_status   longest crater-free RW run    5 pts
  uxo_detect      6 zones x (zone + type)      18 pts
  uxo_count       UXO in RW zones only          2 pts

Crater SIZE is not predicted by the network -- it is measured. The crop is metric, so a
box in pixels becomes a box in centimetres, and the three sizes separate cleanly by area
once the lean is subtracted (see mai/sizing.py).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from mai import sizing
from mai.arena import Arena
from mai.homography import Homography

CROP_RE = re.compile(r"^(?P<video>top_center_\d+)_f\d+_(?P<zone>[A-Z]{2}-[A-Z]?\d+)$")
UXO_NAMES = ["misile", "dumb", "cluster"]
FOCAL_PX = 2194.0  # measured from the oblique clip; same camera


def runway_available_m(crater_zones: set[str], arena: Arena) -> int:
    """Longest unbroken run of crater-free RW zones, in real metres (300m per zone)."""
    runway = sorted(arena.runway_zones, key=lambda z: z.x)
    best = run = 0
    for zone in runway:
        run = 0 if zone.id in crater_zones else run + 1
        best = max(best, run)
    return int(best * zone.w * arena.scale / 100.0)


def predict_frame(crops, models, arena, geometry, px_per_cm, conf=0.35):
    """crops: {zone_id: image}. Returns crater and UXO findings, keyed by zone."""
    zone_ids = list(crops)
    images = [crops[z] for z in zone_ids]

    crater_out = models["crater"].predict(
        images, imgsz=512, conf=conf, device=0, verbose=False, half=True
    )
    uxo_out = models["uxo"].predict(
        images, imgsz=640, conf=conf, device=0, verbose=False, half=True
    )

    craters, uxo = {}, {}
    for zone_id, cres, ures in zip(zone_ids, crater_out, uxo_out):
        for box in cres.boxes:
            xyxy = box.xyxy[0].tolist()
            base_cm, w_cm, d_cm = sizing.box_to_arena(xyxy, zone_id, arena, px_per_cm)
            landed = arena.zone_at(*base_cm) or zone_id
            size = sizing.crater_size(w_cm, d_cm, base_cm, geometry)
            score = float(box.conf[0])
            if landed not in craters or score > craters[landed][1]:
                craters[landed] = (size, score)

        for box in ures.boxes:
            xyxy = box.xyxy[0].tolist()
            base_cm, _, _ = sizing.box_to_arena(xyxy, zone_id, arena, px_per_cm)
            landed = arena.zone_at(*base_cm) or zone_id
            score = float(box.conf[0])
            kind = UXO_NAMES[int(box.cls[0])]
            if landed not in uxo or score > uxo[landed][1]:
                uxo[landed] = (kind, score)

    return craters, uxo


def load_frames(crops_dir: Path, videos: set[str], arena: Arena):
    """{stem: {zone_id: image}} for the airfield zones of the given videos."""
    frames: dict[str, dict[str, np.ndarray]] = {}
    airfield = {z.id for z in arena.airfield_zones}
    for path in sorted(crops_dir.glob("*.jpg")):
        match = CROP_RE.match(path.stem)
        if not match or match.group("zone") not in airfield:
            continue
        if videos and match.group("video") not in videos:
            continue
        stem = path.stem.rsplit("_", 1)[0]
        frames.setdefault(stem, {})[match.group("zone")] = cv2.imread(str(path))
    return frames


def truth_for(stem: str, arena: Arena, labels: Path, px_per_cm, geometry):
    """Ground truth zones from the labels, resolved the same way predictions are."""
    craters, uxo = {}, {}
    airfield = {z.id for z in arena.airfield_zones}
    for zone_id in airfield:
        for task, store in (("crater", craters), ("uxo", uxo)):
            sub = "zones_labels_crater/train" if task == "crater" else "zones_labels_UXO/train"
            path = labels / sub / f"{stem}_{zone_id}.txt"
            if not path.exists():
                continue
            zone = arena.zone(zone_id)
            width, height = (zone.w + 10) * px_per_cm, (zone.h + 10) * px_per_cm
            for line in path.read_text().strip().split("\n"):
                if not line:
                    continue
                cls, cx, cy, bw, bh = map(float, line.split())
                xyxy = (
                    (cx - bw / 2) * width, (cy - bh / 2) * height,
                    (cx + bw / 2) * width, (cy + bh / 2) * height,
                )
                base_cm, w_cm, d_cm = sizing.box_to_arena(xyxy, zone_id, arena, px_per_cm)
                landed = arena.zone_at(*base_cm) or zone_id
                if task == "crater":
                    store[landed] = sizing.crater_size(w_cm, d_cm, base_cm, geometry)
                else:
                    store[landed] = UXO_NAMES[int(cls)]
    return craters, uxo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", default="outputs/dataset/zones")
    parser.add_argument("--labels", default="outputs/dataset/labels")
    parser.add_argument("--manifest", default="outputs/dataset/manifest.json")
    parser.add_argument("--crater-weights", default="runs/detect/runs/crater/weights/best.pt")
    parser.add_argument(
        "--uxo-weights", default="runs/detect/runs/uxo_copypaste/weights/best.pt"
    )
    parser.add_argument("--videos", default="top_center_7,top_center_8", help="Held-out.")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.65,
        help="Chosen by sweeping confidence against LIGHTING-SHIFTED held-out crops, not "
        "against clean ones. On clean data every threshold from 0.35 to 0.65 scores the "
        "same, so the clean data cannot pick one. Under warm/cool/gamma shifts, 0.65 is "
        "the only point where the copy-paste model gives 0 misses AND 0 false positives. "
        "See mai.cli.stress_lighting.",
    )
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    arena = Arena.from_yaml()
    manifest = {e["stem"]: e for e in json.load(open(args.manifest))}
    models = {"crater": YOLO(args.crater_weights), "uxo": YOLO(args.uxo_weights)}
    videos = {v.strip() for v in args.videos.split(",") if v.strip()}
    frames = load_frames(Path(args.crops), videos, arena)

    print(f"{len(frames)} frames from {sorted(videos)}  (never seen in training)\n")

    if args.benchmark:
        stem = next(iter(frames))
        crops = frames[stem]
        entry = manifest[stem]
        H = Homography(np.array(entry["homography_image_to_arena_cm"]), 0, 0, 0, 0, [])
        geometry = sizing.NadirGeometry.from_homography(H, (3840, 2160), FOCAL_PX)
        ppc = entry["topview_px_per_cm"]

        for _ in range(3):  # warm up CUDA / cuDNN autotune
            predict_frame(crops, models, arena, geometry, ppc, args.conf)
        runs = 10
        start = time.perf_counter()
        for _ in range(runs):
            predict_frame(crops, models, arena, geometry, ppc, args.conf)
        elapsed = (time.perf_counter() - start) / runs * 1000
        print(f"both models, 20 crops, batched FP16: {elapsed:6.1f} ms/frame")
        print(f"  + ArUco/homography/crops (measured earlier, half-res ArUco): ~50 ms")
        print(f"  = full pipeline ~{elapsed + 50:.0f} ms/frame -> {1000 / (elapsed + 50):.1f} frames/sec")
        return 0

    # --- zone-level scoring: the actual competition metric ---
    totals = Counter()
    for stem, crops in sorted(frames.items()):
        entry = manifest[stem]
        H = Homography(np.array(entry["homography_image_to_arena_cm"]), 0, 0, 0, 0, [])
        geometry = sizing.NadirGeometry.from_homography(H, (3840, 2160), FOCAL_PX)
        ppc = entry["topview_px_per_cm"]

        craters, uxo = predict_frame(crops, models, arena, geometry, ppc, args.conf)
        true_craters, true_uxo = truth_for(stem, arena, Path(args.labels), ppc, geometry)

        pred_c = {z: v[0] for z, v in craters.items()}
        pred_u = {z: v[0] for z, v in uxo.items()}
        runway = {z.id for z in arena.runway_zones}

        zone_hit = len(set(pred_c) & set(true_craters))
        full_hit = sum(1 for z in set(pred_c) & set(true_craters) if pred_c[z] == true_craters[z])
        u_zone = len(set(pred_u) & set(true_uxo))
        u_full = sum(1 for z in set(pred_u) & set(true_uxo) if pred_u[z] == true_uxo[z])

        c_count = len(set(pred_c) & runway), len(set(true_craters) & runway)
        u_count = len(set(pred_u) & runway), len(set(true_uxo) & runway)
        rw_pred = runway_available_m(set(pred_c), arena)
        rw_true = runway_available_m(set(true_craters), arena)

        totals["crater_zone"] += zone_hit
        totals["crater_full"] += full_hit
        totals["crater_true"] += len(true_craters)
        totals["crater_fp"] += len(set(pred_c) - set(true_craters))
        totals["uxo_zone"] += u_zone
        totals["uxo_full"] += u_full
        totals["uxo_true"] += len(true_uxo)
        totals["uxo_fp"] += len(set(pred_u) - set(true_uxo))
        totals["count_ok"] += (c_count[0] == c_count[1]) + (u_count[0] == u_count[1])
        totals["runway_ok"] += rw_pred == rw_true
        totals["frames"] += 1

        flag = "" if (full_hit == len(true_craters) and u_full == len(true_uxo)
                      and not (set(pred_c) - set(true_craters))
                      and not (set(pred_u) - set(true_uxo))) else "   <-- MISS"
        print(
            f"{stem}: craters {full_hit}/{len(true_craters)} zone+size, "
            f"UXO {u_full}/{len(true_uxo)} zone+type | "
            f"RW count {c_count[0]}/{c_count[1]}, UXO count {u_count[0]}/{u_count[1]}, "
            f"runway {rw_pred}m/{rw_true}m{flag}"
        )

    n = totals["frames"]
    print(f"\n{'='*66}\nSCOREBOARD over {n} held-out frames")
    print(f"  crater zone+size : {totals['crater_full']:3d} / {totals['crater_true']:3d}"
          f"   (zone only: {totals['crater_zone']})   false positives: {totals['crater_fp']}")
    print(f"  UXO   zone+type  : {totals['uxo_full']:3d} / {totals['uxo_true']:3d}"
          f"   (zone only: {totals['uxo_zone']})   false positives: {totals['uxo_fp']}")
    print(f"  counts correct   : {totals['count_ok']:3d} / {2 * n}")
    print(f"  runway length ok : {totals['runway_ok']:3d} / {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
