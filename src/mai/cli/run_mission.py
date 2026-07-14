"""The whole mission, end to end: a drone video in, dashboard JSON out.

    python -m mai.cli.run_mission --source data/test/top_center_1.mp4

    video -> sharp frames -> ArUco -> homography -> 20 airfield zone crops
          -> crater model + UXO model -> vote across frames -> zone answers -> JSON

VOTING IS THE POINT OF BEING FAST. A single frame is one opinion; the drone gives us
hundreds. Inference costs ~190ms/frame, so in the seconds we have spare we can register
many frames, project every detection into ARENA COORDINATES, and let the zones vote.
Single-frame flukes -- a scorch mark read as a crater, a missed ball in one blurred
frame -- do not survive a vote. This is the cheapest accuracy in the project, and it is
why the latency work mattered.

Every answer is derived from ONE state object, so crater_count, runway_status and the
crater_detect list can never contradict each other. Two files disagreeing about the same
craters would lose points twice over.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from mai import aruco, frames, homography, sizing, topview, viz, zones
from mai.arena import Arena
from mai.undistort import CameraProfile, Undistorter

UXO_NAMES = ["misile", "dumb", "cluster"]
FOCAL_PX = 2194.0  # measured from the oblique clip; same camera, same zoom


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--output",
        default=None,
        help="Defaults to outputs/mission/<video name>/, so runs never overwrite each other.",
    )
    parser.add_argument("--mission-code", default="LKUSDC80")
    parser.add_argument("--crater-weights", default="runs/detect/runs/crater/weights/best.pt")
    parser.add_argument(
        "--uxo-weights", default="runs/detect/runs/uxo_copypaste/weights/best.pt"
    )
    parser.add_argument("--conf", type=float, default=0.65)
    parser.add_argument("--sample-sec", type=float, default=0.2)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--max-rms-cm", type=float, default=1.0)
    parser.add_argument(
        "--vote",
        type=float,
        default=0.5,
        help="A zone counts as occupied if it is seen in this fraction of frames.",
    )
    parser.add_argument("--aruco-scale", type=float, default=0.5)
    args = parser.parse_args()

    from ultralytics import YOLO

    arena = Arena.from_yaml()
    undistorter = Undistorter(CameraProfile.load("video_4k"))
    detector = aruco.build_detector(arena.dictionary)
    models = {"crater": YOLO(args.crater_weights), "uxo": YOLO(args.uxo_weights)}
    airfield = arena.airfield_zones
    output = Path(args.output or f"outputs/mission/{Path(args.source).stem}")
    output.mkdir(parents=True, exist_ok=True)

    # Warm up on the SHAPES the mission will actually feed. cuDNN autotunes per input
    # geometry, and the runway crops (~299x449) and taxiway crops (~549x449) are different
    # shapes -- warm only one of them and the first real frame still pays a ~3.3s tuning
    # cost, which then gets averaged into the per-frame latency and, on a 5-frame video,
    # quadruples it. The reported number would be an artefact of our own instrumentation.
    warm = [np.zeros((449, 299, 3), np.uint8) for _ in arena.runway_zones]
    warm += [np.zeros((449, 549, 3), np.uint8) for _ in arena.taxiway_zones]
    for name, imgsz in (("crater", 512), ("uxo", 640)):
        for _ in range(2):
            models[name].predict(warm, imgsz=imgsz, device=0, verbose=False, half=True)

    print(f"source: {args.source}")
    print(f"models: crater={Path(args.crater_weights).parts[-3]} "
          f"uxo={Path(args.uxo_weights).parts[-3]} @ conf {args.conf}\n")

    # --- votes, accumulated in ARENA coordinates across frames ---
    crater_votes: dict[str, list] = defaultdict(list)
    uxo_votes: dict[str, list] = defaultdict(list)
    used = 0
    rejected = 0
    timings = defaultdict(float)
    levels: dict[tuple[int, bool], int] = defaultdict(int)
    reasons: dict[str, int] = defaultdict(int)
    best_frame = None

    for frame in frames.iter_frames(args.source, sample_sec=args.sample_sec):
        if used >= args.max_frames:
            break

        t0 = time.perf_counter()
        try:
            undistorted, solved, n_markers, n_ticks = topview.register_robust(
                frame, arena, undistorter, detector, args.aruco_scale
            )
        except homography.HomographyError as error:
            rejected += 1
            reasons[str(error)[:60]] += 1
            continue
        # A 4-marker fit is checked by its own reprojection error; a tick-rescued fit is
        # already checked against the printed boundaries inside register_robust.
        if n_ticks == 0 and solved.rms_cm > args.max_rms_cm:
            rejected += 1
            reasons[f"reprojection RMS > {args.max_rms_cm}cm"] += 1
            continue
        levels[(n_markers, n_ticks > 0)] += 1
        t1 = time.perf_counter()

        px_per_cm = topview.native_px_per_cm(arena, solved)
        geometry = sizing.NadirGeometry.from_homography(
            solved, (undistorted.shape[1], undistorted.shape[0]), FOCAL_PX
        )
        crops = {z.id: zones.crop_zone(undistorted, z, solved, pad_cm=5.0)[0] for z in airfield}
        t2 = time.perf_counter()

        zone_ids = list(crops)
        images = [crops[z] for z in zone_ids]
        crater_out = models["crater"].predict(
            images, imgsz=512, conf=args.conf, device=0, verbose=False, half=True
        )
        uxo_out = models["uxo"].predict(
            images, imgsz=640, conf=args.conf, device=0, verbose=False, half=True
        )
        t3 = time.perf_counter()

        # ONE vote per zone per frame, the most confident. A frame that fires twice in a
        # zone must not get two votes -- the arena allows at most one object per zone, and
        # letting a frame vote twice is what produced "seen in 105% of frames".
        frame_craters: dict[str, tuple[str, float]] = {}
        frame_uxo: dict[str, tuple[str, float]] = {}
        for zone_id, cres, ures in zip(zone_ids, crater_out, uxo_out):
            for box in cres.boxes:
                base, w_cm, d_cm = sizing.box_to_arena(
                    box.xyxy[0].tolist(), zone_id, arena, px_per_cm
                )
                landed = arena.zone_at(*base) or zone_id
                score = float(box.conf[0])
                if landed not in frame_craters or score > frame_craters[landed][1]:
                    frame_craters[landed] = (
                        sizing.crater_size(w_cm, d_cm, base, geometry), score
                    )
            for box in ures.boxes:
                base, _, _ = sizing.box_to_arena(
                    box.xyxy[0].tolist(), zone_id, arena, px_per_cm
                )
                landed = arena.zone_at(*base) or zone_id
                score = float(box.conf[0])
                if landed not in frame_uxo or score > frame_uxo[landed][1]:
                    frame_uxo[landed] = (UXO_NAMES[int(box.cls[0])], score)

        for landed, vote in frame_craters.items():
            crater_votes[landed].append(vote)
        for landed, vote in frame_uxo.items():
            uxo_votes[landed].append(vote)

        timings["register"] += t1 - t0
        timings["crop"] += t2 - t1
        timings["detect"] += t3 - t2
        used += 1
        if best_frame is None:
            best_frame = (undistorted, solved, px_per_cm, frame.index)

    if not used:
        print(f"No frame could be registered ({rejected} rejected):")
        for reason, count in reasons.items():
            print(f"    {count}x  {reason}")
        return 1

    print(f"{used} frames registered and analysed ({rejected} rejected)")
    for (n_markers, used_ticks), count in sorted(levels.items()):
        how = f"{n_markers} markers" + (" + printed ticks" if used_ticks else "")
        print(f"    {count:2d} frames via {how}")
    print()

    # --- resolve the votes ---
    threshold = max(1, int(round(args.vote * used)))

    def resolve(votes):
        out = {}
        for zone_id, entries in votes.items():
            if len(entries) < threshold:
                continue
            weight = defaultdict(float)
            for label, conf in entries:
                weight[label] += conf
            label = max(weight, key=weight.get)
            out[zone_id] = (label, len(entries) / used, weight[label] / len(entries))
        return out

    craters = resolve(crater_votes)
    uxo = resolve(uxo_votes)

    runway = {z.id for z in arena.runway_zones}
    order = {z.id: (z.y, z.x) for z in arena.zones}

    print("CRATERS")
    for zone_id in sorted(craters, key=lambda z: order[z]):
        size, seen, conf = craters[zone_id]
        print(f"  {zone_id:6s} {size:7s}  seen in {seen:5.0%} of frames, mean conf {conf:.2f}")
    dropped = {z: v for z, v in crater_votes.items() if z not in craters}
    for zone_id, entries in sorted(dropped.items(), key=lambda kv: order[kv[0]]):
        print(f"  {zone_id:6s} {'--':7s}  VOTED OUT (only {len(entries)}/{used} frames)")

    print("\nUXO")
    for zone_id in sorted(uxo, key=lambda z: order[z]):
        kind, seen, conf = uxo[zone_id]
        print(f"  {zone_id:6s} {kind:7s}  seen in {seen:5.0%} of frames, mean conf {conf:.2f}")
    dropped = {z: v for z, v in uxo_votes.items() if z not in uxo}
    for zone_id, entries in sorted(dropped.items(), key=lambda kv: order[kv[0]]):
        print(f"  {zone_id:6s} {'--':7s}  VOTED OUT (only {len(entries)}/{used} frames)")

    # --- the mission answers, all derived from one state so they cannot disagree ---
    crater_rw = sorted(set(craters) & runway, key=lambda z: order[z])
    uxo_rw = sorted(set(uxo) & runway, key=lambda z: order[z])

    runway_sorted = sorted(arena.runway_zones, key=lambda z: z.x)
    best = run = 0
    for zone in runway_sorted:
        run = 0 if zone.id in craters else run + 1
        best = max(best, run)
    available_m = int(best * 50 * arena.scale / 100)

    status = (
        ("정상", "사용 가능") if available_m >= 2100
        else ("제한 운용", "제한적 사용 가능") if available_m >= 1500
        else ("비상 운용", "사용 가능 여부 검토") if available_m >= 900
        else ("운용 불가", "사용 불가, 폐쇄")
    )

    payload = {
        "start.json": {"mission_code": args.mission_code},
        "crater_detect.json": {
            "mission_code": args.mission_code,
            "crater_detect": [
                {"zone": z, "size": craters[z][0]}
                for z in sorted(craters, key=lambda z: order[z])
            ],
        },
        "crater_count.json": {"mission_code": args.mission_code, "crater_count": len(crater_rw)},
        "runway_status.json": {"mission_code": args.mission_code, "runway_status": available_m},
        "uxo_detect.json": {
            "mission_code": args.mission_code,
            "uxo_detect": [
                {"zone": z, "type": uxo[z][0]} for z in sorted(uxo, key=lambda z: order[z])
            ],
        },
        "uxo_count.json": {"mission_code": args.mission_code, "uxo_count": len(uxo_rw)},
    }
    for name, body in payload.items():
        (output / name).write_text(json.dumps(body, ensure_ascii=False, indent=2))

    print(f"\n{'='*70}")
    print(f"crater_detect : {len(craters)} zones   {[z for z in sorted(craters, key=lambda z: order[z])]}")
    print(f"crater_count  : {len(crater_rw)}  (runway only: {crater_rw})")
    print(f"runway_status : {available_m} m  -> {status[0]} / {status[1]}")
    print(f"uxo_detect    : {len(uxo)} zones   {[z for z in sorted(uxo, key=lambda z: order[z])]}")
    print(f"uxo_count     : {len(uxo_rw)}  (runway only: {uxo_rw})")

    # --- QA image: the answers drawn on the arena, so a human can check them ---
    undistorted, solved, px_per_cm, index = best_frame
    warped = topview.warp(undistorted, arena, solved, px_per_cm)
    overlay = viz.draw_zone_grid_on_topview(warped, arena, px_per_cm)
    for zone_id, (size, _, _) in craters.items():
        zone = arena.zone(zone_id)
        p = (np.array(zone.center) * px_per_cm).astype(int)
        cv2.putText(overlay, f"CRATER {size}", (p[0] - 90, p[1] + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(overlay, f"CRATER {size}", (p[0] - 90, p[1] + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 220, 255), 2, cv2.LINE_AA)
    for zone_id, (kind, _, _) in uxo.items():
        zone = arena.zone(zone_id)
        p = (np.array(zone.center) * px_per_cm).astype(int)
        cv2.putText(overlay, f"UXO {kind}", (p[0] - 80, p[1] - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(overlay, f"UXO {kind}", (p[0] - 80, p[1] - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 255, 120), 2, cv2.LINE_AA)
    cv2.imwrite(str(output / "answers_on_arena.jpg"), overlay)

    per = {k: v / used * 1000 for k, v in timings.items()}
    print(f"\nLATENCY per frame: register {per['register']:.0f}ms + crop {per['crop']:.0f}ms "
          f"+ detect {per['detect']:.0f}ms = {sum(per.values()):.0f}ms "
          f"({1000 / sum(per.values()):.1f} fps)")
    print(f"\n  -> {output}   (check answers_on_arena.jpg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
