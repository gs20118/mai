"""Time the full per-frame path, so speed decisions are made on numbers.

    python -m mai.cli.benchmark

Inference is NOT the mission bottleneck -- moving files off the drone is. So the point
of being fast is not meeting the 180-second deadline; it is being able to run MANY
frames and vote. Temporal fusion in arena coordinates is the cheapest accuracy in the
project, and every 100ms saved is another frame in the vote.

Reports the breakdown, because the surprise is usually not where you expect: on this
footage ArUco on the full 4K frame costs more than both networks put together.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from mai import aruco, topview, zones
from mai.arena import Arena
from mai.frames import Frame
from mai.perf import restore_opencv_threads
from mai.undistort import CameraProfile, Undistorter

# MUST come after `from ultralytics import YOLO` above: that import calls
# cv2.setNumThreads(1), which makes ArUco 4.4x slower and says nothing about it.
restore_opencv_threads()


def timeit(fn, repeats: int = 10, warmup: int = 3, gpu: bool = False) -> float:
    """Median-of-repeats wall time in ms.

    `gpu=False` for CPU-only work. This matters: CUDA calls are only needed to flush
    async GPU work, and dropping a torch.cuda.synchronize() into a pure-OpenCV timing
    loop drags the CUDA runtime into the measurement. It made ArUco read 246ms when
    the honest figure is 57ms -- a 4x lie, and in the direction that would have sent us
    optimising the wrong stage.

    Median, not mean, so one scheduling hiccup cannot move the number.
    """
    use_cuda = gpu and torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if use_cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(samples))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/raw/top_center_4.mp4")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--half", action="store_true", default=True)
    args = parser.parse_args()

    device = args.device
    print(f"device: {torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'}\n")

    arena = Arena.from_yaml()
    undistorter = Undistorter(CameraProfile.load("video_4k"))
    detector = aruco.build_detector(arena.dictionary)

    capture = cv2.VideoCapture(args.source)
    _, frame_image = capture.read()
    capture.release()

    # --- geometry ---
    half_image = cv2.resize(frame_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    full_ms = timeit(lambda: aruco.detect(frame_image, detector, keep_ids=set(arena.markers)), 5)
    half_ms = timeit(lambda: aruco.detect(half_image, detector, keep_ids=set(arena.markers)), 5)
    n_half = len(aruco.detect(half_image, detector, keep_ids=set(arena.markers)))

    registration = topview.register(
        Frame(frame_image, 0, args.source), arena, undistorter, detector
    )
    airfield = arena.airfield_zones
    crop_ms = timeit(
        lambda: [
            zones.crop_zone(registration.undistorted, z, registration.homography, pad_cm=5.0)[0]
            for z in airfield
        ],
        5,
    )
    crops = [
        zones.crop_zone(registration.undistorted, z, registration.homography, pad_cm=5.0)[0]
        for z in airfield
    ]

    print("GEOMETRY")
    print(f"  ArUco @ 4K (3840x2160)      {full_ms:7.1f} ms")
    print(f"  ArUco @ half (1920x1080)    {half_ms:7.1f} ms   ({n_half}/4 markers still found)")
    print(f"  {len(airfield)} airfield zone crops       {crop_ms:7.1f} ms")
    print()

    # --- networks ---
    print("NETWORKS  (untrained COCO weights: the ARCHITECTURE cost is what we're timing,")
    print("           and that does not change once we fine-tune on our own classes)")
    results = {}
    for name, weights, imgsz in [
        ("crater  yolo11n @512", "yolo11n.pt", 512),
        ("uxo     yolo11s @640", "yolo11s.pt", 640),
    ]:
        model = YOLO(weights)
        # ultralytics' predict() takes "0", but torch's .to() wants "cuda:0".
        model.to("cpu" if device == "cpu" else f"cuda:{device}")
        batched = timeit(
            lambda: model.predict(
                crops, imgsz=imgsz, device=device, half=args.half, verbose=False
            ),
            8,
            gpu=True,
        )
        looped = timeit(
            lambda: [
                model.predict(c, imgsz=imgsz, device=device, half=args.half, verbose=False)
                for c in crops
            ],
            3,
            gpu=True,
        )
        results[name] = batched
        print(f"  {name}   batched {batched:7.1f} ms   |  one-at-a-time {looped:7.1f} ms")

    net_total = sum(results.values())
    best_aruco = half_ms if n_half == 4 else full_ms
    total = best_aruco + crop_ms + net_total

    print()
    print("PER-FRAME TOTAL (half-res ArUco + crops + both networks, batched)")
    print(f"  {best_aruco:.0f} + {crop_ms:.0f} + {net_total:.0f}  =  {total:.0f} ms   "
          f"->  {1000 / total:.1f} frames/sec")
    print()
    print(f"  In a 30-second window while the drone is still flying: "
          f"~{int(30 * 1000 / total)} frames analysed.")
    print("  We need perhaps 5. The surplus is what we spend on VOTING.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
