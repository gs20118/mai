"""Dial in the lens distortion by eye, then freeze it.

    python -m mai.cli.tune_undistort --image data/frames/hover.jpg --profile video_4k
    python -m mai.cli.tune_undistort --image ... --sweep k1=-0.20:0.05:8   # headless
    python -m mai.cli.tune_undistort --profile video_4k --set k1=-0.03 --set k2=0.01

We have no chessboard, so this is a tuning tool, not a calibration tool. That is
fine: the arena is a plane, and a planar homography absorbs any linear camera
matrix. Only the NONLINEAR terms have to be right, because a homography cannot
represent radial distortion. So the tuner has exactly one job -- make straight
lines in the arena come out straight.

WORKFLOW: tune on train footage, save the profile, then apply the identical
profile to test footage without touching it again.

STRAIGHTNESS PROBE: click two points along something you know is physically
straight (a runway edge is ideal, since it runs the full 500cm). The tool draws
the true straight line between them. If the real edge bows away from that line,
distortion remains. Tune until it doesn't.

The GUI needs a display (WSLg on WSL2). If cv2.imshow fails, use --sweep, which
renders a labelled montage of candidates you can compare with any image viewer,
then commit the winner with --set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from mai.undistort import DEFAULT_CAMERA_CONFIG, CameraProfile, Undistorter

# Trackbars are integer-only, so each parameter maps onto a slider range.
SLIDERS = {
    "k1": (-0.5, 0.5),
    "k2": (-0.3, 0.3),
    "k3": (-0.1, 0.1),
    "p1": (-0.02, 0.02),
    "p2": (-0.02, 0.02),
}
STEPS = 1000
PREVIEW_WIDTH = 1500


def _preview(image: np.ndarray, profile: CameraProfile, probe: list[tuple[int, int]]):
    corrected = Undistorter(profile)(image)
    scale = PREVIEW_WIDTH / corrected.shape[1]
    view = cv2.resize(
        corrected, (PREVIEW_WIDTH, int(round(corrected.shape[0] * scale)))
    )

    if len(probe) == 2:
        # The straight line the arena edge SHOULD lie on, if distortion is gone.
        cv2.line(view, probe[0], probe[1], (0, 0, 255), 1, cv2.LINE_AA)
        for point in probe:
            cv2.circle(view, point, 4, (0, 0, 255), -1)
    for point in probe[:1] if len(probe) == 1 else []:
        cv2.circle(view, point, 4, (0, 255, 255), -1)

    banner = (
        f"k1={profile.k1:+.4f} k2={profile.k2:+.4f} k3={profile.k3:+.4f} "
        f"p1={profile.p1:+.4f} p2={profile.p2:+.4f}"
    )
    cv2.putText(view, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(view, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(
        view,
        "click 2 pts on a known-straight edge | s=save  r=reset probe  q=quit",
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        view,
        "click 2 pts on a known-straight edge | s=save  r=reset probe  q=quit",
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 255, 200),
        1,
    )
    return view


def run_gui(image: np.ndarray, profile: CameraProfile, config_path: Path) -> int:
    window = "tune_undistort"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, PREVIEW_WIDTH, 900)

    values = {name: getattr(profile, name) for name in SLIDERS}
    probe: list[tuple[int, int]] = []

    def to_slider(name: str, value: float) -> int:
        low, high = SLIDERS[name]
        return int(round((value - low) / (high - low) * STEPS))

    def from_slider(name: str, position: int) -> float:
        low, high = SLIDERS[name]
        return low + (high - low) * position / STEPS

    for name in SLIDERS:
        cv2.createTrackbar(
            name, window, to_slider(name, values[name]), STEPS, lambda _v: None
        )

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(probe) >= 2:
                probe.clear()
            probe.append((x, y))

    cv2.setMouseCallback(window, on_mouse)

    current = profile
    while True:
        for name in SLIDERS:
            values[name] = from_slider(name, cv2.getTrackbarPos(name, window))
        current = CameraProfile(
            name=profile.name,
            image_size=profile.image_size,
            fx=profile.fx,
            fy=profile.fy,
            cx=profile.cx,
            cy=profile.cy,
            alpha=profile.alpha,
            **values,
        )
        cv2.imshow(window, _preview(image, current, probe))

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            probe.clear()
        if key == ord("s"):
            current.save(config_path)
            print(f"saved profile '{current.name}' to {config_path}")
            print(
                f"  k1={current.k1:+.5f} k2={current.k2:+.5f} k3={current.k3:+.5f} "
                f"p1={current.p1:+.5f} p2={current.p2:+.5f}"
            )

    cv2.destroyAllWindows()
    return 0


def run_sweep(image: np.ndarray, profile: CameraProfile, spec: str, output: Path) -> int:
    """Render a labelled montage across one parameter, for when there is no display."""
    name, _, rng = spec.partition("=")
    name = name.strip()
    if name not in SLIDERS:
        raise SystemExit(f"--sweep parameter must be one of {sorted(SLIDERS)}, got {name!r}")
    low, high, count = (float(part) for part in rng.split(":"))

    tiles = []
    for value in np.linspace(low, high, int(count)):
        candidate = CameraProfile(
            **{**profile.__dict__, name: float(value)}
        )
        corrected = Undistorter(candidate)(image)
        scale = 640 / corrected.shape[1]
        tile = cv2.resize(corrected, (640, int(round(corrected.shape[0] * scale))))
        label = f"{name}={value:+.4f}"
        cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
        tiles.append(tile)

    height = max(tile.shape[0] for tile in tiles)
    tiles = [
        cv2.copyMakeBorder(tile, 0, height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        for tile in tiles
    ]
    columns = min(len(tiles), 4)
    rows = [
        np.hstack(tiles[index : index + columns]) for index in range(0, len(tiles), columns)
    ]
    width = max(row.shape[1] for row in rows)
    rows = [
        cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT)
        for row in rows
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.vstack(rows))
    print(f"swept {name} over [{low}, {high}] in {int(count)} steps -> {output}")
    print(f"Pick the straightest, then commit it:")
    print(f"  python -m mai.cli.tune_undistort --profile {profile.name} --set {name}=<value>")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=None, help="A frame to tune against.")
    parser.add_argument("--profile", default="video_4k")
    parser.add_argument("--camera", default=str(DEFAULT_CAMERA_CONFIG))
    parser.add_argument("--sweep", default=None, help="Headless, e.g. k1=-0.2:0.05:8")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="K=V",
        help="Write a value directly, e.g. --set k1=-0.03. Repeatable.",
    )
    parser.add_argument("--output", default="outputs/undistort_sweep.jpg")
    args = parser.parse_args()

    config_path = Path(args.camera)
    profile = CameraProfile.load(args.profile, config_path)

    if args.set:
        updates = {}
        for assignment in args.set:
            key, _, value = assignment.partition("=")
            updates[key.strip()] = float(value)
        profile = CameraProfile(**{**profile.__dict__, **updates})
        profile.save(config_path)
        print(f"saved profile '{profile.name}' to {config_path}: {updates}")
        return 0

    if args.image is None:
        parser.error("--image is required unless you are using --set")

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"could not read {args.image}")

    height, width = image.shape[:2]
    if (width, height) != profile.image_size:
        print(
            f"note: frame is {width}x{height} but profile '{profile.name}' was tuned at "
            f"{profile.image_size[0]}x{profile.image_size[1]}; intrinsics will be rescaled."
        )

    if args.sweep:
        return run_sweep(image, profile, args.sweep, Path(args.output))

    try:
        return run_gui(image, profile, config_path)
    except cv2.error as error:
        print(f"GUI unavailable ({error}).")
        print("Fall back to headless mode:")
        print(
            f"  python -m mai.cli.tune_undistort --image {args.image} "
            f"--profile {args.profile} --sweep k1=-0.2:0.05:8"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
