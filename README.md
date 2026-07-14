# mai — ROKAF AI Hackathon, drone ISR / battle damage assessment

A drone flies a 500×400cm diorama airbase for 180 seconds. We report crater zones
and sizes, facility damage states, and UXO zones and types as JSON to a scoring
dashboard. Full task brief in [`TASK.md`](TASK.md).

Scoring is **per zone ID** (`RW-04`, `TW-A2`, `FA-02`), never per bounding box. A
perfect detection projected into the wrong zone scores zero. So everything rests on
the image → arena-coordinate transform, which is what this repo currently builds.

## Status

Phase 1 — top-view geometry. Complete, and verified against synthetic ground truth.

```
frame → undistort → ArUco → homography → per-zone crops
```

No detection or classification models yet. That decision is deliberately gated on
looking at the zone crops and the resolution report first.

## Quick start

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt && uv pip install -e .

# See the whole pipeline run, without a drone:
.venv/bin/python -m mai.cli.build_topview --synthetic --output outputs/demo
.venv/bin/python -m mai.cli.crop_zones --run outputs/demo

.venv/bin/python -m pytest        # 18 tests, no footage required
```

`opencv-contrib-python`, **not** `opencv-python` — `cv2.aruco` and `cv2.SIFT` live
in contrib, and the two packages conflict.

## With real footage

```bash
# 1. Which ArUco dictionary and IDs did the organizers actually use?
python -m mai.cli.scan_aruco --source data/raw/practice.mp4

# 2. Dial in lens distortion by eye on train footage; freeze it for test footage.
python -m mai.cli.tune_undistort --image data/frames/hover.jpg --profile video_4k
#    ...or headless, if the WSL GUI is unavailable:
python -m mai.cli.tune_undistort --image ... --sweep k1=-0.20:0.05:8

# 3. Register a frame to the arena.
python -m mai.cli.build_topview --source data/raw/hover.mp4 --profile video_4k

# 4. Crop every zone, and find out what is actually resolvable.
python -m mai.cli.crop_zones --run outputs/topview_<timestamp>
```

**Look at `grid_on_source.jpg` first.** It projects the arena grid back onto the
drone's own frame. If those lines land on the real runway and taxiway edges, then
the marker map, the distortion profile, the homography and the band layout are all
correct at once. If they don't, one of them is wrong — and it is usually obvious
which.

## The arena

500×400cm, scale 1:600, in five 80cm bands — 26 zones, defined in
[`configs/arena.yaml`](configs/arena.yaml):

```
y=0    FA-01(160) │ FA-02(180) │ FA-03(160)
y=80   TW-A1 .. TW-A5      (5 × 100cm)
y=160  RW-01 .. RW-10     (10 × 50cm)      one zone = 300m real
y=240  TW-B1 .. TW-B5      (5 × 100cm)
y=320  FA-04(160) │ FA-05(180) │ FA-06(160)
```

Arena coordinates are **centimetres**, origin top-left, +x right, +y down. Display
pixels are a separate thing, and only the visualisation layer converts to them.

## What the geometry already tells us

Two findings from the synthetic model, before any real footage:

**A single whole-arena shot may not carry the UXO mission.** At ~300cm the ground
sample distance is ~2.3mm/px, so a 28mm cluster munition is only **~12px** — right
at the edge of detectability, and telling it apart from a 44mm shell at that size is
harder still. Craters are comfortable (43–74px). This argues for a second, lower
pass over the runway strip.

**The 5:4 arena on a 16:9 sensor costs ~30% of the resolution you'd expect.**
Framing the whole arena is limited by its 400cm *depth* against 2160px of frame
height, not by its 500cm width against 3840px. Yaw costs more still, so squaring the
drone up to the arena is worth real pixels.

## Layout

```
configs/arena.yaml     zones, bands, ArUco id→world map   ← single source of truth
configs/camera.yaml    per-capture-mode intrinsics + distortion
src/mai/arena.py       Zone model, world_cm → zone_id lookup
src/mai/undistort.py   barrel correction, resolution-preserving
src/mai/aruco.py       detection tuned for SMALL markers (~40px at altitude)
src/mai/homography.py  image px → arena cm, with a reprojection-error gate
src/mai/zones.py       native-resolution zone crops + the GSD report
src/mai/synthetic.py   fake arena + fake camera, so geometry is testable with no drone
legacy/                the original five scripts, kept for reference
docs/capture_checklist.md   ← read before the practice session
```

## Next

[`docs/capture_checklist.md`](docs/capture_checklist.md). Every `aruco:` value in
`arena.yaml` is a placeholder until measured on-site, and the transfer-chain timing
governs the entire flight plan.
