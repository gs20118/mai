# mai — ROKAF AI Hackathon, drone ISR / battle damage assessment

A drone flies a 500×400cm diorama airbase for 180 seconds. We report crater zones and
sizes, facility damage states, and UXO zones and types as JSON to a scoring dashboard.
Full task brief in [`TASK.md`](TASK.md).

Scoring is **per zone ID** (`RW-04`, `TW-A2`, `FA-02`), never per bounding box. A perfect
detection projected into the wrong zone scores zero — so everything rests on the image →
arena-coordinate transform, and that is where most of the engineering went.

## Status

Crater (25 pts) and UXO (20 pts) run **end to end, video in → dashboard JSON out**, and
score 18/18 zones on a genuinely held-out scene. Facility damage (18 pts) and the LLM
report (7 pts) are next; the pipeline is structured to slot them in as two more per-frame
steps and two more JSON writers.

```
video → sharp frames → undistort → ArUco+ticks → homography → 20 zone crops
      → crater model + UXO model → vote across frames → per-zone answers → JSON
```

## The one command

```bash
python -m mai.cli.run_mission --source data/test/top_center_1.mp4
```

Writes the six dashboard files (`start`, `crater_detect`, `crater_count`, `runway_status`,
`uxo_detect`, `uxo_count`) plus `answers_on_arena.jpg` to `outputs/mission/<video>/`. On
the real test video, on the competition-equivalent laptop (RTX 4070): **260 ms/frame**,
20 frames in ~5s, every zone correct.

## How registration survives the real world

The homography needs four corner ArUco markers, but on real footage you do not always get
four. `topview.register_robust` escalates rather than giving up:

1. **Half-resolution ArUco.** The markers are 102px, so at half scale they are still 51px
   — clear of the ~40px detection floor — and the work drops ~4×, from ~145ms to ~40ms.
2. **Full-resolution retry** if the fast path comes up short, before concluding a marker
   is truly missing.
3. **Markers + printed ticks.** The arena prints yellow ticks on its border at every
   sector boundary, at known arena coordinates. Each tick fixes **one** coordinate — a
   linear constraint that drops straight into the homography DLT beside the marker
   corners. This registers a frame from as few as **two** visible markers.

That third level is not theoretical: in the multi-video test one clip had a corner marker
**clipped by the frame edge** (only 2–3 markers ever visible) and would otherwise have
registered zero frames and scored zero. With ticks it registered cleanly, and the tick
residual doubles as an **independent** accuracy check — the markers cannot vouch for the
corner they cannot see. Same machinery also registers the 45° oblique view, where a
building occludes a marker outright.

## Voting is why speed matters

Inference is not the mission bottleneck — moving files off the drone is. So the point of a
fast per-frame path is not to finish in time; it is to analyse **many** frames and let the
zones vote. Every detection is projected into arena coordinates and accumulated across
frames; a zone is reported only if it clears a vote threshold, one vote per zone per frame.
On the first real video this caught the old crater model firing intermittently in a
UXO zone (6/20 frames) and **voted it out** — a single-frame pipeline would have shipped a
sixth crater and corrupted the count and runway length with it.

Every answer is derived from **one** state object, so `crater_detect`, `crater_count` and
`runway_status` can never contradict each other.

## The two detectors

Deliberately unequal, because the tasks are:

| | model | why |
|---|---|---|
| **crater** | `yolo11n` @ 512 | one class, 50–90px, black on grey — easy |
| **UXO** | `yolo11s` @ 640 | three classes, smallest an ~18px ball, hard missile/cluster split |

Sizing them separately means the pair costs `n + s`, not `2 × s`.

**Crater size is measured, not predicted.** The crops are metric (we know mm/px), so a box
in pixels becomes a box in cm, and the three sizes separate cleanly by area at 155 / 293
cm². A lean correction subtracts the parallax a standing object throws toward the arena
edge — see [`src/mai/sizing.py`](src/mai/sizing.py).

**Copy-paste augmentation respects the physics.** UXO stand upright, so a nadir bbox
carries a lean that encodes height — the cue separating the classes at these sizes. Objects
are pasted only where that lean stays correct (same zone across scenes, or mirrored through
the drone's nadir), never warped, and rescaled to the destination's px/cm. It made no
difference on clean data but held recall at 1.000 under lighting shift, where the baseline
lost a quarter of objects.

### Data discipline

The dataset splits **by scene, not by video** — 13 videos contained only 6 distinct object
layouts (the drone was re-flown over an unchanged arena), so a naïve by-video split leaked
the same objects into train and val and faked a perfect score. `build_yolo_dataset`
fingerprints each video by what-is-where, groups the duplicates, and refuses to let one
straddle the split. Sources are task-aware: a source labelled for UXO only never enters the
crater dataset (unlabelled ≠ background).

## Results (held-out scenes, never in training)

```
crater   18/18 zones          0 missed   0 false positives
UXO      18/18 zones + type   0 missed   0 false positives   (val recall 1.000, mAP50 0.995)
lighting stress (realistic ±20% / gamma 0.8–1.25 / warm-cool):  recall 1.000 everywhere
```

## The arena

500×400cm, scale 1:600, five 80cm bands — 26 zones, in
[`configs/arena.yaml`](configs/arena.yaml), the single source of truth:

```
y=0    FA-01(160) │ FA-02(180) │ FA-03(160)
y=80   TW-A1 .. TW-A5      (5 × 100cm)
y=160  RW-01 .. RW-10     (10 × 50cm)      one zone = 300m real
y=240  TW-B1 .. TW-B5      (5 × 100cm)
y=320  FA-04(160) │ FA-05(180) │ FA-06(160)
```

Arena coordinates are **centimetres**, origin top-left. The ArUco markers are inset ~11cm
from the corners (measured from the arena's own printed ticks by
`mai.cli.calibrate_arena`, not assumed) — assuming they sat on the corners would skew every
zone by up to 11cm.

## Setup

```bash
uv venv --python 3.12
uv pip install -r requirements.txt && uv pip install -e .
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pytest        # 55 tests
```

`opencv-contrib-python`, **not** `opencv-python` — `cv2.aruco` and `cv2.SIFT` live in
contrib, and the two packages conflict. Torch must match the driver's CUDA (cu126 here);
an over-new build silently falls back to CPU.

## Rebuild from scratch

```bash
# 1. dataset from labelled crops (splits by SCENE; zones_3 is UXO-only)
python -m mai.cli.build_yolo_dataset --output data/yolo
python -m mai.cli.build_copypaste --task uxo --per-image 4

# 2. train (batch 16, AMP + RAM cache ≈ 20s/epoch on a 4070)
python -m mai.cli.train_detector --task crater --name crater2
python -m mai.cli.train_detector --task uxo --data data/yolo/uxo_cp/data.yaml --name uxo2_copypaste

# 3. check the honest numbers, and the operating point under lighting shift
python -m mai.cli.stress_lighting --weights runs/detect/runs/uxo2_copypaste/weights/best.pt

# 4. run the mission
python -m mai.cli.run_mission --source data/test/top_center_1.mp4
```

## Layout

```
configs/arena.yaml        zones, bands, measured ArUco map, target sizes  ← source of truth
configs/camera.yaml       per-capture-mode intrinsics (distortion ≈ 0 on this drone)
src/mai/arena.py          Zone model, world_cm → zone_id lookup
src/mai/aruco.py          detection tuned for small markers (~40px at altitude)
src/mai/ticks.py          the printed border ticks, as one-coordinate fiducials
src/mai/homography.py     image px → arena cm; solve() and solve_constrained() (markers+ticks)
src/mai/topview.py        register_robust: the 3-level registration escalation
src/mai/pose.py           camera pose from the homography (for the oblique view's lean)
src/mai/zones.py          native-resolution zone crops + GSD report
src/mai/sizing.py         crater size from metric area, with lean correction
src/mai/copypaste.py      physics-respecting UXO augmentation
src/mai/localize.py       SIFT-to-nadir-map registration (markerless / oblique fallback)
src/mai/synthetic.py      fake arena + camera, so the geometry is testable with no drone
src/mai/cli/run_mission.py   the whole thing, video → dashboard JSON
legacy/                   the original five scripts, kept for reference
docs/capture_checklist.md ← read before any practice session
```

## Open questions for the organizers (blocking the last mile)

Neither is a code problem, and both gate points we cannot otherwise secure:

- **Exact JSON schema** — especially how facilities are keyed (FA zone ID vs building
  name). Getting this wrong forfeits 18 points with every metric still green.
- **Dashboard transmission protocol** — TCP / HTTP / file drop, host, port, framing. We
  write the correct files; we need the wire contract to deliver them.
