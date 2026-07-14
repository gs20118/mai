"""Crater sizing, the lean correction, and the split that must never leak.

Crater size is worth 15 of the 25 crater points and it is not predicted by the network --
it is measured off a metric crop. So the arithmetic here is scored directly, and it gets
tested directly.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from mai import pose as posemod
from mai import sizing
from mai.arena import Arena

ALTITUDE_CM = 446.0
NADIR_CM = (245.0, 194.0)


@pytest.fixture(scope="module")
def arena() -> Arena:
    return Arena.from_yaml()


@pytest.fixture(scope="module")
def geometry() -> sizing.NadirGeometry:
    return sizing.NadirGeometry(
        nadir_cm=np.array(NADIR_CM), altitude_cm=ALTITUDE_CM
    )


def test_lean_is_zero_under_the_drone_and_grows_outward(geometry):
    assert np.allclose(geometry.lean_cm(NADIR_CM, 3.0), [0.0, 0.0])

    near = np.linalg.norm(geometry.lean_cm((300.0, 194.0), 3.0))
    far = np.linalg.norm(geometry.lean_cm((490.0, 194.0), 3.0))
    assert 0 < near < far

    # Similar triangles: a 3cm object 245cm off nadir, seen from 446cm up.
    assert far == pytest.approx(3.0 * 245.0 / ALTITUDE_CM, rel=0.02)


def test_lean_points_away_from_the_drone(geometry):
    """A crater to the RIGHT of the nadir leans further right, never back toward it."""
    right = geometry.lean_cm((450.0, 194.0), 3.0)
    left = geometry.lean_cm((50.0, 194.0), 3.0)
    assert right[0] > 0
    assert left[0] < 0


@pytest.mark.parametrize(
    "size,w_cm,d_cm",
    [("small", 10.0, 10.0), ("medium", 15.9, 15.0), ("big", 17.9, 20.0)],
)
def test_crater_size_recovers_the_physical_class(size, w_cm, d_cm, geometry):
    """A crater of each true size, placed under the drone where there is no lean."""
    assert sizing.crater_size(w_cm, d_cm, NADIR_CM, geometry) == size


def test_the_lean_correction_matters_at_the_arena_edge(geometry):
    """A medium crater at the edge inflates toward BIG unless the lean is removed.

    This is the whole reason the correction exists. Medium and big differ by only 20mm in
    width, and a crater at the arena edge picks up over a centimetre of apparent size from
    its own height. Without the correction the two classes start to touch.
    """
    edge = (490.0, 194.0)
    lean = geometry.lean_cm(edge, sizing.CRATER_HEIGHT_CM["medium"])

    # What the camera actually sees for a medium crater out there: base plus lean.
    seen_w = 15.9 + abs(lean[0])
    seen_d = 15.0 + abs(lean[1])

    assert sizing.crater_size(seen_w, seen_d, edge, geometry) == "medium"
    # And the raw, uncorrected area is measurably closer to the big threshold.
    assert seen_w * seen_d > 15.9 * 15.0


def test_box_to_arena_uses_the_base_not_the_centroid(arena):
    """A tall object leans off its own footprint; its box CENTRE is not where it stands."""
    zone = arena.zone("RW-04")
    px_per_cm = 5.0
    # A 20px-wide, 60px-tall box: an object leaning strongly downward in the crop.
    box = (100.0, 100.0, 120.0, 160.0)

    base_cm, _, _ = sizing.box_to_arena(box, "RW-04", arena, px_per_cm)

    # The base sits at the box's BOTTOM edge (y2), not its middle.
    expected_y = (zone.y - 5.0) + 160.0 / px_per_cm
    assert base_cm[1] == pytest.approx(expected_y)


def test_pose_refuses_a_near_nadir_homography():
    """The bug this caught: focal=100px, altitude=20cm, returned without complaint.

    A fronto-parallel view makes the orthonormality constraints independent of the focal
    length, so every f fits equally well and argmin returns whatever noise preferred. A
    nadir view does not need a pose -- but silently inventing one is far worse than
    refusing.
    """
    # A homography with no perspective at all: pure scale + translation.
    nadir = np.array([[0.2, 0.0, -50.0], [0.0, 0.2, -40.0], [0.0, 0.0, 1.0]])
    with pytest.raises(posemod.PoseError, match="not identifiable"):
        posemod.estimate(np.linalg.inv(nadir), (3840, 2160))


# --- the split ---------------------------------------------------------------------

DATASET = Path("data/yolo")
CROP_RE = re.compile(r"^(?P<video>top_center_\d+)_f\d+_(?P<zone>[A-Z]{2}-[A-Z]?\d+)")


@pytest.mark.skipif(not DATASET.exists(), reason="run build_yolo_dataset first")
@pytest.mark.parametrize("task", ["crater", "uxo"])
def test_train_and_val_share_no_video(task):
    """The leak that would have made every metric a lie.

    Each video contributed 2 frames of a HOVERING drone, so its two frames are the same
    arena a few pixels apart. Split those at random and the same crater lands on both
    sides; validation comes back near-perfect and means nothing. Whole videos, or nothing.
    """
    def videos(split):
        return {
            CROP_RE.match(p.stem).group("video")
            for p in (DATASET / task / "images" / split).glob("*.jpg")
            if CROP_RE.match(p.stem)
        }

    train, val = videos("train"), videos("val")
    assert train and val
    assert not (train & val), f"{task}: videos in BOTH train and val: {sorted(train & val)}"


@pytest.mark.skipif(not DATASET.exists(), reason="run build_yolo_dataset first")
@pytest.mark.parametrize("task", ["crater", "uxo"])
def test_facility_zones_are_excluded(task):
    """Craters and UXO are only ever scored in RW/TW. FA crops are not training signal."""
    for split in ("train", "val"):
        for path in (DATASET / task / "images" / split).glob("*.jpg"):
            match = CROP_RE.match(path.stem)
            assert match and not match.group("zone").startswith("FA"), path.name


@pytest.mark.skipif(not DATASET.exists(), reason="run build_yolo_dataset first")
@pytest.mark.parametrize("task", ["crater", "uxo"])
def test_background_crops_are_present_and_explicitly_empty(task):
    """A crop with no object must be LISTED with an empty label, not simply absent.

    An absent image teaches nothing. An empty one teaches "this is background", which on
    a runway full of scorch marks and tyre streaks is the cheapest false-positive
    suppression we have.
    """
    labels = list((DATASET / task / "labels" / "train").glob("*.txt"))
    empties = [p for p in labels if not p.read_text().strip()]
    images = list((DATASET / task / "images" / "train").glob("*.jpg"))

    assert len(labels) == len(images)  # every image has a label file
    assert len(empties) > 50  # and a good share of them are background
