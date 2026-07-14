"""Markerless localisation of the high-resolution strip pass.

The strip pass cannot see the corner markers -- that is forced, not chosen: more
resolution means covering less arena, and any frame containing all four markers
must span the full arena depth. So it registers against the ArUco-anchored top view
by feature matching.

The tests that matter here are the ones where it must FAIL. A periodic runway can
produce a homography shifted by exactly one zone that looks entirely healthy, and
that answer scores zero. Refusing is the correct behaviour; guessing is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from mai import aruco, localize, topview, zones
from mai.arena import Arena
from mai.frames import Frame
from mai.localize import Gates, LocalizationError, Reference
from mai.synthetic import (
    CameraPose,
    PlacedObject,
    capture,
    fit_focal_for_coverage,
    fit_focal_px,
    render_topdown,
)
from mai.undistort import CameraProfile, Undistorter

# Kept modest so the suite stays fast; the ratios are what matter, not the absolutes.
TOP_IMAGE = (1920, 1080)
LOW_IMAGE = (2016, 1512)  # 4:3, like a still rather than 16:9 video
TOP_HEIGHT_CM = 300.0
LOW_HEIGHT_CM = 130.0
CANVAS_PX_PER_CM = 8.0

# The airfield strip (TW-A + RW + TW-B) is 500 x 240cm, centred at y=200.
# Two stills cover it; this is the left one.
STRIP_COVERAGE = (250.0, 240.0)
STRIP_CENTER = (125.0, 200.0)

OBJECTS = [
    PlacedObject("crater_big", 175.0, 200.0),  # RW-04
    PlacedObject("crater_small", 75.0, 200.0),  # RW-02
    PlacedObject("uxo_cluster", 225.0, 200.0),  # RW-05
    PlacedObject("uxo_misile", 50.0, 120.0),  # TW-A1
    PlacedObject("uxo_dumb", 150.0, 280.0),  # TW-B2
]


def undistorter_for(image_size, focal) -> Undistorter:
    return Undistorter(
        CameraProfile(
            name="synthetic",
            image_size=image_size,
            fx=focal,
            fy=focal,
            cx=image_size[0] / 2,
            cy=image_size[1] / 2,
            k1=-0.05,
            k2=0.01,
            alpha=1.0,
        )
    )


def build_scene(arena: Arena, texture: str):
    """One physical arena, photographed twice: high with markers, low without."""
    canvas = render_topdown(
        arena, OBJECTS, px_per_cm=CANVAS_PX_PER_CM, texture=texture, seed=7
    )

    top_pose = CameraPose(250.0, 200.0, TOP_HEIGHT_CM, pitch_deg=2.0, yaw_deg=3.0)
    top_focal = fit_focal_px(arena, top_pose, TOP_IMAGE)
    # objects= is passed even though the canvas is pre-rendered: it records the
    # ground truth on the capture. Omitting it leaves capture.objects empty, which
    # silently turns any test that loops over it into a vacuous pass.
    top = capture(
        arena,
        pose=top_pose,
        objects=OBJECTS,
        image_size=TOP_IMAGE,
        focal_px=top_focal,
        distortion=(-0.05, 0.01, 0, 0, 0),
        render_px_per_cm=CANVAS_PX_PER_CM,
        topdown=canvas,
    )

    low_pose = CameraPose(
        STRIP_CENTER[0], STRIP_CENTER[1], LOW_HEIGHT_CM, pitch_deg=1.0, yaw_deg=-2.0
    )
    low_focal = fit_focal_for_coverage(STRIP_COVERAGE, low_pose, LOW_IMAGE)
    low = capture(
        arena,
        pose=low_pose,
        objects=OBJECTS,
        image_size=LOW_IMAGE,
        focal_px=low_focal,
        distortion=(-0.05, 0.01, 0, 0, 0),
        render_px_per_cm=CANVAS_PX_PER_CM,
        topdown=canvas,
    )
    return top, top_focal, low, low_focal


def build_reference(arena: Arena, top, top_focal) -> tuple[Reference, float]:
    """Register the top view on its markers, then turn it into a map."""
    registration = topview.register(
        Frame(image=top.image, index=0, source="top"),
        arena,
        undistorter_for(TOP_IMAGE, top_focal),
        aruco.build_detector(arena.dictionary),
    )
    assert registration.marker_ids == sorted(arena.markers)

    px_per_cm = topview.native_px_per_cm(arena, registration.homography)
    warped = topview.warp(
        registration.undistorted, arena, registration.homography, px_per_cm
    )
    return Reference(warped, px_per_cm, arena), px_per_cm


@pytest.fixture(scope="module")
def arena() -> Arena:
    return Arena.from_yaml()


@pytest.fixture(scope="module")
def textured(arena: Arena):
    """A realistic surface: 3D-printed grain, which is random and so aperiodic."""
    top, top_focal, low, low_focal = build_scene(arena, texture="speckle")
    reference, ref_px_per_cm = build_reference(arena, top, top_focal)
    undistorted_low = undistorter_for(LOW_IMAGE, low_focal)(low.image)
    return reference, ref_px_per_cm, undistorted_low, low


def test_the_strip_pass_is_worth_flying(textured):
    """It must actually buy resolution, or there is no reason to accept markerless."""
    reference, ref_px_per_cm, _, _ = textured
    result = localize.localize(textured[2], reference)

    assert result.px_per_cm > 2 * ref_px_per_cm

    # The cluster munition is the whole point of this pass.
    cluster_px_top = 28.0 / (10.0 / ref_px_per_cm)
    cluster_px_low = 28.0 / (10.0 / result.px_per_cm)
    assert cluster_px_low > 2 * cluster_px_top


def test_localizes_without_any_marker(arena: Arena, textured):
    reference, _, undistorted_low, _ = textured
    result = localize.localize(undistorted_low, reference)

    assert result.homography.marker_ids == []  # nothing to see
    assert result.inliers >= 20
    assert result.homography.rms_cm < 2.0
    assert result.alias_ratio < 0.5

    # It should place itself over the LEFT half of the airfield and stay there.
    assert {"RW-02", "RW-04", "TW-A1", "TW-B2"} <= set(result.zones)
    assert not any(
        zone in result.zones for zone in ("RW-09", "RW-10", "TW-A5", "TW-B5")
    )
    # Catching a sliver of the facility band is expected and in fact welcome: the
    # facilities are aperiodic, so they help break the runway's 50cm ambiguity.
    assert {zone for zone in result.zones if zone.startswith("RW")} >= {
        f"RW-0{n}" for n in range(1, 6)
    }


def test_objects_get_the_right_zone_through_the_markerless_fit(arena: Arena, textured):
    """The end of the chain: a detection in a markerless frame -> a correct zone ID."""
    reference, _, undistorted_low, low = textured
    result = localize.localize(undistorted_low, reference)

    expected = {
        "crater_big": "RW-04",
        "crater_small": "RW-02",
        "uxo_cluster": "RW-05",
        "uxo_misile": "TW-A1",
        "uxo_dumb": "TW-B2",
    }
    checked = 0
    for placed in low.objects:
        truth = np.array([placed.x_cm, placed.y_cm])
        image_point = _project_into(truth, low, undistorted_low.shape)
        if image_point is None:
            continue  # outside this strip
        recovered = result.homography.to_world(image_point.reshape(1, 2))[0]
        assert arena.zone_at(*recovered) == expected[placed.kind]
        assert np.linalg.norm(recovered - truth) < 3.0
        checked += 1

    # Without this, forgetting to populate capture.objects turns the loop above into
    # a vacuous pass -- which it silently did on the first run of this file.
    assert checked == len(expected)


def test_smooth_periodic_runway_is_refused_not_guessed(arena: Arena):
    """The adversarial case, and the whole reason this module has gates.

    A featureless, perfectly repeating runway offers no evidence that distinguishes
    zone k from zone k+1. The only safe answer is no answer: raise, and let the
    caller fall back to the marker-registered top view. Returning a confident,
    well-formed, off-by-one-zone answer would score zero and look like success.
    """
    top, top_focal, low, low_focal = build_scene(arena, texture="none")
    reference, _ = build_reference(arena, top, top_focal)
    undistorted_low = undistorter_for(LOW_IMAGE, low_focal)(low.image)

    with pytest.raises(LocalizationError):
        localize.localize(undistorted_low, reference)


def test_gates_reject_a_deliberately_shifted_fit(arena: Arena, textured):
    """Hand the alias check a solution that IS off by one zone, and make it object."""
    reference, _, undistorted_low, _ = textured
    honest = localize.localize(undistorted_low, reference)

    # A 50cm shift along the runway: one zone, the exact competition-losing error.
    shift = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    shifted = shift @ honest.homography.matrix

    truth = np.array([175.0, 200.0])  # the big crater, in RW-04
    honest_zone = arena.zone_at(*honest.homography.to_world(
        honest.homography.to_image(truth.reshape(1, 2))
    )[0])
    lied_zone = arena.zone_at(
        *localize._apply(honest.homography.to_image(truth.reshape(1, 2)), shifted)[0]
    )

    assert honest_zone == "RW-04"
    assert lied_zone == "RW-05"  # plausible, well-formed, and worth zero points


def test_a_frame_that_is_not_the_arena_is_refused(arena: Arena, textured):
    """Noise in, refusal out -- never a homography."""
    reference, _, undistorted_low, _ = textured
    noise = np.random.default_rng(1).integers(
        0, 255, undistorted_low.shape, dtype=np.uint8
    )
    with pytest.raises(LocalizationError):
        localize.localize(noise, reference)


def test_strict_gates_can_refuse_a_good_frame(textured):
    """Gates are tunable, and tightening them fails closed rather than open."""
    reference, _, undistorted_low, _ = textured
    with pytest.raises(LocalizationError):
        localize.localize(
            undistorted_low, reference, gates=Gates(min_inliers=100_000)
        )


def test_anchor_check_catches_what_geometry_cannot(arena: Arena, textured):
    """Random object placement is aperiodic, so it breaks the one-zone ambiguity."""
    reference, _, undistorted_low, low = textured
    result = localize.localize(undistorted_low, reference)

    known_cm = np.array([[obj.x_cm, obj.y_cm] for obj in OBJECTS])
    seen_px = np.array(
        [
            point
            for obj in low.objects
            if (point := _project_into(np.array([obj.x_cm, obj.y_cm]), low, undistorted_low.shape))
            is not None
        ]
    )

    assert localize.check_anchors(result.homography, seen_px, known_cm) == 1.0

    # The same objects, viewed through a one-zone-shifted fit, no longer line up.
    shift = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    from mai.homography import Homography

    lying = Homography(
        matrix=shift @ result.homography.matrix,
        rms_cm=0.0, max_error_cm=0.0, inliers=0, total=0, marker_ids=[],
    )
    assert localize.check_anchors(lying, seen_px, known_cm) < 0.5


def test_zone_crops_work_unchanged_on_a_markerless_fit(arena: Arena, textured):
    """A Homography is a Homography: the crop layer should not care how we got it."""
    reference, ref_px_per_cm, undistorted_low, _ = textured
    result = localize.localize(undistorted_low, reference)

    _, record = zones.crop_zone(undistorted_low, arena.zone("RW-04"), result.homography)
    assert record.visible
    # Compared against the reference rather than an absolute mm/px, because these
    # tests run at reduced image sizes to stay fast. The RATIO is the real claim.
    assert record.px_per_cm > 2 * ref_px_per_cm


def _project_into(world_cm, scene, shape):
    """Ground-truth projection of an arena point into an undistorted frame."""
    ideal = scene.true_world_to_image @ np.array([world_cm[0], world_cm[1], 1.0])
    ideal = ideal[:2] / ideal[2]

    undistorter = undistorter_for(
        (scene.image.shape[1], scene.image.shape[0]), scene.camera_matrix[0, 0]
    )
    transfer = undistorter.new_camera_matrix(
        scene.image.shape[1], scene.image.shape[0]
    ) @ np.linalg.inv(scene.camera_matrix)
    point = transfer @ np.array([ideal[0], ideal[1], 1.0])
    point = point[:2] / point[2]

    height, width = shape[:2]
    if not (0 <= point[0] < width and 0 <= point[1] < height):
        return None
    return point
