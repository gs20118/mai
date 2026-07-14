"""End-to-end geometry, against synthetic ground truth.

Render the arena -> project through a known camera pose -> barrel-distort it, then
make the pipeline recover the arena. Because we know the answer exactly, this
catches the class of bug that would otherwise only surface on competition day: a
rotated marker map, a sign flip in a distortion coefficient, a zone grid off by
one band. All of those produce a plausible-looking image and wrong zone IDs.
"""

from __future__ import annotations

import numpy as np
import pytest

from mai import aruco, homography, topview, zones
from mai.arena import Arena
from mai.frames import Frame
from mai.synthetic import CameraPose, PlacedObject, capture, fit_focal_px
from mai.undistort import CameraProfile, Undistorter

IMAGE_SIZE = (3840, 2160)
HEIGHT_CM = 300.0
DISTORTION = (-0.28, 0.09, 0.0, 0.0, 0.0)

POSE = CameraPose(x_cm=250.0, y_cm=200.0, height_cm=HEIGHT_CM, pitch_deg=3.0, yaw_deg=5.0)
# Longest focal that still frames the whole 500x400cm arena on a 16:9 sensor.
FOCAL_PX = fit_focal_px(Arena.from_yaml(), POSE, IMAGE_SIZE)

# One object per zone we care about, placed at a known arena position.
OBJECTS = [
    PlacedObject("crater_big", 175.0, 200.0),  # RW-04
    PlacedObject("crater_small", 475.0, 200.0),  # RW-10
    PlacedObject("uxo_cluster", 150.0, 120.0),  # TW-A2
    PlacedObject("uxo_misile", 450.0, 280.0),  # TW-B5
]
EXPECTED_ZONES = ["RW-04", "RW-10", "TW-A2", "TW-B5"]


def make_profile(distortion=DISTORTION, alpha: float = 1.0) -> CameraProfile:
    k1, k2, p1, p2, k3 = distortion
    return CameraProfile(
        name="synthetic",
        image_size=IMAGE_SIZE,
        fx=FOCAL_PX,
        fy=FOCAL_PX,
        cx=IMAGE_SIZE[0] / 2,
        cy=IMAGE_SIZE[1] / 2,
        k1=k1,
        k2=k2,
        p1=p1,
        p2=p2,
        k3=k3,
        alpha=alpha,
    )


def project(matrix: np.ndarray, point) -> np.ndarray:
    homogeneous = np.array([point[0], point[1], 1.0], dtype=np.float64)
    projected = matrix @ homogeneous
    return projected[:2] / projected[2]


@pytest.fixture(scope="module")
def arena() -> Arena:
    return Arena.from_yaml()


@pytest.fixture(scope="module")
def scene(arena: Arena):
    """A slightly tilted, slightly yawed hover shot, not a degenerate top-down."""
    return capture(
        arena,
        pose=POSE,
        objects=OBJECTS,
        image_size=IMAGE_SIZE,
        focal_px=FOCAL_PX,
        distortion=DISTORTION,
    )


@pytest.fixture(scope="module")
def undistorter() -> Undistorter:
    return Undistorter(make_profile())


@pytest.fixture(scope="module")
def registration(arena: Arena, scene, undistorter: Undistorter):
    frame = Frame(image=scene.image, index=0, source="synthetic")
    detector = aruco.build_detector(arena.dictionary)
    return topview.register(frame, arena, undistorter, detector)


def world_to_undistorted(world_cm, scene, undistorter: Undistorter) -> np.ndarray:
    """Where a known arena point lands in the undistorted frame, via ground truth.

    Undistortion maps a distorted pixel onto the same viewing ray but expressed in
    the NEW camera matrix that getOptimalNewCameraMatrix produced. So the transfer
    from the ideal (original-K) frame to the undistorted (new-K) frame is
    new_K @ inv(K) -- not the identity, which is the easy mistake here.
    """
    ideal = project(scene.true_world_to_image, world_cm)
    new_camera_matrix = undistorter.new_camera_matrix(*IMAGE_SIZE)
    transfer = new_camera_matrix @ np.linalg.inv(scene.camera_matrix)
    return project(transfer, ideal)


def test_all_four_markers_are_found(arena: Arena, registration):
    # IDs come from the measured arena config (they are 1..4 on the real board),
    # so assert against the config rather than hard-coding them.
    assert registration.marker_ids == sorted(arena.markers)


def test_homography_recovers_arena_coordinates(registration):
    """Sub-centimetre on a 500cm arena. The narrowest zone is 50cm, so this is ample."""
    assert registration.homography.rms_cm < 0.5
    assert registration.homography.max_error_cm < 1.0
    assert registration.homography.inliers == 16  # 4 corners x 4 markers


def test_undistortion_is_load_bearing(arena: Arena):
    """Skipping undistortion must measurably degrade the fit, or the stage is theatre.

    A homography cannot represent radial distortion, so it absorbs it as a
    least-squares compromise that is worst at the frame edges -- exactly where the
    corner zones live.

    Uses a mild barrel (k1=-0.08) rather than the module's punishing default,
    because under the strong one the uncorrected markers do not decode at all --
    which proves the point, but proves it by a different mechanism than the one
    under test here.
    """
    mild = (-0.08, 0.0, 0.0, 0.0, 0.0)
    scene = capture(
        arena, pose=POSE, image_size=IMAGE_SIZE, focal_px=FOCAL_PX, distortion=mild
    )
    frame = Frame(image=scene.image, index=0, source="synthetic")
    detector = aruco.build_detector(arena.dictionary)

    corrected = topview.register(
        frame, arena, Undistorter(make_profile(distortion=mild)), detector
    )
    uncorrected = topview.register(
        frame, arena, Undistorter(make_profile(distortion=(0, 0, 0, 0, 0))), detector
    )

    # Assert on MAX error, not RMS. rms_cm is computed over RANSAC's inliers only, so
    # it flatters a bad fit by ignoring the very correspondences the distortion threw
    # off -- here it hides a 3cm error behind a 0.29cm RMS. Max error is also the
    # quantity that actually matters: it is what pushes an object into the wrong zone.
    assert corrected.homography.max_error_cm < 0.5
    assert corrected.homography.inliers == corrected.homography.total

    # Uncorrected, the fit is off by centimetres at the worst point, and RANSAC has to
    # discard correspondences it cannot reconcile.
    assert uncorrected.homography.max_error_cm > 2.0
    assert uncorrected.homography.inliers < uncorrected.homography.total
    assert uncorrected.homography.max_error_cm > 10 * corrected.homography.max_error_cm


def test_objects_land_in_their_true_zones(arena: Arena, scene, registration, undistorter):
    """The entire competition reduces to this: does a detection get the right zone ID?"""
    solved = registration.homography

    for placed, expected in zip(scene.objects, EXPECTED_ZONES):
        truth = np.array([placed.x_cm, placed.y_cm])
        image_point = world_to_undistorted(truth, scene, undistorter)
        recovered = solved.to_world(image_point.reshape(1, 2))[0]

        assert arena.zone_at(*recovered) == expected
        assert np.linalg.norm(recovered - truth) < 1.0


def test_base_point_beats_centroid_for_tall_objects():
    """A 115mm missile leans away from nadir; its bbox centroid is not its footprint."""
    bbox = (100.0, 200.0, 140.0, 300.0)
    assert homography.base_point(bbox).tolist() == [120.0, 300.0]


def test_a_partial_marker_view_is_refused(arena: Arena, scene, undistorter):
    """Fewer than four markers must fail loudly rather than fit a plausible lie.

    One marker's four corners fit a homography perfectly -- near-zero reprojection
    error -- while being wildly wrong once extrapolated across the arena. Silence
    here would be far more dangerous than an exception.
    """
    frame = Frame(image=scene.image, index=0, source="synthetic")
    detector = aruco.build_detector(arena.dictionary)
    detections = aruco.detect(
        undistorter(frame.image), detector, keep_ids=set(arena.markers)
    )
    assert len(detections) == 4

    subset = detections[:3]
    image_points, world_points = aruco.correspondences(subset, arena)
    with pytest.raises(homography.HomographyError, match="need 4"):
        homography.solve(
            image_points, world_points, [d.id for d in subset]
        )


def test_zone_crops_preserve_source_resolution(arena: Arena, registration, undistorter):
    """Crops must not silently downsample: a 28mm target has no pixels to spare.

    crop_zone derives its scale from the homography's local Jacobian; here we check
    it against the camera geometry (focal / altitude), which is an independent
    derivation of the same quantity.
    """
    solved = registration.homography
    crop, record = zones.crop_zone(
        registration.undistorted, arena.zone("RW-04"), solved, pad_cm=5.0
    )

    new_focal = undistorter.new_camera_matrix(*IMAGE_SIZE)[0, 0]
    expected_px_per_cm = new_focal / HEIGHT_CM
    assert record.px_per_cm == pytest.approx(expected_px_per_cm, rel=0.15)

    # Undistortion must not cost us resolution: the crop should still carry the
    # source's own px/cm (focal/altitude), not the shrunken focal that
    # getOptimalNewCameraMatrix hands back when it squeezes the debulged image
    # into the original canvas.
    source_px_per_cm = FOCAL_PX / HEIGHT_CM
    assert record.px_per_cm > 0.95 * source_px_per_cm

    # The legacy pipeline warped at 2 px/cm. Anything near that has thrown away
    # most of the sensor, and a cluster munition with it.
    assert record.px_per_cm > 4.0
    assert record.visible and record.coverage == 1.0

    # 50cm zone + 5cm padding each side, at the crop's own scale.
    assert crop.shape[1] == pytest.approx(60 * record.px_per_cm, abs=2)
    assert crop.shape[0] == pytest.approx(90 * record.px_per_cm, abs=2)


def test_gsd_report_measures_the_binding_constraint(arena: Arena, registration):
    solved = registration.homography
    records = [
        zones.crop_zone(registration.undistorted, zone, solved)[1] for zone in arena.zones
    ]
    report = zones.gsd_report(arena, records)

    assert report["summary"]["zones_visible"] == 26

    # The SMALLEST target decides whether the 18-point UXO mission is winnable from a
    # given altitude, so find it rather than naming it. This test used to hard-code the
    # cluster munition, on TASK.md's authority that it was a 28mm cube. It is not -- the
    # cluster is a long object and the small ball ("dumb") is the binding constraint. See
    # the note above `targets:` in configs/arena.yaml. Asking the config which target is
    # smallest, instead of assuming, means a future correction to the props cannot
    # silently invalidate the assertion again.
    targets = report["summary"]["smallest_target_px_worst_zone"]
    uxo = {name: px for name, px in targets.items() if name.startswith("uxo_")}
    smallest_name = min(uxo, key=uxo.get)
    smallest_px = uxo[smallest_name]

    assert smallest_name == "uxo_dumb"
    assert smallest_px > 0

    # A big crater must be far easier to see than the smallest UXO.
    assert targets["crater_big"] > 4 * smallest_px
