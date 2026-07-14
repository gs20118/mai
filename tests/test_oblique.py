"""The oblique view: partial markers rescued by printed ticks, and camera pose.

Two things have to hold for the 45-degree footage to be usable at all.

1. With one corner marker buried behind a building, the remaining markers fit a
   homography that looks healthy and is badly wrong in the corner they cannot see.
   The arena's printed ticks each fix ONE coordinate, and folding those linear
   constraints into the DLT has to pin that corner down.

2. A homography rectifies only the ground. To reason about anything with height --
   which at 44 degrees means every building smearing a full runway zone -- we need
   the camera pose, recovered from the homography itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from mai import pose as posemod
from mai.homography import AxisConstraint, HomographyError, axis_residuals, solve, solve_constrained

IMAGE_SIZE = (3840, 2160)


def build_camera(elevation_deg: float, focal_px: float = 2200.0):
    """A camera looking at the arena from a known pose, and its exact homography."""
    width, height = IMAGE_SIZE
    camera_matrix = np.array(
        [[focal_px, 0, width / 2], [0, focal_px, height / 2], [0, 0, 1.0]]
    )

    # Sit off the +x edge, looking back across the board -- like the real footage.
    tilt = np.deg2rad(90.0 - elevation_deg)
    center = np.array([615.0, 200.0, 240.0])

    forward = np.array([-np.sin(tilt), 0.0, -np.cos(tilt)])
    forward /= np.linalg.norm(forward)
    right = np.cross(np.array([0.0, 0.0, 1.0]), forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack([right, down, forward])
    translation = -rotation @ center

    extrinsic = np.column_stack([rotation[:, 0], rotation[:, 1], translation])
    world_to_image = camera_matrix @ extrinsic
    return world_to_image / world_to_image[2, 2], camera_matrix, rotation, center


def project(matrix, points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.c_[points, np.ones(len(points))] @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


# --- pose recovery -----------------------------------------------------------


@pytest.mark.parametrize("elevation", [35.0, 45.0, 60.0, 75.0])
def test_pose_recovers_a_known_camera(elevation):
    world_to_image, _, _, center = build_camera(elevation)
    recovered = posemod.estimate(world_to_image, IMAGE_SIZE)

    assert recovered.focal_px == pytest.approx(2200.0, rel=0.02)
    assert recovered.elevation_deg == pytest.approx(elevation, abs=1.5)
    assert recovered.center_cm == pytest.approx(center, abs=8.0)


def test_pose_reproduces_its_own_homography():
    world_to_image, _, _, _ = build_camera(45.0)
    recovered = posemod.estimate(world_to_image, IMAGE_SIZE)

    ground = np.array([[0, 0], [500, 0], [500, 400], [0, 400], [250, 200.0]])
    direct = project(world_to_image, ground)
    via_pose = recovered.project(np.c_[ground, np.zeros(len(ground))])
    assert np.linalg.norm(direct - via_pose, axis=1).max() < 1.0


def test_focal_is_not_solved_by_the_ill_conditioned_constraint():
    """Solving either orthonormality equation algebraically blows up.

    The orthogonality one divides by c1*c2, which is near zero on a real view -- on
    the actual footage it returns f=392 where the truth is ~2200. We search for the f
    that satisfies BOTH constraints instead. This pins that down.
    """
    world_to_image, _, _, _ = build_camera(45.0)
    width, height = IMAGE_SIZE
    cx, cy = width / 2, height / 2

    a = world_to_image[0, :] - cx * world_to_image[2, :]
    b = world_to_image[1, :] - cy * world_to_image[2, :]
    c = world_to_image[2, :]

    # Here the denominator is exactly zero -- a camera with no yaw makes c1*c2 vanish
    # outright, which is about as loud as an ill-conditioned equation can be. On the
    # real footage it is merely near-zero, which is worse: it returns a plausible
    # number (f=392) instead of an obvious infinity.
    with np.errstate(divide="ignore", invalid="ignore"):
        algebraic = -(a[0] * a[1] + b[0] * b[1]) / (c[0] * c[1])
    assert not np.isfinite(algebraic) or abs(np.sqrt(abs(algebraic)) - 2200.0) > 200.0

    recovered = posemod.estimate(world_to_image, IMAGE_SIZE)
    assert recovered.focal_px == pytest.approx(2200.0, rel=0.02)


def test_lean_equals_object_height_at_45_degrees():
    """A tidy sanity check: at 45 degrees the ground smear equals the height."""
    world_to_image, _, _, _ = build_camera(45.0)
    recovered = posemod.estimate(world_to_image, IMAGE_SIZE)
    assert recovered.lean_cm(50.0) == pytest.approx(50.0, rel=0.06)


def test_buildings_lean_away_from_the_nadir():
    """Aerial fact: a tall object's top is displaced radially AWAY from the nadir."""
    world_to_image, _, _, center = build_camera(45.0)
    recovered = posemod.estimate(world_to_image, IMAGE_SIZE)

    ground = np.array([[100.0, 100.0]])
    nadir_image = project(world_to_image, [center[:2]])[0]
    base_image = project(world_to_image, ground)[0]

    lean = posemod.lean_vectors(recovered, ground, 50.0)[0]
    away = base_image - nadir_image
    cosine = float(lean @ away / (np.linalg.norm(lean) * np.linalg.norm(away)))
    assert cosine > 0.95


# --- markers + ticks ---------------------------------------------------------

# The real board: 2 column splits and 4 band splits, printed on the border.
BOUNDARY_X_CM = [160.0, 340.0]
BOUNDARY_Y_CM = [80.0, 160.0, 240.0, 320.0]
MARKERS_CM = {
    1: (10.57, 11.41),
    2: (488.40, 11.58),
    3: (10.53, 388.48),
    4: (488.71, 388.61),
}
MARKER_SIZE_CM = 20.7


def marker_corners(marker_id):
    x, y = MARKERS_CM[marker_id]
    half = MARKER_SIZE_CM / 2
    return np.array(
        [[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]]
    )


def make_scene(visible_ids, corner_noise_px=0.35, tick_noise_px=0.5, seed=0):
    """Project the board through a 45-degree camera; hide one marker; add corner noise.

    Corner noise is scaled by how SMALL the marker appears. That is not a detail --
    it is the entire mechanism of the real failure. In this view the far markers
    subtend ~56px against ~181px for the near ones (the real footage: 42 vs 165), and
    a subpixel corner error on a small, blurred, steeply-oblique marker translates
    into a far larger arena error than the same pixel error on a big crisp one. Give
    every marker identical noise and the three-marker fit comes out fine, the test
    passes, and it proves nothing about the footage we actually have.
    """
    world_to_image, _, _, _ = build_camera(45.0)
    rng = np.random.default_rng(seed)

    reference_side = 181.0  # apparent size of a near marker
    image_points, world_points, marker_ids = [], [], []
    for marker_id in visible_ids:
        corners = marker_corners(marker_id)
        projected = project(world_to_image, corners)

        side = np.mean(
            [np.linalg.norm(projected[(i + 1) % 4] - projected[i]) for i in range(4)]
        )
        noise = corner_noise_px * (reference_side / max(side, 1.0))

        projected = projected + rng.normal(0, noise, projected.shape)
        image_points.append(projected)
        world_points.append(corners)
        marker_ids += [marker_id] * 4

    # Ticks sit ON the border, a little outside the arena rectangle. Each fixes only
    # the coordinate of the boundary it marks; the other is not known.
    constraints = []
    for x_cm in BOUNDARY_X_CM:
        for y_cm in (-6.0, 406.0):
            point = project(world_to_image, [[x_cm, y_cm]])[0]
            constraints.append(
                AxisConstraint(
                    image_xy=point + rng.normal(0, tick_noise_px, 2), axis=0, value_cm=x_cm
                )
            )
    for y_cm in BOUNDARY_Y_CM:
        for x_cm in (-6.0, 506.0):
            point = project(world_to_image, [[x_cm, y_cm]])[0]
            constraints.append(
                AxisConstraint(
                    image_xy=point + rng.normal(0, tick_noise_px, 2), axis=1, value_cm=y_cm
                )
            )

    return (
        np.vstack(image_points),
        np.vstack(world_points),
        marker_ids,
        constraints,
        world_to_image,
    )


def far_corner_error(solved, world_to_image):
    """Error in the corner where the occluded marker lives -- where nothing checks it."""
    probes = np.array([[0.0, 0.0], [40.0, 40.0], [0.0, 80.0], [80.0, 0.0]])
    image = project(world_to_image, probes)
    recovered = solved.to_world(image)
    return float(np.linalg.norm(recovered - probes, axis=1).max())


def test_three_markers_drift_in_the_corner_they_cannot_see():
    """The dangerous case: healthy-looking reprojection error, badly wrong far corner."""
    image_points, world_points, marker_ids, _, world_to_image = make_scene([2, 3, 4])
    solved = solve(image_points, world_points, marker_ids, min_markers=3)

    # It looks fine where the markers are...
    assert solved.rms_cm < 2.0
    # ...and is wrong where they are not.
    assert far_corner_error(solved, world_to_image) > 3.0


def test_ticks_pin_down_the_corner_the_markers_cannot_reach():
    image_points, world_points, marker_ids, constraints, world_to_image = make_scene([2, 3, 4])

    markers_only = solve(image_points, world_points, marker_ids, min_markers=3)
    combined = solve_constrained(image_points, world_points, constraints, marker_ids)

    assert far_corner_error(combined, world_to_image) < far_corner_error(
        markers_only, world_to_image
    ) / 3
    assert far_corner_error(combined, world_to_image) < 1.5


def test_two_markers_on_one_edge_are_rescued_by_ticks():
    """Most real frames show only markers 3 and 4 -- and they share an edge.

    Two markers on the same edge barely constrain the perpendicular direction, so on
    their own they are close to degenerate. The ticks make those frames usable, which
    is what makes the oblique video worth having at all.
    """
    image_points, world_points, marker_ids, constraints, world_to_image = make_scene([3, 4])

    combined = solve_constrained(image_points, world_points, constraints, marker_ids)
    assert far_corner_error(combined, world_to_image) < 2.0

    residuals = axis_residuals(combined, constraints)
    assert np.abs(residuals).max() < 2.0


def test_a_single_marker_is_still_refused_even_with_ticks():
    image_points, world_points, marker_ids, constraints, _ = make_scene([4])
    with pytest.raises(HomographyError, match="need at least"):
        solve_constrained(image_points, world_points, constraints, marker_ids)


def test_axis_residuals_measure_the_fit_where_markers_do_not():
    """Marker reprojection error only checks the fit where the markers are."""
    image_points, world_points, marker_ids, constraints, _ = make_scene([2, 3, 4])
    markers_only = solve(image_points, world_points, marker_ids, min_markers=3)

    assert markers_only.rms_cm < 2.0  # looks healthy
    assert np.abs(axis_residuals(markers_only, constraints)).max() > 2.0  # isn't


# --- SIFT against the nadir map: a third, denser, occlusion-proof source ---------

def _synthetic_pair(occlude_marker: int = 1):
    """One physical arena, shot twice: nadir (all markers) and 45-degree (one buried).

    The occlusion is applied to the OBLIQUE IMAGE, not to the shared canvas. That is
    how a building actually occludes: it blocks the line of sight from one viewpoint
    and not another, which is precisely why all four markers are visible in the real
    nadir footage and only two or three in the 45-degree footage. Painting the canvas
    instead would take the marker away from both views and quietly destroy the very
    reference map the test depends on.

    The nadir frame becomes the reference map -- anchored on all four markers, so it
    carries absolute arena coordinates. The oblique frame then localises against it.
    """
    import cv2
    from mai.arena import Arena
    from mai.synthetic import CameraPose, capture, render_topdown

    arena = Arena.from_yaml()
    canvas = render_topdown(arena, px_per_cm=6.0, texture="speckle", seed=11)

    size = (1920, 1080)
    nadir = capture(
        arena,
        pose=CameraPose(250, 200, 320, pitch_deg=1.0, yaw_deg=2.0),
        image_size=size, distortion=(0, 0, 0, 0, 0),
        render_px_per_cm=6.0, topdown=canvas,
    )
    oblique = capture(
        arena,
        pose=CameraPose(250, 200, 260, pitch_deg=42.0, yaw_deg=0.0),
        image_size=size, focal_px=1100.0, distortion=(0, 0, 0, 0, 0),
        render_px_per_cm=6.0, topdown=canvas,
    )

    if occlude_marker is not None:
        centre = np.array([arena.markers[occlude_marker].center])
        at = project(oblique.true_world_to_image, centre)[0].astype(int)
        half = 90
        cv2.rectangle(
            oblique.image,
            (at[0] - half, at[1] - half),
            (at[0] + half, at[1] + half),
            (55, 55, 55),
            -1,
        )
    return arena, nadir, oblique


def test_sift_map_registers_a_frame_whose_markers_are_buried():
    """The answer to 'what if the ticks get covered too?'

    Markers and ticks are both SPARSE and individually occludable -- a building already
    buries one marker. SIFT against the ArUco-anchored nadir map yields hundreds of
    full correspondences spread over the whole board, and no single obstruction hides
    hundreds of features. On the real footage it proved to be the MOST reliable of the
    three sources.
    """
    import cv2
    from mai import aruco, localize, topview
    from mai.frames import Frame
    from mai.undistort import CameraProfile, Undistorter

    arena, nadir, oblique = _synthetic_pair(occlude_marker=1)
    undistorter = Undistorter(
        CameraProfile(name="s", image_size=(1920, 1080), fx=1100, fy=1100, cx=960, cy=540)
    )
    detector = aruco.build_detector(arena.dictionary)

    anchor = topview.register(
        Frame(nadir.image, 0, "nadir"), arena, undistorter, detector
    )
    px_per_cm = topview.native_px_per_cm(arena, anchor.homography)
    reference = localize.Reference(
        topview.warp(anchor.undistorted, arena, anchor.homography, px_per_cm),
        px_per_cm,
        arena,
    )

    detections = aruco.detect(oblique.image, detector, keep_ids=set(arena.markers))
    assert 1 not in [d.id for d in detections]  # genuinely buried
    assert len(detections) >= 2

    image_points, world_points = aruco.correspondences(detections, arena)
    seed, _ = cv2.findHomography(
        image_points.reshape(-1, 1, 2), world_points.reshape(-1, 1, 2)
    )

    sift_image, sift_arena = localize.ground_correspondences(
        oblique.image, reference, seed
    )
    assert len(sift_image) >= 50  # hundreds, not a handful

    fused = solve_constrained(
        np.vstack([image_points, sift_image]),
        np.vstack([world_points, sift_arena]),
        [],
        [d.id for d in detections],
    )

    # Ground truth: project known arena points through the TRUE oblique homography.
    truth = np.array([[x, y] for x in (60, 250, 440) for y in (100, 200, 300)], float)
    true_image = project(oblique.true_world_to_image, truth)
    recovered = fused.to_world(true_image)
    assert np.linalg.norm(recovered - truth, axis=1).max() < 4.0

    # And every probe still lands in its true zone.
    for point, back in zip(truth, recovered):
        assert arena.zone_at(*back) == arena.zone_at(*point)
