"""Camera pose from the arena homography, so we can reason about HEIGHT.

A homography rectifies the ground plane and nothing else. In a nadir view that
hardly matters -- objects are nearly under the camera, so they barely lean. In the
45-degree view it matters enormously: a 50cm building seen at 45 degrees projects
its roof a full 50cm away from its base on the ground plane, smearing it across an
entire neighbouring zone. Rectify that view and the buildings turn into streaks.

So for the oblique view we need the real camera, not just the plane-to-plane map.
That means recovering K, R and t, after which any 3D point can be projected -- and a
zone becomes a BOX (its footprint, extruded to the height of what stands on it)
rather than a flat quad.

We have no calibrated intrinsics, but we do not need them: a single homography of a
known plane is enough to solve for the focal length. Writing H = lambda*K[r1 r2 t],
the fact that r1 and r2 are orthonormal columns of a rotation matrix gives two
equations, and with the principal point assumed at the image centre and square
pixels, the only unknown left is f.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class PoseError(RuntimeError):
    pass


@dataclass(frozen=True)
class Pose:
    """The camera that produced an arena homography."""

    camera_matrix: np.ndarray  # 3x3 K
    rotation: np.ndarray  # 3x3 R, arena frame -> camera frame
    translation: np.ndarray  # 3, in cm
    focal_px: float
    elevation_deg: float  # 90 = straight down, 0 = horizontal

    @property
    def center_cm(self) -> np.ndarray:
        """Camera position in arena coordinates (cm). Z is height above the board."""
        return -self.rotation.T @ self.translation

    def project(self, points_cm: np.ndarray) -> np.ndarray:
        """Nx3 arena points (x, y, z = height above the board) -> Nx2 image px."""
        points = np.asarray(points_cm, dtype=np.float64).reshape(-1, 3)
        camera = points @ self.rotation.T + self.translation
        if np.any(camera[:, 2] <= 1e-6):
            raise PoseError("point is behind the camera")
        return _project(camera, self.camera_matrix)

    def lean_cm(self, height_cm: float) -> float:
        """How far the top of an object of this height sits from its base, in ground cm.

        This is the number that says whether a rectified oblique view is usable: at
        45 degrees it equals the object's own height, so a 50cm building lands 50cm
        from where it actually stands.
        """
        return float(height_cm / np.tan(np.deg2rad(self.elevation_deg)))


def _project(camera_points: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    projected = camera_points @ camera_matrix.T
    return projected[:, :2] / projected[:, 2:3]


def estimate(
    world_to_image: np.ndarray, image_size: tuple[int, int]
) -> Pose:
    """Recover K, R, t from a plane homography (arena cm -> image px).

    The principal point is assumed to be the image centre and the pixels square,
    which leaves the focal length as the single unknown. Both orthonormality
    constraints give an estimate of f^2; we take the mean of whichever are valid,
    and disagreement between them is a sign the assumptions do not hold.
    """
    width, height = image_size
    cx, cy = width / 2.0, height / 2.0
    matrix = np.asarray(world_to_image, dtype=np.float64)

    focal = _solve_focal(matrix, cx, cy)
    camera_matrix = np.array(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )

    # B = K^-1 H = lambda * [r1 r2 t]. Recover lambda from the fact that r1 and r2 are
    # unit vectors. Its SIGN is fixed by requiring the board to sit in front of the
    # camera (positive depth), not behind it.
    normalized = np.linalg.inv(camera_matrix) @ matrix
    scale = 2.0 / (
        np.linalg.norm(normalized[:, 0]) + np.linalg.norm(normalized[:, 1])
    )
    if normalized[2, 2] < 0:
        scale = -scale
    normalized = normalized * scale

    r1, r2, translation = normalized[:, 0], normalized[:, 1], normalized[:, 2]
    rotation = _orthonormalize(np.column_stack([r1, r2, np.cross(r1, r2)]))

    # Arena +z must point UP off the board, i.e. towards the camera. If the
    # decomposition produced the other handedness, r3 points into the board and every
    # extruded building would sink through it instead of standing up.
    center = -rotation.T @ translation
    if center[2] < 0:
        rotation = rotation @ np.diag([1.0, 1.0, -1.0])
        center = -rotation.T @ translation

    # Elevation above the board: 90 degrees is straight down.
    view_axis = rotation[2, :]  # camera's optical axis, in arena coordinates
    elevation = float(np.degrees(np.arcsin(min(abs(view_axis[2]), 1.0))))

    return Pose(
        camera_matrix=camera_matrix,
        rotation=rotation,
        translation=translation,
        focal_px=focal,
        elevation_deg=elevation,
    )


def _solve_focal(matrix: np.ndarray, cx: float, cy: float) -> float:
    """Focal length that best satisfies BOTH orthonormality constraints.

    The textbook move is to solve either constraint algebraically for f. Do not: the
    orthogonality one divides by c1*c2, which on a real view is close to zero, so it
    amplifies noise into nonsense -- on this footage it returns f=392 where the truth
    is ~2200. Averaging the two algebraic answers is worse still, since the result
    satisfies neither.

    Instead, sweep f and pick the value that actually makes r1 and r2 orthonormal.
    Cheap, and it cannot be blown up by a near-zero denominator.
    """
    inverse = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    centred = inverse @ matrix

    def residual(focal: float) -> float:
        scaling = np.diag([1.0 / focal, 1.0 / focal, 1.0])
        b1, b2 = (scaling @ centred[:, 0]), (scaling @ centred[:, 1])
        n1, n2 = np.linalg.norm(b1), np.linalg.norm(b2)
        if n1 < 1e-12 or n2 < 1e-12:
            return np.inf
        return (float(b1 @ b2 / (n1 * n2))) ** 2 + (float(n1 / n2) - 1.0) ** 2

    grid = np.geomspace(100.0, 50_000.0, 600)
    scores = np.array([residual(f) for f in grid])
    best = int(np.argmin(scores))
    if not np.isfinite(scores[best]):
        raise PoseError("cannot solve for focal length from this homography")

    # IDENTIFIABILITY, and this check is load-bearing. A low residual only says some
    # focal length FITS; it does not say that focal length is DETERMINED. As the view
    # approaches straight-down the orthonormality constraints stop depending on f at
    # all -- every focal fits equally well -- and the minimum flattens into a valley.
    # Argmin then returns whatever noise favoured. On the real nadir footage this
    # silently produced focal=100px and a camera altitude of 20cm, with a residual
    # small enough to sail through a threshold test. Nadir views do not need a pose,
    # but returning nonsense instead of raising is the exact failure this project
    # keeps guarding against.
    #
    # So measure the width of the valley: if a broad band of focal lengths all fit,
    # the focal is unidentifiable and there is no answer to give.
    tolerance = max(scores[best] * 10.0, 1e-4)
    acceptable = grid[scores < tolerance]
    if acceptable.size and acceptable.max() / acceptable.min() > 3.0:
        raise PoseError(
            f"focal length is not identifiable from this homography: everything from "
            f"{acceptable.min():.0f}px to {acceptable.max():.0f}px fits equally well. "
            "The view is too close to fronto-parallel for the orthonormality "
            "constraints to say anything about f. A near-nadir view needs no pose -- "
            "use the homography directly."
        )

    # Refine around the winner.
    low = grid[max(best - 1, 0)]
    high = grid[min(best + 1, len(grid) - 1)]
    fine = np.linspace(low, high, 400)
    focal = float(fine[int(np.argmin([residual(f) for f in fine]))])

    if residual(focal) > 0.05:
        raise PoseError(
            f"no focal length makes the view consistent (best residual "
            f"{residual(focal):.3f}). The principal point is probably not at the "
            "image centre, or the homography is wrong."
        )
    return focal


def _orthonormalize(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rotation)
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result


def lean_vectors(
    pose: Pose, ground_cm: np.ndarray, height_cm: float
) -> np.ndarray:
    """Image-space offset from a point's BASE to its TOP, for a given height.

    Deliberately returns a DELTA rather than an absolute position. The homography
    knows exactly where the ground is (sub-centimetre); the pose is a 7-DOF model of
    an 8-DOF homography, so it carries a residual of ~16px. Anchoring on the
    homography and taking only the lean from the pose keeps the pose's error out of
    the base position, where it would matter, and confines it to the lean, where a
    few pixels are irrelevant for cropping.
    """
    ground = np.asarray(ground_cm, dtype=np.float64).reshape(-1, 2)
    base = pose.project(np.c_[ground, np.zeros(len(ground))])
    top = pose.project(np.c_[ground, np.full(len(ground), height_cm)])
    return top - base


def zone_box_image_bounds(
    pose: Pose,
    ground_quad_px: np.ndarray,
    ground_quad_cm: np.ndarray,
    height_cm: float,
    image_size: tuple[int, int],
    pad_px: int = 20,
) -> tuple[int, int, int, int]:
    """Image bounding box of a zone's footprint EXTRUDED to `height_cm`.

    The ground quad alone would guillotine every building at its base -- which is
    precisely the detail the oblique view exists to capture. Extruding it gives a
    crop containing the whole structure, facade and all.

    `ground_quad_px` comes from the homography (exact); the extrusion comes from the
    pose.
    """
    ground_quad_px = np.asarray(ground_quad_px, dtype=np.float64).reshape(-1, 2)
    tops = ground_quad_px + lean_vectors(pose, ground_quad_cm, height_cm)
    corners = np.vstack([ground_quad_px, tops])

    width, height = image_size
    left = int(np.clip(corners[:, 0].min() - pad_px, 0, width - 1))
    right = int(np.clip(corners[:, 0].max() + pad_px, 0, width - 1))
    top = int(np.clip(corners[:, 1].min() - pad_px, 0, height - 1))
    bottom = int(np.clip(corners[:, 1].max() + pad_px, 0, height - 1))
    return left, top, right, bottom
