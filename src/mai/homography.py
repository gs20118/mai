"""The image -> arena transform, and its quality gate.

H maps undistorted image pixels directly to arena CENTIMETRES, not to some
display canvas. Keeping "arena coordinates" and "display pixels" as separate
things is deliberate: every downstream mission reasons in cm (zone lookup, crater
size thresholds, runway length), and only the visualisation layer ever converts
to pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class HomographyError(RuntimeError):
    pass


@dataclass(frozen=True)
class Homography:
    matrix: np.ndarray  # 3x3, undistorted image px -> arena cm
    rms_cm: float  # reprojection error of the marker corners, in cm
    max_error_cm: float
    inliers: int
    total: int
    marker_ids: list[int]

    @property
    def inverse(self) -> np.ndarray:
        return np.linalg.inv(self.matrix)

    def to_world(self, image_points: np.ndarray) -> np.ndarray:
        """Nx2 image px -> Nx2 arena cm."""
        return _transform(np.asarray(image_points, dtype=np.float64), self.matrix)

    def to_image(self, world_points: np.ndarray) -> np.ndarray:
        """Nx2 arena cm -> Nx2 image px."""
        return _transform(np.asarray(world_points, dtype=np.float64), self.inverse)

    def topview_matrix(self, px_per_cm: float) -> np.ndarray:
        """Compose with a scale, giving image px -> top-view canvas px."""
        scale = np.array(
            [[px_per_cm, 0.0, 0.0], [0.0, px_per_cm, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return scale @ self.matrix

    def source_px_per_cm(self, world_point: np.ndarray) -> float:
        """How many source pixels cover one arena cm at this arena location.

        This is the number that decides whether a target is detectable: it varies
        across the frame because zones near the edge are viewed more obliquely and
        from further away, so they get fewer pixels per centimetre than the
        centre. Computed as the local Jacobian of the world -> image map.
        """
        world_point = np.asarray(world_point, dtype=np.float64).reshape(2)
        delta = 0.5  # cm
        origin = self.to_image(world_point.reshape(1, 2))[0]
        along_x = self.to_image((world_point + [delta, 0.0]).reshape(1, 2))[0]
        along_y = self.to_image((world_point + [0.0, delta]).reshape(1, 2))[0]
        px_x = np.linalg.norm(along_x - origin) / delta
        px_y = np.linalg.norm(along_y - origin) / delta
        # Geometric mean: the effective linear resolution of an isotropic target.
        return float(np.sqrt(px_x * px_y))

    def mm_per_px(self, world_point: np.ndarray) -> float:
        px_per_cm = self.source_px_per_cm(world_point)
        return float("inf") if px_per_cm <= 0 else 10.0 / px_per_cm


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    source = points.reshape(-1, 1, 2).astype(np.float64)
    return cv2.perspectiveTransform(source, matrix).reshape(-1, 2)


REQUIRED_MARKERS = 4


def solve(
    image_points: np.ndarray,
    world_points: np.ndarray,
    marker_ids: list[int],
    ransac_threshold_cm: float = 2.0,
    min_markers: int = REQUIRED_MARKERS,
) -> Homography:
    """Fit image px -> arena cm from marker correspondences.

    All four corner markers are required. A subset would still *fit*: a single
    marker supplies 4 corners, enough to determine a homography, and it will fit
    them near-perfectly and report a reprojection error close to zero. But
    extrapolating a fit anchored on a ~10cm marker across a 500cm arena amplifies
    any corner noise by ~50x. That failure mode is the dangerous one, because it
    looks like a flawless registration while putting objects in the wrong zones.
    Demanding all four corners braces the fit across the full arena.

    RANSAC's threshold is in centimetres because the target space is centimetres,
    which makes it a physically meaningful number rather than an arbitrary one.
    """
    image_points = np.asarray(image_points, dtype=np.float64)
    world_points = np.asarray(world_points, dtype=np.float64)
    distinct = sorted(set(marker_ids))

    if len(distinct) < min_markers:
        raise HomographyError(
            f"saw {len(distinct)} marker(s) {distinct}, need {min_markers}. "
            "Reframe so all four corner markers are visible."
        )

    if len(image_points) < 4:
        raise HomographyError(
            f"need >=4 point correspondences, got {len(image_points)} "
            f"(markers seen: {distinct})"
        )

    matrix, mask = cv2.findHomography(
        image_points.reshape(-1, 1, 2),
        world_points.reshape(-1, 1, 2),
        cv2.RANSAC,
        ransac_threshold_cm,
    )
    if matrix is None:
        raise HomographyError(f"findHomography failed (markers seen: {marker_ids})")

    projected = _transform(image_points, matrix)
    errors = np.linalg.norm(projected - world_points, axis=1)
    inlier_mask = (
        mask.reshape(-1).astype(bool) if mask is not None else np.ones(len(errors), bool)
    )

    return Homography(
        matrix=matrix,
        rms_cm=float(np.sqrt(np.mean(errors[inlier_mask] ** 2))),
        max_error_cm=float(errors.max()),
        inliers=int(inlier_mask.sum()),
        total=len(errors),
        marker_ids=distinct,
    )


def base_point(bbox_xyxy: tuple[float, float, float, float]) -> np.ndarray:
    """The ground-contact point of a detection: bottom-centre of its box.

    Tall objects lean away from the nadir under perspective, so a 115mm missile's
    bounding-box CENTROID sits well off its actual footprint and can project into
    the neighbouring zone — which scores zero. The bottom edge is much closer to
    where the object actually touches the ground. Craters are flat enough that it
    makes no difference, so this is safe to use everywhere.
    """
    x1, _, x2, y2 = bbox_xyxy
    return np.array([(x1 + x2) / 2.0, y2], dtype=np.float64)
