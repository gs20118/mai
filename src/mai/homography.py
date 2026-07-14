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
class AxisConstraint:
    """An image point whose arena X *or* Y is known, but not both.

    The arena's printed border ticks are exactly this: a tick on the left border
    marking the y=80 band boundary pins that image point to Y=80, while its X is
    only "somewhere on the border" and is not known.

    Half a correspondence is still worth having. It is linear in H -- from
    Y = (h2.p)/(h3.p), fixing Y = Y0 gives (h2 - Y0*h3).p = 0 -- so it drops straight
    into the DLT next to the full point correspondences. That is what lets us pin
    down a corner of the arena whose ArUco marker is occluded.
    """

    image_xy: np.ndarray
    axis: int  # 0 fixes X, 1 fixes Y
    value_cm: float


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


def solve_constrained(
    image_points: np.ndarray,
    world_points: np.ndarray,
    axis_constraints: list[AxisConstraint],
    marker_ids: list[int],
    min_markers: int = 2,
) -> Homography:
    """Fit image px -> arena cm from full correspondences PLUS one-axis constraints.

    For the oblique view, where a building buries one corner marker. Three markers
    still fit a homography, and it will look healthy -- ~1cm reprojection error --
    because it is only checking itself where it has evidence. Out in the corner it
    cannot see, it drifts: on this footage, by ~13cm, which is a quarter of a runway
    zone. The printed ticks put evidence exactly there.

    Everything is linear, so this is one DLT and an SVD:
        full correspondence -> 2 rows    (X and Y both known)
        axis constraint     -> 1 row     (only one of them known)

    Coordinates are pre-normalised to keep the system well conditioned. Hartley's
    data-dependent normalisation cannot be used as-is, because an axis constraint has
    no second coordinate to take a centroid over -- so we use a fixed normalisation
    onto roughly [-1, 1] instead, which is enough.
    """
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    world_points = np.asarray(world_points, dtype=np.float64).reshape(-1, 2)
    distinct = sorted(set(marker_ids))

    if len(distinct) < min_markers:
        raise HomographyError(
            f"saw {len(distinct)} marker(s) {distinct}, need at least {min_markers} "
            "even with tick constraints"
        )

    rows_available = 2 * len(image_points) + len(axis_constraints)
    if rows_available < 8:
        raise HomographyError(
            f"only {rows_available} equations for 8 unknowns "
            f"({len(image_points)} points, {len(axis_constraints)} tick constraints)"
        )

    # Fixed normalisation: image px onto ~[-1,1], arena cm onto ~[-1,1].
    scale_u = max(np.abs(image_points[:, 0]).max(), 1.0)
    scale_v = max(np.abs(image_points[:, 1]).max(), 1.0)
    if axis_constraints:
        tick_xy = np.array([c.image_xy for c in axis_constraints], dtype=np.float64)
        scale_u = max(scale_u, np.abs(tick_xy[:, 0]).max())
        scale_v = max(scale_v, np.abs(tick_xy[:, 1]).max())
    normalize_image = np.diag([1.0 / scale_u, 1.0 / scale_v, 1.0])
    normalize_world = np.diag([1.0 / 250.0, 1.0 / 200.0, 1.0])

    rows = []
    for (u, v), (x_cm, y_cm) in zip(image_points, world_points):
        un, vn = u / scale_u, v / scale_v
        xn, yn = x_cm / 250.0, y_cm / 200.0
        rows.append([un, vn, 1, 0, 0, 0, -xn * un, -xn * vn, -xn])
        rows.append([0, 0, 0, un, vn, 1, -yn * un, -yn * vn, -yn])

    for constraint in axis_constraints:
        u, v = constraint.image_xy
        un, vn = u / scale_u, v / scale_v
        if constraint.axis == 0:
            xn = constraint.value_cm / 250.0
            rows.append([un, vn, 1, 0, 0, 0, -xn * un, -xn * vn, -xn])
        else:
            yn = constraint.value_cm / 200.0
            rows.append([0, 0, 0, un, vn, 1, -yn * un, -yn * vn, -yn])

    _, _, right = np.linalg.svd(np.array(rows, dtype=np.float64))
    normalized = right[-1].reshape(3, 3)

    matrix = np.linalg.inv(normalize_world) @ normalized @ normalize_image
    if abs(matrix[2, 2]) < 1e-12:
        raise HomographyError("degenerate homography from the constrained fit")
    matrix = matrix / matrix[2, 2]

    projected = _transform(image_points, matrix)
    errors = np.linalg.norm(projected - world_points, axis=1)

    return Homography(
        matrix=matrix,
        rms_cm=float(np.sqrt(np.mean(errors**2))),
        max_error_cm=float(errors.max()) if len(errors) else 0.0,
        inliers=len(errors),
        total=len(errors),
        marker_ids=distinct,
    )


def axis_residuals(
    solved: Homography, axis_constraints: list[AxisConstraint]
) -> np.ndarray:
    """Error, in cm, of each one-axis constraint under a fitted homography.

    The honest check on an oblique fit. Marker reprojection error only measures the
    fit where the markers are; these measure it where they aren't.
    """
    if not axis_constraints:
        return np.empty(0)
    points = np.array([c.image_xy for c in axis_constraints], dtype=np.float64)
    projected = solved.to_world(points)
    return np.array(
        [
            projected[i, constraint.axis] - constraint.value_cm
            for i, constraint in enumerate(axis_constraints)
        ]
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
