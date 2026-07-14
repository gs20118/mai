"""Registering frames that contain no ArUco markers, against ones that do.

Resolution comes from covering less arena, not from flying lower: ground sample
distance is (arena covered / pixels), so a wider lens at a lower altitude that
still frames the whole arena buys nothing. And since the markers sit at the four
corners, any frame containing all four must span the full 400cm depth -- which
caps it at ~1.3mm/px. More resolution and visible markers are therefore mutually
exclusive, and a high-resolution pass over the runway strip MUST be registered by
something other than markers.

The arena is planar, so any two views of it are related by a homography. Once the
whole-arena still is ArUco-registered it becomes a map in arena coordinates, and
every subsequent frame localises against that map by feature matching.

THE FAILURE MODE THIS MODULE EXISTS TO PREVENT. The runway is ten identical 50cm
zones with regular centreline dashes. Feature matching on a periodic structure can
converge on a homography shifted by exactly one zone -- with a healthy inlier count
and a low reprojection error. It returns a confident, well-formed answer that is
off by one zone, which scores zero. Every gate below is aimed at that. When we
cannot register a frame safely we raise, and the caller falls back to the
marker-registered top view rather than emitting a wrong zone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .arena import Arena
from .homography import Homography


class LocalizationError(RuntimeError):
    """Registration was not safe to trust. Fail closed; never guess a zone."""


@dataclass
class Gates:
    """Every one of these is a way for a periodic runway to lie to us."""

    min_matches: int = 30
    min_inliers: int = 20
    min_inlier_ratio: float = 0.25
    max_rms_cm: float = 2.0

    # A high-resolution pass must actually be higher resolution than the top view,
    # and not absurdly so. Expressed as a multiple of the reference's px/cm.
    min_scale_ratio: float = 1.2
    max_scale_ratio: float = 6.0

    # The drone is flown deliberately; a wildly rotated solution is a bad fit, not
    # a surprising flight path.
    max_rotation_deg: float = 35.0

    # The frame must land on the arena, and be a sane quad once it gets there.
    min_inside_arena: float = 0.75

    # Ambiguity: if shifting the solution by one runway zone still explains a
    # comparable share of the putative matches, the scene is too periodic to
    # localise safely and we refuse. This is the gate that catches the
    # off-by-one-zone lie.
    max_alias_ratio: float = 0.5
    alias_tolerance_cm: float = 3.0


@dataclass
class Localization:
    homography: Homography  # low-altitude frame px -> arena cm
    matches: int
    inliers: int
    px_per_cm: float
    rotation_deg: float
    inside_arena: float
    alias_ratio: float  # 0 = unambiguous, ->1 = periodic and untrustworthy
    zones: list[str] = field(default_factory=list)

    @property
    def inlier_ratio(self) -> float:
        return self.inliers / max(self.matches, 1)


class Reference:
    """An ArUco-anchored map of the arena that markerless frames localise against.

    Built from the whole-arena top view, whose homography came from the four corner
    markers. Its features are computed once and reused for every frame -- the
    mission clock is 180 seconds and SIFT on a 4K image is not cheap.
    """

    def __init__(
        self,
        topview: np.ndarray,
        px_per_cm: float,
        arena: Arena,
        max_side: int = 1600,
        n_features: int = 8000,
        contrast_threshold: float = 0.015,
    ):
        self.image = topview
        self.px_per_cm = px_per_cm
        self.arena = arena
        # The arena is a low-contrast surface: grey asphalt, matte PLA. SIFT's default
        # contrastThreshold of 0.04 discards most of the faint surface structure that
        # is the only thing a markerless frame has to match on, so we lower it. More
        # weak features is the right trade here -- RANSAC and the gates below can
        # afford to reject bad ones, but they cannot conjure features that were never
        # detected.
        self._detector = cv2.SIFT_create(
            nfeatures=n_features, contrastThreshold=contrast_threshold
        )

        self.scale = min(max_side / max(topview.shape[:2]), 1.0)
        small = (
            cv2.resize(
                topview,
                (
                    int(round(topview.shape[1] * self.scale)),
                    int(round(topview.shape[0] * self.scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )
            if self.scale < 1.0
            else topview
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        self.keypoints, self.descriptors = self._detector.detectAndCompute(gray, None)

    @property
    def feature_count(self) -> int:
        return 0 if self.descriptors is None else len(self.descriptors)

    def canvas_to_arena(self) -> np.ndarray:
        """Reference canvas px -> arena cm. A pure scale, by construction."""
        return np.diag([1.0 / self.px_per_cm, 1.0 / self.px_per_cm, 1.0])

    def features(self, image: np.ndarray, max_side: int = 1600):
        scale = min(max_side / max(image.shape[:2]), 1.0)
        small = (
            cv2.resize(
                image,
                (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
            if scale < 1.0
            else image
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._detector.detectAndCompute(gray, None)
        return scale, keypoints, descriptors


def localize(
    frame: np.ndarray,
    reference: Reference,
    gates: Gates | None = None,
    ratio: float = 0.75,
    ransac_threshold_cm: float = 2.0,
    max_side: int = 1600,
) -> Localization:
    """Register an undistorted markerless frame against the reference map.

    Raises LocalizationError rather than returning an untrustworthy answer.
    """
    gates = gates or Gates()
    arena = reference.arena

    if reference.descriptors is None or reference.feature_count < gates.min_matches:
        raise LocalizationError(
            f"reference has only {reference.feature_count} features; the arena surface "
            "is too smooth to match against"
        )

    frame_scale, frame_keypoints, frame_descriptors = reference.features(frame, max_side)
    if frame_descriptors is None or len(frame_descriptors) < gates.min_matches:
        raise LocalizationError("frame has too few features (blurred, or featureless)")

    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    knn = matcher.knnMatch(frame_descriptors, reference.descriptors, k=2)

    # Lowe's ratio test is itself a periodicity filter: a feature on a repeating
    # runway matches two reference locations about equally well, so the ratio
    # approaches 1 and the match is discarded. That is protective -- but it means a
    # periodic surface yields FEW matches rather than wrong ones, which is why the
    # min_matches gate below is doing real work.
    good = [
        first
        for pair in knn
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < ratio * second.distance
    ]
    if len(good) < gates.min_matches:
        raise LocalizationError(
            f"only {len(good)} distinct matches (need {gates.min_matches}). Either the "
            "surface is too smooth to match, or it is so repetitive that the ratio test "
            "rejected everything. Both mean we cannot place this frame safely."
        )

    frame_points = np.float32([frame_keypoints[m.queryIdx].pt for m in good])
    reference_points = np.float32([reference.keypoints[m.trainIdx].pt for m in good])

    # Work directly in arena cm so the RANSAC threshold is a physical distance.
    arena_points = _apply(
        reference_points / reference.scale, reference.canvas_to_arena()
    )
    full_frame_points = frame_points / frame_scale

    matrix, mask = cv2.findHomography(
        full_frame_points.reshape(-1, 1, 2),
        arena_points.reshape(-1, 1, 2),
        cv2.RANSAC,
        ransac_threshold_cm,
    )
    if matrix is None:
        raise LocalizationError("findHomography failed on the putative matches")

    inlier_mask = mask.reshape(-1).astype(bool)
    inliers = int(inlier_mask.sum())
    if inliers < gates.min_inliers:
        raise LocalizationError(
            f"{inliers} inliers, need {gates.min_inliers}"
        )
    inlier_ratio = inliers / len(good)
    if inlier_ratio < gates.min_inlier_ratio:
        raise LocalizationError(
            f"inlier ratio {inlier_ratio:.2f} below {gates.min_inlier_ratio}; the matches "
            "do not agree on a single placement"
        )

    projected = _apply(full_frame_points, matrix)
    errors = np.linalg.norm(projected - arena_points, axis=1)
    rms_cm = float(np.sqrt(np.mean(errors[inlier_mask] ** 2)))
    if rms_cm > gates.max_rms_cm:
        raise LocalizationError(f"reprojection RMS {rms_cm:.2f}cm exceeds {gates.max_rms_cm}cm")

    solved = Homography(
        matrix=matrix,
        rms_cm=rms_cm,
        max_error_cm=float(errors[inlier_mask].max()),
        inliers=inliers,
        total=len(good),
        marker_ids=[],  # markerless, by definition
    )

    height, width = frame.shape[:2]
    center = np.array([width / 2.0, height / 2.0])
    px_per_cm = solved.source_px_per_cm(solved.to_world(center.reshape(1, 2))[0])
    scale_ratio = px_per_cm / reference.px_per_cm
    if not (gates.min_scale_ratio <= scale_ratio <= gates.max_scale_ratio):
        raise LocalizationError(
            f"implied scale {scale_ratio:.2f}x the reference is outside "
            f"[{gates.min_scale_ratio}, {gates.max_scale_ratio}]x -- this is not a "
            "plausible high-resolution pass"
        )

    rotation_deg = _rotation_deg(solved)
    if abs(rotation_deg) > gates.max_rotation_deg:
        raise LocalizationError(
            f"implied rotation {rotation_deg:+.0f}deg exceeds "
            f"{gates.max_rotation_deg}deg; the fit is bad, not the flight path"
        )

    quad = solved.to_world(
        np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    )
    inside = _fraction_inside(quad, arena)
    if inside < gates.min_inside_arena:
        raise LocalizationError(
            f"only {inside:.0%} of the frame lands inside the arena (need "
            f"{gates.min_inside_arena:.0%})"
        )

    alias_ratio = _alias_ratio(
        full_frame_points, arena_points, matrix, arena, inliers, gates
    )
    if alias_ratio > gates.max_alias_ratio:
        raise LocalizationError(
            f"AMBIGUOUS: shifting the solution by one runway zone still explains "
            f"{alias_ratio:.0%} as many matches. The surface here is too periodic to "
            "localise safely -- refusing rather than risking an off-by-one-zone answer."
        )

    zones = sorted(
        {
            zone_id
            for zone_id in (arena.zone_at(*point) for point in _sample_quad(quad))
            if zone_id is not None
        }
    )

    return Localization(
        homography=solved,
        matches=len(good),
        inliers=inliers,
        px_per_cm=px_per_cm,
        rotation_deg=rotation_deg,
        inside_arena=inside,
        alias_ratio=alias_ratio,
        zones=zones,
    )


def ground_correspondences(
    frame: np.ndarray,
    reference: Reference,
    seed: np.ndarray,
    exclude_bands: tuple[str, ...] = ("facility",),
    ratio: float = 0.75,
    ransac_px: float = 3.0,
    max_side: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """SIFT correspondences from a frame to the reference map: (image px, arena cm).

    A second, independent way to pin down a frame -- and a far denser one than the
    fiducials. Four markers give 16 correspondences and the printed ticks give at most
    12 half-constraints; this gives HUNDREDS of full ones, spread across the whole
    board. That matters because fiducials are individually occludable (a building
    already buries one marker) whereas no single obstruction hides hundreds of
    features.

    Two details do most of the work:

    PRE-WARP. Matching a 45-degree view straight against a top-down map asks SIFT to
    absorb the entire perspective change, which is past what it tolerates -- it found
    211 inliers and landed 3.3cm out. Rectifying the frame first with the (imperfect)
    seed homography leaves SIFT only the residual error to cope with: 591 inliers, and
    1.7cm. The seed's own error does not propagate, because we map the matches back to
    source-image pixels afterwards.

    EXCLUDE THE BUILDINGS. A homography describes a PLANE. The facility bands hold
    30-50cm structures, whose features sit well off that plane and drag the fit toward
    a systematic bias. Masking them out took the error from 1.7cm to 1.1cm. The
    markers and ticks, which sit on the border, constrain those bands instead.
    """
    px_per_cm = reference.px_per_cm
    arena = reference.arena
    scale = np.diag([px_per_cm, px_per_cm, 1.0])
    to_canvas = scale @ seed
    size = (
        int(round(arena.width_cm * px_per_cm)),
        int(round(arena.height_cm * px_per_cm)),
    )
    rectified = cv2.warpPerspective(frame, to_canvas, size)

    mask = np.full(size[::-1], 255, np.uint8)
    for zone in arena.zones:
        if zone.band in exclude_bands:
            x1, y1 = int(zone.x * px_per_cm), int(zone.y * px_per_cm)
            x2, y2 = int(zone.x2 * px_per_cm), int(zone.y2 * px_per_cm)
            mask[y1:y2, x1:x2] = 0

    detector = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.015)
    frame_kp, frame_desc = detector.detectAndCompute(
        cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY), mask
    )
    ref_kp, ref_desc = detector.detectAndCompute(
        cv2.cvtColor(reference.image, cv2.COLOR_BGR2GRAY), mask
    )
    if frame_desc is None or ref_desc is None or len(frame_desc) < 20:
        raise LocalizationError("too few features to match against the reference map")

    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    good = [
        first
        for pair in matcher.knnMatch(frame_desc, ref_desc, k=2)
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < ratio * second.distance
    ]
    if len(good) < 30:
        raise LocalizationError(f"only {len(good)} SIFT matches against the reference map")

    canvas_points = np.float32([frame_kp[m.queryIdx].pt for m in good])
    reference_points = np.float32([ref_kp[m.trainIdx].pt for m in good])
    _, inlier_mask = cv2.findHomography(
        canvas_points.reshape(-1, 1, 2),
        reference_points.reshape(-1, 1, 2),
        cv2.RANSAC,
        ransac_px,
    )
    if inlier_mask is None:
        raise LocalizationError("RANSAC failed on the reference-map matches")
    keep = inlier_mask.reshape(-1).astype(bool)
    if keep.sum() < 30:
        raise LocalizationError(f"only {int(keep.sum())} inliers against the reference map")

    # Back to SOURCE image pixels, so the seed's error never enters the final fit.
    image_points = _apply(canvas_points[keep], np.linalg.inv(to_canvas))
    arena_points = reference_points[keep] / px_per_cm
    return image_points, arena_points


def check_anchors(
    solved: Homography,
    frame_objects_px: np.ndarray,
    arena_objects_cm: np.ndarray,
    tolerance_cm: float = 15.0,
) -> float:
    """Fraction of the frame's objects that land on a known object from the top view.

    The strongest defence against periodicity, once a detector exists to feed it.
    Craters and UXO are placed at random, so their spatial pattern is unique and
    APERIODIC: repeating asphalt can fool SIFT, but a random scatter of craters
    cannot. A registration shifted by one zone will put the objects in the wrong
    places, and this catches it where the geometric gates cannot.

    Returns 1.0 when every detected object matches a known one.
    """
    if len(frame_objects_px) == 0 or len(arena_objects_cm) == 0:
        return 0.0
    projected = solved.to_world(np.asarray(frame_objects_px, dtype=np.float64))
    known = np.asarray(arena_objects_cm, dtype=np.float64)
    distances = np.linalg.norm(projected[:, None, :] - known[None, :, :], axis=2)
    return float((distances.min(axis=1) < tolerance_cm).mean())


def _apply(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)


def _rotation_deg(solved: Homography) -> float:
    """Yaw implied by the fit, read off the arena x-axis."""
    origin = solved.to_image(np.array([[250.0, 200.0]]))[0]
    along_x = solved.to_image(np.array([[260.0, 200.0]]))[0]
    delta = along_x - origin
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def _fraction_inside(quad: np.ndarray, arena: Arena) -> float:
    samples = _sample_quad(quad)
    inside = [
        0.0 <= x <= arena.width_cm and 0.0 <= y <= arena.height_cm for x, y in samples
    ]
    return float(np.mean(inside))


def _sample_quad(quad: np.ndarray, steps: int = 8) -> np.ndarray:
    """A grid of points across the frame's arena footprint."""
    top = np.linspace(quad[0], quad[1], steps)
    bottom = np.linspace(quad[3], quad[2], steps)
    return np.vstack([np.linspace(t, b, steps) for t, b in zip(top, bottom)])


def _alias_ratio(
    frame_points: np.ndarray,
    arena_points: np.ndarray,
    matrix: np.ndarray,
    arena: Arena,
    best_inliers: int,
    gates: Gates,
) -> float:
    """How well a one-zone-shifted placement also explains the matches.

    The runway repeats every 50cm. If a solution translated by exactly that period
    fits nearly as many of the putative matches as the winner does, then the
    evidence genuinely does not distinguish the two, and any answer we give is a
    coin flip between adjacent zones.
    """
    runway = arena.runway_zones
    if not runway:
        return 0.0
    period = runway[0].w

    projected = _apply(frame_points, matrix)
    best = 0
    for shift in (-period, period):
        shifted = projected + np.array([shift, 0.0])
        errors = np.linalg.norm(shifted - arena_points, axis=1)
        best = max(best, int((errors < gates.alias_tolerance_cm).sum()))
    return best / max(best_inliers, 1)
