"""End-to-end: frame -> undistort -> ArUco -> homography -> top-view.

Frame selection matters. A frame with a great homography but heavy motion blur is
useless for detecting a 21-pixel object, and a razor-sharp frame with a bad
homography is worse than useless because it produces *confident wrong* zone IDs.
So we gate on geometry first (all markers present, reprojection RMS under
threshold) and only then rank the survivors by sharpness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import aruco, frames, homography, ticks
from .arena import Arena
from .aruco import Detection
from .frames import Frame
from .homography import Homography
from .undistort import Undistorter


@dataclass
class Registration:
    frame: Frame
    undistorted: np.ndarray
    detections: list[Detection]
    homography: Homography
    sharpness: float

    @property
    def marker_ids(self) -> list[int]:
        return [detection.id for detection in self.detections]


@dataclass
class Rejection:
    frame_index: int
    reason: str
    marker_ids: list[int]
    rms_cm: float | None = None


def register(
    frame: Frame,
    arena: Arena,
    undistorter: Undistorter,
    detector: cv2.aruco.ArucoDetector,
    use_centers_only: bool = False,
) -> Registration:
    """Undistort a frame and solve its homography, or raise HomographyError."""
    undistorted = undistorter(frame.image)
    detections = aruco.detect(undistorted, detector, keep_ids=set(arena.markers))
    image_points, world_points = aruco.correspondences(
        detections, arena, use_centers_only=use_centers_only
    )
    solved = homography.solve(
        image_points, world_points, [detection.id for detection in detections]
    )
    return Registration(
        frame=frame,
        undistorted=undistorted,
        detections=detections,
        homography=solved,
        sharpness=frames.sharpness(undistorted),
    )


def select_best(
    source: str | Path,
    arena: Arena,
    undistorter: Undistorter,
    sample_sec: float = 0.5,
    min_markers: int = 4,
    max_rms_cm: float = 1.0,
    use_centers_only: bool = False,
) -> tuple[Registration | None, list[Rejection]]:
    """Scan a source and return the sharpest frame that is also well registered.

    Returns (best, rejections). The rejection list is the diagnostic you read when
    nothing passes: it tells you whether the markers were never seen (framing or
    altitude problem) or were seen but reprojected badly (marker map, distortion
    profile, or a rotated marker).
    """
    detector = aruco.build_detector(arena.dictionary)
    best: Registration | None = None
    rejections: list[Rejection] = []

    for frame in frames.iter_frames(source, sample_sec=sample_sec):
        try:
            candidate = register(
                frame, arena, undistorter, detector, use_centers_only=use_centers_only
            )
        except homography.HomographyError as error:
            rejections.append(
                Rejection(frame.index, str(error), marker_ids=[])
            )
            continue

        if len(candidate.detections) < min_markers:
            rejections.append(
                Rejection(
                    frame.index,
                    f"saw {len(candidate.detections)} markers, need {min_markers}",
                    candidate.marker_ids,
                )
            )
            continue

        if candidate.homography.rms_cm > max_rms_cm:
            rejections.append(
                Rejection(
                    frame.index,
                    f"reprojection RMS {candidate.homography.rms_cm:.2f}cm "
                    f"exceeds {max_rms_cm}cm",
                    candidate.marker_ids,
                    candidate.homography.rms_cm,
                )
            )
            continue

        if best is None or candidate.sharpness > best.sharpness:
            best = candidate

    return best, rejections


def register_robust(
    frame: Frame,
    arena: Arena,
    undistorter: Undistorter,
    detector: cv2.aruco.ArucoDetector,
    aruco_scale: float = 0.5,
    max_tick_error_cm: float = 1.5,
) -> tuple[np.ndarray, Homography, int, int]:
    """Register a nadir frame, escalating through three levels rather than giving up.

    Returns (undistorted, homography, markers_used, ticks_used).

    LEVEL 1 -- half-resolution ArUco. The markers are 102px, so at half scale they are
    still 51px, well clear of the ~40px floor the tuned detector handles, and the work
    drops ~4x (145ms -> ~40ms). This is the fast path and it covers most frames.

    LEVEL 2 -- full-resolution retry. Downscaling costs a little marker detectability, so
    when the fast path comes back short of four, look again at full resolution before
    concluding anything is actually missing.

    LEVEL 3 -- markers + printed ticks. On test video 9 the drone framed slightly low and
    marker 4 was CLIPPED BY THE BOTTOM EDGE of the sensor: its centre sits 40px from the
    border and the marker is 110px wide, so half of it is simply not in the image. ArUco
    needs a complete quad, so it cannot decode -- at any resolution. Demanding four
    markers means that entire video registers zero frames and scores zero, and there is
    nothing wrong with it that a slightly higher hover would not have fixed.

    The arena prints its own ground truth, and we already solved this for the oblique
    view: each border tick fixes ONE coordinate, which is a linear constraint that drops
    straight into the DLT beside the marker corners. Three markers plus the ticks
    register the frame accurately, and -- crucially -- the tick residual is an INDEPENDENT
    check on the result, because the markers cannot vouch for the corner they cannot see.
    """
    undistorted = undistorter(frame.image)

    def detect_at(scale: float):
        if scale >= 1.0:
            return aruco.detect(undistorted, detector, keep_ids=set(arena.markers))
        small = cv2.resize(
            undistorted, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        found = aruco.detect(small, detector, keep_ids=set(arena.markers))
        return [Detection(id=d.id, corners=d.corners / scale) for d in found]

    detections = detect_at(aruco_scale)
    if len(detections) < homography.REQUIRED_MARKERS and aruco_scale < 1.0:
        detections = detect_at(1.0)

    image_points, world_points = aruco.correspondences(detections, arena)
    marker_ids = [d.id for d in detections]

    if len(detections) >= homography.REQUIRED_MARKERS:
        return undistorted, homography.solve(image_points, world_points, marker_ids), len(detections), 0

    if len(detections) < 2:
        raise homography.HomographyError(
            f"only {len(detections)} marker(s); need >=2 even with the printed ticks"
        )

    seed, _ = cv2.findHomography(
        image_points.reshape(-1, 1, 2), world_points.reshape(-1, 1, 2)
    )
    if seed is None:
        raise homography.HomographyError("seed homography failed")

    found = ticks.find(undistorted, seed, arena)
    constraints = ticks.constraints(found)
    if len(constraints) < 4:
        raise homography.HomographyError(
            f"{len(detections)} markers and only {len(constraints)} ticks: "
            "not enough evidence to pin the missing corner"
        )

    solved = homography.solve_constrained(
        image_points, world_points, constraints, marker_ids
    )
    residual = float(np.abs(homography.axis_residuals(solved, constraints)).max())
    if residual > max_tick_error_cm:
        raise homography.HomographyError(
            f"tick residual {residual:.2f}cm exceeds {max_tick_error_cm}cm -- the fit "
            "does not agree with the arena's printed boundaries"
        )
    return undistorted, solved, len(detections), len(constraints)


def native_px_per_cm(arena: Arena, solved: Homography) -> float:
    """A top-view scale that matches the source resolution rather than guessing one.

    Sampled at every zone centre and taken as the median, so one badly oblique
    corner cannot drag the whole canvas down.
    """
    samples = [
        solved.source_px_per_cm(np.array(zone.center)) for zone in arena.zones
    ]
    return float(np.median(samples))


def warp(
    undistorted: np.ndarray, arena: Arena, solved: Homography, px_per_cm: float
) -> np.ndarray:
    size = (
        int(round(arena.width_cm * px_per_cm)),
        int(round(arena.height_cm * px_per_cm)),
    )
    return cv2.warpPerspective(
        undistorted, solved.topview_matrix(px_per_cm), size, flags=cv2.INTER_LINEAR
    )
