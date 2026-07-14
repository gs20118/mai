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

from . import aruco, frames, homography
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
