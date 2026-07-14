"""ArUco marker detection, and the mapping from marker IDs to arena coordinates.

The organizers place four markers, one at each arena corner. We do NOT infer
which marker is which from their image positions (the approach in
legacy/aruco_homography.py): ArUco already decodes the ID, so we look each ID up
in the arena's marker map. That removes an assumption about detection order and
drone yaw, and it lets us use all four corners of every marker rather than just
its centre, giving 16 point correspondences instead of 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .arena import Arena

# Every dictionary scan_aruco tries when we do not yet know which one the
# organizers used.
CANDIDATE_DICTIONARIES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_7X7_50",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_36h11",
]


@dataclass(frozen=True)
class Detection:
    id: int
    corners: np.ndarray  # 4x2 image px, in cv2.aruco order (TL, TR, BR, BL)

    @property
    def center(self) -> np.ndarray:
        return self.corners.mean(axis=0)

    @property
    def area_px(self) -> float:
        return float(cv2.contourArea(self.corners.astype(np.float32)))


def build_detector(dictionary_name: str) -> cv2.aruco.ArucoDetector:
    """A detector tuned for markers that are SMALL in frame.

    This is the binding constraint and it is easy to underestimate. To frame the
    whole 500x400cm arena the drone must sit high enough that a 10cm corner marker
    spans only ~40 pixels of a 4K frame. OpenCV's defaults are tuned for markers
    that fill much more of the image, and at 40px they start dropping detections --
    and dropping even one marker means the frame cannot be registered at all.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    parameters = cv2.aruco.DetectorParameters()

    # Default minMarkerPerimeterRate of 0.03 means a marker must have a perimeter
    # of at least 3% of the image's longest side: on 4K that is ~29px per side,
    # leaving almost no headroom under a 40px marker. Drop the floor so a small
    # marker is not discarded before it is ever decoded.
    parameters.minMarkerPerimeterRate = 0.01

    # Sweep the adaptive threshold over a finer range of window sizes. A small
    # marker occupies few pixels, so the window that cleanly separates its black
    # border from the background is smaller than the default sweep considers.
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 4

    # With only ~7px per marker cell, the bit-sampling grid needs more slack before
    # a cell is called black or white.
    parameters.perspectiveRemovePixelPerCell = 8
    parameters.maxErroneousBitsInBorderRate = 0.4

    # Subpixel corner refinement: free accuracy, and corner precision is exactly
    # what the homography's reprojection error is made of.
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 50
    parameters.cornerRefinementMinAccuracy = 0.01

    return cv2.aruco.ArucoDetector(dictionary, parameters)


def detect(
    image: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    keep_ids: set[int] | None = None,
) -> list[Detection]:
    """Detect markers, optionally discarding IDs that are not in the arena map.

    The competition hall may well contain other ArUco markers (other teams'
    equipment, signage). An unexpected ID fed into the homography would wreck it,
    so anything not in the arena's marker map is dropped.
    """
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return []

    detections = []
    for marker_corners, marker_id in zip(corners, ids.reshape(-1).tolist()):
        marker_id = int(marker_id)
        if keep_ids is not None and marker_id not in keep_ids:
            continue
        detections.append(
            Detection(id=marker_id, corners=marker_corners.reshape(4, 2).astype(np.float64))
        )
    return sorted(detections, key=lambda detection: detection.id)


def correspondences(
    detections: list[Detection], arena: Arena, use_centers_only: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Pair detected marker corners with their known arena coordinates.

    Returns (image_points Nx2 px, world_points Nx2 cm). With four markers this is
    16 correspondences; `use_centers_only` falls back to 4, which is the bare
    minimum for a homography and only worth using if marker_size_cm is unknown.
    """
    image_points: list[np.ndarray] = []
    world_points: list[np.ndarray] = []

    for detection in detections:
        if detection.id not in arena.markers:
            continue
        if use_centers_only:
            image_points.append(detection.center.reshape(1, 2))
            world_points.append(np.asarray(arena.markers[detection.id].center).reshape(1, 2))
        else:
            image_points.append(detection.corners)
            world_points.append(arena.marker_world_corners(detection.id))

    if not image_points:
        return np.empty((0, 2)), np.empty((0, 2))
    return np.vstack(image_points), np.vstack(world_points)


def scan_dictionaries(
    image: np.ndarray, dictionary_names: list[str] | None = None
) -> dict[str, list[int]]:
    """Which dictionaries yield detections in this frame, and which IDs.

    Used once, to discover what the organizers actually put in the arena. Slow by
    design (it tries every dictionary), so it never runs in the mission path.
    """
    results: dict[str, list[int]] = {}
    for name in dictionary_names or CANDIDATE_DICTIONARIES:
        try:
            detections = detect(image, build_detector(name))
        except AttributeError:
            continue  # dictionary not present in this OpenCV build
        if detections:
            results[name] = sorted(detection.id for detection in detections)
    return results
