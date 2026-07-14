"""Barrel distortion correction.

A homography can model any planar projection but it cannot model radial lens
distortion, so distortion must be removed *before* the homography is solved or
it leaks straight into the zone assignment as a position error that grows toward
the frame edges — exactly where the corner zones live.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import yaml

DEFAULT_CAMERA_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "camera.yaml"

DISTORTION_KEYS = ("k1", "k2", "p1", "p2", "k3")


@dataclass(frozen=True)
class CameraProfile:
    name: str
    image_size: tuple[int, int]  # (width, height) the values were tuned at
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    alpha: float = 0.0

    @classmethod
    def load(
        cls, profile: str, path: str | Path = DEFAULT_CAMERA_CONFIG
    ) -> "CameraProfile":
        with Path(path).open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
        profiles = raw["profiles"]
        if profile not in profiles:
            raise KeyError(
                f"camera profile {profile!r} not in {path}; have {sorted(profiles)}"
            )
        spec = dict(profiles[profile])
        spec["image_size"] = tuple(int(value) for value in spec["image_size"])
        return cls(name=profile, **spec)

    def save(self, path: str | Path = DEFAULT_CAMERA_CONFIG) -> None:
        """Write this profile back into the YAML, leaving other profiles untouched.

        yaml.safe_dump discards comments, and camera.yaml's header is the only place
        the tuning workflow is written down. So we lift the leading comment block
        out first and put it back afterwards.
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        header = "".join(
            line
            for line in text.splitlines(keepends=True)[
                : next(
                    (
                        index
                        for index, line in enumerate(text.splitlines())
                        if line.strip() and not line.lstrip().startswith("#")
                    ),
                    0,
                )
            ]
        )

        raw = yaml.safe_load(text)
        raw["profiles"][self.name] = {
            "image_size": list(self.image_size),
            "fx": round(self.fx, 3),
            "fy": round(self.fy, 3),
            "cx": round(self.cx, 3),
            "cy": round(self.cy, 3),
            "k1": round(self.k1, 6),
            "k2": round(self.k2, 6),
            "p1": round(self.p1, 6),
            "p2": round(self.p2, 6),
            "k3": round(self.k3, 6),
            "alpha": round(self.alpha, 3),
        }
        body = yaml.safe_dump(raw, sort_keys=False, default_flow_style=False)
        path.write_text(header + body, encoding="utf-8")

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def dist_coeffs(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)

    @property
    def has_distortion(self) -> bool:
        return any(abs(getattr(self, key)) > 1e-12 for key in DISTORTION_KEYS)

    def scaled_to(self, width: int, height: int) -> "CameraProfile":
        """Rescale intrinsics to a different resolution of the same sensor crop.

        Tuning at 4K and then running on 1080p footage would otherwise apply an
        fx/cx that are 2x too large. Distortion coefficients are dimensionless
        (defined on normalised coordinates) so they carry over untouched.
        """
        tuned_width, tuned_height = self.image_size
        if (width, height) == (tuned_width, tuned_height):
            return self
        scale_x = width / tuned_width
        scale_y = height / tuned_height
        return replace(
            self,
            image_size=(width, height),
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
        )


class Undistorter:
    """Applies one profile to many frames, building the remap tables only once.

    We are on a 180-second mission clock, and initUndistortRectifyMap is far more
    expensive than the cv2.remap that uses its output.
    """

    def __init__(
        self,
        profile: CameraProfile,
        preserve_resolution: bool = True,
        max_output_scale: float = 2.0,
    ):
        self.profile = profile
        self.preserve_resolution = preserve_resolution
        self.max_output_scale = max_output_scale
        self._cache: dict[tuple[int, int], tuple] = {}

    def _maps_for(self, width: int, height: int):
        key = (width, height)
        if key not in self._cache:
            profile = self.profile.scaled_to(width, height)
            camera_matrix = profile.camera_matrix
            dist_coeffs = profile.dist_coeffs

            # alpha=1 keeps the whole field of view, which we need because the
            # corner ArUco markers live right at the frame edge and alpha=0 would
            # crop them away. But undistorting a barrel image makes it bulge
            # outward, and squeezing that bulge back into the original canvas
            # shrinks the focal length -- i.e. it silently DOWNSAMPLES. On a 28mm
            # target that is resolution we cannot spare.
            #
            # So we let the output canvas grow instead: ask for the new camera
            # matrix, see how much focal length it wants to give away, and enlarge
            # the output by exactly that factor to win it back.
            probe, _ = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (width, height), profile.alpha, (width, height)
            )
            scale = 1.0
            if self.preserve_resolution and probe[0, 0] > 0:
                scale = float(
                    np.clip(camera_matrix[0, 0] / probe[0, 0], 1.0, self.max_output_scale)
                )

            out_size = (int(round(width * scale)), int(round(height * scale)))
            new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (width, height), profile.alpha, out_size
            )
            map_x, map_y = cv2.initUndistortRectifyMap(
                camera_matrix,
                dist_coeffs,
                None,
                new_camera_matrix,
                out_size,
                cv2.CV_16SC2,
            )
            self._cache[key] = (profile, new_camera_matrix, map_x, map_y, out_size)
        return self._cache[key]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Undistort a frame. Returns the input untouched if the profile is distortion-free."""
        if not self.profile.has_distortion:
            return image
        height, width = image.shape[:2]
        _, _, map_x, map_y, _ = self._maps_for(width, height)
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    def new_camera_matrix(self, width: int, height: int) -> np.ndarray:
        """The camera matrix that undistorted pixels live in."""
        if not self.profile.has_distortion:
            return self.profile.scaled_to(width, height).camera_matrix
        _, new_camera_matrix, _, _, _ = self._maps_for(width, height)
        return new_camera_matrix

    def output_size(self, width: int, height: int) -> tuple[int, int]:
        """Size of the undistorted frame, which may be larger than the input."""
        if not self.profile.has_distortion:
            return (width, height)
        return self._maps_for(width, height)[4]

    def undistort_points(self, points: np.ndarray, width: int, height: int) -> np.ndarray:
        """Map Nx2 distorted image points into the undistorted frame.

        Cheaper and lossless compared to undistorting the whole image and
        re-detecting, so ArUco corners could be corrected this way instead. We
        undistort the image anyway because the zone crops need to come from a
        rectified source, but this is here for callers that only need points.
        """
        if not self.profile.has_distortion:
            return np.asarray(points, dtype=np.float64)
        profile, new_camera_matrix, _, _, _ = self._maps_for(width, height)
        source = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(
            source, profile.camera_matrix, profile.dist_coeffs, P=new_camera_matrix
        )
        return undistorted.reshape(-1, 2)
