"""Arena geometry: zones, bands, and the ArUco marker map.

Every mission is scored per zone ID, never per bounding box, so this module is
the coordinate system the whole project speaks. Detections get projected into
arena centimetres and then resolved to a zone ID here.

Arena frame: origin at the top-left corner, +x right, +y down, units centimetres.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

DEFAULT_ARENA_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "arena.yaml"

# Corner order matches cv2.aruco.detectMarkers: the marker's own top-left,
# top-right, bottom-right, bottom-left. Decoding the ID also recovers the
# marker's orientation, so index i is always the same physical corner no matter
# how the drone is rotated above it.
_MARKER_CORNER_OFFSETS = np.array(
    [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=np.float64
)


@dataclass(frozen=True)
class Zone:
    id: str
    x: float
    y: float
    w: float
    h: float
    band: str

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def contains(self, x_cm: float, y_cm: float) -> bool:
        """Half-open on the far edges, so neighbouring zones never both claim a point."""
        return self.x <= x_cm < self.x2 and self.y <= y_cm < self.y2

    def polygon(self, pad_cm: float = 0.0) -> np.ndarray:
        """Corners as a 4x2 array in arena cm, ordered TL, TR, BR, BL."""
        x1, y1 = self.x - pad_cm, self.y - pad_cm
        x2, y2 = self.x2 + pad_cm, self.y2 + pad_cm
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


@dataclass(frozen=True)
class Marker:
    id: int
    center: tuple[float, float]
    rotation_deg: float

    def world_corners(self, size_cm: float) -> np.ndarray:
        """The marker's four corners in arena cm, in cv2.aruco's corner order."""
        theta = np.deg2rad(self.rotation_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
        offsets = (_MARKER_CORNER_OFFSETS * size_cm) @ rotation.T
        return offsets + np.asarray(self.center, dtype=np.float64)


class ArenaConfigError(ValueError):
    pass


class Arena:
    def __init__(
        self,
        width_cm: float,
        height_cm: float,
        scale: int,
        zones: list[Zone],
        markers: dict[int, Marker],
        marker_size_cm: float,
        dictionary: str,
        targets: dict[str, dict[str, float]],
    ):
        self.width_cm = width_cm
        self.height_cm = height_cm
        self.scale = scale
        self.zones = zones
        self.markers = markers
        self.marker_size_cm = marker_size_cm
        self.dictionary = dictionary
        self.targets = targets
        self._by_id = {zone.id: zone for zone in zones}

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_ARENA_CONFIG) -> "Arena":
        with Path(path).open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)

        arena_cfg = raw["arena"]
        aruco_cfg = raw["aruco"]

        markers = {}
        for marker_id, spec in aruco_cfg["markers"].items():
            markers[int(marker_id)] = Marker(
                id=int(marker_id),
                center=tuple(float(value) for value in spec["center"]),
                rotation_deg=float(spec.get("rotation_deg", 0.0)),
            )

        arena = cls(
            width_cm=float(arena_cfg["width_cm"]),
            height_cm=float(arena_cfg["height_cm"]),
            scale=int(arena_cfg["scale"]),
            zones=[Zone(**zone) for zone in raw["zones"]],
            markers=markers,
            marker_size_cm=float(aruco_cfg["marker_size_cm"]),
            dictionary=str(aruco_cfg["dictionary"]),
            targets=raw.get("targets", {}),
        )
        problems = arena.validate()
        if problems:
            raise ArenaConfigError(
                f"{path} is inconsistent:\n" + "\n".join(f"  - {p}" for p in problems)
            )
        return arena

    # --- zone lookup ----------------------------------------------------------

    def zone(self, zone_id: str) -> Zone:
        try:
            return self._by_id[zone_id]
        except KeyError:
            raise KeyError(f"unknown zone id: {zone_id!r}") from None

    def zone_at(self, x_cm: float, y_cm: float) -> str | None:
        """Zone ID containing an arena point, or None if it falls outside the arena.

        Points exactly on the arena's far edge are nudged inside, so a detection
        at x=500.0 lands in RW-10 rather than falling through the half-open
        interval into None.
        """
        if not (0.0 <= x_cm <= self.width_cm and 0.0 <= y_cm <= self.height_cm):
            return None
        epsilon = 1e-9
        x_cm = min(x_cm, self.width_cm - epsilon)
        y_cm = min(y_cm, self.height_cm - epsilon)
        for zone in self.zones:
            if zone.contains(x_cm, y_cm):
                return zone.id
        return None

    def zones_in_band(self, band: str) -> list[Zone]:
        return [zone for zone in self.zones if zone.band == band]

    @property
    def runway_zones(self) -> list[Zone]:
        """The 10 RW zones. Crater count, UXO count and runway length use only these."""
        return self.zones_in_band("runway")

    @property
    def taxiway_zones(self) -> list[Zone]:
        return self.zones_in_band("taxiway_a") + self.zones_in_band("taxiway_b")

    @property
    def facility_zones(self) -> list[Zone]:
        return self.zones_in_band("facility")

    @property
    def airfield_zones(self) -> list[Zone]:
        """RW + TW: the 20 zones where craters and UXO can be scored."""
        return self.runway_zones + self.taxiway_zones

    # --- markers --------------------------------------------------------------

    def marker_world_corners(self, marker_id: int) -> np.ndarray:
        return self.markers[marker_id].world_corners(self.marker_size_cm)

    # --- validation -----------------------------------------------------------

    def validate(self) -> list[str]:
        """Structural problems with the config, as human-readable strings.

        This is cheap and runs on every load. A silently wrong arena config
        produces confident wrong zone IDs, which is the worst failure mode we
        have — it looks like a working system and scores zero.
        """
        problems: list[str] = []

        ids = [zone.id for zone in self.zones]
        duplicates = {zone_id for zone_id in ids if ids.count(zone_id) > 1}
        if duplicates:
            problems.append(f"duplicate zone ids: {sorted(duplicates)}")

        for zone in self.zones:
            if zone.x < 0 or zone.y < 0 or zone.x2 > self.width_cm or zone.y2 > self.height_cm:
                problems.append(f"{zone.id} escapes the arena bounds")

        # The zones must tile the arena exactly: no gaps (a crater landing in a
        # gap gets no zone) and no overlaps (it would get an arbitrary one).
        total_area = sum(zone.w * zone.h for zone in self.zones)
        arena_area = self.width_cm * self.height_cm
        if abs(total_area - arena_area) > 1e-6:
            problems.append(
                f"zones cover {total_area:g}cm^2 but the arena is {arena_area:g}cm^2 "
                "(gap or overlap)"
            )

        for marker in self.markers.values():
            x_cm, y_cm = marker.center
            if not (0.0 <= x_cm <= self.width_cm and 0.0 <= y_cm <= self.height_cm):
                problems.append(f"marker {marker.id} centre {marker.center} is outside the arena")

        if len(self.markers) < 4:
            problems.append(f"need >=4 markers for a homography, config has {len(self.markers)}")

        return problems

    def __repr__(self) -> str:
        return (
            f"Arena({self.width_cm:g}x{self.height_cm:g}cm, 1:{self.scale}, "
            f"{len(self.zones)} zones, {len(self.markers)} markers)"
        )
