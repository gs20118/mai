"""Copy-paste augmentation that respects the physics, instead of quietly breaking it.

There are only ~70 UNIQUE UXO instances in the whole dataset -- 8 scenes, each shot twice
by a hovering drone, so the 140 boxes are really 70 objects seen twice. For the rarest
class that is ~21 examples. That is thin for an 18px target, and copy-paste is the
standard answer.

BUT THESE OBJECTS STAND UP, AND THAT CHANGES WHERE THEY MAY BE PASTED.

A nadir view sees an upright object's base AND its top, displaced by
`height x distance_from_nadir / altitude`. So the same missile looks different depending
on where it stands: at RW-01 it leans LEFT, at RW-10 it leans RIGHT, and directly under
the drone it barely leans at all. That lean is not noise -- it is the height cue, and
height is what separates a 92mm missile from a 6mm ball at these pixel sizes.

Now note that the crop's BACKGROUND identifies its zone: RW-01 carries the "60" runway
threshold marking, RW-10 the "27", and each taxiway has its own road geometry. So pasting
a left-leaning missile onto RW-10's background produces an image that cannot physically
exist. The model either wastes capacity on it or, worse, learns that lean is meaningless.
We would be destroying the very signal we most need.

The fix is not to warp the patch -- a 25px patch would only smear. It is to paste only
where the lean it already carries is still correct:

  1. CROSS-SCENE, SAME ZONE. Take the missile from scene A's RW-04 and drop it into
     scenes B..H's RW-04. Same zone, same geometry, same lean -- exact by construction,
     and it multiplies the data by the number of scenes.
  2. MIRRORED, RADIUS-MATCHED. RW-04 (x=175) and RW-07 (x=325) sit symmetrically about
     the nadir, so a horizontally mirrored patch has exactly the right lean at the
     mirrored zone. Free doubling, still no warping.
  3. INTRA-ZONE JITTER. Moving an object within its own zone changes its lean by at most
     ~1.3cm. Safe variety.

Everything here is exact. Nothing is warped, nothing is invented.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .arena import Arena

# Objects are dark props on light asphalt, and (per the labeller) there is no confounding
# context inside the boxes. So a threshold relative to the local background segments them
# cleanly, which is what makes a cut-out possible at all.
DARKNESS_PERCENTILE = 35


@dataclass
class Patch:
    """A cut-out object, and the zone whose geometry its lean belongs to."""

    image: np.ndarray  # BGR, the tight box
    alpha: np.ndarray  # float 0..1, feathered
    cls: int
    zone_id: str
    source: str

    @property
    def size(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]


def cut(image: np.ndarray, box_xyxy, cls: int, zone_id: str, source: str) -> Patch | None:
    """Cut an object out of a crop, with a soft alpha matte."""
    x1, y1, x2, y2 = (int(round(v)) for v in box_xyxy)
    height, width = image.shape[:2]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, width), min(y2, height)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None

    crop = image[y1:y2, x1:x2]
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Threshold against the box's own brightness distribution, not a global constant --
    # the runway is much brighter than the grass, and the same object sits on both.
    cutoff = np.percentile(grey, DARKNESS_PERCENTILE)
    mask = (grey <= cutoff).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # Keep the largest blob: the object, not the scorch marks around it.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.float32)
    if mask.sum() < 12:
        return None

    alpha = cv2.GaussianBlur(mask, (3, 3), 0)  # feather, or the seam becomes the feature
    return Patch(image=crop.copy(), alpha=alpha, cls=cls, zone_id=zone_id, source=source)


class Placer:
    """Decides where a patch may legally go, and puts it there."""

    def __init__(self, arena: Arena, nadir_cm=(245.0, 194.0), tolerance_cm: float = 25.0):
        self.arena = arena
        self.nadir = np.array(nadir_cm, dtype=np.float64)
        self.tolerance = tolerance_cm

    def _offset(self, zone_id: str) -> np.ndarray:
        return np.array(self.arena.zone(zone_id).center) - self.nadir

    def targets(self, zone_id: str) -> list[tuple[str, bool]]:
        """Zones where this patch's lean is still correct, and whether to mirror it.

        The lean points along the vector from the nadir to the object, with a length set
        by the distance. So a patch is valid at any zone with (near enough) the same
        offset vector -- which is its own zone -- or at the zone mirrored through the
        nadir in x, provided we mirror the patch to match.
        """
        source = self._offset(zone_id)
        out: list[tuple[str, bool]] = [(zone_id, False)]

        mirrored = np.array([-source[0], source[1]])
        for zone in self.arena.airfield_zones:
            if zone.id == zone_id:
                continue
            offset = self._offset(zone.id)
            if np.linalg.norm(offset - mirrored) <= self.tolerance:
                out.append((zone.id, True))
        return out

    def paste(
        self,
        background: np.ndarray,
        patch: Patch,
        mirror: bool,
        rng: random.Random,
        margin: int = 6,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
        """Composite the patch somewhere in the crop, and return the new box."""
        image = patch.image[:, ::-1].copy() if mirror else patch.image
        alpha = patch.alpha[:, ::-1].copy() if mirror else patch.alpha

        height, width = background.shape[:2]
        patch_h, patch_w = image.shape[:2]
        if patch_w + 2 * margin >= width or patch_h + 2 * margin >= height:
            return None

        x = rng.randint(margin, width - patch_w - margin)
        y = rng.randint(margin, height - patch_h - margin)

        region = background[y : y + patch_h, x : x + patch_w]

        # Harmonise brightness with the destination, or the model learns to spot pasted
        # patches by their exposure rather than by what they are.
        source_bg = float(np.median(image[alpha < 0.1])) if (alpha < 0.1).any() else 0.0
        target_bg = float(np.median(region))
        shifted = np.clip(
            image.astype(np.float32) + (target_bg - source_bg) * 0.6, 0, 255
        )

        blend = alpha[..., None]
        composed = shifted * blend + region.astype(np.float32) * (1 - blend)
        # A touch of grain, matched to the destination, so the paste is not suspiciously
        # clean against a noisy runway.
        composed += np.random.normal(0, 1.5, composed.shape)

        out = background.copy()
        out[y : y + patch_h, x : x + patch_w] = np.clip(composed, 0, 255).astype(np.uint8)
        return out, (x, y, x + patch_w, y + patch_h)


def to_yolo(box_xyxy, image_shape) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box_xyxy
    height, width = image_shape[:2]
    return (
        (x1 + x2) / 2 / width,
        (y1 + y2) / 2 / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def boxes_overlap(a, b, slack: int = 4) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (
        ax2 + slack < bx1 or bx2 + slack < ax1 or ay2 + slack < by1 or by2 + slack < ay1
    )
