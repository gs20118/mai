"""A fake arena and a fake drone camera, for verifying the geometry with no drone.

We render the arena top-down from arena.yaml, then push it through a KNOWN camera
pose and a KNOWN barrel distortion to produce a synthetic "drone frame". The
pipeline then has to recover what we started from. Because ground truth is exact,
this catches the errors that are otherwise invisible until competition day: a
rotated marker map, a sign flip in a distortion coefficient, a zone grid that is
off by one band.

When real footage arrives we will be debugging the footage, not this code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .arena import Arena

_BAND_FILL = {
    "facility": (70, 85, 70),
    "taxiway_a": (95, 95, 100),
    "taxiway_b": (95, 95, 100),
    "runway": (60, 60, 65),
}

# Half-width of the white backing plate under each marker, as a multiple of the
# marker's own side length.
QUIET_ZONE_RATIO = 0.75

# Canvas scale for the demo CLIs. It has to out-resolve the finest capture we
# simulate, or the "high-resolution" strip pass is just upsampling a canvas with no
# more detail to give -- the crops would look identical to the top view's and the
# demo would quietly prove nothing. At 16 px/cm the canvas holds 0.63mm/px, finer
# than the ~0.8mm/px the strip pass samples at. Tests use a coarser canvas to stay
# fast, since they assert on geometry rather than on detail.
DEMO_CANVAS_PX_PER_CM = 16.0

# How much floor to render around the arena. The real markers sit with their outer
# corner flush to the arena corner, so their white quiet zone falls outside the
# arena entirely -- without this margin they have no light background to be
# segmented against and none of them decode.
CANVAS_MARGIN_CM = 30.0


@dataclass
class PlacedObject:
    """A mission object at a known arena location, so tests can assert its zone."""

    kind: str  # a key of arena.targets, e.g. "crater_big" or "uxo_cluster"
    x_cm: float
    y_cm: float


@dataclass
class CameraPose:
    """Drone pose above the arena, in arena centimetres."""

    x_cm: float = 250.0
    y_cm: float = 200.0
    height_cm: float = 300.0
    pitch_deg: float = 0.0  # tilt away from straight-down
    roll_deg: float = 0.0
    yaw_deg: float = 0.0

    def world_to_image(self, camera_matrix: np.ndarray) -> np.ndarray:
        """Ground-truth homography: arena cm -> ideal (undistorted) image px.

        For points on the arena plane (Z=0) the projection collapses to
        K @ [r1 r2 t], which is exactly a homography.
        """
        pitch, roll, yaw = np.deg2rad([self.pitch_deg, self.roll_deg, self.yaw_deg])

        # Camera looking straight down: camera +z is world -z, camera +y is world +y.
        base = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])
        rot_x = cv2.Rodrigues(np.array([pitch, 0.0, 0.0]))[0]
        rot_y = cv2.Rodrigues(np.array([0.0, roll, 0.0]))[0]
        rot_z = cv2.Rodrigues(np.array([0.0, 0.0, yaw]))[0]
        rotation = rot_z @ rot_y @ rot_x @ base

        center = np.array([self.x_cm, self.y_cm, self.height_cm])
        translation = -rotation @ center

        extrinsic = np.column_stack([rotation[:, 0], rotation[:, 1], translation])
        matrix = camera_matrix @ extrinsic
        return matrix / matrix[2, 2]


@dataclass
class SyntheticCapture:
    image: np.ndarray  # the distorted "drone frame"
    topdown: np.ndarray  # the ideal top-down render it came from
    true_world_to_image: np.ndarray  # ground truth, arena cm -> ideal image px
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    px_per_cm: float
    objects: list[PlacedObject] = field(default_factory=list)


def render_topdown(
    arena: Arena,
    objects: list[PlacedObject] | None = None,
    px_per_cm: float = 4.0,
    texture: str = "speckle",
    seed: int = 0,
    margin_cm: float = 30.0,
) -> np.ndarray:
    """Draw the arena straight down: bands, zone markings, ArUco markers, objects.

    `texture` decides whether markerless localisation is even possible, so it is a
    first-class knob rather than cosmetic:

      "speckle" -- surface grain, as a real 3D-printed / painted diorama has. It is
                   random, hence APERIODIC, which both gives SIFT something to match
                   on and breaks the runway's 50cm repetition.
      "none"    -- a perfectly smooth surface whose only content is the repeating
                   zone grid and centreline dashes. This is the adversarial case:
                   featureless AND periodic. Localisation must REFUSE here, not
                   guess a zone.
    """
    # The canvas extends past the arena onto the surrounding floor. That is not
    # cosmetic: the real markers sit with their outer corner flush to the arena
    # corner, so their white quiet zone lies OUTSIDE the arena entirely. Render only
    # the arena and the markers lose the light background they are segmented
    # against, and none of them decode.
    width = int(round((arena.width_cm + 2 * margin_cm) * px_per_cm))
    height = int(round((arena.height_cm + 2 * margin_cm) * px_per_cm))
    canvas = np.full((height, width, 3), 120, dtype=np.uint8)  # concrete floor

    def to_px(point) -> tuple[int, int]:
        return (
            int(round((point[0] + margin_cm) * px_per_cm)),
            int(round((point[1] + margin_cm) * px_per_cm)),
        )

    for zone in arena.zones:
        corners = zone.polygon()
        cv2.rectangle(
            canvas, to_px(corners[0]), to_px(corners[2]), _BAND_FILL[zone.band], -1
        )
        # Zone edges: the straight lines the distortion tuner is judged against.
        cv2.rectangle(canvas, to_px(corners[0]), to_px(corners[2]), (200, 200, 200), 1)

    if texture == "speckle":
        # Texture has to be MULTI-SCALE or the test is meaningless. Fine grain alone
        # (~1mm) averages away to nothing by the time the top view samples at
        # ~2.3mm/px, leaving the reference map with almost no features to match
        # against -- which is what a real smooth surface would also do. Real dioramas
        # carry structure at several scales: print lines, paint variation, weathering,
        # scuffs. The coarse component below (~3cm) is what survives downsampling and
        # so is what markerless localisation actually lives on.
        rng = np.random.default_rng(seed)
        fine = rng.normal(0.0, 6.0, (height, width, 1))

        coarse_px = max(int(round(3.0 * px_per_cm)), 2)
        coarse = rng.normal(
            0.0, 15.0, (height // coarse_px + 2, width // coarse_px + 2, 1)
        )
        coarse = cv2.resize(
            coarse, (width, height), interpolation=cv2.INTER_LINEAR
        )[..., None]

        canvas = np.clip(
            canvas.astype(np.float32) + fine + coarse, 0, 255
        ).astype(np.uint8)

    # Runway centreline dashes, so the render has some texture to feature-match on.
    for zone in arena.runway_zones:
        center_x, center_y = zone.center
        cv2.line(
            canvas,
            to_px((center_x - zone.w * 0.3, center_y)),
            to_px((center_x + zone.w * 0.3, center_y)),
            (240, 240, 240),
            2,
        )

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, arena.dictionary))
    for marker in arena.markers.values():
        side_px = int(round(arena.marker_size_cm * px_per_cm))

        # The white quiet zone is not decoration. ArUco finds a marker by looking
        # for a dark quad against a light background, so a marker pasted straight
        # onto the dark facility band has nothing for its black border to contrast
        # against and simply will not segment. Real markers are printed on white
        # card for exactly this reason -- and if the organizers' markers turn out
        # to lack a white margin, detection at altitude will be unreliable.
        quiet_px = int(round(side_px * QUIET_ZONE_RATIO))
        center_px = to_px(marker.center)
        cv2.rectangle(
            canvas,
            (center_px[0] - quiet_px, center_px[1] - quiet_px),
            (center_px[0] + quiet_px, center_px[1] + quiet_px),
            (255, 255, 255),
            -1,
        )

        tile = cv2.aruco.generateImageMarker(dictionary, marker.id, side_px)
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        if marker.rotation_deg:
            rotation = cv2.getRotationMatrix2D(
                (side_px / 2, side_px / 2), -marker.rotation_deg, 1.0
            )
            tile = cv2.warpAffine(
                tile, rotation, (side_px, side_px), borderValue=(255, 255, 255)
            )
        x0, y0 = to_px(
            (
                marker.center[0] - arena.marker_size_cm / 2,
                marker.center[1] - arena.marker_size_cm / 2,
            )
        )
        canvas[y0 : y0 + side_px, x0 : x0 + side_px] = tile

    for placed in objects or []:
        spec = arena.targets[placed.kind]
        half_w = spec["w_mm"] / 10.0 / 2.0
        half_d = spec["d_mm"] / 10.0 / 2.0
        top_left = to_px((placed.x_cm - half_w, placed.y_cm - half_d))
        bottom_right = to_px((placed.x_cm + half_w, placed.y_cm + half_d))
        if placed.kind.startswith("crater"):
            cv2.ellipse(
                canvas,
                to_px((placed.x_cm, placed.y_cm)),
                (max(int(half_w * px_per_cm), 1), max(int(half_d * px_per_cm), 1)),
                0,
                0,
                360,
                (15, 15, 15),
                -1,
            )
        else:
            cv2.rectangle(canvas, top_left, bottom_right, (40, 200, 230), -1)

    return canvas


def fit_focal_px(
    arena: Arena,
    pose: CameraPose,
    image_size: tuple[int, int],
    margin: float = 0.08,
) -> float:
    """The longest focal length that still frames the arena AND all four markers.

    Fitted against the actually-projected marker corners rather than the arena's
    axis-aligned extent, because any yaw or pitch enlarges the projected footprint
    and it is the corner markers -- not the arena rectangle -- that must stay in
    frame. Getting this wrong silently drops a marker, and the pipeline then
    (correctly) refuses to register the frame at all.

    Two facts about the mission fall out of this, and both are worth knowing before
    planning the flight:

    1. The arena is 500x400cm (5:4) but the sensor is 16:9. Fitting the 400cm DEPTH
       into 2160px of frame height binds long before the 500cm width fills 3840px of
       frame width. A whole-arena shot is height-limited, so its ground sample
       distance is set by 400cm/2160px, not the 500cm/3840px one would assume --
       about 30% coarser, and that lands squarely on the 28mm cluster munition.
    2. Any yaw costs further resolution, because a rotated rectangle needs a bigger
       frame. Squaring the drone up to the arena is worth real pixels.
    """
    width_px, height_px = image_size

    # With K = [[f,0,cx],[0,f,cy],[0,0,1]] the projection is cx + f*(Xc/Zc), so the
    # normalised offsets Xc/Zc are independent of f. Recover them with an identity
    # camera matrix, then solve for the largest f that keeps them all in frame.
    normalized = pose.world_to_image(np.eye(3))
    corners = np.vstack(
        [arena.marker_world_corners(marker_id) for marker_id in arena.markers]
        + [
            np.array(
                [[0.0, 0.0], [arena.width_cm, 0.0],
                 [arena.width_cm, arena.height_cm], [0.0, arena.height_cm]]
            )
        ]
    )
    projected = cv2.perspectiveTransform(
        corners.reshape(-1, 1, 2).astype(np.float64), normalized
    ).reshape(-1, 2)

    max_x = np.abs(projected[:, 0]).max()
    max_y = np.abs(projected[:, 1]).max()
    return float(
        min(
            (width_px / 2.0) * (1.0 - margin) / max_x,
            (height_px / 2.0) * (1.0 - margin) / max_y,
        )
    )


def fit_focal_for_coverage(
    coverage_cm: tuple[float, float],
    pose: CameraPose,
    image_size: tuple[int, int],
    margin: float = 0.05,
) -> float:
    """Focal length that frames a region of the arena, for a nadir-ish pose.

    Used to plan the high-resolution strip passes. The airfield (RW + TW bands) is
    only 500x240cm, so two 4:3 stills covering it reach ~12.6 px/cm -- against the
    ~5-7 px/cm of a whole-arena frame. That is what takes a 28mm cluster munition
    from ~12px to ~35px. The price is that neither still contains a corner marker.
    """
    width_px, height_px = image_size
    px_per_cm = min(
        width_px / coverage_cm[0], height_px / coverage_cm[1]
    ) * (1.0 - margin)
    return px_per_cm * pose.height_cm


def capture(
    arena: Arena,
    pose: CameraPose | None = None,
    objects: list[PlacedObject] | None = None,
    image_size: tuple[int, int] = (3840, 2160),
    focal_px: float | None = None,
    distortion: tuple[float, float, float, float, float] = (-0.28, 0.09, 0.0, 0.0, 0.0),
    render_px_per_cm: float = 8.0,
    texture: str = "speckle",
    seed: int = 0,
    topdown: np.ndarray | None = None,
    margin_cm: float = CANVAS_MARGIN_CM,
) -> SyntheticCapture:
    """Render the arena, project it through a camera, then barrel-distort it.

    `focal_px=None` frames the whole arena automatically. The default distortion is
    a pronounced barrel, far stronger than a real DJI lens after in-camera
    correction. That is on purpose: if the pipeline survives this, mild residual
    distortion will not trouble it.

    Pass an existing `topdown` canvas to shoot several captures of the SAME physical
    arena. That matters for markerless localisation: a re-rendered canvas would draw
    a fresh random speckle, so the low pass and the top view would have no shared
    surface detail to match on -- which is not a property real footage has.
    """
    pose = pose or CameraPose()
    objects = objects or []
    width, height = image_size
    if focal_px is None:
        focal_px = fit_focal_px(arena, pose, image_size)

    if topdown is None:
        topdown = render_topdown(
            arena,
            objects,
            px_per_cm=render_px_per_cm,
            texture=texture,
            seed=seed,
            margin_cm=margin_cm,
        )

    camera_matrix = np.array(
        [[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]]
    )
    world_to_image = pose.world_to_image(camera_matrix)

    # warpPerspective interpolates, it does not integrate, so projecting a canvas
    # that is much denser than the target aliases the fine detail away -- and the
    # first thing to die is the ArUco marker's bit pattern. A real sensor averages
    # over each photosite instead, so mimic that with an INTER_AREA prepass down to
    # roughly the projected scale before warping. Without this the synthetic
    # markers fail to decode for reasons that have nothing to do with the pipeline.
    projected_px_per_cm = focal_px / pose.height_cm
    canvas_px_per_cm = render_px_per_cm
    canvas_width_cm = arena.width_cm + 2 * margin_cm
    canvas_height_cm = arena.height_cm + 2 * margin_cm
    if render_px_per_cm > 1.5 * projected_px_per_cm:
        canvas_px_per_cm = 1.5 * projected_px_per_cm
        topdown_for_warp = cv2.resize(
            topdown,
            (
                max(int(round(canvas_width_cm * canvas_px_per_cm)), 1),
                max(int(round(canvas_height_cm * canvas_px_per_cm)), 1),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        topdown_for_warp = topdown

    # Canvas px -> arena cm: undo the canvas scale, then the margin offset (the
    # canvas origin is on the floor outside the arena, not at the arena corner).
    canvas_to_world = np.array(
        [
            [1.0 / canvas_px_per_cm, 0.0, -margin_cm],
            [0.0, 1.0 / canvas_px_per_cm, -margin_cm],
            [0.0, 0.0, 1.0],
        ]
    )
    ideal = cv2.warpPerspective(
        topdown_for_warp,
        world_to_image @ canvas_to_world,
        image_size,
        flags=cv2.INTER_LINEAR,
    )

    dist_coeffs = np.array(distortion, dtype=np.float64)
    distorted = _apply_distortion(ideal, camera_matrix, dist_coeffs)

    return SyntheticCapture(
        image=distorted,
        topdown=topdown,
        true_world_to_image=world_to_image,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        px_per_cm=render_px_per_cm,
        objects=objects,
    )


def _apply_distortion(
    ideal: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
) -> np.ndarray:
    """Bend an ideal pinhole image into a barrel-distorted one.

    To fill each pixel of the distorted output we need to know where it samples in
    the ideal image, which means inverting the distortion polynomial.
    cv2.undistortPoints does exactly that inversion, so we run it over the output
    pixel grid and remap. This is the precise inverse of what Undistorter does,
    which is what makes the round-trip test meaningful.
    """
    height, width = ideal.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    distorted_pixels = np.stack([grid_x, grid_y], axis=-1).astype(np.float32).reshape(-1, 1, 2)

    ideal_pixels = cv2.undistortPoints(
        distorted_pixels, camera_matrix, dist_coeffs, P=camera_matrix
    ).reshape(height, width, 2)

    return cv2.remap(
        ideal,
        ideal_pixels[..., 0],
        ideal_pixels[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
