"""Per-zone rectified crops, a contact sheet, and the resolution report.

    python -m mai.cli.crop_zones --run outputs/topview_20260714_120000

This is the decision gate for the whole project. The contact sheet shows all 26
zones at their native source resolution; the GSD report says, in pixels, how big
each real target actually is in each zone. Between them they answer the only
question that matters right now: is a 28mm cluster munition resolvable from this
altitude, or does the flight plan have to change?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from mai import viz, zones
from mai.arena import Arena
from mai.homography import Homography


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run directory from build_topview.")
    parser.add_argument("--arena", default=None)
    parser.add_argument(
        "--pad-cm",
        type=float,
        default=5.0,
        help="Padding around each zone, so an object on a boundary is not clipped.",
    )
    parser.add_argument("--cell-px", type=int, default=420, help="Contact sheet cell size.")
    args = parser.parse_args()

    run_dir = Path(args.run)
    with (run_dir / "metadata.json").open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    undistorted = cv2.imread(str(run_dir / "undistorted.jpg"))
    if undistorted is None:
        raise FileNotFoundError(run_dir / "undistorted.jpg")

    arena = Arena.from_yaml(args.arena) if args.arena else Arena.from_yaml()
    solved = Homography(
        matrix=np.array(metadata["homography_image_to_arena_cm"], dtype=np.float64),
        rms_cm=metadata["reprojection_rms_cm"],
        max_error_cm=metadata["reprojection_max_cm"],
        inliers=0,
        total=0,
        marker_ids=metadata["marker_ids"],
    )

    crops, records = zones.crop_all(
        undistorted, arena, solved, run_dir / "zones", pad_cm=args.pad_cm
    )
    report = zones.gsd_report(arena, records, run_dir / "gsd_report.json")

    by_id = {record.zone_id: record for record in records}
    captions = {
        record.zone_id: f"{record.mm_per_px:.2f}mm/px" for record in records
    }
    # Rows follow the physical arena: FA / TW-A / RW / TW-B / FA, top to bottom.
    order = [zone.id for zone in sorted(arena.zones, key=lambda z: (z.y, z.x))]
    sheet = viz.contact_sheet(
        crops, order, columns=5, cell_px=args.cell_px, captions=captions
    )
    cv2.imwrite(str(run_dir / "contact_sheet.jpg"), sheet)

    summary = report["summary"]
    print(f"cropped {len(records)} zones -> {run_dir / 'zones'}")
    print(f"  visible: {summary['zones_visible']}/{summary['zones_total']}")
    print(
        f"  ground sample distance: {summary['mm_per_px_best']}mm/px (best zone) .. "
        f"{summary['mm_per_px_worst']}mm/px (worst zone)"
    )

    print("\nTarget size in pixels, in the WORST-resolved zone:")
    print("  (below ~10px a YOLO-class detector is unreliable; 20px+ is comfortable)")
    for name, pixels in summary["smallest_target_px_worst_zone"].items():
        verdict = "OK" if pixels >= 20 else ("MARGINAL" if pixels >= 10 else "TOO SMALL")
        print(f"    {name:15s} {pixels:6.1f}px   {verdict}")

    unseen = [record.zone_id for record in records if not record.visible]
    if unseen:
        print(f"\n  NOT SEEN by this frame: {unseen}")

    print(f"\n  -> {run_dir / 'contact_sheet.jpg'}   <- look at this")
    print(f"  -> {run_dir / 'gsd_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
