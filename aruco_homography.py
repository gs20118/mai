import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ARUCO_DICTIONARY_NAMES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_6X6_50",
    "DICT_6X6_100",
]


# Initial inspection grid. Adjust these after checking the warped image.
ZONE_LAYOUT_CM = {
    "runway": {
        "prefix": "RW",
        "count": 10,
        "x": 0,
        "y": 160,
        "width": 500,
        "height": 80,
    },
    "taxiway_a": {
        "prefix": "TW-A",
        "count": 5,
        "x": 0,
        "y": 80,
        "width": 500,
        "height": 80,
    },
    "taxiway_b": {
        "prefix": "TW-B",
        "count": 5,
        "x": 0,
        "y": 240,
        "width": 500,
        "height": 80,
    },
}


def get_aruco_dictionary(dictionary_name):
    dictionary_id = getattr(cv2.aruco, dictionary_name)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.Dictionary_get(dictionary_id)


def detect_markers(frame, dictionary_name):
    dictionary = get_aruco_dictionary(dictionary_name)

    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(frame)
    else:
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

    if ids is None:
        ids = np.empty((0, 1), dtype=np.int32)
        corners = []

    return corners, ids.reshape(-1).astype(int).tolist(), rejected


def marker_centers(corners):
    centers = []
    for marker_corners in corners:
        points = marker_corners.reshape(4, 2)
        centers.append(points.mean(axis=0))
    if not centers:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(centers, dtype=np.float32)


def order_points(points):
    points = np.asarray(points, dtype=np.float32)
    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1).reshape(-1)

    ordered = np.array(
        [
            points[np.argmin(point_sum)],
            points[np.argmin(point_diff)],
            points[np.argmax(point_sum)],
            points[np.argmax(point_diff)],
        ],
        dtype=np.float32,
    )

    # If one point is selected twice, fall back to a minimum-area rectangle.
    if len({tuple(point) for point in ordered}) < 4:
        rect = cv2.minAreaRect(points)
        ordered = order_points(cv2.boxPoints(rect))

    return ordered


def choose_corner_points(centers):
    if len(centers) < 4:
        return None
    return order_points(centers)


def hull_area(points):
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
    return float(cv2.contourArea(hull))


def draw_aruco_overlay(frame, corners, ids, centers):
    overlay = frame.copy()
    if corners:
        cv2.aruco.drawDetectedMarkers(overlay, corners, np.array(ids, dtype=np.int32))

    for marker_id, center in zip(ids, centers):
        x, y = int(round(center[0])), int(round(center[1]))
        cv2.circle(overlay, (x, y), 6, (0, 255, 255), -1)
        cv2.putText(
            overlay,
            str(marker_id),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def draw_corner_points(frame, points):
    overlay = frame.copy()
    labels = ["TL", "TR", "BR", "BL"]
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]

    for label, point, color in zip(labels, points, colors):
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(overlay, (x, y), 10, color, -1)
        cv2.putText(
            overlay,
            label,
            (x + 12, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )

    polygon = points.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(overlay, [polygon], isClosed=True, color=(0, 255, 0), thickness=3)
    return overlay


def cm_to_px(point_cm, px_per_cm):
    return int(round(point_cm[0] * px_per_cm)), int(round(point_cm[1] * px_per_cm))


def draw_zone_overlay(image, board_width_cm, board_height_cm, px_per_cm):
    overlay = image.copy()
    height, width = overlay.shape[:2]

    cv2.rectangle(overlay, (0, 0), (width - 1, height - 1), (255, 255, 255), 2)

    for layout in ZONE_LAYOUT_CM.values():
        segment_width = layout["width"] / layout["count"]
        for index in range(layout["count"]):
            x1 = layout["x"] + segment_width * index
            x2 = layout["x"] + segment_width * (index + 1)
            y1 = layout["y"]
            y2 = layout["y"] + layout["height"]

            p1 = cm_to_px((x1, y1), px_per_cm)
            p2 = cm_to_px((x2, y2), px_per_cm)
            cv2.rectangle(overlay, p1, p2, (0, 255, 0), 2)

            if layout["prefix"] == "RW":
                label = f"RW-{index + 1:02d}"
            else:
                label = f"{layout['prefix']}{index + 1}"

            text_x = p1[0] + 8
            text_y = p1[1] + 28
            cv2.putText(
                overlay,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    cv2.putText(
        overlay,
        f"Top-view inspection grid: {board_width_cm:g}cm x {board_height_cm:g}cm",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def candidate_score(centers):
    return len(centers), hull_area(centers)


def scan_frame(cap, frame_index, fps, dictionary_name):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        return None

    corners, ids, _ = detect_markers(frame, dictionary_name)
    centers = marker_centers(corners)
    count, area = candidate_score(centers)
    return {
        "frame": frame.copy(),
        "frame_index": frame_index,
        "time_sec": frame_index / fps,
        "dictionary": dictionary_name,
        "corners": corners,
        "ids": ids,
        "centers": centers,
        "count": count,
        "hull_area": area,
    }


def better_candidate(candidate, best):
    if candidate is None:
        return False
    if best is None:
        return True
    return (candidate["count"], candidate["hull_area"]) > (best["count"], best["hull_area"])


def refine_best_candidate(video_path, coarse_best, fps, refine_window_frames):
    if refine_window_frames <= 0 or coarse_best is None:
        return coarse_best

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return coarse_best

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(coarse_best["frame_index"] - refine_window_frames, 0)
    end = coarse_best["frame_index"] + refine_window_frames
    if frame_count > 0:
        end = min(end, frame_count - 1)

    refined_best = coarse_best
    for frame_index in range(start, end + 1):
        candidate = scan_frame(cap, frame_index, fps, coarse_best["dictionary"])
        if better_candidate(candidate, refined_best):
            refined_best = candidate

    cap.release()
    return refined_best


def inspect_video(video_path, sample_sec, refine_window_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS from video: {video_path}")

    step = max(int(round(fps * sample_sec)), 1)
    frame_index = 0
    best = None
    dictionary_summary = {
        name: {"best_count": 0, "best_hull_area": 0.0, "best_frame_index": None}
        for name in ARUCO_DICTIONARY_NAMES
    }

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % step == 0:
            for dictionary_name in ARUCO_DICTIONARY_NAMES:
                corners, ids, _ = detect_markers(frame, dictionary_name)
                centers = marker_centers(corners)
                count, area = candidate_score(centers)

                summary = dictionary_summary[dictionary_name]
                if (count, area) > (summary["best_count"], summary["best_hull_area"]):
                    summary["best_count"] = count
                    summary["best_hull_area"] = area
                    summary["best_frame_index"] = frame_index

                if (count, area) > (best["count"], best["hull_area"]) if best is not None else True:
                    best = {
                        "frame": frame.copy(),
                        "frame_index": frame_index,
                        "time_sec": frame_index / fps,
                        "dictionary": dictionary_name,
                        "corners": corners,
                        "ids": ids,
                        "centers": centers,
                        "count": count,
                        "hull_area": area,
                    }

        frame_index += 1

    cap.release()
    if best is None:
        raise RuntimeError(f"No frames read from video: {video_path}")

    coarse_frame_index = best["frame_index"]
    coarse_time_sec = best["time_sec"]
    coarse_count = best["count"]
    best = refine_best_candidate(video_path, best, fps, refine_window_frames)
    best["fps"] = fps
    best["sample_step_frames"] = step
    best["dictionary_summary"] = dictionary_summary
    best["coarse_frame_index"] = coarse_frame_index
    best["coarse_time_sec"] = coarse_time_sec
    best["coarse_count"] = coarse_count
    best["refine_window_frames"] = refine_window_frames
    return best


def build_homography(corner_points, board_width_cm, board_height_cm, px_per_cm):
    output_width = int(round(board_width_cm * px_per_cm))
    output_height = int(round(board_height_cm * px_per_cm))
    destination = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    homography, mask = cv2.findHomography(corner_points, destination)
    return homography, mask, output_width, output_height


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def run(args):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco is not available. Install opencv-contrib-python, not only opencv-python."
        )

    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best = inspect_video(video_path, args.sample_sec, args.refine_window_frames)
    centers = best["centers"]
    corner_points = choose_corner_points(centers)

    selected_frame = best["frame"]
    aruco_overlay = draw_aruco_overlay(selected_frame, best["corners"], best["ids"], centers)

    cv2.imwrite(str(output_dir / "selected_frame.jpg"), selected_frame)
    cv2.imwrite(str(output_dir / "aruco_overlay.jpg"), aruco_overlay)

    metadata = {
        "status": "not_enough_markers",
        "video": str(video_path),
        "selected_frame_index": int(best["frame_index"]),
        "selected_time_sec": float(best["time_sec"]),
        "fps": float(best["fps"]),
        "sample_sec": float(args.sample_sec),
        "sample_step_frames": int(best["sample_step_frames"]),
        "coarse_frame_index": int(best["coarse_frame_index"]),
        "coarse_time_sec": float(best["coarse_time_sec"]),
        "coarse_marker_count": int(best["coarse_count"]),
        "refine_window_frames": int(best["refine_window_frames"]),
        "dictionary": best["dictionary"],
        "marker_count": int(best["count"]),
        "marker_ids": [int(marker_id) for marker_id in best["ids"]],
        "marker_centers": centers.astype(float).tolist(),
        "dictionary_summary": best["dictionary_summary"],
        "homography": None,
        "corner_points": None,
        "outputs": {
            "selected_frame": str(output_dir / "selected_frame.jpg"),
            "aruco_overlay": str(output_dir / "aruco_overlay.jpg"),
        },
    }

    if corner_points is not None:
        corner_overlay = draw_corner_points(aruco_overlay, corner_points)
        cv2.imwrite(str(output_dir / "corner_overlay.jpg"), corner_overlay)
        metadata["outputs"]["corner_overlay"] = str(output_dir / "corner_overlay.jpg")
        metadata["corner_points"] = corner_points.astype(float).tolist()

    if corner_points is not None and len(centers) >= 4:
        homography, mask, output_width, output_height = build_homography(
            corner_points=corner_points,
            board_width_cm=args.board_width_cm,
            board_height_cm=args.board_height_cm,
            px_per_cm=args.px_per_cm,
        )

        if homography is not None:
            warped = cv2.warpPerspective(selected_frame, homography, (output_width, output_height))
            zone_overlay = draw_zone_overlay(
                warped,
                board_width_cm=args.board_width_cm,
                board_height_cm=args.board_height_cm,
                px_per_cm=args.px_per_cm,
            )

            cv2.imwrite(str(output_dir / "warped_topview.jpg"), warped)
            cv2.imwrite(str(output_dir / "zone_overlay.jpg"), zone_overlay)

            metadata["status"] = "success"
            metadata["homography"] = homography.astype(float).tolist()
            metadata["homography_mask"] = None if mask is None else mask.astype(int).reshape(-1).tolist()
            metadata["topview_size_px"] = [int(output_width), int(output_height)]
            metadata["board_size_cm"] = [float(args.board_width_cm), float(args.board_height_cm)]
            metadata["px_per_cm"] = float(args.px_per_cm)
            metadata["outputs"]["warped_topview"] = str(output_dir / "warped_topview.jpg")
            metadata["outputs"]["zone_overlay"] = str(output_dir / "zone_overlay.jpg")

    save_json(output_dir / "metadata.json", metadata)
    print(f"status={metadata['status']}")
    print(f"selected_frame={metadata['selected_frame_index']} time={metadata['selected_time_sec']:.2f}s")
    print(f"dictionary={metadata['dictionary']} marker_count={metadata['marker_count']}")
    print(f"output_dir={output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find a frame with visible ArUco markers and generate a top-view homography."
    )
    parser.add_argument(
        "--video",
        default="media/2026_07_12 19_09.mp4",
        help="Input video path.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/aruco_homography",
        help="Directory to save inspection artifacts.",
    )
    parser.add_argument(
        "--sample-sec",
        type=float,
        default=1.0,
        help="Sample one frame every N seconds.",
    )
    parser.add_argument(
        "--refine-window-frames",
        type=int,
        default=20,
        help="After coarse sampling, scan every frame in this +/- window around the best frame.",
    )
    parser.add_argument("--board-width-cm", type=float, default=500.0)
    parser.add_argument("--board-height-cm", type=float, default=400.0)
    parser.add_argument(
        "--px-per-cm",
        type=float,
        default=2.0,
        help="Top-view output scale. Default makes a 1000x800 image for 500x400cm.",
    )
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
