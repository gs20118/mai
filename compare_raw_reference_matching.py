import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def build_feature_detector():
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("SIFT is not available. Install opencv-contrib-python.")
    return cv2.SIFT_create(nfeatures=5000)


def resize_for_features(image, max_side):
    height, width = image.shape[:2]
    scale = min(max_side / max(height, width), 1.0)
    if scale == 1.0:
        return image, 1.0
    resized = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))))
    return resized, scale


def detect_features(detector, image, max_side):
    resized, scale = resize_for_features(image, max_side)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return resized, scale, keypoints or [], descriptors


def ratio_test(knn_matches, ratio):
    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            good.append(first)
    return good


def choose_frame_indices(start_frame, frame_count, count, margin_frames, end_frame):
    start = min(max(start_frame + margin_frames, 0), max(frame_count - 1, 0))
    end = frame_count - 1 if end_frame is None else min(max(end_frame, 0), frame_count - 1)
    end = max(end, start)
    if count <= 1:
        return [start]
    return np.linspace(start, end, count).round().astype(int).tolist()


def read_frame(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def transform_point(point, homography):
    point_array = np.float32([[point]]).reshape(1, 1, 2)
    transformed = cv2.perspectiveTransform(point_array, homography).reshape(2)
    return transformed


def clip_point(point, image_shape):
    height, width = image_shape[:2]
    clipped = np.array(point, dtype=np.float32).copy()
    clipped[0] = np.clip(clipped[0], 0, width - 1)
    clipped[1] = np.clip(clipped[1], 0, height - 1)
    return clipped


def match_to_reference(
    frame,
    reference_keypoints,
    reference_descriptors,
    reference_scale,
    detector,
    matcher,
    ratio,
    min_matches,
    ransac_reproj_threshold,
    max_side,
):
    frame_small, _, frame_keypoints, frame_descriptors = detect_features(
        detector, frame, max_side
    )
    result = {
        "frame_small": frame_small,
        "frame_keypoints": frame_keypoints,
        "good_matches": [],
        "inlier_mask": None,
        "homography": None,
        "match_center": None,
        "projected_frame_center": None,
        "projected_frame_center_original": None,
        "status": "no_descriptors",
    }

    if frame_descriptors is None or reference_descriptors is None:
        return result

    knn_matches = matcher.knnMatch(frame_descriptors, reference_descriptors, k=2)
    good_matches = ratio_test(knn_matches, ratio)
    result["good_matches"] = good_matches

    if good_matches:
        target_points_all = np.float32(
            [reference_keypoints[m.trainIdx].pt for m in good_matches]
        )
        result["match_center"] = np.median(target_points_all, axis=0)

    if len(good_matches) < min_matches:
        result["status"] = "not_enough_matches"
        return result

    source_points = np.float32([frame_keypoints[m.queryIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )
    target_points = np.float32([reference_keypoints[m.trainIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )
    homography, inlier_mask = cv2.findHomography(
        source_points,
        target_points,
        cv2.RANSAC,
        ransac_reproj_threshold,
    )
    result["homography"] = homography
    result["inlier_mask"] = inlier_mask
    if homography is None or inlier_mask is None:
        result["status"] = "homography_failed"
        return result

    height, width = frame_small.shape[:2]
    frame_center = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float32)
    result["projected_frame_center"] = transform_point(frame_center, homography)
    result["projected_frame_center_original"] = (
        result["projected_frame_center"] / reference_scale
    )

    inliers = inlier_mask.reshape(-1).astype(bool)
    inlier_count = int(inliers.sum())
    if inlier_count < min_matches:
        result["status"] = "not_enough_inliers"
        return result

    inlier_target_points = target_points.reshape(-1, 2)[inliers]
    result["match_center"] = np.median(inlier_target_points, axis=0)
    result["status"] = "success"
    return result


def draw_comparison_overlay(topview, results):
    overlay = topview.copy()
    cv2.rectangle(overlay, (8, 8), (470, 88), (255, 255, 255), -1)
    cv2.rectangle(overlay, (8, 8), (470, 88), (0, 0, 0), 2)
    cv2.putText(
        overlay,
        "circle: direct top-view match",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "square: raw full-view reference -> top-view",
        (18, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    palette = [
        (255, 80, 80),
        (80, 200, 255),
        (80, 255, 120),
        (255, 180, 60),
        (200, 120, 255),
        (255, 80, 200),
        (120, 255, 255),
        (180, 255, 80),
        (255, 140, 140),
        (140, 180, 255),
    ]

    for result in results:
        color = palette[(result["index"] - 1) % len(palette)]
        direct = result.get("direct_top_center")
        raw = result.get("raw_to_top_center")

        if direct is not None:
            point = clip_point(direct, topview.shape).astype(int)
            cv2.circle(overlay, tuple(point), 13, color, -1)
            cv2.circle(overlay, tuple(point), 17, (0, 0, 0), 3)

        if raw is not None:
            point = clip_point(raw, topview.shape).astype(int)
            cv2.rectangle(overlay, tuple(point - 14), tuple(point + 14), color, 4)
            cv2.rectangle(overlay, tuple(point - 18), tuple(point + 18), (0, 0, 0), 2)

        label_point = raw if raw is not None else direct
        if label_point is None:
            continue
        point = clip_point(label_point, topview.shape)
        label = f"F{result['frame_index']}"
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        text_x = int(np.clip(point[0] + 18, 0, topview.shape[1] - text_size[0] - 4))
        text_y = int(np.clip(point[1] - 12, text_size[1] + 4, topview.shape[0] - 4))
        cv2.putText(
            overlay,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return overlay


def draw_matches_image(frame_result, reference_small, reference_keypoints, max_draw_matches):
    return cv2.drawMatches(
        frame_result["frame_small"],
        frame_result["frame_keypoints"],
        reference_small,
        reference_keypoints,
        frame_result["good_matches"][:max_draw_matches],
        None,
        matchesMask=(
            frame_result["inlier_mask"].reshape(-1).astype(int).tolist()[:max_draw_matches]
            if frame_result["inlier_mask"] is not None
            else None
        ),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def summarize_match(name, match_result, raw_to_top=None):
    inlier_count = (
        int(match_result["inlier_mask"].reshape(-1).sum())
        if match_result["inlier_mask"] is not None
        else 0
    )
    reference_center_original = match_result.get("projected_frame_center_original")
    payload = {
        "status": match_result["status"],
        "match_count": int(len(match_result["good_matches"])),
        "inlier_count": inlier_count,
        "reference_center": (
            None
            if match_result["projected_frame_center"] is None
                else match_result["projected_frame_center"].astype(float).tolist()
        ),
        "reference_center_original": (
            None
            if reference_center_original is None
            else reference_center_original.astype(float).tolist()
        ),
    }
    if raw_to_top is not None and reference_center_original is not None:
        payload["top_center"] = transform_point(reference_center_original, raw_to_top).astype(
            float
        ).tolist()
    else:
        payload["top_center"] = payload["reference_center_original"]
    return name, payload


def run(args):
    total_start = time.perf_counter()
    metadata = load_json(args.metadata)
    raw_to_top = np.array(metadata["homography"], dtype=np.float64)

    topview = cv2.imread(args.topview)
    raw_reference = cv2.imread(args.raw_reference)
    overlay_background = cv2.imread(args.overlay_background)
    if topview is None:
        raise FileNotFoundError(args.topview)
    if raw_reference is None:
        raise FileNotFoundError(args.raw_reference)
    if overlay_background is None:
        overlay_background = topview.copy()

    cap = cv2.VideoCapture(args.video or metadata["video"])
    if not cap.isOpened():
        raise RuntimeError("Failed to open video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(metadata["selected_frame_index"])
    margin_frames = int(round(args.start_after_sec * fps))
    frame_indices = choose_frame_indices(
        start_frame, frame_count, args.count, margin_frames, args.end_frame
    )

    detector = build_feature_detector()
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=80))

    ref_start = time.perf_counter()
    top_small, top_scale, top_keypoints, top_descriptors = detect_features(
        detector, topview, args.max_side
    )
    raw_small, raw_scale, raw_keypoints, raw_descriptors = detect_features(
        detector, raw_reference, args.max_side
    )
    reference_feature_sec = time.perf_counter() - ref_start

    output_dir = Path(args.output_dir)
    direct_dir = output_dir / "direct_top_matches"
    raw_dir = output_dir / "raw_reference_matches"
    frame_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    direct_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    frame_loop_start = time.perf_counter()
    results = []
    for index, frame_index in enumerate(frame_indices, start=1):
        frame_start = time.perf_counter()
        frame = read_frame(cap, frame_index)
        if frame is None:
            continue
        time_sec = frame_index / fps
        stem = f"F{frame_index:04d}_{time_sec:.1f}s"
        frame_path = frame_dir / f"{stem}.jpg"
        cv2.imwrite(str(frame_path), frame)

        direct = match_to_reference(
            frame,
            top_keypoints,
            top_descriptors,
            top_scale,
            detector,
            matcher,
            args.ratio,
            args.min_matches,
            args.ransac_reproj_threshold,
            args.max_side,
        )
        raw = match_to_reference(
            frame,
            raw_keypoints,
            raw_descriptors,
            raw_scale,
            detector,
            matcher,
            args.ratio,
            args.min_matches,
            args.ransac_reproj_threshold,
            args.max_side,
        )

        direct_match_path = direct_dir / f"{stem}_direct_top_sift.jpg"
        raw_match_path = raw_dir / f"{stem}_raw_ref_sift.jpg"
        cv2.imwrite(
            str(direct_match_path),
            draw_matches_image(direct, top_small, top_keypoints, args.max_draw_matches),
        )
        cv2.imwrite(
            str(raw_match_path),
            draw_matches_image(raw, raw_small, raw_keypoints, args.max_draw_matches),
        )

        _, direct_payload = summarize_match("direct_top", direct)
        _, raw_payload = summarize_match("raw_reference", raw, raw_to_top)
        elapsed = time.perf_counter() - frame_start

        results.append(
            {
                "index": index,
                "frame_index": int(frame_index),
                "time_sec": float(time_sec),
                "runtime_sec": float(elapsed),
                "frame_path": str(frame_path),
                "direct_match_path": str(direct_match_path),
                "raw_reference_match_path": str(raw_match_path),
                "direct_top": direct_payload,
                "raw_reference": raw_payload,
                "direct_top_center": direct_payload["top_center"],
                "raw_to_top_center": raw_payload["top_center"],
            }
        )

    cap.release()
    frames_total_sec = time.perf_counter() - frame_loop_start

    overlay = draw_comparison_overlay(overlay_background, results)
    cv2.imwrite(str(output_dir / "comparison_overlay.jpg"), overlay)

    total_sec = time.perf_counter() - total_start
    payload = {
        "status": "success",
        "runtime": {
            "total_sec": float(total_sec),
            "reference_feature_sec": float(reference_feature_sec),
            "frames_total_sec": float(frames_total_sec),
        },
        "video": str(args.video or metadata["video"]),
        "metadata": str(args.metadata),
        "topview": str(args.topview),
        "raw_reference": str(args.raw_reference),
        "overlay_background": str(args.overlay_background),
        "topview_feature_scale": float(top_scale),
        "raw_reference_feature_scale": float(raw_scale),
        "frame_indices": [int(frame_index) for frame_index in frame_indices],
        "results": results,
        "outputs": {
            "comparison_overlay": str(output_dir / "comparison_overlay.jpg"),
            "frames_dir": str(frame_dir),
            "direct_top_matches_dir": str(direct_dir),
            "raw_reference_matches_dir": str(raw_dir),
        },
    }
    save_json(output_dir / "metadata.json", payload)

    direct_success = sum(1 for result in results if result["direct_top"]["status"] == "success")
    raw_success = sum(1 for result in results if result["raw_reference"]["status"] == "success")
    print(f"frames={len(results)} direct_success={direct_success} raw_success={raw_success}")
    print(f"runtime={total_sec:.2f}s")
    print(f"output_dir={output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare direct top-view SIFT matching with raw full-view reference matching."
    )
    parser.add_argument("--metadata", default="outputs/aruco_homography_refined/metadata.json")
    parser.add_argument("--topview", default="outputs/aruco_homography_refined/warped_topview.jpg")
    parser.add_argument(
        "--raw-reference", default="outputs/aruco_homography_refined/selected_frame.jpg"
    )
    parser.add_argument(
        "--overlay-background", default="outputs/aruco_homography_refined/zone_overlay.jpg"
    )
    parser.add_argument("--video", default=None)
    parser.add_argument("--output-dir", default="outputs/raw_reference_comparison")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--start-after-sec", type=float, default=0.8)
    parser.add_argument("--end-frame", type=int, default=549)
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--min-matches", type=int, default=12)
    parser.add_argument("--ransac-reproj-threshold", type=float, default=5.0)
    parser.add_argument("--max-draw-matches", type=int, default=80)
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
