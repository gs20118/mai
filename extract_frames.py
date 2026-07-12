import argparse
from pathlib import Path

import cv2


def variance_of_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_frames(video_path, output_dir, interval_sec, blur_threshold, max_frames):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS from video: {video_path}")

    step = max(int(round(fps * interval_sec)), 1)
    frame_idx = 0
    saved_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step == 0:
            blur_score = variance_of_laplacian(frame)
            if blur_score >= blur_threshold:
                timestamp_sec = frame_idx / fps
                output_path = output_dir / f"frame_{saved_idx:05d}_{timestamp_sec:.1f}s.jpg"
                cv2.imwrite(str(output_path), frame)
                saved_idx += 1

                if max_frames is not None and saved_idx >= max_frames:
                    break

        frame_idx += 1

    cap.release()
    return saved_idx


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract sharp training frames from one drone video."
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument(
        "--output",
        default="datasets/raw_frames",
        help="Directory to save extracted frames.",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=0.5,
        help="Frame sampling interval in seconds.",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=80.0,
        help="Minimum Laplacian variance. Raise this to keep only sharper frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to save.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    saved_count = extract_frames(
        video_path=args.video,
        output_dir=args.output,
        interval_sec=args.interval_sec,
        blur_threshold=args.blur_threshold,
        max_frames=args.max_frames,
    )
    print(f"saved {saved_count} frames to {args.output}")


if __name__ == "__main__":
    main()
