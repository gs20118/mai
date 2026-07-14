"""Frame sources: a video file or a directory of stills, behind one interface.

The mission may end up shooting stills rather than video (a 4K clip can take
longer to transfer off the drone than the entire 180s mission allows), so nothing
downstream should care which it got.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


@dataclass
class Frame:
    image: np.ndarray
    index: int
    source: str
    time_sec: float | None = None

    @property
    def label(self) -> str:
        if self.time_sec is None:
            return f"{Path(self.source).stem}"
        return f"F{self.index:05d}_{self.time_sec:.1f}s"


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian. Higher is sharper; motion blur tanks it."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def iter_frames(source: str | Path, sample_sec: float = 0.5) -> Iterator[Frame]:
    """Yield frames from a video, an image, or a directory of images.

    `sample_sec` only applies to video.
    """
    source = Path(source)

    if source.is_dir():
        paths = sorted(
            path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not paths:
            raise FileNotFoundError(f"no images in {source}")
        for index, path in enumerate(paths):
            image = cv2.imread(str(path))
            if image is not None:
                yield Frame(image=image, index=index, source=str(path))
        return

    if not source.exists():
        raise FileNotFoundError(source)

    if source.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(source))
        if image is None:
            raise RuntimeError(f"failed to read image: {source}")
        yield Frame(image=image, index=0, source=str(source))
        return

    if source.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"unsupported source type: {source}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"video reports invalid fps: {source}")

    step = max(int(round(fps * sample_sec)), 1)
    index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if index % step == 0:
                yield Frame(
                    image=image, index=index, source=str(source), time_sec=index / fps
                )
            index += 1
    finally:
        capture.release()
