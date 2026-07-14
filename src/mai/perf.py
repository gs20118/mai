"""Performance footguns that bite silently.

Nothing here is clever. It exists because these cost real milliseconds on the mission
clock and give no indication that they are doing so.
"""

from __future__ import annotations

import cv2


def restore_opencv_threads(threads: int | None = None) -> int:
    """Undo the thread limit that `import ultralytics` imposes on OpenCV.

    Importing ultralytics calls cv2.setNumThreads(1). It does that so OpenCV cannot
    fight its dataloader workers during TRAINING -- but the call happens at import, so
    it silently applies to inference too, and to every other OpenCV operation in the
    process.

    On this machine that makes ArUco detection 4.4x slower: 20.8ms -> 91.1ms at half
    resolution, 57ms -> 249ms at 4K. Nothing warns you. The geometry stage simply
    becomes the most expensive thing in the pipeline, and stays there.

    Call this AFTER importing ultralytics, in any process that also does OpenCV work.
    """
    if threads is None:
        threads = cv2.getNumberOfCPUs()
    cv2.setNumThreads(threads)
    return cv2.getNumThreads()
