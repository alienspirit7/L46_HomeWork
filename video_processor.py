"""Video frame iteration and annotated video output.

Frames are yielded one at a time (generator) to avoid loading entire
videos into memory.  The annotated-video writer accepts frames
incrementally as well.
"""

import os
from typing import Generator, Optional, Tuple

import cv2
import numpy as np


def iterate_frames(video_path: str) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Yield ``(frame_index, bgr_frame)`` for every frame in a video.

    Args:
        video_path: Path to the input video file.

    Yields:
        Tuple of (frame_index, BGR numpy array).

    Raises:
        FileNotFoundError: If *video_path* does not exist.
        RuntimeError: If the video cannot be opened.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, frame
        idx += 1

    cap.release()


def get_video_info(video_path: str) -> dict:
    """Return basic metadata about a video file.

    Returns:
        Dict with keys: total_frames, fps, width, height, duration_sec.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    info = {
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_sec"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


class VideoWriter:
    """Context-manager wrapper around cv2.VideoWriter for incremental writes."""

    def __init__(self, output_path: str, fps: int, width: int, height: int):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        self._path = output_path

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
