"""YOLO Video Object Detection Comparison — main entry point.

Runs three YOLO models (v8n, v9c, 11n) on three test videos,
processing every frame, and generates a full evaluation report.
"""

import os
import time as time_mod
from typing import Dict

from config import (
    CONFIDENCE_THRESHOLD,
    MODELS,
    MODELS_DIR,
    RESULTS_DIR,
    ANNOTATED_DIR,
    RAW_DIR,
    TEST_FILES,
)
from detectors import YOLOv8Detector, YOLOv9Detector, YOLO11Detector
from detectors.base_detector import BaseDetector, VideoResult
from evaluation.metrics import compute_video_metrics
from evaluation.report import generate_report
from utils import save_raw_json
from video_processor import iterate_frames, get_video_info, VideoWriter


DETECTOR_MAP = {
    "YOLOv8n": YOLOv8Detector,
    "YOLOv9c": YOLOv9Detector,
    "YOLO11n": YOLO11Detector,
}


def _create_dirs() -> None:
    """Ensure all output directories exist."""
    for d in (RESULTS_DIR, ANNOTATED_DIR, RAW_DIR, MODELS_DIR):
        os.makedirs(d, exist_ok=True)


def _run_model_on_video(
    model_name: str,
    model_path: str,
    test_name: str,
    test_cfg: dict,
) -> VideoResult:
    """Process every frame of one video with one model."""
    vid_info = get_video_info(test_cfg["path"])
    total = vid_info["total_frames"]
    print(f"    Video: {total} frames, "
          f"{vid_info['fps']:.0f} FPS, "
          f"{vid_info['duration_sec']:.1f} sec")

    detector: BaseDetector = DETECTOR_MAP[model_name](
        model_path=model_path,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        target_classes=test_cfg["target_classes"],
    )

    vid_path = os.path.join(ANNOTATED_DIR, f"{model_name}_{test_name}.mp4")
    frame_results = []
    t0 = time_mod.perf_counter()

    with VideoWriter(vid_path, int(vid_info["fps"]),
                     vid_info["width"], vid_info["height"]) as writer:
        for idx, frame in iterate_frames(test_cfg["path"]):
            fr = detector.detect_frame(frame)
            fr.frame_index = idx
            writer.write(fr.annotated_frame)
            fr.annotated_frame = None          # free memory
            frame_results.append(fr)

            if (idx + 1) % 100 == 0 or idx == total - 1:
                print(f"    Frame {idx+1:>5}/{total}  "
                      f"({fr.num_detections} det, "
                      f"{fr.inference_time_ms:.0f} ms)")

    elapsed = time_mod.perf_counter() - t0
    result = compute_video_metrics(
        model_name, test_name, test_cfg["task"],
        test_cfg["target_classes"], frame_results, elapsed,
    )
    print(f"    ✓ {len(frame_results)} frames in {elapsed:.1f}s "
          f"({len(frame_results)/elapsed:.1f} FPS)")
    print(f"    Annotated video → {vid_path}")
    save_raw_json(result)
    return result


def main() -> None:
    _create_dirs()
    all_results: Dict[str, Dict[str, VideoResult]] = {}
    t_start = time_mod.perf_counter()

    for model_name, model_path in MODELS.items():
        print(f"\n{'='*60}\n  Model: {model_name}\n{'='*60}")
        all_results[model_name] = {}
        for test_name, test_cfg in TEST_FILES.items():
            print(f"\n  ▸ {test_name}: {test_cfg['description']}")
            all_results[model_name][test_name] = _run_model_on_video(
                model_name, model_path, test_name, test_cfg)

    print(f"\n{'='*60}\n  Generating comparison report …\n{'='*60}\n")
    generate_report(all_results)
    print(f"\n✅  Done! Total pipeline time: "
          f"{time_mod.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
