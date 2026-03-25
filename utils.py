"""Utility functions for the YOLO comparison pipeline."""

import json
import os

from config import RAW_DIR
from detectors.base_detector import VideoResult


def save_raw_json(video_result: VideoResult) -> None:
    """Persist per-frame detection data and metrics as a JSON file.

    Args:
        video_result: Fully populated VideoResult for one model × one video.
    """
    data = {
        "model_name": video_result.model_name,
        "test_file": video_result.test_file_name,
        "task": video_result.task,
        "target_classes": video_result.target_classes,
        "total_frames": len(video_result.frame_results),
        "metrics": {
            "avg_detections_per_frame": video_result.avg_detections_per_frame,
            "avg_confidence": video_result.avg_confidence,
            "max_confidence": video_result.max_confidence,
            "avg_inference_time_ms": video_result.avg_inference_time_ms,
            "detections_per_second": video_result.detections_per_second,
            "high_confidence_ratio": video_result.high_confidence_ratio,
            "detection_consistency": video_result.detection_consistency,
            "total_processing_time_sec": video_result.total_processing_time_sec,
        },
        "frames": [
            {
                "frame_index": fr.frame_index,
                "num_detections": fr.num_detections,
                "inference_time_ms": fr.inference_time_ms,
                "detections": fr.detections,
            }
            for fr in video_result.frame_results
        ],
    }
    filename = f"{video_result.model_name}_{video_result.test_file_name}.json"
    path = os.path.join(RAW_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Raw data → {path}")
