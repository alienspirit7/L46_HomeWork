"""Metric computation for video detection results."""

from typing import List

import numpy as np

from detectors.base_detector import FrameResult, VideoResult


def compute_video_metrics(
    model_name: str,
    test_file_name: str,
    task: str,
    target_classes: List[str],
    frame_results: List[FrameResult],
    total_processing_time_sec: float = 0.0,
) -> VideoResult:
    """Aggregate per-frame detection data into a single VideoResult.

    Computes 7 key metrics + total processing time:
        1. avg_detections_per_frame
        2. avg_confidence
        3. max_confidence
        4. avg_inference_time_ms
        5. detections_per_second
        6. high_confidence_ratio  (fraction of detections with conf >= 0.7)
        7. detection_consistency  (1 / (1 + std_dev(per_frame_count)))
    """
    per_frame_counts = [fr.num_detections for fr in frame_results]
    all_confidences = [
        det["confidence"]
        for fr in frame_results
        for det in fr.detections
    ]
    inference_times = [fr.inference_time_ms for fr in frame_results]

    # 1. Average detections per frame
    avg_det = float(np.mean(per_frame_counts)) if per_frame_counts else 0.0

    # 2. Average confidence
    avg_conf = float(np.mean(all_confidences)) if all_confidences else 0.0

    # 3. Max confidence
    max_conf = float(np.max(all_confidences)) if all_confidences else 0.0

    # 4. Average inference time (ms)
    avg_time = float(np.mean(inference_times)) if inference_times else 0.0

    # 5. Detections per second (model throughput)
    det_per_sec = 1000.0 / avg_time if avg_time > 0 else 0.0

    # 6. High-confidence ratio (>=0.7)
    if all_confidences:
        high = sum(1 for c in all_confidences if c >= 0.7)
        high_ratio = high / len(all_confidences)
    else:
        high_ratio = 0.0

    # 7. Detection consistency: 1 / (1 + std_dev)
    if len(per_frame_counts) > 1:
        std = float(np.std(per_frame_counts, ddof=0))
        consistency = 1.0 / (1.0 + std)
    else:
        consistency = 1.0

    return VideoResult(
        model_name=model_name,
        test_file_name=test_file_name,
        task=task,
        target_classes=target_classes,
        frame_results=frame_results,
        avg_detections_per_frame=avg_det,
        avg_confidence=avg_conf,
        max_confidence=max_conf,
        avg_inference_time_ms=avg_time,
        detections_per_second=det_per_sec,
        high_confidence_ratio=high_ratio,
        detection_consistency=consistency,
        total_processing_time_sec=total_processing_time_sec,
    )
