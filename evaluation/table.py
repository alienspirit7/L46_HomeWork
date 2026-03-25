"""Comparison table builder — formats metrics into a grid table and CSV."""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from tabulate import tabulate

from config import RESULTS_DIR
from detectors.base_detector import VideoResult


METRIC_LABELS = [
    ("avg_detections_per_frame", "Avg Detections/Frame"),
    ("avg_confidence",          "Avg Confidence"),
    ("max_confidence",          "Max Confidence"),
    ("avg_inference_time_ms",   "Avg Inference (ms)"),
    ("detections_per_second",   "Detections/Sec (FPS)"),
    ("high_confidence_ratio",   "High-Conf Ratio (≥0.7)"),
    ("detection_consistency",   "Detection Consistency"),
    ("total_processing_time_sec", "Total Processing Time (s)"),
]

# True = higher is better; False = lower is better
HIGHER_IS_BETTER = {
    "avg_detections_per_frame": True,
    "avg_confidence":           True,
    "max_confidence":           True,
    "avg_inference_time_ms":    False,
    "detections_per_second":    True,
    "high_confidence_ratio":    True,
    "detection_consistency":    True,
    "total_processing_time_sec": False,
}


def build_comparison_table(
    all_results: Dict[str, Dict[str, VideoResult]],
) -> pd.DataFrame:
    """Build, print, and export the comparison table.

    Returns:
        DataFrame with the formatted table (used by conclusion generator).
    """
    model_names = list(all_results.keys())
    test_names = list(next(iter(all_results.values())).keys())

    # Build column groups: per-test columns + AVG column per model
    columns: List[str] = []
    for tf in test_names:
        for mn in model_names:
            columns.append(f"{mn} / {tf}")
    for mn in model_names:
        columns.append(f"{mn} / AVG")

    rows: List[List[str]] = []
    for attr, label in METRIC_LABELS:
        raw_vals: List[float] = []
        for tf in test_names:
            for mn in model_names:
                raw_vals.append(getattr(all_results[mn][tf], attr))
        for mn in model_names:
            vals = [getattr(all_results[mn][tf], attr) for tf in test_names]
            raw_vals.append(float(np.mean(vals)))

        # Mark best value per row with ★
        higher = HIGHER_IS_BETTER[attr]
        best_idx = int(np.argmax(raw_vals) if higher else np.argmin(raw_vals))
        formatted = [f"{v:.4f}" for v in raw_vals]
        formatted[best_idx] = formatted[best_idx] + " ★"
        rows.append([label] + formatted)

    headers = ["Metric"] + columns
    table_str = tabulate(rows, headers=headers, tablefmt="grid")
    print("\n" + table_str + "\n")

    # Export CSV
    df = pd.DataFrame(rows, columns=headers)
    csv_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"Table saved → {csv_path}")

    return df
