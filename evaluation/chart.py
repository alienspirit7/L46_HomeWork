"""Bar chart generator — grouped metrics chart per test file."""

import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR, TEST_FILES
from detectors.base_detector import VideoResult


MODEL_COLORS = {
    "YOLOv8n": "#4C72B0",
    "YOLOv9c": "#DD8452",
    "YOLO11n": "#55A868",
}

CHART_METRICS = [
    "avg_confidence",
    "high_confidence_ratio",
    "detection_consistency",
]
CHART_METRIC_LABELS = ["Avg Confidence", "High-Conf Ratio", "Consistency"]


def save_chart(all_results: Dict[str, Dict[str, VideoResult]]) -> None:
    """Create and save a grouped bar chart with one subplot per test file."""
    model_names = list(all_results.keys())
    test_names = list(next(iter(all_results.values())).keys())

    fig, axes = plt.subplots(
        1, len(test_names), figsize=(6 * len(test_names), 5))
    if len(test_names) == 1:
        axes = [axes]

    bar_width = 0.22
    x = np.arange(len(CHART_METRICS))

    for ax, tf in zip(axes, test_names):
        for i, mn in enumerate(model_names):
            vr = all_results[mn][tf]
            vals = [getattr(vr, m) for m in CHART_METRICS]
            offset = (i - 1) * bar_width
            ax.bar(x + offset, vals, bar_width,
                   label=mn, color=MODEL_COLORS.get(mn, "#999999"))

        ax.set_xticks(x)
        ax.set_xticklabels(CHART_METRIC_LABELS, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(TEST_FILES[tf]["description"], fontsize=11)
        ax.legend(fontsize=8)

    fig.suptitle("YOLO Model Comparison — Key Metrics",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    chart_path = os.path.join(RESULTS_DIR, "metrics_chart.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved → {chart_path}")
