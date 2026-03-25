"""Written conclusion generator — auto-fills winners and statistics."""

import os
from typing import Dict

import numpy as np
import pandas as pd

from config import RESULTS_DIR, TEST_FILES
from detectors.base_detector import VideoResult


def generate_conclusion(
    all_results: Dict[str, Dict[str, VideoResult]],
    table_df: pd.DataFrame,
) -> None:
    """Compute winners and write the conclusion to disk."""
    model_names = list(all_results.keys())
    test_names = list(next(iter(all_results.values())).keys())

    # Total frames analysed (varies per video)
    total_frames = sum(
        len(all_results[model_names[0]][tf].frame_results)
        for tf in test_names
    )

    # Per-model averages across all test files
    avg_fps = _avg_attr(all_results, "detections_per_second")
    avg_hcr = _avg_attr(all_results, "high_confidence_ratio")
    avg_con = _avg_attr(all_results, "detection_consistency")

    speed_winner = max(avg_fps, key=avg_fps.get)
    speed_loser = min(avg_fps, key=avg_fps.get)
    quality_winner = max(avg_hcr, key=avg_hcr.get)
    consistency_winner = max(avg_con, key=avg_con.get)

    speed_pct = (
        ((avg_fps[speed_winner] / avg_fps[speed_loser]) - 1) * 100
        if avg_fps[speed_loser] > 0 else 0.0
    )

    # Count ★ per model
    star_counts = _count_stars(table_df, model_names)
    overall_winner = max(star_counts, key=star_counts.get)

    # Vehicle detection summary
    vehicle_tests = [
        t for t in test_names
        if TEST_FILES[t]["task"] == "vehicle_detection"
    ]
    veh_det = _avg_attr(all_results, "avg_detections_per_frame", vehicle_tests)
    veh_conf = _avg_attr(all_results, "avg_confidence", vehicle_tests)
    veh_fps = _avg_attr(all_results, "detections_per_second", vehicle_tests)

    test_list_str = ", ".join(TEST_FILES[t]["description"] for t in test_names)

    conclusion = f"""\
=== EVALUATION CONCLUSION ===

Models compared: {', '.join(model_names)}
Test videos: {test_list_str}
Frames analysed: {total_frames} total frames across {len(test_names)} videos

SPEED: {speed_winner} averaged {avg_fps[speed_winner]:.1f} FPS — \
{speed_pct:.0f}% faster than {speed_loser}.

DETECTION QUALITY: {quality_winner} achieved a high-confidence ratio of \
{avg_hcr[quality_winner]:.1%}.

CONSISTENCY: {consistency_winner} had the most stable detections \
(score: {avg_con[consistency_winner]:.2f}/1.0).

OVERALL WINNER: {overall_winner} led in \
{star_counts[overall_winner]} out of 8 metric categories.

ON VEHICLE DETECTION ({', '.join(vehicle_tests)}):
{max(veh_det, key=veh_det.get)} detected {veh_det[max(veh_det, key=veh_det.get)]:.1f}/frame. \
{max(veh_conf, key=veh_conf.get)} had best confidence ({veh_conf[max(veh_conf, key=veh_conf.get)]:.2f}). \
{max(veh_fps, key=veh_fps.get)} was fastest ({veh_fps[max(veh_fps, key=veh_fps.get)]:.1f} FPS).
"""

    # Non-vehicle test summary
    non_veh = [t for t in test_names if t not in vehicle_tests]
    if non_veh:
        nv = non_veh[0]
        nv_conf = {mn: all_results[mn][nv].avg_confidence for mn in model_names}
        nv_winner = max(nv_conf, key=nv_conf.get)
        nv_vals = list(nv_conf.values())
        conclusion += f"""\
ON {TEST_FILES[nv]['description'].upper()} ({nv}):
Target classes: {', '.join(TEST_FILES[nv]['target_classes'])}.
{nv_winner} performed best with avg confidence {nv_conf[nv_winner]:.2f}.
Confidence range: {min(nv_vals):.2f}–{max(nv_vals):.2f}.
"""

    print(conclusion)
    path = os.path.join(RESULTS_DIR, "conclusion.txt")
    with open(path, "w") as f:
        f.write(conclusion)
    print(f"Conclusion saved → {path}")


# ── Helpers ───────────────────────────────────────────────────────────

def _avg_attr(
    all_results: Dict[str, Dict[str, VideoResult]],
    attr: str,
    test_subset: list = None,
) -> Dict[str, float]:
    """Average a VideoResult attribute over test files for each model."""
    model_names = list(all_results.keys())
    tests = test_subset or list(next(iter(all_results.values())).keys())
    return {
        mn: float(np.mean([getattr(all_results[mn][tf], attr) for tf in tests]))
        for mn in model_names
    }


def _count_stars(
    table_df: pd.DataFrame, model_names: list
) -> Dict[str, int]:
    """Count ★ symbols per model in the formatted table."""
    counts: Dict[str, int] = {mn: 0 for mn in model_names}
    for _, row in table_df.iterrows():
        for col in table_df.columns[1:]:
            if "★" in str(row[col]):
                for mn in model_names:
                    if mn in col:
                        counts[mn] += 1
                        break
    return counts
