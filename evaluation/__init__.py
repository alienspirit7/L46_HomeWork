"""Evaluation package — metrics computation and report generation."""

from .metrics import compute_video_metrics
from .report import generate_report

__all__ = ["compute_video_metrics", "generate_report"]
