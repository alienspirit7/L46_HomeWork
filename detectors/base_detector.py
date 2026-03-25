"""Shared interface and data classes for all YOLO detectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np


@dataclass
class FrameResult:
    """Result of running detection on a single frame."""

    frame_index: int
    detections: List[Dict]          # [{class_name, confidence, bbox:[x1,y1,x2,y2]}]
    num_detections: int             # filtered to target_classes only
    inference_time_ms: float
    annotated_frame: np.ndarray = field(repr=False)


@dataclass
class VideoResult:
    """Aggregated detection results for one model × one video."""

    model_name: str
    test_file_name: str
    task: str
    target_classes: List[str]
    frame_results: List[FrameResult]

    # Aggregated metrics (computed in evaluation/metrics.py)
    avg_detections_per_frame: float = 0.0
    avg_confidence: float = 0.0
    max_confidence: float = 0.0
    avg_inference_time_ms: float = 0.0
    detections_per_second: float = 0.0
    high_confidence_ratio: float = 0.0   # fraction of detections with conf >= 0.7
    detection_consistency: float = 0.0   # 1/(1+std); 1.0 = perfectly consistent
    total_processing_time_sec: float = 0.0  # wall-clock time for the entire video run


class BaseDetector(ABC):
    """Abstract base class that all YOLO detectors extend."""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        confidence_threshold: float,
        target_classes: List[str],
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes

    @abstractmethod
    def detect_frame(self, frame: np.ndarray) -> FrameResult:
        """Run detection on a single BGR frame and return a FrameResult."""

    def get_name(self) -> str:
        """Return the human-readable model name."""
        return self.model_name
