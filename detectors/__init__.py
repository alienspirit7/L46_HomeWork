"""Detectors package — exposes all three YOLO detector classes."""

from .yolov8_detector import YOLOv8Detector
from .yolov9_detector import YOLOv9Detector
from .yolo11_detector import YOLO11Detector

__all__ = ["YOLOv8Detector", "YOLOv9Detector", "YOLO11Detector"]
