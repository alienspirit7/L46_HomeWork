"""YOLOv8n detector implementation."""

import time

import cv2
import numpy as np
from ultralytics import YOLO

from .base_detector import BaseDetector, FrameResult


class YOLOv8Detector(BaseDetector):
    """Detector using YOLOv8-nano."""

    def __init__(self, model_path: str, confidence_threshold: float,
                 target_classes: list):
        super().__init__("YOLOv8n", model_path,
                         confidence_threshold, target_classes)
        self.model = YOLO(model_path)
        self.model.fuse()  # merge Conv+BN layers for faster CPU inference

    def detect_frame(self, frame: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            verbose=False,
            stream=False,
        )
        inference_ms = (time.perf_counter() - t0) * 1000

        detections = []
        for box in results[0].boxes:
            class_name = self.model.names[int(box.cls)]
            if class_name not in self.target_classes:
                continue
            detections.append({
                "class_name": class_name,
                "confidence": float(box.conf),
                "bbox": [int(x) for x in box.xyxy[0].tolist()],
            })

        annotated = self._draw_boxes(frame.copy(), detections)

        return FrameResult(
            frame_index=0,
            detections=detections,
            num_detections=len(detections),
            inference_time_ms=inference_ms,
            annotated_frame=annotated,
        )

    @staticmethod
    def _draw_boxes(frame: np.ndarray, detections: list) -> np.ndarray:
        color_map: dict = {}
        for det in detections:
            cls = det["class_name"]
            if cls not in color_map:
                h = hash(cls) % (256 ** 3)
                color_map[cls] = (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)
            x1, y1, x2, y2 = det["bbox"]
            color = color_map[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls} {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame
