"""Central configuration for the YOLO Video Object Detection Comparison project."""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILES_DIR = os.path.join(BASE_DIR, "test_files")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ANNOTATED_DIR = os.path.join(RESULTS_DIR, "annotated")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ─── Detection settings ──────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.4  # Lower threshold to catch more detections

# ─── Models ───────────────────────────────────────────────────────────────────
MODELS = {
    "YOLOv8n": os.path.join(MODELS_DIR, "yolov8n.pt"),
    "YOLOv9c": os.path.join(MODELS_DIR, "yolov9c.pt"),
    "YOLO11n": os.path.join(MODELS_DIR, "yolo11n.pt"),
}

# ─── Test files ───────────────────────────────────────────────────────────────
TEST_FILES = {
    "test1": {
        "path": os.path.join(TEST_FILES_DIR, "test1.mp4"),
        "task": "vehicle_detection",
        "target_classes": ["car", "truck", "bus", "motorcycle", "bicycle"],
        "description": "Road traffic (daytime) — vehicle detection",
    },
    "test2": {
        "path": os.path.join(TEST_FILES_DIR, "test2.mp4"),
        "task": "vehicle_detection",
        "target_classes": ["car", "truck", "bus", "motorcycle", "bicycle"],
        "description": "Road traffic at night — vehicle detection",
    },
    "test3": {
        "path": os.path.join(TEST_FILES_DIR, "test3.mp4"),
        "task": "cyclist_pedestrian_detection",
        "target_classes": ["person", "bicycle"],
        "description": "Cyclists and pedestrians detection",
    },
}


