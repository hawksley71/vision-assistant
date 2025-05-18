# Model Weights

This directory contains model weights for object detection.

## Directory Structure

```
models/
└── yolov8/          # YOLOv8 model weights
    ├── yolov8n.pt   # Nano model
    └── yolov8s.pt   # Small model
```

## Usage

The model weights are automatically downloaded on first run and cached locally. You can also manually download the weights:

1. YOLOv8 weights: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

## Version Control

- Model weights are excluded from git via `.gitignore`
- The `models/` directory structure is preserved
- Weights are downloaded and cached on first run

## Supported Models

- YOLOv8n (nano): ~6MB
- YOLOv8s (small): ~22MB 