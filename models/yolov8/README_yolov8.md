# YOLOv8 Model Folder

This folder is required for the project to run, as it is the default location for the YOLOv8 model weights file (`yolov8n.pt`).

## How to Obtain yolov8n.pt

The model file is **not included in the repository** due to its size. To download the required model file, run:

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8/yolov8n.pt
```

Alternatively, follow the instructions in the main `README.md` for more details.

## Note
- This folder is kept in the repository to preserve the required structure.
- The `.gitignore` is set to ignore model weights, so you must download the file manually if setting up the project from scratch. 