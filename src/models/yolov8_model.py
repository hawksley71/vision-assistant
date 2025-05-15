import cv2
import numpy as np
import torch
import os
from pathlib import Path
from ultralytics import YOLO
from .base_model import BaseModel
from ..config.settings import MODEL_SETTINGS

class YOLOv8Model(BaseModel):
    """YOLOv8 model implementation."""
    
    def __init__(self, model_name="yolov8n.pt"):
        """Initialize YOLOv8 model."""
        super().__init__()
        self.model_name = model_name
        self.model = self._load_model()
        
    def _load_model(self):
        """Load YOLOv8 model with local weights or download if needed."""
        # Define paths
        model_dir = Path("models/yolov8")
        model_path = model_dir / self.model_name
        
        # Create model directory if it doesn't exist
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load local weights first
        if model_path.exists():
            print(f"Loading local weights from {model_path}")
            return YOLO(str(model_path))
            
        # Download if not found locally
        print(f"Downloading {self.model_name}...")
        model = YOLO(self.model_name)
        
        # Save the downloaded weights
        model.save(str(model_path))
        print(f"Saved weights to {model_path}")
        
        return model
        
    def detect(self, frame):
        """Run detection on a frame."""
        results = self.model(frame, verbose=False)
        return results[0]
        
    def get_detections(self, frame):
        """Process frame and return detections in standard format."""
        results = self.detect(frame)
        detections = []
        
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'score': float(score),
                'class': int(class_id),
                'class_name': results.names[int(class_id)]
            })
            
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw detection boxes and labels on the frame.
        
        Args:
            frame (numpy.ndarray): Input frame in BGR format
            detections (list): List of detections from detect() method
            
        Returns:
            numpy.ndarray: Frame with detections drawn
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['score']
            class_name = det['class_name']
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f'{class_name} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return frame 