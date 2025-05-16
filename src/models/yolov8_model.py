import cv2
import numpy as np
import torch
import os
from pathlib import Path
from ultralytics import YOLO
from .base_model import BaseModel
from ..config.settings import MODEL_SETTINGS, PATHS

class YOLOv8Model(BaseModel):
    """YOLOv8 model implementation."""
    
    def __init__(self, model_name="yolov8n.pt"):
        """Initialize YOLOv8 model."""
        super().__init__()
        self.model_name = model_name
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the YOLOv8 model."""
        try:
            # Check if model file exists
            model_path = Path(PATHS['models']['yolov8'])
            if not model_path.exists():
                raise RuntimeError(f"Error: YOLOv8 model not found at {model_path}")
                
            print(f"[DEBUG] Loading YOLOv8 model from {model_path}")
            model = YOLO(str(model_path))
            
            # Validate model with a dummy input
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            model(dummy_input)
            print("[DEBUG] Model validation successful")
            
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load YOLOv8 model: {str(e)}")
            raise
        
    def detect(self, frame):
        """Run detection on a frame."""
        if frame is None or not isinstance(frame, np.ndarray):
            print("[DEBUG] Invalid frame input")
            return None
            
        try:
            # Ensure frame is in the correct format
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                print(f"[DEBUG] Invalid frame shape: {frame.shape}")
                return None
                
            results = self.model(frame, verbose=False)
            if not results or len(results) == 0:
                print("[DEBUG] No results from model")
                return None
                
            return results[0]  # Return first result (single image)
        except Exception as e:
            print(f"[DEBUG] Error in YOLOv8 detection: {str(e)}")
            return None
        
    def get_detections(self, frame):
        """Process frame and return detections in standard format."""
        results = self.detect(frame)
        detections = []
        
        if results is None:
            return detections
            
        try:
            if hasattr(results, 'boxes') and results.boxes is not None:
                boxes = results.boxes
                for i in range(len(boxes)):
                    try:
                        box = boxes[i]
                        # Handle both tensor and numpy array formats
                        if hasattr(box.xyxy, 'cpu'):
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        else:
                            x1, y1, x2, y2 = box.xyxy[0]
                            
                        if hasattr(box.conf, 'cpu'):
                            score = box.conf[0].cpu().numpy()
                        else:
                            score = box.conf[0]
                            
                        if hasattr(box.cls, 'cpu'):
                            class_id = int(box.cls[0].cpu().numpy())
                        else:
                            class_id = int(box.cls[0])
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(score),
                            'class': class_id,
                            'class_name': results.names[class_id]
                        })
                    except Exception as e:
                        print(f"[DEBUG] Error processing detection {i}: {str(e)}")
                        continue
        except Exception as e:
            print(f"[DEBUG] Error processing detections: {str(e)}")
            
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
            conf = det['confidence']
            class_name = det['class_name']
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f'{class_name} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return frame 