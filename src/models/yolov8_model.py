import cv2
import numpy as np
import torch
from ultralytics import YOLO
from .base_model import BaseModel

class YOLOv8Model(BaseModel):
    """YOLOv8 model implementation."""
    
    def __init__(self, model_path=None):
        """Initialize YOLOv8 model.
        
        Args:
            model_path (str, optional): Path to the model weights. If None, uses YOLOv8n.
        """
        if model_path is None:
            model_path = 'yolov8n.pt'  # Use nano model by default
        
        # Initialize model with GPU support
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        
        # Set model parameters for faster inference
        self.model.fuse()  # Fuse model layers
        if self.device == 'cuda':
            self.model.to(self.device)
            self.model.model.half()  # Use half precision
        
        # Set inference parameters
        self.conf = 0.25  # Confidence threshold
        self.iou = 0.45   # NMS IoU threshold
        self.max_det = 20  # Maximum detections per frame
        
    def detect(self, frame):
        """Detect objects in a frame using YOLOv8.
        
        Args:
            frame (numpy.ndarray): Input frame in BGR format
            
        Returns:
            list: List of detections, each containing:
                - bbox: [x1, y1, x2, y2] coordinates
                - confidence: float
                - class_id: int
                - class_name: str
        """
        # Do not normalize; pass frame as-is (uint8, 0-255)
        pass
        
        # print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
        
        frame = frame.astype(np.uint8)
        
        # Run inference with optimized parameters
        results = self.model(frame, 
                           verbose=False,
                           conf=self.conf,
                           iou=self.iou,
                           max_det=self.max_det,
                           device=self.device)[0]
        
        detections = []
        
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = r
            class_name = results.names[int(class_id)]
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf),
                'class_id': int(class_id),
                'class_name': class_name
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
            conf = det['confidence']
            class_name = det['class_name']
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f'{class_name} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return frame 