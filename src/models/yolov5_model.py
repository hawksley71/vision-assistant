import cv2
import numpy as np
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device
from .base_model import BaseModel
from ..config.settings import PATHS, MODEL_SETTINGS

class YOLOv5Model(BaseModel):
    """YOLOv5 model implementation."""
    
    def __init__(self):
        """Initialize YOLOv5 model."""
        # Load YOLOv5 model
        model_path = PATHS['models']['yolov5']
        settings = MODEL_SETTINGS['yolov5']
        
        self.device = select_device('')
        self.model = DetectMultiBackend(model_path, device=self.device)
        self.stride = self.model.stride
        self.names = self.model.names
        self.imgsz = check_img_size(settings['image_size'], s=self.stride)
        
        # Store settings for inference
        self.conf = settings['confidence_threshold']
        self.iou = settings['iou_threshold']
        self.max_det = settings['max_detections']
        
    def detect(self, frame):
        """Detect objects in a frame using YOLOv5.
        
        Args:
            frame (numpy.ndarray): Input frame in BGR format
            
        Returns:
            list: List of detections, each containing:
                - bbox: [x1, y1, x2, y2] coordinates
                - confidence: float
                - class_id: int
                - class_name: str
        """
        # Prepare image
        img = cv2.resize(frame, (self.imgsz, self.imgsz))
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if len(img.shape) == 3:
            img = img[None]
            
        # Inference
        pred = self.model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf, self.iou, None, False, max_det=self.max_det)
        
        detections = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    class_id = int(cls)
                    class_name = self.names[class_id]
                    detections.append({
                        'bbox': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                        'confidence': float(conf),
                        'class_id': class_id,
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