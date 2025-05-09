from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """Base class for object detection models."""
    
    @abstractmethod
    def __init__(self, model_path=None):
        """Initialize the model.
        
        Args:
            model_path (str, optional): Path to the model weights. If None, uses default weights.
        """
        pass
    
    @abstractmethod
    def detect(self, frame):
        """Detect objects in a frame.
        
        Args:
            frame (numpy.ndarray): Input frame in BGR format
            
        Returns:
            list: List of detections, each containing:
                - bbox: [x1, y1, x2, y2] coordinates
                - confidence: float
                - class_id: int
                - class_name: str
        """
        pass
    
    @abstractmethod
    def draw_detections(self, frame, detections):
        """Draw detection boxes and labels on the frame.
        
        Args:
            frame (numpy.ndarray): Input frame in BGR format
            detections (list): List of detections from detect() method
            
        Returns:
            numpy.ndarray: Frame with detections drawn
        """
        pass 