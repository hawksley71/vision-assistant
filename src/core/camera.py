import cv2
from src.config.settings import CAMERA_SETTINGS

class Camera:
    def __init__(self, device_id=0):
        self.cap = cv2.VideoCapture(device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SETTINGS['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SETTINGS['height'])
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_SETTINGS['fps'])
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open camera")

    def read(self):
        return self.cap.read()

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.cap.release() 