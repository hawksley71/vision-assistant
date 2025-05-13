"""
Configuration settings for the Vision-Aware Smart Assistant project.

This module centralizes all configuration settings including paths, model parameters,
camera settings, logging options, and more. It also handles automatic directory creation
for all required paths.

Usage:
    from config.settings import PATHS, MODEL_SETTINGS, CAMERA_SETTINGS
    model_path = PATHS['models']['yolov8']
    conf_threshold = MODEL_SETTINGS['yolov8']['confidence_threshold']
"""

import os
from pathlib import Path

# Project root directory - automatically determined based on this file's location
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Directory paths configuration
# All paths are relative to PROJECT_ROOT
PATHS = {
    'data': {
        'raw': os.path.join(PROJECT_ROOT, 'data', 'raw'),        # Raw detection logs
        'processed': os.path.join(PROJECT_ROOT, 'data', 'processed'),  # Processed data files
        'logs': os.path.join(PROJECT_ROOT, 'data', 'logs'),      # Application logs
    },
    'models': {
        'weights': os.path.join(PROJECT_ROOT, 'models', 'weights'),  # Model weights directory
        'yolov5': os.path.join(PROJECT_ROOT, 'models', 'weights', 'yolov5s.pt'),  # YOLOv5 model path
        'yolov8': os.path.join(PROJECT_ROOT, 'models', 'weights', 'yolov8n.pt'),  # YOLOv8 model path
    },
    'outputs': {
        'detections': os.path.join(PROJECT_ROOT, 'outputs', 'detections'),  # Detection results
        'audio': os.path.join(PROJECT_ROOT, 'outputs', 'audio'),    # Generated audio files
        'logs': os.path.join(PROJECT_ROOT, 'outputs', 'logs'),      # Output logs
    },
    'assets': {
        'audio': os.path.join(PROJECT_ROOT, 'assets', 'audio'),     # Audio assets
    }
}

# Model settings for object detection
MODEL_SETTINGS = {
    'yolov8': {
        'confidence_threshold': 0.25,  # Minimum confidence score for detections (0.0 to 1.0)
        'iou_threshold': 0.45,        # IoU threshold for non-maximum suppression (0.0 to 1.0)
        'max_detections': 20,         # Maximum number of detections per frame
        'device': 'cuda',             # Device to run inference on ('cuda' or 'cpu')
    },
    'yolov5': {
        'confidence_threshold': 0.25,  # Minimum confidence score for detections (0.0 to 1.0)
        'iou_threshold': 0.45,        # IoU threshold for non-maximum suppression (0.0 to 1.0)
        'max_detections': 1000,       # Maximum number of detections per frame
        'image_size': (640, 640),     # Input image size (width, height)
    }
}

# Camera settings for video capture
CAMERA_SETTINGS = {
    'width': 640,     # Camera capture width in pixels
    'height': 480,    # Camera capture height in pixels
    'fps': 10,        # Frames per second
}

# Logging settings for detection and application logs
LOGGING_SETTINGS = {
    'log_interval': 1.0,              # Interval between log entries in seconds
    'max_labels_per_log': 3,          # Maximum number of labels to log per entry
    'log_format': '%Y-%m-%d %H:%M:%S',  # Timestamp format for log files
}

# Audio settings for text-to-speech and playback
AUDIO_SETTINGS = {
    'language': 'en',                 # Language for text-to-speech
    'audio_player': 'mpv',           # Audio player command ('mpv', 'vlc', 'aplay', etc.)
    'intro_message': "Hello! I am your vision-aware assistant. I can see and detect objects in my view. Ask me what I see, or about past detections.",
}

# Home Assistant integration settings
HOME_ASSISTANT = {
    'url': 'http://localhost:8123',   # Home Assistant instance URL
    'tts_service': 'tts.google_translate_say',  # Text-to-speech service to use
    'media_player': 'media_player.den_speaker',  # Media player entity ID
}

def create_directories():
    """
    Create all required directories if they don't exist.
    This function is called automatically when the module is imported.
    It ensures all necessary directories for data, models, outputs, and assets exist.
    """
    for category in PATHS.values():
        for path in category.values():
            # Only create if path does not look like a file (no extension)
            if not os.path.splitext(path)[1]:
                os.makedirs(path, exist_ok=True)

# Initialize directories when module is imported
create_directories()

# Example usage:
"""
# Accessing paths
model_path = PATHS['models']['yolov8']
log_dir = PATHS['data']['raw']

# Accessing model settings
conf_threshold = MODEL_SETTINGS['yolov8']['confidence_threshold']
max_detections = MODEL_SETTINGS['yolov5']['max_detections']

# Accessing camera settings
camera_width = CAMERA_SETTINGS['width']
camera_fps = CAMERA_SETTINGS['fps']

# Accessing logging settings
log_interval = LOGGING_SETTINGS['log_interval']
timestamp_format = LOGGING_SETTINGS['log_format']

# Accessing audio settings
tts_language = AUDIO_SETTINGS['language']
audio_player = AUDIO_SETTINGS['audio_player']

# Accessing Home Assistant settings
ha_url = HOME_ASSISTANT['url']
tts_service = HOME_ASSISTANT['tts_service']
""" 