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

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Paths
PATHS = {
    'data': {
        'raw': os.path.join(PROJECT_ROOT, 'data', 'raw'),
        'processed': os.path.join(PROJECT_ROOT, 'data', 'processed'),
        'combined_logs': os.path.join(PROJECT_ROOT, 'data', 'processed', 'combined_logs.csv'),
    },
    'models': {
        'yolov8': os.path.join(PROJECT_ROOT, 'models', 'yolov8', 'yolov8n.pt'),  # YOLOv8 model path
    },
    'logs': os.path.join(PROJECT_ROOT, 'logs'),
}

# Create directories if they don't exist
for path in PATHS['data'].values():
    if not os.path.splitext(path)[1]:  # Only create if no file extension
        os.makedirs(path, exist_ok=True)
os.makedirs(PATHS['logs'], exist_ok=True)

# Model settings
MODEL_SETTINGS = {
    'yolov8': {
        'conf_threshold': 0.25,  # Confidence threshold
        'iou_threshold': 0.45,   # IoU threshold for NMS
        'max_detections': 100,   # Maximum number of detections
        'device': 'cpu',         # Device to run inference on
    }
}

# Camera settings
CAMERA_SETTINGS = {
    'width': 640,
    'height': 480,
    'fps': 30,
    'device': 0,  # Default camera device
}

# Audio settings
AUDIO_SETTINGS = {
    'sample_rate': 16000,
    'chunk_size': 1024,
    'channels': 1,
    'format': 'wav',
}

# Home Assistant settings
HOME_ASSISTANT = {
    'url': os.getenv('HOME_ASSISTANT_URL', 'http://localhost:8123'),
    'token': os.getenv('HOME_ASSISTANT_TOKEN', ''),
    'tts_service': 'tts.google_translate_say',
    'tts_entity': 'media_player.living_room_speaker',
}

# OpenAI settings
OPENAI = {
    'api_key': os.getenv('OPENAI_TOKEN', ''),
    'model': 'gpt-4-turbo-preview',
    'temperature': 0.7,
    'max_tokens': 1000,
}

# Logging settings
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.path.join(PATHS['logs'], 'assistant.log'),
}

def create_directories():
    """
    Create all required directories if they don't exist.
    This function is called automatically when the module is imported.
    It ensures all necessary directories for data, models, and logs exist.
    """
    for category in PATHS.values():
        if isinstance(category, dict):
            for path in category.values():
                # Only create if path does not look like a file (no extension)
                if not os.path.splitext(path)[1]:
                    os.makedirs(path, exist_ok=True)
        elif isinstance(category, str):
            # Only create if path does not look like a file (no extension)
            if not os.path.splitext(category)[1]:
                os.makedirs(category, exist_ok=True)

# Initialize directories when module is imported
create_directories()

# Example usage:
"""
# Accessing paths
model_path = PATHS['models']['yolov8']
log_dir = PATHS['data']['raw']

# Accessing model settings
conf_threshold = MODEL_SETTINGS['yolov8']['confidence_threshold']
max_detections = MODEL_SETTINGS['yolov8']['max_detections']

# Accessing camera settings
camera_width = CAMERA_SETTINGS['width']
camera_fps = CAMERA_SETTINGS['fps']

# Accessing audio settings
tts_language = AUDIO_SETTINGS['language']
audio_player = AUDIO_SETTINGS['audio_player']

# Accessing Home Assistant settings
ha_url = HOME_ASSISTANT['url']
tts_service = HOME_ASSISTANT['tts_service']
""" 