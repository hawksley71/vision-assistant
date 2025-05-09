import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Directory paths
PATHS = {
    'data': {
        'raw': os.path.join(PROJECT_ROOT, 'data', 'raw'),
        'processed': os.path.join(PROJECT_ROOT, 'data', 'processed'),
        'logs': os.path.join(PROJECT_ROOT, 'data', 'logs'),
    },
    'models': {
        'weights': os.path.join(PROJECT_ROOT, 'models', 'weights'),
        'yolov5': os.path.join(PROJECT_ROOT, 'models', 'weights', 'yolov5s.pt'),
        'yolov8': os.path.join(PROJECT_ROOT, 'models', 'weights', 'yolov8n.pt'),
    },
    'outputs': {
        'detections': os.path.join(PROJECT_ROOT, 'outputs', 'detections'),
        'audio': os.path.join(PROJECT_ROOT, 'outputs', 'audio'),
        'logs': os.path.join(PROJECT_ROOT, 'outputs', 'logs'),
    },
    'assets': {
        'audio': os.path.join(PROJECT_ROOT, 'assets', 'audio'),
    }
}

# Model settings
MODEL_SETTINGS = {
    'yolov8': {
        'confidence_threshold': 0.25,
        'iou_threshold': 0.45,
        'max_detections': 20,
        'device': 'cuda',  # or 'cpu'
    },
    'yolov5': {
        'confidence_threshold': 0.25,
        'iou_threshold': 0.45,
        'max_detections': 1000,
        'image_size': (640, 640),
    }
}

# Camera settings
CAMERA_SETTINGS = {
    'width': 640,
    'height': 480,
    'fps': 30,
}

# Logging settings
LOGGING_SETTINGS = {
    'log_interval': 1.0,  # seconds
    'max_labels_per_log': 3,
    'log_format': '%Y-%m-%d %H:%M:%S',
}

# Audio settings
AUDIO_SETTINGS = {
    'language': 'en',
    'audio_player': 'mpv',  # or 'vlc', 'aplay', etc.
    'intro_message': "Hello! I am your vision-aware assistant. I can see and detect objects in my view. Ask me what I see, or about past detections.",
}

# Home Assistant settings
HOME_ASSISTANT = {
    'url': 'http://localhost:8123',
    'tts_service': 'tts.google_translate_say',
    'media_player': 'media_player.den_speaker',
}

# Create directories if they don't exist
def create_directories():
    for category in PATHS.values():
        for path in category.values():
            os.makedirs(path, exist_ok=True)

# Initialize directories when module is imported
create_directories() 