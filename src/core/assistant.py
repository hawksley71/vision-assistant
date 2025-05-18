# Standard library imports
import os
import re
import csv
import json
import time
import random
import threading
import difflib
from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import ast
from pathlib import Path

# Third-party library imports
import cv2
import numpy as np
import pandas as pd
import requests
import speech_recognition as sr
from dotenv import load_dotenv, find_dotenv
from gtts import gTTS
from openai import OpenAI
from sklearn.cluster import KMeans
from spellchecker import SpellChecker

# Local application imports
from src.config.settings import PATHS, CAMERA_SETTINGS, LOGGING, AUDIO_SETTINGS, HOME_ASSISTANT, MODEL_SETTINGS
from src.models.yolov8_model import YOLOv8Model
from src.utils.audio import get_microphone
from src.core.openai_assistant import ask_openai, parse_query_with_openai
from src.core.tts import send_tts_to_ha, wait_for_tts_to_finish
from src.core.camera import Camera
from src.utils.pattern_analyzer import PatternAnalyzer, generate_pattern_summary

# Suppress ALSA and other audio library warnings
os.environ["PYTHONWARNINGS"] = "ignore"
try:
    import ctypes
    asound = ctypes.cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(None)
except Exception:
    pass

load_dotenv()
HOME_ASSISTANT_TOKEN = os.getenv("HOME_ASSISTANT_TOKEN")

HEADLESS = os.environ.get("VISION_ASSISTANT_HEADLESS", "0") == "1"

# Global lock to prevent multiple AI Report Assistants from being created simultaneously
_assistant_lock = threading.Lock()

# Global lock to prevent multiple combined_logs.csv uploads to OpenAI
_combined_logs_lock = threading.Lock()

class ResponseGenerator(ABC):
    """Abstract base class for response generators."""
    @abstractmethod
    def format_single_detection(self, obj: str, timestamp: str, confidence: Optional[float] = None) -> str:
        """Format a single detection response."""
        pass

    @abstractmethod
    def format_multiple_detections(self, obj: str, times: List[str], time_expr: Optional[str] = None) -> str:
        """Format multiple detections response."""
        pass

    @abstractmethod
    def format_confidence(self, label: str, confidence: float) -> str:
        """Format confidence level response."""
        pass

    def _add_article(self, obj: str) -> str:
        """Add the appropriate article ('a', 'an', or 'the') before the object name."""
        if obj.lower().endswith('s'):  # Check if the object is plural
            return f"the {obj}"
        if obj.lower().startswith(('a', 'e', 'i', 'o', 'u')):
            return f"an {obj}"
        return f"a {obj}"

class BasicResponseGenerator(ResponseGenerator):
    """Basic response generator with minimal formatting."""
    def format_single_detection(self, obj: str, timestamp: str, confidence: Optional[float] = None) -> str:
        return f"Yes, I last saw {self._add_article(obj)} at {timestamp}."

    def format_multiple_detections(self, obj: str, times: List[str], time_expr: Optional[str] = None) -> str:
        if len(times) == 1:
            return f"Yes, I saw {self._add_article(obj)} at {times[0]} {('during ' + time_expr) if time_expr else ''}."
        return f"Yes, I saw {self._add_article(obj)} {len(times)} times {('during ' + time_expr) if time_expr else ''}: {', '.join(times)}."

    def format_confidence(self, label: str, confidence: float) -> str:
        percent_conf = int(round(confidence * 100))
        return f"{self._add_article(label)}: {percent_conf}%"

class NaturalResponseGenerator(ResponseGenerator):
    """Natural language response generator with more conversational formatting."""
    def __init__(self):
        self.confidence_phrases = {
            'high': ["I'm quite certain", "I'm very confident", "I'm sure"],
            'medium': ["I think", "I believe", "It looks like"],
            'low': ["I might have seen", "I possibly saw", "I think I saw"]
        }
        self.transition_phrases = [
            "First,", "Then,", "After that,", "Later,", "Finally,"
        ]

    def _get_confidence_phrase(self, confidence: float) -> str:
        if confidence >= 0.8:
            return random.choice(self.confidence_phrases['high'])
        elif confidence >= 0.5:
            return random.choice(self.confidence_phrases['medium'])
        return random.choice(self.confidence_phrases['low'])

    def format_single_detection(self, obj: str, timestamp: str, confidence: Optional[float] = None) -> str:
        if confidence is not None:
            phrase = self._get_confidence_phrase(confidence)
            return f"{phrase} I saw {self._add_article(obj)} at {timestamp}."
        return f"I saw {self._add_article(obj)} at {timestamp}."

    def format_multiple_detections(self, obj: str, times: List[str], time_expr: Optional[str] = None) -> str:
        if len(times) == 1:
            return f"I saw {self._add_article(obj)} at {times[0]} {('during ' + time_expr) if time_expr else ''}."
        
        # Format multiple detections with transitions
        parts = []
        for i, time in enumerate(times):
            if i == 0:
                parts.append(f"I first saw {self._add_article(obj)} at {time}")
            elif i == len(times) - 1:
                parts.append(f"and finally at {time}")
            else:
                parts.append(f"then at {time}")
        
        time_context = f" {('during ' + time_expr) if time_expr else ''}"
        return f"I saw {self._add_article(obj)} several times{time_context}: {' '.join(parts)}."

    def format_confidence(self, label: str, confidence: float) -> str:
        phrase = self._get_confidence_phrase(confidence)
        percent_conf = int(round(confidence * 100))
        return f"{phrase} it's {self._add_article(label)} ({percent_conf}% confidence)"

class ContextualResponseGenerator(ResponseGenerator):
    """Response generator that includes more context about the detection."""
    def __init__(self):
        self.context_phrases = [
            "walking by",
            "in the frame",
            "in view",
            "passing through",
            "in the scene"
        ]

    def format_single_detection(self, obj: str, timestamp: str, confidence: Optional[float] = None) -> str:
        context = random.choice(self.context_phrases)
        return f"I saw {self._add_article(obj)} {context} at {timestamp}."

    def format_multiple_detections(self, obj: str, times: List[str], time_expr: Optional[str] = None) -> str:
        if len(times) == 1:
            context = random.choice(self.context_phrases)
            return f"I saw {self._add_article(obj)} {context} at {times[0]} {('during ' + time_expr) if time_expr else ''}."
        
        parts = []
        for i, time in enumerate(times):
            context = random.choice(self.context_phrases)
            if i == 0:
                parts.append(f"I first saw {self._add_article(obj)} {context} at {time}")
            elif i == len(times) - 1:
                parts.append(f"and finally {context} at {time}")
            else:
                parts.append(f"then {context} at {time}")
        
        time_context = f" {('during ' + time_expr) if time_expr else ''}"
        return f"I saw {self._add_article(obj)} several times{time_context}: {' '.join(parts)}."

    def format_confidence(self, label: str, confidence: float) -> str:
        percent_conf = int(round(confidence * 100))
        if confidence >= 0.8:
            return f"I'm very confident ({percent_conf}%) that I saw {self._add_article(label)}"
        elif confidence >= 0.5:
            return f"I'm reasonably sure ({percent_conf}%) that I saw {self._add_article(label)}"
        return f"I'm not very certain ({percent_conf}%), but I think I saw {self._add_article(label)}"

class DetectionAssistant:
    def __init__(self, mic, response_style: str = "natural", camera=None):
        with _assistant_lock:
            # Initialize response generator based on style
            self.response_generators = {
                "basic": BasicResponseGenerator(),
                "natural": NaturalResponseGenerator(),
                "contextual": ContextualResponseGenerator()
            }
            self.response_generator = self.response_generators.get(
                response_style, 
                self.response_generators["natural"]
            )
            
            # Initialize spell checker
            self.spell = SpellChecker()
            
            # Initialize pattern analyzer
            self.pattern_analyzer = PatternAnalyzer()
            self.pattern_summary = None
            self._load_pattern_summary()
            
            # Initialize other components
            self.mic = mic
            self.camera = camera
            self.buffer = deque(maxlen=100)
            self.last_reported_time = None
            self.last_reported_labels = set()
            self.pending_detections = []
            self.clarification_needed = False
            self.last_query = None
            self.last_object = None
            
        # Initialize camera
        self.camera = camera if camera is not None else Camera()
        self.cap = self.camera.cap

        # Initialize OpenAI client
        _ = load_dotenv(find_dotenv())
        api_key = os.getenv("OPENAI_TOKEN")
        if not api_key:
            raise RuntimeError("Error: OPENAI_TOKEN not found in environment variables")
        self.openai_client = OpenAI(api_key=api_key)
        print("[DEBUG] OpenAI client initialized successfully")

        # Initialize YOLOv8 model with explicit model path
        model_path = PATHS['models']['yolov8']
        if not os.path.exists(model_path):
            raise RuntimeError(f"Error: YOLOv8 model not found at {model_path}")
        self.model = YOLOv8Model()
        self.latest_detections = []
        self.fps = 0
        self.pending_detections = None  # Store pending detections for follow-up

        # Define patterns for handling responses
        self.pending_response_patterns = [
            r"yes|yeah|sure|okay|ok|all of them|all|everything|complete|full|entire",
            r"no|nope|nah|just three|three|first three|most recent|recent|latest"
        ]

        # For logging - use actual current date
        today = datetime.now()
        today_str = today.strftime('%Y_%m_%d')
        sanitized_date = self.sanitize_filename(today_str)
        self.log_path = os.path.join(PATHS['data']['raw'], f"detections_{sanitized_date}.csv")
        self.last_log_time = time.time()
        
        # Use deque with maxlen to keep last 30 detections (about 1 second at 30fps)
        self.detections_buffer = deque(maxlen=30)
        
        # Write header if file does not exist
        if not os.path.exists(self.log_path):
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "label_1", "count_1", "avg_conf_1",
                    "label_2", "count_2", "avg_conf_2",
                    "label_3", "count_3", "avg_conf_3"
                ])

        # For voice
        self.voice_thread = threading.Thread(target=self.voice_query_loop, daemon=True)
        self.voice_active = True
        self.r = sr.Recognizer()
        
        # Initialize microphone with preference for CMTECK
        if mic is None:
            print("[DEBUG] No microphone provided, searching for CMTECK...")
            mic = sr.Microphone()
            available_mics = sr.Microphone.list_microphone_names()
            print("[DEBUG] Available microphones:")
            for i, name in enumerate(available_mics):
                print(f"[DEBUG] Microphone {i}: {name}")
            
            # Try to find CMTECK microphone
            cmteck_index = None
            for i, name in enumerate(available_mics):
                if "CMTECK" in name:
                    cmteck_index = i
                    break
            
            if cmteck_index is not None:
                print(f"[DEBUG] Found CMTECK microphone at index {cmteck_index}")
                mic = sr.Microphone(device_index=cmteck_index)
            else:
                print("[DEBUG] CMTECK microphone not found, using default")
                mic = sr.Microphone()
        
        self.mic = mic

        self.last_reported_labels = []  # Track last reported labels for confidence queries
        self.last_reported_confidences = {}  # Track last reported confidences

        self.combined_logs_path = PATHS['data']['combined_logs']
        self.combined_df = None
        self.last_combined_log_date = None
        self.write_combined_logs_once_per_day(force=True)

        self.show_feed = False  # New flag to control when to show the camera feed

        # PATCH: Initialize last_detected_object for context/pronoun handling
        self.last_detected_object = None
        # PATCH: For testing, if data/raw/detections_test.csv exists, use it for log loading
        self._test_log_path = os.path.join(PATHS['data']['raw'], 'detections_test.csv') if os.path.exists(os.path.join(PATHS['data']['raw'], 'detections_test.csv')) else None

    def _load_pattern_summary(self):
        """Load the pattern summary if it exists."""
        pattern_summary_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'pattern_summary.json'
        if pattern_summary_path.exists():
            with open(pattern_summary_path, 'r') as f:
                self.pattern_summary = json.load(f)
                
    def _update_pattern_summary(self):
        """Update the pattern summary with latest data."""
        df = self.load_combined_logs()
        if not df.empty:
            generate_pattern_summary(df)
            self._load_pattern_summary()
            
    def sanitize_filename(self, name):
        return re.sub(r'[^A-Za-z0-9]+', '_', name)

    def log_top_labels(self):
        """Log the top 3 detected labels to a CSV file."""
        if not self.detections_buffer:
            return
            
        # Count label frequencies and collect confidences
        label_counts = Counter()
        label_confidences = defaultdict(list)
        
        for det in self.detections_buffer:
            label = det['class_name']
            conf = det['confidence']
            label_counts[label] += 1
            label_confidences[label].append(conf)
            
        # Get top 3 labels
        top_labels = label_counts.most_common(3)
        
        # Prepare row with current timestamp
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = [now]
        
        for label, count in top_labels:
            avg_conf = sum(label_confidences[label]) / len(label_confidences[label])
            row.extend([label, count, round(avg_conf, 3)])
            
        # Pad row if fewer than 3 labels
        while len(row) < 10:
            row.extend(["", "", ""])
            
        # Ensure log directory exists
        log_dir = os.path.dirname(self.log_path)
        os.makedirs(log_dir, exist_ok=True)
        
        # Write to CSV
        print(f"[DEBUG] Writing to log file: {self.log_path}")
        print(f"[DEBUG] Row data: {row}")
        
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        # Write combined logs if the day has changed
        self.write_combined_logs_once_per_day()

    def natural_list(self, items):
        """Format a list of items in a natural way (e.g., 'a, b, and c')."""
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    def summarize_buffer_labels(self):
        """Summarize the current buffer of detections."""
        print(f"[DEBUG] Current buffer size: {len(self.detections_buffer)}")
        if not self.detections_buffer:
            self.last_reported_labels = []
            self.last_reported_confidences = {}
            return "Warming up, please wait..."
            
        # Count label frequencies and collect confidences
        label_counts = Counter()
        label_confidences = defaultdict(list)
        
        for det in self.detections_buffer:
            label = det['class_name']
            conf = det['confidence']
            label_counts[label] += 1
            label_confidences[label].append(conf)
            
        # Compute average confidence for each label
        avg_confidences = {label: sum(confs)/len(confs) for label, confs in label_confidences.items()}
        
        # Sort labels by average confidence, descending
        sorted_labels = sorted(avg_confidences.items(), key=lambda x: x[1], reverse=True)
        
        # Only keep top 3
        sorted_labels = sorted_labels[:3]
        
        # Store for confidence queries
        self.last_reported_labels = [label for label, _ in sorted_labels]
        self.last_reported_confidences = {label: label_confidences[label] for label, _ in sorted_labels}
        
        if not sorted_labels:
            return "I'm not seeing anything right now."
            
        # Low confidence logic
        low_confidence_threshold = 0.4
        low_confidence_phrases = [
            "probably a",
            "might be a",
            "may have seen a",
            "possibly a"
        ]
        
        def label_with_conf(label, conf):
            percent = int(round(conf * 100))
            if conf < low_confidence_threshold:
                phrase = random.choice(low_confidence_phrases)
                return f"{phrase} {label}"
            else:
                return label
                
        # Build the list for natural response
        label_phrases = [label_with_conf(label, conf) for label, conf in sorted_labels]
        return "Right now, I am seeing: " + self.natural_list(label_phrases) + "."

    def summarize_buffer_confidence(self):
        # Report the average confidence for the last reported label(s)
        print(f"[DEBUG] last_reported_labels: {self.last_reported_labels}")
        print(f"[DEBUG] last_reported_confidences: {self.last_reported_confidences}")
        
        if not self.last_reported_labels or not self.last_reported_confidences:
            if self.detections_buffer:
                print("[DEBUG] No last reported labels/confidences, summarizing current buffer instead.")
                label_counts = Counter()
                label_confidences = defaultdict(list)
                for det in self.detections_buffer:
                    label = det['class_name']
                    conf = det['confidence']
                    label_counts[label] += 1
                    label_confidences[label].append(conf)
                avg_confidences = {label: sum(confs)/len(confs) for label, confs in label_confidences.items()}
                sorted_labels = sorted(avg_confidences.items(), key=lambda x: x[1], reverse=True)[:3]
                if not sorted_labels:
                    print("[DEBUG] No detections in buffer for confidence summary.")
                    return "I don't have confidence information for the current detection."
                
                parts = []
                for label, conf in sorted_labels:
                    parts.append(self.response_generator.format_confidence(label, conf))
                if len(parts) == 1:
                    return parts[0]
                return "My confidence levels are: " + ", ".join(parts) + "."
            
            print("[DEBUG] No last reported labels/confidences available and buffer is empty.")
            return "I don't have confidence information for the current detection."
        
        parts = []
        for label in self.last_reported_labels:
            confs = self.last_reported_confidences.get(label, [])
            print(f"[DEBUG] Label: {label}, Confidences: {confs}")
            if confs:
                avg_conf = sum(confs) / len(confs)
                parts.append(self.response_generator.format_confidence(label, avg_conf))
        
        if not parts:
            print("[DEBUG] No confidence information for last detection.")
            return "I don't have confidence information for the last detection."
        
        if len(parts) == 1:
            return parts[0]
        return "My confidence levels are: " + ", ".join(parts) + "."

    def parse_time_expression(self, time_expr):
        # Use actual current date
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today = pd.Timestamp(today)
        print(f"[DEBUG] Current date being used: {today}")
        
        if not time_expr or time_expr == "today":
            return today, today
        elif time_expr == "yesterday":
            return today - pd.Timedelta(days=1), today - pd.Timedelta(days=1)
        elif time_expr == "last week":
            # Get current week number
            current_year, current_week, _ = today.isocalendar()
            # Calculate last week's dates
            last_week_start = pd.Timestamp.fromisocalendar(current_year, current_week - 1, 1)  # Monday
            last_week_end = pd.Timestamp.fromisocalendar(current_year, current_week - 1, 7)   # Sunday
            print(f"[DEBUG] Last week range: {last_week_start} to {last_week_end}")
            return last_week_start, last_week_end
        elif time_expr == "this week":
            # Get current week number
            current_year, current_week, _ = today.isocalendar()
            # Calculate this week's dates
            this_week_start = pd.Timestamp.fromisocalendar(current_year, current_week, 1)  # Monday
            this_week_end = pd.Timestamp.fromisocalendar(current_year, current_week, 7)   # Sunday
            return this_week_start, this_week_end
        elif time_expr == "last month":
            first = today.replace(day=1) - pd.Timedelta(days=1)
            start = first.replace(day=1)
            end = first
            return start, end
        elif time_expr == "this month":
            start = today.replace(day=1)
            end = today
            return start, end
        elif time_expr and re.match(r"in [A-Za-z]+", time_expr):
            # e.g., "in May"
            month = time_expr.split()[1]
            year = today.year
            try:
                start = pd.Timestamp(f"{year}-{month}-01")
                end = (start + pd.offsets.MonthEnd(1)).normalize()
                return start, end
            except Exception:
                return None, None
        elif time_expr == "this weekend":
            # Find the most recent Saturday and Sunday (could be today if today is Sat/Sun)
            weekday = today.weekday()
            # Saturday is 5, Sunday is 6
            saturday = today - pd.Timedelta(days=(weekday - 5) % 7)
            sunday = saturday + pd.Timedelta(days=1)
            return saturday, sunday
        elif time_expr == "last weekend":
            # Find the previous week's Saturday and Sunday
            weekday = today.weekday()
            last_saturday = today - pd.Timedelta(days=weekday + 2)
            last_sunday = last_saturday + pd.Timedelta(days=1)
            return last_saturday, last_sunday
        # Handle "X weeks ago" pattern
        elif time_expr and re.match(r"(\d+)\s+weeks?\s+ago", time_expr, re.IGNORECASE):
            weeks_ago = int(re.match(r"(\d+)\s+weeks?\s+ago", time_expr, re.IGNORECASE).group(1))
            current_year, current_week, _ = today.isocalendar()
            target_week = current_week - weeks_ago
            # Handle year boundary
            if target_week <= 0:
                current_year -= 1
                target_week += 52  # Approximate weeks in a year
            week_start = pd.Timestamp.fromisocalendar(current_year, target_week, 1)  # Monday
            week_end = pd.Timestamp.fromisocalendar(current_year, target_week, 7)   # Sunday
            print(f"[DEBUG] {weeks_ago} weeks ago range: {week_start} to {week_end}")
            return week_start, week_end
        # Handle standalone days of the week
        elif time_expr.lower() in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
            # Map day names to weekday numbers (Monday=0, Sunday=6)
            day_map = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }
            target_weekday = day_map[time_expr.lower()]
            current_weekday = today.weekday()
            # Calculate days to subtract to get to the most recent occurrence of the target day
            days_to_subtract = (current_weekday - target_weekday) % 7
            # If today is the target day, use today
            if days_to_subtract == 0:
                return today, today
            # Otherwise, go back to the most recent occurrence
            target_date = today - pd.Timedelta(days=days_to_subtract)
            print(f"[DEBUG] Most recent {time_expr}: {target_date}")
            return target_date, target_date
        # Add more cases as needed
        return None, None

    def find_closest_label(self, partial_label, known_labels):
        # Use difflib to find the closest match
        matches = difflib.get_close_matches(partial_label, known_labels, n=1, cutoff=0.6)
        if matches:
            return matches[0]
        return partial_label

    def correct_spelling(self, word):
        """Correct the spelling of a word using pyspellchecker."""
        misspelled = self.spell.unknown([word])
        if misspelled:
            return self.spell.correction(word)
        return word

    def get_known_objects(self) -> set:
        """Extract a set of all known object names from the logs."""
        try:
            df = self.load_all_logs()
            if df.empty:
                return set()
            
            # Get unique objects from all label columns
            known_objects = set()
            for col in ['label_1', 'label_2', 'label_3']:
                known_objects.update(df[col].dropna().unique())
            
            # Also check current detections
            if self.detections_buffer:
                known_objects.update(d['class_name'] for d in self.detections_buffer)
            
            return known_objects
        except Exception as e:
            print(f"[ERROR] Failed to get known objects: {e}")
            return set()

    def normalize_object_label(self, label: str) -> str:
        """Normalize an object label to handle common variations."""
        if not label:
            return label
            
        # Convert to lowercase and strip whitespace
        label = label.lower().strip()
        
        # Handle common variations
        variations = {
            'buses': 'bus',
            'busses': 'bus',
            'trucks': 'truck',
            'cars': 'car',
            'dogs': 'dog',
            'cats': 'cat',
            'birds': 'bird',
            'people': 'person',
            'persons': 'person',
            'humans': 'person',
            'vehicles': 'vehicle',
            'animals': 'animal'
        }
        
        # Check for exact matches in variations
        if label in variations:
            return variations[label]
            
        # Check for known objects that might be similar
        known_objects = self.get_known_objects()
        if known_objects:
            # Try to find exact match first
            if label in known_objects:
                return label
                
            # Try to find similar objects
            similar = self.find_similar_objects(label)
            if similar:
                return similar[0]  # Return the most similar match
                
        return label

    def find_similar_objects(self, partial_label: str, max_suggestions: int = 3) -> List[str]:
        """Find similar objects from known objects list."""
        known_objects = self.get_known_objects()
        if not known_objects:
            return []
        
        # Convert to lowercase for comparison
        partial_label = partial_label.lower()
        known_objects_lower = {obj.lower(): obj for obj in known_objects}
        
        # Find matches
        matches = []
        
        # First try exact matches
        for obj_lower, obj in known_objects_lower.items():
            if partial_label in obj_lower or obj_lower in partial_label:
                matches.append(obj)
                if len(matches) >= max_suggestions:
                    return matches
        
        # Then try fuzzy matching
        if len(matches) < max_suggestions:
            fuzzy_matches = difflib.get_close_matches(
                partial_label,
                [obj.lower() for obj in known_objects if obj not in matches],
                n=max_suggestions - len(matches),
                cutoff=0.6
            )
            for match in fuzzy_matches:
                for obj in known_objects:
                    if obj.lower() == match and obj not in matches:
                        matches.append(obj)
                        break
        
        return matches

    def load_all_logs(self):
        """Load all detection logs as a DataFrame."""
        try:
            # Try to load the combined logs file
            df = pd.read_csv(PATHS['data']['combined_logs'], parse_dates=['timestamp'])
            print(f"[DEBUG] Loaded {len(df)} rows from combined logs")
            return df
        except Exception as e:
            print(f"[DEBUG] Error loading combined logs: {e}")
            # Return empty DataFrame if file not found or error
            return pd.DataFrame(columns=[
                "timestamp",
                "label_1", "count_1", "avg_conf_1",
                "label_2", "count_2", "avg_conf_2",
                "label_3", "count_3", "avg_conf_3"
            ])

    def write_combined_logs_for_debug(self, df):
        output_path = PATHS['data']['combined_logs']
        df.to_csv(output_path, index=False)
        print(f"Debug: Combined logs written to {output_path}")

    def _wait_for_file_processing(self, file_id, max_retries=10, delay=2):
        """Wait for a file to be processed and ready for use."""
        for i in range(max_retries):
            try:
                file = self.openai_client.files.retrieve(file_id)
                if file.status == 'processed':
                    print(f"[DEBUG] File {file_id} is ready for use")
                    return True
                elif file.status == 'error':
                    raise RuntimeError(f"File processing failed: {file.status_details}")
                print(f"[DEBUG] Waiting for file {file_id} to be processed... (attempt {i+1}/{max_retries})")
                time.sleep(delay)
            except Exception as e:
                print(f"[ERROR] Error checking file status: {e}")
                time.sleep(delay)
        raise RuntimeError(f"File {file_id} failed to process within {max_retries * delay} seconds")

    def write_combined_logs_once_per_day(self, force=False):
        with _combined_logs_lock:
            # Write combined logs only if the day has changed or force is True
            df = self.load_all_logs()
            if df.empty:
                return
            latest_date = df['timestamp'].max().date()
            
            # Check for existing files
            try:
                existing_files = self.openai_client.files.list()
                combined_logs_files = [f for f in existing_files.data if f.filename == 'combined_logs.csv']
                
                if combined_logs_files:
                    # Get the most recent file
                    latest_file = max(combined_logs_files, key=lambda x: x.created_at)
                    file_age = time.time() - latest_file.created_at
                    
                    # If file is less than 2 hours old and not forced, don't upload
                    if file_age < 7200 and not force:  # 7200 seconds = 2 hours
                        print(f"[DEBUG] Using existing file (ID: {latest_file.id}) that is {int(file_age/60)} minutes old")
                        # Verify the file is still processed and ready
                        self._wait_for_file_processing(latest_file.id)
                        return
                    
                    # Delete old files
                    for file in combined_logs_files:
                        try:
                            print(f"[DEBUG] Deleting old file (ID: {file.id})")
                            self.openai_client.files.delete(file.id)
                        except Exception as e:
                            print(f"[WARNING] Failed to delete old file {file.id}: {e}")
                            continue
            except Exception as e:
                print(f"[ERROR] Failed to check existing files: {e}")
                # If we can't check existing files, proceed with upload if forced or date changed
                if not force and self.last_combined_log_date == latest_date:
                    return
            
            # Only proceed with upload if forced or date changed
            if force or self.last_combined_log_date != latest_date:
                # Write to local file first
                try:
                    self.write_combined_logs_for_debug(df)
                    self.last_combined_log_date = latest_date
                except Exception as e:
                    print(f"[ERROR] Failed to write local combined logs: {e}")
                    return
                
                # Upload to OpenAI
                try:
                    print("[DEBUG] Uploading combined logs to OpenAI...")
                    with open(PATHS['data']['combined_logs'], "rb") as f:
                        file_obj = self.openai_client.files.create(file=f, purpose="assistants")
                        print(f"[DEBUG] File uploaded successfully with ID: {file_obj.id}")
                        # Wait for file to be processed
                        self._wait_for_file_processing(file_obj.id)
                except Exception as e:
                    print(f"[ERROR] Failed to upload file to OpenAI: {e}")
                    # Don't raise the exception, just log it and continue
                    # This prevents the program from crashing if the upload fails

        # After writing logs, update pattern summary
        self._update_pattern_summary()

    def format_timestamp(self, timestamp):
        """Format timestamp in a natural way, using weekday names for current week."""
        now = datetime.now()
        timestamp = pd.Timestamp(timestamp)
        
        # Check if timestamp is today
        if timestamp.date() == now.date():
            return timestamp.strftime("%I:%M %p today").lstrip("0")
        # Check if timestamp is yesterday
        elif timestamp.date() == (now.date() - timedelta(days=1)):
            return timestamp.strftime("%I:%M %p yesterday").lstrip("0")
        # If within current week, use weekday name
        elif (now - timestamp).days < 7:
            return timestamp.strftime("%I:%M %p on %A").lstrip("0")
        # If within current year, use month and day
        elif timestamp.year == now.year:
            return timestamp.strftime("%I:%M %p on %B %d").lstrip("0")
        # Otherwise use full date
        else:
            return timestamp.strftime("%I:%M %p on %B %d, %Y").lstrip("0")

    def get_system_message(self):
        """Get the system message with current date."""
        current_date = datetime.now().strftime('%Y-%m-%d')
        return f"""You are a detection log and date/time expert. Use Python and pandas to analyze the detection logs. Only answer using your knowledge of the date and time and the detection logs. Unless the user asks for step-by-step reasoning, always return only the final answer in one sentence.

Today's date is {current_date}. For any questions involving time, such as 'most recent', 'last', 'yesterday', 'last week', 'last month', 'this winter', or any other relative date or time phrase, use this date as the reference for 'today'.

You have access to the detection logs as CSV files and can use Python and pandas to analyze them. If a user asks about an object and you do not find an exact match for the object name in the logs, search for partial string matches (case-insensitive) in the object labels. If you find up to three close matches, suggest them to the user as possible intended objects.

When interpreting user responses about detection history:
- If the user indicates they want to see all detections (e.g., "yes", "all", "everything"), show all detections
- If the user indicates they want to see recent detections (e.g., "no", "just three", "recent"), show only the three most recent detections
- If the response is ambiguous, default to showing the three most recent detections"""

    def handle_pending_detections(self, user_input):
        """Handle user response to pending detections question."""
        if not self.pending_detections:
            return None
            
        user_input = user_input.strip().lower()
        
        # Define patterns for rule-based responses
        yes_patterns = ["yes", "yeah", "sure", "okay", "ok", "all of them", "all", "everything", "complete", "full", "entire"]
        no_patterns = ["no", "nope", "nah", "just three", "three", "first three", "most recent", "recent", "latest"]
        
        # Try rule-based logic first
        if any(p in user_input for p in yes_patterns):
            # Return all detections
            obj, matches = self.pending_detections
            times = [self.format_timestamp(row['timestamp']) for _, row in matches.iterrows()]
            response = self.response_generator.format_multiple_detections(obj, times)
            self.pending_detections = None
            return response
        elif any(p in user_input for p in no_patterns):
            # Return just the three most recent detections
            obj, matches = self.pending_detections
            matches = matches.sort_values("timestamp", ascending=False).head(3)
            times = [self.format_timestamp(row['timestamp']) for _, row in matches.iterrows()]
            response = self.response_generator.format_multiple_detections(obj, times)
            self.pending_detections = None
            return response
            
        # If no match found, fall back to OpenAI
        try:
            obj, matches = self.pending_detections
            # Create a prompt that includes context about the pending detections
            prompt = f"""I asked if you wanted to hear about all {len(matches)} detections of {obj} or just the three most recent ones.
Your response was: "{user_input}"
Please interpret this response and tell me if you want to hear about all detections or just the three most recent ones.
Respond with either 'all' or 'three'."""

            # Get OpenAI's interpretation with current date in system message
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.get_system_message()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10
            )
            
            interpretation = response.choices[0].message.content.strip().lower()
            
            # Process OpenAI's interpretation
            if 'all' in interpretation:
                times = [self.format_timestamp(row['timestamp']) for _, row in matches.iterrows()]
                response = self.response_generator.format_multiple_detections(obj, times)
            else:  # Default to three most recent
                matches = matches.sort_values("timestamp", ascending=False).head(3)
                times = [self.format_timestamp(row['timestamp']) for _, row in matches.iterrows()]
                response = self.response_generator.format_multiple_detections(obj, times)
                
            self.pending_detections = None
            return response
            
        except Exception as e:
            print(f"[DEBUG] Error in OpenAI fallback: {e}")
            # If OpenAI fails, default to showing three most recent
            obj, matches = self.pending_detections
            matches = matches.sort_values("timestamp", ascending=False).head(3)
            times = [self.format_timestamp(row['timestamp']) for _, row in matches.iterrows()]
            response = self.response_generator.format_multiple_detections(obj, times)
            self.pending_detections = None
            return response

    def answer_object_time_query(self, obj, query):
        """Answer a query about when an object was detected."""
        if not obj:
            return "I'm not sure what object you're asking about."
            
        # Load all logs
        all_logs = self.load_all_logs()
        
        # Normalize the object name
        obj = self.normalize_object_label(obj)
        
        # Find all detections of the object in any label column
        matches = all_logs[
            (all_logs['label_1'].fillna('').str.lower() == obj.lower()) |
            (all_logs['label_2'].fillna('').str.lower() == obj.lower()) |
            (all_logs['label_3'].fillna('').str.lower() == obj.lower())
        ]
        
        if len(matches) == 0:
            return f"I haven't seen any {obj} yet."
        
        # Check for pattern-related queries using natural language variations
        pattern_indicators = [
            "pattern", "regular", "usually", "typically", "normally", "often",
            "schedule", "routine", "when does", "what time does", "when do you see",
            "what time do you see", "when is", "what time is", "when are",
            "what time are", "when do", "what time do", "when does it come",
            "what time does it come", "when does it appear", "what time does it appear",
            "when do you detect", "what time do you detect", "when do you notice",
            "what time do you notice", "when do you find", "what time do you find"
        ]
        
        is_pattern_query = False
        if query:
            query_lower = query.lower()
            is_pattern_query = any(indicator in query_lower for indicator in pattern_indicators)
        
        # If we have enough observations, analyze patterns
        if len(matches) >= 6:
            # Analyze patterns directly from the matches DataFrame
            matches = matches.copy()
            matches.loc[:, 'hour'] = matches['timestamp'].dt.hour
            matches.loc[:, 'day'] = matches['timestamp'].dt.day_name()
            matches.loc[:, 'is_weekday'] = matches['timestamp'].dt.weekday < 5
            
            # Count occurrences by hour and day
            hour_counts = matches['hour'].value_counts()
            day_counts = matches['day'].value_counts()
            total = len(matches)
            
            # Determine strong patterns (occurring in >30% of detections)
            strong_hours = hour_counts[hour_counts > total * 0.3].index.tolist()
            strong_days = day_counts[day_counts > total * 0.3].index.tolist()
            
            # Build response
            response = f"I've seen {obj} {total} times"
            
            # Check if detections occur every day
            all_days = set(matches['timestamp'].dt.day_name())
            if len(all_days) >= 6:  # If detected on 6 or more different days
                response += " every day"
            elif strong_days:
                # Check if all detections are on weekdays
                if all(matches['is_weekday']):
                    response += " on weekdays"
                # Check if all detections are on a single day
                elif len(strong_days) == 1 and day_counts[strong_days[0]] >= total * 0.5:
                    response += f" on {strong_days[0]}s"
                else:
                    response += f" on {', '.join(strong_days)}"
            
            # Add time patterns
            if strong_hours:
                time_desc = []
                for hour in strong_hours:
                    if 5 <= hour < 12:
                        time_desc.append("in the morning")
                    elif 12 <= hour < 17:
                        time_desc.append("in the afternoon")
                    else:
                        time_desc.append("in the evening")
                if time_desc:
                    response += f" {', '.join(time_desc)}"
            
            # If no strong pattern is detected and this is a pattern query
            if not strong_hours and not strong_days and is_pattern_query:
                response += ", but it does not have a regular pattern"
            
            return response + "."
        
        # Handle responses based on number of detections
        if len(matches) == 1:
            time = matches.iloc[0]['timestamp']
            return f"I've seen {obj} once, at {time.strftime('%I:%M %p')} on {time.strftime('%A, %B %d')}."
        
        if len(matches) <= 3:
            response = f"I've seen {obj} {len(matches)} times:\n"
            for _, row in matches.iterrows():
                time = row['timestamp']
                response += f"- {time.strftime('%I:%M %p')} on {time.strftime('%A, %B %d')}\n"
            return response.strip()
        
        # For more than 3 detections, ask if user wants to see all or just recent ones
        self.pending_detections = (obj, matches)
        return f"I've seen {obj} {len(matches)} times. Would you like to hear about all the detections, or just the three most recent ones?"

    def analyze_object_pattern(self, object_name: str, df: pd.DataFrame) -> str:
        """Analyze patterns in object detections."""
        try:
            # Normalize object name
            object_name = object_name.lower().strip()
            
            # Load pattern summary
            summary = self.load_pattern_summary()
            if summary is None:
                return self._analyze_object_pattern_fallback(df, object_name)
            
            # Find object patterns
            object_patterns = summary[summary['object'].str.lower() == object_name]
            if len(object_patterns) == 0:
                return self._analyze_object_pattern_fallback(df, object_name)
            
            # Get the first matching pattern
            pattern = object_patterns.iloc[0]
            strong_patterns = pattern['strong_patterns']
            
            if not strong_patterns:
                return f"I've seen {object_name} {pattern['total_detections']} times, but I don't see a strong pattern in when it appears."
            
            # Build response based on patterns
            response = f"I've seen {object_name} {pattern['total_detections']} times"
            
            # Check if we have detections for this object
            matches = df[
                (df['label_1'].fillna('').str.lower() == object_name) |
                (df['label_2'].fillna('').str.lower() == object_name) |
                (df['label_3'].fillna('').str.lower() == object_name)
            ]
            
            if len(matches) == 0:
                return f"I haven't seen any {object_name} yet."
            
            # Add time and day descriptions
            time_desc = []
            day_desc = []
            
            for p in strong_patterns:
                if p['type'] == 'time':
                    if 'hour' in p:
                        hour = p['hour']
                        if 5 <= hour < 12:
                            time_desc.append("in the morning")
                        elif 12 <= hour < 17:
                            time_desc.append("in the afternoon")
                        else:
                            time_desc.append("in the evening")
                    elif 'description' in p:
                        time_desc.append(p['description'])
                elif p['type'] == 'day':
                    if 'day' in p:
                        day_desc.append(f"on {p['day'].lower()}s")
                    elif 'description' in p:
                        day_desc.append(p['description'])
            
            # Special handling for bus patterns
            if object_name == 'bus':
                # Check if detections occur every day
                all_days = set(matches['timestamp'].dt.day_name())
                if len(all_days) >= 6:  # If detected on 6 or more different days
                    day_desc.append("every day")
                # Check if detections occur during school hours
                school_hours = matches[matches['timestamp'].dt.hour.between(7, 15)]
                if len(school_hours) >= len(matches) * 0.3:  # If 30% or more detections are during school hours
                    time_desc.append("during school hours")
            
            # Combine descriptions
            if day_desc:
                response += " " + " and ".join(day_desc)
            if time_desc:
                response += " " + " and ".join(time_desc)
            
            return response
            
        except Exception as e:
            print(f"Error in analyze_object_pattern: {str(e)}")
            return self._analyze_object_pattern_fallback(df, object_name)

    def _analyze_object_pattern_fallback(self, df: pd.DataFrame, object_name: str) -> str:
        try:
            df = df.copy()
            df.loc[:, 'hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df.loc[:, 'day'] = pd.to_datetime(df['timestamp']).dt.day_name()
            hour_counts = df['hour'].value_counts()
            day_counts = df['day'].value_counts()
            total = len(df)
            strong_hours = hour_counts[hour_counts > total * 0.3].index.tolist()
            strong_days = day_counts[day_counts > total * 0.3].index.tolist()
            response = f"I've seen {object_name} {total} times"
            all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            if all(day in day_counts.index for day in all_days):
                response += " every day"
            elif strong_days:
                response += f" on {', '.join(strong_days)}"
            if strong_hours:
                response += f" around {', '.join(map(str, strong_hours))}:00"
            # If no strong pattern is detected, indicate that for pattern queries
            if not strong_hours and not strong_days:
                response += ", but it does not have a regular pattern"
            return response + "."
        except Exception as e:
            print(f"Error in fallback pattern analysis: {e}")
            return f"I've seen {object_name} {len(df)} times, but I couldn't analyze the pattern."

    def handle_clarification_response(self, user_input):
        """Handle user's response to a clarification question about object label."""
        if not hasattr(self, 'pending_clarification') or self.pending_clarification is None:
            return "I'm sorry, there is no pending clarification request. Please ask your question again."
        original_label, options = self.pending_clarification
        user_input = user_input.strip().lower()
        if user_input in ['yes', 'yeah', 'sure', 'okay', 'ok', 'correct', 'right']:
            # Use the first option
            self.pending_clarification = None
            result = self.analyze_object_pattern(self.load_all_logs(), options[0])
            return result if result is not None else f"I couldn't find a pattern for '{options[0]}'."
        elif user_input in ['no', 'nope', 'nah', 'incorrect', 'wrong']:
            # User insists on the original label
            self.pending_clarification = None
            return f"I'll use '{original_label}' as requested."
        else:
            # Check if the user's response matches any of the options
            for option in options:
                if user_input in option.lower():
                    self.pending_clarification = None
                    result = self.analyze_object_pattern(self.load_all_logs(), option)
                    return result if result is not None else f"I couldn't find a pattern for '{option}'."
            # Ambiguous response, default to the original label
            self.pending_clarification = None
            return f"I'll proceed with '{original_label}'."

    def load_pattern_summary(self):
        """Load the pattern summary from the processed data directory."""
        try:
            pattern_file = os.path.join(PATHS['data']['processed'], 'pattern_summary.csv')
            if not os.path.exists(pattern_file):
                print("[DEBUG] Pattern summary file not found")
                return None
            
            df = pd.read_csv(pattern_file)
            if df.empty:
                print("[DEBUG] Pattern summary file is empty")
                return None
            
            # Convert string representation of patterns back to list of dicts
            df['strong_patterns'] = df['strong_patterns'].apply(eval)
            return df
        except Exception as e:
            print(f"[DEBUG] Error loading pattern summary: {e}")
            return None

    def get_time_based_patterns(self, object_name: str) -> List[str]:
        """Get time-based patterns for an object."""
        pattern_summary = self.load_pattern_summary()
        if pattern_summary is None or pattern_summary.empty:
            return []
        
        obj_patterns = pattern_summary[pattern_summary['object'].str.lower() == object_name.lower()]
        if obj_patterns.empty:
            return []
        
        patterns = obj_patterns.iloc[0]['strong_patterns']
        if not patterns:
            return []
        
        pattern_desc = []
        for pattern in patterns:
            if pattern['type'] == 'time':
                pattern_desc.append(f"I usually see it {pattern['description']}")
            elif pattern['type'] == 'day':
                pattern_desc.append(f"I often see it {pattern['description']}")
        
        return pattern_desc

    def get_detection_stats(self, object_name: str, matches: pd.DataFrame) -> str:
        """Get detection statistics for an object."""
        if matches.empty:
            return f"I haven't detected any {object_name} yet."
        
        # Sort matches by timestamp
        matches = matches.sort_values('timestamp', ascending=False)
        
        # Calculate time since last detection
        last_detection = matches.iloc[0]
        time_ago = pd.Timestamp.now() - pd.to_datetime(last_detection['timestamp'])
        hours_ago = time_ago.total_seconds() / 3600
        
        # Format time ago string
        if hours_ago < 24:
            time_ago_str = f"{int(hours_ago)} hours ago"
        else:
            days_ago = hours_ago / 24
            time_ago_str = f"{int(days_ago)} days ago"
        
        # Get total detections
        total_detections = len(matches)
        
        # Get detection frequency
        if total_detections > 1:
            first_detection = matches.iloc[-1]
            total_time = pd.to_datetime(last_detection['timestamp']) - pd.to_datetime(first_detection['timestamp'])
            total_hours = total_time.total_seconds() / 3600
            if total_hours > 0:
                frequency = total_detections / (total_hours / 24)  # detections per day
                if frequency > 0:
                    if frequency >= 1:
                        freq_str = f"about {int(frequency)} times per day"
                    else:
                        freq_str = f"about once every {int(1/frequency)} days"
                else:
                    freq_str = "occasionally"
            else:
                freq_str = "frequently"
        else:
            freq_str = "once"
        
        return f"I've detected {object_name} {total_detections} times ({freq_str}). The most recent detection was {time_ago_str}."

    def update_pattern_summary_if_needed(self):
        """Update pattern summary if it's a new day."""
        summary_path = os.path.join(PATHS['data']['raw'], 'pattern_summary.csv')
        
        # Check if summary exists and is from today
        if os.path.exists(summary_path):
            try:
                # Get file modification time
                mod_time = os.path.getmtime(summary_path)
                mod_date = datetime.fromtimestamp(mod_time).date()
                today = datetime.now().date()
                
                # If summary is from today, no need to update
                if mod_date == today:
                    return
            except Exception as e:
                print(f"[DEBUG] Error checking summary date: {e}")
        
        # Generate new summary
        print("[DEBUG] Generating new pattern summary...")
        self.generate_daily_pattern_summary()

    def process_historical_query(self, query: str) -> str:
        """Process a historical query about object detections."""
        # Extract object name from query
        obj_name = None
        
        # Check for "check the record for" pattern first
        check_record_match = re.search(r'check the record for (?:a |an |the )?([\w\s]+?)(?:\?|$)', query.lower())
        if check_record_match:
            obj_name = check_record_match.group(1).strip()
            
        # If no match found, try other patterns
        if not obj_name:
            patterns = [
                r'last time.*?(?:saw|detected|spotted) (?:a |an |the )?([\w\s]+?)(?:\?|$)',
                r'usually.*?(?:see|detect|spot) (?:a |an |the )?([\w\s]+?)(?:\?|$)',
                r'tell me about (?:a |an |the )?([\w\s]+?)(?:\?|$)',
                r'pattern.*?(?:for|with) (?:a |an |the )?([\w\s]+?)(?:\?|$)',
                r'when.*?(?:see|detect|spot) (?:a |an |the )?([\w\s]+?)(?:\?|$)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, query.lower())
                if match:
                    obj_name = match.group(1).strip()
                    break
                    
        if not obj_name:
            return "I'm not sure which object you're asking about. Could you please specify?"
            
        # Normalize object name
        obj_name = self.normalize_object_label(obj_name)
        
        # Check if we have pattern data for this object
        if self.pattern_summary:
            for obj_data in self.pattern_summary:
                if obj_data['object'].lower() == obj_name.lower():
                    # Format the response
                    response = f"I've detected {obj_data['total_detections']} instances of {obj_name}.\n"
                    
                    if obj_data['strong_patterns']:
                        response += "I've noticed the following patterns:\n"
                        for pattern in obj_data['strong_patterns']:
                            response += f"- {pattern['description']}\n"
                            
                    # Add first/last seen info
                    first_seen = datetime.fromisoformat(obj_data['first_seen'])
                    last_seen = datetime.fromisoformat(obj_data['last_seen'])
                    response += f"\nFirst seen: {first_seen.strftime('%Y-%m-%d %H:%M')}\n"
                    response += f"Last seen: {last_seen.strftime('%Y-%m-%d %H:%M')}"
                    
                    return response
                    
        # If no pattern data, fall back to basic analysis
        df = self.load_combined_logs()
        if df.empty:
            return f"I don't have any historical data about {obj_name} yet."
            
        # Find all detections of this object
        matches = df[
            (df['label_1'].fillna('').str.lower() == obj_name.lower()) |
            (df['label_2'].fillna('').str.lower() == obj_name.lower()) |
            (df['label_3'].fillna('').str.lower() == obj_name.lower())
        ]
        
        if len(matches) == 0:
            return f"I haven't detected any {obj_name} in my records."
            
        # Get basic stats
        total_detections = len(matches)
        first_seen = matches['timestamp'].min()
        last_seen = matches['timestamp'].max()
        
        # Generate response
        response = f"I've detected {total_detections} instances of {obj_name}.\n"
        response += f"First seen: {first_seen}\n"
        response += f"Last seen: {last_seen}"
        
        return response
        
    def write_combined_logs_once_per_day(self):
        """Write combined logs once per day and update pattern summary."""
        with _combined_logs_lock:
            # Get the path for combined logs
            combined_logs_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'combined_logs.csv'
            
            # Check if file exists and is less than 2 hours old
            if combined_logs_path.exists():
                file_age = time.time() - combined_logs_path.stat().st_mtime
                if file_age < 7200:  # 2 hours in seconds
                    return
                    
            # Load and combine all logs
            df = self.load_combined_logs()
            if not df.empty:
                # Write combined logs
                df.to_csv(combined_logs_path, index=False)
                
                # Update pattern summary
                self._update_pattern_summary()

    def answer_live_query(self, query: str) -> str:
        """Answer a query about current detections."""
        if not self.detections_buffer:
            return "I'm not seeing anything right now."
            
        # Count label frequencies and collect confidences
        label_counts = Counter()
        label_confidences = defaultdict(list)
        
        for det in self.detections_buffer:
            label = det['class_name']
            conf = det['confidence']
            label_counts[label] += 1
            label_confidences[label].append(conf)
            
        # Get top 3 labels by frequency
        top_labels = label_counts.most_common(3)
        
        if not top_labels:
            return "I'm not seeing anything right now."
            
        # Build response
        response = "Right now, I am seeing: "
        label_phrases = []
        
        for label, count in top_labels:
            avg_conf = sum(label_confidences[label]) / len(label_confidences[label])
            if avg_conf < 0.4:
                label_phrases.append(f"possibly a {label}")
            else:
                label_phrases.append(label)
                
        response += self.natural_list(label_phrases) + "."
        return response

    def voice_query_loop(self):
        print("Voice assistant is ready. Ask: 'What are you seeing right now?' or 'Did you see a [thing]?' or 'When did you last see [thing]?' or 'What was the first/last thing you saw?'. Say 'check the record for [thing]' to look up historical data. Say 'exit' to quit voice mode.")
        
        # Define regex patterns for each query type
        live_patterns = [
            r"(what|tell me|show me).*(see|detect|seeing|detecting|there|in front)",
            r"what else",
            r"what do you see",
            r"what are you seeing",
            r"what can you see",
            r"what's there",
            r"what is there",
            r"what do you detect",
            r"what are you detecting",
            r"what do you see now",
            r"what are you seeing now",
            r"what can you see now",
            r"what's there now",
            r"what is there now"
        ]
        
        # Add historical data keywords that force log lookup
        historical_keywords = [
            r"check the record",  # New high-priority pattern
            r"log", r"logs",
            r"record", r"records",
            r"report", r"reports",
            r"observation", r"observations",
            r"row", r"rows",
            r"item", r"items",
            r"detection", r"detections",
            r"historical",
            r"history",
            r"past",
            r"previous",
            r"earlier"
        ]
        
        # Combine into a single pattern
        historical_pattern = "|".join(historical_keywords)
        
        did_you_see_patterns = [
            r"did you see (?:a |an |any )?([\w\s]+?)(?:\s+(?:on|in|during|at|this|last|yesterday|today|tomorrow|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december))?(?:\?|$)",
            r"have you seen (?:a |an |any )?([\w\s]+?)(?:\s+(?:on|in|during|at|this|last|yesterday|today|tomorrow|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december))?(?:\?|$)",
            r"have you ever seen (?:a |an |any )?([\w\s]+?)(?:\s+(?:on|in|during|at|this|last|yesterday|today|tomorrow|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december))?(?:\?|$)",
            r"did you ever see (?:a |an |any )?([\w\s]+?)(?:\s+(?:on|in|during|at|this|last|yesterday|today|tomorrow|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december))?(?:\?|$)",
            r"have you detected (?:a |an |any )?([\w\s]+?)(?:\s+(?:on|in|during|at|this|last|yesterday|today|tomorrow|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december))?(?:\?|$)",
            r"did you detect (?:a |an |any )?([\w\s]+?)(?:\s+(?:on|in|during|at|this|last|yesterday|today|tomorrow|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december))?(?:\?|$)"
        ]
        
        last_seen_patterns = [
            r"when did you (?:last )?see (?:a |an )?([\w\s]+)\??",
            r"when was the last time you saw (?:a |an )?([\w\s]+?)\??"
        ]
        
        confidence_patterns = [
            r"confident",
            r"confidence",
            r"sure",
            r"how sure",
            r"how confident",
            r"how certain",
            r"how accurate",
            r"how reliable",
            r"how sure are you",
            r"how confident are you",
            r"how certain are you",
            r"how accurate are you",
            r"how reliable are you",
            r"are you sure",
            r"are you confident",
            r"are you certain"
        ]
        
        last_debug_print = time.time()
        
        while self.voice_active:
            try:
                with self.mic as source:
                    self.r.adjust_for_ambient_noise(source)
                    print("Listening...")
                    audio = self.r.listen(source, timeout=5)
                query = self.r.recognize_google(audio).lower()
                print(f"[DEBUG] Recognized query: {query}")
                
                # First check if we have a pending clarification
                if hasattr(self, 'pending_clarification') and self.pending_clarification is not None:
                    message = self.handle_clarification_response(query)
                    print(f"[DEBUG] TTS will be called with: {message}")
                    send_tts_to_ha(message)
                    continue
                
                intent_info = parse_query_with_openai(query, self.openai_client)
                intent = intent_info.get("intent")
                obj = intent_info.get("object")
                print(f"[DEBUG] OpenAI intent: {intent}, object: {obj}")

                message = "Sorry, I didn't understand that."
                
                # Check for historical keywords first
                if re.search(historical_pattern, query, re.IGNORECASE):
                    print("[DEBUG] Historical keyword detected, checking logs...")
                    # Special handling for "check the record" pattern
                    if "check the record" in query.lower():
                        print("[DEBUG] 'Check the record' pattern detected, explicitly routing to historical data")
                        # Extract object name after "check the record for"
                        match = re.search(r'check the record for (?:a |an |any |the )?([\w\s]+?)(?:\?|$)', query, re.IGNORECASE)
                        if match:
                            obj = match.group(1).strip()
                            obj = self.normalize_object_label(obj)
                            print(f"[DEBUG] Extracted object from 'check the record' pattern: {obj}")
                            df_all = self.load_all_logs()
                            message = self.process_historical_query(query)
                        else:
                            message = "What would you like me to check in the records?"
                        print(f"[DEBUG] TTS will be called with: {message}")
                        send_tts_to_ha(message)
                        continue
                    
                    # Try to extract object from query if not provided by OpenAI intent
                    if not obj:
                        # Try to extract after 'about', 'for', or 'of'
                        match = re.search(r'(?:about|for|of) ([\w\s]+)', query, re.IGNORECASE)
                        if match:
                            obj = match.group(1).strip()
                        else:
                            # Try to extract last word (if user says 'records for cat')
                            words = query.split()
                            if len(words) > 2:
                                obj = words[-1]
                    if obj:
                        obj = self.normalize_object_label(obj)
                        print(f"[DEBUG] Extracted object for historical query: {obj}")
                        df_all = self.load_all_logs()
                        message = self.process_historical_query(query)
                    else:
                        message = "What object would you like me to check in the logs?"
                    print(f"[DEBUG] TTS will be called with: {message}")
                    send_tts_to_ha(message)
                    continue
                
                # Check for "did you see" or "have you seen" patterns
                for pattern in did_you_see_patterns:
                    match = re.search(pattern, query, re.IGNORECASE)
                    if match:
                        extracted_obj = match.group(1).strip() if match.group(1) else None
                        if extracted_obj:
                            obj = extracted_obj
                            print(f"[DEBUG] Extracted object from 'did you see' pattern: {obj}")
                            obj = self.normalize_object_label(obj)
                            df_all = self.load_all_logs()
                            message = self.process_historical_query(query)
                            print(f"[DEBUG] TTS will be called with: {message}")
                            send_tts_to_ha(message)
                            continue
                
                # Handle based on OpenAI intent first
                if intent == "live_view":
                    message = self.summarize_buffer_labels()
                elif intent == "confidence":
                    print("[DEBUG] Confidence query detected. Calling summarize_buffer_confidence().")
                    message = self.summarize_buffer_confidence()
                elif intent == "detection_history":
                    if obj:
                        obj = self.normalize_object_label(obj)
                        df_all = self.load_all_logs()
                        message = self.process_historical_query(query)
                    else:
                        message = "What object are you asking about?"
                
                print(f"[USER QUERY] {query}")
                print(f"[ASSISTANT RESPONSE] {message}")
                print(f"[DEBUG] TTS will be called with: {message}")
                send_tts_to_ha(message)
                
            except sr.WaitTimeoutError:
                print("No speech detected within timeout period.")
            except sr.UnknownValueError:
                print("Sorry, could not understand. Please try again.")
            except sr.RequestError as e:
                print(f"Could not request results from speech recognition service; {e}")
            except Exception as e:
                print(f"Error in voice query loop: {e}")
                import traceback
                traceback.print_exc()

    def wait_for_intro_to_finish(self):
        """Wait for the introduction message to finish playing."""
        url = f"{HOME_ASSISTANT['url']}/api/states/{HOME_ASSISTANT['tts_entity']}"
        headers = {"Authorization": f"Bearer {HOME_ASSISTANT_TOKEN}"}
        intro_text = "Hello! I am your vision-aware assistant. I can see and detect objects in my view. Ask me what I see, or about past detections."
        last_word = "detections"
        
        print("[DEBUG] Waiting for intro message to finish...")
        for _ in range(20):  # Try for 10 seconds (20 * 0.5)
            try:
                resp = requests.get(url, headers=headers, timeout=1)
                state = resp.json()
                if state.get("state") == "idle":
                    # Check if the last word was played
                    media_content_id = state.get("attributes", {}).get("media_content_id", "")
                    if last_word in media_content_id:
                        print("[DEBUG] Intro message finished playing")
                        return True
            except Exception as e:
                print(f"[DEBUG] Error polling media player state: {e}")
            time.sleep(0.5)
        print("[DEBUG] Timeout waiting for intro TTS to finish.")
        return False

    def start_intro_and_voice(self):
        """Start the introduction message and voice assistant."""
        print("Warming up, please wait...")
        time.sleep(0.5)  # Short warmup delay
        
        # Generate and play intro via Home Assistant
        intro_text = "Hello! I am your vision-aware assistant. I can see and detect objects in my view. Ask me what I see, or about past detections."
        
        # Send intro message and wait for it to finish
        send_tts_to_ha(intro_text)
        
        # Wait for intro to finish playing
        if self.wait_for_intro_to_finish():
            # Now allow the camera feed to be shown
            self.show_feed = True
            # Now start the voice thread
            self.voice_thread.start()
        else:
            print("[DEBUG] Starting voice thread despite intro timeout")
            self.show_feed = True
            self.voice_thread.start()

    def run(self):
        frame_count = 0
        start_time = time.time()
        last_debug_print = time.time()
        print("Press 'q' to quit the live feed window.")
        
        # Start intro and voice assistant in a background thread
        intro_thread = threading.Thread(target=self.start_intro_and_voice, daemon=True)
        intro_thread.start()
        
        print("[DEBUG] Starting main detection loop...")
        while True:
            t_loop_start = time.time()
            
            # Skip detection and display if intro hasn't finished
            if not self.show_feed:
                time.sleep(0.1)  # Short sleep to prevent CPU spinning
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
                
            # Run detection
            try:
                detections = self.model.get_detections(frame)
                # Only print detections once per second
                current_time = time.time()
                if current_time - last_debug_print >= 1.0:
                    if detections:
                        print(f"[DEBUG] Detected objects: {[d['class_name'] for d in detections]}")
                    last_debug_print = current_time
                self.latest_detections = detections
                self.detections_buffer.extend(detections)
            except Exception as e:
                print(f"[DEBUG] Error in detection: {e}")
                continue
                
            # Draw detections
            try:
                frame = self.model.draw_detections(frame, detections)
            except Exception as e:
                print(f"[DEBUG] Error drawing detections: {e}")
                
            # Calculate and display FPS
            frame_count += 1
            if frame_count >= 30:
                end_time = time.time()
                self.fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()
                
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
            # Log top 3 labels every second
            now = time.time()
            if now - self.last_log_time >= 1.0:
                self.log_top_labels()
                self.last_log_time = now
                
            # Show frame
            cv2.imshow('Live Detection Assistant', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.voice_active = False
        self.voice_thread.join()
        self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def listen_for_query(self):
        """Listen for a voice query using Whisper."""
        try:
            print("Listening...")
            with self.mic as source:
                # Adjust for ambient noise
                self.r.adjust_for_ambient_noise(source)
                # Increase timeout and phrase time limit
                audio = self.r.listen(source, timeout=10, phrase_time_limit=15)
                print("Processing speech...")
                # Use Whisper for transcription
                result = self.r.recognize_whisper(audio)
                query = result["text"].strip()
                if query:
                    print(f"[DEBUG] Recognized query: {query}")
                    return query
                return None
        except sr.WaitTimeoutError:
            print("No speech detected within timeout period")
            return None
        except Exception as e:
            print(f"Error in speech recognition: {str(e)}")
            return None

    def process_user_query(self, user_input):
        """Main entry point for processing user queries, including handling pending detections."""
        # First, check if there is a pending detections context
        pending_response = self.handle_pending_detections(user_input)
        if pending_response:
            return pending_response

        # Process the query with OpenAI, including current date in system message
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.get_system_message()},
                    {"role": "user", "content": user_input}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[DEBUG] Error processing query with OpenAI: {e}")
            return "I'm sorry, I encountered an error processing your query."

if __name__ == "__main__":
    import speech_recognition as sr
    mic = sr.Microphone()
    assistant = DetectionAssistant(mic)
    assistant.run() 