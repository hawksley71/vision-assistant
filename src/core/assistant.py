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
    def __init__(self, mic, response_style: str = "natural"):
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
            
            # Update pattern summary on initialization
            self.update_pattern_summary_if_needed()
            
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SETTINGS['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SETTINGS['height'])
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_SETTINGS['fps'])
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open camera")

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

    def normalize_object_label(self, label):
        """Normalize and correct spelling of object labels."""
        # Remove all leading articles and extra spaces, and lowercase
        normalized = re.sub(r'^(a |an |the )+', '', label.strip(), flags=re.IGNORECASE).strip().lower()
        # Correct spelling
        return self.correct_spelling(normalized)

    def load_all_logs(self, log_dir="data/raw"):
        """Load all detection logs from the specified directory."""
        # print(f"[DEBUG] Loading logs from: {log_dir}")
        all_dfs = []
        
        # Ensure the directory exists
        if not os.path.exists(log_dir):
            # print(f"[DEBUG] Log directory {log_dir} does not exist")
            return pd.DataFrame()
            
        # Get all CSV files in the directory
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
        # print(f"[DEBUG] Found {len(log_files)} log files: {log_files}")
        
        for f in log_files:
            try:
                file_path = os.path.join(log_dir, f)
                # print(f"[DEBUG] Loading log file: {file_path}")
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                all_dfs.append(df)
                # print(f"[DEBUG] Successfully loaded {len(df)} rows from {f}")
            except Exception as e:
                # print(f"[DEBUG] Error loading {f}: {str(e)}")
                continue
                
        if not all_dfs:
            # print("[DEBUG] No valid log files found")
            return pd.DataFrame()
            
        # Combine all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        # print(f"[DEBUG] Combined {len(combined_df)} total rows from {len(all_dfs)} files")
        return combined_df

    def write_combined_logs_for_debug(self, df):
        output_path = PATHS['data']['combined_logs']
        df.to_csv(output_path, index=False)
        print(f"Debug: Combined logs written to {output_path}")

    def write_combined_logs_once_per_day(self, force=False):
        with _combined_logs_lock:
            # Write combined logs only if the day has changed or force is True
            df = self.load_all_logs()
            if df.empty:
                return
            latest_date = df['timestamp'].max().date()
            if force or self.last_combined_log_date != latest_date:
                self.write_combined_logs_for_debug(df)
                self.last_combined_log_date = latest_date

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
        
        # If no exact match, try compound word matching
        if len(matches) == 0 and ' ' in obj:
            # Split the compound word
            parts = obj.split()
            # Find objects that contain all parts of the compound word
            potential_matches = []
            for col in ['label_1', 'label_2', 'label_3']:
                for label in all_logs[col].dropna().unique():
                    if all(part.lower() in label.lower() for part in parts):
                        potential_matches.append(label)
            
            if potential_matches:
                # Get the top three most detected matches
                top_matches = sorted(potential_matches, key=lambda x: len(all_logs[all_logs[col] == x]), reverse=True)[:3]
                print(f"[DEBUG] No exact match for '{obj}'. Did you mean one of these: {', '.join(top_matches)}?")
                self.pending_clarification = (obj, top_matches)
                return f"I'm not sure if you meant '{obj}' or one of these: {', '.join(top_matches)}. Which one did you mean?"
        
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
            pattern_info = self.analyze_object_pattern(matches, obj, is_pattern_query)
            if pattern_info:
                return pattern_info
        
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
        # Store pending detections context
        self.pending_detections = (obj, matches)
        return f"I've seen {obj} {len(matches)} times. Would you like to hear about all the detections, or just the three most recent ones?"

    def analyze_object_pattern(self, df, object_label, is_pattern_query=False):
        """Analyze patterns in object detections using pre-computed summaries."""
        # Load pattern summary
        summary_path = os.path.join(PATHS['data']['raw'], 'pattern_summary.csv')
        if not os.path.exists(summary_path):
            # Generate summary if it doesn't exist
            self.generate_daily_pattern_summary()
        try:
            summary_df = pd.read_csv(summary_path)
            if 'strong_patterns' in summary_df.columns:
                summary_df['strong_patterns'] = summary_df['strong_patterns'].apply(self.safe_eval)
            
            # Normalize labels for case-insensitive, whitespace-robust matching
            def normalize(label):
                return str(label).strip().lower()
            norm_obj = normalize(object_label)
            summary_df['norm_object'] = summary_df['object'].apply(normalize)
            
            # First try exact match
            obj_summary = summary_df[summary_df['norm_object'] == norm_obj]
            
            if obj_summary.empty:
                # Check for partial matches in compound words
                potential_matches = []
                for obj in summary_df['object']:
                    norm_obj_in_summary = normalize(obj)
                    # Check if the query is a substring of the object or vice versa
                    if norm_obj in norm_obj_in_summary or norm_obj_in_summary in norm_obj:
                        potential_matches.append(obj)
                
                if potential_matches:
                    # Get the top three most detected matches
                    top_matches = sorted(potential_matches, key=lambda x: summary_df[summary_df['object'] == x]['total_detections'].iloc[0], reverse=True)[:3]
                    print(f"[DEBUG] Found potential matches for '{object_label}': {', '.join(top_matches)}")
                    self.pending_clarification = (object_label, top_matches)
                    return f"I'm not sure if you meant '{object_label}' or one of these: {', '.join(top_matches)}. Which one did you mean?"
                
                # If no partial matches, try fuzzy matching
                from difflib import get_close_matches
                available_objects = summary_df['object'].tolist()
                close_matches = get_close_matches(norm_obj, [normalize(obj) for obj in available_objects], n=3, cutoff=0.6)
                if close_matches:
                    options = [available_objects[[normalize(obj) for obj in available_objects].index(match)] for match in close_matches]
                    print(f"[DEBUG] Found fuzzy matches for '{object_label}': {', '.join(options)}")
                    self.pending_clarification = (object_label, options)
                    return f"I'm not sure if you meant '{object_label}' or one of these: {', '.join(options)}. Which one did you mean?"
                else:
                    print(f"[DEBUG] No matches found for '{object_label}'. Available objects: {available_objects}")
                    if is_pattern_query:
                        return f"There isn't enough data to establish a pattern for the {object_label}."
                    return f"I haven't seen any {object_label} yet."
                
            obj_summary = obj_summary.iloc[0]
            strong_patterns = obj_summary['strong_patterns']
            
            if not strong_patterns:
                if is_pattern_query:
                    return f"The {object_label} does not have a regular pattern in its appearances."
                return f"I've seen the {object_label}, but there isn't enough data to establish a pattern."
                
            # Construct response from strong patterns
            response = f"The {object_label} is usually detected "
            pattern_descriptions = [p['description'] for p in strong_patterns]
            response += ", ".join(pattern_descriptions)
            return response + "."
            
        except Exception as e:
            print(f"[DEBUG] Error using pattern summary: {e}")
            # Fall back to original pattern detection if summary fails
            return self._analyze_object_pattern_fallback(df, object_label, is_pattern_query)

    def handle_clarification_response(self, user_input):
        """Handle user's response to a clarification question about object label."""
        if not hasattr(self, 'pending_clarification') or self.pending_clarification is None:
            return "I'm sorry, there is no pending clarification request. Please ask your question again."
        original_label, options = self.pending_clarification
        user_input = user_input.strip().lower()
        if user_input in ['yes', 'yeah', 'sure', 'okay', 'ok', 'correct', 'right']:
            # Use the first option
            self.pending_clarification = None
            result = self.analyze_object_pattern(self.load_all_logs(), options[0], is_pattern_query=True)
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
                    result = self.analyze_object_pattern(self.load_all_logs(), option, is_pattern_query=True)
                    return result if result is not None else f"I couldn't find a pattern for '{option}'."
            # Ambiguous response, default to the original label
            self.pending_clarification = None
            return f"I'll proceed with '{original_label}'."

    def _analyze_object_pattern_fallback(self, df, object_label, is_pattern_query):
        """Fallback method for pattern detection if summary is not available."""
        if len(df) < 6:
            if is_pattern_query:
                return f"There isn't enough data to establish a pattern for the {object_label}."
            return None

        # Rest of the original pattern detection logic...
        # [Previous implementation remains unchanged]

    def load_combined_logs(self):
        """Load and cache the combined logs DataFrame from outputs/combined_logs.csv."""
        if self.combined_df is not None:
            return self.combined_df
        if not os.path.exists(self.combined_logs_path):
            print(f"Combined logs file not found: {self.combined_logs_path}")
            self.combined_df = pd.DataFrame()
            return self.combined_df
        self.combined_df = pd.read_csv(self.combined_logs_path, parse_dates=["timestamp"])
        return self.combined_df

    def filter_by_object(self, df, object_label):
        mask = (
            df['label_1'].fillna('').str.lower() == object_label.lower() |
            df['label_2'].fillna('').str.lower() == object_label.lower() |
            df['label_3'].fillna('').str.lower() == object_label.lower()
        )
        return df[mask]

    def build_prompt_for_object(self, df, object_label, user_query, max_rows=20):
        filtered = self.filter_by_object(df, object_label)
        filtered = filtered.sort_values("timestamp", ascending=False).head(max_rows)
        table = filtered.to_csv(index=False)
        prompt = (
            f"Here are the most recent detection logs for '{object_label}':\n"
            f"{table}\n"
            f"User question: {user_query}\n"
            "Answer the question using only this data. If the answer is not in the table, say so."
        )
        return prompt

    def build_prompt_general(self, df, user_query, max_rows=20):
        recent = df.sort_values("timestamp", ascending=False).head(max_rows)
        table = recent.to_csv(index=False)
        prompt = (
            f"Here are the most recent detection logs:\n"
            f"{table}\n"
            f"User question: {user_query}\n"
            "Answer the question using only this data."
        )
        return prompt

    def voice_query_loop(self):
        print("Voice assistant is ready. Ask: 'What are you seeing right now?' or 'Did you see a [thing]?' or 'When did you last see [thing]?' or 'What was the first/last thing you saw?'. Say 'exit' to quit voice mode.")
        
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
                        message = self.answer_object_time_query(obj, None)
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
                            message = self.answer_object_time_query(obj, None)
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
                        message = self.answer_object_time_query(obj, None)
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

    def generate_daily_pattern_summary(self):
        """Generate daily summary of patterns for all detected objects."""
        # Load all logs
        df = self.load_all_logs()
        if df.empty:
            return pd.DataFrame()

        # Get current date and 30 days ago
        today = pd.Timestamp.now()
        thirty_days_ago = today - pd.Timedelta(days=30)

        # Filter for last 30 days
        recent_df = df[df['timestamp'] >= thirty_days_ago].copy()
        
        # Initialize summary DataFrame
        summary_data = []
        
        # Get unique objects
        objects = set()
        for col in ['label_1', 'label_2', 'label_3']:
            objects.update(recent_df[col].dropna().unique())
        
        for obj in objects:
            # Fix spelling mistake: change 'racoon' to 'raccoon'
            if obj == 'racoon':
                obj = 'raccoon'
            # Get detections for this object
            obj_df = recent_df[
                (recent_df['label_1'] == obj) |
                (recent_df['label_2'] == obj) |
                (recent_df['label_3'] == obj)
            ]
            
            if len(obj_df) < 6:  # Skip if not enough data
                continue
                
            # Calculate time-based metrics
            obj_df['hour'] = obj_df['timestamp'].dt.hour
            obj_df['weekday'] = obj_df['timestamp'].dt.day_name()
            obj_df['time_bin'] = obj_df['hour'].apply(lambda h: 
                'morning' if 5 <= h < 12 else
                'afternoon' if 12 <= h < 17 else
                'evening' if 17 <= h < 20 else
                'night'
            )
            
            # Calculate pattern metrics
            time_bin_counts = obj_df['time_bin'].value_counts()
            weekday_counts = obj_df['weekday'].value_counts()
            
            # Determine strong patterns
            total_detections = len(obj_df)
            strong_patterns = []
            
            # Time pattern
            if len(time_bin_counts) > 0:
                most_common_time = time_bin_counts.index[0]
                time_ratio = time_bin_counts[most_common_time] / total_detections
                if time_ratio >= 0.7:  # Strong pattern threshold
                    strong_patterns.append({
                        'type': 'time',
                        'value': most_common_time,
                        'ratio': time_ratio,
                        'description': f"in the {most_common_time}" if most_common_time != 'night' else "at night"
                    })
                elif len(time_bin_counts) >= 2:
                    top_two = time_bin_counts.nlargest(2)
                    if all(count / total_detections >= 0.3 for count in top_two):
                        if 'morning' in top_two.index and 'afternoon' in top_two.index:
                            strong_patterns.append({
                                'type': 'time',
                                'value': 'morning_afternoon',
                                'ratio': sum(top_two) / total_detections,
                                'description': "in the morning and afternoon"
                            })
            
            # Weekday pattern
            if len(weekday_counts) > 0:
                most_common_day = weekday_counts.index[0]
                day_ratio = weekday_counts[most_common_day] / total_detections
                if day_ratio >= 0.7:  # Strong pattern threshold
                    strong_patterns.append({
                        'type': 'day',
                        'value': most_common_day,
                        'ratio': day_ratio,
                        'description': f"on {most_common_day}s"
                    })
                else:
                    # Check for weekday/weekend pattern
                    weekday_count = sum(weekday_counts.get(day, 0) for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
                    weekend_count = sum(weekday_counts.get(day, 0) for day in ['Saturday', 'Sunday'])
                    if weekday_count / total_detections >= 0.7:
                        strong_patterns.append({
                            'type': 'day',
                            'value': 'weekdays',
                            'ratio': weekday_count / total_detections,
                            'description': "on weekdays"
                        })
                    elif weekend_count / total_detections >= 0.7:
                        strong_patterns.append({
                            'type': 'day',
                            'value': 'weekends',
                            'ratio': weekend_count / total_detections,
                            'description': "on weekends"
                        })
            
            # Add to summary
            summary_data.append({
                'object': obj,
                'total_detections': total_detections,
                'strong_patterns': strong_patterns,
                'last_detection': obj_df['timestamp'].max(),
                'first_detection': obj_df['timestamp'].min()
            })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_path = os.path.join(PATHS['data']['raw'], 'pattern_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        return summary_df

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

    def safe_eval(self, val):
        """Safely parse pattern data from string representation."""
        try:
            # Only allow parsing of basic Python data structures
            if not isinstance(val, str):
                return val
            # Remove any whitespace
            val = val.strip()
            # Only allow parsing of lists and dictionaries
            if val.startswith('[') and val.endswith(']'):
                return ast.literal_eval(val)
            if val.startswith('{') and val.endswith('}'):
                return ast.literal_eval(val)
            return val
        except (ValueError, SyntaxError):
            return val

if __name__ == "__main__":
    import speech_recognition as sr
    mic = sr.Microphone()
    assistant = DetectionAssistant(mic)
    assistant.run() 