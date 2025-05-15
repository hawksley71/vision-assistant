import os
# Suppress ALSA and other audio library warnings
os.environ["PYTHONWARNINGS"] = "ignore"
try:
    import ctypes
    asound = ctypes.cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(None)
except Exception:
    pass
import cv2
import time
from collections import Counter, defaultdict, deque
import csv
from datetime import datetime, timedelta
from src.models.yolov8_model import YOLOv8Model
import os
import threading
import speech_recognition as sr
from gtts import gTTS
import re
import pandas as pd
import requests
from dotenv import load_dotenv, find_dotenv
from src.config.settings import PATHS, CAMERA_SETTINGS, LOGGING, AUDIO_SETTINGS, HOME_ASSISTANT, MODEL_SETTINGS
import json
import random
import difflib
from .openai_assistant import ask_openai, parse_query_with_openai
import numpy as np
from sklearn.cluster import KMeans
from src.utils.audio import get_microphone
from openai import OpenAI
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union

load_dotenv()
HOME_ASSISTANT_TOKEN = os.getenv("HOME_ASSISTANT_TOKEN")

HEADLESS = os.environ.get("VISION_ASSISTANT_HEADLESS", "0") == "1"

def send_tts_to_ha(message):
    """Send text to Home Assistant for TTS."""
    url = f"{HOME_ASSISTANT['url']}/api/services/tts/cloud_say"
    headers = {
        "Authorization": f"Bearer {HOME_ASSISTANT_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "entity_id": HOME_ASSISTANT['tts_entity'],
        "message": message,
        "language": "en-US",
        "cache": False
    }
    print("[DEBUG] Posting to Home Assistant Cloud TTS:")
    print("[DEBUG] URL:", url)
    print("[DEBUG] Headers:", headers)
    print("[DEBUG] Payload:", json.dumps(payload, indent=2))
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        print(f"[DEBUG] TTS Response Status: {response.status_code}")
        try:
            print("[DEBUG] TTS Response JSON:", response.json())
        except Exception:
            print("[DEBUG] TTS Response Text:", response.text)
        return response.status_code == 200
    except Exception as e:
        print(f"[DEBUG] Could not send to Home Assistant: {e}")
        return False

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

class BasicResponseGenerator(ResponseGenerator):
    """Basic response generator with minimal formatting."""
    def format_single_detection(self, obj: str, timestamp: str, confidence: Optional[float] = None) -> str:
        return f"Yes, I last saw {obj} at {timestamp}."

    def format_multiple_detections(self, obj: str, times: List[str], time_expr: Optional[str] = None) -> str:
        if len(times) == 1:
            return f"Yes, I saw {obj} at {times[0]} {('during ' + time_expr) if time_expr else ''}."
        return f"Yes, I saw {obj} {len(times)} times {('during ' + time_expr) if time_expr else ''}: {', '.join(times)}."

    def format_confidence(self, label: str, confidence: float) -> str:
        percent_conf = int(round(confidence * 100))
        return f"{label}: {percent_conf}%"

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
            return f"{phrase} I saw {obj} at {timestamp}."
        return f"I saw {obj} at {timestamp}."

    def format_multiple_detections(self, obj: str, times: List[str], time_expr: Optional[str] = None) -> str:
        if len(times) == 1:
            return f"I saw {obj} at {times[0]} {('during ' + time_expr) if time_expr else ''}."
        
        # Format multiple detections with transitions
        parts = []
        for i, time in enumerate(times):
            if i == 0:
                parts.append(f"I first saw {obj} at {time}")
            elif i == len(times) - 1:
                parts.append(f"and finally at {time}")
            else:
                parts.append(f"then at {time}")
        
        time_context = f" {('during ' + time_expr) if time_expr else ''}"
        return f"I saw {obj} several times{time_context}: {' '.join(parts)}."

    def format_confidence(self, label: str, confidence: float) -> str:
        phrase = self._get_confidence_phrase(confidence)
        percent_conf = int(round(confidence * 100))
        return f"{phrase} it's {label} ({percent_conf}% confidence)"

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
        return f"I saw {obj} {context} at {timestamp}."

    def format_multiple_detections(self, obj: str, times: List[str], time_expr: Optional[str] = None) -> str:
        if len(times) == 1:
            context = random.choice(self.context_phrases)
            return f"I saw {obj} {context} at {times[0]} {('during ' + time_expr) if time_expr else ''}."
        
        parts = []
        for i, time in enumerate(times):
            context = random.choice(self.context_phrases)
            if i == 0:
                parts.append(f"I first saw {obj} {context} at {time}")
            elif i == len(times) - 1:
                parts.append(f"and finally {context} at {time}")
            else:
                parts.append(f"then {context} at {time}")
        
        time_context = f" {('during ' + time_expr) if time_expr else ''}"
        return f"I saw {obj} several times{time_context}: {' '.join(parts)}."

    def format_confidence(self, label: str, confidence: float) -> str:
        percent_conf = int(round(confidence * 100))
        if confidence >= 0.8:
            return f"I'm very confident ({percent_conf}%) that I saw {label}"
        elif confidence >= 0.5:
            return f"I'm reasonably sure ({percent_conf}%) that I saw {label}"
        return f"I'm not very certain ({percent_conf}%), but I think I saw {label}"

class DetectionAssistant:
    def __init__(self, mic, response_style: str = "natural"):
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

    def normalize_object_label(self, label):
        # Remove all leading articles and extra spaces, and lowercase
        return re.sub(r'^(a |an |the )+', '', label.strip(), flags=re.IGNORECASE).strip().lower()

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

    def answer_object_time_query(self, obj, time_expr):
        # Normalize object
        obj = self.normalize_object_label(obj).lower()
        print(f"[DEBUG] Searching for object: {obj}")
        
        # Parse time expression
        start, end = self.parse_time_expression(time_expr)
        # Only filter by date if a time expression is present
        if time_expr and start is not None and end is not None:
            print(f"[DEBUG] Filtering logs from {start} to {end}")
            df_all = self.load_all_logs()
            df_filtered = df_all[(df_all['timestamp'] >= start) & (df_all['timestamp'] <= end)]
            if df_filtered.empty:
                print(f"[DEBUG] Filtered DataFrame is empty for object '{obj}' and time range {start} to {end}")
                print(f"[DEBUG] DataFrame shape before filtering: {df_all.shape}")
        else:
            df_filtered = self.load_all_logs()
            if df_filtered.empty:
                print(f"[DEBUG] DataFrame is empty when searching for object '{obj}' with no time filter.")
        
        # Robustly collect all unique labels
        all_labels = set()
        for col in ['label_1', 'label_2', 'label_3']:
            if col in df_filtered.columns:
                col_labels = (
                    df_filtered[col]
                    .dropna()
                    .astype(str)
                    .str.lower()
                    .apply(self.normalize_object_label)
                    .unique()
                )
                all_labels.update(col_labels)
        print(f"[DEBUG] Available labels in logs: {sorted(all_labels)}")
        # Create masks for each label column (only if column exists)
        mask1 = (
            df_filtered['label_1'].fillna('').astype(str).str.lower().apply(self.normalize_object_label) == obj
        ) if 'label_1' in df_filtered.columns else pd.Series([False]*len(df_filtered))
        mask2 = (
            df_filtered['label_2'].fillna('').astype(str).str.lower().apply(self.normalize_object_label) == obj
        ) if 'label_2' in df_filtered.columns else pd.Series([False]*len(df_filtered))
        mask3 = (
            df_filtered['label_3'].fillna('').astype(str).str.lower().apply(self.normalize_object_label) == obj
        ) if 'label_3' in df_filtered.columns else pd.Series([False]*len(df_filtered))
        combined_mask = mask1 | mask2 | mask3
        matches = df_filtered[combined_mask]
        if not matches.empty:
            # If no time expression, return the most recent detection and stop
            if not time_expr:
                most_recent = matches.sort_values('timestamp', ascending=False).iloc[0]
                most_recent_time = self.format_timestamp(most_recent['timestamp'])
                # Get average confidence if available
                confidence = None
                for col in ['avg_conf_1', 'avg_conf_2', 'avg_conf_3']:
                    if col in most_recent and pd.notna(most_recent[col]):
                        confidence = most_recent[col]
                        break
                return self.response_generator.format_single_detection(obj, most_recent_time, confidence)
            
            # Sort matches by timestamp
            matches = matches.sort_values('timestamp', ascending=False)
            times = [self.format_timestamp(ts) for ts in matches['timestamp']]
            
            # If more than 3 matches, ask if user wants to hear all
            if len(times) > 3:
                self.pending_detections = times  # Store all times for potential follow-up
                return f"I found {len(times)} detections of {obj} {('during ' + time_expr) if time_expr else ''}. Would you like to hear all of them, or just the three most recent?"
            
            # If 3 or fewer matches, return them all
            return self.response_generator.format_multiple_detections(obj, times, time_expr)
        else:
            # Try fuzzy matching if exact match fails
            closest_match = self.find_closest_label(obj, all_labels)
            if closest_match and closest_match != obj:
                print(f"[DEBUG] No exact match found, closest match: {closest_match}")
                mask1 = (
                    df_filtered['label_1'].fillna('').astype(str).str.lower().apply(self.normalize_object_label) == closest_match
                ) if 'label_1' in df_filtered.columns else pd.Series([False]*len(df_filtered))
                mask2 = (
                    df_filtered['label_2'].fillna('').astype(str).str.lower().apply(self.normalize_object_label) == closest_match
                ) if 'label_2' in df_filtered.columns else pd.Series([False]*len(df_filtered))
                mask3 = (
                    df_filtered['label_3'].fillna('').astype(str).str.lower().apply(self.normalize_object_label) == closest_match
                ) if 'label_3' in df_filtered.columns else pd.Series([False]*len(df_filtered))
                combined_mask = mask1 | mask2 | mask3
                matches = df_filtered[combined_mask]
                if not matches.empty:
                    if not time_expr:
                        most_recent = matches.sort_values('timestamp', ascending=False).iloc[0]
                        most_recent_time = self.format_timestamp(most_recent['timestamp'])
                        return f"Yes, I last saw {closest_match} at {most_recent_time}."
                    
                    # Sort matches by timestamp
                    matches = matches.sort_values('timestamp', ascending=False)
                    times = [self.format_timestamp(ts) for ts in matches['timestamp']]
                    
                    # If more than 3 matches, ask if user wants to hear all
                    if len(times) > 3:
                        self.pending_detections = times  # Store all times for potential follow-up
                        return f"I found {len(times)} detections of {closest_match} {('during ' + time_expr) if time_expr else ''}. Would you like to hear all of them, or just the three most recent?"
                    
                    # If 3 or fewer matches, return them all
                    if len(times) == 1:
                        return f"Yes, I saw {closest_match} at {times[0]} {('during ' + time_expr) if time_expr else ''}."
                    else:
                        return f"Yes, I saw {closest_match} {len(times)} times {('during ' + time_expr) if time_expr else ''}: {', '.join(times)}."
            print(f"[DEBUG] No match found for '{obj}' in available labels: {sorted(all_labels)}")
            # Partial substring match: greedy, up to 3 most recent unique matches
            partial_matches = []
            partial_match_counts = {}
            found_labels = set()
            for idx, row in df_filtered.sort_values('timestamp', ascending=False).iterrows():
                for col in ['label_1', 'label_2', 'label_3']:
                    label = str(row.get(col, '')).lower().strip()
                    norm_label = self.normalize_object_label(label)
                    if norm_label and obj in norm_label and norm_label != obj and norm_label not in found_labels:
                        found_labels.add(norm_label)
                        partial_matches.append(norm_label)
                        partial_match_counts[norm_label] = 1
                        if len(partial_matches) == 3:
                            break
                if len(partial_matches) == 3:
                    break
            if partial_matches:
                for norm_label in partial_matches:
                    count = 0
                    for col in ['label_1', 'label_2', 'label_3']:
                        count += (df_filtered[col].fillna('').astype(str).str.lower().apply(self.normalize_object_label) == norm_label).sum()
                    partial_match_counts[norm_label] = count
                partial_matches_sorted = sorted(partial_matches, key=lambda l: -partial_match_counts[l])
                return (f"No, I have not seen {obj}{' ' + time_expr if time_expr else ''}, "
                        f"but I have seen: {', '.join(partial_matches_sorted)}. Did you mean one of these?")
            return f"No, I have not seen {obj}{' ' + time_expr if time_expr else ''}."

    def analyze_object_pattern(self, df, object_label):
        mask = (
            (df['label_1'] == object_label) |
            (df['label_2'] == object_label) |
            (df['label_3'] == object_label)
        )
        obj_df = df[mask].copy()
        if obj_df.empty:
            return f"I have not seen any {object_label} in the logs."
        obj_df['timestamp'] = pd.to_datetime(obj_df['timestamp'])
        obj_df['weekday'] = obj_df['timestamp'].dt.day_name()
        obj_df['month'] = obj_df['timestamp'].dt.month_name()
        obj_df['hour_minute'] = obj_df['timestamp'].dt.hour * 60 + obj_df['timestamp'].dt.minute

        # Days of week
        day_counts = obj_df['weekday'].value_counts()
        common_days = day_counts[day_counts > 1].index.tolist()

        # Months
        month_counts = obj_df['month'].value_counts()
        common_months = month_counts[month_counts > 1].index.tolist()

        # Time clustering (k=2 for morning/afternoon, fallback to 1)
        times = obj_df['hour_minute'].values.reshape(-1, 1)
        n_clusters = 2 if len(times) >= 4 else 1
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(times)
        centers = sorted(kmeans.cluster_centers_.flatten())
        time_strings = [f"{int(c//60):02d}:{int(c%60):02d}" for c in centers]

        response = []
        if common_days:
            response.append(f"The {object_label} is usually detected on {', '.join(common_days)}")
        if n_clusters == 2:
            response.append(f"at around {time_strings[0]} and {time_strings[1]}")
        else:
            response.append(f"at around {time_strings[0]}")
        if common_months:
            response.append(f"in {', '.join(common_months)}")
        if not common_days and not common_months:
            response = [f"The {object_label} does not have a regular pattern of appearance."]
        return " ".join(response) + "."

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
            r"when was the last time you saw (?:a |an )?([\w\s]+)\??"
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
                        obj = match.group(1).strip() if match.group(1) else obj
                        if obj:
                            obj = self.normalize_object_label(obj)
                            print(f"[DEBUG] Extracted object from 'did you see' pattern: {obj}")
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

    def answer_live_query(self, user_input):
        """Given a user query about the live camera feed, return a response string."""
        query = user_input.lower()
        
        # Handle "what do you see" queries
        if "what do you see" in query or "what are you seeing" in query:
            # Get the top 3 most frequent objects from the buffer
            label_counts = Counter()
            for det in self.detections_buffer:
                label_counts[det['class_name']] += 1
            
            top_labels = label_counts.most_common(3)
            if not top_labels:
                return "I'm not seeing anything right now."
            
            # Format the response
            labels = [label for label, _ in top_labels]
            response = "Right now, I am seeing: " + self.natural_list(labels) + "."
            
            # Store the reported labels and their confidences for confidence queries
            self.last_reported_labels = labels
            self.last_reported_confidences = {
                label: [d['confidence'] for d in self.detections_buffer if d['class_name'] == label]
                for label in labels
            }
            
            return response
            
        # Handle confidence queries
        confidence_keywords = ["confident", "confidence", "sure", "certain", "accurate", "reliable"]
        if any(word in query for word in confidence_keywords):
            return self.summarize_buffer_confidence()
            
        # Default response for unrecognized queries
        return "I'm not sure how to answer that about the live feed."

if __name__ == "__main__":
    assistant = DetectionAssistant()
    assistant.run() 