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
from collections import Counter, defaultdict
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
from dotenv import load_dotenv
from ..config.settings import PATHS, CAMERA_SETTINGS, LOGGING_SETTINGS, AUDIO_SETTINGS, HOME_ASSISTANT
import json
import random
import difflib

load_dotenv()
HOME_ASSISTANT_TOKEN = os.getenv("HOME_ASSISTANT_TOKEN")

HEADLESS = os.environ.get("VISION_ASSISTANT_HEADLESS", "0") == "1"

class DetectionAssistant:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SETTINGS['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SETTINGS['height'])
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_SETTINGS['fps'])
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open camera")

        # Initialize YOLOv8 model
        self.model = YOLOv8Model()
        self.latest_detections = []
        self.fps = 0

        # For logging
        self.log_path = os.path.join(PATHS['data']['raw'], f"detections_{datetime.now().strftime(LOGGING_SETTINGS['log_format'])}.csv")
        self.last_log_time = time.time()
        self.detections_buffer = []  # Store detections for the current 1s interval
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
        self.mic = sr.Microphone()

        self.last_reported_labels = []  # Track last reported labels for confidence queries
        self.last_reported_confidences = {}  # Track last reported confidences

    def log_top_labels(self):
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
        # Prepare row
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = [now]
        for label, count in top_labels:
            avg_conf = sum(label_confidences[label]) / len(label_confidences[label])
            row.extend([label, count, round(avg_conf, 3)])
        # Pad row if fewer than 3 labels
        while len(row) < 10:
            row.extend(["", "", ""])
        # Write to CSV
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        # Clear buffer
        self.detections_buffer = []

    def natural_list(self, items):
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    def summarize_buffer_labels(self):
        # Only return the top 1-3 labels (no confidence or count)
        if not self.detections_buffer:
            self.last_reported_labels = []
            self.last_reported_confidences = {}
            return "I'm not seeing anything right now."
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
            print("[DEBUG] No last reported labels/confidences available.")
            return "I haven't reported any detections yet."
        parts = []
        for label in self.last_reported_labels:
            confs = self.last_reported_confidences.get(label, [])
            print(f"[DEBUG] Label: {label}, Confidences: {confs}")
            if confs:
                avg_conf = sum(confs) / len(confs)
                percent_conf = int(round(avg_conf * 100))
                parts.append(f"{label}: {percent_conf}%")
        if not parts:
            print("[DEBUG] No confidence information for last detection.")
            return "I don't have confidence information for the last detection."
        if len(parts) == 1:
            return f"My average confidence for {parts[0].split(':')[0]} is {parts[0].split(':')[1].strip()}."
        return "My average confidences are: " + ", ".join(parts) + "."

    def parse_time_expression(self, time_expr):
        today = pd.Timestamp.today().normalize()
        if not time_expr or time_expr == "today":
            return today, today
        elif time_expr == "yesterday":
            return today - pd.Timedelta(days=1), today - pd.Timedelta(days=1)
        elif time_expr == "last week":
            start = today - pd.Timedelta(days=today.weekday() + 7)
            end = start + pd.Timedelta(days=6)
            return start, end
        elif time_expr == "this week":
            start = today - pd.Timedelta(days=today.weekday())
            end = start + pd.Timedelta(days=6)
            return start, end
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
        # Add more cases as needed
        return None, None

    def find_closest_label(self, partial_label, known_labels):
        # Use difflib to find the closest match
        matches = difflib.get_close_matches(partial_label, known_labels, n=1, cutoff=0.6)
        if matches:
            return matches[0]
        return partial_label

    def normalize_object_label(self, label):
        # Remove leading articles (a, an, the) and extra spaces
        return re.sub(r'^(a |an |the )', '', label.strip(), flags=re.IGNORECASE)

    def answer_object_time_query(self, obj, time_expr):
        # Load logs
        def load_all_logs(log_dir="data/raw"):
            all_dfs = []
            log_files = sorted([f for f in os.listdir(log_dir) if f.endswith(".csv")])
            for f in log_files:
                try:
                    df = pd.read_csv(os.path.join(log_dir, f))
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Warning: Skipped {f} due to read error: {e}")
            if not all_dfs:
                return pd.DataFrame(columns=["timestamp", "label_1", "count_1", "avg_conf_1", "label_2", "count_2", "avg_conf_2", "label_3", "count_3", "avg_conf_3"])
            return pd.concat(all_dfs, ignore_index=True).sort_values("timestamp")
        df_all = load_all_logs()
        if df_all.empty:
            return "I haven't seen anything yet."
        # Normalize object
        obj = self.normalize_object_label(obj)
        # Parse time expression
        start, end = self.parse_time_expression(time_expr)
        if start is not None and end is not None:
            df_filtered = df_all[(df_all['timestamp'] >= start) & (df_all['timestamp'] <= end)]
        else:
            df_filtered = df_all
        # Search for object in filtered logs
        mask = (
            df_filtered['label_1'].fillna('').str.lower() == obj or
            df_filtered['label_2'].fillna('').str.lower() == obj or
            df_filtered['label_3'].fillna('').str.lower() == obj
        )
        matches = df_filtered.loc[mask]
        if not matches.empty:
            times = matches['timestamp'].dt.strftime("%I:%M %p on %A").str.lstrip("0").tolist()
            if len(times) == 1:
                return f"Yes, I saw {obj} at {times[0]} {('during ' + time_expr) if time_expr else ''}."
            else:
                return f"Yes, I saw {obj} {len(times)} times {('during ' + time_expr) if time_expr else ''}: {', '.join(times)}."
        else:
            return f"No, I did not see {obj}{' ' + time_expr if time_expr else ''}."

    def voice_query_loop(self):
        print("Voice assistant is ready. Ask: 'What are you seeing right now?' or 'Did you see a [thing]?' or 'When did you last see [thing]?' or 'What was the first/last thing you saw?'. Say 'exit' to quit voice mode.")
        # Define regex patterns for each query type
        live_patterns = [
            r"(what|tell me|show me).*(see|detect|seeing|detecting|there|in front)",
            r"what else"
        ]
        last_thing_patterns = [
            r"(last|most recent).*(thing|object|detection)(.*(see|detect))?",
            r"what did you see last",
            r"what did you detect last",
            r"what was the last thing you saw",
            r"what was the last object you detected",
            r"what did you just see",
            r"what did you just detect",
            r"what was the last detection",
            r"tell me the last thing you saw",
            r"tell me the last object you detected",
            r"what was the last thing detected",
            r"what did you see a moment ago",
            r"what was the last thing",
            r"last thing",
            r"most recent thing"
        ]
        first_thing_patterns = [
            r"first.*thing.*see|detect",
            r"what was the first thing you saw",
            r"what was the first object you detected",
            r"tell me the first thing you saw",
            r"what did you see first"
        ]
        did_you_see_patterns = [
            r"did you see (?:a |an |any )?([\w\s]+)\??",
            r"have you seen (?:a |an |any )?([\w\s]+)\??"
        ]
        last_seen_patterns = [
            r"when did you (?:last )?see (?:a |an )?([\w\s]+)\??",
            r"when was the last time you saw (?:a |an )?([\w\s]+)\??"
        ]
        confidence_patterns = [
            r"confident",
            r"confidence",
            r"sure"
        ]
        # Add more flexible pattern for follow-up queries about 'that'
        followup_last_seen_patterns = [
            r"when (?:did|have)? ?you (?:see|saw|spotted|detect(?:ed)?) (?:that|it|them)? ?(?:last|previously)?",
            r"when (?:was|is) the last time you (?:saw|spotted|detected) (?:that|it|them)?",
            r"last time you (?:saw|spotted|detected) (?:that|it|them)?",
            r"have you (?:seen|spotted|detected) (?:that|it|them)? before"
        ]
        # Patterns for 'usual time' and 'frequency' queries
        usual_time_patterns = [
            r"when does the ([\w \-]+) usually come",
            r"what time does the ([\w \-]+) usually come",
            r"when is the ([\w \-]+) usually here",
            r"what time does the ([\w \-]+) arrive",
            r"what time do you usually see the ([\w \-]+)",
            r"when do you usually see the ([\w \-]+)",
            r"when is the ([\w \-]+) usually seen",
            r"when does the ([\w \-]+) show up",
            r"what time is the ([\w \-]+) usually seen",
            r"what time does the ([\w \-]+) show up"
        ]
        frequency_patterns = [
            r"how many days a week does the ([\w \-]+) come",
            r"how often does the ([\w \-]+) come",
            r"on which days does the ([\w \-]+) come",
            r"what days does the ([\w \-]+) come",
            r"which days does the ([\w \-]+) come",
            r"what days of the week does the ([\w \-]+) come"
        ]
        day_of_week_patterns = [
            r"what day of the week does the ([\w \-]+) come",
            r"which day does the ([\w \-]+) come",
            r"what days does the ([\w \-]+) come",
            r"which days does the ([\w \-]+) come",
            r"what days of the week does the ([\w \-]+) come"
        ]
        day_of_month_patterns = [
            r"what day of the month does the ([\w \-]+) come",
            r"which day of the month does the ([\w \-]+) come"
        ]
        month_patterns = [
            r"which months does the ([\w \-]+) come",
            r"what month does the ([\w \-]+) come"
        ]
        weekend_patterns = [
            r"does the ([\w \-]+) come on weekends?",
            r"does the ([\w \-]+) come on weekdays?"
        ]
        specific_day_patterns = [
            r"does the ([\w \-]+) come on (monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?"
        ]
        pending_time_expr = None
        while self.voice_active:
            try:
                with self.mic as source:
                    self.r.adjust_for_ambient_noise(source)
                    print("Listening...")
                    audio = self.r.listen(source, timeout=5)
                query = self.r.recognize_google(audio).lower()
                print(f"[DEBUG] Recognized query: {query}")
                if any(word in query for word in ["exit", "quit", "stop", "cancel"]):
                    # Exit acknowledgment phrases
                    exit_phrases = [
                        "Okay, I am shutting down now. Goodbye!",
                        "Exiting the assistant. Have a great day!",
                        "Understood, I'm turning off. See you next time!",
                        "The assistant is now exiting. Take care!"
                    ]
                    exit_message = random.choice(exit_phrases)
                    print(f"[DEBUG] Exit acknowledged: {exit_message}")
                    # Send exit message to Home Assistant TTS
                    def send_tts_to_ha(message):
                        url = "http://localhost:8123/api/services/tts/cloud_say"
                        headers = {
                            "Authorization": f"Bearer {HOME_ASSISTANT_TOKEN}",
                            "Content-Type": "application/json",
                        }
                        payload = {
                            "entity_id": "media_player.den_speaker",
                            "message": message,
                            "language": "en-US",
                            "cache": False
                        }
                        print("[DEBUG] Posting to Home Assistant Cloud TTS (exit):")
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
                        except Exception as e:
                            print(f"[DEBUG] Could not send exit message to Home Assistant: {e}")
                    threading.Thread(target=send_tts_to_ha, args=(exit_message,), daemon=True).start()
                    print("Exiting voice assistant.")
                    self.voice_active = False
                    break
                message = "Sorry, I didn't understand that."
                # Confidence query (check first)
                if any(re.search(pattern, query) for pattern in confidence_patterns):
                    print("[DEBUG] Confidence query detected. Calling summarize_buffer_confidence().")
                    message = self.summarize_buffer_confidence()
                # Live detection query (buffer summary, labels only)
                elif any(re.search(pattern, query) for pattern in live_patterns):
                    message = self.summarize_buffer_labels()
                # Check for follow-up queries about 'that'
                elif any(re.search(pattern, query, re.IGNORECASE) for pattern in followup_last_seen_patterns):
                    # Use self.last_reported_labels for context
                    if not self.last_reported_labels:
                        message = "I'm not sure what 'that' refers to. Please ask what I am seeing first."
                    else:
                        # Load logs
                        def load_all_logs(log_dir="data/raw"):
                            all_dfs = []
                            log_files = sorted([f for f in os.listdir(log_dir) if f.endswith(".csv")])
                            for f in log_files:
                                try:
                                    df = pd.read_csv(os.path.join(log_dir, f))
                                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                                    all_dfs.append(df)
                                except Exception as e:
                                    print(f"Warning: Skipped {f} due to read error: {e}")
                            if not all_dfs:
                                return pd.DataFrame(columns=["timestamp", "label_1", "count_1", "avg_conf_1", "label_2", "count_2", "avg_conf_2", "label_3", "count_3", "avg_conf_3"])
                            return pd.concat(all_dfs, ignore_index=True).sort_values("timestamp")
                        df_all = load_all_logs()
                        responses = []
                        for label in self.last_reported_labels:
                            matches = df_all[(df_all['label_1'].str.lower() == label.lower()) |
                                             (df_all['label_2'].str.lower() == label.lower()) |
                                             (df_all['label_3'].str.lower() == label.lower())]
                            if not matches.empty:
                                last_time = matches.iloc[-1]['timestamp']
                                spoken_time = last_time.strftime("%I:%M %p on %B %d").lstrip("0")
                                responses.append(f"The last time I saw {label} was at {spoken_time}.")
                            else:
                                responses.append(f"I have not seen {label} before.")
                        message = " ".join(responses)
                else:
                    matched = False
                    # Usual time queries
                    for pattern in usual_time_patterns:
                        m = re.search(pattern, query, re.IGNORECASE)
                        if m:
                            obj = self.normalize_object_label(m.group(1).strip())
                            # Load logs
                            def load_all_logs(log_dir="data/raw"):
                                all_dfs = []
                                log_files = sorted([f for f in os.listdir(log_dir) if f.endswith(".csv")])
                                for f in log_files:
                                    try:
                                        df = pd.read_csv(os.path.join(log_dir, f))
                                        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                                        all_dfs.append(df)
                                    except Exception as e:
                                        print(f"Warning: Skipped {f} due to read error: {e}")
                                if not all_dfs:
                                    return pd.DataFrame(columns=["timestamp", "label_1", "count_1", "avg_conf_1", "label_2", "count_2", "avg_conf_2", "label_3", "count_3", "avg_conf_3"])
                                return pd.concat(all_dfs, ignore_index=True).sort_values("timestamp")
                            df_all = load_all_logs()
                            # Find all timestamps for the object (normalized)
                            mask = (
                                df_all['label_1'].fillna('').apply(normalize_label) == obj |
                                df_all['label_2'].fillna('').apply(normalize_label) == obj |
                                df_all['label_3'].fillna('').apply(normalize_label) == obj
                            )
                            times = df_all.loc[mask, 'timestamp']
                            if times.empty:
                                message = f"I have not seen {m.group(1).strip()} before."
                            else:
                                # Find most common hour/minute (rounded to nearest 5 min)
                                rounded_times = times.dt.hour * 60 + (times.dt.minute // 5) * 5
                                from collections import Counter
                                most_common = Counter(rounded_times).most_common(2)
                                spoken_times = []
                                for t, _ in most_common:
                                    hour = t // 60
                                    minute = t % 60
                                    spoken_times.append(datetime(2000, 1, 1, hour, minute).strftime("%-I:%M %p"))
                                message = f"The {m.group(1).strip()} usually comes at " + " and ".join(spoken_times) + "."
                            matched = True
                            break
                    # Frequency queries
                    if not matched:
                        for pattern in frequency_patterns:
                            m = re.search(pattern, query, re.IGNORECASE)
                            if m:
                                obj = self.normalize_object_label(m.group(1).strip())
                                if 'df_all' in locals() and not df_all.empty:
                                    known_labels = set()
                                    for col in ["label_1", "label_2", "label_3"]:
                                        known_labels.update(df_all[col].dropna().str.lower().unique())
                                else:
                                    known_labels = set()
                                obj = self.find_closest_label(obj, known_labels)
                                df_all = load_all_logs()
                                mask = (
                                    df_all['label_1'].fillna('').apply(normalize_label) == obj |
                                    df_all['label_2'].fillna('').apply(normalize_label) == obj |
                                    df_all['label_3'].fillna('').apply(normalize_label) == obj
                                )
                                times = df_all.loc[mask, 'timestamp']
                                if times.empty:
                                    message = f"I have not seen {m.group(1).strip()} before."
                                else:
                                    df = df_all.loc[mask].copy()
                                    df['week'] = df['timestamp'].dt.isocalendar().week
                                    df['year'] = df['timestamp'].dt.year
                                    days_per_week = df.groupby(['year', 'week'])['timestamp'].apply(lambda x: x.dt.date.nunique())
                                    avg_days = days_per_week.mean()
                                    most_common_days = int(round(days_per_week.mode().iloc[0])) if not days_per_week.mode().empty else int(round(avg_days))
                                    message = f"The {m.group(1).strip()} comes about {most_common_days} day{'s' if most_common_days != 1 else ''} per week."
                                matched = True
                                break
                    # Day of week queries
                    if not matched:
                        for pattern in day_of_week_patterns:
                            m = re.search(pattern, query, re.IGNORECASE)
                            if m:
                                obj = self.normalize_object_label(m.group(1).strip())
                                df_all = load_all_logs()
                                mask = (
                                    df_all['label_1'].fillna('').apply(normalize_label) == obj |
                                    df_all['label_2'].fillna('').apply(normalize_label) == obj |
                                    df_all['label_3'].fillna('').apply(normalize_label) == obj
                                )
                                times = df_all.loc[mask, 'timestamp']
                                if times.empty:
                                    message = f"I have not seen {m.group(1).strip()} before."
                                else:
                                    days = times.dt.day_name().value_counts().index.tolist()
                                    message = f"The {m.group(1).strip()} usually comes on " + ", ".join(days) + "."
                                matched = True
                                break
                    # Day of month queries
                    if not matched:
                        for pattern in day_of_month_patterns:
                            m = re.search(pattern, query, re.IGNORECASE)
                            if m:
                                obj = self.normalize_object_label(m.group(1).strip())
                                df_all = load_all_logs()
                                mask = (
                                    df_all['label_1'].fillna('').apply(normalize_label) == obj |
                                    df_all['label_2'].fillna('').apply(normalize_label) == obj |
                                    df_all['label_3'].fillna('').apply(normalize_label) == obj
                                )
                                times = df_all.loc[mask, 'timestamp']
                                if times.empty:
                                    message = f"I have not seen {m.group(1).strip()} before."
                                else:
                                    days = times.dt.day.value_counts().index.tolist()
                                    message = f"The {m.group(1).strip()} usually comes on day(s) " + ", ".join(str(d) for d in days) + " of the month."
                                matched = True
                                break
                    # Month queries
                    if not matched:
                        for pattern in month_patterns:
                            m = re.search(pattern, query, re.IGNORECASE)
                            if m:
                                obj = self.normalize_object_label(m.group(1).strip())
                                df_all = load_all_logs()
                                mask = (
                                    df_all['label_1'].fillna('').apply(normalize_label) == obj |
                                    df_all['label_2'].fillna('').apply(normalize_label) == obj |
                                    df_all['label_3'].fillna('').apply(normalize_label) == obj
                                )
                                times = df_all.loc[mask, 'timestamp']
                                if times.empty:
                                    message = f"I have not seen {m.group(1).strip()} before."
                                else:
                                    months = times.dt.month_name().value_counts().index.tolist()
                                    message = f"The {m.group(1).strip()} usually comes in " + ", ".join(months) + "."
                                matched = True
                                break
                    # Weekend/weekday queries
                    if not matched:
                        for pattern in weekend_patterns:
                            m = re.search(pattern, query, re.IGNORECASE)
                            if m:
                                obj = self.normalize_object_label(m.group(1).strip())
                                df_all = load_all_logs()
                                mask = (
                                    df_all['label_1'].fillna('').apply(normalize_label) == obj |
                                    df_all['label_2'].fillna('').apply(normalize_label) == obj |
                                    df_all['label_3'].fillna('').apply(normalize_label) == obj
                                )
                                times = df_all.loc[mask, 'timestamp']
                                if times.empty:
                                    message = f"I have not seen {m.group(1).strip()} before."
                                else:
                                    if 'weekend' in m.group(0):
                                        weekend_days = times[times.dt.weekday >= 5]
                                        if not weekend_days.empty:
                                            message = f"Yes, the {m.group(1).strip()} comes on weekends."
                                        else:
                                            message = f"No, the {m.group(1).strip()} does not come on weekends."
                                    else:
                                        weekday_days = times[times.dt.weekday < 5]
                                        if not weekday_days.empty:
                                            message = f"Yes, the {m.group(1).strip()} comes on weekdays."
                                        else:
                                            message = f"No, the {m.group(1).strip()} does not come on weekdays."
                                matched = True
                                break
                    # Specific day queries (yes/no)
                    if not matched:
                        for pattern in specific_day_patterns:
                            m = re.search(pattern, query, re.IGNORECASE)
                            if m:
                                obj = self.normalize_object_label(m.group(1).strip())
                                day = m.group(2).capitalize()
                                df_all = load_all_logs()
                                mask = (
                                    df_all['label_1'].fillna('').apply(normalize_label) == obj |
                                    df_all['label_2'].fillna('').apply(normalize_label) == obj |
                                    df_all['label_3'].fillna('').apply(normalize_label) == obj
                                )
                                times = df_all.loc[mask, 'timestamp']
                                if times.empty:
                                    message = f"I have not seen {m.group(1).strip()} before."
                                else:
                                    if day in times.dt.day_name().unique():
                                        message = f"Yes, the {m.group(1).strip()} comes on {day}s."
                                    else:
                                        message = f"No, the {m.group(1).strip()} does not come on {day}s."
                                matched = True
                                break
                    if not matched:
                        # If no object found but a time expression is present, prompt for object
                        time_expr_regex = r"(today|yesterday|last week|this week|last month|this month|this weekend|last weekend|in (?:january|february|march|april|may|june|july|august|september|october|november|december))"
                        time_expr_match = re.search(time_expr_regex, query)
                        found_time = time_expr_match.group(0) if time_expr_match else None
                        if found_time:
                            pending_time_expr = found_time
                            message = "What object are you asking about?"
                        elif pending_time_expr:
                            obj = self.normalize_object_label(query.strip())
                            # Now combine with pending_time_expr and answer
                            message = self.answer_object_time_query(obj, pending_time_expr)
                            pending_time_expr = None
                print(f"[USER QUERY] {query}")
                print(f"[ASSISTANT RESPONSE] {message}")
                # Send message to Home Assistant TTS (cloud_say)
                def send_tts_to_ha(message):
                    url = "http://localhost:8123/api/services/tts/cloud_say"
                    headers = {
                        "Authorization": f"Bearer {HOME_ASSISTANT_TOKEN}",
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "entity_id": "media_player.den_speaker",
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
                    except Exception as e:
                        print(f"[DEBUG] Could not send message to Home Assistant: {e}")
                # Run TTS in a thread (non-blocking)
                threading.Thread(target=send_tts_to_ha, args=(message,), daemon=True).start()
            except sr.WaitTimeoutError:
                print("No speech detected. Try again...")
            except sr.UnknownValueError:
                print("Sorry, could not understand.")
            except Exception as e:
                print(f"Voice assistant error: {e}")

    def start_intro_and_voice(self):
        # Wait 3 seconds while detection buffer fills
        print("Warming up, please wait...")
        time.sleep(3)
        # Generate and play intro via Home Assistant
        intro_text = "Hello! I am your vision-aware assistant. I can see and detect objects in my view. Ask me what I see, or about past detections."
        def send_tts_to_ha(message):
            url = "http://localhost:8123/api/services/tts/cloud_say"
            headers = {
                "Authorization": f"Bearer {HOME_ASSISTANT_TOKEN}",
                "Content-Type": "application/json",
            }
            payload = {
                "entity_id": "media_player.den_speaker",
                "message": message,
                "language": "en-US",
                "cache": False
            }
            print("[DEBUG] Posting to Home Assistant Cloud TTS (intro):")
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
            except Exception as e:
                print(f"[DEBUG] Could not send intro to Home Assistant: {e}")
        threading.Thread(target=send_tts_to_ha, args=(intro_text,), daemon=True).start()
        self.voice_thread.start()

    def run(self):
        frame_count = 0
        start_time = time.time()
        print("Press 'q' to quit the live feed window.")
        # Start intro and voice assistant in a background thread
        intro_thread = threading.Thread(target=self.start_intro_and_voice, daemon=True)
        intro_thread.start()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            # Run detection
            detections = self.model.detect(frame)
            self.latest_detections = detections
            self.detections_buffer.extend(detections)
            # Draw detections
            frame = self.model.draw_detections(frame, detections)
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
            if not HEADLESS:
                cv2.imshow('Live Detection Assistant', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        self.voice_active = False
        self.voice_thread.join()
        self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    assistant = DetectionAssistant()
    assistant.run() 