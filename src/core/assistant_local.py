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

load_dotenv()
HOME_ASSISTANT_TOKEN = os.getenv("HOME_ASSISTANT_TOKEN")

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
        top_labels = [label for label, _ in label_counts.most_common(3)]
        self.last_reported_labels = top_labels
        self.last_reported_confidences = {label: label_confidences[label] for label in top_labels}
        if not top_labels:
            return "I'm not seeing anything right now."
        return "Right now, I am seeing: " + ", ".join(top_labels) + "."

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
                parts.append(f"{label}: {avg_conf:.2f}")
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
        while self.voice_active:
            try:
                with self.mic as source:
                    self.r.adjust_for_ambient_noise(source)
                    print("Listening...")
                    audio = self.r.listen(source, timeout=5)
                query = self.r.recognize_google(audio).lower()
                print(f"[DEBUG] Recognized query: {query}")
                if any(word in query for word in ["exit", "quit", "stop", "cancel"]):
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
                else:
                    # Historical queries from log
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
                        message = "I haven't seen anything yet."
                    else:
                        # Last thing query
                        for pattern in last_thing_patterns:
                            if re.search(pattern, query):
                                print(f"[DEBUG] Matched last_thing_pattern: {pattern}")
                                # Regex for time expressions
                                time_expr_regex = r"(today|yesterday|last week|this week|last month|this month|this weekend|last weekend|in (?:january|february|march|april|may|june|july|august|september|october|november|december))"
                                time_expr_match = re.search(time_expr_regex, query)
                                found_time = time_expr_match.group(0) if time_expr_match else None
                                df_filtered = df_all
                                if found_time:
                                    start, end = self.parse_time_expression(found_time)
                                    if start is not None and end is not None:
                                        df_filtered = df_filtered[(df_filtered['timestamp'] >= start) & (df_filtered['timestamp'] <= end)]
                                        print(f"[DEBUG] Filtering logs from {start} to {end}")
                                    else:
                                        message = f"Sorry, I couldn't understand the time expression '{found_time}'."
                                        break
                                if df_filtered.empty:
                                    message = f"I didn't see anything{(' ' + found_time) if found_time else ''}."
                                    print(f"[DEBUG] No entries found for last thing query in range: {found_time}")
                                    break
                                last_row = df_filtered.iloc[-1]
                                print(f"[DEBUG] Last row: {last_row}")
                                for col in ["label_1", "label_2", "label_3"]:
                                    label = last_row[col]
                                    print(f"[DEBUG] Checking last label column {col}: {label}")
                                    if label:
                                        last_time = last_row['timestamp']
                                        spoken_time = last_time.strftime("%I:%M %p on %B %d").lstrip("0")
                                        message = f"The last thing I saw was {label} at {spoken_time}."
                                        print(f"[DEBUG] Responding with last label: {label}")
                                        break
                                break
                        # First thing query
                        for pattern in first_thing_patterns:
                            if re.search(pattern, query):
                                print(f"[DEBUG] Matched first_thing_pattern: {pattern}")
                                # Regex for time expressions
                                time_expr_regex = r"(today|yesterday|last week|this week|last month|this month|this weekend|last weekend|in (?:january|february|march|april|may|june|july|august|september|october|november|december))"
                                time_expr_match = re.search(time_expr_regex, query)
                                found_time = time_expr_match.group(0) if time_expr_match else None
                                df_filtered = df_all
                                if found_time:
                                    start, end = self.parse_time_expression(found_time)
                                    if start is not None and end is not None:
                                        df_filtered = df_filtered[(df_filtered['timestamp'] >= start) & (df_filtered['timestamp'] <= end)]
                                        print(f"[DEBUG] Filtering logs from {start} to {end}")
                                    else:
                                        message = f"Sorry, I couldn't understand the time expression '{found_time}'."
                                        break
                                if df_filtered.empty:
                                    message = f"I didn't see anything{(' ' + found_time) if found_time else ''}."
                                    print(f"[DEBUG] No entries found for first thing query in range: {found_time}")
                                    break
                                first_row = df_filtered.iloc[0]
                                print(f"[DEBUG] First row: {first_row}")
                                for col in ["label_1", "label_2", "label_3"]:
                                    label = first_row[col]
                                    print(f"[DEBUG] Checking first label column {col}: {label}")
                                    if label:
                                        first_time = first_row['timestamp']
                                        spoken_time = first_time.strftime("%I:%M %p on %B %d").lstrip("0")
                                        message = f"The first thing I saw was {label} at {spoken_time}."
                                        print(f"[DEBUG] Responding with first label: {label}")
                                        break
                                break
                        # Did you see ...
                        else:
                            found = False
                            # List of supported time expressions
                            time_expressions = [
                                "today", "yesterday", "last week", "this week", "last month", "this month",
                                "this weekend", "last weekend"
                            ] + [f"in {month}" for month in [
                                "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"
                            ]]
                            for pattern in did_you_see_patterns:
                                match_see = re.search(pattern, query)
                                if match_see:
                                    label_time = match_see.group(1).strip().lower()
                                    # Regex for time expressions
                                    time_expr_regex = r"(today|yesterday|last week|this week|last month|this month|this weekend|last weekend|in (?:january|february|march|april|may|june|july|august|september|october|november|december))"
                                    time_expr_match = re.search(time_expr_regex, query)
                                    found_time = time_expr_match.group(0) if time_expr_match else None
                                    if found_time:
                                        # Remove the time expression from the label if present
                                        label = re.sub(time_expr_regex, '', label_time).strip()
                                        time_expr = found_time
                                    else:
                                        label = label_time
                                        time_expr = None
                                    print(f"[DEBUG] Did you see: label='{label}', time_expr='{time_expr}'")
                                    # Filter DataFrame by time expression if present
                                    df_filtered = df_all
                                    if time_expr:
                                        start, end = self.parse_time_expression(time_expr)
                                        if start is not None and end is not None:
                                            df_filtered = df_filtered[(df_filtered['timestamp'] >= start) & (df_filtered['timestamp'] <= end)]
                                            print(f"[DEBUG] Filtering logs from {start} to {end}")
                                        else:
                                            message = f"Sorry, I couldn't understand the time expression '{time_expr}'."
                                            found = True
                                            break
                                    matches = df_filtered[(df_filtered['label_1'].str.lower() == label) |
                                                         (df_filtered['label_2'].str.lower() == label) |
                                                         (df_filtered['label_3'].str.lower() == label)]
                                    if not matches.empty:
                                        first_time = matches.iloc[0]['timestamp']
                                        spoken_time = first_time.strftime("%I:%M %p on %B %d").lstrip("0")
                                        message = f"Yes, I saw {label} at {spoken_time}."
                                    else:
                                        message = f"No, I did not see {label}{' ' + time_expr if time_expr else ''}."
                                    found = True
                                    break
                            if not found:
                                # When did you last see ...
                                for pattern in last_seen_patterns:
                                    match_last = re.search(pattern, query)
                                    if match_last:
                                        label_time = match_last.group(1).strip().lower()
                                        time_expr_match = re.search(time_expr_regex, query)
                                        found_time = time_expr_match.group(0) if time_expr_match else None
                                        if found_time:
                                            label = re.sub(time_expr_regex, '', label_time).strip()
                                            time_expr = found_time
                                        else:
                                            label = label_time
                                            time_expr = None
                                        print(f"[DEBUG] Last seen: label='{label}', time_expr='{time_expr}'")
                                        df_filtered = df_all
                                        if time_expr:
                                            start, end = self.parse_time_expression(time_expr)
                                            if start is not None and end is not None:
                                                df_filtered = df_filtered[(df_filtered['timestamp'] >= start) & (df_filtered['timestamp'] <= end)]
                                                print(f"[DEBUG] Filtering logs from {start} to {end}")
                                            else:
                                                message = f"Sorry, I couldn't understand the time expression '{time_expr}'."
                                                break
                                        matches = df_filtered[(df_filtered['label_1'].str.lower() == label) |
                                                             (df_filtered['label_2'].str.lower() == label) |
                                                             (df_filtered['label_3'].str.lower() == label)]
                                        if not matches.empty:
                                            last_time = matches.iloc[-1]['timestamp']
                                            spoken_time = last_time.strftime("%I:%M %p on %B %d").lstrip("0")
                                            message = f"The last time I saw {label} was at {spoken_time}."
                                        else:
                                            message = f"I never saw {label}{' ' + time_expr if time_expr else ''}."
                                        break
                print("Responding:", message)
                # Send message to Home Assistant TTS
                try:
                    headers = {
                        "Authorization": f"Bearer {HOME_ASSISTANT_TOKEN}",
                        "Content-Type": "application/json",
                    }
                    json_data = {
                        "entity_id": "media_player.den_speaker",
                        "message": message,
                    }
                    requests.post(
                        "http://localhost:8123/api/services/tts/google_translate_say",
                        headers=headers,
                        json=json_data,
                    )
                except Exception as e:
                    print(f"[DEBUG] Could not send message to Home Assistant: {e}")
                tts = gTTS(message)
                tts.save("assets/audio/response.mp3")
                os.system("mpv assets/audio/response.mp3")
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
        # Generate and play intro
        intro_text = "Hello! I am your vision-aware assistant. I can see and detect objects in my view. Ask me what I see, or about past detections."
        tts = gTTS(text=intro_text, lang='en')
        tts.save("assets/audio/intro.mp3")
        os.system("mpv assets/audio/intro.mp3")
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