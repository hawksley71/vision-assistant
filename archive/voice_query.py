import speech_recognition as sr
from gtts import gTTS
import pandas as pd
import os
from datetime import datetime
import re
import cv2
from models.yolov8_model import YOLOv8Model

def load_all_logs(log_dir="data"):
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
        return pd.DataFrame(columns=["timestamp", "label", "confidence"])

    return pd.concat(all_dfs, ignore_index=True).sort_values("timestamp")

# Initialize recognizer and mic
r = sr.Recognizer()
mic = sr.Microphone()

# Initialize YOLOv8 model for live queries
model = YOLOv8Model()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Ask: 'Did you see a [thing]?', 'When did you last see [thing]?', 'What was the first thing you saw?', or 'What are you seeing right now?'. Say 'exit' to quit.")

while True:
    try:
        with mic as source:
            r.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = r.listen(source, timeout=5)

        query = r.recognize_google(audio).lower()
        print("You said:", query)

        if any(word in query for word in ["exit", "quit", "stop", "cancel"]):
            print("Exiting.")
            break

        message = "Sorry, I didn't understand that."

        # Live detection query (run YOLOv8 on a live frame)
        trigger_phrases = [
            "what are you seeing",
            "what do you see",
            "what do you see now",
            "what are you detecting",
            "what do you detect",
            "what is in front of you",
            "what's in front of you",
            "what's there",
            "what do you see right now",
            "what else",
            "what about now"
        ]
        if any(phrase in query for phrase in trigger_phrases):
            ret, frame = cap.read()
            if not ret:
                message = "Camera error: could not read frame."
            else:
                detections = model.detect(frame)
                if detections:
                    detection_text = ", ".join([f"{det['class_name']} ({det['confidence']:.2f})" for det in detections])
                    message = f"Right now, I am seeing: {detection_text}."
                else:
                    message = "I'm not seeing anything right now."
        else:
            df_all = load_all_logs()
            if df_all.empty:
                message = "I haven't seen anything yet."
            else:
                # "Did you see a ..." pattern
                match_see = re.search(r"did you see (?:a |an |any )?([\w\s]+)\??", query)
                if match_see:
                    label = match_see.group(1).strip().lower()
                    matches = df_all[df_all['label'].str.lower() == label]
                    if not matches.empty:
                        first_time = matches.iloc[0]['timestamp']
                        spoken_time = first_time.strftime("%I:%M %p on %B %d").lstrip("0")
                        message = f"Yes, I saw {label} at {spoken_time}."
                    else:
                        message = f"No, I did not see {label}."

                # "When did you last see..." pattern
                match_last = re.search(r"when did you (?:last )?see (?:a |an )?([\w\s]+)\??", query)
                if match_last:
                    label = match_last.group(1).strip().lower()
                    matches = df_all[df_all['label'].str.lower() == label]
                    if not matches.empty:
                        last_time = matches.iloc[-1]['timestamp']
                        spoken_time = last_time.strftime("%I:%M %p on %B %d").lstrip("0")
                        message = f"The last time I saw {label} was at {spoken_time}."
                    else:
                        message = f"I never saw {label}."

                # "What was the first thing you saw?" pattern
                if "what was the first thing you saw" in query:
                    first_row = df_all.iloc[0]
                    label = first_row['label']
                    first_time = first_row['timestamp']
                    spoken_time = first_time.strftime("%I:%M %p on %B %d").lstrip("0")
                    message = f"The first thing I saw was {label} at {spoken_time}."

        print("Responding:", message)
        tts = gTTS(message)
        tts.save("response.mp3")
        os.system("mpv response.mp3")

    except sr.WaitTimeoutError:
        print("No speech detected. Try again...")
    except sr.UnknownValueError:
        print("Sorry, could not understand.")
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
        break

cap.release()
