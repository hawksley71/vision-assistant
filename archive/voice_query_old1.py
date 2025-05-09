import speech_recognition as sr
from gtts import gTTS
import pandas as pd
import os
from datetime import datetime, timedelta

# Load most recent log
log_dir = "data"
log_files = sorted([f for f in os.listdir(log_dir) if f.endswith(".csv")])
if not log_files:
    print("No detection logs found.")
    exit()

latest_log = os.path.join(log_dir, log_files[-1])
df = pd.read_csv(latest_log)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Recognizer setup
r = sr.Recognizer()
mic = sr.Microphone()

print("Say 'what did you see' to get a summary. Say 'exit' to quit.")

while True:
    try:
        with mic as source:
            r.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = r.listen(source, timeout=5)

        query = r.recognize_google(audio).lower()
        print("You said:", query)

        if "what did you see" in query:
            # Filter last hour of logged detections
            latest_time = df['timestamp'].max()
            cutoff = latest_time - timedelta(hours=1)
            df_recent = df[df['timestamp'] >= cutoff]

            if df_recent.empty:
                message = "In the past hour of detections, I haven't seen anything."
            else:
                top_labels = df_recent['label'].value_counts().head(10).index.tolist()
                if len(top_labels) == 1:
                    message = f"In the past hour of detections, I saw {top_labels[0]}."
                else:
                    joined = ", ".join(top_labels[:-1]) + ", and " + top_labels[-1]
                    message = f"In the past hour of detections, I saw: {joined}."

            print("Speaking:", message)
            tts = gTTS(message)
            tts.save("response.mp3")
            os.system("mpv response.mp3")

        elif any(word in query for word in ["exit", "quit", "stop", "cancel"]):
            print("Exiting.")
            break

    except sr.WaitTimeoutError:
        print("No speech detected. Try again...")
    except sr.UnknownValueError:
        print("Sorry, could not understand.")
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
        break
