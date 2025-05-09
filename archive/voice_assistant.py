import pandas as pd
from gtts import gTTS
import os
from datetime import datetime, timedelta

# Load latest log file
log_dir = "data"
log_files = sorted([f for f in os.listdir(log_dir) if f.endswith(".csv")])
if not log_files:
    print("No detection logs found.")
    exit()

latest_log = os.path.join(log_dir, log_files[-1])
df = pd.read_csv(latest_log)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter last hour of data (based on timestamp column)
latest_time = df['timestamp'].max()
cutoff = latest_time - timedelta(hours=1)
df_recent = df[df['timestamp'] >= cutoff]

# Compose spoken summary
if df_recent.empty:
    message = "In the past hour of detections, I haven't seen anything."
else:
    top_labels = df_recent['label'].value_counts().head(10).index.tolist()
    if len(top_labels) == 1:
        message = f"In the past hour of detections, I saw {top_labels[0]}."
    else:
        joined = ", ".join(top_labels[:-1]) + ", and " + top_labels[-1]
        message = f"In the past hour of detections, I saw: {joined}."

# Print and speak it using gTTS
print("Speaking:", message)
tts = gTTS(text=message)
tts.save("summary.mp3")
os.system("mpv summary.mp3")  # or replace with "vlc summary.mp3"
