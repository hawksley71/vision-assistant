import torch
import cv2
import time
import csv
import os
from datetime import datetime

# Load YOLOv5 model from local clone
model = torch.hub.load('yolov5', 'yolov5s', source='local')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Create output log folder
log_dir = "data"
os.makedirs(log_dir, exist_ok=True)

# Create timestamped log file
log_filename = os.path.join(log_dir, f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
with open(log_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'label', 'confidence'])

# Track recently seen objects
recent_detections = {}  # {label: last_seen_time}
memory_window = 10  # seconds

# Start video stream
cap = cv2.VideoCapture(0)
print("Logging detections to:", log_filename)
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run inference
    results = model(frame)
    annotated_frame = results.render()[0]
    cv2.imshow('YOLOv5 Object Logger (q to quit)', annotated_frame)

    # Get current time
    now = time.time()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Track new or reappeared objects
    seen_now = set()
    for *box, conf, cls in results.xyxy[0]:
        label = model.names[int(cls)]
        label_key = f"{label}"
        seen_now.add(label_key)

        # Check if object is new or reappeared after memory window
        last_seen = recent_detections.get(label_key, 0)
        if now - last_seen > memory_window:
            print(f"{timestamp} - Logging: {label_key} ({conf:.2f})")
            with open(log_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, label_key, f"{conf:.2f}"])
            recent_detections[label_key] = now  # update last seen

    # Save current detections to live_now.txt
    try:
        with open("data/live_now.txt", "w") as f:
            f.write(", ".join(sorted(seen_now)))
    except Exception as e:
        print(f"Warning: Could not write live_now.txt — {e}")

    # Exit on 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
