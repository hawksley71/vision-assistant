import torch
import cv2
import time
import warnings

# Optional: Suppress deprecation warnings from YOLOv5 internals
warnings.filterwarnings("ignore", category=FutureWarning)

# Load YOLOv5 model from local clone
model = torch.hub.load('yolov5', 'yolov5s', source='local')

# Move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Open default webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

# Track last print time to throttle output
last_print_time = 0
throttle_interval = 1.0  # seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Inference
    results = model(frame)

    # Show annotated frame
    annotated_frame = results.render()[0]
    cv2.imshow('YOLOv5 Detection (press q to quit)', annotated_frame)

    # Throttle detection printing to once per second
    current_time = time.time()
    if current_time - last_print_time > throttle_interval:
        detected = set()
        for *box, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]
            detected.add(f"{label} ({conf:.2f})")

        if detected:
            print("Detected:", ", ".join(sorted(detected)))

        last_print_time = current_time

    # Exit on 'q' key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
