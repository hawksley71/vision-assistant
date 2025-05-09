import csv
import os
from datetime import datetime, timedelta
import random
from dateutil.relativedelta import relativedelta

OBJECTS = ["bus", "truck", "car", "person", "dog", "cat", "bicycle", "motorcycle", "traffic light", "stop sign"]

start_date = datetime(2025, 4, 24)
end_date = datetime(2025, 5, 7)
log_dir = "data"
os.makedirs(log_dir, exist_ok=True)

for i in range((end_date - start_date).days + 1):
    day = start_date + timedelta(days=i)
    filename = os.path.join(log_dir, f"detections_{day.strftime('%Y%m%d')}.csv")
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "label_1", "count_1", "avg_conf_1",
            "label_2", "count_2", "avg_conf_2",
            "label_3", "count_3", "avg_conf_3"
        ])
        for j in range(10):
            ts = day + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
            labels = random.sample(OBJECTS, k=random.randint(1, 3))
            row = [ts.strftime("%Y-%m-%d %H:%M:%S")]
            for label in labels:
                count = random.randint(1, 5)
                conf = round(random.uniform(0.5, 1.0), 2)
                row.extend([label, count, conf])
            # Pad row if fewer than 3 labels
            while len(row) < 10:
                row.extend(["", "", ""])
            writer.writerow(row)
print(f"Fake logs generated in {log_dir}/ for {start_date.date()} to {end_date.date()}.")

# Generate one log for a single day in each of the last 11 months ending March 2025
# Last 11 months ending March 2025 (April 2024 to March 2025)
month_start = datetime(2024, 4, 1)
for m in range(11):
    month = month_start + relativedelta(months=m)
    # Pick the 15th of each month for the log
    log_day = month.replace(day=15)
    filename = os.path.join(log_dir, f"detections_{log_day.strftime('%Y%m%d')}.csv")
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "label_1", "count_1", "avg_conf_1",
            "label_2", "count_2", "avg_conf_2",
            "label_3", "count_3", "avg_conf_3"
        ])
        for j in range(10):
            ts = log_day + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
            labels = random.sample(OBJECTS, k=random.randint(1, 3))
            row = [ts.strftime("%Y-%m-%d %H:%M:%S")]
            for label in labels:
                count = random.randint(1, 5)
                conf = round(random.uniform(0.5, 1.0), 2)
                row.extend([label, count, conf])
            while len(row) < 10:
                row.extend(["", "", ""])
            writer.writerow(row)
print(f"Monthly logs generated for April 2024 to March 2025 in {log_dir}/.") 