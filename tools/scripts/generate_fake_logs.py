import csv
import os
from datetime import datetime, timedelta
import random
from dateutil.relativedelta import relativedelta
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.config.settings import PATHS, LOGGING_SETTINGS

# Special vehicles (excluded from random objects)
SPECIAL_OBJECTS = ["school bus", "garbage truck", "mail truck", "icecream truck", "oil truck"]
OBJECTS = ["truck", "car", "person", "dog", "cat", "bicycle", "motorcycle", "traffic light", "stop sign"]

start_date = datetime(2024, 5, 9)
end_date = datetime(2025, 5, 9)
log_dir = PATHS['data']['raw']
os.makedirs(log_dir, exist_ok=True)

# Helper to generate a random time within a range
def random_time_on_day(day, hour_start, min_start, hour_end, min_end):
    hour = random.randint(hour_start, hour_end)
    if hour == hour_start:
        minute = random.randint(min_start, 59)
    elif hour == hour_end:
        minute = random.randint(0, min_end)
    else:
        minute = random.randint(0, 59)
    return day.replace(hour=hour, minute=minute, second=0)

for i in range((end_date - start_date).days + 1):
    day = start_date + timedelta(days=i)
    filename = os.path.join(log_dir, f"detections_{day.strftime(LOGGING_SETTINGS['log_format'])}.csv")
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "label_1", "count_1", "avg_conf_1",
            "label_2", "count_2", "avg_conf_2",
            "label_3", "count_3", "avg_conf_3"
        ])
        month = day.month
        weekday = day.weekday()  # Monday=0, Sunday=6
        # --- Special Vehicles ---
        # School bus: Weekdays, Sep-Jun
        if weekday < 5 and (month >= 9 or month <= 6):
            # Morning
            ts_morning = random_time_on_day(day, 7, 30, 7, 50)
            row = [ts_morning.strftime("%Y-%m-%d %H:%M:%S"), "school bus", random.randint(1, 3), round(random.uniform(0.5, 1.0), 2)]
            while len(row) < 10:
                row.extend(["", "", ""])
            writer.writerow(row)
            # Afternoon
            ts_afternoon = random_time_on_day(day, 15, 0, 15, 20)
            row = [ts_afternoon.strftime("%Y-%m-%d %H:%M:%S"), "school bus", random.randint(1, 3), round(random.uniform(0.5, 1.0), 2)]
            while len(row) < 10:
                row.extend(["", "", ""])
            writer.writerow(row)
        # Garbage truck: Tuesday mornings
        if weekday == 1:  # Tuesday
            ts_garbage = random_time_on_day(day, 6, 0, 6, 45)
            row = [ts_garbage.strftime("%Y-%m-%d %H:%M:%S"), "garbage truck", random.randint(1, 2), round(random.uniform(0.5, 1.0), 2)]
            while len(row) < 10:
                row.extend(["", "", ""])
            writer.writerow(row)
        # Mail truck: Weekdays, 12:00-1:00 p.m.
        if weekday < 5:
            ts_mail = random_time_on_day(day, 12, 0, 13, 0)
            row = [ts_mail.strftime("%Y-%m-%d %H:%M:%S"), "mail truck", random.randint(1, 2), round(random.uniform(0.5, 1.0), 2)]
            while len(row) < 10:
                row.extend(["", "", ""])
            writer.writerow(row)
        # Icecream truck: Saturdays, June-August, 2:00-4:00 p.m.
        if weekday == 5 and month in [6, 7, 8]:
            ts_icecream = random_time_on_day(day, 14, 0, 16, 0)
            row = [ts_icecream.strftime("%Y-%m-%d %H:%M:%S"), "icecream truck", random.randint(1, 2), round(random.uniform(0.5, 1.0), 2)]
            while len(row) < 10:
                row.extend(["", "", ""])
            writer.writerow(row)
        # Oil truck: First day of each month, 9:00-10:00 a.m.
        if day.day == 1:
            ts_oil = random_time_on_day(day, 9, 0, 10, 0)
            row = [ts_oil.strftime("%Y-%m-%d %H:%M:%S"), "oil truck", random.randint(1, 2), round(random.uniform(0.5, 1.0), 2)]
            while len(row) < 10:
                row.extend(["", "", ""])
            writer.writerow(row)
        # --- Random records (no special vehicles, not 12am-5am) ---
        for j in range(8):
            hour = random.randint(5, 23)  # 5am to 11pm
            minute = random.randint(0, 59)
            ts = day.replace(hour=hour, minute=minute, second=0)
            labels = random.sample(OBJECTS, k=random.randint(1, 3))
            row = [ts.strftime("%Y-%m-%d %H:%M:%S")]
            for label in labels:
                count = random.randint(1, 5)
                conf = round(random.uniform(0.5, 1.0), 2)
                row.extend([label, count, conf])
            while len(row) < 10:
                row.extend(["", "", ""])
            writer.writerow(row)
print(f"Fake logs generated in {log_dir}/ for {start_date.date()} to {end_date.date()}.")

# Monthly logs for the 15th of each month (no special vehicles in random records)
month_start = datetime(2024, 6, 1)
for m in range(12):
    month = month_start + relativedelta(months=m)
    log_day = month.replace(day=15)
    filename = os.path.join(log_dir, f"detections_{log_day.strftime(LOGGING_SETTINGS['log_format'])}.csv")
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "label_1", "count_1", "avg_conf_1",
            "label_2", "count_2", "avg_conf_2",
            "label_3", "count_3", "avg_conf_3"
        ])
        for j in range(10):
            hour = random.randint(5, 23)
            minute = random.randint(0, 59)
            ts = log_day.replace(hour=hour, minute=minute, second=0)
            labels = random.sample(OBJECTS, k=random.randint(1, 3))
            row = [ts.strftime("%Y-%m-%d %H:%M:%S")]
            for label in labels:
                count = random.randint(1, 5)
                conf = round(random.uniform(0.5, 1.0), 2)
                row.extend([label, count, conf])
            while len(row) < 10:
                row.extend(["", "", ""])
            writer.writerow(row)
print(f"Monthly logs generated for June 2024 to May 2025 in {log_dir}/.") 