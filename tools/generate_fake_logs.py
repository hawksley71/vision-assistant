import csv
import os
from datetime import datetime, timedelta
import random
from dateutil.relativedelta import relativedelta
import sys
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.config.settings import PATHS, LOGGING_SETTINGS

# List of objects that can appear randomly during daytime hours
DAYTIME_OBJECTS = [
    "cat", "dog", "bird", "chipmunk", "bicycle", "car", "truck", "stroller",
    "prime van", "ups truck", "fedex truck", "person"
]

# Special objects with rules
SPECIAL_OBJECTS = [
    {
        "name": "school bus",
        "months": list(range(9, 13)) + list(range(1, 7)),  # September (9) to June (6)
        "weekdays": [0, 1, 2, 3, 4],  # Monday=0, ..., Friday=4
        "times": [("07:00", "07:20"), ("15:00", "15:30")],  # Two appearances per day
        "frequency": "every",
    },
    {
        "name": "mail truck",
        "months": list(range(1, 13)),  # All months
        "weekdays": [0, 1, 2, 3, 4],  # Monday=0, ..., Friday=4
        "times": [("14:00", "15:00")],  # Once per weekday, between 2pm and 3pm
        "frequency": "every",
    },
    {
        "name": "garbage truck",
        "months": list(range(1, 13)),
        "weekdays": [1],  # Tuesday
        "times": [("06:30", "07:00")],
        "frequency": "every",
    },
    {
        "name": "recycling truck",
        "months": list(range(1, 13)),
        "weekdays": [1],  # Tuesday
        "times": [("07:30", "08:30")],
        "frequency": "every_other_week",
    },
    {
        "name": "ice cream truck",
        "months": [6, 7, 8],
        "weekdays": list(range(0, 7)),
        "times": [("12:00", "16:00")],
        "frequency": "every",
    },
    {
        "name": "oil truck",
        "months": list(range(1, 13)),
        "weekdays": list(range(0, 7)),
        "times": [("09:00", "17:00")],
        "frequency": "first_day_of_month",
    },
    {
        "name": "newspaper",
        "months": list(range(1, 13)),
        "weekdays": [6],  # Sunday
        "times": [("04:00", "04:45")],
        "frequency": "every",
    },
    {
        "name": "skunk",
        "months": list(range(1, 13)),
        "weekdays": list(range(0, 7)),
        "times": [("00:00", "23:59")],
        "frequency": "once_per_week",
    },
    {
        "name": "racoon",
        "months": list(range(1, 13)),
        "weekdays": list(range(0, 7)),
        "times": [("22:00", "23:59"), ("00:00", "05:00")],
        "frequency": "twice_per_week",
    },
    {
        "name": "streetcleaner",
        "months": [4, 5, 6, 7, 8, 9, 10],
        "weekdays": [0, 1, 2, 3, 4],
        "times": [("07:00", "15:00")],
        "frequency": "once_per_month_first_week",
    },
    {
        "name": "snow plow",
        "months": [12, 1, 2],
        "weekdays": list(range(0, 7)),
        "times": [("00:00", "23:59")],
        "frequency": "twelve_per_winter",
    },
    {
        "name": "police car",
        "months": list(range(1, 13)),
        "weekdays": list(range(0, 7)),
        "times": [("00:00", "23:59")],
        "frequency": "three_per_month",
    },
    {
        "name": "fire truck",
        "months": list(range(1, 13)),
        "weekdays": list(range(0, 7)),
        "times": [("00:00", "23:59")],
        "frequency": "once_per_month",
    },
    {
        "name": "ambulance",
        "months": list(range(1, 13)),
        "weekdays": list(range(0, 7)),
        "times": [("00:00", "23:59")],
        "frequency": "once_per_two_months",
    },
]

# Number of logs to generate per day for random objects
LOGS_PER_DAY = 3

# Helper to generate a random daytime time
def random_daytime_time(date):
    hour = random.randint(6, 18)  # 6:00 to 18:00 (6am to 6:59pm)
    minute = random.randint(0, 59)
    return date.replace(hour=hour, minute=minute, second=0)

def random_time_in_window(date, start_str, end_str):
    start = datetime.strptime(start_str, "%H:%M").time()
    end = datetime.strptime(end_str, "%H:%M").time()
    start_minutes = start.hour * 60 + start.minute
    end_minutes = end.hour * 60 + end.minute
    minute = random.randint(start_minutes, end_minutes)
    hour, minute = divmod(minute, 60)
    return date.replace(hour=hour, minute=minute, second=0)

def generate_special_object_rows(date, week_number, month_days, winter_days, month, two_months_map):
    rows = []
    for obj in SPECIAL_OBJECTS:
        if date.month not in obj["months"] or date.weekday() not in obj["weekdays"]:
            continue
        freq = obj.get("frequency", "every")
        if freq == "every":
            for time_window in obj["times"]:
                ts = random_time_in_window(date, *time_window)
                count = random.randint(1, 3)
                conf = round(random.uniform(0.7, 1.0), 2)
                row = [ts.strftime("%Y-%m-%d %H:%M:%S"), obj["name"], count, conf, "", "", "", "", "", ""]
                rows.append(row)
        elif freq == "every_other_week":
            if week_number % 2 == 0:
                for time_window in obj["times"]:
                    ts = random_time_in_window(date, *time_window)
                    count = random.randint(1, 3)
                    conf = round(random.uniform(0.7, 1.0), 2)
                    row = [ts.strftime("%Y-%m-%d %H:%M:%S"), obj["name"], count, conf, "", "", "", "", "", ""]
                    rows.append(row)
        elif freq == "first_day_of_month":
            if date.day == 1:
                for time_window in obj["times"]:
                    ts = random_time_in_window(date, *time_window)
                    count = random.randint(1, 3)
                    conf = round(random.uniform(0.7, 1.0), 2)
                    row = [ts.strftime("%Y-%m-%d %H:%M:%S"), obj["name"], count, conf, "", "", "", "", "", ""]
                    rows.append(row)
        elif freq == "once_per_week":
            # Only generate on the randomly chosen day for this week
            if month_days[obj["name"]][week_number] == date.weekday():
                for time_window in obj["times"]:
                    ts = random_time_in_window(date, *time_window)
                    count = random.randint(1, 3)
                    conf = round(random.uniform(0.7, 1.0), 2)
                    row = [ts.strftime("%Y-%m-%d %H:%M:%S"), obj["name"], count, conf, "", "", "", "", "", ""]
                    rows.append(row)
        elif freq == "twice_per_week":
            # Only generate on the two randomly chosen days for this week
            if date.weekday() in month_days[obj["name"]][week_number]:
                for time_window in obj["times"]:
                    ts = random_time_in_window(date, *time_window)
                    count = random.randint(1, 3)
                    conf = round(random.uniform(0.7, 1.0), 2)
                    row = [ts.strftime("%Y-%m-%d %H:%M:%S"), obj["name"], count, conf, "", "", "", "", "", ""]
                    rows.append(row)
        elif freq == "once_per_month_first_week":
            # Only generate on the randomly chosen day in the first week of the month
            if date.day <= 7 and month_days[obj["name"]][month] == date.day:
                for time_window in obj["times"]:
                    ts = random_time_in_window(date, *time_window)
                    count = random.randint(1, 3)
                    conf = round(random.uniform(0.7, 1.0), 2)
                    row = [ts.strftime("%Y-%m-%d %H:%M:%S"), obj["name"], count, conf, "", "", "", "", "", ""]
                    rows.append(row)
        elif freq == "twelve_per_winter":
            # Only generate on the 12 randomly chosen days for the winter months
            if date in winter_days:
                for time_window in obj["times"]:
                    ts = random_time_in_window(date, *time_window)
                    count = random.randint(1, 3)
                    conf = round(random.uniform(0.7, 1.0), 2)
                    row = [ts.strftime("%Y-%m-%d %H:%M:%S"), obj["name"], count, conf, "", "", "", "", "", ""]
                    rows.append(row)
        elif freq == "three_per_month":
            # Only generate on the 3 randomly chosen days for this month
            if date.day in month_days[obj["name"]][month]:
                for time_window in obj["times"]:
                    ts = random_time_in_window(date, *time_window)
                    count = random.randint(1, 3)
                    conf = round(random.uniform(0.7, 1.0), 2)
                    row = [ts.strftime("%Y-%m-%d %H:%M:%S"), obj["name"], count, conf, "", "", "", "", "", ""]
                    rows.append(row)
        elif freq == "once_per_month":
            # Only generate on the randomly chosen day for this month
            if date.day == month_days[obj["name"]][month]:
                for time_window in obj["times"]:
                    ts = random_time_in_window(date, *time_window)
                    count = random.randint(1, 3)
                    conf = round(random.uniform(0.7, 1.0), 2)
                    row = [ts.strftime("%Y-%m-%d %H:%M:%S"), obj["name"], count, conf, "", "", "", "", "", ""]
                    rows.append(row)
        elif freq == "once_per_two_months":
            # Only generate on the randomly chosen day for this two-month period
            if (month, date.day) == two_months_map[obj["name"]]:
                for time_window in obj["times"]:
                    ts = random_time_in_window(date, *time_window)
                    count = random.randint(1, 3)
                    conf = round(random.uniform(0.7, 1.0), 2)
                    row = [ts.strftime("%Y-%m-%d %H:%M:%S"), obj["name"], count, conf, "", "", "", "", "", ""]
                    rows.append(row)
    return rows

def sanitize_filename(name):
    return re.sub(r'[^A-Za-z0-9]+', '_', name)

def generate_logs():
    # Set end_date to yesterday
    end_date = datetime.now().date() - timedelta(days=1)
    # Set start_date to one year before yesterday
    start_date = end_date - timedelta(days=365)
    # Convert to datetime objects at midnight
    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.min.time())
    current_date = start_date

    # Precompute random days for special frequencies
    month_days = {}
    winter_days = set()
    two_months_map = {}
    for obj in SPECIAL_OBJECTS:
        if obj["frequency"] in ["once_per_week", "twice_per_week"]:
            # For each week, pick 1 or 2 random weekdays
            month_days[obj["name"]] = {}
            d = start_date
            week = 0
            while d <= end_date:
                if obj["frequency"] == "once_per_week":
                    month_days[obj["name"]][week] = random.choice(obj["weekdays"])
                else:
                    month_days[obj["name"]][week] = random.sample(obj["weekdays"], 2)
                d += timedelta(days=7)
                week += 1
        elif obj["frequency"] == "once_per_month_first_week":
            # For each month, pick a random day in the first week
            month_days[obj["name"]] = {}
            for m in obj["months"]:
                month_days[obj["name"]][m] = random.randint(1, 7)
        elif obj["frequency"] == "twelve_per_winter":
            # For each winter, pick 12 random days
            for year in [2024, 2025]:
                for m in [12, 1, 2]:
                    for _ in range(12):
                        day = random.randint(1, 28)
                        winter_days.add(datetime(year, m, day))
        elif obj["frequency"] == "three_per_month":
            # For each month, pick 3 random days
            month_days[obj["name"]] = {}
            for m in obj["months"]:
                month_days[obj["name"]][m] = random.sample(range(1, 29), 3)
        elif obj["frequency"] == "once_per_month":
            # For each month, pick 1 random day
            month_days[obj["name"]] = {}
            for m in obj["months"]:
                month_days[obj["name"]][m] = random.randint(1, 28)
        elif obj["frequency"] == "once_per_two_months":
            # For each two-month period, pick a random day
            for y in [2024, 2025]:
                for m in range(1, 13, 2):
                    day = random.randint(1, 28)
                    two_months_map[obj["name"]] = (m, day)

    while current_date <= end_date:
        week_number = (current_date - start_date).days // 7
        month = current_date.month
        # Only use date for filename, and sanitize
        date_str = current_date.strftime('%Y-%m-%d')
        filename = os.path.join(PATHS['data']['raw'], f"detections_{sanitize_filename(date_str)}.csv")
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "label_1", "count_1", "avg_conf_1",
                "label_2", "count_2", "avg_conf_2",
                "label_3", "count_3", "avg_conf_3"
            ])
            # Special object rows
            special_rows = generate_special_object_rows(
                current_date, week_number, month_days, winter_days, month, two_months_map
            )
            for row in special_rows:
                writer.writerow(row)
            # Random daytime objects
            for _ in range(LOGS_PER_DAY):
                ts = random_time_in_window(current_date, "06:00", "19:00")
                num_objs = random.randint(1, 3)
                labels = random.sample(DAYTIME_OBJECTS, k=num_objs)
                row = [ts.strftime("%Y-%m-%d %H:%M:%S")]
                for label in labels:
                    count = random.randint(1, 5)
                    conf = round(random.uniform(0.5, 1.0), 2)
                    row.extend([label, count, conf])
                while len(row) < 10:
                    row.extend(["", "", ""])
                writer.writerow(row)
        current_date += timedelta(days=1)

    print(f"Fake logs generated in {PATHS['data']['raw']} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")

if __name__ == "__main__":
    generate_logs()
    # Remove or comment out the following block if you only want the rolling year logs:
    # month_start = datetime(2024, 4, 1)
    # for m in range(11):
    #     ...
    # print(f"Monthly logs generated for April 2024 to March 2025 in {PATHS['data']['raw']}/.") 