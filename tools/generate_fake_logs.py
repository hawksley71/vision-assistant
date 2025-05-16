#!/usr/bin/env python3
"""
Generate fake detection logs for testing the vision assistant.
This script creates realistic detection logs with patterns and variations.
"""

import os
import random
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Common objects and their variations
OBJECTS = {
    'person': ['person', 'man', 'woman', 'child', 'boy', 'girl'],
    'vehicle': ['car', 'truck', 'bus', 'school bus', 'mail truck', 'van'],
    'animal': ['dog', 'cat', 'bird', 'squirrel', 'raccoon', 'rabbit'],
    'furniture': ['chair', 'table', 'desk', 'couch', 'bed'],
    'electronics': ['laptop', 'phone', 'tv', 'monitor', 'keyboard'],
    'food': ['apple', 'banana', 'sandwich', 'pizza', 'coffee'],
    'clothing': ['shirt', 'pants', 'shoes', 'hat', 'jacket'],
    'container': ['bottle', 'cup', 'bag', 'box', 'backpack']
}

def generate_timestamps(start_date, end_date, num_detections):
    """Generate realistic timestamps with patterns."""
    timestamps = []
    current = start_date
    
    # Generate timestamps with some patterns
    while len(timestamps) < num_detections:
        # Add some randomness to the time
        time_delta = timedelta(
            seconds=random.randint(0, 3600),  # Random time within the hour
            minutes=random.randint(0, 59),
            hours=random.randint(0, 23)
        )
        current += time_delta
        
        # Skip if we've gone past the end date
        if current > end_date:
            break
            
        timestamps.append(current)
    
    return sorted(timestamps)

def generate_detection_row(timestamp, objects):
    """Generate a single row of detection data."""
    # Select 3 random objects
    selected_objects = random.sample(list(objects.keys()), min(3, len(objects)))
    
    # Generate counts and confidences
    counts = [random.randint(1, 5) for _ in range(3)]
    confidences = [round(random.uniform(0.5, 0.95), 3) for _ in range(3)]
    
    # Create the row
    row = {
        'timestamp': timestamp,
        'label_1': random.choice(objects[selected_objects[0]]),
        'count_1': counts[0],
        'avg_conf_1': confidences[0]
    }
    
    if len(selected_objects) > 1:
        row.update({
            'label_2': random.choice(objects[selected_objects[1]]),
            'count_2': counts[1],
            'avg_conf_2': confidences[1]
        })
    else:
        row.update({'label_2': '', 'count_2': 0, 'avg_conf_2': 0.0})
        
    if len(selected_objects) > 2:
        row.update({
            'label_3': random.choice(objects[selected_objects[2]]),
            'count_3': counts[2],
            'avg_conf_3': confidences[2]
        })
    else:
        row.update({'label_3': '', 'count_3': 0, 'avg_conf_3': 0.0})
    
    return row

def generate_logs(days=7, detections_per_day=100):
    """Generate detection logs for the specified number of days."""
    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Generate logs for each day
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    current_date = start_date
    while current_date <= end_date:
        # Generate timestamps for this day
        day_start = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        timestamps = generate_timestamps(day_start, day_end, detections_per_day)
        
        # Generate detection rows
        rows = [generate_detection_row(ts, OBJECTS) for ts in timestamps]
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Save to CSV
        filename = f"data/raw/detections_{current_date.strftime('%Y_%m_%d')}.csv"
        df.to_csv(filename, index=False)
        print(f"Generated {len(df)} detections for {current_date.strftime('%Y-%m-%d')}")
        
        current_date += timedelta(days=1)

def main():
    """Main function to generate test data."""
    print("Generating test detection logs...")
    generate_logs()
    print("Done! Test data has been generated in data/raw/")

if __name__ == "__main__":
    main() 