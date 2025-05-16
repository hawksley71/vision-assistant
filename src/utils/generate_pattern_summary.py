import os
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

def load_combined_logs():
    """Load the combined logs from the processed data directory."""
    logs_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'combined_logs.csv')
    if not os.path.exists(logs_path):
        print(f"Error: Combined logs file not found at {logs_path}")
        return None
    
    try:
        df = pd.read_csv(logs_path)
        if df.empty:
            print("Error: Combined logs file is empty")
            return None
        return df
    except Exception as e:
        print(f"Error loading combined logs: {e}")
        return None

def generate_pattern_summary():
    """Generate a summary of detection patterns for each object."""
    # Load combined logs
    df = load_combined_logs()
    if df is None:
        return
    
    # Convert timestamps to datetime if they're strings
    if isinstance(df['timestamp'].iloc[0], str):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Get unique objects
    objects = set()
    for col in ['label_1', 'label_2', 'label_3']:
        objects.update(df[col].dropna().unique())
    
    # Initialize summary data
    summary_data = []
    
    for obj in objects:
        # Find all detections of this object
        matches = df[
            (df['label_1'].fillna('').str.lower() == obj.lower()) |
            (df['label_2'].fillna('').str.lower() == obj.lower()) |
            (df['label_3'].fillna('').str.lower() == obj.lower())
        ]
        
        if len(matches) < 3:  # Skip objects with too few detections
            continue
        
        # Extract time components
        matches = matches.copy()
        matches['hour'] = matches['timestamp'].dt.hour
        matches['day'] = matches['timestamp'].dt.day_name()
        matches['is_weekday'] = matches['timestamp'].dt.weekday < 5
        matches['is_weekend'] = matches['timestamp'].dt.weekday >= 5
        
        # Initialize pattern list
        strong_patterns = []
        
        # Check for time-based patterns
        hour_counts = matches['hour'].value_counts()
        if len(hour_counts) > 0:
            most_common_hour = hour_counts.index[0]
            if hour_counts.iloc[0] >= len(matches) * 0.3:  # At least 30% of detections at this hour
                strong_patterns.append({
                    'type': 'time',
                    'description': f'around {most_common_hour}:00'
                })
        
        # Check for day-based patterns
        day_counts = matches['day'].value_counts()
        if len(day_counts) > 0:
            most_common_day = day_counts.index[0]
            if day_counts.iloc[0] >= len(matches) * 0.4:  # At least 40% of detections on this day
                if len(day_counts) == 7:
                    strong_patterns.append({
                        'type': 'day',
                        'description': 'every day'
                    })
                elif most_common_day in ['Saturday', 'Sunday'] and all(matches['is_weekend']):
                    strong_patterns.append({
                        'type': 'day',
                        'description': 'on weekends'
                    })
                elif most_common_day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] and all(matches['is_weekday']):
                    strong_patterns.append({
                        'type': 'day',
                        'description': 'on weekdays'
                    })
                else:
                    strong_patterns.append({
                        'type': 'day',
                        'description': f'on {most_common_day}s'
                    })
        
        # Add to summary data
        summary_data.append({
            'object': obj,
            'total_detections': len(matches),
            'strong_patterns': strong_patterns
        })
    
    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    output_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'pattern_summary.csv')
    summary_df.to_csv(output_path, index=False)
    print(f"Generated pattern summary with {len(summary_df)} objects")

if __name__ == "__main__":
    generate_pattern_summary() 