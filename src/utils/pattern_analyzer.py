import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
from pathlib import Path

class PatternAnalyzer:
    def __init__(self, min_detections: int = 3, time_threshold: float = 0.3, day_threshold: float = 0.4):
        """
        Initialize the pattern analyzer.
        
        Args:
            min_detections: Minimum number of detections required to consider patterns
            time_threshold: Minimum percentage of detections at a time to consider it a pattern
            day_threshold: Minimum percentage of detections on a day to consider it a pattern
        """
        self.min_detections = min_detections
        self.time_threshold = time_threshold
        self.day_threshold = day_threshold
        
    def analyze_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze patterns in the detection data.
        
        Args:
            df: DataFrame containing detection data
            
        Returns:
            List of dictionaries containing pattern information for each object
        """
        if df.empty:
            return []
            
        # Convert timestamps to datetime if needed
        if isinstance(df['timestamp'].iloc[0], str):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Get unique objects
        objects = set()
        for col in ['label_1', 'label_2', 'label_3']:
            objects.update(df[col].dropna().unique())
            
        patterns = []
        for obj in objects:
            obj_patterns = self._analyze_object_patterns(df, obj)
            if obj_patterns:
                patterns.append(obj_patterns)
                
        return patterns
        
    def _analyze_object_patterns(self, df: pd.DataFrame, obj: str) -> Dict[str, Any]:
        """
        Analyze patterns for a specific object.
        
        Args:
            df: DataFrame containing detection data
            obj: Object name to analyze
            
        Returns:
            Dictionary containing pattern information for the object
        """
        # Find all detections of this object
        matches = df[
            (df['label_1'].fillna('').str.lower() == obj.lower()) |
            (df['label_2'].fillna('').str.lower() == obj.lower()) |
            (df['label_3'].fillna('').str.lower() == obj.lower())
        ]
        
        if len(matches) < self.min_detections:
            return None
            
        # Extract time components
        matches = matches.copy()
        matches['hour'] = matches['timestamp'].dt.hour
        matches['day'] = matches['timestamp'].dt.day_name()
        matches['is_weekday'] = matches['timestamp'].dt.weekday < 5
        matches['is_weekend'] = matches['timestamp'].dt.weekday >= 5
        matches['month'] = matches['timestamp'].dt.month
        matches['season'] = matches['timestamp'].dt.month % 12 // 3 + 1
        
        # Initialize pattern list
        strong_patterns = []
        
        # Analyze time patterns
        time_patterns = self._analyze_time_patterns(matches)
        if time_patterns:
            strong_patterns.extend(time_patterns)
            
        # Analyze day patterns
        day_patterns = self._analyze_day_patterns(matches)
        if day_patterns:
            strong_patterns.extend(day_patterns)
            
        # Analyze seasonal patterns
        seasonal_patterns = self._analyze_seasonal_patterns(matches)
        if seasonal_patterns:
            strong_patterns.extend(seasonal_patterns)
            
        # Analyze frequency patterns
        frequency_patterns = self._analyze_frequency_patterns(matches)
        if frequency_patterns:
            strong_patterns.extend(frequency_patterns)
            
        return {
            'object': obj,
            'total_detections': len(matches),
            'strong_patterns': strong_patterns,
            'first_seen': matches['timestamp'].min().isoformat(),
            'last_seen': matches['timestamp'].max().isoformat(),
            'avg_confidence': matches[['avg_conf_1', 'avg_conf_2', 'avg_conf_3']].mean().mean()
        }
        
    def _analyze_time_patterns(self, matches: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze time-based patterns."""
        patterns = []
        
        # Hour-based patterns
        hour_counts = matches['hour'].value_counts()
        if len(hour_counts) > 0:
            most_common_hour = hour_counts.index[0]
            proportion = hour_counts.iloc[0] / len(matches)
            if proportion >= self.time_threshold:
                patterns.append({
                    'type': 'time',
                    'description': f'around {most_common_hour:02d}:00'
                })
        # Time of day patterns
        morning = len(matches[matches['hour'].between(5, 11)])
        afternoon = len(matches[matches['hour'].between(12, 17)])
        evening = len(matches[matches['hour'].between(18, 23)])
        night = len(matches[matches['hour'].between(0, 4)])
        time_of_day = max([(morning, 'morning'), (afternoon, 'afternoon'),
                          (evening, 'evening'), (night, 'night')])
        if (time_of_day[0] / len(matches)) >= self.time_threshold:
            patterns.append({
                'type': 'time_of_day',
                'description': f'in the {time_of_day[1]}'
            })
        return patterns
        
    def _analyze_day_patterns(self, matches: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze day-based patterns."""
        patterns = []
        # Day of week patterns
        day_counts = matches['day'].value_counts()
        detected_days = set(matches['timestamp'].dt.dayofweek)
        if len(day_counts) > 0:
            most_common_day = day_counts.index[0]
            if day_counts.iloc[0] >= len(matches) * self.day_threshold:
                if detected_days == set(range(7)):
                    patterns.append({
                        'type': 'day',
                        'description': 'every day'
                    })
                elif all(matches['is_weekday']):
                    patterns.append({
                        'type': 'day',
                        'description': 'on weekdays'
                    })
                elif all(matches['is_weekend']):
                    patterns.append({
                        'type': 'day',
                        'description': 'on weekends'
                    })
                else:
                    patterns.append({
                        'type': 'day',
                        'description': f'on {most_common_day}s'
                    })
        return patterns
        
    def _analyze_seasonal_patterns(self, matches: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze seasonal patterns."""
        patterns = []
        
        # Season-based patterns
        season_counts = matches['season'].value_counts()
        if len(season_counts) > 0:
            most_common_season = season_counts.index[0]
            if season_counts.iloc[0] >= len(matches) * self.time_threshold:
                season_names = {1: 'winter', 2: 'spring', 3: 'summer', 4: 'fall'}
                patterns.append({
                    'type': 'season',
                    'description': f'in {season_names[most_common_season]}'
                })
                
        return patterns
        
    def _analyze_frequency_patterns(self, matches: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze frequency patterns."""
        patterns = []
        
        # Calculate time between detections
        matches = matches.sort_values('timestamp')
        time_diffs = matches['timestamp'].diff().dt.total_seconds() / 3600  # hours
        
        if len(time_diffs) > 1:
            avg_frequency = time_diffs.mean()
            if avg_frequency <= 24:  # Daily or more frequent
                patterns.append({
                    'type': 'frequency',
                    'description': 'daily'
                })
            elif avg_frequency <= 168:  # Weekly or more frequent
                patterns.append({
                    'type': 'frequency',
                    'description': 'weekly'
                })
            elif avg_frequency <= 720:  # Monthly or more frequent
                patterns.append({
                    'type': 'frequency',
                    'description': 'monthly'
                })
                
        return patterns

def generate_pattern_summary(df: pd.DataFrame = None) -> None:
    """
    Generate a summary of detection patterns and save it to a file.
    
    Args:
        df: Optional DataFrame containing detection data. If None, loads from combined logs.
    """
    if df is None:
        # Load combined logs
        combined_logs_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'combined_logs.csv'
        if not combined_logs_path.exists():
            print(f"Error: Combined logs file not found at {combined_logs_path}")
            return
        df = pd.read_csv(combined_logs_path)
        
    # Initialize analyzer and generate patterns
    analyzer = PatternAnalyzer()
    patterns = analyzer.analyze_patterns(df)
    
    # Save patterns to file
    output_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'pattern_summary.json'
    with open(output_path, 'w') as f:
        json.dump(patterns, f, indent=2)
    print(f"Generated pattern summary with {len(patterns)} objects")

if __name__ == "__main__":
    generate_pattern_summary() 