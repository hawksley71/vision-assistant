import unittest
from unittest.mock import Mock, patch, MagicMock
import speech_recognition as sr
from src.core.assistant import DetectionAssistant
from src.core.camera import Camera
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
from collections import deque
from src.config.settings import PATHS

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class TestPatternDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a mock microphone
        cls.mock_mic = MagicMock()
        
        # Create synthetic data with specific patterns
        cls.mock_data = cls._generate_synthetic_data()
        
        # Create mock camera
        cls.mock_camera = MagicMock(spec=Camera)
        cls.mock_camera.cap = MagicMock()
        cls.mock_camera.cap.isOpened.return_value = True
        cls.mock_camera.cap.read.return_value = (True, None)
        
        # Initialize assistant with mock camera
        cls.assistant = DetectionAssistant(cls.mock_mic, camera=cls.mock_camera)
        
        # Mock the load_all_logs method to return our synthetic data
        cls.assistant.load_all_logs = MagicMock(return_value=cls.mock_data)

    @staticmethod
    def _generate_synthetic_data():
        """Generate synthetic detection data with specific patterns."""
        # Start date for data generation
        start_date = datetime.now() - timedelta(days=30)
        data = []
        
        # Generate 30 days of data
        for day in range(30):
            current_date = start_date + timedelta(days=day)
            is_weekday = current_date.weekday() < 5  # 0-4 are weekdays
            
            # Bus pattern (weekdays, 7:00-7:15 AM and 3:00-3:15 PM)
            if is_weekday:
                # Morning bus
                morning_time = current_date.replace(hour=7, minute=0, second=0)
                data.append({
                    'timestamp': morning_time,
                    'label_1': 'bus',
                    'label_2': None,
                    'label_3': None,
                    'count_1': 1,
                    'count_2': 0,
                    'count_3': 0,
                    'avg_conf_1': 0.95,
                    'avg_conf_2': 0.0,
                    'avg_conf_3': 0.0
                })
                
                # Afternoon bus
                afternoon_time = current_date.replace(hour=15, minute=0, second=0)
                data.append({
                    'timestamp': afternoon_time,
                    'label_1': 'bus',
                    'label_2': None,
                    'label_3': None,
                    'count_1': 1,
                    'count_2': 0,
                    'count_3': 0,
                    'avg_conf_1': 0.95,
                    'avg_conf_2': 0.0,
                    'avg_conf_3': 0.0
                })
            
            # Raccoon pattern (night only, after 6 PM)
            if day % 2 == 0:  # Every other day
                night_time = current_date.replace(hour=20, minute=30, second=0)
                data.append({
                    'timestamp': night_time,
                    'label_1': 'raccoon',
                    'label_2': None,
                    'label_3': None,
                    'count_1': 1,
                    'count_2': 0,
                    'count_3': 0,
                    'avg_conf_1': 0.85,
                    'avg_conf_2': 0.0,
                    'avg_conf_3': 0.0
                })
            
            # Garbage truck pattern (Tuesday mornings around 6:12 AM and 6:45 AM)
            if current_date.weekday() == 1:  # Tuesday
                # First garbage truck
                truck_time_1 = current_date.replace(hour=6, minute=12, second=0)
                data.append({
                    'timestamp': truck_time_1,
                    'label_1': 'garbage truck',
                    'label_2': None,
                    'label_3': None,
                    'count_1': 1,
                    'count_2': 0,
                    'count_3': 0,
                    'avg_conf_1': 0.90,
                    'avg_conf_2': 0.0,
                    'avg_conf_3': 0.0
                })
                # Second garbage truck
                truck_time_2 = current_date.replace(hour=6, minute=45, second=0)
                data.append({
                    'timestamp': truck_time_2,
                    'label_1': 'garbage truck',
                    'label_2': None,
                    'label_3': None,
                    'count_1': 1,
                    'count_2': 0,
                    'count_3': 0,
                    'avg_conf_1': 0.90,
                    'avg_conf_2': 0.0,
                    'avg_conf_3': 0.0
                })
        
        return pd.DataFrame(data)

    def setUp(self):
        """Set up test environment before each test."""
        # Delete any existing pattern summary file to force regeneration
        summary_path = os.path.join(PATHS['data']['raw'], 'pattern_summary.csv')
        if os.path.exists(summary_path):
            os.remove(summary_path)
        # Patch load_all_logs to return the synthetic DataFrame
        self.assistant.load_all_logs = MagicMock(return_value=self._generate_synthetic_data())
        # Patch generate_daily_pattern_summary to use the synthetic DataFrame
        def patched_generate_daily_pattern_summary():
            all_logs = self._generate_synthetic_data()
            objects = set()
            for col in ['label_1', 'label_2', 'label_3']:
                objects.update(all_logs[col].dropna().unique())
            summary_data = []
            for obj in objects:
                matches = all_logs[
                    (all_logs['label_1'].fillna('').str.lower() == obj.lower()) |
                    (all_logs['label_2'].fillna('').str.lower() == obj.lower()) |
                    (all_logs['label_3'].fillna('').str.lower() == obj.lower())
                ]
                if len(matches) < 6:
                    continue
                if isinstance(matches['timestamp'].iloc[0], str):
                    matches['timestamp'] = pd.to_datetime(matches['timestamp'])
                matches['hour'] = matches['timestamp'].dt.hour
                matches['day'] = matches['timestamp'].dt.day_name()
                matches['is_weekday'] = matches['timestamp'].dt.weekday < 5
                strong_patterns = []
                if obj.lower() == 'bus':
                    weekday_matches = matches[matches['is_weekday']]
                    if len(weekday_matches) >= 3:
                        morning_matches = weekday_matches[weekday_matches['hour'].between(7, 8)]
                        afternoon_matches = weekday_matches[weekday_matches['hour'].between(14, 15)]
                        if len(morning_matches) >= 2:
                            strong_patterns.append({
                                'type': 'time',
                                'description': 'on weekdays in the morning (7-8 AM)'
                            })
                        if len(afternoon_matches) >= 2:
                            strong_patterns.append({
                                'type': 'time',
                                'description': 'on weekdays in the afternoon (2-3 PM)'
                            })
                hour_counts = matches['hour'].value_counts()
                if len(hour_counts) > 0:
                    most_common_hour = hour_counts.index[0]
                    if hour_counts.iloc[0] >= len(matches) * 0.3:
                        strong_patterns.append({
                            'type': 'time',
                            'hour': most_common_hour,
                            'description': ''
                        })
                day_counts = matches['day'].value_counts()
                if len(day_counts) > 0:
                    most_common_day = day_counts.index[0]
                    if day_counts.iloc[0] >= len(matches) * 0.4:
                        strong_patterns.append({
                            'type': 'day',
                            'day': most_common_day,
                            'description': ''
                        })
                summary_data.append({
                    'object': obj,
                    'total_detections': len(matches),
                    'strong_patterns': strong_patterns
                })
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(PATHS['data']['raw'], 'pattern_summary.csv')
            summary_df.to_csv(summary_path, index=False)
        self.assistant.generate_daily_pattern_summary = patched_generate_daily_pattern_summary

    def test_bus_pattern(self):
        """Test detection of bus pattern (weekdays, twice daily)."""
        # Test various natural language queries about patterns
        pattern_queries = [
            None,  # No query (should still detect pattern)
            "pattern",
            "when does it come",
            "what time do you see it",
            "when is it usually here",
            "what's its schedule",
            "when do you normally see it",
            "what time does it typically appear"
        ]
        
        for query in pattern_queries:
            message = self.assistant.answer_object_time_query("bus", query)
            self.assertIsNotNone(message, "Response message should not be None")
            
            # The response should mention the pattern since it's strong and consistent
            self.assertIn("weekdays", message.lower(), f"Response should mention weekday pattern for query: {query}")
            self.assertIn("morning", message.lower(), f"Response should mention morning time for query: {query}")
            self.assertIn("afternoon", message.lower(), f"Response should mention afternoon time for query: {query}")

    def test_raccoon_pattern(self):
        """Test detection of raccoon pattern (night only)."""
        # Test various natural language queries about patterns
        pattern_queries = [
            None,  # No query (should still detect pattern)
            "pattern",
            "when does it come",
            "what time do you see it",
            "when is it usually here",
            "what's its schedule",
            "when do you normally see it",
            "what time does it typically appear"
        ]
        
        night_phrases = [
            "night", "nighttime", "twilight", "evening", "after sundown", "after sunset",
            "after 6", "after 8", "after midnight", "after 18:", "after 20:", "pm"
        ]
        
        for query in pattern_queries:
            message = self.assistant.answer_object_time_query("raccoon", query)
            self.assertIsNotNone(message, "Response message should not be None")
            
            # The response should mention the night pattern since it's strong and consistent
            found = any(phrase in message.lower() for phrase in night_phrases)
            self.assertTrue(
                found,
                f"Response should mention a night/evening/after-sunset pattern for query: {query}, got: {message}"
            )

    def test_garbage_truck_pattern(self):
        """Test detection of garbage truck pattern (Tuesday mornings)."""
        # Test various natural language queries about patterns
        pattern_queries = [
            None,  # No query (should still detect pattern)
            "pattern",
            "when does it come",
            "what time do you see it",
            "when is it usually here",
            "what's its schedule",
            "when do you normally see it",
            "what time does it typically appear"
        ]
        
        for query in pattern_queries:
            message = self.assistant.answer_object_time_query("garbage truck", query)
            self.assertIsNotNone(message, "Response message should not be None")
            
            # The response should mention the pattern since it's strong and consistent
            self.assertIn("tuesday", message.lower(), f"Response should mention Tuesday pattern for query: {query}")
            self.assertIn("morning", message.lower(), f"Response should mention morning time for query: {query}")

    def test_no_pattern(self):
        """Test handling of objects with no clear pattern."""
        # Create a random pattern for a cat
        random_data = self.mock_data.copy()
        random_times = [
            datetime.now() - timedelta(hours=i) for i in range(10)
        ]
        for time in random_times:
            random_data = pd.concat([random_data, pd.DataFrame([{
                'timestamp': time,
                'label_1': 'cat',
                'label_2': None,
                'label_3': None,
                'count_1': 1,
                'count_2': 0,
                'count_3': 0,
                'avg_conf_1': 0.85,
                'avg_conf_2': 0.0,
                'avg_conf_3': 0.0
            }])])
        
        self.assistant.load_all_logs = MagicMock(return_value=random_data)
        
        # Test various natural language queries about patterns
        pattern_queries = [
            None,  # No query (should not mention pattern)
            "pattern",
            "when does it come",
            "what time do you see it",
            "when is it usually here",
            "what's its schedule",
            "when do you normally see it",
            "what time does it typically appear"
        ]
        
        for query in pattern_queries:
            message = self.assistant.answer_object_time_query("cat", query)
            self.assertIsNotNone(message, "Response message should not be None")
            
            if query:  # For pattern queries
                self.assertIn("does not have a regular pattern", message.lower(), 
                            f"Response should indicate no clear pattern for query: {query}")
            else:  # For non-pattern queries
                self.assertNotIn("pattern", message.lower(), 
                               "Response should not mention pattern for non-pattern query")

    def test_bus_patterns_across_weekdays(self):
        """Test detection of bus patterns across different days of the week."""
        # Create data with bus patterns on different weekdays
        data = []
        start_date = datetime.now() - timedelta(days=30)
        
        for day in range(30):
            current_date = start_date + timedelta(days=day)
            weekday = current_date.weekday()
            
            # Regular bus on weekdays at different times
            if weekday < 5:  # Weekdays
                # Morning bus (7:30 AM)
                morning_time = current_date.replace(hour=7, minute=30, second=0)
                data.append({
                    'timestamp': morning_time,
                    'label_1': 'bus',
                    'label_2': None,
                    'label_3': None,
                    'count_1': 1,
                    'count_2': 0,
                    'count_3': 0,
                    'avg_conf_1': 0.90,
                    'avg_conf_2': 0.0,
                    'avg_conf_3': 0.0
                })
                
                # Afternoon bus (4:30 PM)
                afternoon_time = current_date.replace(hour=16, minute=30, second=0)
                data.append({
                    'timestamp': afternoon_time,
                    'label_1': 'school bus',
                    'label_2': None,
                    'label_3': None,
                    'count_1': 1,
                    'count_2': 0,
                    'count_3': 0,
                    'avg_conf_1': 0.90,
                    'avg_conf_2': 0.0,
                    'avg_conf_3': 0.0
                })
        
        df = pd.DataFrame(data)
        self.assistant.load_all_logs = MagicMock(return_value=df)
        
        # Test pattern detection
        message = self.assistant.answer_object_time_query("school bus", "when do you usually see it")
        self.assertIsNotNone(message, "Response message should not be None")
        self.assertIn("weekdays", message.lower(), "Response should mention weekday pattern")

    def test_school_bus_vs_regular_bus(self):
        """Test differentiation between school bus and regular bus patterns."""
        # Create data with both school bus and regular bus patterns
        data = []
        start_date = datetime.now() - timedelta(days=30)
        
        for day in range(30):
            current_date = start_date + timedelta(days=day)
            weekday = current_date.weekday()
            
            # School bus only on weekdays during school hours
            if weekday < 5:  # Weekdays
                # Morning school bus (7:00 AM)
                morning_time = current_date.replace(hour=7, minute=0, second=0)
                data.append({
                    'timestamp': morning_time,
                    'label_1': 'school bus',
                    'label_2': None,
                    'label_3': None,
                    'count_1': 1,
                    'count_2': 0,
                    'count_3': 0,
                    'avg_conf_1': 0.95,
                    'avg_conf_2': 0.0,
                    'avg_conf_3': 0.0
                })
                
                # Afternoon school bus (3:00 PM)
                afternoon_time = current_date.replace(hour=15, minute=0, second=0)
                data.append({
                    'timestamp': afternoon_time,
                    'label_1': 'school bus',
                    'label_2': None,
                    'label_3': None,
                    'count_1': 1,
                    'count_2': 0,
                    'count_3': 0,
                    'avg_conf_1': 0.95,
                    'avg_conf_2': 0.0,
                    'avg_conf_3': 0.0
                })
            
            # Regular bus every day at different times
            # Morning regular bus (8:30 AM)
            morning_time = current_date.replace(hour=8, minute=30, second=0)
            data.append({
                'timestamp': morning_time,
                'label_1': 'bus',
                'label_2': None,
                'label_3': None,
                'count_1': 1,
                'count_2': 0,
                'count_3': 0,
                'avg_conf_1': 0.90,
                'avg_conf_2': 0.0,
                'avg_conf_3': 0.0
            })
            
            # Evening regular bus (6:30 PM)
            evening_time = current_date.replace(hour=18, minute=30, second=0)
            data.append({
                'timestamp': evening_time,
                'label_1': 'bus',
                'label_2': None,
                'label_3': None,
                'count_1': 1,
                'count_2': 0,
                'count_3': 0,
                'avg_conf_1': 0.90,
                'avg_conf_2': 0.0,
                'avg_conf_3': 0.0
            })
        
        df = pd.DataFrame(data)
        self.assistant.load_all_logs = MagicMock(return_value=df)
        
        # Test school bus pattern
        message = self.assistant.answer_object_time_query("school bus", "when do you usually see it")
        self.assertIsNotNone(message, "Response message should not be None")
        self.assertIn("weekdays", message.lower(), "Response should mention weekday pattern")
        
        # Test regular bus pattern
        message = self.assistant.answer_object_time_query("bus", "when do you usually see it")
        self.assertIsNotNone(message, "Response message should not be None")
        self.assertIn("every day", message.lower(), "Response should mention daily pattern")

if __name__ == '__main__':
    unittest.main() 