import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
from collections import deque

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.assistant import DetectionAssistant

class TestPatternDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a mock microphone
        cls.mock_mic = MagicMock()
        
        # Create synthetic data with specific patterns
        cls.mock_data = cls._generate_synthetic_data()
        
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
            
            # School bus pattern (weekdays, 7:00-7:15 AM and 3:00-3:15 PM)
            if is_weekday:
                # Morning bus
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
                
                # Afternoon bus
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
        # Mock the camera and other dependencies
        with patch('cv2.VideoCapture') as mock_camera, \
             patch('src.core.assistant.YOLOv8Model') as mock_model, \
             patch('src.core.assistant.OpenAI') as mock_openai:
            
            # Configure the mock camera
            mock_camera.return_value.isOpened.return_value = True
            
            # Configure the mock model
            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance
            
            # Configure the mock OpenAI client
            mock_openai_instance = MagicMock()
            mock_openai.return_value = mock_openai_instance
            
            # Create the assistant instance with mocked dependencies
            self.assistant = DetectionAssistant(self.mock_mic)
            
            # Mock the load_all_logs method to return our synthetic data
            self.assistant.load_all_logs = MagicMock(return_value=self.mock_data)

    def test_school_bus_pattern(self):
        """Test detection of school bus pattern (weekdays, twice daily)."""
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
            message = self.assistant.answer_object_time_query("school bus", query)
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

    def create_test_data(self, dates, times, weekdays):
        """Helper function to create test data with specific patterns."""
        data = []
        for date, time, weekday in zip(dates, times, weekdays):
            timestamp = pd.Timestamp(f"{date} {time}")
            data.append({
                'timestamp': timestamp,
                'label_1': 'test_object',
                'count_1': 1,
                'avg_conf_1': 0.9
            })
        return pd.DataFrame(data)

    def test_strong_pattern_only(self):
        # Create data with a strong morning pattern (7 out of 10 detections in morning)
        dates = ['2024-03-01'] * 10
        times = ['08:00:00'] * 7 + ['14:00:00', '15:00:00', '16:00:00']
        weekdays = ['Monday'] * 10
        df = self.create_test_data(dates, times, weekdays)
        
        result = self.assistant.analyze_object_pattern(df, 'test_object', is_pattern_query=True)
        self.assertIn("in the morning", result)
        self.assertNotIn("does not have a regular pattern", result)

    def test_moderate_pattern_only(self):
        # Create data with only moderate patterns (5 out of 10 detections in morning)
        dates = ['2024-03-01'] * 10
        times = ['08:00:00'] * 5 + ['14:00:00', '15:00:00', '16:00:00', '17:00:00', '18:00:00']
        weekdays = ['Monday'] * 10
        df = self.create_test_data(dates, times, weekdays)
        
        result = self.assistant.analyze_object_pattern(df, 'test_object', is_pattern_query=True)
        self.assertEqual(result, "The test_object does not have a regular pattern in its appearances.")

    def test_strong_and_moderate_patterns(self):
        # Create data with strong morning pattern and moderate weekday pattern
        dates = ['2024-03-01'] * 7 + ['2024-03-02'] * 3
        times = ['08:00:00'] * 7 + ['14:00:00', '15:00:00', '16:00:00']
        weekdays = ['Monday'] * 7 + ['Tuesday'] * 3
        df = self.create_test_data(dates, times, weekdays)
        
        result = self.assistant.analyze_object_pattern(df, 'test_object', is_pattern_query=True)
        self.assertIn("in the morning", result)
        self.assertIn("sometimes", result)
        self.assertNotIn("does not have a regular pattern", result)

    def test_insufficient_data(self):
        # Create data with only 5 observations
        dates = ['2024-03-01'] * 5
        times = ['08:00:00'] * 5
        weekdays = ['Monday'] * 5
        df = self.create_test_data(dates, times, weekdays)
        
        result = self.assistant.analyze_object_pattern(df, 'test_object', is_pattern_query=True)
        self.assertIn("isn't enough to establish a pattern", result)

    def test_no_patterns(self):
        # Create data with no clear patterns
        dates = ['2024-03-01'] * 10
        times = [f"{hour:02d}:00:00" for hour in range(8, 18)]
        weekdays = ['Monday'] * 10
        df = self.create_test_data(dates, times, weekdays)
        
        result = self.assistant.analyze_object_pattern(df, 'test_object', is_pattern_query=True)
        self.assertEqual(result, "The test_object does not have a regular pattern in its appearances.")

    def test_compound_object_name(self):
        """Test detection of compound object names and clarification logic."""
        # Create a mock summary DataFrame without the compound object name
        mock_summary = pd.DataFrame({
            'object': ['mail van', 'delivery truck', 'postal vehicle'],
            'total_detections': [10, 8, 5],
            'strong_patterns': ['[]', '[]', '[]']
        })
        
        # Mock the pattern summary loading
        with patch('pandas.read_csv', return_value=mock_summary):
            # Simulate a query for a compound object name not in the summary
            query = "mail truck"
            response = self.assistant.analyze_object_pattern(self.assistant.load_all_logs(), query, is_pattern_query=True)
            self.assertIn("I'm not sure if you meant", response)

            # Simulate a user response to the clarification
            user_response = "mail van"
            clarification_response = self.assistant.handle_clarification_response(user_response)
            self.assertIn("mail van", clarification_response)

if __name__ == '__main__':
    unittest.main() 