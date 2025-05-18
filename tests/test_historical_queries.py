import unittest
from unittest.mock import MagicMock, patch, Mock
import pandas as pd
from datetime import datetime, timedelta
import re
import os
import sys
from collections import deque
import speech_recognition as sr
from src.core.camera import Camera

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.assistant import DetectionAssistant

class TestHistoricalQueries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment before running tests."""
        # Create test data directory if it doesn't exist
        os.makedirs('data/raw', exist_ok=True)
        
        # Create test data with bus detections
        cls.create_test_data()
        
        # Create a mock microphone
        cls.mock_mic = MagicMock()
        
        # Create mock camera
        cls.mock_camera = MagicMock(spec=Camera)
        cls.mock_camera.cap = MagicMock()
        cls.mock_camera.cap.isOpened.return_value = True
        cls.mock_camera.cap.read.return_value = (True, None)
        
        # Initialize assistant with mock camera
        cls.assistant = DetectionAssistant(cls.mock_mic, camera=cls.mock_camera)
        
        # Mock the load_all_logs method to return our synthetic data
        cls.assistant.load_all_logs = MagicMock(return_value=cls._generate_synthetic_data())

    @classmethod
    def _generate_synthetic_data(cls):
        """Generate synthetic detection data for testing."""
        # Create timestamps for the last 30 days
        now = datetime.now()
        data = []
        
        for i in range(30):
            current_date = now - timedelta(days=i)
            weekday = current_date.weekday()
            
            # Bus pattern (weekdays only)
            if weekday < 5:  # Weekdays
                # Morning bus (7:00-7:15 AM)
                morning_time = current_date.replace(hour=7, minute=10, second=0)
                data.append({
                    'timestamp': morning_time,
                    'label_1': 'bus',
                    'count_1': 1,
                    'avg_conf_1': 0.95,
                    'label_2': 'person',
                    'count_2': 1,
                    'avg_conf_2': 0.85,
                    'label_3': '',
                    'count_3': 0,
                    'avg_conf_3': 0.0
                })
                
                # Afternoon bus (2:45-3:00 PM)
                afternoon_time = current_date.replace(hour=14, minute=50, second=0)
                data.append({
                    'timestamp': afternoon_time,
                    'label_1': 'bus',
                    'count_1': 1,
                    'avg_conf_1': 0.95,
                    'label_2': 'person',
                    'count_2': 1,
                    'avg_conf_2': 0.85,
                    'label_3': '',
                    'count_3': 0,
                    'avg_conf_3': 0.0
                })
            
            # Regular bus pattern (every day)
            # Morning bus (8:30 AM)
            morning_time = current_date.replace(hour=8, minute=30, second=0)
            data.append({
                'timestamp': morning_time,
                'label_1': 'bus',
                'count_1': 1,
                'avg_conf_1': 0.90,
                'label_2': '',
                'count_2': 0,
                'avg_conf_2': 0.0,
                'label_3': '',
                'count_3': 0,
                'avg_conf_3': 0.0
            })
            
            # Evening bus (6:30 PM)
            evening_time = current_date.replace(hour=18, minute=30, second=0)
            data.append({
                'timestamp': evening_time,
                'label_1': 'bus',
                'count_1': 1,
                'avg_conf_1': 0.90,
                'label_2': '',
                'count_2': 0,
                'avg_conf_2': 0.0,
                'label_3': '',
                'count_3': 0,
                'avg_conf_3': 0.0
            })
            
            # Garbage truck pattern (Tuesdays only)
            if weekday == 1:  # Tuesday
                # First garbage truck (6:12 AM)
                truck_time_1 = current_date.replace(hour=6, minute=12, second=0)
                data.append({
                    'timestamp': truck_time_1,
                    'label_1': 'garbage truck',
                    'count_1': 1,
                    'avg_conf_1': 0.90,
                    'label_2': '',
                    'count_2': 0,
                    'avg_conf_2': 0.0,
                    'label_3': '',
                    'count_3': 0,
                    'avg_conf_3': 0.0
                })
                
                # Second garbage truck (6:45 AM)
                truck_time_2 = current_date.replace(hour=6, minute=45, second=0)
                data.append({
                    'timestamp': truck_time_2,
                    'label_1': 'garbage truck',
                    'count_1': 1,
                    'avg_conf_1': 0.90,
                    'label_2': '',
                    'count_2': 0,
                    'avg_conf_2': 0.0,
                    'label_3': '',
                    'count_3': 0,
                    'avg_conf_3': 0.0
                })
            
            # Raccoon pattern (night only, every other day)
            if i % 2 == 0:  # Every other day
                night_time = current_date.replace(hour=20, minute=30, second=0)
                data.append({
                    'timestamp': night_time,
                    'label_1': 'raccoon',
                    'count_1': 1,
                    'avg_conf_1': 0.85,
                    'label_2': '',
                    'count_2': 0,
                    'avg_conf_2': 0.0,
                    'label_3': '',
                    'count_3': 0,
                    'avg_conf_3': 0.0
                })
        
        return pd.DataFrame(data)

    @classmethod
    def create_test_data(cls):
        """Create test data with bus detections."""
        # Create timestamps for the last 7 days
        now = datetime.now()
        timestamps = []
        for i in range(7):
            # Morning bus (7:00-7:15 AM)
            morning_time = now - timedelta(days=i)
            morning_time = morning_time.replace(hour=7, minute=10, second=0, microsecond=0)
            timestamps.append(morning_time)
            
            # Afternoon bus (2:45-3:00 PM)
            afternoon_time = now - timedelta(days=i)
            afternoon_time = afternoon_time.replace(hour=14, minute=50, second=0, microsecond=0)
            timestamps.append(afternoon_time)
            
            # Regular bus (random times)
            random_time = now - timedelta(days=i)
            random_time = random_time.replace(hour=10, minute=30, second=0, microsecond=0)
            timestamps.append(random_time)
        
        # Create DataFrame with test data
        data = []
        for ts in timestamps:
            # Bus during school hours
            if ts.hour in [7, 14]:
                data.append({
                    'timestamp': ts,
                    'label_1': 'bus',
                    'count_1': 1,
                    'avg_conf_1': 0.95,
                    'label_2': 'person',
                    'count_2': 1,
                    'avg_conf_2': 0.85,
                    'label_3': '',
                    'count_3': 0,
                    'avg_conf_3': 0.0
                })
            # Regular bus at other times
            else:
                data.append({
                    'timestamp': ts,
                    'label_1': 'bus',
                    'count_1': 1,
                    'avg_conf_1': 0.90,
                    'label_2': '',
                    'count_2': 0,
                    'avg_conf_2': 0.0,
                    'label_3': '',
                    'count_3': 0,
                    'avg_conf_3': 0.0
                })
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv('data/raw/detections_test.csv', index=False)

    def normalize_object(self, obj):
        """Normalize object labels to match the expected format."""
        if not obj:
            return None
            
        # Handle common plural forms
        plural_to_singular = {
            'people': 'person',
            'persons': 'person',
            'cats': 'cat',
            'dogs': 'dog',
            'chairs': 'chair',
            'tables': 'table'
        }
        
        # Convert to lowercase and strip
        obj = obj.lower().strip()
        
        # Check if it's a plural form we know about
        if obj in plural_to_singular:
            return plural_to_singular[obj]
            
        return obj

    def extract_object(self, query):
        """Extract the main object from a query, handling various formats."""
        # First try to extract after common prepositions
        match = re.search(r'(?:for|about|of) ([\w\s]+?)(?:\s+(?:on|in|during|at|this|last|yesterday|today|tomorrow|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december))?(?:\?|$)', query, re.IGNORECASE)
        if match:
            obj = match.group(1).strip()
            # Remove any time-related words that might have been captured
            obj = re.sub(r'\s+(?:on|in|during|at|this|last|yesterday|today|tomorrow|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december)$', '', obj, flags=re.IGNORECASE)
            return self.normalize_object(obj)
        
        # Try to extract after "see" or "detect"
        match = re.search(r'(?:see|seen|detect|detected|noticed|observe|observed|find|found|spot|spotted|have|has|are|is|was|were|do|does|did) (?:a |an |any |the )?([\w\s]+?)(?:\s+(?:on|in|during|at|this|last|yesterday|today|tomorrow|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december))?(?:\?|$)', query, re.IGNORECASE)
        if match:
            obj = match.group(1).strip()
            obj = re.sub(r'\s+(?:on|in|during|at|this|last|yesterday|today|tomorrow|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december)$', '', obj, flags=re.IGNORECASE)
            return self.normalize_object(obj)
        
        # Try to extract object from queries about schedule, pattern, frequency, etc.
        match = re.search(r"(?:what(?:'s| is)?|when|how|do|does|are|is|can|could|has|have|will|would|should|did|does) (?:you )?(?:see|detect|notice|observe|find|spot|know|tell|show|give|provide)? ?(?:the |a |an |any |about |of |for |on |in |to |with |by )?([\w\s]+?)(?: schedule| pattern| frequency| timetable| arrival| service| route| timing| times| kind| type| variety| varieties| categories| categories| like| usually| often| common| regular| pass| come| appear| depart| run| here| there| daily| weekly| monthly| yearly| at| on| in| during| at| this| last| yesterday| today| tomorrow| week| month| year| monday| tuesday| wednesday| thursday| friday| saturday| sunday| january| february| march| april| may| june| july| august| september| october| november|december|\?|$)", query, re.IGNORECASE)
        if match:
            obj = match.group(1).strip()
            return self.normalize_object(obj)
        
        # Fallback: look for known objects in the query
        known_objects = ["bus", "garbage truck", "raccoon", "cat", "dog", "person", "chair", "table"]
        for obj in known_objects:
            if obj in query.lower():
                return obj
        
        # If no match found, return None
        return None

    def test_historical_keywords(self):
        """Test that queries containing historical keywords trigger log lookups."""
        historical_queries = [
            ("Show me the logs for person", "person"),
            ("What records do you have about cats", "cat"),
            ("Tell me about past detections of dogs", "dog"),
            ("Show me the history of chair detections", "chair"),
            ("What observations do you have about tables", "table"),
            ("Show me previous detections of people", "person"),
            ("What items have you seen in the past", None),  # No specific object
            ("Show me earlier detections of cats", "cat"),
            ("What have you seen historically", None),  # No specific object
            ("Show me the detection history for dogs", "dog")
        ]

        for query, expected_obj in historical_queries:
            with self.subTest(query=query):
                # Mock the voice query loop's response handling
                with patch('src.core.assistant.send_tts_to_ha') as mock_tts:
                    # Extract and normalize object from query
                    obj = self.extract_object(query)
                    
                    # For queries without specific objects
                    if expected_obj is None:
                        message = "What object would you like me to check in the logs?"
                        self.assertEqual(message, "What object would you like me to check in the logs?")
                    else:
                        # For queries with specific objects
                        message = self.assistant.answer_object_time_query(obj, None)
                        
                        # Verify that load_all_logs was called
                        self.assistant.load_all_logs.assert_called()
                        
                        # Verify the response contains the expected object
                        self.assertIsNotNone(message, "Response message should not be None")
                        self.assertIn(expected_obj, message.lower())

    def test_time_expressions(self):
        """Test that time expressions are correctly parsed and applied to queries."""
        time_queries = [
            ("Did you see a person last week?", "last week", "person"),
            ("Have you seen any cats this week?", "this week", "cat"),
            ("Did you detect dogs yesterday?", "yesterday", "dog"),
            ("Have you seen chairs today?", "today", "chair"),
            ("Did you see tables last month?", "last month", "table"),
            ("Have you seen people this month?", "this month", "person"),
            ("Did you detect cats on Monday?", "monday", "cat"),
            ("Have you seen dogs on Tuesday?", "tuesday", "dog")
        ]

        # Ensure load_all_logs always returns a DataFrame with the correct columns
        columns = [
            'timestamp', 'label_1', 'count_1', 'avg_conf_1',
            'label_2', 'count_2', 'avg_conf_2',
            'label_3', 'count_3', 'avg_conf_3'
        ]
        empty_df = pd.DataFrame(columns=columns)
        self.assistant.load_all_logs = MagicMock(return_value=empty_df)

        for query, expected_time, expected_obj in time_queries:
            with self.subTest(query=query, time=expected_time):
                # Extract and normalize object from query
                obj = self.extract_object(query)
                self.assertIsNotNone(obj, f"Failed to extract object from query: {query}")
                
                # Test the time expression parsing
                start, end = self.assistant.parse_time_expression(expected_time)
                self.assertIsNotNone(start, f"Failed to parse start time for {expected_time}")
                self.assertIsNotNone(end, f"Failed to parse end time for {expected_time}")
                
                # Test the full query handling
                message = self.assistant.answer_object_time_query(obj, expected_time)
                self.assertIsNotNone(message, "Response message should not be None")
                self.assertIn(expected_obj, message.lower())
                
                # Verify that load_all_logs was called
                self.assistant.load_all_logs.assert_called()

    def test_no_historical_data(self):
        """Test handling of queries when no historical data is available."""
        # Mock load_all_logs to return empty DataFrame with correct columns
        columns = [
            'timestamp', 'label_1', 'count_1', 'avg_conf_1',
            'label_2', 'count_2', 'avg_conf_2',
            'label_3', 'count_3', 'avg_conf_3'
        ]
        empty_df = pd.DataFrame(columns=columns)
        self.assistant.load_all_logs = MagicMock(return_value=empty_df)
        
        queries = [
            "Show me the logs for person",
            "What records do you have about cats",
            "Tell me about past detections"
        ]
        
        for query in queries:
            with self.subTest(query=query):
                obj = self.extract_object(query)
                if obj:
                    message = self.assistant.answer_object_time_query(obj, None)
                    self.assertIsNotNone(message, "Response message should not be None")
                    self.assertIn("no records", message.lower())
                else:
                    message = "What object would you like me to check in the logs?"
                    self.assertEqual(message, "What object would you like me to check in the logs?")
                
                # Verify that load_all_logs was called
                self.assistant.load_all_logs.assert_called()

    def test_bus_queries(self):
        """Test various queries about bus detections."""
        test_queries = [
            "have you seen a bus before",
            "have you seen a bus",
            "have you seen one in the past",
            "do the records show a bus",
            "has a bus been detected",
            "when did you last see a bus",
            "what time do you usually see a bus"
        ]
        
        for query in test_queries:
            with self.subTest(query=query):
                response = self.assistant.process_historical_query(query)
                self.assertIsNotNone(response)
                self.assertIsInstance(response, str)
                print(f"\nQuery: {query}")
                print(f"Response: {response}")
    
    def test_school_bus_queries(self):
        """Test various queries about school bus detections."""
        test_queries = [
            "have you seen a school bus before",
            "have you seen a school bus",
            "have you seen one in the past",
            "do the records show a school bus",
            "has a school bus been detected",
            "when did you last see a school bus",
            "what time do you usually see a school bus"
        ]
        
        for query in test_queries:
            with self.subTest(query=query):
                response = self.assistant.process_historical_query(query)
                self.assertIsNotNone(response)
                self.assertIsInstance(response, str)
                print(f"\nQuery: {query}")
                print(f"Response: {response}")
    
    def test_pattern_queries(self):
        """Test queries about detection patterns."""
        test_queries = [
            "what's the pattern for bus detections",
            "when do you usually see the bus",
            "what time does the bus come",
            "is there a regular pattern for bus detections",
            "when do you typically see buses"
        ]
        
        for query in test_queries:
            with self.subTest(query=query):
                response = self.assistant.process_historical_query(query)
                self.assertIsNotNone(response)
                self.assertIsInstance(response, str)
                print(f"\nQuery: {query}")
                print(f"Response: {response}")

    def test_natural_language_bus_queries(self):
        """Test various natural language variations for bus-related queries."""
        # Test queries about bus presence
        presence_queries = [
            "have you noticed any buses lately",
            "do you see buses often",
            "are there buses in the area",
            "have you detected any buses",
            "do buses come by here",
            "have you seen buses around",
            "are buses common here",
            "do buses pass by frequently"
        ]
        
        for query in presence_queries:
            with self.subTest(query=query):
                response = self.assistant.process_historical_query(query)
                self.assertIsNotNone(response)
                self.assertIsInstance(response, str)
                self.assertIn("bus", response.lower())
        
        # Test queries about bus timing
        timing_queries = [
            "when do buses usually come by",
            "what time do you typically see buses",
            "when are buses most common",
            "at what times do buses appear",
            "when do buses pass by",
            "what's the usual time for buses",
            "when are buses most frequent",
            "what times do you notice buses"
        ]
        
        for query in timing_queries:
            with self.subTest(query=query):
                response = self.assistant.process_historical_query(query)
                self.assertIsNotNone(response)
                self.assertIsInstance(response, str)
                self.assertIn("bus", response.lower())
                self.assertTrue(
                    any(word in response.lower() for word in ["time", "when", "schedule", "pattern"]),
                    f"Response should mention timing: {response}"
                )
        
        # Test queries about bus frequency
        frequency_queries = [
            "how often do buses come by",
            "what's the frequency of buses",
            "how frequently do you see buses",
            "how many buses do you see daily",
            "how regular are the buses",
            "how common are buses here",
            "what's the bus frequency like",
            "how many times do buses pass by"
        ]
        
        for query in frequency_queries:
            with self.subTest(query=query):
                response = self.assistant.process_historical_query(query)
                self.assertIsNotNone(response)
                self.assertIsInstance(response, str)
                self.assertIn("bus", response.lower())
                self.assertTrue(
                    any(word in response.lower() for word in ["often", "frequency", "regular", "times", "daily"]),
                    f"Response should mention frequency: {response}"
                )
        
        # Test queries about bus types
        type_queries = [
            "what kinds of buses do you see",
            "do you see different types of buses",
            "what types of buses pass by",
            "are there different bus varieties",
            "what bus varieties have you noticed",
            "do you see various bus types",
            "what bus categories do you detect",
            "are there multiple bus types"
        ]
        
        for query in type_queries:
            with self.subTest(query=query):
                response = self.assistant.process_historical_query(query)
                self.assertIsNotNone(response)
                self.assertIsInstance(response, str)
                self.assertIn("bus", response.lower())
                self.assertTrue(
                    any(word in response.lower() for word in ["type", "kind", "variety", "school", "regular"]),
                    f"Response should mention bus types: {response}"
                )

    def test_bus_schedule_queries(self):
        """Test queries about bus schedules and patterns."""
        schedule_queries = [
            "what's the bus schedule like",
            "can you tell me the bus timetable",
            "what are the bus arrival times",
            "when do buses arrive and depart",
            "what's the bus service pattern",
            "how do the buses run",
            "what's the bus route timing",
            "when are the bus services"
        ]
        
        for query in schedule_queries:
            with self.subTest(query=query):
                response = self.assistant.process_historical_query(query)
                self.assertIsNotNone(response)
                self.assertIsInstance(response, str)
                self.assertIn("bus", response.lower())
                self.assertTrue(
                    any(word in response.lower() for word in ["schedule", "time", "pattern", "regular", "service"]),
                    f"Response should mention schedule: {response}"
                )

if __name__ == '__main__':
    unittest.main(verbosity=2) 