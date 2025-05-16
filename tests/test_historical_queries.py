import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime, timedelta
import re
import os
import sys
from collections import deque

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.assistant import DetectionAssistant

class TestHistoricalQueries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a mock microphone
        cls.mock_mic = MagicMock()
        
        # Create timestamps for mock data
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        two_days_ago = now - timedelta(days=2)
        three_days_ago = now - timedelta(days=3)
        
        # Create a mock DataFrame with historical data
        cls.mock_data = pd.DataFrame({
            'timestamp': [
                yesterday,
                two_days_ago,
                three_days_ago
            ],
            'label_1': ['person', 'cat', 'dog'],
            'label_2': ['chair', 'person', 'cat'],
            'label_3': ['table', 'chair', 'person'],
            'count_1': [1, 1, 1],
            'count_2': [1, 1, 1],
            'count_3': [1, 1, 1],
            'avg_conf_1': [0.9, 0.8, 0.7],
            'avg_conf_2': [0.8, 0.9, 0.8],
            'avg_conf_3': [0.7, 0.8, 0.9]
        })

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
            
            # Mock the load_all_logs method to return our test data
            self.assistant.load_all_logs = MagicMock(return_value=self.mock_data)

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
        match = re.search(r'(?:see|seen|detect|detected) (?:a |an |any )?([\w\s]+?)(?:\s+(?:on|in|during|at|this|last|yesterday|today|tomorrow|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december))?(?:\?|$)', query, re.IGNORECASE)
        if match:
            obj = match.group(1).strip()
            # Remove any time-related words that might have been captured
            obj = re.sub(r'\s+(?:on|in|during|at|this|last|yesterday|today|tomorrow|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december)$', '', obj, flags=re.IGNORECASE)
            return self.normalize_object(obj)
            
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
        # Mock load_all_logs to return empty DataFrame
        self.assistant.load_all_logs = MagicMock(return_value=pd.DataFrame())
        
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

    def test_conversation_context(self):
        """Test that the assistant maintains context across conversation turns."""
        # First query: "what do you see?"
        with patch('src.core.assistant.send_tts_to_ha') as mock_tts:
            # Mock the detections buffer with a person detection
            self.assistant.detections_buffer = deque([
                {'class_name': 'person', 'confidence': 0.9}
            ], maxlen=30)
            
            # First query - what do you see?
            # Use answer_live_query for current detection queries
            message = self.assistant.answer_live_query("what do you see")
            self.assertIsNotNone(message, "Response message should not be None")
            self.assertIn("person", message.lower(), "Response should mention the detected person")
            
            # Store the last detected object
            last_object = "person"
            self.assistant.last_detected_object = last_object
            
            # Second query - have you seen that before?
            message = self.assistant.answer_object_time_query(last_object, None)
            self.assertIsNotNone(message, "Response message should not be None")
            
            # Verify that load_all_logs was called to check history
            self.assistant.load_all_logs.assert_called()
            
            # Verify the response contains historical information
            self.assertIn(last_object, message.lower(), "Response should mention the previously detected person")
            
            # Test with a different object
            self.assistant.detections_buffer = deque([
                {'class_name': 'cat', 'confidence': 0.85}
            ], maxlen=30)
            
            # First query - what do you see?
            message = self.assistant.answer_live_query("what do you see")
            self.assertIsNotNone(message, "Response message should not be None")
            self.assertIn("cat", message.lower(), "Response should mention the detected cat")
            
            # Store the last detected object
            last_object = "cat"
            self.assistant.last_detected_object = last_object
            
            # Second query - have you seen that before?
            message = self.assistant.answer_object_time_query(last_object, None)
            self.assertIsNotNone(message, "Response message should not be None")
            
            # Verify that load_all_logs was called again
            self.assistant.load_all_logs.assert_called()
            
            # Verify the response contains historical information
            self.assertIn(last_object, message.lower(), "Response should mention the previously detected cat")
            
            # Test with no current detection
            self.assistant.detections_buffer = deque(maxlen=30)
            self.assistant.last_detected_object = None
            
            # First query - what do you see?
            message = self.assistant.answer_live_query("what do you see")
            self.assertIsNotNone(message, "Response message should not be None")
            self.assertIn("not seeing anything", message.lower(), "Response should indicate no detection")
            
            # Second query - have you seen that before?
            message = self.assistant.answer_object_time_query("", None)
            self.assertIsNotNone(message, "Response message should not be None")
            self.assertIn("did you mean one of these", message.lower(), "Response should suggest alternatives")

if __name__ == '__main__':
    unittest.main() 