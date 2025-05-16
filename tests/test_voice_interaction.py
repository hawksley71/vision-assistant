import unittest
from unittest.mock import Mock, patch, MagicMock
import speech_recognition as sr
from src.core.assistant import DetectionAssistant
from src.core.voice_loop import VoiceLoop
from src.core.tts import send_tts_to_ha, wait_for_tts_to_finish
from src.config.settings import HOME_ASSISTANT
import time
import threading
import pandas as pd
from datetime import datetime, timedelta

class TestVoiceInteraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test class - runs once for all tests."""
        # Create mock microphone
        cls.mock_mic = Mock(spec=sr.Microphone)
        cls.mock_mic.__enter__ = Mock(return_value=cls.mock_mic)
        cls.mock_mic.__exit__ = Mock(return_value=None)
        
        # Create mock recognizer
        cls.mock_recognizer = Mock(spec=sr.Recognizer)
        cls.mock_recognizer.recognize_google = Mock(return_value="what do you see")
        
        # Mock camera
        cls.mock_camera = Mock()
        cls.mock_camera.read.return_value = (True, None)
        cls.mock_camera.isOpened.return_value = True
        
        # Mock TTS before initializing assistant
        cls.tts_patcher = patch('src.core.tts.send_tts_to_ha', new_callable=MagicMock)
        cls.mock_tts = cls.tts_patcher.start()
        cls.mock_tts.return_value = True
        
        # Initialize assistant with mock components
        with patch('cv2.VideoCapture', return_value=cls.mock_camera):
            cls.assistant = DetectionAssistant(cls.mock_mic, response_style="natural")
            cls.assistant.r = cls.mock_recognizer
            
        # Fill detection buffer with mock data
        cls.assistant.detections_buffer = [
            {'class_name': 'person', 'confidence': 0.95},
            {'class_name': 'chair', 'confidence': 0.85},
            {'class_name': 'table', 'confidence': 0.75}
        ] * 10  # Fill buffer with 30 detections
        
        # Create mock historical data with proper timestamps
        now = datetime.now()
        cls.mock_historical_data = pd.DataFrame({
            'timestamp': [now - timedelta(hours=i) for i in range(24)],
            'label_1': ['person'] * 24,
            'avg_conf_1': [0.95] * 24,
            'label_2': ['chair'] * 24,
            'avg_conf_2': [0.85] * 24,
            'label_3': ['table'] * 24,
            'avg_conf_3': [0.75] * 24
        })
        
        # Mock the load_all_logs method
        cls.assistant.load_all_logs = Mock(return_value=cls.mock_historical_data)
        
        # Wait for warm-up period
        time.sleep(2)  # Reduced from 10-15s since we're mocking TTS

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        cls.tts_patcher.stop()

    def setUp(self):
        """Set up each test - runs before each test method."""
        # Reset mocks for each test
        self.mock_recognizer.recognize_google.reset_mock()
        self.mock_tts.reset_mock()
        # Reset the detection buffer
        self.assistant.detections_buffer = [
            {'class_name': 'person', 'confidence': 0.95},
            {'class_name': 'chair', 'confidence': 0.85},
            {'class_name': 'table', 'confidence': 0.75}
        ] * 10

    def test_what_do_you_see_query(self):
        """Test the 'what do you see' query handling."""
        # Process the query
        response = self.assistant.answer_live_query("what do you see")
        
        # Verify response
        self.assertIn("person", response.lower())
        self.assertIn("chair", response.lower())
        self.assertIn("table", response.lower())
        
        # Verify TTS was called
        self.mock_tts.assert_called()

    def test_confidence_query(self):
        """Test handling of confidence queries."""
        # Set up last reported labels and confidences
        self.assistant.last_reported_labels = ['person', 'chair', 'table']
        self.assistant.last_reported_confidences = {
            'person': [0.95],
            'chair': [0.85],
            'table': [0.75]
        }
        
        # Test confidence query
        response = self.assistant.answer_live_query("how confident are you")
        self.assertIn("confidence", response.lower())
        
        # Verify TTS was called
        self.mock_tts.assert_called()

    def test_historical_query(self):
        """Test historical query handling."""
        # Process the query
        response = self.assistant.answer_object_time_query("person", "yesterday")
        
        # Verify response format
        self.assertIn("person", response.lower())
        self.assertIn("yesterday", response.lower())
        
        # Verify TTS was called
        self.mock_tts.assert_called()

    def test_voice_loop_error_handling(self):
        """Test error handling in voice loop."""
        # Create a mock audio object
        mock_audio = Mock()
        self.mock_mic.__enter__.return_value = mock_audio
        
        # Set up the mock to raise exceptions in sequence
        exceptions = [
            sr.WaitTimeoutError(),
            sr.UnknownValueError(),
            sr.RequestError("API Error"),
            Exception("Unexpected error")
        ]
        
        def mock_recognize(*args, **kwargs):
            if not exceptions:
                return "exit"  # Return exit to stop the loop
            raise exceptions.pop(0)
        
        self.mock_recognizer.recognize_google.side_effect = mock_recognize
        
        # Run the voice query loop in a separate thread with a timeout
        def run_voice_loop():
            try:
                self.assistant.voice_query_loop()
            except Exception as e:
                if not isinstance(e, (sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError)):
                    raise
        
        thread = threading.Thread(target=run_voice_loop)
        thread.daemon = True
        thread.start()
        
        # Wait for the thread to complete or timeout after 5 seconds
        thread.join(timeout=5)
        
        # Verify that all exceptions were handled
        self.assertEqual(len(exceptions), 0, "Not all exceptions were handled")
        
        # Verify that the mock was called at least 5 times (4 exceptions + 1 exit query)
        self.assertGreaterEqual(self.mock_recognizer.recognize_google.call_count, 5)

    def test_tts_retry_logic(self):
        """Test TTS retry logic."""
        # Mock TTS to fail twice then succeed
        self.mock_tts.side_effect = [
            Exception("TTS Error"),
            Exception("TTS Error"),
            True
        ]
        
        # Call TTS directly
        result = send_tts_to_ha("Test message")
        
        # Verify TTS was called at least 3 times
        self.assertGreaterEqual(self.mock_tts.call_count, 3)
        # Verify final result
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main() 