import unittest
from unittest.mock import Mock, patch
import speech_recognition as sr
from src.core.assistant import DetectionAssistant
from src.core.voice_loop import VoiceLoop
from src.config.settings import HOME_ASSISTANT
import time
import threading

class TestVoiceInteraction(unittest.TestCase):
    def setUp(self):
        # Create mock microphone
        self.mock_mic = Mock(spec=sr.Microphone)
        self.mock_mic.__enter__ = Mock(return_value=self.mock_mic)
        self.mock_mic.__exit__ = Mock(return_value=None)
        
        # Create mock recognizer
        self.mock_recognizer = Mock(spec=sr.Recognizer)
        self.mock_recognizer.recognize_google = Mock(return_value="what do you see")
        
        # Mock camera
        self.mock_camera = Mock()
        self.mock_camera.read.return_value = (True, None)
        self.mock_camera.isOpened.return_value = True
        
        # Initialize assistant with mock components
        with patch('cv2.VideoCapture', return_value=self.mock_camera):
            self.assistant = DetectionAssistant(self.mock_mic, response_style="natural")
            self.assistant.r = self.mock_recognizer

    @patch('src.core.voice_loop.send_tts_to_ha')
    def test_what_do_you_see_query(self, mock_tts):
        """Test the 'what do you see' query handling."""
        # Mock some detections
        self.assistant.detections_buffer = [
            {'class_name': 'person', 'confidence': 0.95},
            {'class_name': 'chair', 'confidence': 0.85},
            {'class_name': 'table', 'confidence': 0.75}
        ]
        
        # Process the query
        response = self.assistant.answer_live_query("what do you see")
        
        # Verify response
        self.assertIn("person", response.lower())
        self.assertIn("chair", response.lower())
        self.assertIn("table", response.lower())
        
        # Verify TTS was called
        mock_tts.assert_called_once_with(response)

    @patch('src.core.voice_loop.send_tts_to_ha')
    def test_confidence_query(self, mock_tts):
        """Test confidence query handling."""
        # Mock some detections with confidence
        self.assistant.last_reported_labels = ['person', 'chair']
        self.assistant.last_reported_confidences = {
            'person': [0.95, 0.92, 0.94],
            'chair': [0.85, 0.82, 0.88]
        }
        
        # Process the query
        response = self.assistant.answer_live_query("how confident are you")
        
        # Verify response contains confidence information
        self.assertIn("quite certain", response.lower())
        self.assertIn("person", response.lower())
        self.assertIn("chair", response.lower())
        
        # Verify TTS was called
        mock_tts.assert_called_once_with(response)

    @patch('src.core.voice_loop.send_tts_to_ha')
    def test_historical_query(self, mock_tts):
        """Test historical query handling."""
        # Mock the query
        query = "did you see a person yesterday"
        
        # Process the query
        response = self.assistant.answer_object_time_query("person", "yesterday")
        
        # Verify response format
        self.assertIn("person", response.lower())
        self.assertIn("yesterday", response.lower())
        
        # Verify TTS was called
        mock_tts.assert_called_once_with(response)

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
        
        # Verify that the mock was called the expected number of times
        self.assertEqual(self.mock_recognizer.recognize_google.call_count, 5)  # 4 exceptions + 1 exit query

    @patch('src.core.voice_loop.send_tts_to_ha')
    def test_tts_retry_logic(self, mock_tts):
        """Test TTS retry logic."""
        # Mock TTS to fail twice then succeed
        mock_tts.side_effect = [
            Exception("TTS Error"),
            Exception("TTS Error"),
            True
        ]
        
        # Call TTS
        from src.core.voice_loop import send_tts_to_ha
        result = send_tts_to_ha("Test message", max_retries=3)
        
        # Verify TTS was called 3 times
        self.assertEqual(mock_tts.call_count, 3)
        # Verify final result
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main() 