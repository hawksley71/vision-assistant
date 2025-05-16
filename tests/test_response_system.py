import unittest
from unittest.mock import Mock, patch
import speech_recognition as sr
from src.core.assistant import DetectionAssistant
from src.core.voice_loop import VoiceLoop
from src.config.settings import HOME_ASSISTANT
import time
import pandas as pd

class TestResponseSystem(unittest.TestCase):
    def setUp(self):
        print("[DEBUG] Setting up test...")
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
        
        print("[DEBUG] Initializing assistant...")
        # Initialize assistant with mock components
        with patch('cv2.VideoCapture', return_value=self.mock_camera):
            self.assistant = DetectionAssistant(self.mock_mic, response_style="natural")
            self.assistant.r = self.mock_recognizer
        print("[DEBUG] Setup complete")

    def test_confidence_formatting(self):
        """Test confidence level formatting."""
        print("[DEBUG] Testing confidence formatting...")
        # Test high confidence
        high_conf = self.assistant.response_generator.format_confidence("person", 0.95)
        self.assertTrue(any(phrase in high_conf.lower() for phrase in ["quite certain", "very confident", "i'm sure"]))
        self.assertIn("95%", high_conf)
        
        # Test medium confidence
        med_conf = self.assistant.response_generator.format_confidence("chair", 0.75)
        self.assertTrue(any(phrase in med_conf.lower() for phrase in ["i think", "i believe", "it looks like"]))
        self.assertIn("75%", med_conf)
        
        # Test low confidence
        low_conf = self.assistant.response_generator.format_confidence("table", 0.45)
        self.assertTrue(any(phrase in low_conf.lower() for phrase in [
            "i might have seen",
            "i possibly saw",
            "i think i saw"
        ]))
        self.assertIn("45%", low_conf)
        print("[DEBUG] Confidence formatting tests complete")

    def test_response_generators(self):
        """Test different response generators with various scenarios."""
        print("[DEBUG] Testing response generators...")
        # Test natural style response
        self.assistant.response_style = "natural"
        natural_response = self.assistant.response_generator.format_single_detection("person", "today", 0.95)
        self.assertIn("person", natural_response.lower())
        self.assertIn("today", natural_response.lower())
        self.assertIn("today", natural_response.lower())
        
        # Test technical style response
        self.assistant.response_style = "natural"
        technical_response = self.assistant.response_generator.format_single_detection("person", "today", 0.95)
        # Check for natural language phrases
        self.assertIn("person", technical_response.lower())
        self.assertTrue(any(phrase in technical_response.lower() for phrase in [
            "quite certain",
            "very confident",
            "sure"
        ]))
        self.assertIn("today", technical_response.lower())
        
        # Test concise style response
        self.assistant.response_style = "concise"
        concise_response = self.assistant.response_generator.format_single_detection("person", "today", 0.95)
        self.assertIn("person", concise_response.lower())
        self.assertIn("today", concise_response.lower())
        print("[DEBUG] Response generator tests complete")

    def test_timestamp_formatting(self):
        """Test the timestamp formatting for different time periods."""
        print("[DEBUG] Testing timestamp formatting...")
        # Test today
        today_response = self.assistant.format_timestamp(pd.Timestamp.now())
        self.assertIn("today", today_response.lower())
        
        # Test yesterday
        yesterday_response = self.assistant.format_timestamp(pd.Timestamp.now() - pd.Timedelta(days=1))
        self.assertIn("yesterday", yesterday_response.lower())
        
        # Test this week
        week_response = self.assistant.format_timestamp(pd.Timestamp.now() - pd.Timedelta(days=7))
        self.assertIn("on", week_response.lower())
        print("[DEBUG] Timestamp formatting tests complete")

    def test_article_inclusion(self):
        """Test that responses include the appropriate article before object names."""
        print("[DEBUG] Testing article inclusion...")
        # Test with a singular object that starts with a consonant
        response = self.assistant.response_generator.format_single_detection("dog", "today", 0.95)
        self.assertTrue(any(article in response.lower() for article in ["a dog", "the dog"]), "Response should include 'a dog' or 'the dog'")
        
        # Test with a singular object that starts with a vowel
        response = self.assistant.response_generator.format_single_detection("apple", "today", 0.95)
        self.assertTrue(any(article in response.lower() for article in ["an apple", "the apple"]), "Response should include 'an apple' or 'the apple'")
        
        # Test with a plural object
        response = self.assistant.response_generator.format_single_detection("dogs", "today", 0.95)
        self.assertTrue("the dogs" in response.lower(), "Response should include 'the dogs'")
        
        print("[DEBUG] Article inclusion tests complete")

if __name__ == '__main__':
    unittest.main() 