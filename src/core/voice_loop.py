import time
import requests
import json
import os
import threading
import cv2
from dotenv import load_dotenv
from src.core.openai_assistant import OpenAIAssistantSession
from src.utils.audio import get_microphone
import speech_recognition as sr
from src.core.assistant import DetectionAssistant
import re
from collections import deque
from src.config.settings import PATHS, CAMERA_SETTINGS, LOGGING, AUDIO_SETTINGS, HOME_ASSISTANT, MODEL_SETTINGS
from src.core.tts import send_tts_to_ha, wait_for_tts_to_finish

# Load environment variables
load_dotenv()
HOME_ASSISTANT_TOKEN = os.getenv("HOME_ASSISTANT_TOKEN")

# Constants
BUFFER_SIZE = 30  # Number of frames to keep in detection buffer

def clean_object_name(raw_name):
    stopwords = {
        "before", "after", "night", "day", "week", "month", "year", "first", "last",
        "yesterday", "today", "tomorrow", "recent", "previous", "earlier", "later"
    }
    articles = {"a", "an", "the"}
    words = raw_name.lower().split()
    timewords = []
    # Remove leading articles
    while words and words[0] in articles:
        words.pop(0)
    # Remove leading stopwords and collect them
    while words and words[0] in stopwords:
        timewords.append(words.pop(0))
    # Remove trailing stopwords and collect them
    while words and words[-1] in stopwords:
        timewords.append(words.pop())
    return " ".join(words), timewords

class VoiceLoop:
    """
    Main voice loop for handling user queries, routing to live or historical Q&A, and speaking responses.
    """
    def __init__(self, csv_path):
        print("\n[DEBUG] Initializing VoiceLoop...")
        self.oa_session = OpenAIAssistantSession(csv_path)
        self.last_object = None
        
        print("[DEBUG] Initializing microphone...")
        try:
            self.mic = get_microphone()
            print("[DEBUG] Microphone initialized successfully")
        except Exception as e:
            print(f"[ERROR] Failed to initialize microphone: {e}")
            raise
        
        print("[DEBUG] Initializing speech recognizer...")
        self.recognizer = sr.Recognizer()
        # Adjust for ambient noise with shorter duration
        print("[DEBUG] Adjusting for ambient noise...")
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Reduced from 2 seconds
        print("[DEBUG] Ambient noise adjustment complete")
        
        # Optimize voice recognition settings
        self.recognizer.energy_threshold = 1000  # Lower threshold for better sensitivity
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Shorter pause threshold
        
        print("[DEBUG] Initializing live detection assistant...")
        self.live_assistant = DetectionAssistant(self.mic)
        # Initialize detection buffer using deque
        self.live_assistant.detections_buffer = deque(maxlen=BUFFER_SIZE)
        print("[DEBUG] VoiceLoop initialization complete")
        self.voice_active = True
        self.voice_thread = None

    def test_microphone(self, timeout=2):
        """Test if the microphone is working by recording a short sample."""
        print("\n[DEBUG] Testing microphone...")
        try:
            with self.mic as source:
                print("[DEBUG] Recording test sample...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=2)
                print("[DEBUG] Test recording successful")
                return True
        except sr.WaitTimeoutError:
            print("[DEBUG] No audio detected during test")
            return False
        except Exception as e:
            print(f"[ERROR] Microphone test failed: {e}")
            return False

    def is_historical_query(self, user_input):
        """
        Use regex patterns to robustly detect if a query is about historical detections.
        """
        query = user_input.lower()
        patterns = [
            r"have you seen", r"did you see", r"ever seen", r"seen before", r"before", r"previously",
            r"last time", r"when did you see", r"how many times", r"was there a", r"were there any",
            r"when was the last", r"how often", r"in the past", r"history", r"detected.*yesterday",
            r"detected.*last week", r"detected.*last month", r"detected.*ago", r"detected.*previous",
            r"record", r"log", r"detection log", r"show", r"does the.*log", r"does the.*record",
            r"is there a record", r"is there a log", r"do the logs show", r"do the records show"
        ]
        for pat in patterns:
            if re.search(pat, query):
                print(f"[DEBUG] Historical query pattern matched: {pat}")
                return True
        return False

    def handle_live_query(self, user_input):
        return self.live_assistant.answer_live_query(user_input)

    def voice_interaction_loop(self):
        """Background thread for handling voice interactions."""
        print("\n[DEBUG] Starting voice interaction loop...")
        question_prefixes = [
            "have you ever seen", "have you seen", "did you ever see", "did you see",
            "does the detection log have a record of", "does the record show", "do the logs show",
            "do the records show", "is there a record of", "is there a log of"
        ]
        # Patterns for 'usual time' queries
        usual_time_patterns = [
            r"when does the ([\w \-]+) come",
            r"what time does the ([\w \-]+) come",
            r"when do you usually see the ([\w \-]+)",
            r"what time does the ([\w \-]+) arrive",
            r"when is the ([\w \-]+) usually here",
            r"when is the ([\w \-]+) usually seen",
            r"when does the ([\w \-]+) show up",
            r"what time is the ([\w \-]+) usually seen",
            r"what time does the ([\w \-]+) show up",
            r"when is the ([\w \-]+) usually detected",
            r"when does the ([\w \-]+) usually get detected",
        ]
        # Patterns for pattern-exploration queries
        pattern_exploration_patterns = [
            r"does the ([\w \-]+) have a pattern",
            r"does the ([\w \-]+) show a pattern",
            r"have you seen any patterns with ([\w \-]+)",
            r"does the ([\w \-]+) come regularly",
            r"is there a pattern for ([\w \-]+)",
            r"are there any patterns with ([\w \-]+)",
            r"do you notice any patterns with ([\w \-]+)",
            r"does ([\w \-]+) have a pattern",
            r"does ([\w \-]+) show a pattern",
            r"have you seen any patterns with ([\w \-]+)",
            r"does ([\w \-]+) come regularly",
            r"is there a pattern for ([\w \-]+)",
            r"are there any patterns with ([\w \-]+)",
            r"do you notice any patterns with ([\w \-]+)"
        ]
        while self.voice_active:
            try:
                print("\n[DEBUG] Listening for voice input...")
                with self.mic as source:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=7)
                print("[DEBUG] Audio captured, attempting recognition...")
                
                try:
                    user_input = self.recognizer.recognize_google(audio)
                    print(f"[DEBUG] Recognized: {user_input}")
                    
                    if user_input.lower() == "exit":
                        print("[DEBUG] Exit command received")
                        self.voice_active = False
                        break
                    
                    # 1. Check for pattern-exploration queries (route to OpenAI assistant)
                    matched = False
                    for pattern in pattern_exploration_patterns:
                        m = re.search(pattern, user_input.lower())
                        if m:
                            obj = clean_object_name(m.group(1).strip())[0]
                            if obj in {"one", "it", "that", "this"} and self.live_assistant.last_reported_labels:
                                obj = self.live_assistant.last_reported_labels[0]
                            print(f"[DEBUG] Pattern exploration matched for object: {obj}")
                            # Route to OpenAI assistant for pattern analysis
                            response = self.oa_session.ask_historical_question(
                                f"Are there any patterns for {obj}? If so, describe them in terms of days, times, or months."
                            )
                            print(f"[DEBUG] TTS will be called with: {response}")
                            send_tts_to_ha(response)
                            matched = True
                            break
                    if matched:
                        continue
                    # 2. Check for 'usual time' queries (route to local pattern analysis)
                    for pattern in usual_time_patterns:
                        m = re.search(pattern, user_input.lower())
                        if m:
                            obj = clean_object_name(m.group(1).strip())[0]
                            if obj in {"one", "it", "that", "this"} and self.live_assistant.last_reported_labels:
                                obj = self.live_assistant.last_reported_labels[0]
                            print(f"[DEBUG] Usual time pattern matched for object: {obj}")
                            df_all = self.live_assistant.load_all_logs()
                            response = self.live_assistant.analyze_object_pattern(df_all, obj)
                            print(f"[DEBUG] TTS will be called with: {response}")
                            send_tts_to_ha(response)
                            matched = True
                            break
                    if matched:
                        continue
                    # 3. Historical queries (existing logic)
                    if self.is_historical_query(user_input):
                        print("[DEBUG] Processing historical query...")
                        match = re.search(
                            r"(?:have you(?: ever)? seen|did you(?: ever)? see|does the detection log have a record of|does the record show|do the logs show|do the records show|is there a record of|is there a log of) (.+)",
                            user_input.lower()
                        )
                        if match:
                            object_name, timewords = clean_object_name(match.group(1).strip().rstrip('?'))
                            if object_name in {"one", "it", "that", "this"}:
                                if self.live_assistant.last_reported_labels:
                                    object_name = self.live_assistant.last_reported_labels[0]
                            time_expr = timewords[0] if timewords else None
                            print(f"[DEBUG] Cleaned and resolved object for historical query: {object_name}, time_expr: {time_expr}")
                            response = self.live_assistant.answer_object_time_query(object_name, time_expr)
                        else:
                            cleaned = user_input.lower().strip()
                            for prefix in question_prefixes:
                                if cleaned.startswith(prefix):
                                    cleaned = cleaned[len(prefix):].strip()
                                    break
                            object_name, timewords = clean_object_name(cleaned)
                            if object_name in {"one", "it", "that", "this"}:
                                if self.live_assistant.last_reported_labels:
                                    object_name = self.live_assistant.last_reported_labels[0]
                            time_expr = timewords[0] if timewords else None
                            print(f"[DEBUG] Fallback cleaned and resolved object for historical query: {object_name}, time_expr: {time_expr}")
                            response = self.live_assistant.answer_object_time_query(object_name, time_expr)
                        print(f"[DEBUG] TTS will be called with: {response}")
                        send_tts_to_ha(response)
                        continue
                    # 4. Live queries (existing logic)
                    print("[DEBUG] Processing live query...")
                    response = self.live_assistant.answer_live_query(user_input)
                    print(f"[DEBUG] TTS will be called with: {response}")
                    send_tts_to_ha(response)
                    
                except sr.UnknownValueError:
                    print("[DEBUG] Could not understand audio")
                except sr.RequestError as e:
                    print(f"[ERROR] Could not request results from speech recognition service: {e}")
                
            except sr.WaitTimeoutError:
                print("[DEBUG] No audio detected within timeout")
            except Exception as e:
                print(f"[ERROR] Error in voice interaction loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)  # Prevent tight loop on error

    def wait_for_intro_to_finish(self):
        """Wait for the intro TTS to finish playing."""
        url = f"{HOME_ASSISTANT['url']}/api/states/{HOME_ASSISTANT['tts_entity']}"
        headers = {"Authorization": f"Bearer {HOME_ASSISTANT['token']}"}
        
        print("[DEBUG] Waiting for intro message to finish...")
        for _ in range(20):  # Wait up to 10 seconds (20 * 0.5)
            try:
                resp = requests.get(url, headers=headers, timeout=2)
                if resp.status_code == 200:
                    state = resp.json()
                    if state.get("state") == "idle":
                        print("[DEBUG] Intro message finished playing")
                        return True
                time.sleep(0.5)
            except Exception as e:
                print(f"[DEBUG] Error checking media player state: {e}")
                time.sleep(0.5)
        
        print("[DEBUG] Timeout waiting for intro TTS to finish")
        return False

    def start_intro_and_voice(self):
        """Start the intro message and voice thread."""
        try:
            # Wait 3 seconds while detection buffer fills
            print("[DEBUG] Warming up, please wait...")
            time.sleep(3)
            
            # Test microphone before proceeding
            if not self.test_microphone():
                print("[DEBUG] WARNING: Microphone test failed. Voice recognition may not work.")
            
            # Generate and play intro via Home Assistant
            intro_text = "Hello! I am your vision-aware assistant. I can see and detect objects in my view. Ask me what I see, or about past detections."
            print("[DEBUG] Sending intro message to Home Assistant")
            send_tts_to_ha(intro_text)
            # Wait for intro to finish playing
            print("[DEBUG] Waiting for intro to finish")
            self.wait_for_intro_to_finish()
            # Now start the voice thread
            print("[DEBUG] Starting voice thread")
            self.voice_thread = threading.Thread(target=self.voice_interaction_loop, daemon=True)
            self.voice_thread.start()
            print("[DEBUG] Voice thread started")
        except Exception as e:
            print(f"[DEBUG] Error in start_intro_and_voice: {e}")
            import traceback
            traceback.print_exc()

    def run_voice_loop(self):
        """Run the main detection and voice loop."""
        print("[DEBUG] Starting voice loop...")
        
        # Initialize microphone
        try:
            mic = get_microphone()
            if not mic:
                print("[ERROR] Failed to initialize microphone")
                return
            print("[DEBUG] Microphone initialized successfully")
        except Exception as e:
            print(f"[ERROR] Error initializing microphone: {e}")
            return

        # Initialize speech recognizer
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 4000
        recognizer.dynamic_energy_threshold = True
        
        print("[DEBUG] Speech recognizer initialized")
        print("[DEBUG] Listening for voice commands...")

        while True:
            try:
                # Get frame and run detection
                ret, frame = self.live_assistant.cap.read()
                if not ret or frame is None:
                    print("[DEBUG] No frame available")
                    continue

                # Run detection
                detections = self.live_assistant.model.get_detections(frame)
                if detections:
                    self.live_assistant.latest_detections = detections
                    self.live_assistant.detections_buffer.extend(detections)
                    if len(self.live_assistant.detections_buffer) > BUFFER_SIZE:
                        self.live_assistant.detections_buffer = self.live_assistant.detections_buffer[-BUFFER_SIZE:]

                # Listen for voice command
                with mic as source:
                    print("[DEBUG] Adjusting for ambient noise...")
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    print("[DEBUG] Listening...")
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

                try:
                    print("[DEBUG] Processing speech...")
                    text = recognizer.recognize_google(audio)
                    print(f"[DEBUG] Recognized: {text}")

                    if "what do you see" in text.lower():
                        print("[DEBUG] Processing 'what do you see' query")
                        response = self.live_assistant.process_what_do_you_see()
                        if response:
                            print(f"[DEBUG] Sending TTS response: {response}")
                            if not send_tts_to_ha(response):
                                print("[ERROR] Failed to send TTS response")
                        else:
                            print("[DEBUG] No response generated for 'what do you see' query")

                except sr.WaitTimeoutError:
                    print("[DEBUG] No speech detected within timeout")
                    continue
                except sr.UnknownValueError:
                    print("[DEBUG] Could not understand audio")
                    continue
                except sr.RequestError as e:
                    print(f"[ERROR] Could not request results from speech recognition service: {e}")
                    continue
                except Exception as e:
                    print(f"[ERROR] Unexpected error in voice loop: {e}")
                    continue

            except Exception as e:
                print(f"[ERROR] Error in main loop: {e}")
                continue

    def run(self):
        """Run the main detection and voice loop."""
        # Play intro and wait for it to finish before starting detection and voice
        intro_text = "Hello! I am your vision-aware assistant. I can see and detect objects in my view. Ask me what I see, or about past detections."
        print("[DEBUG] Sending intro message to Home Assistant")
        send_tts_to_ha(intro_text)
        # Wait for intro to finish playing
        print("[DEBUG] Waiting for intro to finish")
        self.wait_for_intro_to_finish()
        print("[DEBUG] Intro finished. Starting detection and voice threads.")
        # Start the voice assistant in a background thread
        self.voice_thread = threading.Thread(target=self.run_voice_loop, daemon=True)
        self.voice_thread.start()
        print("[DEBUG] Voice thread started")
        # Now start the main detection loop
        print("[DEBUG] Starting main detection loop...")
        while self.voice_active:
            ret, frame = self.live_assistant.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
                
            # Run detection
            try:
                detections = self.live_assistant.model.get_detections(frame)
                if detections:
                    print(f"[DEBUG] Detected objects: {[d['class_name'] for d in detections]}")
                self.live_assistant.latest_detections = detections
                self.live_assistant.detections_buffer.extend(detections)
            except Exception as e:
                print(f"[DEBUG] Error in detection: {e}")
                continue
                
            # Draw detections
            try:
                if detections:  # Only draw if we have detections
                    frame = self.live_assistant.model.draw_detections(frame, detections)
            except Exception as e:
                print(f"[DEBUG] Error drawing detections: {e}")
                
            # Calculate and display FPS
            frame_count = getattr(self, 'frame_count', 0) + 1
            self.frame_count = frame_count
            if frame_count >= 30:
                end_time = time.time()
                elapsed = end_time - getattr(self, 'start_time', time.time())
                if elapsed > 0:
                    self.fps = frame_count / elapsed
                else:
                    self.fps = 0
                self.frame_count = 0
                self.start_time = time.time()
                
            cv2.putText(frame, f"FPS: {getattr(self, 'fps', 0):.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
            # Log top 3 labels every second
            now = time.time()
            if now - getattr(self.live_assistant, 'last_log_time', 0) >= 1.0:
                self.live_assistant.log_top_labels()
                self.live_assistant.last_log_time = now
                
            # Show frame
            cv2.imshow('Live Detection Assistant', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.voice_active = False
        if self.voice_thread:
            self.voice_thread.join()
        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.live_assistant.cap.release()
        cv2.destroyAllWindows()

def main():
    # Example usage
    csv_path = PATHS['data']['combined_logs']
    voice_loop = VoiceLoop(csv_path)
    voice_loop.run()

if __name__ == "__main__":
    main() 