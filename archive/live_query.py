import speech_recognition as sr
from gtts import gTTS
import os
import cv2
import time
import torch
import threading
from queue import Queue
from models.yolov8_model import YOLOv8Model

class VoiceProcessor:
    def __init__(self):
        self.r = sr.Recognizer()
        self.mic = sr.Microphone()
        self.audio_queue = Queue()
        self.is_running = True
        self.last_response_time = 0
        self.response_cooldown = 2.0  # Minimum seconds between responses
        
    def start(self):
        self.thread = threading.Thread(target=self._process_voice)
        self.thread.daemon = True
        self.thread.start()
        
    def _process_voice(self):
        with self.mic as source:
            self.r.adjust_for_ambient_noise(source)
            
        while self.is_running:
            try:
                with self.mic as source:
                    audio = self.r.listen(source, timeout=1, phrase_time_limit=3)
                    self.audio_queue.put(audio)
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"Voice processing error: {e}")
                continue
                
    def process_audio(self, current_detections):
        if self.audio_queue.empty():
            return
            
        current_time = time.time()
        if current_time - self.last_response_time < self.response_cooldown:
            return
            
        try:
            audio = self.audio_queue.get_nowait()
            query = self.r.recognize_google(audio).lower()
            print("You said:", query)

            # More robust query recognition
            trigger_phrases = [
                "what are you seeing",
                "what do you see",
                "what do you see now",
                "what are you detecting",
                "what do you detect",
                "what is in front of you",
                "what's in front of you",
                "what's there",
                "what do you see right now"
            ]
            if any(phrase in query for phrase in trigger_phrases):
                if current_detections:
                    detection_text = ", ".join([f"{det['class_name']} ({det['confidence']:.2f})" for det in current_detections])
                    message = f"Right now, I am seeing: {detection_text}."
                else:
                    message = "I'm not seeing anything right now."
            elif any(word in query for word in ["exit", "quit", "stop", "cancel"]):
                print("Exiting.")
                return True
            else:
                message = "Please ask me what I'm seeing right now."

            print("Responding:", message)
            tts = gTTS(message)
            tts.save("response.mp3")
            os.system("mpv response.mp3")
            self.last_response_time = current_time
            
        except sr.UnknownValueError:
            print("Sorry, could not understand.")
        except Exception as e:
            print(f"Error processing audio: {e}")
            
        return False

def main():
    # Initialize camera with optimized parameters
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Initialize model
    model = YOLOv8Model()
    
    # Initialize voice processor
    voice_processor = VoiceProcessor()
    voice_processor.start()
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    print("Ask: 'What are you seeing right now?' or 'What do you see now?'. Say 'exit' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Get detections
        detections = model.detect(frame)
        print(f"Detections: {detections}")  # Debug print
        
        # Draw detections (always use the returned frame)
        frame = model.draw_detections(frame, detections)
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count >= 30:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()
        
        # Display FPS on frame
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Process voice input
        if voice_processor.process_audio(detections):
            break
            
        # Show frame
        cv2.imshow('Object Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    voice_processor.is_running = False
    voice_processor.thread.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 