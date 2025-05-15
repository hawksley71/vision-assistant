import speech_recognition as sr
import time
import os
from dotenv import load_dotenv

class VoiceTestAssistant:
    def __init__(self, mic):
        self.mic = mic
        self.r = sr.Recognizer()
        print("[DEBUG] Voice test assistant initialized")
    
    def process_query(self, query):
        """Process voice queries without requiring camera/detection."""
        query = query.lower()
        
        # Voice-specific queries
        if "did you hear me" in query or "can you hear me" in query:
            return "Yes, I can hear you clearly!"
        elif "test" in query:
            return "Voice test successful!"
        elif "hello" in query or "hi" in query:
            return "Hello! I'm listening."
        elif "how are you" in query:
            return "I'm functioning well, thank you for asking!"
        elif "what can you do" in query:
            return "I can listen to your voice commands and respond. Try asking me 'did you hear me' or 'test'."
        elif "exit" in query or "quit" in query or "stop" in query:
            return "EXIT"
        else:
            return f"I heard you say: '{query}'. This is a voice-only test mode."

def test_voice_query():
    print("[DEBUG] Starting voice query test...")
    
    # Initialize speech recognizer
    r = sr.Recognizer()
    
    # Initialize microphone with preference for CMTECK
    print("[DEBUG] Initializing microphone...")
    mic = sr.Microphone()
    available_mics = sr.Microphone.list_microphone_names()
    print("[DEBUG] Available microphones:")
    for i, name in enumerate(available_mics):
        print(f"[DEBUG] Microphone {i}: {name}")
    
    # Try to find CMTECK microphone
    cmteck_index = None
    for i, name in enumerate(available_mics):
        if "CMTECK" in name:
            cmteck_index = i
            break
    
    if cmteck_index is not None:
        print(f"[DEBUG] Found CMTECK microphone at index {cmteck_index}")
        mic = sr.Microphone(device_index=cmteck_index)
    else:
        print("[DEBUG] CMTECK microphone not found, using default")
        mic = sr.Microphone()
    
    # Initialize voice test assistant
    print("[DEBUG] Initializing voice test assistant...")
    assistant = VoiceTestAssistant(mic)
    
    print("\nVoice test is ready!")
    print("Try saying:")
    print("- 'Did you hear me?'")
    print("- 'Test'")
    print("- 'Hello'")
    print("- 'What can you do?'")
    print("Say 'exit' to quit\n")
    
    while True:
        try:
            with mic as source:
                print("\nListening...")
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source, timeout=5)
            
            print("Processing speech...")
            query = r.recognize_google(audio).lower()
            print(f"[DEBUG] Recognized query: {query}")
            
            # Process the query
            response = assistant.process_query(query)
            if response == "EXIT":
                print("Exiting voice test...")
                break
                
            print(f"\nAssistant: {response}")
            
        except sr.WaitTimeoutError:
            print("No speech detected. Try again...")
        except sr.UnknownValueError:
            print("Sorry, could not understand. Please try again.")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("[DEBUG] Test complete.")

if __name__ == "__main__":
    test_voice_query() 