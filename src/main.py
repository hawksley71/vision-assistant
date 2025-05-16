from core.assistant import DetectionAssistant
import speech_recognition as sr

def main():
    # Initialize microphone
    mic = sr.Microphone()
    
    # Create and run assistant
    assistant = DetectionAssistant(mic)
    assistant.run()

if __name__ == "__main__":
    main() 