#!/usr/bin/env python3
import os
import sys
from src.core.assistant import DetectionAssistant
from src.utils.audio import get_microphone

def main():
    print("\n[DEBUG] Starting Vision Assistant...")
    
    # Initialize microphone
    try:
        print("[DEBUG] Initializing microphone...")
        mic = get_microphone()
        if not mic:
            print("[ERROR] Failed to initialize microphone")
            sys.exit(1)
        print("[DEBUG] Microphone initialized successfully")
    except Exception as e:
        print(f"[ERROR] Error initializing microphone: {e}")
        sys.exit(1)

    # Initialize and run the assistant
    try:
        print("[DEBUG] Initializing DetectionAssistant...")
        assistant = DetectionAssistant(mic)
        print("[DEBUG] Starting assistant...")
        assistant.run()
    except KeyboardInterrupt:
        print("\n[DEBUG] Assistant stopped by user")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[DEBUG] Cleaning up...")
        if 'assistant' in locals():
            assistant.cleanup()
        print("[DEBUG] Assistant stopped")

if __name__ == "__main__":
    main() 