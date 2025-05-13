import os
import platform
import speech_recognition as sr

def list_microphones():
    """List all available microphones."""
    mics = []
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        mics.append(f"Microphone {index}: {name}")
    return mics

def get_microphone():
    """
    Get the appropriate microphone for the system.
    Handles both Linux and Windows systems.
    """
    r = sr.Recognizer()
    
    # List available microphones
    print("\n[DEBUG] Available microphones:")
    for i, mic in enumerate(sr.Microphone.list_microphone_names()):
        print(f"[DEBUG] Microphone {i}: {mic}")
    
    # Try to find a suitable microphone
    if platform.system() == "Windows":
        # On Windows, try to find a USB microphone or webcam microphone
        for i, mic in enumerate(sr.Microphone.list_microphone_names()):
            if "USB" in mic or "Webcam" in mic or "Camera" in mic:
                print(f"[DEBUG] Attempting to use {mic} (device {i})...")
                try:
                    return sr.Microphone(device_index=i)
                except Exception as e:
                    print(f"[ERROR] Failed to initialize {mic}: {e}")
                    continue
        # If no USB/webcam mic found, use default
        print("[DEBUG] No USB/webcam microphone found, using default...")
        return sr.Microphone()
    else:
        # On Linux, try to find a USB microphone
        for i, mic in enumerate(sr.Microphone.list_microphone_names()):
            if "USB" in mic:
                print(f"[DEBUG] Attempting to use {mic} (device {i})...")
                try:
                    return sr.Microphone(device_index=i)
                except Exception as e:
                    print(f"[ERROR] Failed to initialize {mic}: {e}")
                    continue
        # If no USB mic found, use default
        print("[DEBUG] No USB microphone found, using default...")
        return sr.Microphone() 