import os
import platform
import speech_recognition as sr
import pyaudio

def list_microphones():
    """List all available microphones using PyAudio."""
    p = pyaudio.PyAudio()
    mics = []
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info.get('maxInputChannels') > 0:  # Only input devices
            mics.append(f"Microphone {i}: {dev_info.get('name')}")
    p.terminate()
    return mics

def get_microphone():
    """
    Get the appropriate microphone for the system using PyAudio.
    Handles both Linux and Windows systems.
    """
    p = pyaudio.PyAudio()
    
    # List available microphones
    print("\n[DEBUG] Available microphones:")
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info.get('maxInputChannels') > 0:  # Only input devices
            print(f"[DEBUG] Microphone {i}: {dev_info.get('name')}")
    
    # Try to find a suitable microphone
    preferred_keywords = ["USB", "Webcam", "Camera", "CMTECK", "Microphone"]
    
    # First try to find a preferred microphone
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info.get('maxInputChannels') > 0:  # Only input devices
            name = dev_info.get('name', '').lower()
            if any(keyword.lower() in name for keyword in preferred_keywords):
                print(f"[DEBUG] Attempting to use {dev_info.get('name')} (device {i})...")
                try:
                    # Create a PyAudio stream to test the device
                    stream = p.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=int(dev_info.get('defaultSampleRate', 44100)),
                        input=True,
                        input_device_index=i,
                        frames_per_buffer=1024
                    )
                    stream.close()
                    p.terminate()
                    return sr.Microphone(device_index=i)
                except Exception as e:
                    print(f"[ERROR] Failed to initialize {dev_info.get('name')}: {e}")
                    continue
    
    # If no preferred microphone found, use default
    print("[DEBUG] No preferred microphone found, using default...")
    p.terminate()
    return sr.Microphone()

def test_microphone(mic):
    """Test if a microphone is working by recording a short sample."""
    p = pyaudio.PyAudio()
    try:
        # Get device info
        dev_info = p.get_device_info_by_index(mic.device_index)
        # Create a test stream
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=int(dev_info.get('defaultSampleRate', 44100)),
            input=True,
            input_device_index=mic.device_index,
            frames_per_buffer=1024
        )
        # Record a short sample
        data = stream.read(1024)
        stream.stop_stream()
        stream.close()
        p.terminate()
        return True
    except Exception as e:
        print(f"[ERROR] Microphone test failed: {e}")
        p.terminate()
        return False 