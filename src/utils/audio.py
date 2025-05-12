import speech_recognition as sr

def get_microphone():
    """
    Try to use CMTECK USB Microphone (device 4), fall back to default mic if not available.
    Returns:
        sr.Microphone: The selected microphone object.
    """
    try:
        mic = sr.Microphone(device_index=4)
        print("Using CMTECK USB Microphone (device 4).")
    except Exception as e:
        print(f"Could not use CMTECK mic (device 4): {e}")
        print("Falling back to default microphone.")
        mic = sr.Microphone()  # Use default mic
    return mic 