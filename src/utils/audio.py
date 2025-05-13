import speech_recognition as sr

def list_microphones():
    """List all available microphones."""
    mics = []
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        mics.append(f"Microphone {index}: {name}")
    return mics

def get_microphone():
    """
    Dynamically search for 'CMTECK' in the microphone names and use its index if found.
    If not found, fall back to the default microphone.
    Returns:
        sr.Microphone: The selected microphone object.
    """
    print("\n[DEBUG] Available microphones:")
    mic_names = sr.Microphone.list_microphone_names()
    for i, name in enumerate(mic_names):
        print(f"[DEBUG] Microphone {i}: {name}")
    
    cmteck_index = None
    for i, name in enumerate(mic_names):
        if "cmteck" in name.lower():
            cmteck_index = i
            break
    
    if cmteck_index is not None:
        try:
            print(f"\n[DEBUG] Attempting to use CMTECK USB Microphone (device {cmteck_index})...")
            mic = sr.Microphone(device_index=cmteck_index)
            print(f"[DEBUG] Successfully initialized CMTECK USB Microphone (device {cmteck_index})")
            return mic
        except Exception as e:
            print(f"[DEBUG] Could not use CMTECK mic (device {cmteck_index}): {e}")
            print("[DEBUG] Falling back to default microphone...")
    else:
        print("[DEBUG] CMTECK microphone not found. Falling back to default microphone...")
    try:
        mic = sr.Microphone()  # Use default mic
        print("[DEBUG] Successfully initialized default microphone")
        return mic
    except Exception as e:
        print(f"[DEBUG] Could not initialize default microphone: {e}")
        raise RuntimeError("No working microphone found") 