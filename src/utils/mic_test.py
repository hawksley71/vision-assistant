import speech_recognition as sr

print("Available microphones:")
for i, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"{i}: {name}")

index = int(input("Enter the device index to test: "))
r = sr.Recognizer()
with sr.Microphone(device_index=index) as source:
    print("Say something!")
    audio = r.listen(source, timeout=5, phrase_time_limit=5)
    print("Got audio, recognizing...")
    try:
        text = r.recognize_google(audio)
        print("You said:", text)
    except Exception as e:
        print("Recognition error:", e)