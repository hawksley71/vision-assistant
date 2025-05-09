from gtts import gTTS
import os

text = "In the past hour, I saw a person and a cat."
tts = gTTS(text)
tts.save("output.mp3")
os.system("mpv output.mp3")  # or "vlc", "aplay", etc.
