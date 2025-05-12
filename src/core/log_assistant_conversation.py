import time
from src.utils.openai_utils import get_openai_client, create_log_assistant, create_thread, send_message_and_get_response
from src.utils.audio import get_microphone
import speech_recognition as sr
from dotenv import load_dotenv, find_dotenv

# Load environment variables (if needed)
_ = load_dotenv(find_dotenv())

# Set up OpenAI client and assistant
client = get_openai_client()
# You may want to load file_id from a config or previous step
file_id = None  # <-- Set this to your actual file_id
if file_id is None:
    raise ValueError("file_id must be set to your uploaded detection log file ID.")
assistant = create_log_assistant(client, file_id)

# Create a thread for the session
thread = create_thread(client)

# Set up speech recognizer and microphone
recognizer = sr.Recognizer()
mic = get_microphone()

print("Say 'exit' to end the session.")

while True:
    with mic as source:
        print("Listening for your question...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        user_input = recognizer.recognize_google(audio)
        print(f"You said: {user_input}")
        if user_input.lower() == "exit":
            break
        # Send user input to the assistant and get response
        answer = send_message_and_get_response(client, thread, assistant, user_input)
        print(f"Assistant: {answer}")
        # Optionally, use TTS to read the answer aloud
    except Exception as e:
        print(f"Error: {e}")

print("Session ended.")