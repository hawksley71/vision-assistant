import requests
import os
from dotenv import load_dotenv
from pathlib import Path
import json

# Load .env with token
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
token = os.getenv("HOME_ASSISTANT_TOKEN")

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

url = "http://localhost:8123/api/services/tts/speak"
payload = {
    "media_player_entity_id": "media_player.den_speaker",
    "message": "Hello from Python using tts.speak"
}

print("[DEBUG] Posting to URL:", url)
print("[DEBUG] Headers:\n", json.dumps(headers, indent=2))
print("[DEBUG] Payload:\n", json.dumps(payload, indent=2))

response = requests.post(url, headers=headers, json=payload)

print("[DEBUG] Status Code:", response.status_code)
try:
    print("[DEBUG] Response JSON:", response.json())
except Exception:
    print("[DEBUG] Response Text:", response.text)

if response.status_code == 200:
    print("[SUCCESS] Cloud TTS request accepted by Home Assistant.")
else:
    print("[FAILURE] Cloud TTS request failed.")
