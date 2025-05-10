from flask import Flask, request
import subprocess
import os
import requests

app = Flask(__name__)

HOME_ASSISTANT_TOKEN = os.getenv("HOME_ASSISTANT_TOKEN")  # Make sure this is set in your .env

@app.route("/trigger", methods=["POST"])
def trigger_assistant():
    print("Trigger received from Home Assistant.")

    # Send TTS "I'm ready!" to your smart speaker
    url = "http://localhost:8123/api/services/tts/cloud_say"
    headers = {
        "Authorization": f"Bearer {HOME_ASSISTANT_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "entity_id": "media_player.den_speaker",  # Change to your speaker entity if needed
        "message": "I'm ready!",
        "language": "en-US",
        "cache": False
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        print(f"[DEBUG] TTS Response Status: {response.status_code}")
        try:
            print("[DEBUG] TTS Response JSON:", response.json())
        except Exception:
            print("[DEBUG] TTS Response Text:", response.text)
    except Exception as e:
        print(f"[DEBUG] Could not send TTS message: {e}")

    return "Assistant triggered.", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)
