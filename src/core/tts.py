import requests
import json
import time
from src.config.settings import HOME_ASSISTANT

def send_tts_to_ha(message):
    """Send text to Home Assistant for TTS with retry logic."""
    url = f"{HOME_ASSISTANT['url']}/api/services/tts/{HOME_ASSISTANT['tts_service'].split('.')[-1]}"
    headers = {
        "Authorization": f"Bearer {HOME_ASSISTANT['token']}",
        "Content-Type": "application/json",
    }
    payload = {
        "entity_id": HOME_ASSISTANT['tts_entity'],
        "message": message,
        "language": "en-US",
        "cache": False
    }
    
    print("[DEBUG] Posting to Home Assistant Cloud TTS:")
    print("[DEBUG] URL:", url)
    print("[DEBUG] Headers:", headers)
    print("[DEBUG] Payload:", json.dumps(payload, indent=2))
    
    for attempt in range(HOME_ASSISTANT['tts_retry_attempts']):
        try:
            print(f"[DEBUG] TTS attempt {attempt + 1}/{HOME_ASSISTANT['tts_retry_attempts']}")
            response = requests.post(url, headers=headers, json=payload, timeout=HOME_ASSISTANT['tts_timeout'])
            print(f"[DEBUG] TTS Response Status: {response.status_code}")
            
            try:
                print("[DEBUG] TTS Response JSON:", response.json())
            except Exception:
                print("[DEBUG] TTS Response Text:", response.text)
            
            if response.status_code == 200:
                print("[DEBUG] TTS request successful")
                return True
            else:
                print(f"[DEBUG] TTS request failed with status {response.status_code}")
                print(f"[DEBUG] Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"[DEBUG] TTS request timed out after {HOME_ASSISTANT['tts_timeout']} seconds")
        except Exception as e:
            print(f"[DEBUG] TTS request failed: {str(e)}")
        
        if attempt < HOME_ASSISTANT['tts_retry_attempts'] - 1:
            print("[DEBUG] Retrying TTS request...")
            time.sleep(1)  # Wait before retry
    
    print("[DEBUG] All TTS attempts failed")
    return False

def wait_for_tts_to_finish():
    """Wait for the TTS to finish playing."""
    url = f"{HOME_ASSISTANT['url']}/api/states/{HOME_ASSISTANT['tts_entity']}"
    headers = {"Authorization": f"Bearer {HOME_ASSISTANT['token']}"}
    
    print("[DEBUG] Waiting for TTS to finish...")
    for _ in range(20):  # Wait up to 10 seconds (20 * 0.5)
        try:
            resp = requests.get(url, headers=headers, timeout=2)
            if resp.status_code == 200:
                state = resp.json()
                if state.get("state") == "idle":
                    print("[DEBUG] TTS finished playing")
                    return True
            time.sleep(0.5)
        except Exception as e:
            print(f"[DEBUG] Error checking media player state: {e}")
            time.sleep(0.5)
    
    print("[DEBUG] Timeout waiting for TTS to finish")
    return False
