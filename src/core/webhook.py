from flask import Flask, request
import subprocess

app = Flask(__name__)

@app.route("/trigger", methods=["POST"])
def trigger_assistant():
    print("Trigger received from Home Assistant.")
    try:
        subprocess.Popen(["python3", "src/main_assistant.py"])
        return "Assistant triggered.", 200
    except Exception as e:
        return f"Error triggering assistant: {e}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)
