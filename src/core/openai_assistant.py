import os
from dotenv import load_dotenv
import openai
import json
from datetime import datetime

load_dotenv()
OPENAI_TOKEN = os.getenv("OPENAI_TOKEN")
client = openai.OpenAI(api_key=OPENAI_TOKEN)

# Get the current date
current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Create the system message with the f-string evaluated
system_message = f"""You are a vision-aware assistant that can see and detect objects. 
You can answer questions about what you see in real-time and what you've seen in the past. 
If you don't know something or can't see it, say so.
Today's date is {current_date}. Answer time-sensitive queries relative to this date.

Example queries and responses:
- "What do you see?" -> "Right now, I am seeing a person and a chair."
- "Have you seen a cat?" -> "Yes, I saw a cat at 2:30 PM on Monday."
- "How confident are you?" -> "I am 85% confident about the person and 60% confident about the chair."
- "When does the bus usually come?" -> "The bus is usually detected around 8:30 AM and 5:15 PM on weekdays."
- "What was the last thing you saw?" -> "The last thing I saw was a dog at 3:45 PM today."

If you don't know something or can't see it, respond with phrases like:
- "I'm not seeing anything right now."
- "I haven't seen that before."
- "I'm not sure what you're asking about."
- "Could you please specify what object you're asking about?"
"""

def ask_openai(prompt, model="gpt-3.5-turbo", max_tokens=300, temperature=0.2):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content

def parse_query_with_openai(query):
    system_prompt = (
        "You are an assistant that extracts structured intents from user queries about object detections. "
        f"Today's date is {current_date}. Answer time-sensitive queries relative to this date."
        "Return a JSON object with keys: 'intent' (choose from: 'live_view', 'detection_history', 'usual_time', 'frequency', 'days_absent', 'months_present', 'confidence') "
        "and 'object' (the object of interest, e.g., 'bus', 'cat'). "
        "If the query is about the live camera view, set intent to 'live_view'. "
        "If the query is about whether an object has ever been seen or detected, set intent to 'detection_history'. "
        "If the query is about confidence levels, set intent to 'confidence'. "
        "If the query does not fit any of these, return intent as 'unknown'.\n\n"
        "Example queries and their parsed intents:\n"
        "- 'What do you see?' -> {'intent': 'live_view', 'object': None}\n"
        "- 'Have you seen a cat?' -> {'intent': 'detection_history', 'object': 'cat'}\n"
        "- 'When does the bus usually come?' -> {'intent': 'usual_time', 'object': 'bus'}\n"
        "- 'How often does the mail truck come?' -> {'intent': 'frequency', 'object': 'mail truck'}\n"
        "- 'Are there any days the bus doesn't come?' -> {'intent': 'days_absent', 'object': 'bus'}\n"
        "- 'What months does the ice cream truck come?' -> {'intent': 'months_present', 'object': 'ice cream truck'}\n"
        "- 'How confident are you?' -> {'intent': 'confidence', 'object': None}\n"
        "- 'How sure are you about that?' -> {'intent': 'confidence', 'object': None}\n"
        "- 'Have you seen' -> {'intent': 'unknown', 'object': None}\n"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        max_tokens=100,
        temperature=0,
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"intent": "unknown", "object": None}

if __name__ == "__main__":
    # Example usage
    question = "What are you seeing right now?"
    prompt = f"User query: {question}"
    answer = ask_openai(prompt)
    print("OpenAI answer:", answer) 