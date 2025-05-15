import os
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import json

class OpenAIAssistantSession:
    """
    Manages an OpenAI assistant session for historical/pattern queries using the code interpreter tool.
    Handles file upload, assistant and thread creation, and message exchange.
    """
    # Class-level cache for file IDs and assistant ID
    _file_cache = {}
    _assistant_id = None
    
    def __init__(self, csv_path):
        """
        Initialize the session: load API key, upload CSV, create assistant and thread, attach file to thread.
        Args:
            csv_path (str): Path to the combined detection logs CSV file.
        """
        _ = load_dotenv(find_dotenv())
        api_key = os.getenv("OPENAI_TOKEN")
        print(f"[DEBUG] OpenAI API Key loaded: {api_key[:8]}...{api_key[-4:] if api_key else 'None'}")
        print(f"[DEBUG] API Key length: {len(api_key) if api_key else 0}")
        print(f"[DEBUG] API Key contains whitespace: {bool(api_key and any(c.isspace() for c in api_key))}")
        
        try:
            self.client = OpenAI(
                api_key=api_key,
                default_headers={"OpenAI-Beta": "assistants=v2"}
            )
            print("[DEBUG] OpenAI client initialized successfully")
            
            # Test basic API functionality
            print("[DEBUG] Testing API key with basic API call...")
            test_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=5
            )
            print("[DEBUG] Basic API test successful")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize OpenAI client: {str(e)}")
            raise
        
        # Check if file is already uploaded
        if csv_path in self._file_cache:
            print("[DEBUG] Reusing existing file ID from cache")
            self.file_id = self._file_cache[csv_path]
        else:
            print("[DEBUG] Uploading new file")
            try:
                # Upload CSV file
                with open(csv_path, "rb") as f:
                    file_obj = self.client.files.create(file=f, purpose="assistants")
                    self.file_id = file_obj.id
                    # Cache the file ID
                    self._file_cache[csv_path] = self.file_id
                print("[DEBUG] File uploaded successfully")
            except Exception as e:
                print(f"[ERROR] Failed to upload file: {str(e)}")
                raise
        
        # Create or reuse assistant
        if self._assistant_id is None:
            print("[DEBUG] Creating new assistant")
            today = datetime.now().strftime("%Y-%m-%d")
            instructions = (
                "You are a detection log and date/time expert. Use Python and pandas to analyze the detection logs. "
                "Only answer using your knowledge of the date and time and the detection logs. "
                "Unless the user asks for step-by-step reasoning, always return only the final answer in one sentence.\n"
                f"Today's date is {today}. For any questions involving time, such as 'most recent', "
                "'last', 'yesterday', 'last week', 'last month', 'this winter', or any other relative "
                "date or time phrase, use this date as the reference for 'today'. "
                "You have access to the detection logs as CSV files and can use Python and pandas to analyze them. "
                "If a user asks about an object and you do not find an exact match for the object name in the logs, "
                "search for partial string matches (case-insensitive) in the object labels. If you find up to three close matches, suggest them to the user as possible intended objects. "
            )
            self.assistant = self.client.beta.assistants.create(
                name="AI Report Assistant",
                instructions=instructions,
                model="gpt-4o",
                tools=[{"type": "code_interpreter"}]
            )
            self._assistant_id = self.assistant.id
            print(f"[DEBUG] Created new assistant with ID: {self._assistant_id}")
        else:
            print(f"[DEBUG] Reusing existing assistant with ID: {self._assistant_id}")
            self.assistant = self.client.beta.assistants.retrieve(self._assistant_id)
        
        # Create thread
        self.thread = self.client.beta.threads.create()
        # Attach the file to the thread with an initial message
        self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content="Please load the attached detection log for future questions.",
            attachments=[{"file_id": self.file_id, "tools": [{"type": "code_interpreter"}]}]
        )

    def ask_historical_question(self, user_input, last_object=None):
        """
        Send a question to the assistant and return the latest response.
        Args:
            user_input (str): The user's question.
            last_object (str, optional): If provided and 'this' is in the question, replace 'this' with last_object.
        Returns:
            str: The assistant's response.
        """
        if last_object and "this" in user_input.lower():
            user_input = user_input.lower().replace("this", last_object)
        # Send message
        self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=user_input
        )
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id
        )
        # Wait for completion
        while True:
            run_status = self.client.beta.threads.runs.retrieve(thread_id=self.thread.id, run_id=run.id)
            if run_status.status == "completed":
                break
            elif run_status.status in {"failed", "cancelled", "expired"}:
                raise RuntimeError(f"Run failed: {run_status.status}")
            time.sleep(1)
        # Get latest assistant response
        messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
        latest_msg = messages.data[0]
        for content in latest_msg.content:
            if hasattr(content, "text"):
                return content.text.value
        return "No response from assistant."

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