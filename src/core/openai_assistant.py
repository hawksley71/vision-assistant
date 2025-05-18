import os
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import json

# Define current_date at module level
current_date = datetime.now().strftime("%Y-%m-%d")

def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format."""
    if not api_key:
        return False
    # OpenAI API keys start with 'sk-' and are 51 characters long
    return api_key.startswith('sk-') and len(api_key) == 51

def get_api_key() -> str:
    """Get and validate OpenAI API key from environment."""
    api_key = os.getenv("OPENAI_TOKEN")
    if not api_key:
        raise RuntimeError("Error: OPENAI_TOKEN not found in environment variables")
    if not validate_api_key(api_key):
        raise RuntimeError("Error: Invalid OpenAI API key format")
    return api_key

class OpenAIAssistantSession:
    """
    Manages an OpenAI assistant session for historical/pattern queries using the code interpreter tool.
    Handles file upload, assistant and thread creation, and message exchange.
    """
    # Class-level cache for file IDs, assistant ID, and thread ID
    _file_cache = {}
    _assistant_id = None
    _thread_id = None
    _cache_file = "openai_cache.json"
    
    @classmethod
    def _load_cache(cls):
        """Load cached IDs from file."""
        try:
            if os.path.exists(cls._cache_file):
                with open(cls._cache_file, 'r') as f:
                    cache = json.load(f)
                    cls._assistant_id = cache.get('assistant_id')
                    cls._thread_id = cache.get('thread_id')
                    cls._file_cache = cache.get('file_cache', {})
                    print(f"[DEBUG] Loaded cache: assistant_id={cls._assistant_id}, thread_id={cls._thread_id}")
        except Exception as e:
            print(f"[DEBUG] Error loading cache: {e}")
    
    @classmethod
    def _save_cache(cls):
        """Save cached IDs to file."""
        try:
            cache = {
                'assistant_id': cls._assistant_id,
                'thread_id': cls._thread_id,
                'file_cache': cls._file_cache
            }
            with open(cls._cache_file, 'w') as f:
                json.dump(cache, f)
            print(f"[DEBUG] Saved cache: assistant_id={cls._assistant_id}, thread_id={cls._thread_id}")
        except Exception as e:
            print(f"[DEBUG] Error saving cache: {e}")
    
    def _wait_for_file_processing(self, file_id, max_retries=10, delay=2):
        """Wait for a file to be processed and ready for use."""
        for i in range(max_retries):
            try:
                file = self.client.files.retrieve(file_id)
                if file.status == 'processed':
                    print(f"[DEBUG] File {file_id} is ready for use")
                    return True
                elif file.status == 'error':
                    raise RuntimeError(f"File processing failed: {file.status_details}")
                print(f"[DEBUG] Waiting for file {file_id} to be processed... (attempt {i+1}/{max_retries})")
                time.sleep(delay)
            except Exception as e:
                print(f"[ERROR] Error checking file status: {e}")
                time.sleep(delay)
        raise RuntimeError(f"File {file_id} failed to process within {max_retries * delay} seconds")

    def _upload_new_file(self, csv_path):
        """Helper method to upload a new file."""
        print("[DEBUG] Uploading new file")
        try:
            with open(csv_path, "rb") as f:
                file_obj = self.client.files.create(file=f, purpose="assistants")
                self.file_id = file_obj.id
                # Cache the file ID
                self._file_cache[csv_path] = self.file_id
                self._save_cache()  # Save cache after updating file ID
            print("[DEBUG] File uploaded successfully")
            # Wait for file to be processed
            self._wait_for_file_processing(self.file_id)
        except Exception as e:
            print(f"[ERROR] Failed to upload file: {str(e)}")
            raise

    def __init__(self, csv_path):
        """
        Initialize the session: load API key, upload CSV, create assistant and thread, attach file to thread.
        Args:
            csv_path (str): Path to the combined detection logs CSV file.
        """
        # Load cached IDs
        self._load_cache()
        
        # Load and validate API key
        try:
            api_key = get_api_key()
            self.client = OpenAI(api_key=api_key)
            print("[DEBUG] OpenAI client initialized successfully")
        except Exception as e:
            print(f"[ERROR] Failed to initialize OpenAI client: {str(e)}")
            raise

        # Test API key with minimal request
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            print("[DEBUG] API key test successful")
        except Exception as e:
            print(f"[ERROR] API key test failed: {str(e)}")
            raise
        
        # Check for existing files first
        try:
            existing_files = self.client.files.list()
            combined_logs_files = [f for f in existing_files.data if f.filename == 'combined_logs.csv']
            
            if combined_logs_files:
                # Get the most recent file
                latest_file = max(combined_logs_files, key=lambda x: x.created_at)
                file_age = time.time() - latest_file.created_at
                
                # If file is less than 2 hours old, use it
                if file_age < 7200:  # 7200 seconds = 2 hours
                    print(f"[DEBUG] Using existing file (ID: {latest_file.id}) that is {int(file_age/60)} minutes old")
                    self.file_id = latest_file.id
                    # Verify the file is still processed and ready
                    self._wait_for_file_processing(self.file_id)
                    self._file_cache[csv_path] = self.file_id
                    self._save_cache()
                else:
                    # Delete old files and upload new one
                    for file in combined_logs_files:
                        try:
                            print(f"[DEBUG] Deleting old file (ID: {file.id})")
                            self.client.files.delete(file.id)
                        except Exception as e:
                            print(f"[WARNING] Failed to delete old file {file.id}: {e}")
                            continue
                    self._upload_new_file(csv_path)
            else:
                self._upload_new_file(csv_path)
        except Exception as e:
            print(f"[ERROR] Failed to check existing files: {e}")
            # If we can't check existing files, try to upload a new one
            try:
                self._upload_new_file(csv_path)
            except Exception as upload_error:
                print(f"[ERROR] Failed to upload new file: {upload_error}")
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
            try:
                self.assistant = self.client.beta.assistants.create(
                    name="AI Report Assistant",
                    instructions=instructions,
                    model="gpt-4",
                    tools=[{"type": "code_interpreter"}]
                )
                self._assistant_id = self.assistant.id
                self._save_cache()  # Save cache after creating assistant
                print(f"[DEBUG] Created new assistant with ID: {self._assistant_id}")
            except Exception as e:
                print(f"[ERROR] Failed to create assistant: {e}")
                raise
        else:
            print(f"[DEBUG] Reusing existing assistant with ID: {self._assistant_id}")
            try:
                self.assistant = self.client.beta.assistants.retrieve(self._assistant_id)
            except Exception as e:
                print(f"[ERROR] Failed to retrieve assistant: {e}")
                # If retrieval fails, create a new assistant
                self._assistant_id = None
                self.__init__(csv_path)
                return
        
        # Create or reuse thread
        if self._thread_id is None:
            print("[DEBUG] Creating new thread")
            try:
                self.thread = self.client.beta.threads.create()
                self._thread_id = self.thread.id
                self._save_cache()  # Save cache after creating thread
                print(f"[DEBUG] Created new thread with ID: {self._thread_id}")
                
                # Attach the file to the thread with an initial message
                self.client.beta.threads.messages.create(
                    thread_id=self.thread.id,
                    role="user",
                    content="Please load the attached detection log for future questions.",
                    attachments=[{"file_id": self.file_id, "tools": [{"type": "code_interpreter"}]}]
                )
            except Exception as e:
                print(f"[ERROR] Failed to create thread: {e}")
                raise
        else:
            print(f"[DEBUG] Reusing existing thread with ID: {self._thread_id}")
            try:
                self.thread = self.client.beta.threads.retrieve(self._thread_id)
            except Exception as e:
                print(f"[ERROR] Failed to retrieve thread: {e}")
                # If retrieval fails, create a new thread
                self._thread_id = None
                self.__init__(csv_path)
                return

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

def parse_query_with_openai(query: str, client) -> dict:
    """
    Parse a natural language query using OpenAI to determine intent and extract objects.
    
    Args:
        query (str): The natural language query to parse
        client: The OpenAI client instance
        
    Returns:
        dict: A dictionary containing:
            - intent: The detected intent (live_view, confidence, detection_history, etc.)
            - object: The object being queried about (if any)
    """
    try:
        # Construct system prompt for intent extraction
        system_prompt = """You are an intent classifier for a computer vision assistant. 
        Classify the user's query into one of these intents:
        - live_view: Questions about what is currently being seen
        - confidence: Questions about detection confidence or certainty
        - detection_history: Questions about when something was last seen
        - usual_time: Questions about typical times something is seen
        - frequency: Questions about how often something is seen
        - days_absent: Questions about when something was not seen
        - months_present: Questions about which months something was seen
        - unknown: If the query doesn't match any other intent
        
        Also extract any object being queried about.
        
        Return your response as a JSON object with 'intent' and 'object' fields.
        The object field should be null if no specific object is mentioned."""
        
        # Get response from OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.1,  # Lower temperature for more consistent results
            max_tokens=150
        )
        
        # Extract and parse the response
        try:
            result = json.loads(response.choices[0].message.content)
            return {
                "intent": result.get("intent", "unknown"),
                "object": result.get("object")
            }
        except json.JSONDecodeError:
            print(f"[DEBUG] Failed to parse OpenAI response as JSON: {response.choices[0].message.content}")
            return {"intent": "unknown", "object": None}
            
    except Exception as e:
        print(f"[DEBUG] Error in OpenAI query parsing: {e}")
        return {"intent": "unknown", "object": None}

if __name__ == "__main__":
    # Example usage
    question = "What are you seeing right now?"
    prompt = f"User query: {question}"
    answer = ask_openai(prompt)
    print("OpenAI answer:", answer) 