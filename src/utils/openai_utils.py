import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import time
from datetime import datetime

def get_openai_client():
    _ = load_dotenv(find_dotenv())
    return OpenAI(
        api_key=os.getenv("OPENAI_TOKEN"),
        default_headers={"OpenAI-Beta": "assistants=v2"}
    )

def create_log_assistant(client, file_ids, instructions=None, model="gpt-4o"):
    today = datetime.now().strftime("%Y-%m-%d")
    date_instruction = (
        f"Today's date is {today}. For any questions involving time, such as 'most recent', "
        "'last', 'yesterday', 'last week', 'last month', 'this winter', or any other relative "
        "date or time phrase, use this date as the reference for 'today'. "
        "You have access to the detection logs as CSV files and can use Python and pandas to analyze them."
    )
    if instructions is None:
        instructions = (
            "You are a detection log and date/time expert. Use Python and pandas to analyze the detection logs. "
            "Only answer using your knowledge of the date and time and the detection logs."
        )
    instructions = instructions.strip() + "\n" + date_instruction
    assistant = client.beta.assistants.create(
        name="AI Report Assistant",
        instructions=instructions,
        model=model,
        tools=[{"type": "code_interpreter"}],
        file_ids=file_ids
    )
    return assistant

def create_thread(client):
    return client.beta.threads.create()

def send_message_and_get_response(client, thread, assistant, user_input):
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    # Wait for completion
    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run_status.status == "completed":
            break
        time.sleep(1)
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    return messages.data[0].content[0].text.value 