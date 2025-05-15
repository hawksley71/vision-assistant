import os
from dotenv import load_dotenv
from openai import OpenAI
import time

def cleanup_openai_resources():
    """
    Cleanup utility to delete all OpenAI assistants and files.
    Keeps only the most recent assistant and its associated files.
    """
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_TOKEN")
    
    if not api_key:
        print("Error: OPENAI_TOKEN not found in environment variables")
        return
    
    # Initialize client
    client = OpenAI(
        api_key=api_key,
        default_headers={"OpenAI-Beta": "assistants=v2"}
    )
    
    try:
        # List all assistants
        assistants = client.beta.assistants.list()
        print(f"\nFound {len(assistants.data)} assistants")
        
        # Keep track of the most recent assistant
        most_recent = None
        most_recent_time = 0
        
        # Find the most recent assistant
        for assistant in assistants.data:
            if assistant.name == "AI Report Assistant":
                created_at = int(assistant.created_at.timestamp())
                if created_at > most_recent_time:
                    most_recent_time = created_at
                    most_recent = assistant
        
        if most_recent:
            print(f"\nMost recent assistant ID: {most_recent.id}")
            print(f"Created at: {most_recent.created_at}")
            
            # Delete all other assistants
            for assistant in assistants.data:
                if assistant.id != most_recent.id:
                    print(f"Deleting assistant: {assistant.id}")
                    client.beta.assistants.delete(assistant.id)
                    time.sleep(1)  # Rate limiting
            
            print("\nAll other assistants deleted")
        else:
            print("\nNo 'AI Report Assistant' found")
        
        # List all files
        files = client.files.list()
        print(f"\nFound {len(files.data)} files")
        
        # Delete all files
        for file in files.data:
            print(f"Deleting file: {file.id} ({file.filename})")
            client.files.delete(file.id)
            time.sleep(1)  # Rate limiting
        
        print("\nAll files deleted")
        
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    print("Starting OpenAI resources cleanup...")
    cleanup_openai_resources()
    print("\nCleanup completed!") 