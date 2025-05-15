import os
from dotenv import load_dotenv
from openai import OpenAI

def test_api_key():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_TOKEN")
    
    print("\n=== API Key Diagnostics ===")
    print(f"API Key loaded: {api_key[:8]}...{api_key[-4:] if api_key else 'None'}")
    print(f"API Key length: {len(api_key) if api_key else 0}")
    print(f"API Key contains whitespace: {bool(api_key and any(c.isspace() for c in api_key))}")
    print(f"API Key contains newlines: {bool(api_key and (chr(10) in api_key or chr(13) in api_key))}")
    contains_quotes = False
    if api_key:
        contains_quotes = ('"' in api_key) or ("'" in api_key)
    print(f"API Key contains quotes: {contains_quotes}")
    print(f"API Key contains non-printable chars: {bool(api_key and any(not c.isprintable() for c in api_key))}")
    
    if not api_key:
        print("\nError: No API key found in environment variables")
        return
    
    # Initialize client with different configurations
    print("\n=== Testing API Key with Different Configurations ===")
    
    # Test 1: Basic client
    print("\nTest 1: Basic client")
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=5
        )
        print("✓ Basic client test successful")
    except Exception as e:
        print(f"✗ Basic client test failed: {str(e)}")
    
    # Test 2: Client with beta headers
    print("\nTest 2: Client with beta headers")
    try:
        client = OpenAI(
            api_key=api_key,
            default_headers={"OpenAI-Beta": "assistants=v2"}
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=5
        )
        print("✓ Beta headers test successful")
    except Exception as e:
        print(f"✗ Beta headers test failed: {str(e)}")
    
    # Test 3: Client with organization
    print("\nTest 3: Client with organization")
    try:
        client = OpenAI(
            api_key=api_key,
            organization="org-..."  # You'll need to add your org ID here
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=5
        )
        print("✓ Organization test successful")
    except Exception as e:
        print(f"✗ Organization test failed: {str(e)}")

if __name__ == "__main__":
    test_api_key() 