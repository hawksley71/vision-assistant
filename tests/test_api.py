import os
from dotenv import load_dotenv
from openai import OpenAI

def test_api_key():
    """Test that the OpenAI API key is properly configured."""
    api_key = os.getenv("OPENAI_TOKEN")
    
    print("\n=== API Key Diagnostics ===")
    print(f"API Key present: {'Yes' if api_key else 'No'}")
    print(f"API Key length: {len(api_key) if api_key else 0}")
    
    if not api_key:
        print("\nError: No API key found in environment variables")
        return False
    
    print("\n=== Testing API Key with Different Configurations ===")
    
    try:
        # Test with default configuration
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        print("✓ Default configuration successful")
        
        # Test with explicit API key
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            api_key=api_key,
            max_tokens=5
        )
        print("✓ Explicit API key successful")
        
        # Test with environment variable
        os.environ["OPENAI_API_KEY"] = api_key
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        print("✓ Environment variable successful")
        
        return True
        
    except Exception as e:
        print(f"Error testing API key: {str(e)}")
        return False

if __name__ == "__main__":
    test_api_key() 