import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("OPENAI_API_KEY")

print(f"API Key found: {'Yes' if api_key else 'No'}")
print(f"API Key starts with: {api_key[:10] if api_key else 'None'}")
print(f"Is placeholder: {'Yes' if api_key == 'your-openai-api-key-here' else 'No'}")

# Test the API
if api_key and api_key != "your-openai-api-key-here":
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        print("API Test: SUCCESS")
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"API Test: FAILED - {e}")
else:
    print("Cannot test API - no valid key found")