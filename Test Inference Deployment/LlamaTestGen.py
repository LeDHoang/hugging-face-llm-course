import requests
import json

# Test chat completion with advanced parameters
def test_chat_completion():
    chat_payload = {
        "messages": [
            {"role": "system", "content": "You are a creative storyteller."},
            {"role": "user", "content": "Write a creative story about space exploration"}
        ],
        "max_tokens": 200,
        "temperature": 0.8,
        "top_p": 0.95,
        "repetition_penalty": 1.1
    }
    
    response = requests.post("http://localhost:8080/v1/chat/completions",
                           json=chat_payload, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        print("=== Chat Completion Response ===")
        print(result['choices'][0]['message']['content'])
        return result
    else:
        print(f"Chat completion error: {response.status_code} - {response.text}")
        return None

# Test completion endpoint with advanced parameters
def test_completion():
    completion_payload = {
        "prompt": "Write a creative story about space exploration",
        "max_tokens": 200,
        "temperature": 0.8,
        "top_p": 0.95,
        "repetition_penalty": 1.1
    }
    
    response = requests.post("http://localhost:8080/v1/completions",
                           json=completion_payload, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        print("\n=== Completion Response ===")
        print(result['choices'][0]['text'])
        return result
    else:
        print(f"Completion error: {response.status_code} - {response.text}")
        return None

# Run both tests
print("Testing llama.cpp server with advanced parameters...")
test_chat_completion()
test_completion()
print("\nAdvanced parameter testing completed!")
