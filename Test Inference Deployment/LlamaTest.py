# llama.cpp is easy to install and use, requiring minimal dependencies and supporting both CPU and GPU inference.

# First, install and build llama.cpp:

# Copied
# # Clone the repository
# git clone https://github.com/ggerganov/llama.cpp
# cd llama.cpp

# # Build the project
# make

# # Download the SmolLM2-1.7B-Instruct-GGUF model
# curl -L -O https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/resolve/main/smollm2-1.7b-instruct.Q4_K_M.gguf
# Then, launch the server (with OpenAI API compatibility):

# Copied
# # Start the server
# ./server \
#     -m smollm2-1.7b-instruct.Q4_K_M.gguf \
#     --host 0.0.0.0 \
#     --port 8080 \
#     -c 4096 \
#     --n-gpu-layers 0  # Set to a higher number to use GPU
# Let's test with direct HTTP requests first
import requests
import json

# Test basic connectivity
try:
    response = requests.get("http://localhost:8080/health", timeout=5)
    print(f"Server health: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()}")
except Exception as e:
    print(f"Health check failed: {e}")

# Test models endpoint
try:
    response = requests.get("http://localhost:8080/v1/models", timeout=5)
    print(f"Models endpoint: {response.status_code}")
    if response.status_code == 200:
        models = response.json()
        print(f"Available models: {[model.get('id', 'unknown') for model in models.get('data', [])]}")
except Exception as e:
    print(f"Models check failed: {e}")

# Test completion endpoint
try:
    payload = {
        "prompt": "Tell me a story",
        "max_tokens": 50,
        "temperature": 0.7
    }
    response = requests.post("http://localhost:8080/v1/completions",
                           json=payload, timeout=30)
    print(f"Completion endpoint: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Generated text: {result['choices'][0]['text']}")
    else:
        print(f"Error response: {response.text}")
except Exception as e:
    print(f"Completion test failed: {e}")

# Test chat completion endpoint
try:
    chat_payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a short story"}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }
    response = requests.post("http://localhost:8080/v1/chat/completions",
                           json=chat_payload, timeout=30)
    print(f"Chat completion endpoint: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Chat response: {result['choices'][0]['message']['content']}")
    else:
        print(f"Error response: {response.text}")
except Exception as e:
    print(f"Chat completion test failed: {e}")

print("\nLlama.cpp server test completed!")