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
# Interact with the server using Hugging Face’s InferenceClient:
from huggingface_hub import InferenceClient

# Initialize client pointing to llama.cpp server
client = InferenceClient(
    model="http://localhost:8080/v1",  # URL to the llama.cpp server
    token="sk-no-key-required",  # llama.cpp server requires this placeholder
)

# Text generation
response = client.text_generation(
    "Tell me a story",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.95,
    details=True,
)
print(response.generated_text)

# For chat format
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)