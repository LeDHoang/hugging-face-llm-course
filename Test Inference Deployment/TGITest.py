# from huggingface_hub import InferenceClient
# TGI is easy to install and use, with deep integration into the Hugging Face ecosystem.

# First, launch the TGI server using Docker:

# Copied
# docker run --gpus all \
#     --shm-size 1g \
#     -p 8080:80 \
#     -v ~/.cache/huggingface:/data \
#     ghcr.io/huggingface/text-generation-inference:latest \
#     --model-id HuggingFaceTB/SmolLM2-360M-Instruct
# Then interact with it using Hugging Face’s InferenceClient:
# Initialize client pointing to TGI endpoint
client = InferenceClient(
    model="http://localhost:8080",  # URL to the TGI server
)

# Text generation
response = client.text_generation(
    "Tell me a story",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.95,
    details=True,
    stop_sequences=[],
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