from openai import OpenAI

# Initialize client pointing to llama.cpp server
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key-required",  # llama.cpp server requires this placeholder
)

# Chat completion
response = client.chat.completions.create(
    model="smollm2-1.7b-instruct",  # Model identifier can be anything as server only loads one model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)