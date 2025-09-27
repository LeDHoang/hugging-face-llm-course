from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8000/v1")

# Advanced parameters example
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,
    max_tokens=200,
    top_p=0.95,
)
print(response.choices[0].message.content)

# For direct text generation
response = client.text_generation(
    "Write a creative story about space exploration",
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.95,
    details=True,
)
print(response.generated_text)