from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Advanced parameters example
response = client.chat.completions.create(
    model="tinyllama-test.gguf",
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,
    top_p=0.95,
    max_tokens=200,
)
print(response.choices[0].message.content)