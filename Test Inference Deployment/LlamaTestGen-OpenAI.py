from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-no-key-required")

# Advanced parameters example
response = client.chat.completions.create(
    model="smollm2-1.7b-instruct",
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,  # Higher for more creativity
    top_p=0.95,  # Nucleus sampling probability
    frequency_penalty=0.5,  # Reduce repetition of frequent tokens
    presence_penalty=0.5,  # Reduce repetition by penalizing tokens already present
    max_tokens=200,  # Maximum generation length
)
print(response.choices[0].message.content)