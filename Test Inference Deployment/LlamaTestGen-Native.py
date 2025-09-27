# Using llama-cpp-python package for direct model access
from llama_cpp import Llama

# Load the model
llm = Llama(
    model_path="../llama.cpp/tinyllama-test.gguf",
    n_ctx=4096,  # Context window size
    n_threads=8,  # CPU threads
    n_gpu_layers=0,  # GPU layers (0 = CPU only)
)

# Format prompt according to the model's expected format
prompt = """<|im_start|>system
You are a creative storyteller.
<|im_end|>
<|im_start|>user
Write a creative story
<|im_end|>
<|im_start|>assistant
"""

# Generate response with precise parameter control
output = llm(
    prompt,
    max_tokens=200,
    temperature=0.8,
    top_p=0.95,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    stop=["\n\n"],
)

print(output["choices"][0]["text"])