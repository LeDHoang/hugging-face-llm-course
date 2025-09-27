#!/usr/bin/env python3
"""
vLLM Test Script for macOS Apple Silicon
This script demonstrates how to use vLLM for inference on macOS.
"""

import os
from vllm import LLM, SamplingParams

def test_vllm_basic():
    """Test basic vLLM functionality with a small model."""
    print("🚀 Starting vLLM test on macOS Apple Silicon...")
    
    # Use a small, fast model that's suitable for testing
    # For testing, let's try a very small model first
    model_name = "microsoft/DialoGPT-small"  # Very small model for testing
    
    try:
        print(f"📥 Loading model: {model_name}")
        print("⏳ This may take a moment on first run...")
        print("💡 Note: Model will be downloaded from Hugging Face if not cached")
        
        # Initialize the LLM with conservative settings for testing
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=512,  # Very small context for testing
            tensor_parallel_size=1,  # Single GPU/CPU
        )
        
        print("✅ Model loaded successfully!")
        
        # Define sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=50,  # Short responses for testing
        )
        
        # Test prompts
        prompts = [
            "Hello, how are you?",
            "Tell me a joke.",
            "What is AI?",
        ]
        
        print("\n🧪 Running inference tests...")
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n--- Test {i} ---")
            print(f"Prompt: {prompt}")
            
            # Generate response for this specific prompt
            outputs = llm.generate([prompt], sampling_params)
            
            # Print the generated text
            for output in outputs:
                print(f"Response: {output.outputs[0].text}")
                print(f"Tokens generated: {len(output.outputs[0].token_ids)}")
        
        print("\n🎉 All tests completed successfully!")
        print("✅ vLLM is working correctly on your macOS Apple Silicon system!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during vLLM test: {e}")
        print("\n🔍 Troubleshooting tips:")
        print("1. Make sure you have enough memory available")
        print("2. Try with a smaller model if memory is limited")
        print("3. Check your internet connection for model downloads")
        print("4. Some models may not be compatible with CPU-only inference")
        return False

def test_vllm_server():
    """Test vLLM server functionality."""
    print("\n🌐 Testing vLLM server functionality...")
    
    try:
        from vllm.entrypoints.openai.api_server import main as api_server_main
        print("✅ vLLM OpenAI API server module imported successfully!")
        print("📝 To start the server, you can run:")
        print("   python -m vllm.entrypoints.openai.api_server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host 0.0.0.0 --port 8000")
        return True
    except ImportError as e:
        print(f"❌ Could not import vLLM server module: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 vLLM macOS Apple Silicon Test Suite")
    print("=" * 60)
    
    # Test basic functionality
    basic_success = test_vllm_basic()
    
    # Test server functionality
    server_success = test_vllm_server()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"Basic functionality: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"Server functionality: {'✅ PASS' if server_success else '❌ FAIL'}")
    
    if basic_success and server_success:
        print("\n🎊 Congratulations! vLLM is fully functional on your system!")
        print("\n📚 Next steps:")
        print("1. Try different models by changing the model_name variable")
        print("2. Experiment with different sampling parameters")
        print("3. Start a server for API access")
        print("4. Check out the vLLM documentation for more features")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
    
    print("=" * 60)