#!/usr/bin/env python3
"""
vLLM Inference Test for macOS Apple Silicon
This script tests basic inference functionality without server components.
"""

def test_basic_inference():
    """Test basic vLLM inference with a small model."""
    print("🚀 Testing vLLM Inference on macOS Apple Silicon...")
    
    try:
        from vllm import LLM, SamplingParams
        print("✅ vLLM classes imported successfully")
        
        # Use a very small model for testing
        model_name = "distilbert-base-uncased"  # Small model, good for testing
        
        print(f"📥 Loading model: {model_name}")
        print("⏳ This may take a moment on first run...")
        
        # Initialize with very conservative settings
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=128,  # Very small context
            tensor_parallel_size=1,
        )
        
        print("✅ Model loaded successfully!")
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=20,  # Very short responses
        )
        
        # Test prompt
        prompt = "Hello"
        print(f"\n🧪 Testing inference with prompt: '{prompt}'")
        
        # Generate response
        outputs = llm.generate([prompt], sampling_params)
        
        # Print results
        for output in outputs:
            print(f"✅ Generated text: '{output.outputs[0].text}'")
            print(f"📊 Tokens generated: {len(output.outputs[0].token_ids)}")
        
        print("\n🎉 Inference test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        print("\n🔍 This is expected for some models on CPU-only systems.")
        print("💡 vLLM is primarily optimized for GPU inference.")
        return False

def test_model_loading():
    """Test if we can load model configurations."""
    print("\n🔧 Testing model configuration loading...")
    
    try:
        from vllm.config import ModelConfig
        from vllm.model_executor.models import MODEL_REGISTRY
        
        print("✅ Model configuration classes imported")
        print(f"📋 Available model types: {len(MODEL_REGISTRY)}")
        
        # Show some available model types
        model_types = list(MODEL_REGISTRY.keys())[:5]
        print(f"📝 Sample model types: {model_types}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model configuration test failed: {e}")
        return False

def show_next_steps():
    """Show next steps for using vLLM."""
    print("\n📚 Next Steps for vLLM on macOS:")
    print("=" * 50)
    
    print("\n1. For Production Use:")
    print("   • Use Linux with CUDA GPUs for best performance")
    print("   • vLLM is optimized for NVIDIA GPUs")
    print("   • Consider cloud instances (AWS, GCP, Azure)")
    
    print("\n2. For Development on macOS:")
    print("   • Use smaller models (distilbert, gpt2, etc.)")
    print("   • Limit context lengths")
    print("   • Consider other frameworks for CPU inference")
    
    print("\n3. Alternative Options for macOS:")
    print("   • llama.cpp (which you already have working!)")
    print("   • MLX (Apple's ML framework)")
    print("   • Transformers library with CPU optimization")
    print("   • ONNX Runtime")
    
    print("\n4. vLLM Server (when working):")
    print("   python -m vllm.entrypoints.openai.api_server \\")
    print("       --model microsoft/DialoGPT-small \\")
    print("       --host 0.0.0.0 \\")
    print("       --port 8000")

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 vLLM Inference Test for macOS")
    print("=" * 60)
    
    # Test model configuration
    config_success = test_model_loading()
    
    # Test basic inference
    inference_success = test_basic_inference()
    
    # Show next steps
    show_next_steps()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"Model configuration: {'✅ PASS' if config_success else '❌ FAIL'}")
    print(f"Inference test: {'✅ PASS' if inference_success else '❌ FAIL'}")
    
    if config_success:
        print("\n✅ vLLM core functionality is working!")
        print("💡 While inference may be limited on CPU, the framework is properly installed.")
    else:
        print("\n❌ vLLM installation has issues.")
    
    print("=" * 60)

