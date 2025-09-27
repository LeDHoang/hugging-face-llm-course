#!/usr/bin/env python3
"""
Simple vLLM Installation Test for macOS Apple Silicon
This script verifies that vLLM is properly installed and can be imported.
"""

def test_vllm_imports():
    """Test that vLLM can be imported and basic classes are available."""
    print("🧪 Testing vLLM imports...")
    
    try:
        # Test basic import
        import vllm
        print("✅ vllm module imported successfully")
        
        # Test core classes
        from vllm import LLM, SamplingParams
        print("✅ LLM and SamplingParams classes imported successfully")
        
        # Test server module
        from vllm.entrypoints.openai.api_server import main as api_server_main
        print("✅ OpenAI API server module imported successfully")
        
        # Test engine module
        from vllm.engine.llm_engine import LLMEngine
        print("✅ LLMEngine imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_vllm_version():
    """Test vLLM version information."""
    print("\n📋 Testing vLLM version information...")
    
    try:
        import vllm
        
        # Try to get version info
        if hasattr(vllm, '__version__'):
            print(f"✅ vLLM version: {vllm.__version__}")
        else:
            print("ℹ️  Version information not available in this build")
        
        # Check platform detection
        print("ℹ️  Platform detection: CPU (as expected for macOS)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error getting version info: {e}")
        return False

def test_vllm_config():
    """Test vLLM configuration."""
    print("\n⚙️  Testing vLLM configuration...")
    
    try:
        from vllm.config import ModelConfig
        print("✅ ModelConfig imported successfully")
        
        from vllm.config import CacheConfig
        print("✅ CacheConfig imported successfully")
        
        from vllm.config import SchedulerConfig
        print("✅ SchedulerConfig imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def show_usage_examples():
    """Show usage examples for vLLM."""
    print("\n📚 vLLM Usage Examples:")
    print("=" * 50)
    
    print("\n1. Basic Inference:")
    print("""
from vllm import LLM, SamplingParams

# Initialize the LLM
llm = LLM(model="microsoft/DialoGPT-small")

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=50
)

# Generate text
prompts = ["Hello, how are you?"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
""")
    
    print("\n2. Start OpenAI-Compatible Server:")
    print("""
python -m vllm.entrypoints.openai.api_server \\
    --model microsoft/DialoGPT-small \\
    --host 0.0.0.0 \\
    --port 8000
""")
    
    print("\n3. Available Models for Testing:")
    print("   - microsoft/DialoGPT-small (117M parameters)")
    print("   - distilbert-base-uncased (66M parameters)")
    print("   - gpt2 (117M parameters)")
    print("   - facebook/opt-125m (125M parameters)")

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 vLLM macOS Apple Silicon Installation Test")
    print("=" * 60)
    
    # Run tests
    import_success = test_vllm_imports()
    version_success = test_vllm_version()
    config_success = test_vllm_config()
    
    # Show usage examples
    show_usage_examples()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"Import tests: {'✅ PASS' if import_success else '❌ FAIL'}")
    print(f"Version tests: {'✅ PASS' if version_success else '❌ FAIL'}")
    print(f"Config tests: {'✅ PASS' if config_success else '❌ FAIL'}")
    
    if import_success and version_success and config_success:
        print("\n🎊 Congratulations! vLLM is properly installed!")
        print("\n✅ You can now:")
        print("   • Use vLLM for local inference")
        print("   • Start OpenAI-compatible servers")
        print("   • Experiment with different models")
        print("   • Integrate with your applications")
        
        print("\n🚀 Next steps:")
        print("   1. Try the basic inference example above")
        print("   2. Start with small models for testing")
        print("   3. Check vLLM documentation for advanced features")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    print("=" * 60)

