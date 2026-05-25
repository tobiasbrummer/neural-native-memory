#!/usr/bin/env python3
"""
Test per-layer KV cache extraction using llama-cpp-python + custom library.
Uses llama-cpp-python for model loading, then accesses our custom API.
"""
import sys
import os
import time
import numpy as np

# Add lib to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Model path
MODEL_PATH = "/media/project_1/AI/models/phi-4-Q4_K_M.gguf"
LIBLLAMA_PATH = "/home/t0bybr/Dokumente/Projekte/kv_llm_vectorstore/llama.cpp/build_container/bin/libllama.so"

def main():
    print("=" * 70)
    print("llama.cpp Per-Layer KV Cache Extraction Test")
    print("=" * 70)
    
    # Import llama-cpp-python (it uses its own libllama internally)
    from llama_cpp import Llama
    
    # Load our custom API separately
    from src.legacy.llama_cpp_kv_layer import LlamaKVLayerAPI
    
    print(f"\n1. Loading model with llama-cpp-python...")
    print(f"   Path: {MODEL_PATH}")
    
    t0 = time.time()
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=512,
        n_gpu_layers=-1,
        verbose=False,
    )
    print(f"   ✅ Model loaded in {time.time() - t0:.1f}s")
    
    # Process some text to populate the KV cache
    print("\n2. Processing text to populate KV cache...")
    text = "The quick brown fox jumps over the lazy dog. This is a test sentence."
    
    tokens = llm.tokenize(text.encode())
    print(f"   Text: '{text[:50]}...'")
    print(f"   Tokens: {len(tokens)}")
    
    # Evaluate tokens to fill KV cache
    llm.eval(tokens)
    print("   ✅ Tokens evaluated, KV cache populated")
    
    # Now access our custom API
    print("\n3. Testing per-layer KV API...")
    api = LlamaKVLayerAPI(LIBLLAMA_PATH)
    
    # Get the llama_context pointer from llama-cpp-python
    # The internal structure is: llm._ctx.ctx
    try:
        ctx = llm._ctx.ctx
        print(f"   Context pointer: {ctx}")
    except AttributeError:
        print("   ⚠️ Could not access internal context pointer")
        print("   Trying alternative access method...")
        ctx = None
    
    if ctx is None:
        # Try to use ctypes to get the context from the llama-cpp-python library
        import ctypes
        # The llama-cpp-python uses its own libllama, not ours
        print("\n   Note: llama-cpp-python uses its own libllama.so")
        print("   Our API needs the same library to work correctly.")
        print("\n   Solution: Rebuild llama-cpp-python against our modified llama.cpp")
        return 0
    
    # Test the API
    n_layers = api.get_n_layers(ctx)
    print(f"   KV cache layers: {n_layers}")
    
    if n_layers > 0:
        # Get layer 0 K
        k0 = api.get_layer_k(ctx, 0)
        if k0 is not None:
            print(f"   Layer 0 K shape: {k0.shape}")
            print(f"   Layer 0 K mean: {k0.mean():.6f}, std: {k0.std():.6f}")
        
        # Get layer 0 V
        v0 = api.get_layer_v(ctx, 0)
        if v0 is not None:
            print(f"   Layer 0 V shape: {v0.shape}")
    
    print("\n" + "=" * 70)
    print("✅ Test Complete!")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
