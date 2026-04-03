#!/usr/bin/env python3
"""
Test per-layer KV extraction by patching llama-cpp-python's ctypes bindings.
Adds our custom llama_kv_* functions to the already-loaded library.
"""
import sys
import os
import ctypes
from ctypes import c_int32, c_size_t, c_float, c_void_p, POINTER, byref
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = "/media/project_1/AI/models/phi-4-Q4_K_M.gguf"


def patch_llama_cpp_lib():
    """Add our custom KV layer functions to llama-cpp-python's library."""
    import llama_cpp.llama_cpp as ll
    lib = ll._lib
    
    # Add our custom functions
    lib.llama_kv_n_layers.argtypes = [c_void_p]
    lib.llama_kv_n_layers.restype = c_int32
    
    lib.llama_kv_get_layer_k.argtypes = [
        c_void_p, c_int32, POINTER(c_float), c_size_t, 
        POINTER(c_int32), POINTER(c_int32)
    ]
    lib.llama_kv_get_layer_k.restype = c_size_t
    
    lib.llama_kv_get_layer_v.argtypes = [
        c_void_p, c_int32, POINTER(c_float), c_size_t,
        POINTER(c_int32), POINTER(c_int32)
    ]
    lib.llama_kv_get_layer_v.restype = c_size_t
    
    return lib


def get_layer_k(lib, ctx, layer_id: int) -> np.ndarray | None:
    """Extract K tensor from a specific layer."""
    n_tokens = c_int32()
    n_embd = c_int32()
    
    size = lib.llama_kv_get_layer_k(ctx, layer_id, None, 0, byref(n_tokens), byref(n_embd))
    if size == 0:
        return None
    
    buf = np.zeros(size // 4, dtype=np.float32)
    lib.llama_kv_get_layer_k(
        ctx, layer_id,
        buf.ctypes.data_as(POINTER(c_float)),
        size, byref(n_tokens), byref(n_embd)
    )
    
    return buf.reshape(-1, n_embd.value)[:n_tokens.value]


def main():
    print("=" * 70)
    print("llama-cpp-python + Custom KV Layer API Test")
    print("=" * 70)
    
    # First, patch the library
    print("\n1. Patching llama-cpp-python with custom API...")
    try:
        lib = patch_llama_cpp_lib()
        print("   ✅ Library patched successfully")
    except AttributeError as e:
        print(f"   ❌ Custom functions not found in library: {e}")
        print("   Note: The llama-cpp-python library doesn't have our custom functions.")
        print("   This is expected - we need to replace their libllama.so with ours.")
        return 1
    
    # Load model
    from llama_cpp import Llama
    
    print(f"\n2. Loading model: {MODEL_PATH}")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=512,
        n_gpu_layers=-1,
        verbose=False,
    )
    print("   ✅ Model loaded")
    
    # Process text
    print("\n3. Processing text to populate KV cache...")
    text = "The capital of France is Paris. Berlin is in Germany."
    tokens = llm.tokenize(text.encode())
    print(f"   Text: '{text}'")
    print(f"   Tokens: {len(tokens)}")
    
    llm.eval(tokens)
    print("   ✅ KV cache populated")
    
    # Get context pointer
    print("\n4. Extracting per-layer KV data...")
    ctx = llm._ctx.ctx
    
    n_layers = lib.llama_kv_n_layers(ctx)
    print(f"   KV layers: {n_layers}")
    
    if n_layers > 0:
        # Get layer 0
        k0 = get_layer_k(lib, ctx, 0)
        if k0 is not None:
            print(f"   Layer 0 K shape: {k0.shape}")
            print(f"   Layer 0 K mean: {k0.mean():.4f}, std: {k0.std():.4f}")
        
        # Get last layer
        k_last = get_layer_k(lib, ctx, n_layers - 1)
        if k_last is not None:
            print(f"   Layer {n_layers-1} K shape: {k_last.shape}")
    else:
        print("   ⚠️  No KV layers found")
    
    print("\n" + "=" * 70)
    print("✅ Test Complete!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
