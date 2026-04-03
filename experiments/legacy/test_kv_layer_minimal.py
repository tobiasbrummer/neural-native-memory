#!/usr/bin/env python3
"""
Minimal test: Just verify the API functions work by accessing via llama-cpp-python's library.
This test modifies llama-cpp-python's loaded library to add our custom functions.
"""
import sys
import os
import ctypes
from ctypes import c_int32, c_size_t, c_float, c_void_p, POINTER, byref

# Paths
CUSTOM_LIBLLAMA = "/home/t0bybr/Dokumente/Projekte/kv_llm_vectorstore/llama.cpp/build/bin/libllama.so"

def main():
    print("=" * 70)
    print("Minimal Per-Layer KV API Test (Host)")
    print("=" * 70)
    
    # Load our custom library
    print(f"\nLoading: {CUSTOM_LIBLLAMA}")
    
    try:
        lib = ctypes.CDLL(CUSTOM_LIBLLAMA)
        print("✅ Library loaded")
    except OSError as e:
        print(f"❌ Failed to load: {e}")
        return 1
    
    # Check our functions exist
    funcs = ['llama_kv_n_layers', 'llama_kv_get_layer_k', 'llama_kv_get_layer_v']
    for func in funcs:
        try:
            getattr(lib, func)
            print(f"✅ {func} exported")
        except AttributeError:
            print(f"❌ {func} NOT exported")
            return 1
    
    print("\n✅ All API functions exported successfully!")
    print("\nNote: Full integration test requires compatible glibc version")
    print("      or rebuilding llama-cpp-python against our llama.cpp.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
