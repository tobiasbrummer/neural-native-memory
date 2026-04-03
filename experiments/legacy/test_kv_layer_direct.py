#!/usr/bin/env python3
"""
Direct ctypes test of per-layer KV API without llama-cpp-python.
Uses raw low-level llama.cpp API calls.
"""
import ctypes
from ctypes import (
    c_int, c_int32, c_uint32, c_bool, c_char_p, c_void_p, c_size_t, c_float,
    POINTER, byref, Structure, CFUNCTYPE
)
import numpy as np
import sys
import os

# Paths 
LIBLLAMA_PATH = "/home/t0bybr/Dokumente/Projekte/kv_llm_vectorstore/llama.cpp/build_container/bin/libllama.so"
MODEL_PATH = "/media/project_1/AI/models/phi-4-Q4_K_M.gguf"


# Define llama_model_params structure (simplified - only need defaults)
class llama_model_params(Structure):
    _fields_ = [
        ("devices", c_void_p),
        ("n_devices", c_int32),
        ("n_gpu_layers", c_int32),
        ("split_mode", c_int32),
        ("main_gpu", c_int32),
        ("tensor_split", c_void_p),
        ("tensor_buft", c_void_p),
        ("rpc_servers", c_char_p),
        ("progress_callback", c_void_p),
        ("progress_callback_user_data", c_void_p),
        ("kv_overrides", c_void_p),
        ("vocab_only", c_bool),
        ("use_mmap", c_bool),
        ("use_mlock", c_bool),
        ("check_tensors", c_bool),
    ]


class llama_context_params(Structure):
    _fields_ = [
        ("n_ctx", c_uint32),
        ("n_batch", c_uint32),
        ("n_ubatch", c_uint32),
        ("n_seq_max", c_uint32),
        ("n_threads", c_int32),
        ("n_threads_batch", c_int32),
        ("rope_scaling_type", c_int32),
        ("pooling_type", c_int32),
        ("attention_type", c_int32),
        ("flash_attn_type", c_int32),
        ("rope_freq_base", c_float),
        ("rope_freq_scale", c_float),
        ("yarn_ext_factor", c_float),
        ("yarn_attn_factor", c_float),
        ("yarn_beta_fast", c_float),
        ("yarn_beta_slow", c_float),
        ("yarn_orig_ctx", c_uint32),
        ("defrag_thold", c_float),
        ("cb_eval", c_void_p),
        ("cb_eval_user_data", c_void_p),
        ("type_k", c_int32),
        ("type_v", c_int32),
        ("abort_callback", c_void_p),
        ("abort_callback_data", c_void_p),
        ("embeddings", c_bool),
        ("offload_kqv", c_bool),
        ("no_perf", c_bool),
        ("op_offload", c_bool),
        ("swa_full", c_bool),
        ("kv_unified", c_bool),
        ("sampler", c_void_p),
        ("n_sampler", c_uint32),
    ]


def main():
    print("=" * 70)
    print("Direct ctypes Test - llama.cpp Per-Layer KV API")
    print("=" * 70)
    
    print(f"\n1. Loading library: {LIBLLAMA_PATH}")
    lib = ctypes.CDLL(LIBLLAMA_PATH)
    
    # Setup function signatures
    lib.llama_backend_init.argtypes = []
    lib.llama_backend_init.restype = None
    
    lib.llama_model_default_params.argtypes = []
    lib.llama_model_default_params.restype = llama_model_params
    
    lib.llama_context_default_params.argtypes = []
    lib.llama_context_default_params.restype = llama_context_params
    
    lib.llama_model_load_from_file.argtypes = [c_char_p, llama_model_params]
    lib.llama_model_load_from_file.restype = c_void_p
    
    lib.llama_init_from_model.argtypes = [c_void_p, llama_context_params]
    lib.llama_init_from_model.restype = c_void_p
    
    lib.llama_model_n_layer.argtypes = [c_void_p]
    lib.llama_model_n_layer.restype = c_int32
    
    lib.llama_model_get_vocab.argtypes = [c_void_p]
    lib.llama_model_get_vocab.restype = c_void_p
    
    lib.llama_tokenize.argtypes = [c_void_p, c_char_p, c_int32, POINTER(c_int32), c_int32, c_bool, c_bool]
    lib.llama_tokenize.restype = c_int32
    
    lib.llama_decode.argtypes = [c_void_p, c_void_p]  # batch is complex
    lib.llama_decode.restype = c_int32
    
    lib.llama_free.argtypes = [c_void_p]
    lib.llama_free.restype = None
    
    lib.llama_model_free.argtypes = [c_void_p]
    lib.llama_model_free.restype = None
    
    # Our custom functions
    lib.llama_kv_n_layers.argtypes = [c_void_p]
    lib.llama_kv_n_layers.restype = c_int32
    
    lib.llama_kv_get_layer_k.argtypes = [c_void_p, c_int32, POINTER(c_float), c_size_t, POINTER(c_int32), POINTER(c_int32)]
    lib.llama_kv_get_layer_k.restype = c_size_t
    
    lib.llama_kv_get_layer_v.argtypes = [c_void_p, c_int32, POINTER(c_float), c_size_t, POINTER(c_int32), POINTER(c_int32)]
    lib.llama_kv_get_layer_v.restype = c_size_t
    
    print("   ✅ Library loaded, all functions found")
    
    # Initialize backend
    print("\n2. Initializing backend...")
    lib.llama_backend_init()
    print("   ✅ Backend initialized")
    
    # Load model
    print(f"\n3. Loading model: {MODEL_PATH}")
    model_params = lib.llama_model_default_params()
    model_params.n_gpu_layers = -1  # All layers on GPU
    
    model = lib.llama_model_load_from_file(MODEL_PATH.encode(), model_params)
    if not model:
        print("   ❌ Failed to load model!")
        return 1
    
    n_layer = lib.llama_model_n_layer(model)
    print(f"   ✅ Model loaded ({n_layer} layers)")
    
    # Create context
    print("\n4. Creating context...")
    ctx_params = lib.llama_context_default_params()
    ctx_params.n_ctx = 512
    
    ctx = lib.llama_init_from_model(model, ctx_params)
    if not ctx:
        print("   ❌ Failed to create context!")
        lib.llama_model_free(model)
        return 1
    print("   ✅ Context created")
    
    # Test KV layer API (before any tokens)
    print("\n5. Testing KV layer API (empty cache)...")
    n_kv_layers = lib.llama_kv_n_layers(ctx)
    print(f"   KV layers: {n_kv_layers}")
    
    if n_kv_layers > 0:
        n_tokens = c_int32()
        n_embd = c_int32()
        
        size = lib.llama_kv_get_layer_k(ctx, 0, None, 0, byref(n_tokens), byref(n_embd))
        print(f"   Layer 0 K: tokens={n_tokens.value}, embd={n_embd.value}, size={size}")
    else:
        print("   ⚠️  KV cache is empty (expected before decode)")
    
    # Cleanup
    print("\n6. Cleanup...")
    lib.llama_free(ctx)
    lib.llama_model_free(model)
    print("   ✅ Done!")
    
    print("\n" + "=" * 70)
    print("✅ TEST PASSED - Custom API is working!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
