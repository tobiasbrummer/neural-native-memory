"""
Python ctypes wrapper for llama.cpp per-layer KV cache API.

This module provides access to the custom llama.cpp extension that
enables per-layer K/V tensor extraction for KV-Embedding implementation.
"""
import ctypes
from ctypes import c_int32, c_size_t, c_float, c_void_p, POINTER, byref
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


# Default path to our custom-built libllama.so
DEFAULT_LIBLLAMA_PATH = "/home/t0bybr/Dokumente/Projekte/kv_llm_vectorstore/llama.cpp/build/bin/libllama.so"


@dataclass
class KVLayerData:
    """Container for K/V tensor data from a specific layer."""
    layer_id: int
    k: np.ndarray  # Shape: [n_tokens, n_embd_k]
    v: np.ndarray  # Shape: [n_tokens, n_embd_v]
    

class LlamaKVLayerAPI:
    """
    Python wrapper for llama.cpp per-layer KV cache API.
    
    Usage:
        api = LlamaKVLayerAPI()
        # Use with llama-cpp-python context
        n_layers = api.get_n_layers(ctx._ctx.ctx)
        k_data = api.get_layer_k(ctx._ctx.ctx, layer_id=0)
    """
    
    def __init__(self, lib_path: str = DEFAULT_LIBLLAMA_PATH):
        """Initialize the API wrapper by loading libllama.so."""
        self.lib = ctypes.CDLL(lib_path)
        self._setup_functions()
    
    def _setup_functions(self):
        """Setup ctypes function signatures."""
        # llama_kv_n_layers
        self.lib.llama_kv_n_layers.argtypes = [c_void_p]
        self.lib.llama_kv_n_layers.restype = c_int32
        
        # llama_kv_get_layer_k
        self.lib.llama_kv_get_layer_k.argtypes = [
            c_void_p,           # ctx
            c_int32,            # layer_id
            POINTER(c_float),   # dst
            c_size_t,           # dst_size
            POINTER(c_int32),   # n_tokens
            POINTER(c_int32),   # n_embd
        ]
        self.lib.llama_kv_get_layer_k.restype = c_size_t
        
        # llama_kv_get_layer_v
        self.lib.llama_kv_get_layer_v.argtypes = [
            c_void_p,           # ctx
            c_int32,            # layer_id
            POINTER(c_float),   # dst
            c_size_t,           # dst_size
            POINTER(c_int32),   # n_tokens
            POINTER(c_int32),   # n_embd
        ]
        self.lib.llama_kv_get_layer_v.restype = c_size_t
    
    def get_n_layers(self, ctx: c_void_p) -> int:
        """Get the number of layers in the KV cache."""
        return self.lib.llama_kv_n_layers(ctx)
    
    def get_layer_k(self, ctx: c_void_p, layer_id: int) -> Optional[np.ndarray]:
        """
        Get K tensor data for a specific layer.
        
        Args:
            ctx: llama_context pointer
            layer_id: Layer index (0 to n_layers-1)
            
        Returns:
            numpy array of shape [n_tokens, n_embd_k] or None if error
        """
        n_tokens = c_int32()
        n_embd = c_int32()
        
        # First call to get size
        size = self.lib.llama_kv_get_layer_k(
            ctx, layer_id, None, 0,
            byref(n_tokens), byref(n_embd)
        )
        
        if size == 0:
            return None
        
        # Allocate buffer and copy data
        n_floats = size // 4
        buf = np.zeros(n_floats, dtype=np.float32)
        
        self.lib.llama_kv_get_layer_k(
            ctx, layer_id,
            buf.ctypes.data_as(POINTER(c_float)),
            size,
            byref(n_tokens), byref(n_embd)
        )
        
        # Reshape to [n_tokens, n_embd] (first stream only for now)
        # Note: Full tensor is [n_embd, kv_size, n_stream]
        return buf.reshape(-1, n_embd.value)[:n_tokens.value]
    
    def get_layer_v(self, ctx: c_void_p, layer_id: int) -> Optional[np.ndarray]:
        """
        Get V tensor data for a specific layer.
        
        Args:
            ctx: llama_context pointer
            layer_id: Layer index (0 to n_layers-1)
            
        Returns:
            numpy array of shape [n_tokens, n_embd_v] or None if error
        """
        n_tokens = c_int32()
        n_embd = c_int32()
        
        # First call to get size
        size = self.lib.llama_kv_get_layer_v(
            ctx, layer_id, None, 0,
            byref(n_tokens), byref(n_embd)
        )
        
        if size == 0:
            return None
        
        # Allocate buffer and copy data
        n_floats = size // 4
        buf = np.zeros(n_floats, dtype=np.float32)
        
        self.lib.llama_kv_get_layer_v(
            ctx, layer_id,
            buf.ctypes.data_as(POINTER(c_float)),
            size,
            byref(n_tokens), byref(n_embd)
        )
        
        return buf.reshape(-1, n_embd.value)[:n_tokens.value]
    
    def get_layer_kv(self, ctx: c_void_p, layer_id: int) -> Optional[KVLayerData]:
        """
        Get both K and V tensors for a specific layer.
        
        Args:
            ctx: llama_context pointer
            layer_id: Layer index
            
        Returns:
            KVLayerData containing both K and V arrays, or None if error
        """
        k = self.get_layer_k(ctx, layer_id)
        if k is None:
            return None
            
        v = self.get_layer_v(ctx, layer_id)
        if v is None:
            return None
            
        return KVLayerData(layer_id=layer_id, k=k, v=v)
    
    def get_all_layers_kv(self, ctx: c_void_p) -> list[KVLayerData]:
        """
        Get K/V data from all layers.
        
        Returns:
            List of KVLayerData for each layer
        """
        n_layers = self.get_n_layers(ctx)
        results = []
        
        for layer_id in range(n_layers):
            data = self.get_layer_kv(ctx, layer_id)
            if data is not None:
                results.append(data)
                
        return results


# Singleton instance for convenience
_api_instance: Optional[LlamaKVLayerAPI] = None

def get_api(lib_path: str = DEFAULT_LIBLLAMA_PATH) -> LlamaKVLayerAPI:
    """Get or create a singleton API instance."""
    global _api_instance
    if _api_instance is None:
        _api_instance = LlamaKVLayerAPI(lib_path)
    return _api_instance
