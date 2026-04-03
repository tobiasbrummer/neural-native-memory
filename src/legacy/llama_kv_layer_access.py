"""
ctypes bindings for custom llama.cpp KV layer access functions.

Provides direct access to per-layer K/V tensors from the KV cache,
enabling KV-Embedding extraction and injection.
"""

import ctypes
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

# Default path to the custom-built llama.cpp library
DEFAULT_LIB_PATH = Path(__file__).parent.parent / "llama.cpp" / "build" / "bin" / "libllama.so"


class LlamaKVLayerAccess:
    """
    Wrapper for custom llama.cpp KV layer access functions.

    These functions are added to llama.cpp to enable:
    - Extracting K/V tensors per layer for storage
    - Direct manipulation of KV cache for injection
    """

    def __init__(self, lib_path: Optional[str] = None):
        """
        Initialize the ctypes bindings.

        Args:
            lib_path: Path to libllama.so with custom KV functions.
                      If None, uses the default build path.
        """
        if lib_path is None:
            lib_path = str(DEFAULT_LIB_PATH)

        self.lib_path = lib_path
        self._lib = None
        self._load_library()

    def _load_library(self):
        """Load the shared library and set up function signatures."""
        try:
            self._lib = ctypes.CDLL(self.lib_path)
            logger.info(f"Loaded llama library from {self.lib_path}")
        except OSError as e:
            raise RuntimeError(f"Failed to load llama library from {self.lib_path}: {e}")

        # int32_t llama_kv_n_layers(const struct llama_context * ctx)
        self._lib.llama_kv_n_layers.argtypes = [ctypes.c_void_p]
        self._lib.llama_kv_n_layers.restype = ctypes.c_int32

        # size_t llama_kv_get_layer_k(
        #     struct llama_context * ctx,
        #     int32_t layer_id,
        #     float * dst,
        #     size_t dst_size,
        #     int32_t * n_tokens,
        #     int32_t * n_embd)
        self._lib.llama_kv_get_layer_k.argtypes = [
            ctypes.c_void_p,      # ctx
            ctypes.c_int32,       # layer_id
            ctypes.POINTER(ctypes.c_float),  # dst
            ctypes.c_size_t,      # dst_size
            ctypes.POINTER(ctypes.c_int32),  # n_tokens
            ctypes.POINTER(ctypes.c_int32),  # n_embd
        ]
        self._lib.llama_kv_get_layer_k.restype = ctypes.c_size_t

        # size_t llama_kv_get_layer_v(...)  - same signature
        self._lib.llama_kv_get_layer_v.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        ]
        self._lib.llama_kv_get_layer_v.restype = ctypes.c_size_t

        logger.info("KV layer access functions bound successfully")

    def get_n_layers(self, ctx_ptr: int) -> int:
        """
        Get the number of layers in the KV cache.

        Args:
            ctx_ptr: Pointer to llama_context (as int)

        Returns:
            Number of KV cache layers
        """
        return self._lib.llama_kv_n_layers(ctypes.c_void_p(ctx_ptr))

    def get_layer_k(self, ctx_ptr: int, layer_id: int) -> Tuple[np.ndarray, int, int]:
        """
        Get the K tensor for a specific layer.

        Args:
            ctx_ptr: Pointer to llama_context (as int)
            layer_id: Layer index (0 to n_layers-1)

        Returns:
            Tuple of (k_data, n_tokens, n_embd)
            k_data: numpy array of shape [n_tokens, n_embd]
        """
        n_tokens = ctypes.c_int32()
        n_embd = ctypes.c_int32()

        # First call to get size
        size = self._lib.llama_kv_get_layer_k(
            ctypes.c_void_p(ctx_ptr),
            ctypes.c_int32(layer_id),
            None,
            0,
            ctypes.byref(n_tokens),
            ctypes.byref(n_embd),
        )

        if size == 0:
            raise ValueError(f"Failed to get K tensor for layer {layer_id}")

        # Allocate buffer and get data
        n_floats = size // ctypes.sizeof(ctypes.c_float)
        buffer = (ctypes.c_float * n_floats)()

        self._lib.llama_kv_get_layer_k(
            ctypes.c_void_p(ctx_ptr),
            ctypes.c_int32(layer_id),
            buffer,
            size,
            ctypes.byref(n_tokens),
            ctypes.byref(n_embd),
        )

        # Convert to numpy and reshape
        k_data = np.ctypeslib.as_array(buffer).copy()
        k_data = k_data.reshape(n_tokens.value, n_embd.value)

        return k_data, n_tokens.value, n_embd.value

    def get_layer_v(self, ctx_ptr: int, layer_id: int) -> Tuple[np.ndarray, int, int]:
        """
        Get the V tensor for a specific layer.

        Args:
            ctx_ptr: Pointer to llama_context (as int)
            layer_id: Layer index (0 to n_layers-1)

        Returns:
            Tuple of (v_data, n_tokens, n_embd)
            v_data: numpy array of shape [n_tokens, n_embd]
        """
        n_tokens = ctypes.c_int32()
        n_embd = ctypes.c_int32()

        # First call to get size
        size = self._lib.llama_kv_get_layer_v(
            ctypes.c_void_p(ctx_ptr),
            ctypes.c_int32(layer_id),
            None,
            0,
            ctypes.byref(n_tokens),
            ctypes.byref(n_embd),
        )

        if size == 0:
            raise ValueError(f"Failed to get V tensor for layer {layer_id}")

        # Allocate buffer and get data
        n_floats = size // ctypes.sizeof(ctypes.c_float)
        buffer = (ctypes.c_float * n_floats)()

        self._lib.llama_kv_get_layer_v(
            ctypes.c_void_p(ctx_ptr),
            ctypes.c_int32(layer_id),
            buffer,
            size,
            ctypes.byref(n_tokens),
            ctypes.byref(n_embd),
        )

        # Convert to numpy and reshape
        v_data = np.ctypeslib.as_array(buffer).copy()
        v_data = v_data.reshape(n_tokens.value, n_embd.value)

        return v_data, n_tokens.value, n_embd.value

    def get_all_kv(self, ctx_ptr: int, layers: Optional[list] = None) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Get K and V tensors for multiple layers.

        Args:
            ctx_ptr: Pointer to llama_context (as int)
            layers: List of layer indices, or None for all layers

        Returns:
            Dict mapping layer_id -> (K, V) numpy arrays
        """
        n_layers = self.get_n_layers(ctx_ptr)

        if layers is None:
            layers = list(range(n_layers))

        result = {}
        for layer_id in layers:
            if layer_id < 0 or layer_id >= n_layers:
                logger.warning(f"Skipping invalid layer {layer_id} (n_layers={n_layers})")
                continue

            k_data, _, _ = self.get_layer_k(ctx_ptr, layer_id)
            v_data, _, _ = self.get_layer_v(ctx_ptr, layer_id)
            result[layer_id] = (k_data, v_data)

        return result


def get_context_ptr_from_llama(llm) -> int:
    """
    Extract the raw context pointer from a llama-cpp-python Llama instance.

    Args:
        llm: llama_cpp.Llama instance

    Returns:
        Context pointer as integer
    """
    # llama-cpp-python stores the context in _ctx
    if hasattr(llm, '_ctx') and llm._ctx is not None:
        return llm._ctx.ctx
    elif hasattr(llm, 'ctx'):
        return llm.ctx
    else:
        raise ValueError("Cannot extract context pointer from Llama instance")


# Convenience function for quick testing
def test_kv_access(model_path: str, test_text: str = "Hello, world!"):
    """
    Quick test of KV layer access.

    Args:
        model_path: Path to GGUF model
        test_text: Text to process
    """
    from llama_cpp import Llama

    print(f"Loading model: {model_path}")
    llm = Llama(model_path=model_path, n_ctx=512, n_gpu_layers=0, verbose=False)

    print(f"Processing: {test_text}")
    tokens = llm.tokenize(test_text.encode("utf-8"))
    llm.eval(tokens)

    print("Accessing KV cache...")
    kv_access = LlamaKVLayerAccess()
    ctx_ptr = get_context_ptr_from_llama(llm)

    n_layers = kv_access.get_n_layers(ctx_ptr)
    print(f"KV cache has {n_layers} layers")

    # Get first and last layer
    for layer_id in [0, n_layers - 1]:
        k_data, n_tokens, n_embd = kv_access.get_layer_k(ctx_ptr, layer_id)
        v_data, _, _ = kv_access.get_layer_v(ctx_ptr, layer_id)
        print(f"Layer {layer_id}: K={k_data.shape}, V={v_data.shape}")

    print("Success!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_kv_access(sys.argv[1])
    else:
        print("Usage: python llama_kv_layer_access.py <model.gguf>")
