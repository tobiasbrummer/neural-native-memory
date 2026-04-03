"""
Minimal ctypes bindings for llama.cpp.

Direct bindings to our custom-built llama.cpp library,
avoiding llama-cpp-python version conflicts.
"""

import ctypes
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default path to the custom-built llama.cpp library
LIB_DIR = Path(__file__).parent.parent / "llama.cpp" / "build" / "bin"


def _load_libs():
    """Load all required shared libraries in correct order."""
    lib_dir = LIB_DIR

    # Preload CUDA driver library if available so driver API symbols are resolvable
    for candidate in (
        "/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/lib/x86_64-linux-gnu/libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
        "libcuda.so.1",
        "libcuda.so",
    ):
        try:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            logger.debug(f"Preloaded CUDA driver: {candidate}")
            break
        except OSError:
            continue

    # Load dependencies first
    libs = {}
    for name in ["libggml-base.so", "libggml.so", "libggml-cpu.so", "libggml-cuda.so", "libllama.so"]:
        path = lib_dir / name
        if path.exists():
            try:
                libs[name] = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                logger.debug(f"Loaded {name}")
            except OSError as e:
                if "cuda" in name.lower():
                    logger.debug(f"Skipping optional {name}: {e}")
                else:
                    raise

    return libs.get("libllama.so")


# Load the library
_lib = _load_libs()
if _lib is None:
    raise RuntimeError(f"Failed to load libllama.so from {LIB_DIR}")


# ============================================================================
# Type definitions
# ============================================================================

llama_model_p = ctypes.c_void_p
llama_context_p = ctypes.c_void_p
llama_token = ctypes.c_int32


@dataclass
class LlamaModelParams:
    """Simplified model params - we'll use defaults mostly."""
    n_gpu_layers: int = 0
    use_mmap: bool = True
    use_mlock: bool = False


@dataclass
class LlamaContextParams:
    """Simplified context params."""
    n_ctx: int = 512
    n_batch: int = 512
    n_threads: int = 4


# ============================================================================
# Struct definitions for llama.cpp
# ============================================================================

class llama_model_params(ctypes.Structure):
    """llama_model_params struct from llama.h - updated for current API"""
    _fields_ = [
        ("devices", ctypes.c_void_p),
        ("tensor_buft_overrides", ctypes.c_void_p),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int32),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.c_void_p),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        # booleans
        ("vocab_only", ctypes.c_bool),
        ("use_mmap", ctypes.c_bool),
        ("use_direct_io", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
    ]


class llama_context_params(ctypes.Structure):
    """llama_context_params struct from llama.h - updated for current API"""
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int32),
        ("pooling_type", ctypes.c_int32),
        ("attention_type", ctypes.c_int32),
        ("flash_attn_type", ctypes.c_int32),  # NEW
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int32),
        ("type_v", ctypes.c_int32),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        # booleans at end
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),      # NEW
        ("kv_unified", ctypes.c_bool),    # NEW
        ("samplers", ctypes.c_void_p),    # NEW
    ]


class llama_batch(ctypes.Structure):
    """llama_batch struct"""
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(llama_token)),
        ("embd", ctypes.c_void_p),
        ("pos", ctypes.POINTER(ctypes.c_int32)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]


# ============================================================================
# Function bindings
# ============================================================================

# llama_model_default_params
_lib.llama_model_default_params.argtypes = []
_lib.llama_model_default_params.restype = llama_model_params

# llama_context_default_params
_lib.llama_context_default_params.argtypes = []
_lib.llama_context_default_params.restype = llama_context_params

# llama_load_model_from_file
_lib.llama_load_model_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
_lib.llama_load_model_from_file.restype = llama_model_p

# llama_model_free
_lib.llama_model_free.argtypes = [llama_model_p]
_lib.llama_model_free.restype = None

# llama_init_from_model
_lib.llama_init_from_model.argtypes = [llama_model_p, llama_context_params]
_lib.llama_init_from_model.restype = llama_context_p

# llama_free
_lib.llama_free.argtypes = [llama_context_p]
_lib.llama_free.restype = None

# llama_model_get_vocab
llama_vocab_p = ctypes.c_void_p
_lib.llama_model_get_vocab.argtypes = [llama_model_p]
_lib.llama_model_get_vocab.restype = llama_vocab_p

# llama_tokenize (now takes vocab, not model)
_lib.llama_tokenize.argtypes = [
    llama_vocab_p,           # vocab
    ctypes.c_char_p,         # text
    ctypes.c_int32,          # text_len
    ctypes.POINTER(llama_token),  # tokens
    ctypes.c_int32,          # n_tokens_max
    ctypes.c_bool,           # add_special
    ctypes.c_bool,           # parse_special
]
_lib.llama_tokenize.restype = ctypes.c_int32

# llama_decode
_lib.llama_decode.argtypes = [llama_context_p, llama_batch]
_lib.llama_decode.restype = ctypes.c_int32

# llama_batch_init
_lib.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
_lib.llama_batch_init.restype = llama_batch

# llama_backend_init
_lib.llama_backend_init.argtypes = []
_lib.llama_backend_init.restype = None
_backend_initialized = False

# llama_batch_free
_lib.llama_batch_free.argtypes = [llama_batch]
_lib.llama_batch_free.restype = None

# llama_n_ctx
_lib.llama_n_ctx.argtypes = [llama_context_p]
_lib.llama_n_ctx.restype = ctypes.c_uint32

# llama_n_embd
_lib.llama_n_embd.argtypes = [llama_model_p]
_lib.llama_n_embd.restype = ctypes.c_int32

# llama_n_layer
_lib.llama_n_layer.argtypes = [llama_model_p]
_lib.llama_n_layer.restype = ctypes.c_int32

# llama_get_memory
llama_memory_p = ctypes.c_void_p
_lib.llama_get_memory.argtypes = [llama_context_p]
_lib.llama_get_memory.restype = llama_memory_p

# llama_memory_clear
_lib.llama_memory_clear.argtypes = [llama_memory_p, ctypes.c_bool]
_lib.llama_memory_clear.restype = None

# llama_token_to_piece (now takes vocab, not model)
_lib.llama_token_to_piece.argtypes = [
    llama_vocab_p,
    llama_token,
    ctypes.c_char_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_bool,
]
_lib.llama_token_to_piece.restype = ctypes.c_int32

# ============================================================================
# Custom KV layer access functions
# ============================================================================

# llama_kv_n_layers
_lib.llama_kv_n_layers.argtypes = [llama_context_p]
_lib.llama_kv_n_layers.restype = ctypes.c_int32

# llama_kv_has_layer
_lib.llama_kv_has_layer.argtypes = [llama_context_p, ctypes.c_int32]
_lib.llama_kv_has_layer.restype = ctypes.c_bool

# llama_kv_get_layer_k
_lib.llama_kv_get_layer_k.argtypes = [
    llama_context_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
]
_lib.llama_kv_get_layer_k.restype = ctypes.c_size_t

# llama_kv_get_layer_v
_lib.llama_kv_get_layer_v.argtypes = [
    llama_context_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
]
_lib.llama_kv_get_layer_v.restype = ctypes.c_size_t

# llama_kv_set_layer_k
_lib.llama_kv_set_layer_k.argtypes = [
    llama_context_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_int32,
]
_lib.llama_kv_set_layer_k.restype = ctypes.c_bool

# llama_kv_set_layer_v
_lib.llama_kv_set_layer_v.argtypes = [
    llama_context_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_int32,
]
_lib.llama_kv_set_layer_v.restype = ctypes.c_bool

# llama_kv_prefix_set
_lib.llama_kv_prefix_set.argtypes = [
    llama_context_p,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_size_t,
    ctypes.c_int32,
]
_lib.llama_kv_prefix_set.restype = None

# llama_kv_prefix_clear
_lib.llama_kv_prefix_clear.argtypes = [llama_context_p]
_lib.llama_kv_prefix_clear.restype = None

# ============================================================================
# Embedding matrix access
# ============================================================================

# llama_get_embed_matrix
_lib.llama_get_embed_matrix.argtypes = [
    llama_model_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
]
_lib.llama_get_embed_matrix.restype = ctypes.c_size_t

# ============================================================================
# Hidden states extraction
# ============================================================================

llama_hidden_states_p = ctypes.c_void_p

# llama_hidden_states_init
_lib.llama_hidden_states_init.argtypes = [
    llama_context_p,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32,
]
_lib.llama_hidden_states_init.restype = llama_hidden_states_p

# llama_hidden_states_free
_lib.llama_hidden_states_free.argtypes = [llama_hidden_states_p]
_lib.llama_hidden_states_free.restype = None

# llama_hidden_states_get
_lib.llama_hidden_states_get.argtypes = [
    llama_hidden_states_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
]
_lib.llama_hidden_states_get.restype = ctypes.c_size_t

# llama_hidden_states_enable_capture
_lib.llama_hidden_states_enable_capture.argtypes = [
    llama_context_p,
    llama_hidden_states_p,
]
_lib.llama_hidden_states_enable_capture.restype = ctypes.c_bool

# llama_hidden_states_disable_capture
_lib.llama_hidden_states_disable_capture.argtypes = [llama_context_p]
_lib.llama_hidden_states_disable_capture.restype = None


# ============================================================================
# High-level wrapper class
# ============================================================================

class LlamaModel:
    """
    High-level wrapper for llama.cpp model and context.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        n_threads: int = 4,
    ):
        """
        Load a model and create a context.

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context size
            n_gpu_layers: Number of layers to offload to GPU
            n_threads: Number of CPU threads
        """
        self.model_path = model_path
        self._model = None
        self._ctx = None

        # Initialize backend once
        global _backend_initialized
        if not _backend_initialized:
            _lib.llama_backend_init()
            _backend_initialized = True

        # Load model
        model_params = _lib.llama_model_default_params()
        model_params.n_gpu_layers = n_gpu_layers

        logger.info(f"Loading model: {model_path}")
        self._model = _lib.llama_load_model_from_file(
            model_path.encode("utf-8"),
            model_params
        )
        if not self._model:
            raise RuntimeError(f"Failed to load model: {model_path}")

        # Create context
        ctx_params = _lib.llama_context_default_params()
        ctx_params.n_ctx = n_ctx
        ctx_params.n_batch = min(n_ctx, 512)
        ctx_params.n_threads = n_threads
        ctx_params.n_threads_batch = n_threads

        self._ctx = _lib.llama_init_from_model(self._model, ctx_params)
        if not self._ctx:
            _lib.llama_model_free(self._model)
            raise RuntimeError("Failed to create context")

        self.n_ctx = n_ctx
        self.n_embd = _lib.llama_n_embd(self._model)
        self.n_layer = _lib.llama_n_layer(self._model)
        self._vocab = _lib.llama_model_get_vocab(self._model)

        logger.info(f"Model loaded: n_embd={self.n_embd}, n_layer={self.n_layer}, n_ctx={n_ctx}")

    def __del__(self):
        """Clean up resources."""
        if self._ctx:
            _lib.llama_free(self._ctx)
        if self._model:
            _lib.llama_model_free(self._model)

    def tokenize(self, text: str, add_bos: bool = True) -> List[int]:
        """
        Tokenize text.

        Args:
            text: Text to tokenize
            add_bos: Add beginning-of-sequence token

        Returns:
            List of token IDs
        """
        text_bytes = text.encode("utf-8")
        max_tokens = len(text_bytes) + 16
        tokens = (llama_token * max_tokens)()

        n_tokens = _lib.llama_tokenize(
            self._vocab,  # Now uses vocab instead of model
            text_bytes,
            len(text_bytes),
            tokens,
            max_tokens,
            add_bos,
            False,  # parse_special
        )

        if n_tokens < 0:
            raise RuntimeError(f"Tokenization failed: {n_tokens}")

        return list(tokens[:n_tokens])

    def detokenize(self, tokens: List[int]) -> str:
        """
        Convert tokens back to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        result = []
        buf = ctypes.create_string_buffer(256)

        for token in tokens:
            n = _lib.llama_token_to_piece(
                self._vocab,  # Now uses vocab instead of model
                token,
                buf,
                256,
                0,
                False,
            )
            if n > 0:
                result.append(buf.value[:n].decode("utf-8", errors="replace"))

        return "".join(result)

    def eval(self, tokens: List[int], pos_offset: int = 0, seq_id: int = 0) -> None:
        """
        Evaluate tokens to fill KV cache.

        Args:
            tokens: Token IDs to process
            pos_offset: Position offset for the tokens
            seq_id: Sequence ID
        """
        n_tokens = len(tokens)
        batch = _lib.llama_batch_init(n_tokens, 0, 1)

        try:
            # Fill batch
            batch.n_tokens = n_tokens
            for i, token in enumerate(tokens):
                batch.token[i] = token
                batch.pos[i] = pos_offset + i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = seq_id
                batch.logits[i] = 0

            # Last token needs logits for generation
            batch.logits[n_tokens - 1] = 1

            # Decode
            ret = _lib.llama_decode(self._ctx, batch)
            if ret != 0:
                raise RuntimeError(f"llama_decode failed: {ret}")

        finally:
            _lib.llama_batch_free(batch)

    def reset(self, clear_data: bool = True) -> None:
        """
        Clear the KV cache / memory state for the context.

        Args:
            clear_data: If True, also zero the underlying KV buffers.
        """
        mem = _lib.llama_get_memory(self._ctx)
        if mem:
            _lib.llama_memory_clear(mem, clear_data)

    # ========================================================================
    # KV Cache access
    # ========================================================================

    def kv_n_layers(self) -> int:
        """Get number of KV cache layers."""
        return _lib.llama_kv_n_layers(self._ctx)

    def kv_has_layer(self, layer_id: int) -> bool:
        """Return True if a model layer has a KV cache."""
        return bool(_lib.llama_kv_has_layer(self._ctx, layer_id))

    def kv_get_layer_k(self, layer_id: int) -> Tuple[np.ndarray, int, int]:
        """
        Get K tensor for a layer.

        Returns:
            (k_data, n_tokens, n_embd)
        """
        n_tokens = ctypes.c_int32()
        n_embd = ctypes.c_int32()

        # Get size first (returns bytes)
        size = _lib.llama_kv_get_layer_k(
            self._ctx, layer_id, None, 0,
            ctypes.byref(n_tokens), ctypes.byref(n_embd)
        )

        if size == 0 and n_embd.value == 0:
            raise ValueError(f"Failed to get K for layer {layer_id}")

        # Allocate float buffer and get data (API returns float32)
        if size == 0:
            data = np.empty((0, n_embd.value), dtype=np.float32)
            return data, 0, n_embd.value

        n_floats = size // ctypes.sizeof(ctypes.c_float)
        buf = (ctypes.c_float * n_floats)()

        _lib.llama_kv_get_layer_k(
            self._ctx, layer_id, buf, size,
            ctypes.byref(n_tokens), ctypes.byref(n_embd)
        )

        data = np.ctypeslib.as_array(buf).copy()
        data = data.reshape(n_tokens.value, n_embd.value)
        return data, n_tokens.value, n_embd.value

    def kv_get_layer_v(self, layer_id: int) -> Tuple[np.ndarray, int, int]:
        """
        Get V tensor for a layer.

        Returns:
            (v_data, n_tokens, n_embd)
        """
        n_tokens = ctypes.c_int32()
        n_embd = ctypes.c_int32()

        # Get size first (returns bytes)
        size = _lib.llama_kv_get_layer_v(
            self._ctx, layer_id, None, 0,
            ctypes.byref(n_tokens), ctypes.byref(n_embd)
        )

        if size == 0 and n_embd.value == 0:
            raise ValueError(f"Failed to get V for layer {layer_id}")

        # Allocate float buffer and get data (API returns float32)
        if size == 0:
            data = np.empty((0, n_embd.value), dtype=np.float32)
            return data, 0, n_embd.value

        n_floats = size // ctypes.sizeof(ctypes.c_float)
        buf = (ctypes.c_float * n_floats)()

        _lib.llama_kv_get_layer_v(
            self._ctx, layer_id, buf, size,
            ctypes.byref(n_tokens), ctypes.byref(n_embd)
        )

        data = np.ctypeslib.as_array(buf).copy()
        data = data.reshape(n_tokens.value, n_embd.value)
        return data, n_tokens.value, n_embd.value

    def kv_get_all(self, layers: Optional[List[int]] = None) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Get K and V for multiple layers.

        Args:
            layers: Layer indices, or None for all

        Returns:
            Dict mapping layer_id -> (K, V) arrays
        """
        n_layers = self.kv_n_layers()

        if layers is None:
            layers = list(range(n_layers))

        result = {}
        for layer_id in layers:
            if 0 <= layer_id < n_layers:
                k, _, _ = self.kv_get_layer_k(layer_id)
                v, _, _ = self.kv_get_layer_v(layer_id)
                result[layer_id] = (k, v)

        return result

    # ========================================================================
    # KV Cache Injection
    # ========================================================================

    def kv_set_layer_k(self, layer_id: int, data: np.ndarray) -> bool:
        """
        Set K tensor for a layer.

        Args:
            layer_id: Layer index
            data: K data as numpy array, shape [n_tokens, n_embd]

        Returns:
            True on success
        """
        data = np.ascontiguousarray(data, dtype=np.float32)
        n_tokens, n_embd = data.shape

        return _lib.llama_kv_set_layer_k(
            self._ctx,
            layer_id,
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_tokens,
            n_embd
        )

    def kv_set_layer_v(self, layer_id: int, data: np.ndarray) -> bool:
        """
        Set V tensor for a layer.

        Args:
            layer_id: Layer index
            data: V data as numpy array, shape [n_tokens, n_embd]

        Returns:
            True on success
        """
        data = np.ascontiguousarray(data, dtype=np.float32)
        n_tokens, n_embd = data.shape

        return _lib.llama_kv_set_layer_v(
            self._ctx,
            layer_id,
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_tokens,
            n_embd
        )

    def kv_set_all(self, kv_data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> bool:
        """
        Set K and V for multiple layers.

        Args:
            kv_data: Dict mapping layer_id -> (K, V) arrays

        Returns:
            True if all succeeded
        """
        success = True
        for layer_id, (k, v) in kv_data.items():
            if not self.kv_set_layer_k(layer_id, k):
                success = False
            if not self.kv_set_layer_v(layer_id, v):
                success = False
        return success

    def kv_prefix_set(self, layer_ids: Optional[List[int]], prefix_len: int) -> None:
        """
        Configure layer-specific prefix masking for KV injection.

        Args:
            layer_ids: Layers to enable prefix for, or None for all layers
            prefix_len: Number of prefix tokens (from position 0)
        """
        if layer_ids:
            arr = (ctypes.c_int32 * len(layer_ids))(*layer_ids)
            n_layers = len(layer_ids)
        else:
            arr = None
            n_layers = 0

        _lib.llama_kv_prefix_set(self._ctx, arr, n_layers, int(prefix_len))

    def kv_prefix_clear(self) -> None:
        """Clear prefix masking (no virtual prefix)."""
        _lib.llama_kv_prefix_clear(self._ctx)

    # ========================================================================
    # Embedding Matrix Access
    # ========================================================================

    def get_embed_matrix(self) -> Tuple[np.ndarray, int, int]:
        """
        Get the static token embedding matrix.

        Returns:
            (embed_matrix, n_vocab, n_embd)
            embed_matrix: shape [n_vocab, n_embd]
        """
        n_vocab = ctypes.c_int32()
        n_embd = ctypes.c_int32()

        # Get size first
        size = _lib.llama_get_embed_matrix(
            self._model, None, 0,
            ctypes.byref(n_vocab), ctypes.byref(n_embd)
        )

        if size == 0:
            raise ValueError("Failed to get embedding matrix")

        # Allocate and get data
        n_floats = size // ctypes.sizeof(ctypes.c_float)
        buf = (ctypes.c_float * n_floats)()

        _lib.llama_get_embed_matrix(
            self._model, buf, size,
            ctypes.byref(n_vocab), ctypes.byref(n_embd)
        )

        data = np.ctypeslib.as_array(buf).copy()
        data = data.reshape(n_vocab.value, n_embd.value)

        return data, n_vocab.value, n_embd.value

    # ========================================================================
    # Static Embedding Lookup
    # ========================================================================

    def get_static_embeddings(self, token_ids: List[int]) -> np.ndarray:
        """
        Get static embeddings for token IDs (lookup from embedding matrix).

        Args:
            token_ids: List of token IDs

        Returns:
            Array of shape [n_tokens, n_embd]
        """
        embed_matrix, _, _ = self.get_embed_matrix()
        return embed_matrix[token_ids]


# ============================================================================
# Hidden States Extractor (context manager)
# ============================================================================

class HiddenStatesExtractor:
    """
    Context manager for extracting hidden states during forward pass.

    Usage:
        with HiddenStatesExtractor(model, layers=[10, 20, 30]) as hs:
            model.eval(tokens)
            states = hs.get_all()
    """

    def __init__(self, model: LlamaModel, layers: Optional[List[int]] = None):
        """
        Initialize hidden states extractor.

        Args:
            model: LlamaModel instance
            layers: Layer indices to capture, or None for all
        """
        self._model = model
        self._layers = layers
        self._hs = None

    def __enter__(self):
        # Create the hidden states struct
        if self._layers:
            layer_arr = (ctypes.c_int32 * len(self._layers))(*self._layers)
            self._hs = _lib.llama_hidden_states_init(
                self._model._ctx, layer_arr, len(self._layers)
            )
        else:
            self._hs = _lib.llama_hidden_states_init(
                self._model._ctx, None, 0
            )

        if not self._hs:
            raise RuntimeError("Failed to create hidden states extractor")

        # Enable capture on the scheduler
        if not _lib.llama_hidden_states_enable_capture(self._model._ctx, self._hs):
            _lib.llama_hidden_states_free(self._hs)
            self._hs = None
            raise RuntimeError("Failed to enable hidden states capture")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._hs:
            # Disable capture first
            _lib.llama_hidden_states_disable_capture(self._model._ctx)
            # Free the struct
            _lib.llama_hidden_states_free(self._hs)
            self._hs = None
        return False

    def get_layer(self, layer_id: int) -> Tuple[np.ndarray, int, int]:
        """
        Get hidden state for a specific layer.

        Returns:
            (hidden_state, n_tokens, n_embd)
        """
        if not self._hs:
            raise RuntimeError("Extractor not active")

        n_tokens = ctypes.c_int32()
        n_embd = ctypes.c_int32()

        # Get size
        size = _lib.llama_hidden_states_get(
            self._hs, layer_id, None, 0,
            ctypes.byref(n_tokens), ctypes.byref(n_embd)
        )

        if size == 0:
            raise ValueError(f"No hidden state for layer {layer_id}")

        # Get data
        n_floats = size // ctypes.sizeof(ctypes.c_float)
        buf = (ctypes.c_float * n_floats)()

        _lib.llama_hidden_states_get(
            self._hs, layer_id, buf, size,
            ctypes.byref(n_tokens), ctypes.byref(n_embd)
        )

        data = np.ctypeslib.as_array(buf).copy()
        data = data.reshape(n_tokens.value, n_embd.value)

        return data, n_tokens.value, n_embd.value

    def get_all(self) -> Dict[int, np.ndarray]:
        """
        Get all captured hidden states.

        Returns:
            Dict mapping layer_id -> hidden_state array
        """
        if not self._hs:
            raise RuntimeError("Extractor not active")

        result = {}
        layers = self._layers if self._layers else list(range(self._model.n_layer))

        for layer_id in layers:
            try:
                state, _, _ = self.get_layer(layer_id)
                result[layer_id] = state
            except ValueError:
                pass  # Layer not captured

        return result


# ============================================================================
# Quick test
# ============================================================================

def test(model_path: str):
    """Quick test of the bindings."""
    print(f"Loading: {model_path}")
    model = LlamaModel(model_path, n_ctx=512, n_gpu_layers=20)

    text = "The capital of France is Paris."
    print(f"Tokenizing: {text}")
    tokens = model.tokenize(text)
    print(f"Tokens: {tokens}")

    print("Evaluating...")
    model.eval(tokens)

    print(f"KV layers: {model.kv_n_layers()}")

    if model.kv_n_layers() > 0:
        k, n_tok, n_embd = model.kv_get_layer_k(0)
        v, _, _ = model.kv_get_layer_v(0)
        print(f"Layer 0: K={k.shape}, V={v.shape}")
        print(f"K[0,:5]: {k[0,:5]}")
        print("SUCCESS!")
    else:
        print("ERROR: No KV layers")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test(sys.argv[1])
    else:
        print("Usage: python llama_raw.py <model.gguf>")
