"""
Virtual Prefix Injection for LLMs.

Core concept: Store Pre-RoPE Hidden States, JIT convert to K/V pairs,
inject as virtual prefix into model cache.

Based on insights from:
- KV-Embedding paper (arXiv:2601.01046v1)
- Rotary Position Embedding mechanics
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model_loader import load_model

logger = logging.getLogger(__name__)


@dataclass
class VirtualPrefixData:
    """
    Container for stored virtual prefix data.

    Attributes:
        hidden_states: Dict mapping layer_idx -> [seq_len, hidden_dim] arrays
        token_ids: Token IDs of the sequence [seq_len]
        text: Original text
        metadata: Additional info (model, timestamp, etc.)
    """
    hidden_states: Dict[int, np.ndarray]
    token_ids: np.ndarray
    text: str
    metadata: Dict[str, any]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "hidden_states": {str(k): v.tolist() for k, v in self.hidden_states.items()},
            "token_ids": self.token_ids.tolist(),
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "VirtualPrefixData":
        """Create from dictionary."""
        hidden_states = {
            int(k): np.array(v, dtype=np.float32)
            for k, v in data.get("hidden_states", {}).items()
        }
        return cls(
            hidden_states=hidden_states,
            token_ids=np.array(data["token_ids"], dtype=np.int64),
            text=data["text"],
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Phase 1: Hidden State Extraction (Pre-RoPE)
# =============================================================================

class PreRopeExtractor:
    """
    Extract Pre-RoPE hidden states using forward hooks.

    The key insight: We need hidden states BEFORE they enter the
    self-attention mechanism (i.e., before RoPE is applied).

    For Qwen/Llama architectures:
    - Input to self_attn layer = Pre-RoPE hidden state
    - This is what we need to store
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        target_layers: Optional[List[int]] = None,
    ):
        """
        Initialize the Pre-RoPE extractor.

        Args:
            model: The language model
            tokenizer: The tokenizer
            target_layers: Layer indices to extract from (auto-selected if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.target_layers = target_layers
        self._extracted_states: Dict[int, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._extract_all: bool = False
        self._all_layers: List[int] = []

    def _get_model_layers(self) -> Tuple[List, str]:
        """
        Get the list of transformer layers from the model.

        Handles various architectures:
        - Standard CausalLM: model.model.layers or model.layers
        - Qwen3-VL: model.model.layers (text model inside)
        - Llama/Qwen2: model.model.layers

        Returns:
            Tuple of (layers list, path_description)
        """
        # Try standard paths
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers, "model.model.layers"
        elif hasattr(self.model, "layers"):
            return self.model.layers, "model.layers"
        # Qwen3-VL specific: nested structure
        elif hasattr(self.model, "model") and hasattr(self.model.model, "model"):
            if hasattr(self.model.model.model, "layers"):
                return self.model.model.model.layers, "model.model.model.layers"
        # Try language_model attribute (some VL models)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
            if hasattr(self.model.model.language_model, "layers"):
                return self.model.model.language_model.layers, "model.model.language_model.layers"
        elif hasattr(self.model, "language_model"):
            if hasattr(self.model.language_model, "layers"):
                return self.model.language_model.layers, "model.language_model.layers"

        raise ValueError(f"Cannot find model layers. Model type: {type(self.model)}")

    def _register_hooks(self) -> None:
        """Register forward hooks to capture Pre-RoPE hidden states."""
        self._extracted_states.clear()
        self._hooks.clear()

        # Access the model layers (handles Qwen, Llama, Qwen3-VL, etc.)
        layers, path = self._get_model_layers()
        logger.info(f"Using layer path: {path}")

        if self.target_layers is None:
            # Default: use middle layers (will be overridden by KV-Embedding selection)
            n_layers = len(layers)
            self.target_layers = list(range(n_layers // 2, n_layers))
        
        # For full cache extraction, use all layers
        layers_to_hook = self._all_layers if self._extract_all else self.target_layers

        for layer_idx in layers_to_hook:
            if layer_idx >= len(layers):
                logger.warning(f"Layer {layer_idx} out of range, skipping")
                continue

            layer = layers[layer_idx]

            # Register hook on the DECODER LAYER (not self_attn)
            # The decoder layer receives hidden_states as first positional arg
            def make_hook(idx):
                def hook(module, args, kwargs):
                    # Decoder layer receives: (hidden_states, ...) as positional args
                    # args is a tuple where args[0] is hidden_states

                    if args and len(args) > 0:
                        hidden_state = args[0]
                        if isinstance(hidden_state, torch.Tensor):
                            # This is the Pre-RoPE hidden state!
                            # Shape: [batch, seq_len, hidden_dim]
                            self._extracted_states[idx] = hidden_state.clone()
                            logger.info(f"  Captured layer {idx}: shape {hidden_state.shape}")
                    elif "hidden_states" in kwargs:
                        hidden_state = kwargs["hidden_states"]
                        if isinstance(hidden_state, torch.Tensor):
                            self._extracted_states[idx] = hidden_state.clone()
                            logger.info(f"  Captured layer {idx} from kwargs: shape {hidden_state.shape}")
                    else:
                        logger.debug(f"  Layer {idx}: no hidden_states found")

                    # Return BOTH args and kwargs unchanged
                    return args, kwargs
                return hook

            # Register on the decoder layer, not self_attn
            handle = layer.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True)
            self._hooks.append(handle)

        logger.info(f"Registered hooks for layers: {self.target_layers}")

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    @torch.no_grad()
    def extract(
        self,
        text: str,
        use_prompt: bool = True,
        prompt_template: str = '"{context}" Compress the Context in one word:',
        extract_all_layers: bool = False,
    ) -> VirtualPrefixData:
        """
        Extract Pre-RoPE hidden states for a text.

        Args:
            text: Input text
            use_prompt: Whether to wrap with compression prompt
            prompt_template: Prompt template
            extract_all_layers: If True, extract from ALL layers (for cache injection)

        Returns:
            VirtualPrefixData with extracted states
        """
        # Prepare input
        if use_prompt:
            processed_text = prompt_template.format(context=text)
        else:
            processed_text = text

        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Set extraction mode
        self._extract_all = extract_all_layers
        if extract_all_layers:
            layers, _ = self._get_model_layers()
            self._all_layers = list(range(len(layers)))

        # Register hooks for this extraction
        self._register_hooks()

        try:
            # Forward pass (hooks will capture Pre-RoPE states)
            _ = self.model(**inputs, output_hidden_states=True)
        finally:
            # Always remove hooks
            self._remove_hooks()

        # Convert captured states to numpy
        hidden_states_np = {}
        state_np = None
        for layer_idx, state_tensor in self._extracted_states.items():
            # Remove batch dim: [1, seq_len, hidden] -> [seq_len, hidden]
            state_np = state_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            hidden_states_np[layer_idx] = state_np
            logger.debug(f"  Captured layer {layer_idx}: shape {state_np.shape}")

        if not hidden_states_np:
            raise ValueError(
                f"No hidden states captured! Target layers: {self.target_layers}. "
                f"The hook may not be receiving the expected input format."
            )

        token_ids = inputs["input_ids"].squeeze(0).cpu().numpy()

        metadata = {
            "model": getattr(self.model.config, "name_or_path", "unknown"),
            "target_layers": self.target_layers,
            "seq_len": len(token_ids),
            "hidden_dim": state_np.shape[-1] if state_np is not None else 0,
        }

        return VirtualPrefixData(
            hidden_states=hidden_states_np,
            token_ids=token_ids,
            text=text,
            metadata=metadata,
        )


def extract_virtual_prefix(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    target_layers: Optional[List[int]] = None,
    extract_all_layers: bool = False,
) -> VirtualPrefixData:
    """
    Convenience function to extract virtual prefix data.

    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text
        target_layers: Layer indices for embedding selection (ignored if extract_all_layers=True)
        extract_all_layers: If True, extract from ALL layers (needed for KV cache injection)

    Returns:
        VirtualPrefixData with Pre-RoPE hidden states
    """
    extractor = PreRopeExtractor(model, tokenizer, target_layers)
    return extractor.extract(text, extract_all_layers=extract_all_layers)


# =============================================================================
# Phase 2: JIT Projection (H -> K, V)
# =============================================================================

def _get_model_layer(model, layer_idx: int):
    """
    Get a specific transformer layer from the model.

    Handles various architectures like Qwen, Llama, Qwen3-VL, etc.

    Args:
        model: The language model
        layer_idx: Layer index

    Returns:
        The transformer layer
    """
    # Try standard paths
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    elif hasattr(model, "layers"):
        return model.layers[layer_idx]
    # Qwen3-VL specific: nested structure
    elif hasattr(model, "model") and hasattr(model.model, "model"):
        if hasattr(model.model.model, "layers"):
            return model.model.model.layers[layer_idx]
    # Try language_model attribute (some VL models)
    elif hasattr(model, "model") and hasattr(model.model, "language_model"):
        if hasattr(model.model.language_model, "layers"):
            return model.model.language_model.layers[layer_idx]
    elif hasattr(model, "language_model"):
        if hasattr(model.language_model, "layers"):
            return model.language_model.layers[layer_idx]

    raise ValueError(f"Cannot access layer {layer_idx}. Model type: {type(model)}")


def project_hidden_to_kv(
    hidden_states: np.ndarray,
    layer_idx: int,
    model: AutoModelForCausalLM,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project hidden states to Keys and Values using model weights.

    This is the "JIT compilation" step: we take the stored hidden states
    and run them through k_proj and v_proj to get Keys and Values.

    Args:
        hidden_states: Hidden states [seq_len, hidden_dim]
        layer_idx: Layer to use for projection
        model: The language model

    Returns:
        Tuple of (keys, values)
        - keys: [seq_len, num_kv_heads, head_dim]
        - values: [seq_len, num_kv_heads, head_dim]
    """
    # Get the model dtype from first parameter
    model_dtype = next(model.parameters()).dtype

    # Convert to torch with model's dtype
    h_tensor = torch.from_numpy(hidden_states).to(model.device).to(model_dtype)  # [seq_len, hidden]
    h_tensor = h_tensor.unsqueeze(0)  # Add batch dim: [1, seq_len, hidden]

    # Get the layer (handles various architectures)
    layer = _get_model_layer(model, layer_idx)
    attn = layer.self_attn
    seq_len = hidden_states.shape[0]
    hidden_size = h_tensor.shape[-1]

    # Project to K and V
    with torch.no_grad():
        # Apply input_layernorm before projection — the stored hidden states
        # are captured at the decoder layer input (pre-norm), but k_proj/v_proj
        # expect post-norm input (as in the model's internal forward pass).
        if hasattr(layer, "input_layernorm"):
            h_tensor = layer.input_layernorm(h_tensor)
        elif hasattr(layer, "ln_1"):
            h_tensor = layer.ln_1(h_tensor)
        else:
            logger.warning(f"Layer {layer_idx}: No input_layernorm found, projecting raw hidden states")

        k_tensor = attn.k_proj(h_tensor)  # [1, seq_len, hidden]
        v_tensor = attn.v_proj(h_tensor)  # [1, seq_len, hidden]

        # Get attention configuration for reshaping
        # Try to get num_kv_heads (key-value heads for GQA)
        if hasattr(attn, "num_key_value_heads"):
            num_kv_heads = attn.num_key_value_heads
        elif hasattr(attn, "kv_heads"):
            num_kv_heads = attn.kv_heads
        elif hasattr(attn.config, "num_key_value_heads"):
            num_kv_heads = attn.config.num_key_value_heads
        else:
            # Infer from tensor shape and known configs
            # For Qwen3-VL: hidden_size=4096, num_kv_heads=8, head_dim=128
            for kv_h in [1, 2, 4, 8, 16, 32, 64]:
                if hidden_size % kv_h == 0:
                    head_dim = hidden_size // kv_h
                    if head_dim in [64, 80, 96, 100, 128, 256]:
                        num_kv_heads = kv_h
                        break
            else:
                num_kv_heads = 1
                head_dim = hidden_size

        # Get head_dim
        if hasattr(attn, "head_dim"):
            head_dim = attn.head_dim
        elif hasattr(attn.config, "head_dim"):
            head_dim = attn.config.head_dim
        else:
            head_dim = hidden_size // num_kv_heads

        # Reshape for multi-head attention
        # K: [1, seq_len, num_kv_heads, head_dim]
        # V: [1, seq_len, num_kv_heads, head_dim]
        k_tensor = k_tensor.view(1, seq_len, num_kv_heads, head_dim)
        v_tensor = v_tensor.view(1, seq_len, num_kv_heads, head_dim)

    # Convert back to numpy and remove batch dim
    keys = k_tensor.squeeze(0).cpu().numpy().astype(np.float32)  # [seq_len, num_kv_heads, head_dim]
    values = v_tensor.squeeze(0).cpu().numpy().astype(np.float32)

    return keys, values


# =============================================================================
# Phase 3: RoPE Engine
# =============================================================================

def rotate_half_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input (torch version).

    Matches transformers' rotate_half exactly.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_torch(
    keys: torch.Tensor,
    positions: torch.Tensor,
    cos_cached: torch.Tensor,
    sin_cached: torch.Tensor,
) -> torch.Tensor:
    """
    Apply Rotary Position Embedding to keys using torch.

    This follows the transformers implementation of apply_rotary_pos_emb:
        K_embed = (K * cos) + (rotate_half(K) * sin)

    Args:
        keys: Keys [seq_len, num_heads, head_dim]
        positions: Position IDs [seq_len]
        cos_cached: Pre-computed cos values [max_pos, rope_dim]
        sin_cached: Pre-computed sin values [max_pos, rope_dim]

    Returns:
        Rotated keys [seq_len, num_heads, head_dim]
    """
    seq_len, num_heads, head_dim = keys.shape
    rope_dim = cos_cached.shape[-1]

    # Gather cos/sin for our positions
    cos = cos_cached[positions]  # [seq_len, rope_dim]
    sin = sin_cached[positions]  # [seq_len, rope_dim]

    # Reshape for broadcasting: [1, num_heads, seq_len, rope_dim]
    cos = cos.view(1, 1, seq_len, rope_dim).expand(1, num_heads, -1, -1)
    sin = sin.view(1, 1, seq_len, rope_dim).expand(1, num_heads, -1, -1)

    # Reshape keys to [batch, num_heads, seq_len, head_dim]
    keys_reshaped = keys.transpose(0, 1).unsqueeze(0)  # [1, num_heads, seq_len, head_dim]

    # Extract rotary portion (first rope_dim elements)
    keys_rot = keys_reshaped[..., :rope_dim]  # [1, num_heads, seq_len, rope_dim]
    keys_pass = keys_reshaped[..., rope_dim:]  # [1, num_heads, seq_len, head_dim - rope_dim]

    # Apply RoPE: k_rotated = (k * cos) + (rotate_half(k) * sin)
    keys_rot_embed = (keys_rot * cos) + (rotate_half_torch(keys_rot) * sin)

    # Concatenate rotary and non-rotary parts
    keys_embed = torch.cat([keys_rot_embed, keys_pass], dim=-1)

    # Remove batch dim and transpose back: [seq_len, num_heads, head_dim]
    return keys_embed[0].transpose(0, 1)


def get_rope_cache(
    model: AutoModelForCausalLM,
    layer_idx: int,
    max_pos: int = 2048,
    head_dim: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get pre-computed RoPE cos/sin values for a layer.

    For Qwen3-VL with mRoPE: Uses text-only mode (1D positions, no interleaving).

    Args:
        model: The language model
        layer_idx: Layer index
        max_pos: Maximum position to compute
        head_dim: Override head_dim (for Qwen3-VL with GQA)

    Returns:
        Tuple of (cos, sin) tensors [max_pos, rope_dim]
        where rope_dim = head_dim for full RoPE
    """
    device = next(model.parameters()).device

    with torch.no_grad():
        # Create 1D position indices for text-only mode
        positions = torch.arange(max_pos, device=device)

        # Try to get rotary_emb from the model's language model
        rotary_emb = None
        inv_freq = None
        attention_scaling = 1.0
        rope_dim = None

        # Try multiple paths to find rotary_emb
        # Path 1: Qwen3-VL: model.model.language_model.rotary_emb
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            rotary_emb = model.model.language_model.rotary_emb
        # Path 2: Standard: model.model.rotary_emb
        elif hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
            rotary_emb = model.model.rotary_emb
        # Path 3: Direct: model.rotary_emb
        elif hasattr(model, "rotary_emb"):
            rotary_emb = model.rotary_emb
        # Path 4: Try through layer
        else:
            layer = _get_model_layer(model, layer_idx)
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "rotary_emb"):
                rotary_emb = layer.self_attn.rotary_emb

        if rotary_emb is not None and hasattr(rotary_emb, "inv_freq"):
            inv_freq = rotary_emb.inv_freq
            if hasattr(rotary_emb, "attention_scaling"):
                attention_scaling = rotary_emb.attention_scaling

            # For Qwen3-VL, inv_freq has shape [head_dim // 2]
            # We need full head_dim cos/sin for apply_rotary_pos_emb
            rope_dim = inv_freq.shape[0] * 2  # Full rotary dimension

        # Fallback: compute standard inv_freq
        if inv_freq is None:
            if head_dim is None:
                # Try to get head_dim from config
                layer = _get_model_layer(model, layer_idx)
                attn = layer.self_attn
                if hasattr(attn, "head_dim"):
                    head_dim = attn.head_dim
                elif hasattr(attn, "config") and hasattr(attn.config, "head_dim"):
                    head_dim = attn.config.head_dim
                else:
                    head_dim = 128
            rope_dim = head_dim
            dim = rope_dim // 2
            rope_theta = 500000.0  # Standard default
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, device=device).float() / dim))

        # Compute angles: positions * inv_freq
        # positions: [max_pos], inv_freq: [rope_dim // 2]
        # angles: [max_pos, rope_dim // 2]
        angles = positions.float().unsqueeze(-1) @ inv_freq.unsqueeze(0)

        # Duplicate to get full dimension (like transformers: emb = cat((freqs, freqs), dim=-1))
        angles = torch.cat([angles, angles], dim=-1)  # [max_pos, rope_dim]

        # Compute cos and sin with attention_scaling
        cos = torch.cos(angles) * attention_scaling
        sin = torch.sin(angles) * attention_scaling

    return cos, sin


def apply_rope_to_keys(
    keys: np.ndarray,
    positions: np.ndarray,
    model: AutoModelForCausalLM,
    layer_idx: int,
) -> np.ndarray:
    """
    Apply RoPE to keys for specific positions.

    Args:
        keys: Keys [seq_len, num_heads, head_dim]
        positions: Position IDs [seq_len] (e.g., [0, 1, 2, ...])
        model: The language model
        layer_idx: Layer index

    Returns:
        Rotated keys [seq_len, num_heads, head_dim]
    """
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    seq_len = keys.shape[0]
    max_pos = int(max(positions)) + 1
    head_dim = keys.shape[-1]  # Use actual key dimension

    # Get RoPE cache (tensors on device)
    cos, sin = get_rope_cache(model, layer_idx, max_pos, head_dim)

    # Convert to torch tensors
    keys_torch = torch.from_numpy(keys).to(device).to(model_dtype)  # [seq_len, num_heads, head_dim]
    positions_torch = torch.from_numpy(positions).to(device)  # [seq_len]

    # Apply RoPE
    keys_rotated = apply_rope_torch(keys_torch, positions_torch, cos, sin)

    # Convert back to numpy
    return keys_rotated.cpu().numpy().astype(np.float32)


# =============================================================================
# Phase 4: Cache Injection
# =============================================================================

def prepare_virtual_prefix_kv(
    prefix_data: VirtualPrefixData,
    model: AutoModelForCausalLM,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Prepare K/V pairs from stored virtual prefix data.

    This is the full JIT pipeline:
    1. Load hidden states
    2. Project to K, V for each layer
    3. Apply RoPE for positions 0..N

    Args:
        prefix_data: Stored virtual prefix data
        model: The language model

    Returns:
        Tuple of (keys_per_layer, values_per_layer)
        - keys_per_layer: {layer_idx: [seq_len, num_heads, head_dim]}
        - values_per_layer: {layer_idx: [seq_len, num_heads, head_dim]}
    """
    keys_per_layer = {}
    values_per_layer = {}

    seq_len = prefix_data.metadata.get("seq_len", 0)

    # Virtual positions: 0 to seq_len-1
    positions = np.arange(seq_len, dtype=np.int64)

    for layer_idx in prefix_data.hidden_states.keys():
        # Get hidden states for this layer
        hidden = prefix_data.hidden_states[layer_idx]  # [seq_len, hidden]

        # Project to K, V
        keys, values = project_hidden_to_kv(hidden, layer_idx, model)

        # Apply RoPE to keys (text-only mode, no mRoPE interleaving)
        keys_rotated = apply_rope_to_keys(keys, positions, model, layer_idx)
        keys_per_layer[layer_idx] = keys_rotated
        values_per_layer[layer_idx] = values  # Values don't get RoPE

    logger.info(f"Prepared virtual prefix K/V for {len(keys_per_layer)} layers, seq_len={seq_len}")

    return keys_per_layer, values_per_layer


def inject_virtual_prefix(
    model: AutoModelForCausalLM,
    keys_per_layer: Dict[int, np.ndarray],
    values_per_layer: Dict[int, np.ndarray],
) -> "DynamicCache":
    """
    Inject virtual prefix K/V pairs into a new cache.

    This creates a fresh cache populated with the virtual prefix.
    Subsequent generation will attend to this prefix.

    Args:
        model: The language model
        keys_per_layer: Keys per layer {layer_idx: [seq_len, num_heads, head_dim]}
        values_per_layer: Values per layer {layer_idx: [seq_len, num_heads, head_dim]}

    Returns:
        DynamicCache populated with virtual prefix
    """
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        raise ImportError("DynamicCache not available. Update transformers.")

    cache = DynamicCache()
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    for layer_idx, keys_np in keys_per_layer.items():
        values_np = values_per_layer[layer_idx]

        # Convert to torch and add batch dimension
        # [seq_len, num_heads, head_dim] -> [1, num_heads, seq_len, head_dim]
        keys_tensor = torch.from_numpy(keys_np).to(device).to(model_dtype)  # [seq_len, num_heads, head_dim]
        values_tensor = torch.from_numpy(values_np).to(device).to(model_dtype)

        # Transpose for expected cache format: [batch, num_heads, seq_len, head_dim]
        keys_tensor = keys_tensor.transpose(0, 1).unsqueeze(0)  # [1, num_heads, seq_len, head_dim]
        values_tensor = values_tensor.transpose(0, 1).unsqueeze(0)

        with torch.no_grad():
            cache.update(keys_tensor, values_tensor, layer_idx)

    logger.info(f"Injected virtual prefix into cache, {len(cache)} layers populated")

    return cache


def generate_with_virtual_prefix(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix_data: VirtualPrefixData,
    query: str,
    max_new_tokens: int = 100,
    **generation_kwargs,
) -> str:
    device = next(model.parameters()).device
    
    # Step 1: Prepare K/V from prefix
    keys_per_layer, values_per_layer = prepare_virtual_prefix_kv(prefix_data, model)
    
    # Step 2: Create cache with prefix
    cache = inject_virtual_prefix(model, keys_per_layer, values_per_layer)
    
    # Step 3: Prepare query inputs
    query_inputs = tokenizer(query, return_tensors="pt")
    query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
    
    # Präfix-Länge aus prefix_data (Anzahl gespeicherter Tokens)
    prefix_len = len(prefix_data.token_ids)
    query_len = query_inputs["input_ids"].shape[1]
    
    # Position IDs - Handle mRoPE for Qwen3-VL
    position_ids = torch.arange(
        prefix_len,
        prefix_len + query_len,
        device=device
    )
    
    # Check if model uses mRoPE (Qwen3-VL)
    if hasattr(model, 'config') and getattr(model.config, 'rope_scaling', None):
        rope_type = model.config.rope_scaling.get('type', '') if isinstance(model.config.rope_scaling, dict) else ''
        if 'mrope' in rope_type.lower() or 'qwen3_vl' in type(model).__name__.lower():
            # mRoPE needs [3, seq_len] format
            position_ids = position_ids.unsqueeze(0).expand(3, -1)  # [3, query_len]
            logger.debug(f"Using mRoPE position_ids: {position_ids.shape}")
        else:
            position_ids = position_ids.unsqueeze(0)  # [1, query_len]
    else:
        position_ids = position_ids.unsqueeze(0)  # [1, query_len]
    
    query_inputs["position_ids"] = position_ids
    
    # Cache position für HuggingFace generate
    cache_position = torch.arange(
        prefix_len, 
        prefix_len + query_len, 
        device=device
    )
    query_inputs["cache_position"] = cache_position
    
    # Attention mask muss den Präfix einschließen
    attention_mask = torch.ones(1, prefix_len + query_len, device=device)
    query_inputs["attention_mask"] = attention_mask
    
    # DEBUG: Print shapes
    print(f"DEBUG: cache type: {type(cache)}")
    print(f"DEBUG: input_ids shape: {query_inputs['input_ids'].shape}")
    print(f"DEBUG: position_ids shape: {query_inputs['position_ids'].shape}")
    print(f"DEBUG: attention_mask shape: {query_inputs['attention_mask'].shape}")
    print(f"DEBUG: cache_position shape: {query_inputs['cache_position'].shape}")
    print(f"DEBUG: prefix_len: {prefix_len}, query_len: {query_len}")
    
    # Step 4: Generate
    with torch.no_grad():
        outputs = model.generate(
            **query_inputs,
            past_key_values=cache,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            **generation_kwargs,
        )
    
    generated_ids = outputs[0][query_inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text


# =============================================================================
# Phase 5: Storage I/O
# =============================================================================

def save_virtual_prefix(
    prefix_data: VirtualPrefixData,
    filepath: str,
) -> None:
    """
    Save virtual prefix data to .npz file.

    Args:
        prefix_data: Data to save
        filepath: Output file path (will add .npz if not present)
    """
    import json
    from pathlib import Path

    filepath = Path(filepath)
    if filepath.suffix != ".npz":
        filepath = filepath.with_suffix(".npz")

    # Prepare arrays for npz
    arrays = {}
    for layer_idx, hidden in prefix_data.hidden_states.items():
        arrays[f"hidden_{layer_idx}"] = hidden

    arrays["token_ids"] = prefix_data.token_ids

    # Save metadata as JSON string
    metadata_json = json.dumps({
        "text": prefix_data.text,
        "metadata": prefix_data.metadata,
    })
    arrays["metadata_json"] = np.array([metadata_json], dtype="S")

    np.savez_compressed(filepath, **arrays)
    logger.info(f"Saved virtual prefix to {filepath}")


def load_virtual_prefix(
    filepath: str,
) -> VirtualPrefixData:
    """
    Load virtual prefix data from .npz file.

    Args:
        filepath: Input file path

    Returns:
        VirtualPrefixData
    """
    import json

    data = np.load(filepath, allow_pickle=True)

    # Load hidden states
    hidden_states = {}
    for key in data.files:
        if key.startswith("hidden_"):
            layer_idx = int(key.split("_")[1])
            hidden_states[layer_idx] = data[key].astype(np.float32)

    # Load token IDs
    token_ids = data["token_ids"].astype(np.int64)

    # Load metadata
    metadata_json = bytes(data["metadata_json"][0]).decode("utf-8")
    metadata_dict = json.loads(metadata_json)

    return VirtualPrefixData(
        hidden_states=hidden_states,
        token_ids=token_ids,
        text=metadata_dict["text"],
        metadata=metadata_dict["metadata"],
    )


# =============================================================================
# Phase 6: Direct KV-Cache Storage (Post-RoPE, correct by construction)
# =============================================================================

@dataclass
class StoredKVCache:
    """
    Container for a stored KV cache (post-RoPE, from model forward pass).

    This stores the actual KV cache that the model produced — RoPE is already
    applied, so injection is just loading into a DynamicCache.

    Attributes:
        keys: Dict mapping layer_idx -> [num_kv_heads, seq_len, head_dim] (float16)
        values: Dict mapping layer_idx -> [num_kv_heads, seq_len, head_dim] (float16)
        token_ids: Token IDs of the input sequence
        text: Original memory text
        prefix_len: Number of tokens in the prefix (for position offset)
        metadata: Additional info (model, prompt used, etc.)
    """
    keys: Dict[int, np.ndarray]
    values: Dict[int, np.ndarray]
    token_ids: np.ndarray
    text: str
    prefix_len: int
    metadata: Dict[str, any]


@torch.no_grad()
def extract_kv_cache(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    use_prompt: bool = True,
    prompt_template: str = '"{context}" Compress the Context in one word:',
    max_length: int = 8192,
) -> StoredKVCache:
    """
    Extract the KV cache from a model forward pass.

    This stores the post-RoPE KV cache — correct by construction, no manual
    RoPE application needed at injection time.

    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Memory text to encode
        use_prompt: Whether to use compress prompt (for bidirectional context)
        prompt_template: Prompt template for compression
        max_length: Maximum token length (default 8192, Qwen3-VL supports 32768)

    Returns:
        StoredKVCache with post-RoPE K/V for all layers
    """
    device = next(model.parameters()).device

    # Prepare input (with or without compress prompt)
    if use_prompt:
        input_text = prompt_template.format(context=text)
    else:
        input_text = text

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logger.info(f"Tokenized input: {inputs['input_ids'].shape[1]} tokens (max_length={max_length})")

    # Forward pass — model applies RoPE internally
    outputs = model(**inputs, use_cache=True)
    kv_cache = outputs.past_key_values

    # Extract K/V tensors per layer
    if hasattr(kv_cache, 'key_cache'):
        num_layers = len(kv_cache.key_cache)
        get_kv = lambda idx: (kv_cache.key_cache[idx], kv_cache.value_cache[idx])
    elif hasattr(kv_cache, '__getitem__'):
        num_layers = len(kv_cache)
        get_kv = lambda idx: kv_cache[idx]
    else:
        raise ValueError(f"Unknown cache format: {type(kv_cache)}")

    keys_dict = {}
    values_dict = {}
    for layer_idx in range(num_layers):
        k, v = get_kv(layer_idx)
        # Shape: [batch, num_kv_heads, seq_len, head_dim] -> remove batch dim
        keys_dict[layer_idx] = k.squeeze(0).cpu().numpy().astype(np.float16)
        values_dict[layer_idx] = v.squeeze(0).cpu().numpy().astype(np.float16)

    token_ids = inputs["input_ids"].squeeze(0).cpu().numpy()
    prefix_len = len(token_ids)

    logger.info(f"Extracted KV cache: {num_layers} layers, prefix_len={prefix_len}")

    return StoredKVCache(
        keys=keys_dict,
        values=values_dict,
        token_ids=token_ids,
        text=text,
        prefix_len=prefix_len,
        metadata={
            "model": getattr(model.config, "name_or_path", "unknown"),
            "num_layers": num_layers,
            "use_prompt": use_prompt,
            "prompt_template": prompt_template if use_prompt else None,
            "kv_shape": list(keys_dict[0].shape) if keys_dict else [],
        },
    )


def generate_with_stored_kv(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    stored_kv: StoredKVCache,
    query: str,
    max_new_tokens: int = 100,
    **generation_kwargs,
) -> str:
    """
    Generate text using a stored KV cache as prefix.

    Injects the stored post-RoPE KV cache into a DynamicCache and generates.
    No manual RoPE or projection needed — the cache is correct by construction.

    Args:
        model: The language model (must be same architecture as extraction)
        tokenizer: The tokenizer
        stored_kv: Previously extracted KV cache
        query: Query text to generate from
        max_new_tokens: Maximum tokens to generate
        **generation_kwargs: Additional generation parameters

    Returns:
        Generated text (query response)
    """
    from transformers.cache_utils import DynamicCache

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    # Step 1: Rebuild DynamicCache from stored data
    prefix_cache = DynamicCache()
    for layer_idx in sorted(stored_kv.keys.keys()):
        k = torch.from_numpy(stored_kv.keys[layer_idx]).to(device).to(model_dtype)
        v = torch.from_numpy(stored_kv.values[layer_idx]).to(device).to(model_dtype)
        # Add batch dim: [num_kv_heads, seq_len, head_dim] -> [1, num_kv_heads, seq_len, head_dim]
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        prefix_cache.update(k, v, layer_idx)

    prefix_len = stored_kv.prefix_len

    # Step 2: Prepare query inputs
    query_inputs = tokenizer(query, return_tensors="pt")
    query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
    query_len = query_inputs["input_ids"].shape[1]

    # Position IDs — handle mRoPE for Qwen3-VL
    position_ids = torch.arange(
        prefix_len,
        prefix_len + query_len,
        device=device,
    )

    if hasattr(model, 'config') and getattr(model.config, 'rope_scaling', None):
        rope_type = model.config.rope_scaling.get('type', '') if isinstance(model.config.rope_scaling, dict) else ''
        if 'mrope' in rope_type.lower() or 'qwen3_vl' in type(model).__name__.lower():
            position_ids = position_ids.unsqueeze(0).expand(3, -1)
        else:
            position_ids = position_ids.unsqueeze(0)
    else:
        position_ids = position_ids.unsqueeze(0)

    query_inputs["position_ids"] = position_ids

    cache_position = torch.arange(prefix_len, prefix_len + query_len, device=device)
    query_inputs["cache_position"] = cache_position

    attention_mask = torch.ones(1, prefix_len + query_len, device=device)
    query_inputs["attention_mask"] = attention_mask

    # Step 3: Generate
    logger.info(f"Generating with stored KV prefix (prefix_len={prefix_len}, query_len={query_len})")
    with torch.no_grad():
        outputs = model.generate(
            **query_inputs,
            past_key_values=prefix_cache,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            **generation_kwargs,
        )

    generated_ids = outputs[0][query_inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def save_kv_cache(stored_kv: StoredKVCache, filepath: str) -> None:
    """Save a StoredKVCache to .npz file."""
    import json
    from pathlib import Path

    filepath = Path(filepath)
    if filepath.suffix != ".npz":
        filepath = filepath.with_suffix(".npz")

    arrays = {}
    for layer_idx in stored_kv.keys:
        arrays[f"k_{layer_idx}"] = stored_kv.keys[layer_idx]
        arrays[f"v_{layer_idx}"] = stored_kv.values[layer_idx]

    arrays["token_ids"] = stored_kv.token_ids

    metadata_json = json.dumps({
        "text": stored_kv.text,
        "prefix_len": stored_kv.prefix_len,
        "metadata": stored_kv.metadata,
    })
    arrays["metadata_json"] = np.array([metadata_json], dtype="S")

    np.savez_compressed(filepath, **arrays)
    logger.info(f"Saved KV cache to {filepath}")


def load_kv_cache(filepath: str) -> StoredKVCache:
    """Load a StoredKVCache from .npz file."""
    import json

    data = np.load(filepath, allow_pickle=True)

    keys = {}
    values = {}
    for key in data.files:
        if key.startswith("k_"):
            layer_idx = int(key[2:])
            keys[layer_idx] = data[key].astype(np.float16)
        elif key.startswith("v_"):
            layer_idx = int(key[2:])
            values[layer_idx] = data[key].astype(np.float16)

    token_ids = data["token_ids"].astype(np.int64)
    metadata_json = bytes(data["metadata_json"][0]).decode("utf-8")
    metadata_dict = json.loads(metadata_json)

    return StoredKVCache(
        keys=keys,
        values=values,
        token_ids=token_ids,
        text=metadata_dict["text"],
        prefix_len=metadata_dict["prefix_len"],
        metadata=metadata_dict.get("metadata", {}),
    )


# =============================================================================
# Paper-Based KV Re-routing for Generation
# =============================================================================
def generate_with_kv_rerouting(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    memory_text: str,
    query: str,
    target_layers: List[int],
    max_new_tokens: int = 100,
    use_prompt: bool = True,
    prompt_template: str = '"{context}" Compress the Context in one word:',
    **generation_kwargs,
) -> str:
    from transformers.cache_utils import DynamicCache
    
    device = next(model.parameters()).device
    
    # Prepare memory text
    if use_prompt:
        memory_prompted = prompt_template.format(context=memory_text)
    else:
        memory_prompted = memory_text
    
    # Tokenize memory
    memory_inputs = tokenizer(
        memory_prompted,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    memory_inputs = {k: v.to(device) for k, v in memory_inputs.items()}
    
    # Pass 1: Extract KV states from memory
    logger.info("Pass 1: Extracting KV states from memory...")
    with torch.no_grad():
        memory_outputs = model(**memory_inputs, output_hidden_states=True, use_cache=True)
        memory_kv = memory_outputs.past_key_values
    
    # Create prefix cache
    prefix_cache = DynamicCache()
    
    # Handle different cache formats
    if hasattr(memory_kv, 'key_cache'):
        num_layers = len(memory_kv.key_cache)
        get_kv = lambda idx: (memory_kv.key_cache[idx], memory_kv.value_cache[idx])
    elif hasattr(memory_kv, '__getitem__'):
        num_layers = len(memory_kv)
        get_kv = lambda idx: memory_kv[idx]
    else:
        raise ValueError(f"Unknown cache format: {type(memory_kv)}")
    
    logger.info(f"Cache format: {type(memory_kv).__name__}, {num_layers} layers")
    
    for layer_idx in range(num_layers):
        k, v = get_kv(layer_idx)
        
        # if layer_idx in target_layers:
        #     k_prefix = k
        #     v_prefix = v
        #     logger.debug(f"Layer {layer_idx}: Using full KV, shape {k_prefix.shape}")
        # else:
        #     k_prefix = k[:, :, :0, :]
        #     v_prefix = v[:, :, :0, :]
        
        # prefix_cache.update(k_prefix, v_prefix, layer_idx)
        prefix_cache.update(k, v, layer_idx)
    
    prefix_len = memory_inputs["input_ids"].shape[1]
    
    # Tokenize query
    query_inputs = tokenizer(query, return_tensors="pt")
    query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
    query_len = query_inputs["input_ids"].shape[1]
    
    # Position IDs - Handle mRoPE for Qwen3-VL
    position_ids = torch.arange(
        prefix_len,
        prefix_len + query_len,
        device=device
    )
    
    # Check if model uses mRoPE (Qwen3-VL)
    if hasattr(model, 'config') and getattr(model.config, 'rope_scaling', None):
        rope_type = model.config.rope_scaling.get('type', '') if isinstance(model.config.rope_scaling, dict) else ''
        if 'mrope' in rope_type.lower() or 'qwen3_vl' in type(model).__name__.lower():
            # mRoPE needs [3, seq_len] format
            position_ids = position_ids.unsqueeze(0).expand(3, -1)  # [3, query_len]
            logger.debug(f"Using mRoPE position_ids: {position_ids.shape}")
        else:
            position_ids = position_ids.unsqueeze(0)  # [1, query_len]
    else:
        position_ids = position_ids.unsqueeze(0)  # [1, query_len]
    
    query_inputs["position_ids"] = position_ids
    
    # Cache position
    cache_position = torch.arange(
        prefix_len,
        prefix_len + query_len,
        device=device
    )
    query_inputs["cache_position"] = cache_position
    
    # Attention mask inkl. Präfix
    attention_mask = torch.ones(1, prefix_len + query_len, device=device)
    query_inputs["attention_mask"] = attention_mask
    
    # Pass 2: Generate
    logger.info("Pass 2: Generating with KV prefix...")
    with torch.no_grad():
        outputs = model.generate(
            **query_inputs,
            past_key_values=prefix_cache,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            **generation_kwargs,
        )
    
    generated_ids = outputs[0][query_inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text