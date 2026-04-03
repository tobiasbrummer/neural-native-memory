# Virtual Prefix Injection – Implementation Plan

+doc:planning +project:kv-llm-vectorstore

## Overview

Proof-of-Concept for "Virtual Prefix Injection": Store Pre-RoPE Hidden States, JIT convert to K/V pairs, inject as virtual prefix into model cache.

## Hypothesis

By storing raw Hidden States (Pre-RoPE) and converting them Just-in-Time to K/V pairs with proper RoPE, we can:
1. Save significant storage (H instead of K+V)
2. Inject full contextual sequences as virtual prefix
3. Maintain mathematical consistency with Transformer attention

---

## Phase 1: Core Utilities (lib/virtual_prefix.py)

### 1.1 Hidden State Extractor

Extract Pre-RoPE hidden states from specific layers during forward pass.

**Challenge:** Standard `output_hidden_states=True` returns states *after* attention. We need states *before* attention enters the layer.

**Solution:** Use forward hook or extract from embedding layer + intermediate states.

```python
def extract_pre_rope_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    target_layers: List[int],
) -> Dict[int, np.ndarray]:
    """
    Extract hidden states BEFORE RoPE is applied.
    Returns: {layer_idx: hidden_state_array}
    """
```

### 1.2 JIT Projector (H -> K, V)

Project hidden states to Keys and Values using model weights.

```python
def project_to_kv(
    hidden_states: np.ndarray,  # [seq_len, hidden_dim]
    layer_idx: int,
    model: AutoModelForCausalLM,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project H to (K, V) using k_proj, v_proj weights.
    Returns: keys, values (both [seq_len, num_heads, head_dim])
    """
```

### 1.3 RoPE Engine

Apply Rotary Position Embedding for positions 0..N.

```python
def apply_rope_to_keys(
    keys: np.ndarray,
    positions: np.ndarray,  # [0, 1, 2, ..., seq_len-1]
    layer_idx: int,
    model: AutoModelForCausalLM,
) -> np.ndarray:
    """
    Apply RoPE rotation to keys for given positions.
    """
```

### 1.4 Cache Injector

Inject prepared K/V pairs into model cache.

```python
def inject_virtual_prefix(
    model: AutoModelForCausalLM,
    kv_cache: DynamicCache,
    keys_per_layer: Dict[int, np.ndarray],
    values_per_layer: Dict[int, np.ndarray],
) -> int:
    """
    Inject K/V as virtual prefix.
    Returns: seq_len (offset for real input)
    """
```

---

## Phase 2: Storage Format (data/virtual_prefix/)

### 2.1 File Naming

```
virtual_prefix_{doc_id}_{timestamp}.npz
```

### 2.2 Numpy Structure

```python
{
    'hidden_states': np.ndarray,  # [n_layers, seq_len, hidden_dim] or dict
    'token_ids': np.ndarray,      # [seq_len]
    'text': str,                  # Original text
    'metadata': {
        'model': str,
        'layers': List[int],
        'timestamp': str,
    }
}
```

Use `.npz` format for efficient storage of multiple arrays.

---

## Phase 3: Experiment 5 – End-to-End Validation

### 3.1 Objective

Validate that injected virtual prefix affects generation correctly.

### 3.2 Test Design

1. **Store:** Process a "memory" text, extract H, save to .npz
2. **Load:** Read H from .npz
3. **JIT:** Convert to K,V with RoPE
4. **Inject:** Add as prefix to cache
5. **Generate:** Run inference with query
6. **Validate:** Check if generation incorporates the memory

### 3.3 Test Cases

| Case | Memory Text | Query | Expected Behavior |
|------|-------------|-------|-------------------|
| Simple fact | "The sky is blue." | "What color is the sky?" | Should mention blue |
| Context | "User prefers Python." | "Which language should I use?" | Should suggest Python |
| Control (no injection) | — | Same query | Different/generic answer |

### 3.4 Success Criterion

Generated text with injected prefix shows clear influence of memory compared to baseline (no injection).

---

## Phase 4: Implementation Details

### 4.1 Model-Specific Handling

**Qwen3 Architecture:**
- GQA (Grouped Query Attention)
- Rotary embedding in `model.layers[N].self_attn.rotary_emb`
- Projection: `self_attn.k_proj`, `self_attn.v_proj`

**Access Pattern:**
```python
layer = model.model.layers[layer_idx]
k_proj = layer.self_attn.k_proj
v_proj = layer.self_attn.v_proj
rot_emb = layer.self_attn.rotary_emb
```

### 4.2 Position Offset Handling

When injecting N tokens as prefix:
- Real input tokens need `position_ids += N`
- Attention mask needs adjustment
- Cache length increases by N

---

## Phase 5: Files to Create

| File | Purpose |
|------|---------|
| `lib/virtual_prefix.py` | Core utilities (extract, project, rope, inject) |
| `experiments/5_test_virtual_prefix.py` | End-to-end validation |
| `data/virtual_prefix/test_memory.npz` | Stored test data |

---

## Phase 6: Open Questions

1. **Hook vs. Manual Forward:** Use forward hooks or manual layer-by-layer forward?
2. **Which Hidden States:** Last token only (like KV-Embedding) or full sequence?
3. **Layer Selection:** Reuse KV-Embedding target_layers or all layers?

---

## Next Steps

1. Implement `lib/virtual_prefix.py` with H extraction and H->K,V projection
2. Create storage format test
3. Implement injection mechanism
4. Run end-to-end validation
5. If successful: Integrate with Qdrant
