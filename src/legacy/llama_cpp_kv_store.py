"""
llama-cpp-python KV Cache Store.

Provides KV cache persistence and injection for GGUF models
using llama-cpp-python's state save/load API.
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Any

logger = logging.getLogger(__name__)


@dataclass
class LlamaCppKVState:
    """Stored KV cache state from llama-cpp-python."""
    state_data: bytes  # Raw state from save_state()
    input_ids: list  # Token IDs that generated this state
    scores: bytes  # Scores as bytes
    scores_shape: tuple  # Shape of scores array
    n_tokens: int  # Number of tokens processed
    seed: int  # Random seed
    input_text: str  # Original context text
    token_count: int  # Number of tokens in context
    model_path: str  # Model used to create this state
    metadata: dict  # Additional metadata


def extract_kv_state(
    llm: Any,
    context_text: str,
    model_path: str = "",
) -> LlamaCppKVState:
    """
    Process context text and extract KV cache state.
    
    Args:
        llm: Llama model instance
        context_text: Text to process into KV cache
        model_path: Path to model (for metadata)
        
    Returns:
        LlamaCppKVState containing the cached state
    """
    logger.info(f"Processing context ({len(context_text)} chars)...")
    
    # Tokenize to get token count
    tokens = llm.tokenize(context_text.encode("utf-8"))
    token_count = len(tokens)
    logger.info(f"Context has {token_count} tokens")
    
    # Process context to fill KV cache (no generation)
    llm.eval(tokens)
    
    # Save complete state including KV cache
    state = llm.save_state()
    logger.info(f"Saved state: {len(state.llama_state)} bytes")
    
    return LlamaCppKVState(
        state_data=bytes(state.llama_state),
        input_ids=list(state.input_ids),
        scores=state.scores.tobytes(),  # Store as bytes to preserve shape
        scores_shape=state.scores.shape,  # Store shape separately
        n_tokens=state.n_tokens,
        seed=state.seed,
        input_text=context_text,
        token_count=token_count,
        model_path=model_path,
        metadata={},
    )


def inject_kv_state(
    llm: Any,
    kv_state: LlamaCppKVState,
) -> int:
    """
    Inject previously saved KV state into model.
    
    Args:
        llm: Llama model instance
        kv_state: Previously extracted KV state
        
    Returns:
        Number of tokens in injected context
    """
    import numpy as np
    logger.info(f"Injecting KV state ({kv_state.token_count} tokens)...")
    
    # Create state object for loading with all required fields
    from llama_cpp import LlamaState
    
    # Reconstruct numpy arrays
    input_ids = np.array(kv_state.input_ids, dtype=np.intc)
    scores = np.frombuffer(kv_state.scores, dtype=np.single).reshape(kv_state.scores_shape)
    
    state = LlamaState(
        input_ids=input_ids,
        scores=scores,
        n_tokens=kv_state.n_tokens,
        llama_state=bytes(kv_state.state_data),
        llama_state_size=len(kv_state.state_data),
        seed=kv_state.seed,
    )
    
    llm.load_state(state)
    logger.info("KV state injected successfully")
    
    return kv_state.token_count


def save_kv_state(
    kv_state: LlamaCppKVState,
    path: str,
) -> None:
    """
    Save KV state to disk.
    
    Args:
        kv_state: KV state to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(kv_state, f)
    
    logger.info(f"Saved KV state to {path} ({path.stat().st_size / 1024:.1f} KB)")


def load_kv_state(path: str) -> LlamaCppKVState:
    """
    Load KV state from disk.
    
    Args:
        path: Input file path
        
    Returns:
        Loaded KV state
    """
    with open(path, "rb") as f:
        kv_state = pickle.load(f)
    
    logger.info(f"Loaded KV state: {kv_state.token_count} tokens")
    return kv_state


def generate_with_kv_state(
    llm: Any,
    kv_state: LlamaCppKVState,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.0,
) -> str:
    """
    Generate text with injected KV state as context.
    
    Args:
        llm: Llama model instance
        kv_state: KV state to inject as context
        prompt: Continuation prompt
        max_tokens: Max tokens to generate
        temperature: Sampling temperature (0 = greedy)
        
    Returns:
        Generated text
    """
    # Inject the stored context
    inject_kv_state(llm, kv_state)
    
    # Tokenize the prompt
    prompt_tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=False)
    
    # Evaluate prompt tokens (extends the KV cache with prompt)
    llm.eval(prompt_tokens)
    
    # Generate tokens one by one
    output_tokens = []
    for _ in range(max_tokens):
        # Sample next token
        token = llm.sample(
            temp=temperature,
            top_k=40,
            top_p=0.95,
        )
        
        # Check for EOS
        if token == llm.token_eos():
            break
            
        output_tokens.append(token)
        
        # Evaluate the new token to update KV cache
        llm.eval([token])
    
    # Decode output
    output_text = llm.detokenize(output_tokens).decode("utf-8", errors="ignore")
    return output_text

