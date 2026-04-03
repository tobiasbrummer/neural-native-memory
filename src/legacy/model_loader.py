"""
Model loading utilities for KV-Embedding experiments.

Provides model-agnostic loading via HuggingFace transformers.
"""

import logging
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Default model for experiments
DEFAULT_MODEL = "Qwen/Qwen2-1.5B"


def get_device() -> torch.device:
    """Determine best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_model(
    model_name: str = DEFAULT_MODEL,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float16,
    load_in_4bit: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a decoder-only LLM and its tokenizer.
    
    Args:
        model_name: HuggingFace model identifier
        device: Target device (auto-detected if None)
        dtype: Model precision (default: float16)
        load_in_4bit: Whether to load the model in 4-bit precision
    
    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        device = get_device()
    
    logger.info(f"Loading model {model_name} on {device} with dtype {dtype} (4bit={load_in_4bit})")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device if device.type == "cuda" else None,
            quantization_config=quantization_config,
            trust_remote_code=True,
            output_hidden_states=True,  # Required for layer access
        )
    except ValueError as e:
        if "Unrecognized configuration class" in str(e) or "Vision" in str(e) or "VL" in model_name:
            logger.warning(f"Failed to load as CausalLM, trying AutoModelForVision2Seq (likely Vision-Language model): {e}")
            logger.warning(f"Failed to load as CausalLM, trying AutoModelForVision2Seq (likely Vision-Language model): {e}")
            try:
                from transformers import AutoModelForVision2Seq
                model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map=device if device.type == "cuda" else None,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    output_hidden_states=True,
                )
            except (ImportError, Exception) as e2:
                logger.warning(f"AutoModelForVision2Seq failed (or not found), trying generic AutoModel: {e2}")
                from transformers import AutoModel
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map=device if device.type == "cuda" else None,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    output_hidden_states=True,
                )
        else:
            raise e
    
    if device.type != "cuda" and not load_in_4bit:
         model = model.to(device)
    
    model.eval()
    
    logger.info(f"Model loaded: {type(model).__name__}")
    
    return model, tokenizer


def get_embedding_matrix(model: AutoModelForCausalLM) -> torch.Tensor:
    """
    Extract the input embedding matrix from a model.
    
    Args:
        model: The loaded language model
    
    Returns:
        Embedding matrix of shape (vocab_size, hidden_size)
    """
    # Most models use this structure
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.weight.data
    elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte.weight.data
    elif hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings().weight.data
    else:
        raise ValueError("Could not find embedding matrix in model architecture")
