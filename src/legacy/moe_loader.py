"""
MoE Model Loader with Expert CPU Offloading.

Loads AWQ/BF16 MoE models with experts offloaded to CPU RAM,
keeping backbone (attention, router, head) on GPU.
"""

import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)


def get_moe_device_map(
    config: AutoConfig,
    gpu_device: int = 0,
    offload_experts: bool = True,
) -> Dict[str, Any]:
    """
    Create device_map for MoE model with expert offloading.
    
    Args:
        config: Model config
        gpu_device: GPU device ID
        offload_experts: If True, put experts on CPU
        
    Returns:
        Device map dictionary
    """
    device_map = {}
    num_layers = config.num_hidden_layers
    
    # Embedding and head always on GPU
    device_map["model.embed_tokens"] = gpu_device
    device_map["model.norm"] = gpu_device
    device_map["lm_head"] = gpu_device
    
    # Check for different MoE architectures
    # Granite uses "model.layers.X.block_sparse_moe"
    # Mixtral uses "model.layers.X.block_sparse_moe"
    # Qwen3 uses "model.layers.X.mlp" with experts inside
    
    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        
        # Attention always on GPU
        device_map[f"{prefix}.self_attn"] = gpu_device
        device_map[f"{prefix}.input_layernorm"] = gpu_device
        device_map[f"{prefix}.post_attention_layernorm"] = gpu_device
        
        # Check if this is a MoE layer or dense
        # For hybrid models (Granite-4), some layers are dense
        
        if offload_experts:
            # MoE components: gate on GPU, experts on CPU
            device_map[f"{prefix}.block_sparse_moe.gate"] = gpu_device
            device_map[f"{prefix}.block_sparse_moe.experts"] = "cpu"
            
            # Alternative naming for Qwen3/other models
            device_map[f"{prefix}.mlp.gate"] = gpu_device
            device_map[f"{prefix}.mlp.experts"] = "cpu"
            device_map[f"{prefix}.mlp.shared_expert"] = gpu_device
            device_map[f"{prefix}.mlp.shared_expert_gate"] = gpu_device
        else:
            # All on GPU
            device_map[f"{prefix}.block_sparse_moe"] = gpu_device
            device_map[f"{prefix}.mlp"] = gpu_device
    
    return device_map


def load_moe_model(
    model_path: str,
    offload_experts: bool = True,
    max_memory: Optional[Dict] = None,
    torch_dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load MoE model with expert CPU offloading.
    
    Args:
        model_path: Path to HuggingFace model or AWQ model
        offload_experts: If True, put MoE experts on CPU
        max_memory: Optional memory limits per device
        torch_dtype: Data type for model
        trust_remote_code: Trust remote code in model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading MoE model: {model_path}")
    
    # Load config first
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    
    # Log MoE info
    num_experts = getattr(config, "num_local_experts", None) or \
                  getattr(config, "num_experts", None) or \
                  getattr(config, "n_routed_experts", None)
    num_active = getattr(config, "num_experts_per_tok", None) or \
                 getattr(config, "top_k", None)
    
    logger.info(f"Model: {config.model_type}")
    logger.info(f"Layers: {config.num_hidden_layers}")
    logger.info(f"Hidden size: {config.hidden_size}")
    if num_experts:
        logger.info(f"Experts: {num_experts} total, {num_active} active per token")
    
    # Create device map and load model
    is_awq = hasattr(config, "quantization_config") and \
             config.quantization_config.get("quant_method") == "awq"
    
    if max_memory is None:
        max_memory = {
            0: "8GiB",   # Strict GPU limit to force offloading
            "cpu": "60GiB",
        }
    
    # Load model
    logger.info("Loading model weights...")
    
    if is_awq:
        # Use AutoAWQ for native 4-bit loading (no FP16 decompression)
        logger.info("AWQ model detected: using AutoAWQ for native 4-bit loading")
        try:
            from awq import AutoAWQForCausalLM
            model = AutoAWQForCausalLM.from_quantized(
                model_path,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=trust_remote_code,
            )
        except ImportError:
            logger.warning("AutoAWQ not installed, falling back to transformers")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                device_map="auto",
                max_memory=max_memory,
                dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
                offload_state_dict=True,
            )
    elif offload_experts and num_experts:
        device_map = get_moe_device_map(config, offload_experts=True)
        logger.info("Expert offloading: ENABLED (experts on CPU)")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            device_map=device_map,
            max_memory=max_memory,
            dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
    else:
        logger.info("Using device_map='auto'")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            device_map="auto",
            max_memory=max_memory,
            dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Log memory usage
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"GPU memory used: {gpu_mem:.2f} GB")
    
    return model, tokenizer


def get_expert_locations(model: AutoModelForCausalLM) -> Dict[str, str]:
    """Get the device location of each expert module."""
    locations = {}
    for name, param in model.named_parameters():
        if "expert" in name.lower():
            locations[name] = str(param.device)
    return locations


if __name__ == "__main__":
    # Quick test
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python moe_loader.py <model_path>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    
    model_path = sys.argv[1]
    model, tokenizer = load_moe_model(model_path, offload_experts=True)
    
    # Show expert locations
    locations = get_expert_locations(model)
    cpu_count = sum(1 for v in locations.values() if "cpu" in v)
    gpu_count = sum(1 for v in locations.values() if "cuda" in v)
    print(f"\nExpert locations: {cpu_count} on CPU, {gpu_count} on GPU")
