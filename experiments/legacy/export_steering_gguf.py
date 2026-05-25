#!/usr/bin/env python3
"""
Export Steering Vectors to GGUF format for llama.cpp.

Objective:
Calculate a steering vector (Difference-in-Means) from Positive and Negative prompts
and save it as a .gguf file usable with `llama.cpp`'s `--control-vector` argument.

Mechanism:
1. Extract Pre-RoPE hidden states (using lib.virtual_prefix).
2. Compute Mean(Pos) - Mean(Neg) per layer.
3. Write to GGUF as `direction.{layer_idx}` tensors (Float32).
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add llama.cpp/gguf-py to path
GGUF_PY_PATH = PROJECT_ROOT / "llama.cpp" / "gguf-py"
if GGUF_PY_PATH.exists():
    sys.path.insert(0, str(GGUF_PY_PATH))
else:
    print(f"WARNING: gguf-py not found at {GGUF_PY_PATH}. Assuming installed in env.")

try:
    import gguf
except ImportError:
    print("ERROR: Failed to import gguf. Please ensure llama.cpp submodule is initialized or gguf is installed.")
    sys.exit(1)

from src.legacy.model_loader import load_model, DEFAULT_MODEL
from src.legacy.virtual_prefix import PreRopeExtractor
from src.legacy.io_utils import setup_logging, create_results_dir

logger = logging.getLogger(__name__)

def get_mean_hidden_states(
    extractor: PreRopeExtractor,
    texts: list[str],
    layers: list[int]
) -> dict[int, np.ndarray]:
    """Compute mean hidden state vector per layer for a list of texts."""
    layer_sums = {l: None for l in layers}
    count = 0

    for text in texts:
        # Extract Pre-RoPE states
        # We use use_prompt=True inside extract() by default in experiments usually, 
        # but for pure concept vectors raw text might be better?
        # Actually, Exp 9 used use_prompt=False. Let's stick to that for raw concepts.
        prefix_data = extractor.extract(
            text, 
            use_prompt=False, 
            extract_all_layers=True
        )
        
        for layer in layers:
            # Shape: [seq_len, hidden_dim]
            # Mean pool over sequence
            vec = np.mean(prefix_data.hidden_states[layer], axis=0)
            
            if layer_sums[layer] is None:
                layer_sums[layer] = vec
            else:
                layer_sums[layer] += vec
        
        count += 1
        print(f".", end="", flush=True)

    print()
    return {l: layer_sums[l] / count for l in layers}

def export_to_gguf(
    output_path: str, 
    steering_vectors: dict[int, np.ndarray],
    model_arch: str = "llama"
):
    """Write steering vectors to GGUF file."""
    gguf_writer = gguf.GGUFWriter(output_path, model_arch)
    
    # Metadata usually required? For control vectors minimal metadata seems ok.
    # common.cpp checks for `direction.{layer}` tensors.
    
    gguf_writer.add_string("control_vector.model_hint", model_arch)
    
    logger.info(f"Writing tensors to {output_path}...")
    
    for layer_idx, vec in steering_vectors.items():
        if layer_idx == 0:
            continue # Layer 0 is ignored by llama.cpp control vectors usually
            
        tensor_name = f"direction.{layer_idx}"
        
        # Must be F32
        data_f32 = vec.astype(np.float32)
        gguf_writer.add_tensor(tensor_name, data_f32)
        
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    logger.info("Export complete.")

def main():
    parser = argparse.ArgumentParser(description="Export Steering Vectors to GGUF")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HuggingFace model")
    parser.add_argument("--output", type=str, required=True, help="Output GGUF file path")
    parser.add_argument("--positive", type=str, nargs="+", required=True, help="Positive prompts")
    parser.add_argument("--negative", type=str, nargs="+", required=True, help="Negative prompts")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layers (default: all - first/last)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load Model
    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, load_in_4bit=args.load_in_4bit)
    
    extractor = PreRopeExtractor(model, tokenizer)
    
    # Determine Layers
    # If not specified, we take all layers logic or range?
    # extract(extract_all_layers=True) gets everything.
    # We might want to filter later.
    # For now, let's get all layers available in the model.
    if hasattr(model.config, "num_hidden_layers"):
        n_layers = model.config.num_hidden_layers
    else:
        # Fallback
        dummy = extractor.extract("test", extract_all_layers=True)
        n_layers = len(dummy.hidden_states)
        
    target_layers = list(range(n_layers))
    if args.layers:
        target_layers = [int(x) for x in args.layers.split(",")]
        
    logger.info(f"Computing steering vector for {len(target_layers)} layers...")
    logger.info(f"Positive examples: {len(args.positive)}")
    logger.info(f"Negative examples: {len(args.negative)}")
    
    # Compute Vectors
    logger.info("Encoding Positive...")
    pos_means = get_mean_hidden_states(extractor, args.positive, target_layers)
    
    logger.info("Encoding Negative...")
    neg_means = get_mean_hidden_states(extractor, args.negative, target_layers)
    
    # Compute Difference
    steering_vectors = {}
    for l in target_layers:
        diff = pos_means[l] - neg_means[l]
        steering_vectors[l] = diff
        
    # Export
    export_to_gguf(args.output, steering_vectors)

if __name__ == "__main__":
    main()
