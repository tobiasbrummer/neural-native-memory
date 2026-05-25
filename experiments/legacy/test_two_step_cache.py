#!/usr/bin/env python3
"""Test two-step generation with cache for Qwen3-VL."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.legacy.model_loader import load_model
from transformers.cache_utils import DynamicCache
import torch


def main():
    model, tokenizer = load_model(load_in_4bit=True)

    # Test 1: Normal generation (no cache)
    print("=== Test 1: Normal generation ===")
    inputs = tokenizer("The sky is blue. What color is the sky?", return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Normal: {result}")

    # Test 2: Two-step generation with cache
    print("\n=== Test 2: Two-step with cache ===")

    # First step
    inputs1 = tokenizer("The sky is blue.", return_tensors="pt")
    inputs1 = {k: v.to(model.device) for k, v in inputs1.items()}

    cache1 = DynamicCache()
    with torch.no_grad():
        outputs1 = model(**inputs1, past_key_values=cache1, max_new_tokens=0)

    print(f"Cache seq_len after first step: {cache1.get_seq_length()}")

    # Second step - continue generation
    inputs2 = tokenizer("What color is the sky?", return_tensors="pt")
    inputs2 = {k: v.to(model.device) for k, v in inputs2.items()}

    with torch.no_grad():
        outputs2 = model(**inputs2, past_key_values=cache1, max_new_tokens=10)

    result2 = tokenizer.decode(outputs2[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"With cache: {result2}")


if __name__ == "__main__":
    main()
