
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

def debug_qwen3_rope():
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        load_in_4bit=True, 
        trust_remote_code=True
    )

    prefix_text = "The quick brown fox jumps over the lazy dog."
    query_text = "What did the fox do?"
    
    # 1. Run Baseline (Standard Generation) to see expected behavior
    full_text = prefix_text + " " + query_text
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    # print("Baseline generation...")
    # model.generate(**inputs, max_new_tokens=10)

    # 2. Manual Cache Injection Simulation
    print("\nSimulating Manual Cache Injection...")
    
    # Create dummy cache (zeros) just to test shapes
    prefix_tokens = tokenizer(prefix_text, return_tensors="pt").input_ids.to(model.device)
    prefix_len = prefix_tokens.shape[1]
    
    query_tokens = tokenizer(query_text, return_tensors="pt").input_ids.to(model.device)
    query_len = query_tokens.shape[1]
    
    print(f"Prefix len: {prefix_len}, Query len: {query_len}")
    
    cache = DynamicCache()
    num_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    
    # Populate with dummy data
    for i in range(model.config.num_hidden_layers):
        k = torch.zeros(1, num_heads, prefix_len, head_dim, device=model.device, dtype=model.dtype)
        v = torch.zeros(1, num_heads, prefix_len, head_dim, device=model.device, dtype=model.dtype)
        cache.update(k, v, i)
        
    print(f"Cache populated. Seen tokens: {cache.get_seq_length()}")
    
    # Prepare inputs for generate
    position_ids = torch.arange(prefix_len, prefix_len + query_len, device=model.device)
    # mRoPE check
    if 'mrope' in str(model.config.rope_scaling) or 'qwen3_vl' in type(model).__name__.lower():
        position_ids = position_ids.unsqueeze(0).expand(3, -1)
    else:
        position_ids = position_ids.unsqueeze(0)
        
    print(f"Position IDs shape: {position_ids.shape}")
    
    attention_mask = torch.ones(1, prefix_len + query_len, device=model.device)
    cache_position = torch.arange(prefix_len, prefix_len + query_len, device=model.device)
    
    # Clear rope_deltas if they exist
    if hasattr(model, 'rope_deltas'):
        print("Clearing model.rope_deltas")
        model.rope_deltas = None
        
    print("Calling generate...")
    try:
        model.generate(
            input_ids=query_tokens,
            past_key_values=cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            max_new_tokens=5,
            use_cache=True
        )
        print("Generate SUCCESS")
    except Exception as e:
        print(f"Generate FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_qwen3_rope()
