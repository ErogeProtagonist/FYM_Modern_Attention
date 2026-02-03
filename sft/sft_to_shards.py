"""
Convert SFT JSONL data to binary shards for fast loading.

This creates memory-mapped binary files similar to pre-training,
enabling O(1) random access and zero-copy loading.

Output format:
- sft_train_00000.bin, sft_train_00001.bin, etc.
- Each shard contains packed sequences of (tokens, labels) pairs
- Metadata JSON with sequence boundaries
"""

import os
import json
import argparse
import numpy as np
import tiktoken
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Convert SFT JSONL to binary shards")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for shards")
    parser.add_argument("--block_size", type=int, default=2048, help="Sequence length")
    parser.add_argument("--shard_size", type=int, default=100_000_000, help="Tokens per shard (~100M)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Determine split from filename
    if "train" in args.input:
        split = "train"
    elif "val" in args.input:
        split = "val"
    else:
        split = "data"
    
    print(f"Loading and tokenizing {args.input}...")
    
    # Load and tokenize all examples
    all_tokens = []
    all_labels = []
    
    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    for line in tqdm(lines, desc="Tokenizing"):
        ex = json.loads(line.strip())
        prompt = ex["prompt"]
        completion = ex["completion"]
        
        # Tokenize with ChatML special tokens
        prompt_tokens = tokenizer.encode(prompt, allowed_special="all")
        completion_tokens = tokenizer.encode(completion, allowed_special="all")
        
        # Build full sequence
        full_tokens = prompt_tokens + completion_tokens + [tokenizer.eot_token]
        labels = [-100] * len(prompt_tokens) + completion_tokens + [tokenizer.eot_token]
        
        # Truncate to block_size
        full_tokens = full_tokens[:args.block_size]
        labels = labels[:args.block_size]
        
        # Pad to block_size
        pad_len = args.block_size - len(full_tokens)
        if pad_len > 0:
            full_tokens = full_tokens + [tokenizer.eot_token] * pad_len
            labels = labels + [-100] * pad_len
        
        all_tokens.append(full_tokens)
        all_labels.append(labels)
    
    print(f"Tokenized {len(all_tokens)} examples")
    
    # Convert to numpy arrays
    # Use int32 for labels (to handle -100), uint16 for tokens
    tokens_array = np.array(all_tokens, dtype=np.uint16)
    labels_array = np.array(all_labels, dtype=np.int32)
    
    print(f"Token array shape: {tokens_array.shape}")
    print(f"Labels array shape: {labels_array.shape}")
    
    # Calculate number of shards
    total_tokens = tokens_array.size
    num_shards = max(1, (total_tokens + args.shard_size - 1) // args.shard_size)
    examples_per_shard = (len(all_tokens) + num_shards - 1) // num_shards
    
    print(f"Creating {num_shards} shards with ~{examples_per_shard} examples each")
    
    # Write shards
    for shard_idx in range(num_shards):
        start_idx = shard_idx * examples_per_shard
        end_idx = min((shard_idx + 1) * examples_per_shard, len(all_tokens))
        
        shard_tokens = tokens_array[start_idx:end_idx]
        shard_labels = labels_array[start_idx:end_idx]
        
        # Save tokens shard
        tokens_path = os.path.join(args.output_dir, f"sft_{split}_{shard_idx:05d}_tokens.bin")
        shard_tokens.tofile(tokens_path)
        
        # Save labels shard
        labels_path = os.path.join(args.output_dir, f"sft_{split}_{shard_idx:05d}_labels.bin")
        shard_labels.tofile(labels_path)
        
        print(f"Wrote shard {shard_idx}: {end_idx - start_idx} examples")
    
    # Save metadata
    metadata = {
        "split": split,
        "num_shards": num_shards,
        "num_examples": len(all_tokens),
        "block_size": args.block_size,
        "examples_per_shard": examples_per_shard,
        "tokens_dtype": "uint16",
        "labels_dtype": "int32"
    }
    
    metadata_path = os.path.join(args.output_dir, f"sft_{split}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDone! Metadata saved to {metadata_path}")
    print(f"Total size: {(tokens_array.nbytes + labels_array.nbytes) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
