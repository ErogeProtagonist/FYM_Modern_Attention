"""
SFT Data Preparation Script for smol-smoltalk Dataset

Downloads and prepares the HuggingFace smol-smoltalk dataset, which is
specifically designed for sub-1B parameter models (SmolLM2-135M to 360M).

Key advantages over generic datasets (Alpaca, OpenHermes):
- Shorter conversations (small models can't track long multi-turn context)
- No function calling (too complex, causes forgetting)
- No advanced math (CoT chains overflow capacity)
- Reduced task variety (less rewriting/summarization)

Reference: SmolLM2 Paper (https://huggingface.co/papers/2502.02737)

Usage:
    python sft_data_prep.py --output_dir sft_data
"""

import os
import json
import random
import argparse
from tqdm import tqdm


def format_smoltalk_example(example, chat_template: str = "chatml"):
    """
    Format smol-smoltalk conversation into prompt/completion format.
    
    The dataset uses a standard messages format:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    
    Args:
        example: Dataset example with 'messages' field
        chat_template: Format to use ('chatml', 'alpaca', 'simple')
    
    Returns:
        Dict with 'prompt', 'completion', 'source', 'num_turns'
    """
    messages = example.get("messages", [])
    
    if not messages or len(messages) < 2:
        return None
    
    # Build conversation turns
    prompt_parts = []
    completion = ""
    num_turns = 0
    
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "").strip()
        
        if not content:
            continue
        
        if role == "user":
            if chat_template == "chatml":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif chat_template == "alpaca":
                prompt_parts.append(f"### User:\n{content}")
            else:  # simple
                prompt_parts.append(f"User: {content}")
            num_turns += 1
            
        elif role == "assistant":
            # For training, we only supervise the LAST assistant response
            if i == len(messages) - 1:
                completion = content
            else:
                # Include previous assistant responses in the prompt
                if chat_template == "chatml":
                    prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
                elif chat_template == "alpaca":
                    prompt_parts.append(f"### Assistant:\n{content}")
                else:
                    prompt_parts.append(f"Assistant: {content}")
    
    if not completion:
        return None
    
    # Add the assistant prefix for the response we're training on
    if chat_template == "chatml":
        prompt_parts.append("<|im_start|>assistant\n")
    elif chat_template == "alpaca":
        prompt_parts.append("### Assistant:\n")
    else:
        prompt_parts.append("Assistant: ")
    
    prompt = "\n".join(prompt_parts) if chat_template != "simple" else "\n\n".join(prompt_parts)
    
    return {
        "prompt": prompt,
        "completion": completion,
        "source": example.get("source", "smoltalk"),
        "num_turns": num_turns
    }


def prepare_sft_data(
    output_dir: str,
    max_samples: int = None,
    chat_template: str = "chatml",
    max_completion_tokens: int = 512,
    seed: int = 42
):
    """
    Download and prepare smol-smoltalk dataset for SFT.
    
    Args:
        output_dir: Directory to save processed data
        max_samples: Maximum samples to use (None = all)
        chat_template: Chat format ('chatml', 'alpaca', 'simple')
        max_completion_tokens: Filter out examples with longer completions
        seed: Random seed for reproducibility
    """
    from datasets import load_dataset
    import tiktoken
    
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer for length filtering
    enc = tiktoken.get_encoding("gpt2")
    
    # === Load smol-smoltalk ===
    print("Loading HuggingFaceTB/smol-smoltalk...")
    dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    print(f"Loaded {len(dataset)} examples")
    
    # === Process examples ===
    all_examples = []
    sources_count = {}
    skipped_long = 0
    skipped_invalid = 0
    
    for example in tqdm(dataset, desc="Processing"):
        formatted = format_smoltalk_example(example, chat_template)
        
        if formatted is None:
            skipped_invalid += 1
            continue
        
        # Filter by completion length (avoid long outputs that small models struggle with)
        completion_tokens = len(enc.encode(formatted["completion"]))
        if completion_tokens > max_completion_tokens:
            skipped_long += 1
            continue
        
        all_examples.append(formatted)
        
        # Track source distribution
        source = formatted["source"]
        sources_count[source] = sources_count.get(source, 0) + 1
    
    print(f"\nProcessed {len(all_examples)} valid examples")
    print(f"Skipped {skipped_invalid} invalid examples")
    print(f"Skipped {skipped_long} examples with completion > {max_completion_tokens} tokens")
    
    # === Subsample if requested ===
    if max_samples and len(all_examples) > max_samples:
        random.shuffle(all_examples)
        all_examples = all_examples[:max_samples]
        print(f"Subsampled to {max_samples} examples")
    
    # === Shuffle and split ===
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * 0.95)
    train_data = all_examples[:split_idx]
    val_data = all_examples[split_idx:]
    
    # === Save as JSONL ===
    train_path = os.path.join(output_dir, "sft_train.jsonl")
    val_path = os.path.join(output_dir, "sft_val.jsonl")
    
    with open(train_path, "w", encoding="utf-8") as f:
        for example in train_data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    with open(val_path, "w", encoding="utf-8") as f:
        for example in val_data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"\nSaved {len(train_data)} training samples to {train_path}")
    print(f"Saved {len(val_data)} validation samples to {val_path}")
    
    # === Save metadata ===
    metadata = {
        "dataset": "HuggingFaceTB/smol-smoltalk",
        "chat_template": chat_template,
        "total_examples": len(all_examples),
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "max_completion_tokens": max_completion_tokens,
        "sources": sources_count,
        "seed": seed
    }
    
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # === Print statistics ===
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    print(f"Chat template: {chat_template}")
    print(f"\nSource distribution:")
    for source, count in sorted(sources_count.items(), key=lambda x: -x[1])[:10]:
        pct = 100 * count / sum(sources_count.values())
        print(f"  {source}: {count} ({pct:.1f}%)")
    
    # Compute avg lengths
    avg_prompt_len = sum(len(enc.encode(ex["prompt"])) for ex in all_examples[:1000]) / min(1000, len(all_examples))
    avg_comp_len = sum(len(enc.encode(ex["completion"])) for ex in all_examples[:1000]) / min(1000, len(all_examples))
    
    print(f"\nAverage prompt length: {avg_prompt_len:.0f} tokens")
    print(f"Average completion length: {avg_comp_len:.0f} tokens")
    print(f"Average total length: {avg_prompt_len + avg_comp_len:.0f} tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SFT Data Preparation using smol-smoltalk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (all samples, ChatML format)
    python sft_data_prep.py --output_dir sft_data
    
    # Limit samples for testing
    python sft_data_prep.py --output_dir sft_data --max_samples 10000
    
    # Use Alpaca-style formatting
    python sft_data_prep.py --output_dir sft_data --chat_template alpaca
        """
    )
    parser.add_argument("--output_dir", type=str, default="sft_data",
                        help="Output directory for prepared data")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples (None = use all)")
    parser.add_argument("--chat_template", type=str, default="chatml",
                        choices=["chatml", "alpaca", "simple"],
                        help="Chat template format")
    parser.add_argument("--max_completion_tokens", type=int, default=512,
                        help="Filter out examples with longer completions")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    prepare_sft_data(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        chat_template=args.chat_template,
        max_completion_tokens=args.max_completion_tokens,
        seed=args.seed
    )
