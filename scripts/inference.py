"""
Inference Script for Hybrid SWA and MLA Transformers.

Portable inference that works on any hardware (RTX 3090, CPU, etc.)
by forcing the use of naive PyTorch attention implementations.

Usage:
    python inference.py --checkpoint log/hybrid_05000.pt --prompt "Hello, I am"
"""

import os
import sys
import argparse
import torch
import tiktoken

# Add parent directory to path for models import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import ModelConfig
from models.transformer import Transformer



def load_model(checkpoint_path: str, device: str = "cuda") -> tuple:
    """
    Load a trained model from checkpoint.
    
    Forces 'inference' mode to use naive attention implementations
    that work on any hardware.
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Reconstruct config from saved dict
    config_dict = checkpoint['config']
    config = ModelConfig(**config_dict)
    
    # Create model in inference mode (forces naive kernels)
    model = Transformer(config, mode="inference")
    
    # Handle checkpoints saved from compiled models (keys have '_orig_mod.' prefix)
    state_dict = checkpoint['model']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("Detected compiled model checkpoint, stripping '_orig_mod.' prefix...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    # Handle MLA checkpoints saved from FlashMLAttention wrapper
    # Training uses FlashMLAttention which wraps NaiveMLAttention as 'naive_impl'
    # Inference uses NaiveMLAttention directly, so strip the wrapper prefix
    if any('.naive_impl.' in k for k in state_dict.keys()):
        print("Detected FlashMLA training checkpoint, stripping 'naive_impl.' prefix...")
        state_dict = {k.replace('.naive_impl.', '.'): v for k, v in state_dict.items()}
    
    # Load weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Loaded {config.model_type.upper()} model from step {checkpoint.get('step', 'unknown')}")
    if 'val_loss' in checkpoint and checkpoint['val_loss'] is not None:
        print(f"Validation loss at checkpoint: {checkpoint['val_loss']:.4f}")
    
    return model, config


def generate_text(
    model: Transformer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = "cuda"
) -> str:
    """
    Generate text from a prompt using the trained model.
    """
    enc = tiktoken.get_encoding("gpt2")
    
    # Encode prompt
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generating {max_new_tokens} tokens with temperature={temperature}, top_k={top_k}...")
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode
    text = enc.decode(generated[0].tolist())
    return text


def interactive_mode(model: Transformer, device: str = "cuda"):
    """
    Interactive generation mode - keep generating until user quits.
    """
    print("\n" + "="*60)
    print("Interactive Mode - Enter prompts to generate text")
    print("Type 'quit' or 'exit' to stop")
    print("="*60 + "\n")
    
    while True:
        try:
            prompt = input("Prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not prompt:
                continue
            
            text = generate_text(model, prompt, device=device)
            print(f"\nGenerated:\n{text}\n")
            print("-"*40)
            
        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            break


def benchmark_inference(
    model: Transformer,
    config: ModelConfig,
    device: str = "cuda",
    num_tokens: int = 100,
    num_runs: int = 5
):
    """
    Benchmark inference speed and memory usage.
    """
    import time
    
    print("\n" + "="*60)
    print("Inference Benchmark")
    print("="*60)
    
    enc = tiktoken.get_encoding("gpt2")
    prompt = "The quick brown fox jumps over the lazy dog"
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        _ = model.generate(tokens, max_new_tokens=10)
    
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark runs
    times = []
    for run in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(tokens.clone(), max_new_tokens=num_tokens)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.3f}s ({num_tokens/elapsed:.1f} tok/s)")
    
    avg_time = sum(times) / len(times)
    avg_tps = num_tokens / avg_time
    
    print(f"\nAverage: {avg_time:.3f}s ({avg_tps:.1f} tokens/sec)")
    
    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory: {peak_memory:.2f} GB")
    
    # KV cache size estimate
    if config.model_type == "mla":
        # MLA: c_kv (kv_lora_rank) + k_rope (rope_dim) per layer
        cache_per_token = config.n_layers * (config.kv_lora_rank + config.rope_dim) * 2  # bytes (bf16)
    else:
        # Hybrid: varies by layer type
        # Simplified: assume average is half global, half local
        avg_cache = config.n_layers * config.n_heads * config.head_dim * 2 * 2  # K+V, bytes
        # Local layers only keep window_size tokens
        cache_per_token = avg_cache  # Simplified
    
    print(f"Estimated KV cache: {cache_per_token * num_tokens / 1024:.1f} KB for {num_tokens} tokens")


def main():
    parser = argparse.ArgumentParser(description="Inference for Hybrid/MLA Transformer")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter interactive generation mode")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run inference benchmark")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (auto-detected if not specified)")
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    
    # Run benchmark if requested
    if args.benchmark:
        benchmark_inference(model, config, device)
    
    # Interactive mode
    if args.interactive:
        interactive_mode(model, device)
    
    # Single generation
    elif args.prompt:
        text = generate_text(
            model, 
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )
        print(f"\n{'='*60}")
        print(text)
        print('='*60)
    
    else:
        print("\nNo prompt provided. Use --prompt, --interactive, or --benchmark")
        print("Example: python inference.py --checkpoint log/hybrid_05000.pt --prompt 'Hello'")


if __name__ == "__main__":
    main()
