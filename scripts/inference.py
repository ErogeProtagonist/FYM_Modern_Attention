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
    
    # Strip the legacy 'naive_impl.' prefix from older MLA checkpoints. They
    # were saved by the now-removed FlashMLAttention wrapper, which embedded
    # NaiveMLAttention as a sub-module called 'naive_impl'. New checkpoints
    # don't carry the prefix; the guard makes this a no-op for them.
    if any('.naive_impl.' in k for k in state_dict.keys()):
        print("Stripping legacy 'naive_impl.' prefix from MLA checkpoint...")
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
        print("\nNo prompt provided. Use --prompt or --interactive")
        print("Example: python inference.py --checkpoint log/hybrid_05000.pt --prompt 'Hello'")


if __name__ == "__main__":
    main()
