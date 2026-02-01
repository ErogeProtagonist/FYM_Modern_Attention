"""
SFT Training Script for Hybrid SWA and MLA Transformers.

Fine-tunes a pretrained checkpoint on instruction-following data
from the smol-smoltalk dataset (designed for sub-1B models).

Features:
- ChatML-style tokenization with proper masking
- Gradient accumulation for larger effective batch sizes
- Cosine LR schedule with warmup
- Validation loss tracking
- Compatible with both Hybrid and MLA architectures

Usage:
    # Full fine-tuning on local GPU
    python sft_train.py \\
        --checkpoint ../Checkpoints/hybrid_19073.pt \\
        --data_dir sft_data \\
        --batch_size 4 \\
        --epochs 3
    
    # Lower memory with gradient accumulation
    python sft_train.py \\
        --checkpoint ../Checkpoints/mla_19073.pt \\
        --data_dir sft_data \\
        --batch_size 2 \\
        --grad_accum 8
"""

import os
import sys
import json
import math
import time
import argparse
from dataclasses import asdict

# Add yadda to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "yadda"))

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

from models.config import ModelConfig
from models.transformer import Transformer


class SFTDataLoader:
    """
    Dataloader for JSONL SFT data with proper prompt/completion masking.
    
    Only computes loss on completion tokens (prompt tokens masked with -100).
    """
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer, 
        block_size: int, 
        batch_size: int, 
        device: str,
        shuffle: bool = True
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        
        # Load all examples
        self.examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line.strip())
                self.examples.append(example)
        
        print(f"Loaded {len(self.examples)} examples from {data_path}")
        
        # Pre-tokenize for efficiency
        self._tokenize_all()
        
        self.indices = list(range(len(self.tokenized_examples)))
        if self.shuffle:
            import random
            random.shuffle(self.indices)
        self.current_idx = 0
    
    def _tokenize_all(self):
        """Pre-tokenize all examples for faster training."""
        self.tokenized_examples = []
        
        for ex in self.examples:
            prompt = ex["prompt"]
            completion = ex["completion"]
            
            # Tokenize separately to know where prompt ends
            prompt_tokens = self.tokenizer.encode(prompt)
            completion_tokens = self.tokenizer.encode(completion)
            
            # Add EOS at end
            full_tokens = prompt_tokens + completion_tokens + [self.tokenizer.eot_token]
            
            # Create labels: -100 for prompt, actual tokens for completion
            labels = (
                [-100] * len(prompt_tokens) +  # Mask prompt
                completion_tokens +             # Train on completion
                [self.tokenizer.eot_token]      # Train on EOS
            )
            
            # Truncate if needed
            if len(full_tokens) > self.block_size:
                full_tokens = full_tokens[:self.block_size]
                labels = labels[:self.block_size]
            
            self.tokenized_examples.append({
                "tokens": full_tokens,
                "labels": labels
            })
    
    def reset(self):
        """Reset for new epoch."""
        if self.shuffle:
            import random
            random.shuffle(self.indices)
        self.current_idx = 0
    
    def __len__(self):
        return len(self.tokenized_examples) // self.batch_size
    
    def next_batch(self):
        """Get next batch with proper padding."""
        batch_tokens = []
        batch_labels = []
        
        for _ in range(self.batch_size):
            if self.current_idx >= len(self.indices):
                self.current_idx = 0
                if self.shuffle:
                    import random
                    random.shuffle(self.indices)
            
            idx = self.indices[self.current_idx]
            self.current_idx += 1
            
            ex = self.tokenized_examples[idx]
            tokens = ex["tokens"].copy()
            labels = ex["labels"].copy()
            
            # Pad to block_size
            pad_len = self.block_size - len(tokens)
            if pad_len > 0:
                tokens = tokens + [self.tokenizer.eot_token] * pad_len
                labels = labels + [-100] * pad_len  # Don't train on padding
            
            batch_tokens.append(tokens)
            batch_labels.append(labels)
        
        # Convert to tensors
        x = torch.tensor(batch_tokens, dtype=torch.long, device=self.device)
        y = torch.tensor(batch_labels, dtype=torch.long, device=self.device)
        
        return x, y


def load_pretrained_model(checkpoint_path: str, device: str):
    """Load pretrained model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Reconstruct config
    config_dict = checkpoint['config']
    config = ModelConfig(**config_dict)
    
    # Create model in training mode (uses optimized attention)
    model = Transformer(config, mode="train")
    
    # Handle compiled model checkpoints
    state_dict = checkpoint['model']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("Stripping '_orig_mod.' prefix from compiled checkpoint...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    # Handle MLA wrapper prefix
    if any('.naive_impl.' in k for k in state_dict.keys()):
        print("Stripping 'naive_impl.' prefix from FlashMLA checkpoint...")
        state_dict = {k.replace('.naive_impl.', '.'): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    
    step = checkpoint.get('step', 'unknown')
    val_loss = checkpoint.get('val_loss', None)
    print(f"Loaded {config.model_type.upper()} model from step {step}")
    if val_loss:
        print(f"Pretrain validation loss: {val_loss:.4f}")
    
    return model, config


def compute_sft_loss(model, x, y):
    """
    Compute cross-entropy loss only on completion tokens.
    
    Labels of -100 are ignored (prompt and padding tokens).
    """
    logits, _, _ = model(x)
    
    # Shift for next-token prediction: predict token i+1 from token i
    logits = logits[:, :-1, :].contiguous()
    labels = y[:, 1:].contiguous()
    
    # Flatten for cross-entropy
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    
    # Compute loss (automatically ignores -100 labels)
    loss = F.cross_entropy(logits, labels, ignore_index=-100)
    
    return loss


def main():
    parser = argparse.ArgumentParser(
        description="SFT Training for Hybrid/MLA Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained checkpoint")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="sft_data",
                        help="Directory containing sft_train.jsonl and sft_val.jsonl")
    parser.add_argument("--output_dir", type=str, default="sft_checkpoints",
                        help="Output directory for fine-tuned model")
    
    # Training
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Micro batch size per GPU")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=2e-6,
                        help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--save_interval", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--val_interval", type=int, default=250,
                        help="Run validation every N steps")
    
    # System
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not specified)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for faster training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model
    model, config = load_pretrained_model(args.checkpoint, device)
    
    # Compile if requested
    if args.compile and device == "cuda":
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Setup tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Setup data loaders
    train_path = os.path.join(args.data_dir, "sft_train.jsonl")
    val_path = os.path.join(args.data_dir, "sft_val.jsonl")
    
    if not os.path.exists(train_path):
        print(f"ERROR: Training data not found at {train_path}")
        print("Run sft_data_prep.py first to prepare the dataset.")
        sys.exit(1)
    
    train_loader = SFTDataLoader(
        train_path, tokenizer, config.block_size, 
        args.batch_size, device, shuffle=True
    )
    val_loader = SFTDataLoader(
        val_path, tokenizer, config.block_size,
        args.batch_size, device, shuffle=False
    )
    
    # Calculate steps
    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    effective_batch_size = args.batch_size * args.grad_accum
    
    print(f"\n{'='*60}")
    print("Training Configuration")
    print('='*60)
    print(f"Model type: {config.model_type.upper()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Epochs: {args.epochs}")
    print(f"Micro batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.grad_accum}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total optimizer steps: {total_steps}")
    print(f"Learning rate: {args.lr} -> {args.min_lr}")
    print(f"Warmup steps: {args.warmup_steps}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        fused=device == "cuda"  # Use fused kernel on GPU
    )
    
    # LR scheduler with warmup + cosine decay
    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        if step >= total_steps:
            return args.min_lr
        # Cosine decay
        progress = (step - args.warmup_steps) / (total_steps - args.warmup_steps)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save training config
    train_config = {
        "checkpoint": args.checkpoint,
        "model_type": config.model_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "total_steps": total_steps
    }
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=2)
    
    # Training loop
    model.train()
    optimizer.zero_grad()
    
    step = 0  # Optimizer steps
    micro_step = 0  # Total forward passes
    best_val_loss = float('inf')
    running_loss = 0.0
    
    t_start = time.time()
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print('='*60)
        
        train_loader.reset()
        
        for batch_idx in range(len(train_loader)):
            t0 = time.time()
            
            # Get batch
            x, y = train_loader.next_batch()
            
            # Forward pass with autocast
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                loss = compute_sft_loss(model, x, y)
                loss = loss / args.grad_accum  # Scale for accumulation
            
            # Backward pass
            loss.backward()
            running_loss += loss.item() * args.grad_accum
            micro_step += 1
            
            # Optimizer step after accumulation
            if micro_step % args.grad_accum == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Update LR
                lr = get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Step optimizer
                optimizer.step()
                optimizer.zero_grad()
                
                step += 1
                
                # Logging
                if step % args.log_interval == 0:
                    avg_loss = running_loss / args.log_interval
                    dt = (time.time() - t0) * 1000 / args.grad_accum
                    elapsed = time.time() - t_start
                    
                    # Calculate throughput
                    tokens_per_step = args.batch_size * args.grad_accum * config.block_size
                    tokens_per_sec = tokens_per_step / (dt * args.grad_accum / 1000)
                    
                    # ETA calculation
                    steps_remaining = total_steps - step
                    eta_seconds = steps_remaining * (dt * args.grad_accum / 1000)
                    eta_str = f"{eta_seconds/60:.0f}m" if eta_seconds < 3600 else f"{eta_seconds/3600:.1f}h"
                    
                    print(f"step {step:5d}/{total_steps} | loss: {avg_loss:.4f} | "
                          f"lr: {lr:.2e} | {tokens_per_sec/1000:.1f}k tok/s | "
                          f"dt: {dt:.0f}ms | ETA: {eta_str}")
                    running_loss = 0.0
                
                # Validation
                if step % args.val_interval == 0:
                    model.eval()
                    val_loader.reset()
                    val_loss = 0.0
                    val_batches = min(len(val_loader), 50)  # Cap validation
                    
                    with torch.no_grad():
                        for _ in range(val_batches):
                            x, y = val_loader.next_batch()
                            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                                loss = compute_sft_loss(model, x, y)
                            val_loss += loss.item()
                    
                    val_loss /= val_batches
                    print(f"  -> validation loss: {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = os.path.join(args.output_dir, f"sft_{config.model_type}_best.pt")
                        torch.save({
                            'model': model.state_dict(),
                            'config': asdict(config),
                            'step': step,
                            'val_loss': val_loss,
                            'epoch': epoch
                        }, best_path)
                        print(f"  -> new best! saved to {best_path}")
                    
                    model.train()
                
                # Save checkpoint
                if step % args.save_interval == 0:
                    ckpt_path = os.path.join(args.output_dir, f"sft_{config.model_type}_{step:05d}.pt")
                    torch.save({
                        'model': model.state_dict(),
                        'config': asdict(config),
                        'step': step,
                        'epoch': epoch,
                        'optimizer': optimizer.state_dict()
                    }, ckpt_path)
                    print(f"  -> checkpoint saved to {ckpt_path}")
    
    # Save final checkpoint
    final_path = os.path.join(args.output_dir, f"sft_{config.model_type}_final.pt")
    torch.save({
        'model': model.state_dict(),
        'config': asdict(config),
        'step': step,
        'epoch': args.epochs,
        'val_loss': best_val_loss
    }, final_path)
    
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print("Training Complete!")
    print('='*60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Final step: {step}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model: {final_path}")
    print(f"Best model: {os.path.join(args.output_dir, f'sft_{config.model_type}_best.pt')}")


if __name__ == "__main__":
    main()
