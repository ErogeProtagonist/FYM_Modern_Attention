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
        --checkpoint ../Checkpoints/hybrid_19072.pt \\
        --data_dir sft_data \\
        --batch_size 4 \\
        --epochs 3

    # Lower memory with gradient accumulation
    python sft_train.py \\
        --checkpoint ../Checkpoints/mla_19072.pt \\
        --data_dir sft_data \\
        --batch_size 2 \\
        --grad_accum 8
"""

import os
import sys
import json
import math
import time
import queue
import argparse
import platform
import threading
from dataclasses import asdict

# Add parent directory to path for models import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

from models.config import ModelConfig
from models.transformer import Transformer


# =============================================================================
# GPU PROFILE DETECTION (same as train_edge.py)
# =============================================================================

def get_gpu_profile():
    """
    Detect GPU and return optimal SFT settings.
    
    Returns:
        dict with keys: name, vram_gb, batch_size, profile
    """
    if not torch.cuda.is_available():
        return {
            "name": "CPU",
            "vram_gb": 0,
            "batch_size": 2,
            "profile": "cpu"
        }
    
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    name = props.name.upper()
    
    # B200: 192GB, H200: 141GB, H100: 80GB, A100: 40/80GB, 3090: 24GB
    if "B200" in name or vram_gb >= 180:
        return {"name": props.name, "vram_gb": vram_gb, "batch_size": 32, "profile": "b200"}
    elif "H200" in name or vram_gb >= 130:
        return {"name": props.name, "vram_gb": vram_gb, "batch_size": 32, "profile": "h200"}
    elif vram_gb >= 70:
        return {"name": props.name, "vram_gb": vram_gb, "batch_size": 16, "profile": "h100"}
    elif vram_gb >= 35:
        return {"name": props.name, "vram_gb": vram_gb, "batch_size": 8, "profile": "a100"}
    else:
        return {"name": props.name, "vram_gb": vram_gb, "batch_size": 2, "profile": "consumer"}


# =============================================================================
# ASYNC PREFETCHING (same as train_edge.py)
# =============================================================================

class PrefetchedWrapper:
    """
    Async Data Prefetching - loads next batch in background thread.
    Combined with pin_memory() and non_blocking transfers.
    """
    def __init__(self, loader, device, prefetch_factor=2):
        self.loader = loader
        self.device = device
        self.queue = queue.Queue(maxsize=prefetch_factor)
        self.stop_event = threading.Event()
        self.loader_thread = threading.Thread(target=self._worker, daemon=True)
        self.loader_thread.start()
        
    def _worker(self):
        while not self.stop_event.is_set():
            try:
                x, y = self.loader.next_batch()
                # pin_memory enables async DMA from CPU->GPU
                if x.device.type == 'cpu':
                    x = x.pin_memory()
                    y = y.pin_memory()
                self.queue.put((x, y))
            except Exception as e:
                print(f"Error in prefetch worker: {e}")
                self.stop_event.set()
                return

    def next_batch(self):
        x, y = self.queue.get()
        # non_blocking=True overlaps transfer with GPU compute
        if self.device != 'cpu':
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        return x, y
        
    def reset(self):
        # Drain queue and reset loader
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                pass
        self.loader.reset()


class SFTDataLoader:
    """
    High-performance dataloader for SFT data.
    
    OPTIMIZATION: Pre-builds all data as tensors at startup.
    This eliminates the Python list → tensor conversion bottleneck.
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
        
        # Load and tokenize all examples
        examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                examples.append(json.loads(line.strip()))
        
        print(f"Loaded {len(examples)} examples from {data_path}")
        
        # Pre-tokenize and pad all examples
        print("Pre-tokenizing and building tensor cache...")
        all_tokens = []
        all_labels = []
        
        for ex in examples:
            prompt = ex["prompt"]
            completion = ex["completion"]
            
            # Tokenize (allow ChatML special tokens)
            prompt_tokens = tokenizer.encode(prompt, allowed_special="all")
            completion_tokens = tokenizer.encode(completion, allowed_special="all")
            
            # Build sequence with EOS
            full_tokens = prompt_tokens + completion_tokens + [tokenizer.eot_token]
            
            # Build labels: -100 for prompt, actual tokens for completion
            labels = (
                [-100] * len(prompt_tokens) +
                completion_tokens +
                [tokenizer.eot_token]
            )
            
            # Truncate if needed
            if len(full_tokens) > block_size:
                full_tokens = full_tokens[:block_size]
                labels = labels[:block_size]
            
            # Pad to block_size
            pad_len = block_size - len(full_tokens)
            if pad_len > 0:
                full_tokens = full_tokens + [tokenizer.eot_token] * pad_len
                labels = labels + [-100] * pad_len
            
            all_tokens.append(full_tokens)
            all_labels.append(labels)
        
        # Convert to tensors ONCE at startup (this is the key optimization)
        # Using int32 saves memory bandwidth (like pre-training)
        self.tokens = torch.tensor(all_tokens, dtype=torch.int32)
        self.labels = torch.tensor(all_labels, dtype=torch.int32)
        self.num_examples = len(all_tokens)
        
        print(f"Built tensor cache: {self.tokens.shape} ({self.tokens.nbytes / 1e9:.2f} GB)")
        
        # Shuffle indices
        self.indices = torch.randperm(self.num_examples) if shuffle else torch.arange(self.num_examples)
        self.current_idx = 0
    
    def reset(self):
        """Reset for new epoch."""
        if self.shuffle:
            self.indices = torch.randperm(self.num_examples)
        self.current_idx = 0
    
    def __len__(self):
        return self.num_examples // self.batch_size
    
    def next_batch(self):
        """Get next batch using fast tensor indexing."""
        # Handle wraparound
        if self.current_idx + self.batch_size > self.num_examples:
            self.current_idx = 0
            if self.shuffle:
                self.indices = torch.randperm(self.num_examples)
        
        # Get batch indices
        batch_indices = self.indices[self.current_idx : self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        
        # Fast tensor indexing (no Python loops!)
        x = self.tokens[batch_indices].to(torch.long)  # Cast to long for embedding
        y = self.labels[batch_indices].to(torch.long)
        
        return x, y


def load_pretrained_model(checkpoint_path: str, device: str):
    """Load pretrained model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Reconstruct config
    config_dict = checkpoint['config']
    config = ModelConfig(**config_dict)
    
    # Create model in training mode for optimized attention kernels
    # Disable checkpointing - model is small enough
    model = Transformer(config, mode="train", use_checkpointing=False)

    # Handle compiled model checkpoints
    state_dict = checkpoint['model']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("Stripping '_orig_mod.' prefix from compiled checkpoint...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Strip the legacy 'naive_impl.' prefix that older MLA checkpoints carry.
    # Pre-training and earlier SFT runs used the FlashMLAttention wrapper class,
    # which embedded its real layers under `<...>.attn.naive_impl.<param>`.
    # FlashMLAttention has since been removed and MLA uses NaiveMLAttention
    # directly, so the prefix must be dropped to match the new key layout.
    # New checkpoints saved after this change will not contain the prefix; the
    # `if any(...)` guard makes this a no-op for them.
    if any('.naive_impl.' in k for k in state_dict.keys()):
        print("Stripping legacy 'naive_impl.' prefix from MLA checkpoint...")
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
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Stop after N optimizer steps (overrides epoch end)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Micro batch size (auto-detected if --auto_optimize)")
    parser.add_argument("--grad_accum", type=int, default=None,
                        help="Gradient accumulation steps (auto-calculated if not set)")
    parser.add_argument("--total_batch_size", type=int, default=65536,
                        help="Target total batch size in tokens (~64k for SFT)")
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
    parser.add_argument("--auto_optimize", action="store_true",
                        help="Auto-detect GPU and use optimal batch size")
    
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
    
    # =========================================================================
    # GPU PROFILE AND AUTO-OPTIMIZATION
    # =========================================================================
    gpu_profile = get_gpu_profile()
    
    if args.auto_optimize or args.batch_size is None:
        args.batch_size = gpu_profile["batch_size"]
        print(f"\n{'='*60}")
        print(f"AUTO-OPTIMIZE: Detected {gpu_profile['name']}")
        print(f"  VRAM: {gpu_profile['vram_gb']:.1f} GB")
        print(f"  Profile: {gpu_profile['profile'].upper()}")
        print(f"  Micro Batch Size: {args.batch_size}")
        print(f"{'='*60}")
    else:
        print(f"GPU: {gpu_profile['name']} ({gpu_profile['vram_gb']:.1f} GB)")
    
    # Load model
    model, config = load_pretrained_model(args.checkpoint, device)
    
    # Calculate gradient accumulation
    # Priority: 1. Manual --grad_accum  2. Auto-calc from --total_batch_size
    T = config.block_size
    tokens_per_micro = args.batch_size * T
    
    if args.grad_accum is not None:
        grad_accum = args.grad_accum
        print(f"Manual gradient accumulation: {grad_accum}")
    else:
        if args.total_batch_size % tokens_per_micro != 0:
            grad_accum = max(1, args.total_batch_size // tokens_per_micro)
            actual_total = tokens_per_micro * grad_accum
            print(f"Note: Adjusted total_batch_size from {args.total_batch_size} to {actual_total}")
        else:
            grad_accum = args.total_batch_size // tokens_per_micro
    
    print(f"Gradient accumulation: {grad_accum} (total_batch={args.batch_size * grad_accum * T:,} tokens)")
    
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
    
    train_loader_inner = SFTDataLoader(
        train_path, tokenizer, config.block_size, 
        args.batch_size, device, shuffle=True
    )
    val_loader_inner = SFTDataLoader(
        val_path, tokenizer, config.block_size,
        args.batch_size, device, shuffle=False
    )
    
    # Wrap with async prefetching for overlapped GPU transfer
    train_loader = PrefetchedWrapper(train_loader_inner, device)
    val_loader = val_loader_inner  # Val doesn't need prefetch (small batches)
    
    # Calculate steps (use inner loader for len since wrapper doesn't have it)
    steps_per_epoch = len(train_loader_inner) // grad_accum
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    effective_batch_size = args.batch_size * grad_accum
    
    print(f"\n{'='*60}")
    print("Training Configuration")
    print('='*60)
    print(f"Model type: {config.model_type.upper()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Epochs: {args.epochs}")
    print(f"Micro batch size: {args.batch_size}")
    print(f"Gradient accumulation: {grad_accum}")
    print(f"Effective batch size: {effective_batch_size} sequences ({effective_batch_size * T:,} tokens)")
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
        "grad_accum": grad_accum,
        "lr": args.lr,
        "total_batch_size": args.batch_size * grad_accum * T,
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
        
        for batch_idx in range(len(train_loader_inner)):
            # Only start timing at the beginning of each optimizer step
            if micro_step % grad_accum == 0:
                t0 = time.time()
            
            # Get batch
            x, y = train_loader.next_batch()
            
            # Forward pass with autocast
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                loss = compute_sft_loss(model, x, y)
                loss = loss / grad_accum  # Scale for accumulation
            
            # Backward pass
            loss.backward()
            running_loss += loss.item() * grad_accum
            micro_step += 1
            
            # Optimizer step after accumulation
            if micro_step % grad_accum == 0:
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
                    avg_loss = running_loss / (args.log_interval * grad_accum)  # Divide by total micro-batches
                    dt = (time.time() - t0) * 1000 / grad_accum
                    elapsed = time.time() - t_start
                    
                    # Calculate throughput
                    tokens_per_step = args.batch_size * grad_accum * config.block_size
                    tokens_per_sec = tokens_per_step / (dt * grad_accum / 1000)
                    
                    # ETA calculation
                    steps_remaining = total_steps - step
                    eta_seconds = steps_remaining * (dt * grad_accum / 1000)
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
                            x, y = x.to(device), y.to(device)  # Manual transfer for val
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

                if args.max_steps is not None and step >= args.max_steps:
                    print(f"Reached --max_steps={args.max_steps}; stopping.")
                    break
        if args.max_steps is not None and step >= args.max_steps:
            break

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
