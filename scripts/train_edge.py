"""
Training Script for Hybrid SWA and MLA Transformers.

Based on the proven train_gpt2.py but adapted for:
- Both Hybrid (Gemma-style) and MLA (DeepSeek-style) models
- Automatic kernel selection based on hardware
- Integration with edu_fineweb10B.py data pipeline

Usage:
    # Single GPU
    python train_edge.py --model_type hybrid
    
    # Multi-GPU DDP
    torchrun --standalone --nproc_per_node=8 train_edge.py --model_type mla
"""

import os
import sys
import math
import time
import argparse
import threading
import queue
from dataclasses import asdict

# Add parent directory to path for models import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import numpy as np
import tiktoken

# Import our models (from parent/models/)
from models.config import ModelConfig, HYBRID_500M, MLA_500M
from models.transformer import Transformer



# =============================================================================
# DATA LOADING (from train_gpt2.py)
# =============================================================================

def load_tokens(filename):
    """
    Load tokenized data.
    Optimization: Keep as int16/int32 to save RAM, only cast to long when needed.
    """
    npt = np.memmap(filename, dtype=np.uint16, mode='r')
    # We cast to int32 because PyTorch doesn't support uint16. 
    # int32 is 2x smaller than int64 (default for Embedding), saving memory bandwidth.
    ptt = torch.from_numpy(npt.astype(np.int32))
    return ptt


class DataLoaderLite:
    """Lightweight data loader for sharded binary token files."""
    
    def __init__(self, B, T, process_rank, num_processes, split, data_root, master_process=True):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.master_process = master_process
        self.split = split
        assert split in {'train', 'val'}

        # Get shard filenames
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s and s.endswith('.bin')]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split {split} in {data_root}"
        
        if master_process:
            print(f"Found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        # In packed mode, we shift by B*T per process to ensure each GPU gets unique chunks
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        
        # We need B*T + 1 tokens (X is buf[:-1], Y is buf[1:])
        tokens_needed = B * T + 1
        
        # If we don't have enough tokens in the current shard, we must concatenate with the next
        if self.current_position + tokens_needed > len(self.tokens):
            # Get remaining tokens from current shard
            remaining = self.tokens[self.current_position:]
            
            # Move to next shard
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
            
            # Combine remaining with start of next shard
            buf = torch.cat([remaining, self.tokens[:tokens_needed - len(remaining)]])
            self.current_position += (tokens_needed - len(remaining))
        else:
            buf = self.tokens[self.current_position : self.current_position + tokens_needed]
            self.current_position += B * T * self.num_processes
            
        # Cast to long (int64) here, just for the batch
        buf = buf.to(torch.long)
        
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        
        # Handle wraparound/shard advance for next call
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
            
        return x, y


class PrefetchedWrapper:
    """
    OPTIMIZATION: Async Data Prefetching
    
    Loads next batch in background thread while GPU processes current batch.
    Combined with pin_memory() and non_blocking transfers, eliminates I/O stalls.
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
                # Get next batch from CPU loader
                x, y = self.loader.next_batch()
                
                # OPTIMIZATION: pin_memory() enables async DMA from CPU->GPU
                if x.device.type == 'cpu':
                    x = x.pin_memory()
                    y = y.pin_memory()
                    
                self.queue.put((x, y))
            except Exception as e:
                print(f"Error in prefetch worker: {e}")
                self.stop_event.set()
                return

    def next_batch(self):
        # Get from queue (blocking if empty, which means we wait for IO)
        x, y = self.queue.get()
        
        # OPTIMIZATION: non_blocking=True overlaps transfer with GPU compute
        if self.device != 'cpu':
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
        return x, y
        
    def reset(self):
        # For validation resets, we need to pause/reset the worker
        # Ideally, we just create a new wrapper for validation each time
        # or just rely on the queue draining.
        # Simple hack: Reuse loader but aware that queue has old data?
        # For val, it's okay to just use loader directly without prefetch if needed,
        # or rebuild.
        pass


# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine learning rate schedule with linear warmup."""
    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    # After max_steps, return min_lr
    if step > max_steps:
        return min_lr
    
    # Cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# =============================================================================
# GPU PROFILE DETECTION
# =============================================================================

def get_gpu_profile(device_idx=None):
    """
    Detect GPU and return optimal training settings.
    
    Note: Batch sizes must evenly divide total_batch_size / (T * world_size).
    With T=2048 and total_batch_size=524288, valid batch sizes are:
    1, 2, 4, 8, 16, 32, 64, 128, 256
    
    Args:
        device_idx: GPU device index (for DDP). If None, uses current device.
    
    Returns:
        dict with keys: name, vram_gb, batch_size, use_checkpointing, profile
    """
    if not torch.cuda.is_available():
        return {
            "name": "CPU",
            "vram_gb": 0,
            "batch_size": 4,
            "use_checkpointing": True,
            "profile": "cpu"
        }
    
    # Get GPU properties for current device (DDP-safe)
    if device_idx is None:
        device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    vram_gb = props.total_memory / (1024**3)
    name = props.name
    
    # Determine profile based on VRAM
    # B200: 192GB, H200: 141GB, H100: 80GB, A100: 40/80GB, RTX 3090: 24GB
    # Batch sizes must be powers of 2 or valid divisors of 256
    if vram_gb >= 180:
        # B200 (192GB) - No checkpointing needed, max batch size
        return {
            "name": name,
            "vram_gb": vram_gb,
            "batch_size": 64,  # Max that fits in 192GB
            "use_checkpointing": False,
            "profile": "b200"
        }
    elif vram_gb >= 130:
        # H200 (141GB) - No checkpointing needed, large batch size
        return {
            "name": name,
            "vram_gb": vram_gb,
            "batch_size": 32,  # Conservative for 141GB without checkpointing
            "use_checkpointing": False,
            "profile": "h200"
        }
    elif vram_gb >= 70:
        # H100 (80GB) - Selective checkpointing, standard batch size
        return {
            "name": name,
            "vram_gb": vram_gb,
            "batch_size": 16,
            "use_checkpointing": True,
            "profile": "h100"
        }
    elif vram_gb >= 35:
        # A100-40/80GB - Checkpointing, medium batch
        return {
            "name": name,
            "vram_gb": vram_gb,
            "batch_size": 8,
            "use_checkpointing": True,
            "profile": "a100"
        }
    else:
        # Consumer GPU (RTX 3090, etc) - Full checkpointing, small batch
        return {
            "name": name,
            "vram_gb": vram_gb,
            "batch_size": 4,
            "use_checkpointing": True,
            "profile": "consumer"
        }


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Hybrid or MLA Transformer")
    parser.add_argument("--model_type", type=str, default="hybrid", choices=["hybrid", "mla"],
                        help="Model architecture: 'hybrid' (Gemma-style) or 'mla' (DeepSeek-style)")
    parser.add_argument("--shards_dir", type=str, default="/lambda/nfs/riddy/edu_fineweb10B",
                        help="Directory containing tokenized shards (Lambda: /lambda/nfs/riddy/edu_fineweb10B)")
    parser.add_argument("--log_dir", type=str, default="log",
                        help="Directory for logs and checkpoints")
    parser.add_argument("--max_steps", type=int, default=19073,
                        help="Maximum training steps (~1 epoch for 10B tokens)")
    parser.add_argument("--val_interval", type=int, default=250,
                        help="Validation interval")
    parser.add_argument("--save_interval", type=int, default=5000,
                        help="Checkpoint save interval")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Micro batch size per GPU (auto-detected if --auto_optimize)")
    parser.add_argument("--total_batch_size", type=int, default=524288,
                        help="Total batch size in tokens (~0.5M)")
    parser.add_argument("--max_lr", type=float, default=6e-4,
                        help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=6e-5,
                        help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=715,
                        help="Learning rate warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile")
    parser.add_argument("--auto_optimize", action="store_true",
                        help="Auto-detect GPU and use optimal settings (batch size, checkpointing)")
    parser.add_argument("--no_checkpointing", action="store_true",
                        help="Disable gradient checkpointing (requires more VRAM, faster training)")
    args = parser.parse_args()

    # =========================================================================
    # DDP SETUP
    # =========================================================================
    ddp = int(os.environ.get('RANK', -1)) != -1
    
    if ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        if master_process:
            print(f"Using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    
    # Reproducibility
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # =========================================================================
    # GPU PROFILE AND AUTO-OPTIMIZATION
    # =========================================================================
    gpu_profile = get_gpu_profile()
    
    # Apply auto-optimization settings
    if args.auto_optimize:
        if args.batch_size is None:
            args.batch_size = gpu_profile["batch_size"]
        use_checkpointing = gpu_profile["use_checkpointing"]
        if master_process:
            print(f"\n{'='*60}")
            print(f"AUTO-OPTIMIZE: Detected {gpu_profile['name']}")
            print(f"  VRAM: {gpu_profile['vram_gb']:.1f} GB")
            print(f"  Profile: {gpu_profile['profile'].upper()}")
            print(f"  Batch Size: {args.batch_size}")
            print(f"  Checkpointing: {'Enabled (selective)' if use_checkpointing else 'Disabled'}")
            print(f"{'='*60}\n")
    else:
        # Default batch size if not specified
        if args.batch_size is None:
            args.batch_size = 16
        # Use checkpointing unless explicitly disabled
        use_checkpointing = not args.no_checkpointing
        if master_process:
            print(f"\n{'='*60}")
            print(f"GPU: {gpu_profile['name']} ({gpu_profile['vram_gb']:.1f} GB)")
            print(f"Profile: {gpu_profile['profile'].upper()} (use --auto_optimize for auto-tuning)")
            print(f"Batch Size: {args.batch_size}")
            print(f"Checkpointing: {'Enabled' if use_checkpointing else 'Disabled'}")
            print(f"{'='*60}\n")

    # =========================================================================
    # MODEL SETUP
    # =========================================================================
    if master_process:
        print(f"\n{'='*60}")
        print(f"Training {args.model_type.upper()} Transformer")
        from models.attention import FLEX_AVAILABLE, FLASH_MLA_AVAILABLE
        print(f"Flex attention available: {FLEX_AVAILABLE}")
        print(f"FlashMLA (SDPA) available: {FLASH_MLA_AVAILABLE}")
        print(f"Using torch.compile: {args.compile}")
        print(f"{'='*60}\n")
    
    # Select config based on model type
    if args.model_type == "hybrid":
        config = HYBRID_500M
    else:
        config = MLA_500M
    
    # Determine mode based on hardware
    mode = "train"  # Will use optimized kernels if available
    
    # Pass checkpointing flag to model
    model = Transformer(config, mode=mode, use_checkpointing=use_checkpointing)
    model.to(device)
    
    if args.compile:
        if master_process:
            print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    raw_model = model.module if ddp else model

    # =========================================================================
    # DATA LOADER SETUP
    # =========================================================================
    B = args.batch_size
    T = config.block_size
    
    assert args.total_batch_size % (B * T * ddp_world_size) == 0, \
        "total_batch_size must be divisible by B * T * world_size"
    grad_accum_steps = args.total_batch_size // (B * T * ddp_world_size)
    
    if master_process:
        print(f"Total batch size: {args.total_batch_size:,} tokens")
        print(f"Micro batch size: {B} sequences x {T} tokens = {B*T:,} tokens/GPU")
        print(f"Gradient accumulation steps: {grad_accum_steps}")
        print(f"Effective batch: {B * T * grad_accum_steps * ddp_world_size:,} tokens")

    train_loader_inner = DataLoaderLite(
        B=B, T=T, 
        process_rank=ddp_rank, 
        num_processes=ddp_world_size,
        split="train", 
        data_root=args.shards_dir,
        master_process=master_process
    )
    val_loader_inner = DataLoaderLite(
        B=B, T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="val",
        data_root=args.shards_dir,
        master_process=master_process
    )
    
    # Wrap with prefetcher (only for training is critical, but val helps too)
    train_loader = PrefetchedWrapper(train_loader_inner, device)
    val_loader = val_loader_inner # Keep val simple or wrap if needed

    # =========================================================================
    # OPTIMIZER SETUP
    # =========================================================================
    # OPTIMIZATION: Enable TF32 for matmuls (H100/A100) and cuDNN auto-tuning
    torch.set_float32_matmul_precision('high')  # Allows TF32 on Tensor Cores
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for matmuls
    torch.backends.cudnn.allow_tf32 = True        # Enable TF32 for cuDNN ops
    torch.backends.cudnn.benchmark = True         # Auto-tune convolution algorithms
    
    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.max_lr,
        device_type=device_type
    )

    # =========================================================================
    # LOGGING SETUP
    # =========================================================================
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"log_{args.model_type}.txt")
    if master_process:
        with open(log_file, "w") as f:
            f.write(f"Training {args.model_type} model\n")
            f.write(f"Config: {asdict(config)}\n\n")

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    enc = tiktoken.get_encoding("gpt2")
    
    for step in range(args.max_steps):
        t0 = time.time()
        last_step = (step == args.max_steps - 1)

        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------
        if step % args.val_interval == 0 or last_step:
            model.eval()
            val_loader.reset()
            
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        _, loss, _ = model(x, targets=y)
                    
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            
            if master_process:
                print(f"Step {step:5d} | val loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")

        # ---------------------------------------------------------------------
        # CHECKPOINTING
        # ---------------------------------------------------------------------
        if master_process and step > 0 and (step % args.save_interval == 0 or last_step):
            checkpoint_path = os.path.join(args.log_dir, f"{args.model_type}_{step:05d}.pt")
            checkpoint = {
                'model': raw_model.state_dict(),
                'config': asdict(config),
                'step': step,
                'val_loss': val_loss_accum.item() if 'val_loss_accum' in dir() else None,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # ---------------------------------------------------------------------
        # SAMPLE GENERATION
        # ---------------------------------------------------------------------
        if master_process and ((step > 0 and step % args.val_interval == 0) or last_step):
            model.eval()
            prompt = "The future of artificial intelligence"
            tokens = enc.encode(prompt)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            
            with torch.no_grad():
                generated = raw_model.generate(tokens, max_new_tokens=50, temperature=0.8)
            
            text = enc.decode(generated[0].tolist())
            print(f"\nGenerated: {text}\n")

        # ---------------------------------------------------------------------
        # TRAINING STEP
        # ---------------------------------------------------------------------
        model.train()
        optimizer.zero_grad()
        loss_accum = torch.zeros(1, device=device)
        
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss, _ = model(x, targets=y)
            
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        
        # Gradient clipping
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Learning rate schedule
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.max_lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.step()
        
        if device_type == "cuda":
            torch.cuda.synchronize()
        
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = B * T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr: {lr:.2e} | "
                  f"norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:,.0f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    # =========================================================================
    # CLEANUP
    # =========================================================================
    if ddp:
        destroy_process_group()
    
    if master_process:
        print("\nTraining complete!")


if __name__ == "__main__":
    main()
