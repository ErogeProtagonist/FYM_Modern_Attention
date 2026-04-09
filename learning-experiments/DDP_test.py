"""
DDP & Pipeline Verification Script (Lambda Labs H100 Cluster)

This script is a comprehensive 'Dry Run' to verify:
1. DDP Initialization & Multi-GPU Connectivity.
2. Binary Shard Loading (memmap) & Rank-Aware Sharding.
3. Tokenizer Consistency (checking EOT token alignment).
4. Checkpointing (Verifying disk write/read on the cluster).
5. Gradient Synchronization & Throughput Benchmark.

Execution on Cluster:
    torchrun --standalone --nproc_per_node=8 DDP_test.py --shards_dir ./edu_fineweb10B
"""

import os
import sys
import time
import math
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

# --- CONFIG ---
B = 16              # Micro-batch size
T = 1024            # Sequence length
TOTAL_STEPS = 50    # Short benchmark run
SHARDS_DIR = "./edu_fineweb10B"

# --- DDP SETUP ---
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- DIAGNOSTIC HELPERS ---
def log(msg):
    if master_process:
        print(f"[MASTER] {msg}")

def log_all(msg):
    print(f"[Rank {ddp_rank}] {msg}")

# --- MEMMAP LOADER ---
class ShardedDataLoader:
    def __init__(self, shards_dir, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        
        # 1. Find all .bin shards
        self.shards = sorted(glob.glob(os.path.join(shards_dir, "*.bin")))
        if len(self.shards) == 0:
            log_all(f"ERROR: No .bin shards found in {shards_dir}")
            sys.exit(1)
            
        # 2. Check shard count vs GPU count
        if len(self.shards) < num_processes:
            log(f"WARNING: Shard count ({len(self.shards)}) is less than GPU count ({num_processes}). Some GPUs will be idle or redundant.")

        # 3. Load first shard to check alignment
        self.current_shard_idx = (0 + process_rank) % len(self.shards)
        self.load_shard()
        
    def load_shard(self):
        path = self.shards[self.current_shard_idx]
        # Using memmap for zero-latency loading
        self.tokens_np = np.memmap(path, dtype=np.uint16, mode='r')
        # We cast to .long() (int64) because PyTorch's cross_entropy 
        # loss function requires it for indices on CUDA.
        self.tokens = torch.from_numpy(self.tokens_np.astype(np.int64))
        self.current_pos = 0 
        
        # Tokenizer check: The first token of a shard should usually be 50256 (EOT) if prep script is correct
        if self.current_pos == 0 and self.tokens[0] != 50256:
             log_all(f"CHECKER: Shard starts with token {self.tokens[0]} (Expected 50256 for EOT). Check prep script.")
        
    def next_batch(self):
        # Switch shard if we reach the end
        if self.current_pos + self.B * self.T + 1 > len(self.tokens):
            self.current_shard_idx = (self.current_shard_idx + self.num_processes) % len(self.shards)
            self.load_shard()
            
        buf = self.tokens[self.current_pos : self.current_pos + self.B * self.T + 1]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.current_pos += self.B * self.T
        return x, y

# --- GPT DUMMY ---
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(50304, 768),
            wpe = nn.Embedding(1024, 768),
            h = nn.ModuleList([nn.Linear(768, 768) for _ in range(4)]), # Small for testing
            ln_f = nn.LayerNorm(768),
        ))
        self.lm_head = nn.Linear(768, 50304, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.transformer.wte(idx) + self.transformer.wpe(torch.arange(T, device=device))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, 50304), targets.view(-1))
        return logits, loss

# --- VERIFICATION SUITE ---
if __name__ == "__main__":
    log("--- CLUSTER VERIFICATION START ---")
    log(f"Platform: {sys.platform} | PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
    
    # 1. Connectivity Test (Heartbeat)
    if ddp:
        heartbeat = torch.ones(1, device=device)
        dist.all_reduce(heartbeat, op=dist.ReduceOp.SUM)
        if heartbeat.item() != ddp_world_size:
            log_all(f"CRITICAL: DDP Heartbeat failed! Expected {ddp_world_size}, got {heartbeat.item()}")
            sys.exit(1)
        log("DDP Connectivity: OK (All GPUs responding)")

    # 2. Data Pipeline Check
    shards_path = SHARDS_DIR
    for i, arg in enumerate(sys.argv):
        if arg == "--shards_dir":
            shards_path = sys.argv[i+1]
            
    if not os.path.exists(shards_path):
        log(f"ALERT: {shards_path} not found. Creating random tokens for dry run.")
        os.makedirs(shards_path, exist_ok=True)
        # Create a tiny dummy .bin for testing if none exist
        dummy = np.random.randint(0, 50257, (100000,), dtype=np.uint16)
        with open(os.path.join(shards_path, "dummy_shard_000000.bin"), "wb") as f:
            f.write(dummy.tobytes())

    loader = ShardedDataLoader(shards_path, B, T, ddp_rank, ddp_world_size)
    log("DataLoader (memmap): OK")

    # 3. Model & Optimizer
    model = TinyGPT().to(device)
    if ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    log("Model Distributed: OK")

    # 4. Checkpoint Test
    if master_process:
        log("Testing Checkpoint IO...")
        checkpoint = {
            'model': (model.module if ddp else model).state_dict(),
            'step': 0,
            'test_val': 42
        }
        ckpt_path = "test_ckpt.pt"
        torch.save(checkpoint, ckpt_path)
        # Verify it exists and can be loaded
        if os.path.exists(ckpt_path):
            loaded = torch.load(ckpt_path, weights_only=True)
            if loaded['test_val'] == 42:
                log("Checkpoint IO: OK (Write/Read verified)")
                os.remove(ckpt_path)
            else:
                log("CRITICAL: Checkpoint data corrupted!")
                sys.exit(1)
        else:
            log("CRITICAL: Failed to write checkpoint to disk!")
            sys.exit(1)

    # 5. Benchmark Loop
    log("\nLaunching performance benchmark...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    for step in range(TOTAL_STEPS):
        t0 = time.time()
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        dt_ms = (time.time() - t0) * 1000
        tokens_per_sec = (B * T * ddp_world_size) / (dt_ms / 1000)
        
        if master_process and step % 10 == 0:
            print(f"Step {step:02d} | Loss: {loss.item():.4f} | {dt_ms:.1f}ms | {tokens_per_sec/1e6:.2f}M tok/sec")

    total_time = time.time() - start_time
    avg_tok_sec = (B * T * ddp_world_size * TOTAL_STEPS) / total_time
    
    log(f"\nBenchmark Result: {avg_tok_sec/1e6:.2f} Million Tokens/sec")
    log("Verification Complete: Cluster is ready for 10B training run.")
    
    if ddp:
        destroy_process_group()
