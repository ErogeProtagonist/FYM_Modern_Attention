"""
Sanity check for the Hybrid SWA rolling-buffer cache.

Runs a local layer and a global layer through 6 generation steps with a
window_size of 4, and confirms:
  - Local layer cache stops growing once it hits window_size.
  - Global layer cache grows linearly with the number of tokens.

Uses a fresh ModelConfig rather than mutating HYBRID_500M so the rest of
the process state is unaffected.
"""

import torch

from models.config import ModelConfig
from models.attention import NaiveHybridAttention


def verify_swa_shapes():
    print("Verifying SWA KV Cache Shapes...")

    # Small test config: window=4, period=2 (layer 0 local, layer 1 global).
    # Kept structurally identical to HYBRID_500M otherwise so the attention
    # class's assumptions about n_heads / n_kv_heads / head_dim still hold.
    config = ModelConfig(
        model_type="hybrid",
        d_model=1280,
        n_layers=24,
        n_heads=20,
        n_kv_heads=4,
        window_size=4,
        global_layer_period=2,
    )

    print(f"Test Configuration: Window={config.window_size}, "
          f"Global Period={config.global_layer_period}")

    local_attn = NaiveHybridAttention(config, layer_idx=0)   # local (SWA)
    global_attn = NaiveHybridAttention(config, layer_idx=1)  # global

    B = 1
    num_steps = 6

    cache_local = None
    cache_global = None

    print("\n--- Starting Generation Steps ---")

    for i in range(num_steps):
        x = torch.randn(B, 1, config.d_model)  # single-token input
        # Absolute position of the new token = number of tokens already seen.
        position_ids = torch.tensor([i])

        _, cache_local = local_attn(
            x, position_ids=position_ids, kv_cache=cache_local, use_cache=True
        )
        _, cache_global = global_attn(
            x, position_ids=position_ids, kv_cache=cache_global, use_cache=True
        )

        k_local, _ = cache_local
        k_global, _ = cache_global

        print(f"Step {i+1}:")
        capped = " (Capped at window!)" if k_local.shape[2] == config.window_size else ""
        print(f"  Local Layer Cache Shape:  {list(k_local.shape)}{capped}")
        print(f"  Global Layer Cache Shape: {list(k_global.shape)}")

    if (cache_local[0].shape[2] == config.window_size
            and cache_global[0].shape[2] == num_steps):
        print("\nSUCCESS: Local layer capped at window size, Global layer grew indefinitely.")
    else:
        print("\nFAILURE: Cache shapes did not match expected behavior.")


if __name__ == "__main__":
    verify_swa_shapes()
