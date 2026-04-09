"""
Sanity check: print actual KV cache tensor shapes for one forward pass.

Goal: confirm whether the Hybrid cache stores tensors at n_heads or n_kv_heads.
If at n_heads, the GQA expansion is happening before the cache write
(memory-suboptimal — cache is n_rep× larger than necessary).

Usage:
    python check_kv_cache_shapes.py --checkpoint ../../Checkpoints/hybrid_19072.pt
    python check_kv_cache_shapes.py --checkpoint ../../Checkpoints/mla_19072.pt
"""

import argparse
import torch
from evaluate import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ctx_len", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, config = load_model(args.checkpoint, device=args.device)

    print(f"\nConfig: model_type={config.model_type}, n_heads={config.n_heads}, "
          f"n_kv_heads={getattr(config, 'n_kv_heads', 'n/a')}, head_dim={config.head_dim}")
    print(f"n_rep (n_heads / n_kv_heads) = "
          f"{config.n_heads // getattr(config, 'n_kv_heads', config.n_heads)}")

    input_ids = torch.zeros((1, args.ctx_len), dtype=torch.long, device=args.device)

    with torch.no_grad():
        out = model(input_ids, use_cache=True)

    # Model returns (logits, loss, kv_cache)
    kv_cache = out[-1]
    assert kv_cache is not None, "Model did not return a KV cache"

    print(f"\nNumber of cached layers: {len([c for c in kv_cache if c is not None])}")
    print(f"Input ctx_len: {args.ctx_len}\n")

    print(f"{'Layer':>6} | {'Tensor':>10} | {'Shape':<35} | {'Bytes':>12}")
    print("-" * 75)
    total_bytes = 0
    for i, layer_cache in enumerate(kv_cache):
        if layer_cache is None:
            print(f"{i:>6} | (None)")
            continue
        a, b = layer_cache
        a_bytes = a.nelement() * a.element_size()
        b_bytes = b.nelement() * b.element_size()
        total_bytes += a_bytes + b_bytes

        if config.model_type == "mla":
            print(f"{i:>6} | {'c_kv':>10} | {str(list(a.shape)):<35} | {a_bytes:>12,}")
            print(f"{i:>6} | {'k_rope':>10} | {str(list(b.shape)):<35} | {b_bytes:>12,}")
        else:
            print(f"{i:>6} | {'K':>10} | {str(list(a.shape)):<35} | {a_bytes:>12,}")
            print(f"{i:>6} | {'V':>10} | {str(list(b.shape)):<35} | {b_bytes:>12,}")

    print("-" * 75)
    print(f"Total cache: {total_bytes:,} bytes = {total_bytes / 1024 / 1024:.2f} MB")

    # Diagnostic: for Hybrid, check whether cache dim 1 is n_heads or n_kv_heads
    if config.model_type != "mla":
        first = kv_cache[0][0]  # K of layer 0
        cache_heads = first.shape[1]
        if cache_heads == config.n_heads:
            print(f"\n>>> CACHE STORES AT n_heads={config.n_heads} "
                  f"(GQA expansion happens BEFORE cache write — cache is "
                  f"{config.n_heads // config.n_kv_heads}x larger than optimal)")
        elif cache_heads == config.n_kv_heads:
            print(f"\n>>> Cache stores at n_kv_heads={config.n_kv_heads} (optimal GQA caching)")
        else:
            print(f"\n>>> Unexpected cache head dim: {cache_heads}")


if __name__ == "__main__":
    main()
