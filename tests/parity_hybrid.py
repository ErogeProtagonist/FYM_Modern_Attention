"""
Train-mode vs inference-mode logit parity test for the Hybrid SWA+GQA model.

Re-confirms the 2.4e-05 max-diff result reported in the dissertation methodology
chapter, after the Bug 5 GQA-cache fix moved `repeat_interleave` to occur after
the cache write in all three Hybrid attention classes.

What this test compares:
    - Model A: Hybrid loaded with mode="train"
        -> FlashSWAHybridAttention (flash_attn package) if available, otherwise
           NaiveHybridAttention via SDPA. With use_cache=False the forward takes
           the *training* branch of whichever class is selected.
    - Model B: Hybrid loaded with mode="inference"
        -> NaiveHybridAttention via SDPA (post-Bug-5 fix).

Both models receive the same state_dict and the same fixed input. We then
compare logits and report the max absolute difference.

Note: on hardware without flash_attn (e.g. local 3090), both branches resolve
to NaiveHybridAttention and the parity test becomes trivially zero. Run on a
machine with flash_attn installed for a meaningful comparison.

Usage:
    cd swa-mla-500m
    python -m tests.parity_hybrid --checkpoint ../Checkpoints/hybrid_19073.pt
"""

import argparse
import os
import sys

import torch

# Add parent directory so `models` is importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import ModelConfig
from models.transformer import Transformer


def _strip_prefixes(state_dict):
    """Match the prefix-handling in scripts/evaluate.py:load_model."""
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    if any(".naive_impl." in k for k in state_dict.keys()):
        state_dict = {k.replace(".naive_impl.", "."): v for k, v in state_dict.items()}
    return state_dict


def _build(checkpoint_path, mode, device, dtype):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig(**ckpt["config"])
    assert config.model_type == "hybrid", (
        f"This parity test is for the Hybrid model only; got {config.model_type}"
    )
    model = Transformer(config, mode=mode)
    model.load_state_dict(_strip_prefixes(ckpt["model"]))
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="float32 gives the cleanest comparison; bf16 matches deployment.",
    )
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--rtol", type=float, default=1e-4,
        help="Relative tolerance for the allclose check.",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-4,
        help="Absolute tolerance for the allclose check.",
    )
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}  dtype: {args.dtype}\n")

    print("Building train-mode model...")
    model_train, config = _build(args.checkpoint, "train", args.device, dtype)
    print("Building inference-mode model...")
    model_infer, _ = _build(args.checkpoint, "inference", args.device, dtype)

    # Fixed input
    torch.manual_seed(args.seed)
    seq_len = min(args.seq_len, config.block_size)
    input_ids = torch.randint(
        0, config.vocab_size, (args.batch, seq_len), device=args.device
    )

    print(f"\nRunning forward passes (B={args.batch}, S={seq_len})...")
    with torch.no_grad():
        out_train = model_train(input_ids, use_cache=False)
        out_infer = model_infer(input_ids, use_cache=False)

    # Both forwards return (logits, loss, kv_cache); take logits
    logits_train = out_train[0].float()
    logits_infer = out_infer[0].float()

    diff = (logits_train - logits_infer).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    allclose = torch.allclose(logits_train, logits_infer, rtol=args.rtol, atol=args.atol)

    print("\n" + "=" * 60)
    print("Hybrid train-mode vs inference-mode logit parity")
    print("=" * 60)
    print(f"Logits shape   : {tuple(logits_train.shape)}")
    print(f"Max abs diff   : {max_diff:.3e}")
    print(f"Mean abs diff  : {mean_diff:.3e}")
    print(f"allclose       : {allclose}  (rtol={args.rtol}, atol={args.atol})")

    if max_diff < 1e-3:
        print("\nPASS  -- well within float accumulation noise; the post-Bug-5")
        print("        inference path is numerically equivalent to training.")
    else:
        print("\nFAIL  -- max diff exceeds 1e-3; investigate.")
        sys.exit(1)


if __name__ == "__main__":
    main()
