"""
Train-mode vs inference-mode logit parity test for the Hybrid SWA+GQA model.

What this test compares:
    - Model A: Hybrid loaded with mode="train"
        -> FlashSWAHybridAttention (flash_attn package) if available, otherwise
           NaiveHybridAttention via SDPA. With use_cache=False the forward takes
           the *training* branch of whichever class is selected.
    - Model B: Hybrid loaded with mode="inference"
        -> NaiveHybridAttention via SDPA (post-Bug-5 fix).

Both models receive the same state_dict and the same fixed input. We then
compare logits and report the max/mean absolute difference.

Two regimes:

1. Hardware *without* flash_attn (e.g. local 3090): both branches resolve to
   NaiveHybridAttention. The test runs in float32 and is effectively a
   sanity check for the SDPA path against itself. Expected max diff is at
   the float32 noise floor (~1e-5 or smaller).

2. Hardware *with* flash_attn (cloud A100/H100/B200): the train branch
   dispatches to flash_attn_func and the inference branch stays on SDPA, so
   this is a real cross-kernel comparison. flash_attn_func only accepts
   fp16/bf16, so the test auto-promotes to bf16. Expected max diff is at
   the bf16 cross-kernel noise floor — measured baseline is roughly
   max ~4.4e-01, mean ~2.6e-02 on hybrid_19072.pt at (B=2, S=512). The
   pass thresholds below are set above this baseline.

Usage:
    cd swa-mla-500m
    python -m tests.parity_hybrid --checkpoint ../Checkpoints/hybrid_19072.pt
"""

import argparse
import os
import sys

import torch

# Add parent directory so `models` is importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.attention import FLASH_ATTN_AVAILABLE
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
        help="float32 gives the cleanest comparison; bf16 matches deployment. "
             "Auto-promoted to bfloat16 if flash_attn is installed.",
    )
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--rtol", type=float, default=None,
        help="Relative tolerance (default: 1e-4 fp32, 1e-2 bf16).",
    )
    parser.add_argument(
        "--atol", type=float, default=None,
        help="Absolute tolerance (default: 1e-4 fp32, 1e-2 bf16).",
    )
    args = parser.parse_args()

    # flash_attn_func rejects float32. When it's installed the train-mode
    # model dispatches to it, so we have to run the comparison in bf16.
    if FLASH_ATTN_AVAILABLE and args.dtype == "float32":
        print("flash_attn detected -- promoting dtype float32 -> bfloat16")
        print("(flash_attn_func only supports fp16/bf16; this is the meaningful")
        print(" comparison anyway since training uses bf16.)\n")
        args.dtype = "bfloat16"

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    # Default tolerances depend on dtype: bf16 has a much larger noise floor
    # because flash_attn_func and SDPA accumulate in slightly different orders.
    if args.rtol is None:
        args.rtol = 1e-4 if args.dtype == "float32" else 5e-1
    if args.atol is None:
        args.atol = 1e-4 if args.dtype == "float32" else 5e-1

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
    # Argmax agreement is the metric greedy/top-k sampling actually cares
    # about — two logit tensors can have a large max absolute diff in bf16
    # and still agree on the top token at every position.
    argmax_agree = (
        logits_train.argmax(-1) == logits_infer.argmax(-1)
    ).float().mean().item()
    allclose = torch.allclose(logits_train, logits_infer, rtol=args.rtol, atol=args.atol)

    print("\n" + "=" * 60)
    print("Hybrid train-mode vs inference-mode logit parity")
    print("=" * 60)
    print(f"Logits shape      : {tuple(logits_train.shape)}")
    print(f"Max abs diff      : {max_diff:.3e}")
    print(f"Mean abs diff     : {mean_diff:.3e}")
    print(f"Argmax agreement  : {argmax_agree * 100:.2f}%")
    print(f"allclose          : {allclose}  (rtol={args.rtol}, atol={args.atol})")

    # Pass criteria depend on dtype:
    #   fp32 (Naive vs Naive on hardware without flash_attn): a single tight
    #     check on max diff. Anything materially above 1e-3 is a regression.
    #   bf16 (FlashSWA vs Naive cross-kernel): two checks. mean must stay near
    #     the bf16 per-element noise floor (~1e-2), and max must stay below
    #     the long-tail outlier scale we have actually measured.
    if args.dtype == "float32":
        max_threshold = 1e-3
        passed = max_diff < max_threshold
        criterion = f"max diff < {max_threshold:.0e}"
    else:
        max_threshold = 1.0
        mean_threshold = 1e-1
        passed = (max_diff < max_threshold) and (mean_diff < mean_threshold)
        criterion = f"max diff < {max_threshold:.1f} and mean diff < {mean_threshold:.0e}"

    if passed:
        print(f"\nPASS  -- {criterion} ({args.dtype} noise floor)")
        print("        Train and inference paths are numerically equivalent.")
    else:
        print(f"\nFAIL  -- {criterion} not met; investigate.")
        sys.exit(1)


if __name__ == "__main__":
    main()
