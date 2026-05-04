"""
Parity tests for NaiveMLAttention.

MLA's attention forward has two internal branches:

  - Fast path  : `use_cache=False` and `kv_cache is None` -> SDPA with
                 `is_causal=True` (attention.py: NaiveMLAttention.forward:426).
  - Slow path  : anything else -> manual `matmul -> triu additive mask ->
                 softmax -> matmul` (attention.py: NaiveMLAttention.forward:433).

Those two branches compute the same mathematical function but use different
kernels and reduction orders, so bf16 disagreement is expected.

This test runs two checks, both entirely locally (MLA uses SDPA everywhere,
so no flash_attn dependency — safe on a Windows / 3090 box):

  Test 1 — Fast path vs slow path, no cache
      `model(x, use_cache=False)`                (SDPA branch)
      vs.
      `model(x, use_cache=True)` with no prior cache   (manual-matmul branch)
      Both feed the same full sequence through, q_len == k_len, so the
      answers should match up to bf16 reduction-order noise.

  Test 2 — Incremental vs full-sequence (last position)
      `model(x, use_cache=False)`                               (fast path)
      vs.
      `model(x[:, :-1], use_cache=True)` then
      `model(x[:, -1:], kv_cache=<populated>, use_cache=True)` (slow path, q_len=1)

      This exercises the parts of the code that only run during real cached
      generation:
        - k_rope cached at (B, 1, S, rope_dim) — Bug 5 fix.
        - kv_up_proj reshape-then-chunk — Bug 1 fix.
        - triu additive mask for q_len=1, k_len=S.
        - position_ids derived from cache length in Transformer.forward.

Usage:
    cd swa-mla-500m
    python -m tests.parity_mla --checkpoint ../Checkpoints/mla_19072.pt
    python -m tests.parity_mla --checkpoint ../Checkpoints/mla_19072.pt --dtype float32
"""

import argparse
import os
import sys

import torch

# Allow running as a script (not just -m) by adding the repo root to sys.path.
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


def _report(name, logits_a, logits_b, rtol, atol, bf16):
    """Print a block of stats for one comparison and return (passed, criterion)."""
    diff = (logits_a - logits_b).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    argmax_agree = (logits_a.argmax(-1) == logits_b.argmax(-1)).float().mean().item()
    allclose = torch.allclose(logits_a, logits_b, rtol=rtol, atol=atol)

    print(f"\n{'-' * 60}")
    print(name)
    print("-" * 60)
    print(f"  Shape             : {tuple(logits_a.shape)}")
    print(f"  Max abs diff      : {max_diff:.3e}")
    print(f"  Mean abs diff     : {mean_diff:.3e}")
    print(f"  Argmax agreement  : {argmax_agree * 100:.2f}%")
    print(f"  allclose          : {allclose}  (rtol={rtol}, atol={atol})")

    # Pass criteria mirror parity_hybrid: a single tight check at fp32, and
    # a two-part mean+max check at bf16 to allow for the larger per-element
    # noise floor while still catching a real regression on the bulk.
    if bf16:
        max_threshold = 1.0
        mean_threshold = 1e-1
        passed = (max_diff < max_threshold) and (mean_diff < mean_threshold)
        criterion = f"max < {max_threshold:.1f} and mean < {mean_threshold:.0e}"
    else:
        max_threshold = 1e-3
        passed = max_diff < max_threshold
        criterion = f"max < {max_threshold:.0e}"
    return passed, criterion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="bf16 matches deployment; fp32 gives tighter numerics.",
    )
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--rtol", type=float, default=None,
        help="Relative tolerance for allclose (default: 1e-4 fp32, 5e-1 bf16).",
    )
    parser.add_argument(
        "--atol", type=float, default=None,
        help="Absolute tolerance for allclose (default: 1e-4 fp32, 5e-1 bf16).",
    )
    args = parser.parse_args()

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    bf16 = args.dtype != "float32"

    if args.rtol is None:
        args.rtol = 1e-4 if not bf16 else 5e-1
    if args.atol is None:
        args.atol = 1e-4 if not bf16 else 5e-1

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}  dtype: {args.dtype}")

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = ModelConfig(**ckpt["config"])
    assert config.model_type == "mla", (
        f"This parity test is for the MLA model only; got {config.model_type}"
    )

    model = Transformer(config, mode="inference")
    model.load_state_dict(_strip_prefixes(ckpt["model"]))
    model.to(device=args.device, dtype=dtype)
    model.eval()

    torch.manual_seed(args.seed)
    seq_len = min(args.seq_len, config.block_size)
    input_ids = torch.randint(
        0, config.vocab_size, (args.batch, seq_len), device=args.device
    )

    print(f"\nRunning forward passes (B={args.batch}, S={seq_len})...")

    # -------------------------------------------------------------------
    # Test 1: fast path (SDPA) vs slow path (manual matmul), same input,
    #         no prior cache on either side.
    # -------------------------------------------------------------------
    with torch.no_grad():
        # use_cache=False, kv_cache=None -> fast branch
        out_fast = model(input_ids, use_cache=False)
        # use_cache=True, kv_cache=None -> slow branch (q_len==k_len==S)
        out_slow = model(input_ids, use_cache=True)

    t1_pass, t1_crit = _report(
        "Test 1: fast path (SDPA) vs slow path (manual matmul), no cache",
        out_fast[0].float(),
        out_slow[0].float(),
        args.rtol, args.atol, bf16,
    )

    # -------------------------------------------------------------------
    # Test 2: incremental prefill+step vs full-sequence, compare at the
    #         final position. Exercises the real cached-generation code.
    # -------------------------------------------------------------------
    with torch.no_grad():
        # Full sequence in one shot.
        logits_full, _, _ = model(input_ids, use_cache=False)

        # Prefill the first S-1 tokens; keep the populated cache.
        prefill = input_ids[:, :-1]
        _, _, kv_cache = model(prefill, use_cache=True)

        # Feed the last token with the cache; position_ids are derived
        # from cache length inside Transformer.forward.
        last = input_ids[:, -1:]
        logits_step, _, _ = model(last, kv_cache=kv_cache, use_cache=True)

    # Compare logits at the final position of the full sequence against
    # the single logit produced by the incremental step.
    logits_full_last = logits_full[:, -1, :].float()
    logits_step_last = logits_step[:, -1, :].float()

    t2_pass, t2_crit = _report(
        "Test 2: incremental (prefill + step) vs full-sequence, last position",
        logits_full_last,
        logits_step_last,
        args.rtol, args.atol, bf16,
    )

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("MLA PARITY SUMMARY")
    print("=" * 60)
    print(f"  Test 1 (fast vs slow, no cache)     : {'PASS' if t1_pass else 'FAIL'}  ({t1_crit})")
    print(f"  Test 2 (incremental vs full seq)    : {'PASS' if t2_pass else 'FAIL'}  ({t2_crit})")

    if t1_pass and t2_pass:
        print(f"\nAll MLA parity checks passed at {args.dtype} noise floor.")
    else:
        print("\nOne or more checks failed; investigate.")
        sys.exit(1)


if __name__ == "__main__":
    main()
