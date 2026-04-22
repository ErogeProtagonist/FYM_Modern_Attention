# Repo Notes

Concise design and implementation notes for readers who open this repo cold.
For the dissertation-level context (motivations, results, write-up decisions)
see the paper. This file covers only what you need to navigate the code.

---

## The dual-path architecture

Both models expose a single `Transformer` class but pick a different attention
kernel depending on `mode` and whether `flash_attn` is installed.

|                | `mode='train'` + `flash_attn` | `mode='train'` no `flash_attn` | `mode='inference'` |
| -------------- | :---------------------------: | :----------------------------: | :----------------: |
| **Hybrid**     | `FlashSWAHybridAttention`     | `NaiveHybridAttention`         | `NaiveHybridAttention` |
| **MLA**        | `NaiveMLAttention`            | `NaiveMLAttention`             | `NaiveMLAttention`     |

Dispatch lives in `models/attention.py::get_attention`. The Hybrid training
kernel calls `flash_attn_func` with native sliding window; the inference
kernel uses PyTorch SDPA and is portable. MLA uses SDPA everywhere because
the `flash_mla` package is inference-only and didn't help training throughput.

Inside `NaiveMLAttention.forward` there is a further split:

- **fast path** — `use_cache=False` and `kv_cache is None`: one SDPA call
  with `is_causal=True`. Used in training and lm-eval-harness loglikelihood.
- **slow path** — anything else: manual `matmul → triu → softmax → matmul`.
  Necessary because cached generation has `q_len=1` while `k_len` is the
  full cached context, which SDPA's `is_causal` flag does not handle.

The parity tests (`tests/parity_*.py`) exist to keep these branches in sync.

---

## Files, in order of importance

- `models/attention.py` — three attention classes + factory. Densest file.
- `models/transformer.py` — block / full model / generate loop. Note the
  `position_ids` derivation from cache length in `Transformer.forward`; this
  is load-bearing for incremental decoding.
- `models/config.py` — `ModelConfig` dataclass + `HYBRID_500M` / `MLA_500M`.
- `models/rope.py` — standard and single-tensor (decoupled) RoPE.
- `scripts/train_edge.py` — DDP pre-training loop.
- `scripts/inference.py` — portable inference (bf16 by default to match eval).
- `scripts/evaluate.py` — lm-eval-harness wrapper + rolling-ctx wikitext PPL.
- `sft/` — supervised fine-tuning on smol-smoltalk.
- `tests/parity_hybrid.py` — train vs inference kernel parity (cross-kernel
  on cloud, Naive-vs-Naive on local).
- `tests/parity_mla.py` — MLA self-consistency (fast vs slow path, full-seq
  vs incremental cached step).
- `tests/verify_swa.py` — SWA rolling-cache shape sanity check.

Checkpoints are kept **outside** the repo at `../Checkpoints/hybrid_19072.pt`
and `../Checkpoints/mla_19072.pt` (too large to track).

---

## Bug history (for breadcrumbs in the code)

The six numbered bugs referenced in code comments are:

1. **MLA kv_up_proj reshape order (Gibberish Bug).** Must be
   `view(B, S, n_heads, 2*d_h) → transpose → chunk(2, dim=-1)`, not
   chunk-then-reshape. The linear weights assume interleaved layout;
   swapping produces gibberish at reasonable-looking training loss.
2. *(evaluate.py dtype default — now bf16 by default, matches training.)*
3. **MLA causal mask dtype.** The manual additive mask in the slow path
   must match `attn_weights.dtype` (bf16/fp16) or the graph stays mixed and
   fails under autocast inference.
4. **Hybrid SWA mask dtype.** Same idea for the SDPA fallback path in
   `NaiveHybridAttention._get_sliding_window_mask`.
5. **MLA k_rope cache shape.** `NaiveMLAttention` must cache the decoupled
   RoPE key at `(B, 1, S, rope_dim)` and only broadcast to `n_heads` for the
   attention matmul. Caching at `n_heads` concatenated incorrectly on
   subsequent cached steps.
6. **GQA cache size.** `NaiveHybridAttention` caches at `n_kv_heads`, not
   `n_heads`. Expanding with `repeat_interleave` happens *after* the cache
   block. The old code cached at `n_heads`, defeating the whole GQA win.

---

## Parity test baselines

All on 500M / 24 layers / 50,304-vocab.

| Test | dtype | max abs diff | mean abs diff | argmax | Notes |
|---|---|---:|---:|---:|---|
| Hybrid (Naive vs Naive, no flash_attn) | fp32 | ~4.5e-05 | ~2.7e-06 | — | self-check, 3090 |
| Hybrid (FlashSWA vs Naive) | bf16 | ~4.38e-01 | ~2.64e-02 | 97.46% | cloud B200, real cross-kernel |
| MLA Test 1 (fast vs slow, no cache) | bf16 | ~4.5e-01 | ~3.0e-02 | 96.88% | 3090 |
| MLA Test 2 (incremental vs full, last pos) | bf16 | ~1.6e-01 | ~2.8e-02 | 100.00% | exercises bugs 1 & 6 |

Pass thresholds in the test scripts: fp32 requires `max < 1e-3`; bf16
requires `max < 1.0` and `mean < 1e-1`. Anything materially tighter in bf16
is also suspicious (likely a dispatch change) — check argmax agreement too.

---

## Running things

```bash
# Local (Windows/3090, no flash_attn)
python -m tests.parity_mla --checkpoint ../Checkpoints/mla_19072.pt
python -m tests.parity_hybrid --checkpoint ../Checkpoints/hybrid_19072.pt  # Naive vs Naive
python scripts/inference.py --checkpoint ../Checkpoints/hybrid_19072.pt --interactive

# Cloud (flash_attn available)
python -m tests.parity_hybrid --checkpoint ../Checkpoints/hybrid_19072.pt  # real FlashSWA vs SDPA
torchrun --standalone --nproc_per_node=2 scripts/train_edge.py --model_type mla --auto_optimize --compile
```

---

## Things that look weird but aren't

- `FlashSWAHybridAttention` inherits from `NaiveHybridAttention` and only
  overrides the non-cached forward — the cached path is shared.
- `ModelConfig.q_lora_rank` and `rope_scaling` are unused; kept for checkpoint
  backward-compat (the dataclass must accept the keys that older checkpoints
  were saved with).
- `parity_hybrid.py` auto-promotes `float32` → `bfloat16` when `flash_attn` is
  installed, because `flash_attn_func` rejects fp32. On a 3090 without
  `flash_attn` it stays fp32 and acts as a SDPA self-consistency check.
