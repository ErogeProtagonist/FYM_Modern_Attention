"""
Evaluation Script for Hybrid SWA+GQA and MLA Transformers.

Wraps our custom Transformer for EleutherAI's lm-evaluation-harness.
Supports HellaSwag, ARC-Easy, PIQA and perplexity evaluation.
Also includes a KV cache memory benchmark.

Usage:
    # Run standard benchmarks
    python evaluate.py --checkpoint ../../Checkpoints/sft_hybrid_best.pt --tasks hellaswag,arc_easy,piqa

    # Quick test (limit to 50 examples)
    python evaluate.py --checkpoint ../../Checkpoints/sft_hybrid_best.pt --tasks hellaswag --limit 50

    # KV cache memory profile only
    python evaluate.py --checkpoint ../../Checkpoints/sft_hybrid_best.pt --memory_profile
"""

import os
import sys
import argparse
import json
import time
from typing import List, Tuple, Optional, Union

import torch
import torch.nn.functional as F
import tiktoken
import numpy as np

# Add parent directory to path for models import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import ModelConfig
from models.transformer import Transformer


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(checkpoint_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Load a trained model from checkpoint, forcing inference mode.

    Casts the model (and therefore its KV cache) to `dtype`. Defaults to bf16
    to match production inference and to make the KV-cache memory benchmark
    reflect realistic deployment cost.
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config_dict = checkpoint['config']
    config = ModelConfig(**config_dict)
    model = Transformer(config, mode="inference")

    state_dict = checkpoint['model']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("Stripping '_orig_mod.' prefix...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    if any('.naive_impl.' in k for k in state_dict.keys()):
        print("Stripping 'naive_impl.' prefix...")
        state_dict = {k.replace('.naive_impl.', '.'): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device=device, dtype=dtype)
    model.eval()

    step = checkpoint.get('step', 'unknown')
    val_loss = checkpoint.get('val_loss', None)
    print(f"Loaded {config.model_type.upper()} model (step {step}, val_loss={val_loss}, dtype={dtype})")
    return model, config


# =============================================================================
# LM-EVAL HARNESS WRAPPER
# =============================================================================

try:
    from lm_eval.api.model import LM
    from lm_eval.api.instance import Instance
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False
    LM = object  # fallback so we can define the class


class SwaMlaLM(LM):
    """
    lm-eval-harness wrapper for our custom Transformer.
    
    Implements loglikelihood (for HellaSwag/ARC/PIQA),
    loglikelihood_rolling (for perplexity), and generate_until.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda", batch_size: int = 4,
                 dtype: torch.dtype = torch.bfloat16):
        if LM_EVAL_AVAILABLE:
            super().__init__()
        self.model, self.config = load_model(checkpoint_path, device, dtype=dtype)
        self._device = device
        self._batch_size = batch_size
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot_token_id = self.enc.eot_token
        self.max_length = self.config.block_size  # 2048

    @property
    def eot_token(self) -> int:
        return self.eot_token_id

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        return self.enc.encode(string, allowed_special=set())

    def tok_decode(self, tokens: List[int], **kwargs) -> str:
        return self.enc.decode(tokens)

    def _model_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run forward pass, return logits."""
        with torch.no_grad():
            logits, _, _ = self.model(input_ids)
        return logits

    def loglikelihood(self, requests: list) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood of continuation given context, batched.

        Tokenizes every (context, continuation) up front, sorts by total length
        to minimize padding waste, then runs forward passes in batches of
        self._batch_size. Right-pads with EOT; padding tokens never affect the
        logits we extract because attention is causal and we only read the
        positions corresponding to real continuation tokens.
        """
        # 1. Tokenize all requests, recording per-sample metadata
        prepared = []
        for orig_idx, req in enumerate(requests):
            if isinstance(req, Instance):
                context, continuation = req.args
            else:
                context, continuation = req

            ctx_tokens = self.tok_encode(context)
            cont_tokens = self.tok_encode(continuation)

            # Combine and truncate from the left if needed
            all_tokens = ctx_tokens + cont_tokens
            if len(all_tokens) > self.max_length:
                all_tokens = all_tokens[-self.max_length:]
                ctx_len = len(all_tokens) - len(cont_tokens)
            else:
                ctx_len = len(ctx_tokens)

            prepared.append({
                "orig_idx": orig_idx,
                "tokens": all_tokens,
                "ctx_len": ctx_len,
                "cont_len": len(cont_tokens),
            })

        # 2. Sort by length (descending) so each batch has similar-length samples
        prepared.sort(key=lambda x: len(x["tokens"]), reverse=True)

        # 3. Process in batches
        results = [None] * len(requests)
        pad_token = self.eot_token_id

        for batch_start in range(0, len(prepared), self._batch_size):
            batch = prepared[batch_start: batch_start + self._batch_size]
            max_len = max(len(item["tokens"]) for item in batch)

            # Right-pad each sample to max_len with EOT
            batch_input = torch.full(
                (len(batch), max_len), pad_token,
                dtype=torch.long, device=self._device,
            )
            for i, item in enumerate(batch):
                tokens = item["tokens"]
                batch_input[i, :len(tokens)] = torch.tensor(
                    tokens, dtype=torch.long, device=self._device,
                )

            # Single forward pass on the whole batch
            logits = self._model_call(batch_input)  # (batch, max_len, vocab)

            # Extract per-sample loglikelihoods
            for i, item in enumerate(batch):
                ctx_len = item["ctx_len"]
                cont_len = item["cont_len"]

                # logits[i, ctx_len-1 : ctx_len-1+cont_len] predicts continuation
                shift_logits = logits[i, ctx_len - 1: ctx_len - 1 + cont_len, :]
                shift_labels = batch_input[i, ctx_len: ctx_len + cont_len]

                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(
                    1, shift_labels.unsqueeze(-1)
                ).squeeze(-1)
                total_log_prob = token_log_probs.sum().item()

                # Check greedy: would greedy decoding produce the same continuation?
                greedy_tokens = shift_logits.argmax(dim=-1)
                is_greedy = bool((greedy_tokens == shift_labels).all().item())

                results[item["orig_idx"]] = (total_log_prob, is_greedy)

        return results

    def loglikelihood_rolling(self, requests: list) -> List[float]:
        """
        Compute rolling log-likelihood (for perplexity).
        Processes the entire text as if context = EOT.

        Returns a plain float per request (the harness's process_results
        wraps it with word/byte counts for weighted_perplexity).
        """
        results = []

        for req in requests:
            if isinstance(req, Instance):
                (text,) = req.args
            else:
                text = req[0] if isinstance(req, tuple) else req

            tokens = self.tok_encode(text)

            total_ll = 0.0
            # Process in windows of max_length
            for start in range(0, len(tokens), self.max_length):
                chunk = tokens[start: start + self.max_length]
                if len(chunk) < 2:
                    continue

                input_ids = torch.tensor([chunk], dtype=torch.long, device=self._device)
                logits = self._model_call(input_ids)

                # Log probs: logits[i] predicts chunk[i+1]
                shift_logits = logits[:, :-1, :]
                shift_labels = input_ids[:, 1:]

                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
                total_ll += token_log_probs.sum().item()

            results.append(total_ll)

        return results

    def generate_until(self, requests: list) -> List[str]:
        """Generate text until stop sequences or max tokens."""
        results = []

        for req in requests:
            if isinstance(req, Instance):
                context, gen_kwargs = req.args
            else:
                context, gen_kwargs = req

            until = gen_kwargs.get("until", ["\n\n"])
            max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            tokens = self.tok_encode(context)
            # Truncate from left if needed
            if len(tokens) > self.max_length - max_gen:
                tokens = tokens[-(self.max_length - max_gen):]

            input_ids = torch.tensor([tokens], dtype=torch.long, device=self._device)

            with torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    max_new_tokens=max_gen,
                    temperature=0.0,  # Greedy for evaluation
                    top_k=1,
                )

            gen_tokens = generated[0, len(tokens):].tolist()
            gen_text = self.tok_decode(gen_tokens)

            # Truncate at stop sequences
            for stop in until:
                if stop in gen_text:
                    gen_text = gen_text[:gen_text.index(stop)]

            results.append(gen_text)

        return results


# =============================================================================
# KV CACHE MEMORY BENCHMARK
# =============================================================================

def memory_benchmark(model, config, device="cuda"):
    """
    Measure actual KV cache memory usage at various context lengths.
    This is the core thesis metric: GB of VRAM per context length.
    """
    print("\n" + "=" * 70)
    print(f"KV CACHE MEMORY BENCHMARK — {config.model_type.upper()}")
    print("=" * 70)

    enc = tiktoken.get_encoding("gpt2")
    context_lengths = [128, 256, 512, 1024, 1536, 2048]
    results = []

    # Dummy text that fills tokens
    dummy_text = "The quick brown fox jumps over the lazy dog. " * 500
    dummy_tokens = enc.encode(dummy_text)

    for ctx_len in context_lengths:
        if ctx_len > config.block_size:
            continue

        tokens = dummy_tokens[:ctx_len]
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        # Clear GPU cache
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Run forward with KV cache to simulate generation
        with torch.no_grad():
            _, _, kv_cache = model(input_ids, use_cache=True)

        # Measure KV cache size
        total_cache_bytes = 0
        cache_details = []
        for layer_idx, layer_cache in enumerate(kv_cache):
            if layer_cache is None:
                continue
            if config.model_type == "mla":
                # MLA: (c_kv, k_rope) — compressed latent
                c_kv, k_rope = layer_cache
                layer_bytes = c_kv.nelement() * c_kv.element_size()
                layer_bytes += k_rope.nelement() * k_rope.element_size()
                cache_details.append(f"  Layer {layer_idx}: c_kv={list(c_kv.shape)}, k_rope={list(k_rope.shape)}")
            else:
                # Hybrid: (k, v) tensors
                k, v = layer_cache
                layer_bytes = k.nelement() * k.element_size() + v.nelement() * v.element_size()
                cache_details.append(f"  Layer {layer_idx}: K={list(k.shape)}, V={list(v.shape)}")
            total_cache_bytes += layer_bytes

        cache_mb = total_cache_bytes / (1024 * 1024)
        
        # Peak VRAM (includes model weights + activations + cache)
        if device == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_vram_mb = 0

        results.append({
            "context_length": ctx_len,
            "kv_cache_mb": round(cache_mb, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
        })

        print(f"\nContext Length: {ctx_len}")
        print(f"  KV Cache: {cache_mb:.2f} MB")
        if device == "cuda":
            print(f"  Peak VRAM: {peak_vram_mb:.1f} MB")

        # Print first and last layer detail for insight
        if cache_details:
            print(cache_details[0])
            if len(cache_details) > 1:
                print(f"  ... ({len(cache_details) - 2} more layers)")
                print(cache_details[-1])

    # Summary table
    print("\n" + "=" * 70)
    print(f"SUMMARY: {config.model_type.upper()} KV Cache Scaling")
    print("=" * 70)
    print(f"{'Context':>10} | {'KV Cache (MB)':>15} | {'Per-Token (KB)':>15}")
    print("-" * 50)
    for r in results:
        per_token_kb = (r['kv_cache_mb'] * 1024) / r['context_length']
        print(f"{r['context_length']:>10} | {r['kv_cache_mb']:>15.2f} | {per_token_kb:>15.3f}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid/MLA Transformer")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated tasks: hellaswag,arc_easy,piqa")
    parser.add_argument("--num_fewshot", type=int, default=0,
                        help="Number of few-shot examples (0 for zero-shot)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit examples per task (for testing)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Evaluation batch size")
    parser.add_argument("--memory_profile", action="store_true",
                        help="Run KV cache memory benchmark")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not specified)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Inference dtype. bf16 matches production inference.")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Directory to save results")
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    inference_dtype = dtype_map[args.dtype]

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # Validate requested device is actually available
    if device == "cuda" and not torch.cuda.is_available():
        print(f"ERROR: --device cuda requested but torch.cuda.is_available() is False.")
        print(f"  Installed PyTorch: {torch.__version__}")
        print(f"  This is a CPU-only build. Reinstall with CUDA support:")
        print(f"    pip uninstall torch")
        print(f"    pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print(f"  Or re-run with --device cpu (much slower for lm-eval benchmarks).")
        sys.exit(1)

    print(f"Using device: {device}")

    # ---- KV Cache Memory Benchmark ----
    if args.memory_profile:
        model, config = load_model(args.checkpoint, device, dtype=inference_dtype)
        mem_results = memory_benchmark(model, config, device)

        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"memory_{config.model_type}.json")
        with open(out_path, "w") as f:
            json.dump({"model_type": config.model_type, "results": mem_results}, f, indent=2)
        print(f"\nMemory results saved to {out_path}")

        if args.tasks is None:
            return  # Exit if only memory profiling was requested

    # ---- lm-eval-harness Benchmarks ----
    if args.tasks:
        if not LM_EVAL_AVAILABLE:
            print("ERROR: lm-eval-harness not installed!")
            print("Install with: pip install lm-eval")
            sys.exit(1)

        # Create our wrapper
        lm = SwaMlaLM(
            checkpoint_path=args.checkpoint,
            device=device,
            batch_size=args.batch_size,
            dtype=inference_dtype,
        )

        # Parse tasks
        task_list = [t.strip() for t in args.tasks.split(",")]
        print(f"\nRunning evaluation on: {task_list}")
        print(f"Few-shot: {args.num_fewshot}, Limit: {args.limit or 'all'}")

        # Run evaluation
        import lm_eval
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=task_list,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            batch_size=args.batch_size,
        )

        # Print results
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)

        model_type = lm.config.model_type.upper()
        for task_name, task_results in results["results"].items():
            print(f"\n{model_type} — {task_name}:")
            for metric, value in sorted(task_results.items()):
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"eval_{lm.config.model_type}.json")
        with open(out_path, "w") as f:
            # Filter to serializable data
            save_data = {
                "model_type": lm.config.model_type,
                "checkpoint": args.checkpoint,
                "tasks": task_list,
                "num_fewshot": args.num_fewshot,
                "results": results["results"],
            }
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

    if not args.tasks and not args.memory_profile:
        print("No action specified. Use --tasks and/or --memory_profile")
        print("Example: python evaluate.py --checkpoint path/to/model.pt --tasks hellaswag,arc_easy,piqa --memory_profile")


if __name__ == "__main__":
    main()
