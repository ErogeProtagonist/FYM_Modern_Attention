"""
Attention Implementations for Hybrid SWA and MLA Transformers.

This module provides 3 attention variants:
- NaiveHybridAttention: SDPA-based Hybrid (used for inference and as a training
  fallback when flash_attn is unavailable)
- FlashSWAHybridAttention: Optimized training for Hybrid via flash_attn package
- NaiveMLAttention: SDPA-based MLA used for both training and inference
  (the FlashMLA package is inference-only and didn't help training, so we use
  SDPA for everything; the no-cache forward dispatches to FlashAttention via
  the SDPA backend selector)

The factory function `get_attention()` selects the right implementation based
on hardware (flash_attn availability) and mode.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import ModelConfig
from .rope import RotaryEmbedding, apply_rotary_pos_emb, apply_rotary_pos_emb_single


# FlashAttention-2 native sliding window (fastest option for Hybrid training)
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func = None





# ============================================================================
# HYBRID ATTENTION IMPLEMENTATIONS
# ============================================================================

class NaiveHybridAttention(nn.Module):
    """
    Naive PyTorch implementation of Hybrid Sliding Window + Global GQA attention.
    
    Implements Grouped Query Attention (GQA) where multiple query heads share
    fewer KV heads. This reduces KV cache size while maintaining expressiveness.
    
    Optimized to use:
    - Cached causal/window masks (avoids CPU loop overhead)
    - Vectorized mask creation
    - Explicit SDPA backend selection
    """
    
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_global = config.is_global_layer(layer_idx)
        
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_rep  # Number of times to repeat KV heads
        self.head_dim = config.head_dim
        self.window_size = config.window_size
        
        # GQA: Separate Q and KV projections
        # Q: d_model -> n_heads * head_dim
        # K,V: d_model -> n_kv_heads * head_dim (smaller!)
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.kv_proj = nn.Linear(config.d_model, 2 * config.n_kv_heads * config.head_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # RoPE
        self.rope = RotaryEmbedding(self.head_dim, config.block_size, config.rope_base)
        
        # Mark output projection for scaled init
        self.out_proj.RESIDUAL_SCALE_INIT = True
        
        # OPTIMIZATION: Cache attention mask for training (avoids recreating each forward pass)
        self.register_buffer("mask_cache", None, persistent=False)
        
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            position_ids: Position indices for RoPE
            kv_cache: Cached (keys, values) for generation
            use_cache: Whether to return updated cache
            
        Returns:
            output: Attention output (batch, seq_len, d_model)
            new_cache: Updated KV cache if use_cache=True
        """
        B, S, D = x.shape
        
        # GQA Projections
        q = self.q_proj(x)  # (B, S, n_heads * head_dim)
        kv = self.kv_proj(x)  # (B, S, 2 * n_kv_heads * head_dim)
        k, v = kv.chunk(2, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)      # (B, n_heads, S, head_dim)
        k = k.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, n_kv_heads, S, head_dim)
        v = v.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, n_kv_heads, S, head_dim)
        
        # Apply RoPE (on n_kv_heads tensors — before any GQA expansion)
        q, k = self.rope(q, k, position_ids)

        # Handle KV cache for generation. The cache stores tensors at
        # n_kv_heads (NOT n_heads), so the GQA cache is n_rep× smaller.
        # Query-side broadcasting via repeat_interleave happens AFTER this block.
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        if use_cache:
            if not self.is_global:
                # LOCAL LAYER: Cap cache at window_size (rolling buffer)
                if k.shape[2] > self.window_size:
                    k_cache = k[:, :, -self.window_size:, :]
                    v_cache = v[:, :, -self.window_size:, :]
                else:
                    k_cache = k
                    v_cache = v
                new_cache = (k_cache, v_cache)
            else:
                # GLOBAL LAYER: Keep full cache
                new_cache = (k, v)
        else:
            new_cache = None

        # GQA: now expand KV heads to match Q heads for the attention compute.
        # (B, n_kv_heads, S, head_dim) -> (B, n_heads, S, head_dim)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Attention Logic
        if self.is_global:
            # Global: Standard causal SDPA
            # We prefer FLASH_ATTENTION for speed on Ampere+
            with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION, torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION, torch.nn.attention.SDPBackend.MATH]):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Sliding Window
            kv_len = k.shape[2]
            q_len = q.shape[2]
            
            # Efficiently get or create mask (dtype must match q for bf16/fp16 SDPA)
            mask = self._get_sliding_window_mask(q_len, kv_len, q.device, q.dtype)
            
            # Use SDPA with explicit mask
            # Note: FlashAttention 2 supports slight window attention via specialized kernels,
            # but standard sdp_kernel might fall back to efficient_attention or math if mask is dense-ish.
            with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION, torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION, torch.nn.attention.SDPBackend.MATH]):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)
        
        return out, new_cache
    
    def _get_sliding_window_mask(
        self, q_len: int, kv_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Get cached mask or create one efficiently using vectorized ops.
        Avoids slow Python loops. Mask dtype matches the attention tensors so
        bf16/fp16 inference does not crash on the SDPA additive mask.
        """
        # 1. Check if we can reuse the cached mask (Training scenario)
        if self.mask_cache is not None and \
           self.mask_cache.shape == (q_len, kv_len) and \
           self.mask_cache.device == device and \
           self.mask_cache.dtype == dtype:
            return self.mask_cache

        # 2. Vectorized mask creation
        # Indices: (Q, 1) - (1, KV) gives relative distance
        # q_idx[i] = i (if q is full seq) or offset+i (if q is chunk)
        # But for standard forward passes, q is aligned at end of kv usually?

        # Assumption: In standard causal attention (train or inference):
        # The query tokens Q[0..q_len] align with Keys K[kv_len-q_len .. kv_len]
        # i.e. the last q_len keys are the ones matching Q

        # Construct absolute positions
        # KV indices: 0, 1, ..., kv_len-1
        # Q indices:  kv_len-q_len, ..., kv_len-1

        kv_indices = torch.arange(kv_len, device=device).unsqueeze(0)  # (1, KV)
        q_indices = torch.arange(kv_len - q_len, kv_len, device=device).unsqueeze(1) # (Q, 1)

        diff = q_indices - kv_indices

        # Mask conditions:
        # 1. Causal: q >= k (diff >= 0)
        # 2. Window: q - k < window (diff < window)
        # Valid: 0 <= diff < window

        # Create mask initialized to -inf
        mask = torch.full((q_len, kv_len), float("-inf"), device=device, dtype=dtype)

        # Set valid positions to 0.0
        # This is VASTLY faster than a python loop for 2048x2048
        valid_mask = (diff >= 0) & (diff < self.window_size)
        mask.masked_fill_(valid_mask, 0.0)

        # Cache it if it matches block size (typical training case)
        if q_len == self.config.block_size and kv_len == self.config.block_size:
            self.mask_cache = mask

        return mask


class FlashSWAHybridAttention(nn.Module):
    """
    Hybrid attention using FlashAttention-2's native sliding window.

    Uses flash_attn_func with the window_size parameter for O(n * window)
    complexity, plus Grouped Query Attention (GQA) where multiple query heads
    share fewer KV heads. This is the production training kernel; on hardware
    without flash_attn the factory falls back to NaiveHybridAttention (SDPA).

    Requires: pip install flash-attn (H100/A100/B200)
    """
    
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        
        if not FLASH_ATTN_AVAILABLE:
            raise RuntimeError("flash-attn not available. Install with: pip install flash-attn")
            
        self.config = config
        self.layer_idx = layer_idx
        self.is_global = config.is_global_layer(layer_idx)
        
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_rep
        self.head_dim = config.head_dim
        self.window_size = config.window_size
        
        # GQA: Separate Q and KV projections
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.kv_proj = nn.Linear(config.d_model, 2 * config.n_kv_heads * config.head_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.rope = RotaryEmbedding(self.head_dim, config.block_size, config.rope_base)
        self.out_proj.RESIDUAL_SCALE_INIT = True
        
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, S, D = x.shape
        
        # GQA Projections
        q = self.q_proj(x)
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)
        
        # Reshape: (B, S, D) -> (B, S, nh, hd) for flash_attn format
        q = q.view(B, S, self.n_heads, self.head_dim)
        k = k.view(B, S, self.n_kv_heads, self.head_dim)
        v = v.view(B, S, self.n_kv_heads, self.head_dim)

        # Apply RoPE (RoPE needs (B, nh, S, hd) layout)
        q = q.transpose(1, 2)  # (B, n_heads, S, hd)
        k = k.transpose(1, 2)  # (B, n_kv_heads, S, hd)
        q, k = self.rope(q, k, position_ids)

        if kv_cache is not None or use_cache:
            # Inference fallback: SDPA (flash_attn doesn't support KV cache directly).
            # Cache stores tensors at n_kv_heads — expansion to n_heads happens
            # AFTER the cache write, so the cached tensors are n_rep× smaller.
            # q is already (B, n_heads, S, hd); k is (B, n_kv_heads, S, hd).
            v_inf = v.transpose(1, 2)  # (B, n_kv_heads, S, hd)

            if kv_cache is not None:
                past_k, past_v = kv_cache
                k = torch.cat([past_k, k], dim=2)
                v_inf = torch.cat([past_v, v_inf], dim=2)

            if use_cache:
                if not self.is_global:
                    if k.shape[2] > self.window_size:
                        k_cache = k[:, :, -self.window_size:, :]
                        v_cache = v_inf[:, :, -self.window_size:, :]
                    else:
                        k_cache = k
                        v_cache = v_inf
                    new_cache = (k_cache, v_cache)
                else:
                    new_cache = (k, v_inf)
            else:
                new_cache = None

            # Expand KV heads for the attention compute (NOT cached)
            if self.n_rep > 1:
                k_attn = k.repeat_interleave(self.n_rep, dim=1)
                v_attn = v_inf.repeat_interleave(self.n_rep, dim=1)
            else:
                k_attn, v_attn = k, v_inf

            # Use SDPA for inference
            if self.is_global:
                out = F.scaled_dot_product_attention(q, k_attn, v_attn, is_causal=True)
            else:
                kv_len = k_attn.shape[2]
                mask = self._make_sliding_window_mask(q.shape[2], kv_len, x.device)
                out = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=mask)

            out = out.transpose(1, 2).contiguous().view(B, S, D)
        else:
            # Training: use flash_attn with native sliding window.
            # Keep the explicit GQA expansion + (B, S, nh, hd) layout — bit-identical
            # to before this fix.
            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v_expanded = v.transpose(1, 2).repeat_interleave(self.n_rep, dim=1)
            else:
                v_expanded = v.transpose(1, 2)

            q = q.transpose(1, 2)  # Back to (B, S, nh, hd) for flash_attn
            k = k.transpose(1, 2)
            v = v_expanded.transpose(1, 2)  # (B, S, n_heads, hd)

            new_cache = None

            if self.is_global:
                # Global layer: full causal attention
                out = flash_attn_func(q, k, v, causal=True)
            else:
                # Local layer: sliding window attention
                # window_size=(left, right): (window-1, 0) for causal sliding window
                out = flash_attn_func(
                    q, k, v,
                    causal=True,
                    window_size=(self.window_size - 1, 0)
                )

            out = out.view(B, S, D)
        
        out = self.out_proj(out)
        return out, new_cache
    
    def _make_sliding_window_mask(self, q_len, kv_len, device):
        """Fallback mask for inference."""
        mask = torch.full((q_len, kv_len), float("-inf"), device=device)
        for i in range(q_len):
            abs_pos = kv_len - q_len + i
            start = max(0, abs_pos - self.window_size + 1)
            end = abs_pos + 1
            mask[i, start:end] = 0.0
        return mask


# ============================================================================
# MLA (Multi-Head Latent Attention) IMPLEMENTATIONS
# ============================================================================

class NaiveMLAttention(nn.Module):
    """
    Naive PyTorch implementation of Multi-Head Latent Attention.
    
    Implements the DeepSeek-V2/V3 MLA mechanism with:
    - Low-rank KV compression into latent vector c_KV
    - Decoupled RoPE for positional information
    - Standard matmul operations (no FlashMLA kernel)
    
    Works on any hardware (RTX 3090, CPU, etc.).
    """
    
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.kv_lora_rank = config.kv_lora_rank  # d_c (latent dimension)
        self.rope_dim = config.rope_dim           # d_R (RoPE dimension)
        
        # Query projection (full dimension)
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        
        # Decoupled Query RoPE projection
        self.q_rope_proj = nn.Linear(config.d_model, config.n_heads * self.rope_dim, bias=False)
        
        # KV down-projection to latent space
        self.kv_down_proj = nn.Linear(config.d_model, self.kv_lora_rank, bias=False)
        
        # KV up-projection from latent space (Fused for speed)
        self.kv_up_proj = nn.Linear(self.kv_lora_rank, 2 * config.n_heads * config.head_dim, bias=False)

        
        # Decoupled Key RoPE projection (shared across heads in DeepSeek style)
        self.k_rope_proj = nn.Linear(config.d_model, self.rope_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj.RESIDUAL_SCALE_INIT = True
        
        # RoPE for the decoupled positional embeddings
        self.rope_content = RotaryEmbedding(config.head_dim, config.block_size, config.rope_base)
        self.rope_decoupled = RotaryEmbedding(self.rope_dim, config.block_size, config.rope_base)
        
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input (batch, seq_len, d_model)
            kv_cache: Cached (c_KV, k_rope) for generation
            
        Returns:
            output, new_cache
        """
        B, S, D = x.shape
        
        # === Query Path ===
        # Content query
        q_content = self.q_proj(x)  # (B, S, n_heads * head_dim)
        q_content = q_content.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Decoupled RoPE query
        q_rope = self.q_rope_proj(x)  # (B, S, n_heads * rope_dim)
        q_rope = q_rope.view(B, S, self.n_heads, self.rope_dim).transpose(1, 2)
        q_rope = self.rope_decoupled.forward_single(q_rope, position_ids)
        
        # === Key-Value Path ===
        # Compress to latent space
        c_kv = self.kv_down_proj(x)  # (B, S, kv_lora_rank)
        
        # Decoupled RoPE key (shared across heads — DeepSeek's key MLA trick)
        k_rope = self.k_rope_proj(x)  # (B, S, rope_dim)
        k_rope = k_rope.unsqueeze(2)  # (B, S, 1, rope_dim)
        k_rope = k_rope.transpose(1, 2)  # (B, 1, S, rope_dim)
        k_rope = self.rope_decoupled.forward_single(k_rope, position_ids)
        # NOTE: do NOT expand to n_heads here — we cache the shared (B, 1, S, rope_dim)
        # form and only broadcast for the attention matmul below.

        # Handle KV cache
        # In MLA, we cache the compressed c_kv and the (shared) RoPE key
        if kv_cache is not None:
            past_c_kv, past_k_rope = kv_cache
            c_kv = torch.cat([past_c_kv, c_kv], dim=1)
            k_rope = torch.cat([past_k_rope, k_rope], dim=2)

        new_cache = (c_kv, k_rope) if use_cache else None

        # Up-project keys and values from latent space.
        # CRITICAL: reshape BEFORE chunking. view(B, S, n_heads, 2*head_dim) then
        # chunk(2, dim=-1) gives interleaved [k,v] per head, NOT contiguous
        # [all_k, all_v]. Getting this wrong silently produces wrong outputs
        # (the model generates gibberish even with reasonable training loss)
        # because the linear layer's weight order corresponds to the interleaved
        # layout. See Training_Run_Summary_Internal.md section 2.B for the
        # original incident — the "MLA gibberish" bug fixed in commit a13cd78.
        kv_content = self.kv_up_proj(c_kv)  # (B, S_kv, 2 * n_heads * head_dim)
        S_kv = kv_content.shape[1]
        kv_content = kv_content.view(B, S_kv, self.n_heads, 2 * self.head_dim).transpose(1, 2)
        # Shape: (B, n_heads, S_kv, 2 * head_dim)
        k_content, v = kv_content.chunk(2, dim=-1)  # Each: (B, n_heads, S_kv, head_dim)

        # Expand k_rope across heads for the attention matmul only (after caching).
        # This is a view, not a copy — no extra memory allocated.
        k_rope_for_attn = k_rope.expand(-1, self.n_heads, -1, -1)

        # === Attention Computation ===
        # Concatenate content and RoPE dimensions for query and key
        # q_full: (B, n_heads, S_q, head_dim + rope_dim)
        # k_full: (B, n_heads, S_kv, head_dim + rope_dim)
        q_full = torch.cat([q_content, q_rope], dim=-1)
        k_full = torch.cat([k_content, k_rope_for_attn], dim=-1)

        # Fast path: when not using a KV cache (e.g. training, or
        # lm-eval-harness loglikelihood, or any non-cached forward), dispatch to
        # SDPA / FlashAttention. q_len == k_len here so is_causal=True is
        # correct. SDPA's default scale is 1/sqrt(last_dim) =
        # 1/sqrt(head_dim + rope_dim), which exactly matches DeepSeek's MLA
        # scaling.
        if kv_cache is None and not use_cache:
            with torch.nn.attention.sdpa_kernel([
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]):
                out = F.scaled_dot_product_attention(q_full, k_full, v, is_causal=True)
        else:
            # Slow path: cached generation. q_len may be 1 while k_len is the full
            # cached context, so we need an explicit additive mask rather than
            # is_causal=True. Manual matmul handles this correctly.
            scale = 1.0 / math.sqrt(self.head_dim + self.rope_dim)
            attn_weights = torch.matmul(q_full, k_full.transpose(-1, -2)) * scale

            # Apply causal mask (match attn dtype so bf16/fp16 inference works)
            S_q = q_full.shape[2]
            S_kv = k_full.shape[2]
            causal_mask = torch.triu(
                torch.full((S_q, S_kv), float("-inf"), device=x.device, dtype=attn_weights.dtype),
                diagonal=S_kv - S_q + 1
            )
            attn_weights = attn_weights + causal_mask

            # Softmax and apply to values
            attn_weights = F.softmax(attn_weights, dim=-1)
            out = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)
        
        return out, new_cache


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def get_attention(
    config: ModelConfig, 
    layer_idx: int, 
    mode: str = "train"
) -> nn.Module:
    """
    Factory function to get the appropriate attention implementation.

    Hybrid: FlashSWAHybridAttention if flash_attn is installed (used on cloud
    GPUs), otherwise NaiveHybridAttention SDPA fallback.

    MLA: NaiveMLAttention always. The flash_mla package is inference-only and
    didn't help training, so MLA uses SDPA for both phases — its non-cache
    forward dispatches to FlashAttention via the SDPA backend selector.

    Args:
        config: Model configuration
        layer_idx: Layer index (for Hybrid layer type selection)
        mode: "train" (use optimized kernels) or "inference"

    Returns:
        Attention module instance
    """
    attn_cls = None

    if config.model_type == "hybrid":
        if mode == "train" and FLASH_ATTN_AVAILABLE:
            attn_cls = FlashSWAHybridAttention
        else:
            attn_cls = NaiveHybridAttention

    elif config.model_type == "mla":
        attn_cls = NaiveMLAttention

    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    # Debug log for first layer
    if layer_idx == 0:
        print(f"Layer 0 Attention: {attn_cls.__name__} (mode={mode}, flash_attn={FLASH_ATTN_AVAILABLE})")

    return attn_cls(config, layer_idx)

