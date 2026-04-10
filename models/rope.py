"""
Rotary Positional Embeddings (RoPE) implementation.

This module provides RoPE for both standard attention and the
Decoupled RoPE strategy used in MLA (Multi-Head Latent Attention).
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the cosine and sine frequencies for RoPE.
    
    Args:
        dim: Dimension of the embeddings (must be even)
        max_seq_len: Maximum sequence length
        base: Base frequency for the rotations
        device: Device to create tensors on
        
    Returns:
        Tuple of (cos, sin) tensors of shape (max_seq_len, dim)
    """
    assert dim % 2 == 0, "Dimension must be even for RoPE"
    
    # Compute inverse frequencies: theta_i = 1 / (base^(2i/dim))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    
    # Compute position indices
    positions = torch.arange(max_seq_len, device=device).float()
    
    # Outer product: (seq_len,) x (dim/2,) -> (seq_len, dim/2)
    freqs = torch.outer(positions, inv_freq)
    
    # Duplicate for pairs: (seq_len, dim/2) -> (seq_len, dim)
    freqs = torch.cat([freqs, freqs], dim=-1)
    
    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input for RoPE.
    
    For input [..., d], returns [..., d] where:
    - First half becomes negative second half
    - Second half becomes first half
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to query and key tensors.
    
    Args:
        q: Query tensor of shape (batch, n_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, n_heads, seq_len, head_dim)
        cos: Cosine frequencies of shape (seq_len, head_dim)
        sin: Sine frequencies of shape (seq_len, head_dim)
        position_ids: Optional position indices for non-contiguous positions
        
    Returns:
        Tuple of rotated (q, k) tensors
    """
    # Handle position_ids for KV cache scenarios
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]
    else:
        seq_len = q.shape[2]
        cos = cos[:seq_len]
        sin = sin[:seq_len]
    
    # Expand dims: (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Apply rotation: x * cos + rotate_half(x) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


def apply_rotary_pos_emb_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply rotary positional embeddings to a single tensor (for MLA decoupled RoPE).
    
    Args:
        x: Input tensor of shape (batch, n_heads, seq_len, rope_dim)
        cos: Cosine frequencies of shape (seq_len, rope_dim)
        sin: Sine frequencies of shape (seq_len, rope_dim)
        position_ids: Optional position indices
        
    Returns:
        Rotated tensor of same shape
    """
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]
    else:
        seq_len = x.shape[2]
        cos = cos[:seq_len]
        sin = sin[:seq_len]
    
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding module that caches cos/sin frequencies.
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute and register as buffers (not parameters)
        cos, sin = precompute_freqs_cis(dim, max_seq_len, base)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors."""
        return apply_rotary_pos_emb(
            q, k, 
            self.cos_cached.to(q.dtype), 
            self.sin_cached.to(q.dtype),
            position_ids,
        )
    
    def forward_single(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply RoPE to a single tensor (for MLA decoupled RoPE)."""
        return apply_rotary_pos_emb_single(
            x,
            self.cos_cached.to(x.dtype),
            self.sin_cached.to(x.dtype),
            position_ids,
        )
