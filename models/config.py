"""
Model Configuration for Hybrid SWA and MLA Transformers.

This module defines the hyperparameters for 500M parameter models
following the Gemma (Hybrid) and DeepSeek (MLA) architectures.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    """Configuration for both Hybrid SWA and MLA transformer models."""
    
    # Model type: "hybrid" (Gemma-style) or "mla" (DeepSeek-style)
    model_type: Literal["hybrid", "mla"] = "hybrid"
    
    # Core architecture (same for both models to ensure fair comparison)
    d_model: int = 1024          # Hidden dimension
    n_layers: int = 24           # Number of transformer layers
    n_heads: int = 16            # Number of query attention heads
    n_kv_heads: int = 4          # Number of KV heads (for GQA). If None, uses n_heads (MHA)
    head_dim: int = 64           # Dimension per head (d_model // n_heads)
    vocab_size: int = 50304      # Vocabulary size (padded for efficiency)
    block_size: int = 2048       # Maximum sequence length
    
    # FFN configuration (SwiGLU style)
    ffn_hidden_mult: float = 2.667  # FFN hidden = d_model * mult (gives ~2730)
    
    # Hybrid-specific (Gemma-style)
    window_size: int = 512           # Sliding window size for local attention
    global_layer_period: int = 6     # Global attention every N layers (5:1 ratio)
    
    # MLA-specific (DeepSeek-style)
    kv_lora_rank: int = 512          # Latent compression dimension (d_c)
    q_lora_rank: int = 384           # Query compression dimension (optional)
    rope_dim: int = 64               # Decoupled RoPE dimension (d_R)
    
    # Shared attention settings
    rope_base: float = 10000.0       # RoPE base frequency
    rope_scaling: float = 1.0        # RoPE scaling factor for extended context
    
    # Initialization
    init_std: float = 0.006          # DeepSeek-style small init for residual projections
    
    # Dropout (typically 0 for pretraining)
    dropout: float = 0.0
    
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads for GQA"
        self.head_dim = self.d_model // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # Number of times to repeat KV heads

        
    @property
    def ffn_hidden_dim(self) -> int:
        """Compute FFN hidden dimension (SwiGLU uses 2/3 * 4 * d_model ≈ 2.667 * d_model)."""
        return int(self.d_model * self.ffn_hidden_mult)
    
    def is_global_layer(self, layer_idx: int) -> bool:
        """Check if a layer should use global attention (Hybrid model only)."""
        # Pattern: 5 local layers, then 1 global (0-indexed)
        # Layer 5, 11, 17, 23 are global in a 24-layer model with period=6
        return (layer_idx + 1) % self.global_layer_period == 0


# Predefined configurations for the thesis comparison
# Target: ~500M parameters. With d_model=1280, n_layers=24, vocab=50k, we get ~500M
HYBRID_500M = ModelConfig(
    model_type="hybrid",
    d_model=1280,
    n_layers=24,
    n_heads=20,            # 20 query heads
    n_kv_heads=4,          # 4 KV heads (5:1 GQA ratio, like Gemma 2/3)
    window_size=512,
    global_layer_period=6,  # 5:1 local:global ratio
)

MLA_500M = ModelConfig(
    model_type="mla",
    d_model=1280,
    n_layers=24,
    n_heads=20,
    n_kv_heads=20,         # MLA uses full MHA (GQA not applicable - uses latent compression instead)
    kv_lora_rank=512,
    rope_dim=64,
)
