"""
Transformer Model for Hybrid SWA and MLA architectures.

This module contains the main Transformer class that can be configured
for either Hybrid (Gemma-style) or MLA (DeepSeek-style) attention.

Features:
- RMSNorm (not LayerNorm)
- SwiGLU FFN (not GELU MLP)
- RoPE (not learned positional embeddings)
- Weight tying (input embed = output proj)
- DeepSeek-style initialization (0.006 std for residual projections)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import inspect

from .config import ModelConfig
from .attention import get_attention, FLASH_ATTN_AVAILABLE


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm as it doesn't require mean computation.
    Used in Llama, Gemma, DeepSeek, etc.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    
    SwiGLU(x) = (x @ W_gate * SiLU(x @ W_up)) @ W_down
    
    More expressive than standard GELU/ReLU MLP, used in modern LLMs.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = config.ffn_hidden_dim
        
        # OPTIMIZATION: Fused gate+up projection (d_model -> 2*hidden_dim)
        # Reduces kernel launches from 2 to 1, improving throughput by ~5-10%
        self.w1 = nn.Linear(config.d_model, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.d_model, bias=False)
        
        # Mark down projection for scaled init
        self.down_proj.RESIDUAL_SCALE_INIT = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused projection and split
        x_inter = self.w1(x)
        gate, up = x_inter.chunk(2, dim=-1)
        
        # SiLU(gate) * up, then project down
        return self.down_proj(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    """
    Single Transformer block with pre-norm architecture.
    
    Architecture: x -> RMSNorm -> Attention -> + -> RMSNorm -> FFN -> +
    """
    
    def __init__(self, config: ModelConfig, layer_idx: int, mode: str = "train"):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Pre-attention norm
        self.norm1 = RMSNorm(config.d_model)
        
        # Attention (selected based on model type and mode)
        self.attention = get_attention(config, layer_idx, mode)
        
        # Pre-FFN norm
        self.norm2 = RMSNorm(config.d_model)
        
        # Feed-forward network
        self.ffn = SwiGLU(config)
        
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Attention block with residual
        residual = x
        x = self.norm1(x)
        x, new_cache = self.attention(x, position_ids, kv_cache, use_cache)
        x = residual + x
        
        # FFN block with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x, new_cache


class Transformer(nn.Module):
    """
    Full Transformer model for language modeling.
    
    Supports both Hybrid SWA (Gemma-style) and MLA (DeepSeek-style) attention.
    Uses RoPE for positional encoding (no learned positional embeddings).
    """
    
    def __init__(self, config: ModelConfig, mode: str = "train", use_checkpointing: bool = True):
        super().__init__()
        self.config = config
        self.mode = mode
        self.use_checkpointing = use_checkpointing  # Can be disabled for high-VRAM GPUs (H200/B200)
        
        # Token embedding (no positional embeddings - using RoPE)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx, mode)
            for layer_idx in range(config.n_layers)
        ])
        
        # Final norm
        self.norm_f = RMSNorm(config.d_model)
        
        # Output projection (lm_head)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying: share embedding weights with output projection
        self.embed_tokens.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Print model info
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Initialized {config.model_type.upper()} Transformer with {n_params:,} parameters")
        print(f"  - Mode: {mode}")
        print(f"  - flash_attn available: {FLASH_ATTN_AVAILABLE}")
        
    def _init_weights(self, module: nn.Module):
        """
        Initialize weights using DeepSeek-style small init for residual projections.
        """
        if isinstance(module, nn.Linear):
            # Check if this is a residual projection (marked by our attention/ffn classes)
            if hasattr(module, 'RESIDUAL_SCALE_INIT'):
                # Small init for residual projections (DeepSeek style)
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            else:
                # Standard init for other projections
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List]]:
        """
        Forward pass for language modeling.
        
        Args:
            input_ids: Token indices (batch, seq_len)
            targets: Target token indices for loss computation
            position_ids: Position indices for RoPE (auto-generated if None)
            kv_cache: List of cached KV pairs per layer
            use_cache: Whether to return updated KV cache
            
        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided
            new_cache: Updated KV cache if use_cache=True
        """
        B, T = input_ids.shape
        
        # Validate sequence length
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block size {self.config.block_size}"
        
        # Token embeddings
        x = self.embed_tokens(input_ids)
        
        # Generate position_ids if not provided. During incremental generation
        # the new token's absolute position is `past_length + i`, where
        # past_length is read from the cache shape of layer 0:
        #   MLA      : c_kv is [B, T_past, d_c]            -> dim 1
        #   Hybrid   : k    is [B, n_kv_heads, T_past, dh] -> dim 2
        # On the prefill pass (no cache yet) we leave position_ids as None and
        # let the per-layer RoPE module derive positions from sequence length.
        if position_ids is None and kv_cache is not None and kv_cache[0] is not None:
            if self.config.model_type == "mla":
                past_length = kv_cache[0][0].shape[1]
            else:
                past_length = kv_cache[0][0].shape[2]
            position_ids = torch.arange(past_length, past_length + T, device=input_ids.device)
        
        # Process through layers
        new_cache = [] if use_cache else None
        
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = kv_cache[layer_idx] if kv_cache is not None else None
            
            # OPTIMIZATION: Selective Activation Checkpointing (even layers only)
            # Checkpointing discards activations and recomputes during backward pass.
            # Selective (vs all layers) reduces VRAM by ~40% while maintaining ~95% speed.
            # Can be disabled for high-VRAM GPUs (H200/B200) to maximize throughput.
            do_checkpoint = (self.use_checkpointing and 
                           self.training and 
                           not use_cache and 
                           torch.is_grad_enabled())
            if do_checkpoint and (layer_idx % 2 == 0):
                x, layer_new_cache = torch.utils.checkpoint.checkpoint(
                    layer, 
                    x, 
                    position_ids, 
                    layer_cache, 
                    use_cache,
                    use_reentrant=False  # False is recommended for new code (no side effects)
                )
            else:
                x, layer_new_cache = layer(x, position_ids, layer_cache, use_cache)
            
            if use_cache:
                new_cache.append(layer_new_cache)
        
        # Final norm
        x = self.norm_f(x)
        
        # Compute logits
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=-100
            )
        
        return logits, loss, new_cache
    
    def configure_optimizers(
        self, 
        weight_decay: float, 
        learning_rate: float, 
        device_type: str
    ) -> torch.optim.Optimizer:
        """
        Configure AdamW optimizer with weight decay on 2D+ tensors only.
        """
        # Separate parameters into decay and no-decay groups
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"Optimizer: {len(decay_params)} decay tensors ({num_decay:,} params), "
              f"{len(nodecay_params)} no-decay tensors ({num_nodecay:,} params)")
        
        # Use fused AdamW if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"Using fused AdamW: {use_fused}")
        
        optimizer = torch.optim.AdamW(
            optim_groups, 
            lr=learning_rate, 
            betas=(0.9, 0.95), 
            eps=1e-8, 
            fused=use_fused
        )
        
        return optimizer
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Initial tokens (batch, seq_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            Generated tokens (batch, seq_len + max_new_tokens)
        """
        self.eval()
        
        # Initialize KV cache
        kv_cache = [None] * self.config.n_layers
        
        # Process prompt
        if input_ids.shape[1] > 1:
            # Prefill: process all prompt tokens at once
            _, _, kv_cache = self.forward(input_ids, use_cache=True)
            # Get last token for continuation
            next_input = input_ids[:, -1:]
        else:
            next_input = input_ids
        
        generated = input_ids
        
        for _ in range(max_new_tokens):
            # Forward pass with cache
            logits, _, kv_cache = self.forward(
                next_input, 
                kv_cache=kv_cache, 
                use_cache=True
            )
            
            # Get logits for next token
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            next_input = next_token
            
            # Check for max length
            if generated.shape[1] >= self.config.block_size:
                break
        
        return generated
