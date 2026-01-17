import torch
from models.config import HYBRID_500M
from models.attention import NaiveHybridAttention

def verify_swa_shapes():
    print("Verifying SWA KV Cache Shapes...")
    
    # Config: Window size 512, Global period 6
    config = HYBRID_500M
    config.window_size = 4  # Small window for easy testing
    config.global_layer_period = 2 # Every 2nd layer is global (layers 1, 3, etc. are global; 0, 2 are local)
    
    print(f"Test Configuration: Window={config.window_size}, Global Period={config.global_layer_period}")
    
    # 1. Local Layer (Layer 0) - Should act as Sliding Window
    local_attn = NaiveHybridAttention(config, layer_idx=0)
    
    # 2. Global Layer (Layer 1) - Should act as Global Attention
    global_attn = NaiveHybridAttention(config, layer_idx=1)
    
    # Mock input
    B, n_heads, head_dim = 1, config.n_heads, config.head_dim
    
    # Simulate generation steps
    # We will simulate more steps than window_size (e.g., 6 steps > 4 window)
    
    cache_local = None
    cache_global = None
    
    print("\n--- Starting Generation Steps ---")
    
    for i in range(6):
        x = torch.randn(B, 1, config.d_model) # Single token input
        
        # Forward pass Local
        _, cache_local = local_attn(x, kv_cache=cache_local, use_cache=True)
        
        # Forward pass Global
        _, cache_global = global_attn(x, kv_cache=cache_global, use_cache=True)
        
        k_local, _ = cache_local
        k_global, _ = cache_global
        
        print(f"Step {i+1}:")
        print(f"  Local Layer Cache Shape:  {k_local.shape} {'(Capped at 4!)' if k_local.shape[2] == 4 else ''}")
        print(f"  Global Layer Cache Shape: {k_global.shape}")
        
    # Verification
    if cache_local[0].shape[2] == 4 and cache_global[0].shape[2] == 6:
        print("\nSUCCESS: Local layer capped at window size, Global layer grew indefinitely.")
    else:
        print("\nFAILURE: Cache shapes did not match expected behavior.")

if __name__ == "__main__":
    verify_swa_shapes()
