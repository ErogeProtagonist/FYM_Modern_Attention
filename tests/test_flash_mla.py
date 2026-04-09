import torch
from models.config import MLA_500M
from models.attention import FlashMLAttention

def test_flash_mla_forward():
    print("Testing FlashMLAttention Forward Pass (SDPA)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    config = MLA_500M
    config.d_model = 128  # Small model for test
    config.n_heads = 4
    config.head_dim = 32
    config.rope_dim = 16
    config.kv_lora_rank = 64
    
    attn = FlashMLAttention(config, layer_idx=0).to(device)
    
    B, S = 2, 128
    x = torch.randn(B, S, config.d_model, device=device)
    
    # Forward pass
    try:
        y, _ = attn(x)
        print(f"Output shape: {y.shape}")
        assert y.shape == (B, S, config.d_model)
        with open("test_result.txt", "w") as f:
            f.write("SUCCESS")
    except Exception as e:
        with open("test_result.txt", "w") as f:
            f.write(f"FAILURE: {e}")
        print(f"FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_flash_mla_forward()
