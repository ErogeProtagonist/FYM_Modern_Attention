import torch
import sys
import os

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

# Check FlexAttention
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    print("FlexAttention: AVAILABLE (torch.nn.attention.flex_attention)")
    FLEX_AVAILABLE = True
except ImportError:
    try:
        from torch.nn.attention.flex import flex_attention, create_block_mask
        print("FlexAttention: AVAILABLE (torch.nn.attention.flex)")
        FLEX_AVAILABLE = True
    except ImportError as e:
        print(f"FlexAttention: NOT AVAILABLE ({e})")
        FLEX_AVAILABLE = False

# Check SDPA
print("SDPA Backends:")
# This logic works for standard SDPA, not necessarily FlashMLA which is custom
# But we can check if naive SDPA works
try:
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
        print("  FLASH_ATTENTION: Supported (context manager didn't fail)")
except Exception as e:
    print(f"  FLASH_ATTENTION: Not supported/Failed ({e})")

