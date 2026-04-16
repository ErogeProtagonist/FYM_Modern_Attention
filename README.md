# FYM_Modern_Attention

**Efficiency at the Edge: Hybrid Sparsity vs. Latent Compression in 500M Transformers**

Comparative study of two state-of-the-art attention mechanisms for edge deployment:
- **Hybrid SWA+GQA** (Google Gemma-style): Sliding Window + Grouped Query Attention
- **MLA** (DeepSeek-style): Multi-Head Latent Attention with low-rank KV compression

## Results

| Model | Final Val Loss | Throughput (2x B200) |
|-------|----------------|----------------------|
| **Hybrid SWA+GQA** | 2.7207 | ~364k tok/sec |
| **MLA** | 2.7290 | ~187k tok/sec |
| **Δ Gap** | **0.0083** | - |

**Key Finding**: Both architectures reach equivalent quality (~2.72 val loss
and within-noise on HellaSwag/ARC/PIQA). They differ in how they manage the
inference KV cache: Hybrid truncates context via SWA, MLA compresses every
token into a shared latent — a trade-off between forgetting and full-context
preservation, not a single-axis winner.

---

## Project Structure

```
swa-mla-500m/
├── models/                    # Core model implementations
│   ├── attention.py           # Hybrid SWA+GQA + MLA attention mechanisms
│   ├── transformer.py         # Transformer with selective checkpointing
│   ├── config.py              # 500M model configurations
│   └── rope.py                # Rotary Positional Embeddings (standard + decoupled)
├── scripts/                   # Training, inference, evaluation
│   ├── train_edge.py          # DDP training (H100/H200/B200)
│   ├── inference.py           # Portable inference (interactive + single-prompt)
│   ├── evaluate.py            # lm-eval-harness wrapper (HellaSwag/ARC/PIQA)
│   ├── check_kv_cache_shapes.py  # KV-cache shape sanity check
│   ├── plot_kv_cache.py       # Cache scaling figure
│   ├── plot_val_loss.py       # Pre-training loss curve figure
│   └── edu_fineweb10B.py      # FineWeb-Edu dataset tokenization
├── sft/                       # Supervised Fine-Tuning
│   ├── sft_data_prep.py       # Download & prepare smol-smoltalk dataset
│   └── sft_train.py           # SFT training with ChatML format
├── tests/                     # Parity / equivalence tests
│   ├── parity_hybrid.py       # Train-mode vs inference-mode logit parity
│   └── verify_swa.py          # SWA cache shape sanity check
└── learning-experiments/      # Notebooks and exploratory work (reference only)
```

Trained checkpoints live **outside** the repo at `../Checkpoints/`
(`hybrid_19072.pt` and `mla_19072.pt`).

---

## Quick Start

### 1. Pre-training (10B tokens on FineWeb-Edu)

```bash
cd scripts

# Single GPU with auto-optimization
python train_edge.py --model_type hybrid --auto_optimize --compile

# Multi-GPU DDP (2x B200)
torchrun --standalone --nproc_per_node=2 train_edge.py \
    --model_type mla --auto_optimize --compile
```

### 2. Supervised Fine-Tuning (smol-smoltalk)

```bash
cd sft

# Step 1: Prepare data (downloads HuggingFaceTB/smol-smoltalk)
python sft_data_prep.py --output_dir sft_data

# Step 2: Fine-tune pretrained model
python sft_train.py \
    --checkpoint ../../Checkpoints/hybrid_19072.pt \
    --data_dir sft_data \
    --batch_size 16 \
    --epochs 3
```

### 3. Inference

```bash
cd scripts

# Interactive mode
python inference.py --checkpoint ../../Checkpoints/hybrid_19072.pt --interactive

# Single prompt
python inference.py --checkpoint ../../Checkpoints/mla_19072.pt --prompt "Hello, I am"
```

---

## Data Preparation

### Pre-training Data
```bash
cd scripts
# Tokenize FineWeb-Edu (10B tokens)
python edu_fineweb10B.py --output_dir /path/to/edu_fineweb10B
```

### SFT Data
Using **smol-smoltalk** from HuggingFace — specifically designed for sub-1B models:
- Shorter conversations (avoids capacity overflow)
- No advanced math/function calling (prevents catastrophic forgetting)
- Reference: [SmolLM2 Paper](https://huggingface.co/papers/2502.02737)

---

## Key Features

### Model Architecture
- **~500M parameters** with 24 layers, 1280 d_model, 20 query heads
- **Hybrid**: 5:1 ratio of SWA (window=512) to Global GQA layers
- **MLA**: Low-rank KV compression ($d_c$=512) with decoupled RoPE

### Training Optimizations
- **Auto-optimization**: Detects GPU (B200/H200/H100/3090) and adjusts batch size + checkpointing
- **FlashAttention-2 native sliding window**: training SWA layers use `window_size=(window-1, 0)`, no Python-side mask construction
- **Fused SwiGLU FFN**: Single linear + chunk instead of two projections
- **Selective gradient checkpointing**: Alternating layers for memory/speed balance

### Dual-Path Architecture
- **Training**: FlashAttention-2 (`flash_attn` package) with native sliding window
- **Inference**: PyTorch SDPA for portability — works on any GPU/CPU (SDPA still dispatches to FlashAttention-2 on Ampere+)

---

## Requirements

```bash
pip install torch>=2.5 tiktoken datasets tqdm
# Optional for SFT
pip install transformers
```

Hardware: NVIDIA A100/H100/H200/B200 for training, RTX 3090+ for local inference.

---

## Citation

If you use this code, please cite:
```
@misc{rahman2026fym,
  title={Efficiency at the Edge: Hybrid Sparsity vs. Latent Compression in 500M Transformers},
  author={Rahman, Ridwanur},
  year={2026}
}
```

## References
- [Gemma 2/3 Technical Reports](https://ai.google.dev/gemma) (Google DeepMind)
- [DeepSeek-V2/V3 Technical Reports](https://github.com/deepseek-ai) (DeepSeek AI)
- [SmolLM2 Paper](https://huggingface.co/papers/2502.02737) (HuggingFace)
- [FineWeb-Edu Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (HuggingFace)
