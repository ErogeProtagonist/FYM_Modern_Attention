# FYM_Modern_Attention

**Efficiency at the Edge: Hybrid Sparsity vs. Latent Compression in 500M Transformers**

Comparative study of two state-of-the-art attention mechanisms for edge deployment:
- **Hybrid SWA+GQA** (Google Gemma-style): Sliding Window + Grouped Query Attention
- **MLA** (DeepSeek-style): Multi-Head Latent Attention with low-rank KV compression

## Results

| Model | Final Val Loss | Throughput (2x B200) | Training Cost |
|-------|----------------|----------------------|---------------|
| **Hybrid SWA+GQA** | 2.7207 | ~364k tok/sec | ~$76 |
| **MLA** | 2.7290 | ~187k tok/sec | ~$150 |
| **Δ Gap** | **0.0083** | - | **$226 total** |

**Key Finding**: Both architectures achieve equivalent language modelling quality (~2.72 loss), but MLA offers **~60% KV cache reduction** for inference on memory-constrained devices.

---

## Project Structure

```
yadda/
├── models/              # Core model implementations
│   ├── attention.py     # Hybrid SWA + MLA attention mechanisms
│   ├── transformer.py   # Transformer with selective checkpointing
│   ├── config.py        # 500M model configurations
│   └── rope.py          # Rotary Positional Embeddings (standard + decoupled)
├── scripts/             # Pre-training scripts
│   ├── train_edge.py    # DDP training (H100/H200/B200)
│   ├── inference.py     # Portable inference + benchmarking
│   └── edu_fineweb10B.py # FineWeb-Edu dataset tokenization
├── sft/                 # Supervised Fine-Tuning
│   ├── sft_data_prep.py # Download & prepare smol-smoltalk dataset
│   └── sft_train.py     # SFT training with ChatML format
└── archive/             # Old experiments (reference only)
```

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
    --checkpoint ../checkpoints/hybrid_19073.pt \
    --data_dir sft_data \
    --batch_size 16 \
    --epochs 3
```

### 3. Inference

```bash
cd scripts

# Interactive mode
python inference.py --checkpoint ../checkpoints/hybrid_19073.pt --interactive

# Benchmark memory + speed
python inference.py --checkpoint ../checkpoints/mla_19073.pt --benchmark
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
- **536M parameters** with 24 layers, 1280 d_model, 20 query heads
- **Hybrid**: 5:1 ratio of SWA (window=512) to Global GQA layers
- **MLA**: Low-rank KV compression ($d_c$=512) with decoupled RoPE

### Training Optimizations
- **Auto-optimization**: Detects GPU (B200/H200/H100/3090) and adjusts batch size + checkpointing
- **Block mask caching**: 10x speedup for FlexAttention sliding window
- **Fused SwiGLU FFN**: Single linear + chunk instead of two projections
- **Selective gradient checkpointing**: Alternating layers for memory/speed balance

### Dual-Path Architecture
- **Training**: FlexAttention/FlashAttention for maximum throughput
- **Inference**: Naive PyTorch for portability (works on any GPU/CPU)

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
