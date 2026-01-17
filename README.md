# Yadda - Efficient Attention Research

Comparative study of 500M parameter transformers for edge deployment:
- **Hybrid SWA** (Gemma-style): Alternating Sliding Window + Global GQA Attention
- **MLA** (DeepSeek-style): Multi-Head Latent Attention with KV compression

## Project Structure

```
yadda/
├── models/          # Core model implementations
│   ├── attention.py # Hybrid SWA + MLA attention mechanisms
│   ├── transformer.py # Main Transformer with checkpointing
│   ├── config.py    # 500M model configurations
│   └── rope.py      # Rotary Positional Embeddings
├── scripts/         # Training and inference scripts
│   ├── train_edge.py      # DDP training (H100/H200)
│   ├── inference.py       # Portable inference + benchmarking
│   └── edu_fineweb10B.py  # FineWeb-Edu dataset tokenization
├── archive/         # Old experiments (for reference)
└── log/             # Checkpoints and training logs
```

## Quick Start

### Training on H200 (Recommended)
```bash
cd scripts
# Single GPU with auto-optimization
python train_edge.py --model_type hybrid --auto_optimize --compile

# Multi-GPU DDP
torchrun --standalone --nproc_per_node=8 train_edge.py --model_type mla --auto_optimize --compile
```

### Inference on Local GPU
```bash
cd scripts
python inference.py --checkpoint ../log/hybrid_05000.pt --interactive
```

## Data Preparation
```bash
cd scripts
python edu_fineweb10B.py --output_dir /path/to/edu_fineweb10B
```

## Key Features
- **Auto-optimization**: Detects GPU (H200/H100/A100/3090) and adjusts batch size + checkpointing
- **Model checkpoints**: Saved every 5000 steps to `log/{model_type}_{step:05d}.pt`
- **Dual-path training**: Uses FlexAttention/FlashAttention for training, naive PyTorch for portable inference
