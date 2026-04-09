"""Plot KV cache memory vs context length for Hybrid SWA+GQA vs MLA.

Reads the JSON outputs of evaluate.py --memory-only from
./eval_results/memory_{hybrid,mla}.json and produces a single chart with both
architectures on shared axes. Output: kv_cache.png next to this script.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

EVAL_DIR = Path(__file__).resolve().parent / "eval_results"
OUT_PATH = Path(__file__).resolve().parent / "kv_cache.png"


def load_memory(path: Path) -> tuple[list[int], list[float]]:
    data = json.loads(path.read_text())
    contexts = [r["context_length"] for r in data["results"]]
    cache_mb = [r["kv_cache_mb"] for r in data["results"]]
    return contexts, cache_mb


def main() -> None:
    h_ctx, h_mb = load_memory(EVAL_DIR / "memory_hybrid.json")
    m_ctx, m_mb = load_memory(EVAL_DIR / "memory_mla.json")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    ax.plot(
        h_ctx, h_mb,
        marker="o", lw=1.8, color="#1f77b4",
        label="Hybrid SWA+GQA",
    )
    ax.plot(
        m_ctx, m_mb,
        marker="s", lw=1.8, color="#d62728",
        label="MLA",
    )

    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("KV cache size (MB, bf16)")
    ax.set_title(
        "KV cache memory vs context length\n"
        "500M params, 24 layers, n_heads=20, head_dim=64"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=False)

    # Annotate the MLA/Hybrid ratio at each context length, matching the
    # table convention in FYP_thoughts.md ("Hybrid is N× smaller").
    for ctx, h, m in zip(h_ctx, h_mb, m_mb):
        ratio = m / h if h > 0 else float("nan")
        ax.annotate(
            f"{ratio:.1f}x",
            xy=(ctx, max(h, m)),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#444",
        )

    # Force the x-axis to show actual context lengths as ticks.
    ax.set_xticks(h_ctx)
    ax.set_xticklabels([str(c) for c in h_ctx])

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
