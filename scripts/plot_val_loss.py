"""Plot pre-training validation loss curves for Hybrid SWA+GQA vs MLA.

Reads the plain-text training logs in ../../Checkpoints/log_{hybrid,mla}.txt,
extracts the validation loss entries (lines of form "<step> val <loss>"), and
produces a side-by-side figure: full curves on the left, zoomed final-phase
gap on the right. Output: val_loss.png next to this script.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt

VAL_LINE = re.compile(r"^(\d+)\s+val\s+([0-9.]+)\s*$")

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = REPO_ROOT / "Checkpoints"
OUT_PATH = Path(__file__).resolve().parent / "val_loss.png"


def parse_val_losses(log_path: Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    losses: list[float] = []
    for line in log_path.read_text().splitlines():
        match = VAL_LINE.match(line)
        if match:
            steps.append(int(match.group(1)))
            losses.append(float(match.group(2)))
    return steps, losses


def main() -> None:
    h_steps, h_loss = parse_val_losses(CHECKPOINTS / "log_hybrid.txt")
    m_steps, m_loss = parse_val_losses(CHECKPOINTS / "log_mla.txt")

    fig, (ax_full, ax_zoom) = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.4, 1.0]}
    )

    # Left: full validation curves, log y-axis to show the early collapse.
    ax_full.plot(h_steps, h_loss, label="Hybrid SWA+GQA", color="#1f77b4", lw=1.6)
    ax_full.plot(m_steps, m_loss, label="MLA", color="#d62728", lw=1.6)
    ax_full.set_yscale("log")
    ax_full.set_xlabel("Training step")
    ax_full.set_ylabel("Validation loss (FineWeb-Edu, log scale)")
    ax_full.set_title("Full pre-training trajectory")
    ax_full.legend(loc="upper right", frameon=False)
    ax_full.grid(True, which="both", alpha=0.25)

    # Right: zoomed final phase where the gap closes.
    zoom_start = 5000
    h_zoom_idx = [i for i, s in enumerate(h_steps) if s >= zoom_start]
    m_zoom_idx = [i for i, s in enumerate(m_steps) if s >= zoom_start]
    ax_zoom.plot(
        [h_steps[i] for i in h_zoom_idx],
        [h_loss[i] for i in h_zoom_idx],
        label="Hybrid SWA+GQA",
        color="#1f77b4",
        lw=1.6,
    )
    ax_zoom.plot(
        [m_steps[i] for i in m_zoom_idx],
        [m_loss[i] for i in m_zoom_idx],
        label="MLA",
        color="#d62728",
        lw=1.6,
    )
    ax_zoom.set_xlabel("Training step")
    ax_zoom.set_ylabel("Validation loss")
    ax_zoom.set_title(f"Zoom: step {zoom_start}+ (gap closure phase)")
    ax_zoom.grid(True, alpha=0.25)
    ax_zoom.legend(loc="upper right", frameon=False)

    # Annotate final gap on the zoomed plot.
    final_step = h_steps[-1]
    final_h = h_loss[-1]
    final_m = m_loss[-1]
    final_gap = final_m - final_h
    ax_zoom.annotate(
        f"final gap: +{final_gap:.4f} nats",
        xy=(final_step, (final_h + final_m) / 2),
        xytext=(0.55, 0.85),
        textcoords="axes fraction",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.8, color="black"),
    )

    fig.suptitle(
        "Pre-training validation loss: Hybrid SWA+GQA vs MLA "
        "(500M params, FineWeb-Edu, ~10B tokens)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
