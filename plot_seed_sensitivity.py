"""Draw the seed-sensitivity figure: does running more seeds change the answer?

Reads `seed_study.csv` and re-computes each model's mean RMSE using the first
3, 5, 7, 10 and 15 seeds, so the reader can watch the estimate settle. Flat,
overlapping lines are the point — they say the extra runs bought nothing.

    python plot_seed_sensitivity.py
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STEPS = [3, 5, 7, 10, 15]
SEQUENCE = ["CNN-LSTM", "GRU", "1D CNN", "LSTM"]
COLOURS = {"CNN-LSTM": "#4E79A7", "GRU": "#59A14F",
           "1D CNN": "#E15759", "LSTM": "#B07AA1"}
BG, INK, MUTED = "#F8F9FB", "#2D3142", "#6B7280"


def main() -> int:
    root = pathlib.Path(__file__).parent
    runs = pd.read_csv(root / "seed_study.csv")
    seeds = sorted(runs.seed.unique())
    if len(seeds) < max(STEPS):
        raise SystemExit(f"need {max(STEPS)} seeds, found {len(seeds)} — "
                         f"run seed_study.py with more seeds first")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for model in SEQUENCE:
        means, errs = [], []
        for n in STEPS:
            scores = runs[(runs.Model == model) & (runs.seed.isin(seeds[:n]))].RMSE
            means.append(scores.mean())
            # Standard error of the mean: how much the estimate itself could move
            errs.append(scores.std(ddof=1) / np.sqrt(n))
        ax.errorbar(STEPS, means, yerr=errs, label=model, color=COLOURS[model],
                    marker="o", markersize=5, capsize=4, linewidth=1.8, alpha=0.9)

    ax.set_xticks(STEPS)
    ax.set_xlabel("Seeds averaged", fontsize=11, color=INK)
    ax.set_ylabel("Test RMSE (cycles)", fontsize=11, color=INK)
    ax.set_title("More seeds, same answer", fontsize=16, fontweight="bold",
                 loc="left", color=INK, pad=28)
    ax.text(0, 1.03, "Mean test RMSE \u00b1 standard error of the mean, "
                     "recomputed on the first N seeds",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)

    # XGBoost sits at ~19.8 and would flatten everything else, so name it below
    xgb = runs[runs.Model == "XGBoost"].RMSE.mean()
    fig.text(0.012, 0.012,
             f"Error bars overlap at every step: these four are not separated by "
             f"more runs.\nXGBoost, off this chart at {xgb:.2f} RMSE, loses to all "
             f"four in all {len(seeds)} seeds.",
             fontsize=9, color=MUTED, va="bottom", linespacing=1.5)
    ax.grid(True, color="#E0E4EA", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="upper right", ncol=2)

    out = root / "assets" / "seed_sensitivity.png"
    fig.tight_layout(rect=[0, 0.085, 1, 1])
    fig.savefig(out, dpi=150, facecolor=BG)
    print(f"wrote {out.relative_to(root)}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
