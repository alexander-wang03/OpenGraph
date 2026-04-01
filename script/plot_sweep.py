#!/usr/bin/env python3
"""
Plot compression sweep results — delta vs precision & compression ratio.

Usage:
    cd /home/awang/Documents/TRAILbot/OpenGraph
    python script/plot_sweep.py --config-name=isaac_warehouse sequence=04
"""

import hydra
from omegaconf import DictConfig
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


@hydra.main(version_base=None, config_path="../config")
def main(cfg):
    seq = f"{cfg.sequence:02d}" if isinstance(cfg.sequence, int) else cfg.sequence
    result_dir = Path(f"../results/warehouse_{seq}/pcd")
    csv_path = result_dir / "sweep_compression_results.csv"
    out = result_dir

    print(f"Reading sweep results from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Separate baseline and compressed rows
    baseline = df[df["task_set"] == "none"].iloc[0]
    generic = df[df["task_set"] == "generic"].sort_values("delta")
    eval_ = df[df["task_set"] == "eval"].sort_values("delta")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"IB Compression Sweep — Seq {seq}", fontsize=14, fontweight="bold")

    # ---- Plot 1: Compression ratio vs delta ----
    ax = axes[0, 0]
    ax.plot(generic["delta"], generic["compression_ratio"],
            "o-", color="#2196F3", label="Generic tasks", linewidth=2, markersize=7)
    ax.plot(eval_["delta"], eval_["compression_ratio"],
            "s--", color="#FF9800", label="Eval tasks", linewidth=2, markersize=7)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="No compression")
    ax.set_xlabel("Delta (IB threshold)")
    ax.set_ylabel("Compression ratio (objects / units)")
    ax.set_title("Compression Ratio vs Delta")
    ax.legend()
    ax.set_xlim(0.03, 0.32)

    # ---- Plot 2: Units after vs delta ----
    ax = axes[0, 1]
    ax.plot(generic["delta"], generic["units_after"],
            "o-", color="#2196F3", label="Generic tasks", linewidth=2, markersize=7)
    ax.plot(eval_["delta"], eval_["units_after"],
            "s--", color="#FF9800", label="Eval tasks", linewidth=2, markersize=7)
    ax.axhline(y=baseline["units_after"], color="gray", linestyle=":",
               alpha=0.5, label=f"Baseline ({int(baseline['units_after'])} objects)")
    ax.set_xlabel("Delta (IB threshold)")
    ax.set_ylabel("Retrieval units (objects + clusters)")
    ax.set_title("Node Count vs Delta")
    ax.legend()
    ax.set_xlim(0.03, 0.32)

    # ---- Plot 3: Flat P@1 (all queries) vs delta ----
    ax = axes[1, 0]
    ax.plot(generic["delta"], generic["flat_P@1_all"],
            "o-", color="#2196F3", label="Generic (flat)", linewidth=2, markersize=7)
    ax.plot(eval_["delta"], eval_["flat_P@1_all"],
            "s--", color="#FF9800", label="Eval (flat)", linewidth=2, markersize=7)
    ax.plot(generic["delta"], generic["hier_P@1_all"],
            "^-", color="#4CAF50", label="Generic (hier)", linewidth=2, markersize=7)
    ax.plot(eval_["delta"], eval_["hier_P@1_all"],
            "d--", color="#F44336", label="Eval (hier)", linewidth=2, markersize=7)
    ax.axhline(y=baseline["flat_P@1_all"], color="#2196F3", linestyle=":",
               alpha=0.4, label=f"Baseline flat ({baseline['flat_P@1_all']}%)")
    ax.axhline(y=baseline["hier_P@1_all"], color="#4CAF50", linestyle=":",
               alpha=0.4, label=f"Baseline hier ({baseline['hier_P@1_all']}%)")
    ax.set_xlabel("Delta (IB threshold)")
    ax.set_ylabel("Precision@1 (%)")
    ax.set_title("P@1 (All Queries) vs Delta")
    ax.legend(fontsize=8)
    ax.set_xlim(0.03, 0.32)
    ax.set_ylim(0, 50)

    # ---- Plot 4: Flat P@1 (reachable) vs compression ratio ----
    ax = axes[1, 1]
    ax.scatter(baseline["compression_ratio"], baseline["flat_P@1_reach"],
               marker="*", s=200, color="black", zorder=5, label="Baseline (flat)")
    ax.scatter(baseline["compression_ratio"], baseline["hier_P@1_reach"],
               marker="*", s=200, color="#4CAF50", zorder=5, label="Baseline (hier)")

    ax.plot(generic["compression_ratio"], generic["flat_P@1_reach"],
            "o-", color="#2196F3", label="Generic (flat)", linewidth=2, markersize=7)
    ax.plot(eval_["compression_ratio"], eval_["flat_P@1_reach"],
            "s--", color="#FF9800", label="Eval (flat)", linewidth=2, markersize=7)
    ax.plot(generic["compression_ratio"], generic["hier_P@1_reach"],
            "^-", color="#4CAF50", label="Generic (hier)", linewidth=2, markersize=6)
    ax.plot(eval_["compression_ratio"], eval_["hier_P@1_reach"],
            "d--", color="#F44336", label="Eval (hier)", linewidth=2, markersize=6)

    ax.set_xlabel("Compression ratio")
    ax.set_ylabel("Precision@1 — reachable (%)")
    ax.set_title("P@1 (Reachable) vs Compression Ratio")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 60)

    plt.tight_layout()
    plot_path = out / "sweep_compression_plot.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
