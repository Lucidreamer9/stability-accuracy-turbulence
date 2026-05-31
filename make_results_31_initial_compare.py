#!/usr/bin/env python3
"""Plot matched NN/DA initial-state error versus horizon for Results 3.1."""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = "/export/yexiaohe/sdsu/ge"
CSV = os.path.join(
    ROOT,
    "manuscript/turbulence_reverse_simulation/tables/initial_stability_diagnostics.csv",
)
OUT = os.path.join(
    ROOT,
    "manuscript/turbulence_reverse_simulation/eval_outputs/initial_error_NN_DA_matched.png",
)


def load_grouped():
    grouped = defaultdict(list)
    with open(CSV, newline="") as f:
        for row in csv.DictReader(f):
            key = (float(row["T"]), row["method"])
            grouped[key].append(float(row["relL2_init"]))
    return grouped


def summarize(grouped, method):
    ts = sorted({t for (t, m) in grouped if m == method})
    mean = np.array([np.mean(grouped[(t, method)]) for t in ts])
    std = np.array([np.std(grouped[(t, method)], ddof=0) for t in ts])
    return np.array(ts), mean, std


def main():
    grouped = load_grouped()
    fig, ax = plt.subplots(figsize=(4.8, 3.4))

    styles = {
        "NN": {"color": "#1f77b4", "label": "Data-driven"},
        "DA": {"color": "#d62728", "label": "Physics-based"},
    }
    for method in ["NN", "DA"]:
        t, mean, std = summarize(grouped, method)
        color = styles[method]["color"]
        ax.plot(t, mean, "o-", color=color, lw=2.0, ms=4.5, label=styles[method]["label"])
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)

    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$\|\hat{\omega}_0-\omega_0\|_2/\|\omega_0\|_2$")
    ax.set_xlim(0.15, 2.05)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
