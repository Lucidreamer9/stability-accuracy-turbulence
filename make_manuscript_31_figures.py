#!/usr/bin/env python3
"""
Generate manuscript figures for Results 3.1.

Outputs:
  manuscript/turbulence_reverse_simulation/eval_outputs/sample_reconstructions_nn_with_T.png
  manuscript/turbulence_reverse_simulation/eval_outputs/sample_reconstructions_DA_with_T.png
  manuscript/turbulence_reverse_simulation/eval_outputs/sample_reconstructions_NN_DA_combined.png
  manuscript/turbulence_reverse_simulation/eval_outputs/vorticity_spectrum_nn_all_T.png
  manuscript/turbulence_reverse_simulation/eval_outputs/vorticity_spectrum_DA_all_T.png
"""

import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator

ROOT = "/export/yexiaohe/sdsu/ge"
MANU_OUT = os.path.join(ROOT, "manuscript/turbulence_reverse_simulation/eval_outputs")
DL_DIR = os.path.join(ROOT, "eval_outputs_phy")
DA_DIR = "/export/yexiaohe/sdsu/DA_DHIT_pngs_and_obs"
TS = [0.2, 0.5, 0.8, 1.2, 1.6, 2.0]
EX_NUM = 1

plt.rcParams.update({
    "axes.grid": False,
    "legend.frameon": False,
})


def vorticity_spectrum(omega_phys):
    """Shell-averaged vorticity spectrum Z(k) = <|omega_hat(k)|^2>."""
    n = omega_phys.shape[0]
    omega_hat = np.fft.fft2(omega_phys, norm="ortho")
    kx = np.fft.fftfreq(n) * n
    ky = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    density = np.abs(omega_hat) ** 2
    k_mag = np.sqrt(kx**2 + ky**2)
    k_bin = np.rint(k_mag).astype(int)
    k_max = n // 2
    spec = np.zeros(k_max, dtype=np.float64)
    counts = np.zeros(k_max, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            k = k_bin[i, j]
            if 1 <= k < k_max:
                spec[k] += density[i, j]
                counts[k] += 1
    counts = np.maximum(counts, 1)
    return spec / counts


def fmt_t(t):
    return f"{t:g}"


def apply_da_sign_convention(omega_true, omega_re):
    corr = np.corrcoef(omega_re[:, :, 0].ravel(), omega_true[:, :, 0].ravel())[0, 1]
    if corr < 0:
        omega_re = -omega_re
    else:
        omega_re = -omega_re
        omega_true = -omega_true
    return omega_true, omega_re


def load_dl_columns():
    cols = []
    for T in TS:
        path = os.path.join(DL_DIR, f"forward_trajectory_data_T_{fmt_t(T)}_ex{EX_NUM}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        z = np.load(path)
        cols.append(
            {
                "T": float(z["T"]),
                "input": z["obs_phys"],
                "recon": z["traj_pred_phys"][0],
                "true": z["traj_true_phys"][0],
            }
        )
    return cols


def load_da_columns():
    cols = []
    for T in TS:
        pattern = os.path.join(
            DA_DIR,
            f"DA_DHIT_*_T_{fmt_t(T)}_ex{EX_NUM}",
            "observation_*.mat",
        )
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(pattern)
        data = sio.loadmat(matches[0])
        omega_true = data["omega_true"]
        omega_re = data["omega_re"]
        omega_true, omega_re = apply_da_sign_convention(omega_true, omega_re)
        cols.append(
            {
                "T": float(data["T_final"].flat[0]),
                "input": omega_true[:, :, -1],
                "recon": omega_re[:, :, 0],
                "true": omega_true[:, :, 0],
            }
        )
    return cols


def plot_recon_grid(cols, out_path):
    nT = len(cols)
    rows = [
        (r"$\omega_T$", [c["input"] for c in cols], "state"),
        (r"$\hat{\omega}_0$", [c["recon"] for c in cols], "state"),
        (r"$\omega_0$", [c["true"] for c in cols], "state"),
        (r"$\hat{\omega}_0-\omega_0$", [c["recon"] - c["true"] for c in cols], "error"),
    ]

    state_vlim = max(
        np.max(np.abs(x))
        for _, arrs, kind in rows
        if kind == "state"
        for x in arrs
    )
    input_vlim = max(np.max(np.abs(x)) for x in rows[0][1])
    error_vlim = max(np.max(np.abs(x)) for x in rows[-1][1])
    vlims = [input_vlim, state_vlim, state_vlim, error_vlim]

    fig = plt.figure(figsize=(1.9 * nT + 0.85, 8.15))
    gs = GridSpec(
        4,
        nT + 2,
        figure=fig,
        width_ratios=[0.12] + [1] * nT + [0.045],
        wspace=0.0,
        hspace=0.08,
    )

    for r, (row_label, arrs, _) in enumerate(rows):
        ax_label = fig.add_subplot(gs[r, 0])
        ax_label.axis("off")
        ax_label.text(
            0.5,
            0.5,
            row_label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        last_im = None
        for c, arr in enumerate(arrs):
            ax = fig.add_subplot(gs[r, c + 1])
            last_im = ax.imshow(
                arr,
                origin="lower",
                cmap="jet",
                vmin=-vlims[r],
                vmax=vlims[r],
                interpolation="nearest",
            )
            if r == 0:
                ax.set_title(rf"$T={cols[c]['T']:g}$", fontsize=12, pad=4)
            ax.axis("off")
        cax = fig.add_subplot(gs[r, -1])
        cb = fig.colorbar(last_im, cax=cax)
        cb.ax.tick_params(labelsize=8, pad=1)
        cb.locator = MaxNLocator(nbins=5)
        cb.update_ticks()

    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_combined_recon_grid(nn_cols, da_cols, out_path,
                             T_picks=(0.2, 0.5, 1.2, 2.0)):
    """Transposed layout: rows = horizons (4 picked), columns = quantities.

    Every cell has its own vertical colorbar on its right, using per-cell
    symmetric scaling (95th-percentile of |value|).
    """
    def pick(cols):
        return [min(cols, key=lambda c: abs(c['T'] - tp)) for tp in T_picks]
    nn_cols = pick(nn_cols)
    da_cols = pick(da_cols)
    nT = len(nn_cols)

    columns = [
        (r"$\omega_T$",                              [c["input"]              for c in nn_cols]),
        (r"$\omega_0$",                               [c["true"]               for c in nn_cols]),
        (r"$\hat{\omega}_0^{\mathrm{NN}}$",           [c["recon"]              for c in nn_cols]),
        (r"$\hat{\omega}_0^{\mathrm{DA}}$",           [c["recon"]              for c in da_cols]),
        (r"$\hat{\omega}_0^{\mathrm{NN}}-\omega_0$",  [c["recon"] - c["true"]  for c in nn_cols]),
        (r"$\hat{\omega}_0^{\mathrm{DA}}-\omega_0$",  [c["recon"] - c["true"]  for c in da_cols]),
    ]
    nC = len(columns)

    # Outer grid: label column + nC image columns; rows are horizons.
    # Each (r, c) cell is split into (image, thin colorbar) via sub-gridspec
    # so colorbars sit BELOW the image, not overlapping it, while the image
    # columns themselves stay tightly packed horizontally.
    width_ratios = [0.30] + [1.0] * nC
    fig = plt.figure(figsize=(1.55 * nC + 0.55, 1.38 * nT + 0.32))
    outer = GridSpec(nT, 1 + nC, figure=fig,
                     width_ratios=width_ratios,
                     wspace=0.04, hspace=0.03)

    # T labels on the left
    for r in range(nT):
        ax_label = fig.add_subplot(outer[r, 0])
        ax_label.axis("off")
        ax_label.text(0.5, 0.5, rf"$T={nn_cols[r]['T']:g}$",
                      rotation=0, ha="center", va="center",
                      fontsize=11, fontweight="bold")

    def _fmt(v):
        av = abs(v)
        if av >= 10:
            return f"{v:.0f}"
        if av >= 1:
            return f"{v:.1f}"
        return f"{v:.2f}"

    for c, (label, arrs) in enumerate(columns):
        for r in range(nT):
            vlim = float(np.percentile(np.abs(arrs[r]), 95)) or 1.0
            inner = GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[r, 1 + c],
                height_ratios=[1.0, 0.05], hspace=0.04,
            )
            ax = fig.add_subplot(inner[0])
            im = ax.imshow(
                arrs[r], origin="lower", cmap="RdBu_r",
                vmin=-vlim, vmax=vlim,
                interpolation="nearest",
            )
            if r == 0:
                ax.set_title(label, fontsize=11, pad=3)
            ax.axis("off")
            # Footer band: short colorbar in the middle, ±vmax text on the sides.
            footer = fig.add_subplot(inner[1])
            footer.axis("off")
            footer.set_xlim(0, 1)
            footer.set_ylim(0, 1)
            bar_l, bar_r = 0.22, 0.78
            cax = footer.inset_axes([bar_l, 0.05, bar_r - bar_l, 0.90])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[])
            cb.outline.set_linewidth(0.3)
            cb.outline.set_edgecolor("black")
            footer.text(bar_l - 0.02, 0.5, _fmt(-vlim),
                        ha="right", va="center", fontsize=5)
            footer.text(bar_r + 0.02, 0.5, _fmt(vlim),
                        ha="left", va="center", fontsize=5)

    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_vorticity_spectrum(cols, out_path):
    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    cmap = plt.get_cmap("viridis", len(cols))
    for i, col in enumerate(cols):
        color = cmap(i)
        spec_true = vorticity_spectrum(col["true"])
        spec_recon = vorticity_spectrum(col["recon"])
        k = np.arange(len(spec_true))
        ax.loglog(k[1:], spec_true[1:], "--", color=color, alpha=0.75, linewidth=1.2)
        ax.loglog(
            k[1:],
            spec_recon[1:],
            "-",
            color=color,
            linewidth=1.6,
            label=rf"$T={col['T']:g}$",
        )
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$Z(k)$")
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(MANU_OUT, exist_ok=True)
    dl_cols = load_dl_columns()
    da_cols = load_da_columns()

    plot_recon_grid(
        dl_cols,
        os.path.join(MANU_OUT, "sample_reconstructions_nn_with_T.png"),
    )
    plot_recon_grid(
        da_cols,
        os.path.join(MANU_OUT, "sample_reconstructions_DA_with_T.png"),
    )
    plot_combined_recon_grid(
        dl_cols,
        da_cols,
        os.path.join(MANU_OUT, "sample_reconstructions_NN_DA_combined.png"),
    )
    plot_vorticity_spectrum(
        dl_cols,
        os.path.join(MANU_OUT, "vorticity_spectrum_nn_all_T.png"),
    )
    plot_vorticity_spectrum(
        da_cols,
        os.path.join(MANU_OUT, "vorticity_spectrum_DA_all_T.png"),
    )

    print("Saved manuscript 3.1 figures:")
    print(os.path.join(MANU_OUT, "sample_reconstructions_nn_with_T.png"))
    print(os.path.join(MANU_OUT, "sample_reconstructions_DA_with_T.png"))
    print(os.path.join(MANU_OUT, "sample_reconstructions_NN_DA_combined.png"))
    print(os.path.join(MANU_OUT, "vorticity_spectrum_nn_all_T.png"))
    print(os.path.join(MANU_OUT, "vorticity_spectrum_DA_all_T.png"))


if __name__ == "__main__":
    main()
