#!/usr/bin/env python3
"""Plot matched DA/NN terminal forward-consistency error for Results 3.2."""

import csv
import glob
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


ROOT = Path("/export/yexiaohe/sdsu/ge")
DA_ROOT = Path("/export/yexiaohe/sdsu/DA_DHIT_pngs_and_obs")
DL_ROOT = ROOT / "eval_outputs_phy"
OUT_DIR = ROOT / "manuscript/turbulence_reverse_simulation/eval_outputs/da_dl_compare"
OUT_CSV = OUT_DIR / "forward_terminal_error_NN_DA.csv"
OUT_PNG = OUT_DIR / "forward_terminal_error_NN_DA.png"
OUT_SPEC_PNG = OUT_DIR / "forward_terminal_spectrum_error_NN_DA.png"


def parse_da_cases():
    pat = re.compile(r"DA_DHIT_1000_uv_from_vorticity_T_([0-9.]+)_ex(\d+)$")
    for folder in sorted(DA_ROOT.glob("DA_DHIT_1000_uv_from_vorticity_T_*_ex*")):
        match = pat.match(folder.name)
        if match:
            yield float(match.group(1)), match.group(1), int(match.group(2)), folder


def load_da_case(folder):
    mats = glob.glob(os.path.join(folder, "observation_*.mat"))
    if not mats:
        return None
    data = sio.loadmat(mats[0])
    omega_true = data["omega_true"]
    omega_re = data["omega_re"]

    # Match the sign convention used by plot_da_dl_with_error_curve.py.
    corr = np.corrcoef(omega_re[:, :, 0].ravel(), omega_true[:, :, 0].ravel())[0, 1]
    if corr < 0:
        omega_re = -omega_re
    else:
        omega_re = -omega_re
        omega_true = -omega_true
    return omega_true[:, :, -1], omega_re[:, :, -1]


def load_dl_terminal(t_str, ex_num):
    path = DL_ROOT / f"forward_trajectory_data_T_{t_str}_ex{ex_num}.npz"
    if not path.exists():
        candidates = sorted(DL_ROOT.glob(f"forward_trajectory_data_T_*_ex{ex_num}.npz"))
        for candidate in candidates:
            match = re.search(r"T_([0-9.]+)_", candidate.name)
            if match and abs(float(match.group(1)) - float(t_str)) < 1e-6:
                path = candidate
                break
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return z["traj_pred_phys"][-1]


def rel_l2(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)) / (np.sqrt(np.mean(true ** 2)) + 1e-12))


def energy_spectrum_from_vorticity(omega):
    n = omega.shape[0]
    omega_hat = np.fft.fft2(omega, norm="ortho")
    kx = np.fft.fftfreq(n) * n
    ky = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    k2 = kx**2 + ky**2
    k2[0, 0] = np.inf
    density = 0.5 * (np.abs(omega_hat) ** 2) / k2
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


def spectrum_shape_distance(spec_pred, spec_true, kmin_ratio=0.4, kmax_ratio=0.75, eps=1e-12):
    k_max = len(spec_true) - 1
    k_low = max(1, int(np.floor(kmin_ratio * k_max)))
    k_high = max(k_low, int(np.floor(kmax_ratio * k_max)))
    k_high = min(k_high, k_max)
    band = np.arange(k_low, k_high + 1)
    p_pred = spec_pred / (np.sum(spec_pred[1:]) + eps)
    p_true = spec_true / (np.sum(spec_true[1:]) + eps)
    return float(np.mean(np.abs(np.log(p_pred[band] + eps) - np.log(p_true[band] + eps))))


def spec_distance(pred, true):
    return spectrum_shape_distance(
        energy_spectrum_from_vorticity(pred),
        energy_spectrum_from_vorticity(true),
    )


def compute():
    rows = []
    grouped = defaultdict(lambda: {"da": [], "nn": [], "da_spec": [], "nn_spec": []})
    for t, t_str, ex_num, folder in parse_da_cases():
        da_case = load_da_case(folder)
        dl_terminal = load_dl_terminal(t_str, ex_num)
        if da_case is None or dl_terminal is None:
            continue
        true_terminal, da_terminal = da_case
        da_err = rel_l2(da_terminal, true_terminal)
        nn_err = rel_l2(dl_terminal, true_terminal)
        da_spec = spec_distance(da_terminal, true_terminal)
        nn_spec = spec_distance(dl_terminal, true_terminal)
        rows.append((t, ex_num, da_err, nn_err, da_spec, nn_spec))
        grouped[t]["da"].append(da_err)
        grouped[t]["nn"].append(nn_err)
        grouped[t]["da_spec"].append(da_spec)
        grouped[t]["nn_spec"].append(nn_spec)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "T",
            "ex",
            "DA_forward_relL2",
            "NN_forward_relL2",
            "DA_forward_specShape",
            "NN_forward_specShape",
        ])
        writer.writerows(rows)
    return grouped


def plot_rel_l2(grouped):
    ts = sorted(grouped)
    da_mean = np.array([np.mean(grouped[t]["da"]) for t in ts])
    da_std = np.array([np.std(grouped[t]["da"], ddof=0) for t in ts])
    nn_mean = np.array([np.mean(grouped[t]["nn"]) for t in ts])
    nn_std = np.array([np.std(grouped[t]["nn"], ddof=0) for t in ts])

    fig, ax = plt.subplots(figsize=(4.9, 3.25))
    ax.plot(ts, da_mean, "o-", color="#d62728", lw=2.0, ms=4.8, label="Physics-based")
    ax.fill_between(ts, da_mean - da_std, da_mean + da_std, color="#d62728", alpha=0.18, linewidth=0)
    ax.plot(ts, nn_mean, "o-", color="#1f77b4", lw=2.0, ms=4.8, label="Data-driven")
    ax.fill_between(ts, nn_mean - nn_std, nn_mean + nn_std, color="#1f77b4", alpha=0.18, linewidth=0)
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(
        r"$\epsilon_{\mathrm{fwd}}$"
        "\n"
        r"$=\|\hat{\omega}_T-\omega_T\|_2/\|\omega_T\|_2$",
        labelpad=8,
    )
    ax.set_xlim(0.15, 2.05)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.04)


def plot_spec(grouped):
    ts = sorted(grouped)
    da_mean = np.array([np.mean(grouped[t]["da_spec"]) for t in ts])
    da_std = np.array([np.std(grouped[t]["da_spec"], ddof=0) for t in ts])
    nn_mean = np.array([np.mean(grouped[t]["nn_spec"]) for t in ts])
    nn_std = np.array([np.std(grouped[t]["nn_spec"], ddof=0) for t in ts])

    fig, ax = plt.subplots(figsize=(4.9, 3.25))
    ax.plot(ts, da_mean, "o-", color="#d62728", lw=2.0, ms=4.8, label="Physics-based")
    ax.fill_between(ts, da_mean - da_std, da_mean + da_std, color="#d62728", alpha=0.18, linewidth=0)
    ax.plot(ts, nn_mean, "o-", color="#1f77b4", lw=2.0, ms=4.8, label="Data-driven")
    ax.fill_between(ts, nn_mean - nn_std, nn_mean + nn_std, color="#1f77b4", alpha=0.18, linewidth=0)
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(
        r"$D_{\mathrm{spec}}(T)$"
        "\n"
        r"$=\langle|\log\tilde Z_{\hat{\omega}_T}(k)-\log\tilde Z_{\omega_T}(k)|\rangle_k$",
        labelpad=8,
    )
    ax.set_xlim(0.15, 2.05)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_SPEC_PNG, dpi=300, bbox_inches="tight", pad_inches=0.04)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grouped = compute()
    plot_rel_l2(grouped)
    plot_spec(grouped)
    print(OUT_CSV)
    print(OUT_PNG)
    print(OUT_SPEC_PNG)


if __name__ == "__main__":
    main()
