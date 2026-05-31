#!/usr/bin/env python3
"""Plot NN/DA spectral distance to ground truth for Results 3.1."""

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


ROOT = Path("/export/yexiaohe/sdsu/ge")
DA_ROOT = ROOT / "results_from_DA/result_only"
MANU_OUT = ROOT / "manuscript/turbulence_reverse_simulation/eval_outputs"
NN_SUMMARY = MANU_OUT / "summary_by_T.csv"
DA_SUMMARY = MANU_OUT / "da_spectrum_error_by_T.csv"
OUT = MANU_OUT / "spectrum_error_NN_DA_full.png"


def circdiff(a, shift):
    return np.roll(a, (-shift[0], -shift[1]), axis=(0, 1)) - a


def true_uv_from_psi(psi, dx, dy):
    return circdiff(psi, (0, 1)) / dy, -circdiff(psi, (1, 0)) / dx


def vorticity_from_uv(u, v, dx, dy):
    return circdiff(u, (0, -1)) / dy - circdiff(v, (-1, 0)) / dx


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


def parse_cases():
    pat = re.compile(r"DA_DHIT_gaussian_uv_T_([0-9.]+)_ex(\d+)$")
    for path in sorted(DA_ROOT.iterdir()):
        if not path.is_dir():
            continue
        match = pat.match(path.name)
        if match:
            yield float(match.group(1)), int(match.group(2)), path


def load_case(path):
    ic = sio.loadmat(path / "InitialCondition.mat")["ic_psi"]
    da = sio.loadmat(path / "DA_results_uv.mat", squeeze_me=True, struct_as_record=False)
    params = da["Setups"].Parameters
    n = int(params.Nx)
    dx = float(getattr(params, "Lx", 1.0)) / n
    dy = float(getattr(params, "Ly", 1.0)) / n
    psi = ic.reshape((n, n), order="F")
    u_true, v_true = true_uv_from_psi(psi, dx, dy)
    cv = np.asarray(da["updated_control_vector"]).reshape(-1, order="F")
    u_rec = cv[: n * n].reshape((n, n), order="F")
    v_rec = cv[n * n : 2 * n * n].reshape((n, n), order="F")
    omega_true = vorticity_from_uv(u_true, v_true, dx, dy)
    omega_rec = vorticity_from_uv(u_rec, v_rec, dx, dy)
    return omega_true, omega_rec


def compute_da():
    grouped = defaultdict(list)
    for t, _, path in parse_cases():
        omega_true, omega_rec = load_case(path)
        spec_true = energy_spectrum_from_vorticity(omega_true)
        spec_rec = energy_spectrum_from_vorticity(omega_rec)
        grouped[t].append(spectrum_shape_distance(spec_rec, spec_true))
    with open(DA_SUMMARY, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["T", "n", "specShape_mean", "specShape_std"])
        for t in sorted(grouped):
            arr = np.asarray(grouped[t], dtype=float)
            writer.writerow([f"{t:g}", arr.size, arr.mean(), arr.std(ddof=0)])
    return grouped


def load_nn():
    out = {}
    with open(NN_SUMMARY, newline="") as f:
        for row in csv.DictReader(f):
            out[float(row["T"])] = (
                float(row["specShape_mean"]),
                float(row["specShape_std"]),
            )
    return out


def plot(grouped_da):
    nn = load_nn()
    ts = sorted(set(nn) & set(grouped_da))
    nn_mean = np.array([nn[t][0] for t in ts])
    nn_std = np.array([nn[t][1] for t in ts])
    da_mean = np.array([np.mean(grouped_da[t]) for t in ts])
    da_std = np.array([np.std(grouped_da[t], ddof=0) for t in ts])

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(ts, nn_mean, "o-", color="#1f77b4", lw=2.0, ms=4.5, label="Data-driven")
    ax.fill_between(ts, nn_mean - nn_std, nn_mean + nn_std, color="#1f77b4", alpha=0.18, linewidth=0)
    ax.plot(ts, da_mean, "o-", color="#d62728", lw=2.0, ms=4.5, label="Physics-based")
    ax.fill_between(ts, da_mean - da_std, da_mean + da_std, color="#d62728", alpha=0.18, linewidth=0)
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(
        r"$D_{\mathrm{spec}}$"
        "\n"
        r"$=\langle|\log\tilde Z_{\hat{\omega}_0}(k)-\log\tilde Z_{\omega_0}(k)|\rangle_k$",
        labelpad=8,
    )
    ax.set_xlim(0.15, 2.05)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.04)
    print(DA_SUMMARY)
    print(OUT)


def main():
    grouped_da = compute_da()
    plot(grouped_da)


if __name__ == "__main__":
    main()
