#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_reverse_turbulence.py

Paper-ready evaluation for reverse simulation of 2D turbulence.

Outputs (in --out_dir):
1) Sample reconstructions for selected T:
   - Layout: 3 rows × K cols (K = len(Ts_to_show), typically 5)
     Row 1: Input vorticity ω(T)
     Row 2: Reconstructed initial vorticity \hat{ω}(0)
     Row 3: True initial vorticity ω(0)
   - One colorbar per row (rightmost column) + row labels (y-axis).

2) Quantitative errors vs T:
   - Rel. L2(ω0), Rel. gradient error, spectral shape distance, relative total-energy error
   - Plots + summary CSV + LaTeX table snippet.

3) Energy spectrum comparisons (log-log) for selected T.

4) Forward trajectory consistency (physics check):
   - Layout: 2 rows × (Observation + 5 snapshots) + 2 colorbars (aligned columns)
     Col 0: Input observation ω(T)
     Row 1: True ω0 and forward snapshots at t=[0,0.25T,0.5T,0.75T,T]
     Row 2: Reconstructed \hat{ω0} and corresponding forward snapshots
     Last column: colorbar for row 1 (top) and row 2 (bottom), fixed positions.

Assumptions (matching your codebase):
- Model: SimpleUNet in train_u_multiT_gradloss.py
- Solver: FiniteDifferenceNSolver in data_gen.py
- Test files: *.pt containing keys: 'x0','xT','T_end' (optional 'steps','u0','v0')

Example:
python evaluate_reverse_turbulence.py \
  --model_path unet_turbulence_inverse_multiT_gradloss_with_initial.pth \
  --test_dir dataset_test_multiT_with_initial \
  --out_dir eval_outputs_phy \
  --dt 1e-3 --Re 1000 --N 64 \
  --scale_factor 20.0 --T_max 1.0 \
  --sample_T 0.2 0.5 0.8 1.2 1.6 2.0 \
  --forward_check --forward_per_T 2
"""

import os
import glob
import math
import argparse
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from train_u_multiT_gradloss import SimpleUNet
from data_gen import FiniteDifferenceNSolver


# -----------------------------
# Metrics
# -----------------------------
def _safe_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.sum(x * x) + eps)


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    num = _safe_norm(pred - target, eps)
    den = _safe_norm(target, eps)
    return float((num / den).detach().cpu())


def _grad_forward(field: torch.Tensor):
    # forward differences (no padding) as in your training gradient loss
    dx = field[:, :, :, 1:] - field[:, :, :, :-1]
    dy = field[:, :, 1:, :] - field[:, :, :-1, :]
    return dx, dy


def gradient_rel_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    dx_p, dy_p = _grad_forward(pred)
    dx_t, dy_t = _grad_forward(target)
    num = _safe_norm(dx_p - dx_t, eps) + _safe_norm(dy_p - dy_t, eps)
    den = _safe_norm(dx_t, eps) + _safe_norm(dy_t, eps)
    return float((num / den).detach().cpu())


# -----------------------------
# Spectrum utilities (vorticity -> kinetic energy spectrum)
# -----------------------------
def energy_spectrum_from_vorticity(omega_phys: np.ndarray) -> np.ndarray:
    """
    Shell-averaged kinetic energy spectrum E(k) computed from vorticity omega.

    For k != 0:
      |u_hat|^2 + |v_hat|^2 = |omega_hat|^2 / k^2
    so E_k ~ 0.5 * |omega_hat|^2 / k^2 (up to constants).

    omega_phys: (N,N) real array
    returns: spec length k_max = N//2, with spec[0]=0
    """
    N = omega_phys.shape[0]
    omega_hat = np.fft.fft2(omega_phys, norm="ortho")

    kx = np.fft.fftfreq(N) * N
    ky = np.fft.fftfreq(N) * N
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    k2 = kx**2 + ky**2
    k2[0, 0] = np.inf

    energy_density = 0.5 * (np.abs(omega_hat) ** 2) / k2

    k_mag = np.sqrt(kx**2 + ky**2)
    k_bin = np.rint(k_mag).astype(int)

    k_max = N // 2
    spec = np.zeros(k_max, dtype=np.float64)
    counts = np.zeros(k_max, dtype=np.int64)

    for i in range(N):
        for j in range(N):
            k = k_bin[i, j]
            if 1 <= k < k_max:
                spec[k] += energy_density[i, j]
                counts[k] += 1

    counts = np.maximum(counts, 1)
    spec = spec / counts
    return spec


def spectrum_shape_distance(
    spec_pred: np.ndarray,
    spec_true: np.ndarray,
    kmin_ratio: float = 0.4,
    kmax_ratio: float = 0.75,
    eps: float = 1e-12,
) -> float:
    """
    Mean absolute log-difference between normalized spectra over band [k_low, k_high].
    """
    k_max = len(spec_true) - 1
    k_low = max(1, int(math.floor(kmin_ratio * k_max)))
    k_high = max(k_low, int(math.floor(kmax_ratio * k_max)))
    k_high = min(k_high, k_max)

    band = np.arange(k_low, k_high + 1)
    total_pred = np.sum(spec_pred[1:]) + eps
    total_true = np.sum(spec_true[1:]) + eps
    p_pred = spec_pred / total_pred
    p_true = spec_true / total_true

    d = np.mean(np.abs(np.log(p_pred[band] + eps) - np.log(p_true[band] + eps)))
    return float(d)


def total_energy_rel_error(spec_pred: np.ndarray, spec_true: np.ndarray, eps: float = 1e-12) -> float:
    E_pred = np.sum(spec_pred[1:]) + eps
    E_true = np.sum(spec_true[1:]) + eps
    return float(abs(E_pred - E_true) / E_true)


def bandlimited_phase_error(
    omega_pred: np.ndarray,
    omega_true: np.ndarray,
    kmin_ratio: float = 0.4,
    kmax_ratio: float = 0.75,
    eps: float = 1e-12,
) -> float:
    """
    Band-limited phase error: relative L1 error between band-pass filtered pred and true
    (same band as training bandpass loss). Captures spatial organization at intermediate scales.
    """
    N = omega_pred.shape[0]
    kx = np.fft.fftfreq(N) * N
    ky = np.fft.fftfreq(N) * N
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    kk = np.sqrt(kx**2 + ky**2)
    k_max = N // 2
    k_low = kmin_ratio * k_max
    k_high = kmax_ratio * k_max
    mask = ((kk >= k_low) & (kk <= k_high)).astype(np.float64)
    pred_hat = np.fft.fft2(omega_pred, norm="ortho")
    true_hat = np.fft.fft2(omega_true, norm="ortho")
    pred_bp = np.fft.ifft2(pred_hat * mask, norm="ortho").real
    true_bp = np.fft.ifft2(true_hat * mask, norm="ortho").real
    err = np.mean(np.abs(pred_bp - true_bp))
    norm_true = np.mean(np.abs(true_bp)) + eps
    return float(err / norm_true)


# -----------------------------
# Forward simulation utilities
# -----------------------------
def forward_from_omega(solver: FiniteDifferenceNSolver, omega0_phys: np.ndarray, steps: int) -> np.ndarray:
    u, v = solver.converter.velocity_from_vorticity(omega0_phys)
    for _ in range(steps):
        u, v = solver.forward_step(u, v)
    return solver.converter.vorticity_from_velocity(u, v)


def forward_from_uv(
    solver: FiniteDifferenceNSolver,
    u0_phys: np.ndarray,
    v0_phys: np.ndarray,
    steps: int,
) -> np.ndarray:
    u = u0_phys.copy()
    v = v0_phys.copy()
    for _ in range(steps):
        u, v = solver.forward_step(u, v)
    return solver.converter.vorticity_from_velocity(u, v)


# -----------------------------
# Plot helpers (Physics of Fluids–style; five metrics = five colors, match paper)
# -----------------------------
# Colors for the five error metrics (hex); use same in LaTeX \definecolor for text
ERROR_COLORS = [
    "#1f77b4",  # (i) epsilon_L2   – blue
    "#ff7f0e",  # (ii) epsilon_grad – orange
    "#2ca02c",  # (iii) epsilon_shape – green
    "#d62728",  # (iv) epsilon_energy – red
    "#9467bd",  # (v) epsilon_phase – purple
]


def _style_error_axis(ax, Ts_arr, means, stds, ylabel_symbol, color_hex):
    """Apply PoF-style styling; color_hex used for curve and band (lighter)."""
    ax.fill_between(
        Ts_arr, means - stds, means + stds,
        color=color_hex, alpha=0.35, linewidth=0,
    )
    ax.plot(
        Ts_arr, means,
        color=color_hex, marker="o", markersize=4, markeredgewidth=0,
        linewidth=1.8, zorder=5,
    )
    ax.set_xlabel(r"$T$", fontsize=11)
    ax.set_ylabel(ylabel_symbol, fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    # Use nan-aware min/max so plots with NaN (e.g. energyRel) get same axis behavior as others
    low = np.nanmin(means - stds)
    high = np.nanmax(means)
    spread = high - low
    if np.isfinite(low) and np.isfinite(high) and spread >= 0:
        y_min = max(0.0, low - 0.02 * spread)
    else:
        y_min = 0.0
    ax.set_ylim(bottom=y_min)
    ax.set_xlim(0.0, 2.0)


def save_error_vs_T_plot(Ts, stats_dict, out_path, ylabel, title, color_hex=None):
    """ylabel symbol only; color_hex one of ERROR_COLORS or None (uses first)."""
    if color_hex is None:
        color_hex = ERROR_COLORS[0]
    means = np.asarray([stats_dict[T]["mean"] for T in Ts])
    stds = np.asarray([stats_dict[T]["std"] for T in Ts])
    Ts_arr = np.asarray(Ts)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    _style_error_axis(ax, Ts_arr, means, stds, ylabel, color_hex)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_error_vs_T_combined(
    Ts,
    stats_relL2,
    stats_grad,
    stats_specshape,
    stats_energy,
    stats_phase,
    out_path,
):
    """Single figure with 5 subplots; each metric has its own color (match paper)."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.8), sharex=True)
    axes = axes.flatten()
    configs = [
        (stats_relL2, r"$\epsilon_{L^2}$", ERROR_COLORS[0]),
        (stats_grad, r"$\epsilon_{\nabla}$", ERROR_COLORS[1]),
        (stats_specshape, r"$\epsilon_{\mathrm{shape}}$", ERROR_COLORS[2]),
        (stats_energy, r"$\epsilon_{\mathrm{energy}}$", ERROR_COLORS[3]),
        (stats_phase, r"$\epsilon_{\mathrm{phase}}$", ERROR_COLORS[4]),
    ]
    Ts_arr = np.asarray(Ts)
    for idx, (stats_dict, ylabel, color_hex) in enumerate(configs):
        ax = axes[idx]
        means = np.asarray([stats_dict[T]["mean"] for T in Ts])
        stds = np.asarray([stats_dict[T]["std"] for T in Ts])
        _style_error_axis(ax, Ts_arr, means, stds, ylabel, color_hex)
        ax.set_xlabel("")
    axes[5].set_visible(False)
    for idx in [3, 4]:
        axes[idx].set_xlabel(r"$T$", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


from matplotlib.ticker import MultipleLocator

def save_sample_grid(sample_cache, Ts_to_show, out_path, scale_factor=20.0, vlim=None):
    """
    Sample reconstructions figure.
    Layout:
      Row 0: [Empty] [Title: Input (Vert)] [Input ω(T) per T] [Cbar]
      Row 1: [True ω0] [Title: Recon (Vert)] [Recon ω̂(0) per T] [Cbar]
      Row 2: [Empty] [Title: Error (Vert)] [Error per T] [Cbar]
    """
    nT = len(Ts_to_show)
    if nT == 0:
        return

    # Grid: 3 rows.
    # Cols: 
    # 0: True x0 (Reference)
    # 1: Row Title (Vertical Text)
    # 2..nT+1: Data Columns
    # nT+2: Colorbar
    
    fig = plt.figure(figsize=(2.5 * (nT + 1) + 1.5, 6.8))
    gs = GridSpec(
        3, nT + 3, figure=fig,
        width_ratios=[1.0, 0.12] + [1] * nT + [0.06], 
        wspace=0.05,
        hspace=0.12
    )

    # Get representative True x0 from the first T
    first_T = Ts_to_show[0]
    _, _, x0_example = sample_cache[first_T]
    x0_true_np = x0_example[0, 0].numpy() * scale_factor # Physical units

    # Prepare data
    inputs = []
    recons = []
    errors = []
    
    for T in Ts_to_show:
        xT, pred, x0 = sample_cache[T]
        # Convert to physical units
        inputs.append(xT[0, 0].numpy() * scale_factor)
        recons.append(pred[0, 0].numpy() * scale_factor)
        errors.append((pred - x0)[0, 0].numpy() * scale_factor)

    # Vlims
    max_true = np.max(np.abs(x0_true_np))
    max_recon = max([np.max(np.abs(x)) for x in recons])
    vlim_state = max(max_true, max_recon)
    if vlim is not None:
        vlim_state = vlim[1]

    max_input = max([np.max(np.abs(x)) for x in inputs])
    vlim_input = max_input

    max_error = max([np.max(np.abs(x)) for x in errors])
    vlim_error = max_error

    # Helper
    def plot_row(row_idx, data_list, v_limit, cmap="RdBu_r", row_title=None):
        # Plot Row Title (Vertical) in Col 1
        ax_title = fig.add_subplot(gs[row_idx, 1])
        ax_title.axis("off")
        # Align text to the right of the cell to be closer to Data? 
        # Or center? Center of a narrow cell is fine.
        ax_title.text(0.5, 0.5, row_title, ha='center', va='center', rotation=90, fontsize=12, fontweight='bold')

        # Plot Data Columns (Col 2 to nT+1)
        last_im = None
        for i, data in enumerate(data_list):
            ax = fig.add_subplot(gs[row_idx, i + 2])
            im = ax.imshow(data, origin="lower", cmap=cmap, vmin=-v_limit, vmax=v_limit)
            last_im = im
            
            # Column titles only on top row
            if row_idx == 0:
                ax.set_title(rf"$T={Ts_to_show[i]:g}$", fontsize=11)
            
            ax.axis("off")

        # Colorbar (Col nT+2)
        cax = fig.add_subplot(gs[row_idx, nT + 2])
        cb = fig.colorbar(last_im, cax=cax)
        cb.ax.tick_params(labelsize=10)
        
        # FIX: User requested ticks every 1.0
        # Only apply to Input and Recon rows where values are likely > 1.0
        # For Error row, if values are small, 1.0 ticks will hide everything.
        # We check the v_limit.
        if v_limit >= 5.0:  # Increased threshold for 1.0 ticks, or maybe increase step size?
            # If physical units (scale ~20), 1.0 ticks might be too dense (20 ticks).
            # Let's use MaxNLocator or set step to 5.0 or 10.0 depending on v_limit
            # Or just let matplotlib decide but with fewer bins.
            from matplotlib.ticker import MaxNLocator
            cb.locator = MaxNLocator(nbins=5) # Limit to ~5 ticks
            cb.update_ticks()
        elif v_limit >= 0.8: 
            cb.locator = MultipleLocator(1.0)
            cb.update_ticks()
        else:
            # For small errors, keep auto or scientific
            cb.formatter.set_powerlimits((-2, 3))
            cb.update_ticks()
        
        return last_im

    # --- Row 0: Input ---
    # Col 0: Empty (no reference in first row)
    ax_empty0 = fig.add_subplot(gs[0, 0])
    ax_empty0.axis("off")
    plot_row(0, inputs, vlim_input, row_title=r"Input $\omega(T)$")

    # --- Row 1: True ω0 (reference) + Recon ---
    # Col 0: True x0 (Reference), moved to second row
    ax_true = fig.add_subplot(gs[1, 0])
    ax_true.imshow(x0_true_np, origin="lower", cmap="RdBu_r", vmin=-vlim_state, vmax=vlim_state)
    ax_true.set_title(r"True $\omega(0)$", fontsize=11)
    ax_true.axis("off")
    plot_row(1, recons, vlim_state, row_title=r"Recon $\hat{\omega}(0)$")

    # --- Row 2: Error ---
    ax_empty2 = fig.add_subplot(gs[2, 0])
    ax_empty2.axis("off")
    plot_row(2, errors, vlim_error, row_title="Error\n" + r"$\hat{\omega}_0 - \omega_0$")

    # fig.suptitle("Sample reconstructions", y=0.98, fontsize=14)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()




def save_spectrum_plot(k, spec_true, spec_pred, out_path, title):
    plt.figure()
    plt.loglog(k[1:], spec_true[1:], marker="o", linestyle="-", label="True")
    plt.loglog(k[1:], spec_pred[1:], marker="s", linestyle="--", label="Reconstructed")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$E(k)$")
    # plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_spectrum_plot_combined(spectrum_cache, Ts_to_show, T_round, out_path):
    """
    Single figure: one True curve and one Reconstructed curve per T, with different colors per T.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    spec_true = None
    k_axis = None
    for T in Ts_to_show:
        T_key = round(float(T), T_round)
        if T_key not in spectrum_cache:
            continue
        st, sp = spectrum_cache[T_key]
        if spec_true is None:
            spec_true = st
            k_axis = np.arange(len(spec_true))[1:]
        break
    if spec_true is not None:
        ax.loglog(k_axis, spec_true[1:], color="k", linestyle="--", linewidth=2, label="True", zorder=10)
    for T in Ts_to_show:
        T_key = round(float(T), T_round)
        if T_key not in spectrum_cache:
            continue
        _, sp = spectrum_cache[T_key]
        ax.loglog(k_axis, sp[1:], linestyle="-", linewidth=1.5, label=rf"Recon $T={T_key:g}$")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E(k)$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_forward_trajectory_figure(traj_true, traj_pred, times, T, out_path, vlim=None, obs_T=None):
    """
    Paper-ready forward trajectory figure with ALIGNED colorbars.

    Layout: 2 rows × (snapshots + optional observation + colorbar) columns
      Col 0 (optional): Observation ω(T) used as UNet input
      Next 5 cols: snapshots (t = [0, 0.25T, 0.5T, 0.75T, T])
      Last col: colorbar column (top = TRUE row colorbar, bottom = RECON row colorbar)

    This guarantees BOTH colorbars share the same x-position (aligned).
    """
    assert len(traj_true) == len(traj_pred) == len(times)

    has_obs = obs_T is not None
    n_snap = len(traj_true)
    n_img_cols = n_snap + (1 if has_obs else 0)
    fig_w = 13.8 * (n_img_cols + 1) / 6.0
    fig = plt.figure(figsize=(fig_w, 5.0)) # Reduced height from 5.7
    gs = GridSpec(
        2, n_img_cols + 1, figure=fig,
        width_ratios=[1] * n_img_cols + [0.055],
        wspace=0.08, hspace=0.05  # Reduced spacing
    )

    # Row-wise symmetric limits by default (readable); set vlim=(vmin,vmax) for fixed limits
    if vlim is None:
        vmax_true = max(np.max(np.abs(x)) for x in traj_true)
        vmax_pred = max(np.max(np.abs(x)) for x in traj_pred)
        if has_obs:
            vmax_obs = float(np.max(np.abs(obs_T)))
            vmax_true = max(vmax_true, vmax_obs)
            vmax_pred = max(vmax_pred, vmax_obs)
        vmin_true, vmax_true = -vmax_true, vmax_true
        vmin_pred, vmax_pred = -vmax_pred, vmax_pred
    else:
        vmin_true, vmax_true = vlim
        vmin_pred, vmax_pred = vlim

    # ---------------- Row 1: True ----------------
    ims_true = []
    col_offset = 0
    if has_obs:
        ax = fig.add_subplot(gs[0, 0])
        im = ax.imshow(obs_T, origin="lower", cmap="RdBu_r", vmin=vmin_true, vmax=vmax_true)
        ims_true.append(im)
        # Vertical label on the left (closer)
        ax.text(-0.02, 0.5, r"Input $\omega(T)$", transform=ax.transAxes, 
                rotation=90, va='center', ha='right', fontsize=11, fontweight="bold")
        ax.axis("off")
        col_offset = 1

    for j in range(n_snap):
        ax = fig.add_subplot(gs[0, j + col_offset])
        im = ax.imshow(traj_true[j], origin="lower", cmap="RdBu_r", vmin=vmin_true, vmax=vmax_true)
        ims_true.append(im)
        
        if j == 0:
            # Vertical label for t=0 (closer)
            ax.text(-0.02, 0.5, r"True $\omega_0$", transform=ax.transAxes, 
                    rotation=90, va='center', ha='right', fontsize=11, fontweight="bold")
        else:
            # Time title for t>0
            ax.set_title(rf"$t={times[j]:.3f}$", fontsize=11)
            
        ax.axis("off")

    cax_true = fig.add_subplot(gs[0, n_img_cols])
    cb_true = fig.colorbar(ims_true[-1], cax=cax_true)
    cb_true.ax.tick_params(labelsize=9)

    # ---------------- Row 2: Reconstructed ----------------
    ims_pred = []
    # if has_obs:
    #     ax = fig.add_subplot(gs[1, 0])
    #     im = ax.imshow(obs_T, origin="lower", cmap="RdBu_r", vmin=vmin_pred, vmax=vmax_pred)
    #     ims_pred.append(im)
    #     # Vertical label on the left
    #     ax.text(-0.02, 0.5, r"Input $\omega(T)$", transform=ax.transAxes, 
    #             rotation=90, va='center', ha='right', fontsize=11, fontweight="bold")
    #     ax.axis("off")

    for j in range(n_snap):
        # Shift column index by 1 if has_obs (to skip the first column which is empty in row 2)
        col_idx = j + col_offset
        ax = fig.add_subplot(gs[1, col_idx])
        im = ax.imshow(traj_pred[j], origin="lower", cmap="RdBu_r", vmin=vmin_pred, vmax=vmax_pred)
        ims_pred.append(im)
        
        if j == 0:
            # Vertical label for t=0 (closer)
            ax.text(-0.02, 0.5, r"Reconstructed $\hat{\omega}_0$", transform=ax.transAxes, 
                    rotation=90, va='center', ha='right', fontsize=11, fontweight="bold")
        
        # No time titles for Row 2
        ax.axis("off")

    cax_pred = fig.add_subplot(gs[1, n_img_cols])
    cb_pred = fig.colorbar(ims_pred[-1], cax=cax_pred)
    cb_pred.ax.tick_params(labelsize=9)

    # fig.suptitle(rf"Forward trajectory consistency from $t=0$ to $T={T:g}$", y=0.99, fontsize=12)

    # Use rect to reserve space for suptitle (prevents squeezing / shifts)
    # Reduce margins to minimize whitespace
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 1.0], pad=0.5) 
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_forward_trajectory_figure_3row(traj_true, traj_da, traj_dl, times, T, out_path, vlim=None, obs_T=None):
    """
    Same layout as save_forward_trajectory_figure but 3 rows for DA vs DL comparison.

    Row 0: Input ω(T) + True ω0 and its forward evolution (shared reference)
    Row 1: DA reconstructed ω0 and its forward evolution
    Row 2: DL reconstructed ω0 and its forward evolution

    traj_true, traj_da, traj_dl: each list of (N,N) arrays, length = len(times) (e.g. 5).
    """
    assert len(traj_true) == len(traj_da) == len(traj_dl) == len(times)
    has_obs = obs_T is not None
    n_snap = len(traj_true)
    n_img_cols = n_snap + (1 if has_obs else 0)
    fig_w = 13.8 * (n_img_cols + 1) / 6.0
    fig = plt.figure(figsize=(fig_w, 5.0 * 1.5))  # taller for 3 rows
    gs = GridSpec(
        3, n_img_cols + 1, figure=fig,
        width_ratios=[1] * n_img_cols + [0.055],
        wspace=0.08, hspace=0.05
    )
    col_offset = 1 if has_obs else 0

    # Per-row symmetric vlim (same logic as 2-row)
    def row_vmax(arrs, obs=None):
        v = max(np.max(np.abs(x)) for x in arrs)
        if obs is not None:
            v = max(v, float(np.max(np.abs(obs))))
        return v
    if vlim is None:
        vmin_t, vmax_t = -row_vmax(traj_true, obs_T), row_vmax(traj_true, obs_T)
        vmin_da, vmax_da = -row_vmax(traj_da), row_vmax(traj_da)
        vmin_dl, vmax_dl = -row_vmax(traj_dl), row_vmax(traj_dl)
    else:
        vmin_t, vmax_t = vlim
        vmin_da, vmax_da = vlim
        vmin_dl, vmax_dl = vlim

    # ---------------- Row 0: Input + True ----------------
    ims = []
    if has_obs:
        ax = fig.add_subplot(gs[0, 0])
        im = ax.imshow(obs_T, origin="lower", cmap="RdBu_r", vmin=vmin_t, vmax=vmax_t)
        ims.append(im)
        ax.text(-0.02, 0.5, r"$\omega(T)$", transform=ax.transAxes,
                rotation=90, va='center', ha='right', fontsize=11, fontweight="bold")
        ax.axis("off")
    for j in range(n_snap):
        ax = fig.add_subplot(gs[0, j + col_offset])
        im = ax.imshow(traj_true[j], origin="lower", cmap="RdBu_r", vmin=vmin_t, vmax=vmax_t)
        ims.append(im)
        if j == 0:
            ax.text(-0.02, 0.5, r"$\omega_0$", transform=ax.transAxes,
                    rotation=90, va='center', ha='right', fontsize=11, fontweight="bold")
        else:
            ax.set_title(rf"$t={times[j]:.3f}$", fontsize=11)
        ax.axis("off")
    cax = fig.add_subplot(gs[0, n_img_cols])
    cb = plt.colorbar(ims[-1], cax=cax)
    cb.ax.tick_params(labelsize=9)

    # ---------------- Row 1: DA Recon ----------------
    ims_da = []
    for j in range(n_snap):
        ax = fig.add_subplot(gs[1, j + col_offset])
        im = ax.imshow(traj_da[j], origin="lower", cmap="RdBu_r", vmin=vmin_da, vmax=vmax_da)
        ims_da.append(im)
        if j == 0:
            ax.text(-0.02, 0.5, r"$\hat{\omega}_0^{\mathrm{DA}}$", transform=ax.transAxes,
                    rotation=90, va='center', ha='right', fontsize=11, fontweight="bold")
        ax.axis("off")
    cax_da = fig.add_subplot(gs[1, n_img_cols])
    cb_da = plt.colorbar(ims_da[-1], cax=cax_da)
    cb_da.ax.tick_params(labelsize=9)

    # ---------------- Row 2: DL Recon ----------------
    ims_dl = []
    for j in range(n_snap):
        ax = fig.add_subplot(gs[2, j + col_offset])
        im = ax.imshow(traj_dl[j], origin="lower", cmap="RdBu_r", vmin=vmin_dl, vmax=vmax_dl)
        ims_dl.append(im)
        if j == 0:
            ax.text(-0.02, 0.5, r"$\hat{\omega}_0^{\mathrm{NN}}$", transform=ax.transAxes,
                    rotation=90, va='center', ha='right', fontsize=11, fontweight="bold")
        ax.axis("off")
    cax_dl = fig.add_subplot(gs[2, n_img_cols])
    cb_dl = plt.colorbar(ims_dl[-1], cax=cax_dl)
    cb_dl.ax.tick_params(labelsize=9)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 1.0], pad=0.5)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_forward_spectrum_comparison(traj_true_phys, traj_pred_phys, times, T_key, out_path):
    """
    Plot energy spectrum comparison for each time step in the forward trajectory.
    traj_true_phys: list of (N, N) arrays (physical units)
    traj_pred_phys: list of (N, N) arrays (physical units)
    times: list of time values
    """
    n_snaps = len(times)
    # Create a figure with n_snaps subplots in a row
    fig, axes = plt.subplots(1, n_snaps, figsize=(3.5 * n_snaps, 3.5), sharey=True)
    if n_snaps == 1:
        axes = [axes]

    for i, (ax, t_val) in enumerate(zip(axes, times)):
        spec_true = energy_spectrum_from_vorticity(traj_true_phys[i])
        spec_pred = energy_spectrum_from_vorticity(traj_pred_phys[i])
        k = np.arange(len(spec_true))

        ax.loglog(k[1:], spec_true[1:], marker="o", markersize=3, linestyle="-", label="True", alpha=0.7)
        ax.loglog(k[1:], spec_pred[1:], marker="s", markersize=3, linestyle="--", label="Recon", alpha=0.7)
        
        ax.set_title(rf"$t={t_val:.3f}$", fontsize=10)
        ax.set_xlabel(r"$k$")
        if i == 0:
            ax.set_ylabel(r"$E(k)$")
        
        # Only show legend in the first plot to save space
        if i == 0:
            ax.legend(fontsize=8)

    # fig.suptitle(rf"Forward evolution of energy spectrum ($T={T_key:g}$)", y=0.98, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 1.0])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_forward_spectrum_comparison_3way(traj_true, traj_da, traj_dl, times, T_key, out_path):
    """
    One row × n_snaps panels; each panel: True vs DA recon vs DL recon spectrum (same as
    forward_spectrum_evolution but three curves for DA/DL comparison).
    """
    n_snaps = len(times)
    assert len(traj_true) == len(traj_da) == len(traj_dl) == n_snaps
    fig, axes = plt.subplots(1, n_snaps, figsize=(3.5 * n_snaps, 3.5), sharey=True)
    if n_snaps == 1:
        axes = [axes]
    for i, (ax, t_val) in enumerate(zip(axes, times)):
        spec_true = energy_spectrum_from_vorticity(traj_true[i])
        spec_da = energy_spectrum_from_vorticity(traj_da[i])
        spec_dl = energy_spectrum_from_vorticity(traj_dl[i])
        k = np.arange(len(spec_true))
        ax.loglog(k[1:], spec_true[1:], color="k", linestyle="--", linewidth=1.5, label=r"$\omega$")
        ax.loglog(k[1:], spec_da[1:], linestyle="-", linewidth=1.5, label=r"$\hat{\omega}^{\mathrm{DA}}$", alpha=0.8)
        ax.loglog(k[1:], spec_dl[1:], linestyle="-", linewidth=1.5, label=r"$\hat{\omega}^{\mathrm{NN}}$", alpha=0.8)
        ax.set_title(rf"$t={t_val:.3f}$", fontsize=10)
        ax.set_xlabel(r"$k$")
        if i == 0:
            ax.set_ylabel(r"$E(k)$")
            ax.legend(fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 1.0])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate reverse turbulence model (paper-ready outputs).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth model weights.")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory with test .pt files.")
    parser.add_argument("--out_dir", type=str, default="eval_outputs", help="Output directory.")

    parser.add_argument("--N", type=int, default=64, help="Grid resolution N (NxN).")
    parser.add_argument("--dt", type=float, default=1e-3, help="Time step for solver (forward trajectory check).")
    parser.add_argument("--Re", type=float, default=1000.0, help="Reynolds number for solver.")
    parser.add_argument("--L", type=float, default=1.0, help="Domain length (assumed square).")

    parser.add_argument("--scale_factor", type=float, default=20.0,
                        help="Scaling used in training/eval: x_scaled = x_raw / scale_factor.")
    parser.add_argument("--T_max", type=float, default=1.0,
                        help="Time normalization constant for time channel: tau = clip(T/T_max,0,1).")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--sample_T", type=float, nargs="*", default=[0.2, 0.5, 0.8, 1.2, 1.6, 2.0],
                        help="T values to visualize (nearest available).")
    parser.add_argument("--max_files", type=int, default=-1, help="Limit number of test files (-1 = all).")
    parser.add_argument("--T_round", type=int, default=6, help="Round T when grouping (float safety).")

    parser.add_argument("--forward_check", action="store_true",
                        help="Run forward trajectory consistency check (subset).")
    parser.add_argument("--forward_per_T", type=int, default=2, help="Examples per T for forward trajectory figure.")

    parser.add_argument("--kmin_ratio", type=float, default=0.4, help="Spectrum band lower ratio.")
    parser.add_argument("--kmax_ratio", type=float, default=0.75, help="Spectrum band upper ratio.")
    parser.add_argument("--save_obs_gt", action="store_true",
                        help="Save observation (omega(T)) and ground truth (omega(0)) for colleague to reproduce the five errors.")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # load model
    model = SimpleUNet(n_channels=2, n_classes=1).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # list test files
    files = sorted(glob.glob(os.path.join(args.test_dir, "*.pt")))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise RuntimeError(f"No .pt files found in: {args.test_dir}")

    print(f"[Eval] device={device} | files={len(files)}")
    print(f"[Eval] model={args.model_path}")
    print(f"[Eval] out_dir={args.out_dir}")

    # accumulators by T
    errs_relL2 = defaultdict(list)
    errs_grad = defaultdict(list)
    errs_specshape = defaultdict(list)
    errs_energy = defaultdict(list)
    errs_phase = defaultdict(list)

    # representative cache per T: (xT_scaled, pred_scaled, x0_scaled) on CPU
    sample_cache = {}
    spectrum_cache = {}
    forward_pool = defaultdict(list)
    # per-realization spectra for combined plot (one True, preds per T): rid -> {'spec_true': array, 'preds': {T_key: array}}
    by_rid = defaultdict(lambda: {"spec_true": None, "preds": {}})
    # for --save_obs_gt: collect (observation=omega(T), ground_truth=omega(0)) in physical space
    obs_gt_list = []  # list of (obs_phys, gt_phys, T)

    def _realization_id(filepath):
        name = os.path.basename(filepath).replace(".pt", "")
        return name.rsplit("_T", 1)[0] if "_T" in name else name

    for fp in files:
        data = torch.load(fp, map_location="cpu")

        x0_raw = data["x0"].float()  # (1,1,N,N) physical scale (generator-normalized)
        xT_raw = data["xT"].float()
        T = float(data.get("T_end", 0.0))
        steps = int(data.get("steps", round(T / args.dt)))
        T_key = round(T, args.T_round)

        # scale for model
        x0 = x0_raw / args.scale_factor
        xT = xT_raw / args.scale_factor

        # time channel
        t_norm = T / args.T_max if args.T_max > 0 else 0.0
        t_norm = max(0.0, min(1.0, t_norm))
        t_channel = torch.full_like(xT, float(t_norm))

        inp = torch.cat([xT, t_channel], dim=1).to(device)

        with torch.no_grad():
            pred = model(inp)  # scaled

        # all errors in physical space (same scale for fair comparison / alignment with others)
        pred_phys = (pred.detach().cpu() * args.scale_factor).squeeze(0).squeeze(0).numpy()
        true_phys = x0_raw.squeeze(0).squeeze(0).numpy()

        pred_phys_t = torch.from_numpy(pred_phys).float().unsqueeze(0).unsqueeze(0).to(device)
        true_phys_t = torch.from_numpy(true_phys).float().unsqueeze(0).unsqueeze(0).to(device)
        errs_relL2[T_key].append(relative_l2(pred_phys_t, true_phys_t))
        errs_grad[T_key].append(gradient_rel_l2(pred_phys_t, true_phys_t))

        spec_pred = energy_spectrum_from_vorticity(pred_phys)
        spec_true = energy_spectrum_from_vorticity(true_phys)

        errs_specshape[T_key].append(
            spectrum_shape_distance(spec_pred, spec_true, args.kmin_ratio, args.kmax_ratio)
        )
        errs_energy[T_key].append(total_energy_rel_error(spec_pred, spec_true))
        errs_phase[T_key].append(
            bandlimited_phase_error(pred_phys, true_phys, args.kmin_ratio, args.kmax_ratio)
        )

        if args.save_obs_gt:
            obs_phys = xT_raw.squeeze(0).squeeze(0).numpy()
            obs_gt_list.append((obs_phys, true_phys.copy(), T))

        if T_key not in sample_cache:
            sample_cache[T_key] = (xT.clone(), pred.detach().cpu(), x0.clone())
            spectrum_cache[T_key] = (spec_true, spec_pred)

        rid = _realization_id(fp)
        by_rid[rid]["spec_true"] = spec_true  # same x0 for all T of this rid
        by_rid[rid]["preds"][T_key] = spec_pred

        forward_pool[T_key].append((fp, steps))

    Ts = sorted(errs_relL2.keys())

    def summarize(vals):
        arr = np.asarray(vals, dtype=np.float64)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}

    stats_relL2 = {T: summarize(errs_relL2[T]) for T in Ts}
    stats_grad = {T: summarize(errs_grad[T]) for T in Ts}
    stats_specshape = {T: summarize(errs_specshape[T]) for T in Ts}
    stats_energy = {T: summarize(errs_energy[T]) for T in Ts}
    stats_phase = {T: summarize(errs_phase[T]) for T in Ts}

    # -----------------------------
    # Save observation + ground truth for colleague (--save_obs_gt)
    # -----------------------------
    if args.save_obs_gt and obs_gt_list:
        observation = np.stack([x[0] for x in obs_gt_list], axis=0)
        ground_truth = np.stack([x[1] for x in obs_gt_list], axis=0)
        T_values = np.array([x[2] for x in obs_gt_list], dtype=np.float64)
        npz_path = os.path.join(args.out_dir, "eval_observation_groundtruth_for_colleague.npz")
        np.savez_compressed(
            npz_path,
            observation=observation,
            ground_truth=ground_truth,
            T_values=T_values,
            N=np.array(args.N),
        )
        readme_path = os.path.join(args.out_dir, "README_observation_groundtruth.txt")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("eval_observation_groundtruth_for_colleague.npz\n")
            f.write("================================================\n")
            f.write("Physical-space vorticity used to compute the five error metrics.\n\n")
            f.write("Arrays (all numpy):\n")
            f.write("  observation  shape (n_samples, N, N)  = omega(T), input to the inverse problem\n")
            f.write("  ground_truth shape (n_samples, N, N)  = omega(0), true initial vorticity\n")
            f.write("  T_values     shape (n_samples,)      = reverse-time horizon for each sample\n")
            f.write("  N            scalar                  = grid size (NxN)\n\n")
            f.write("Sample order matches the order of test .pt files. Use T_values to group by T.\n")
            f.write("See ERROR_METRICS_DEFINITIONS.md for the exact formulas of the five errors.\n")
        print(f"[Saved] {npz_path} + {readme_path}")

    # -----------------------------
    # Save CSV + LaTeX table
    # -----------------------------
    csv_path = os.path.join(args.out_dir, "summary_by_T.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("T,n,relL2_mean,relL2_std,gradRel_mean,gradRel_std,specShape_mean,specShape_std,energyRel_mean,energyRel_std,phaseRel_mean,phaseRel_std\n")
        for T in Ts:
            f.write(
                f"{T},{stats_relL2[T]['n']},"
                f"{stats_relL2[T]['mean']},{stats_relL2[T]['std']},"
                f"{stats_grad[T]['mean']},{stats_grad[T]['std']},"
                f"{stats_specshape[T]['mean']},{stats_specshape[T]['std']},"
                f"{stats_energy[T]['mean']},{stats_energy[T]['std']},"
                f"{stats_phase[T]['mean']},{stats_phase[T]['std']}\n"
            )
    print(f"[Saved] {csv_path}")

    tex_path = os.path.join(args.out_dir, "table_summary_by_T.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated table (paste into Overleaf)\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Reconstruction errors grouped by reverse-time horizon $T$ (mean $\\pm$ std).}\n")
        f.write("\\label{tab:errors_by_T}\n")
        f.write("\\begin{tabular}{c c c c c}\n")
        f.write("\\hline\n")
        f.write("$T$ & $n$ & Rel. $L^2$ & Grad. Rel. $L^2$ & Spec. shape \\\\\n")
        f.write("\\hline\n")
        for T in Ts:
            f.write(
                f"{T:g} & {stats_relL2[T]['n']} & "
                f"{stats_relL2[T]['mean']:.4f} $\\pm$ {stats_relL2[T]['std']:.4f} & "
                f"{stats_grad[T]['mean']:.4f} $\\pm$ {stats_grad[T]['std']:.4f} & "
                f"{stats_specshape[T]['mean']:.4f} $\\pm$ {stats_specshape[T]['std']:.4f} \\\\\n"
            )
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"[Saved] {tex_path}")

    # -----------------------------
    # Plots: error vs T
    # -----------------------------
    save_error_vs_T_plot(
        Ts, stats_relL2,
        os.path.join(args.out_dir, "error_vs_T_relL2.png"),
        ylabel=r"$\epsilon_{L^2}$",
        title="",
        color_hex=ERROR_COLORS[0],
    )
    save_error_vs_T_plot(
        Ts, stats_grad,
        os.path.join(args.out_dir, "error_vs_T_gradRel.png"),
        ylabel=r"$\epsilon_{\nabla}$",
        title="",
        color_hex=ERROR_COLORS[1],
    )
    save_error_vs_T_plot(
        Ts, stats_specshape,
        os.path.join(args.out_dir, "error_vs_T_specShape.png"),
        ylabel=r"$\epsilon_{\mathrm{shape}}$",
        title="",
        color_hex=ERROR_COLORS[2],
    )
    save_error_vs_T_plot(
        Ts, stats_energy,
        os.path.join(args.out_dir, "error_vs_T_energyRel.png"),
        ylabel=r"$\epsilon_{\mathrm{energy}}$",
        title="",
        color_hex=ERROR_COLORS[3],
    )
    save_error_vs_T_plot(
        Ts, stats_phase,
        os.path.join(args.out_dir, "error_vs_T_phaseRel.png"),
        ylabel=r"$\epsilon_{\mathrm{phase}}$",
        title="",
        color_hex=ERROR_COLORS[4],
    )
    save_error_vs_T_combined(
        Ts, stats_relL2, stats_grad, stats_specshape, stats_energy, stats_phase,
        os.path.join(args.out_dir, "error_vs_T_combined.png"),
    )
    print("[Saved] error-vs-T plots (incl. phase) + error_vs_T_combined.png")

    # -----------------------------
    # Choose Ts to show (nearest available to requested sample_T)
    # -----------------------------
    available_Ts = np.array(sorted(sample_cache.keys()), dtype=np.float64)
    Ts_to_show = []
    for t in args.sample_T:
        if available_Ts.size == 0:
            break
        idx = int(np.argmin(np.abs(available_Ts - float(t))))
        Ts_to_show.append(float(available_Ts[idx]))
    Ts_to_show = list(OrderedDict((t, None) for t in Ts_to_show).keys())

    # -----------------------------
    # Sample reconstructions figure
    # -----------------------------
    if Ts_to_show:
        save_sample_grid(
            sample_cache,
            Ts_to_show,
            os.path.join(args.out_dir, "sample_reconstructions.png"),
            scale_factor=args.scale_factor,
            vlim=None
        )
        print(f"[Saved] sample_reconstructions.png (Ts={Ts_to_show})")

    # -----------------------------
    # Energy spectrum: one combined figure, one realization (same True, pred per T)
    # -----------------------------
    if Ts_to_show:
        Ts_keys_show = [round(float(T), args.T_round) for T in Ts_to_show]
        chosen_rid = None
        for rid, data in by_rid.items():
            if data["spec_true"] is None:
                continue
            if all(Tk in data["preds"] for Tk in Ts_keys_show):
                chosen_rid = rid
                break
        if chosen_rid is not None:
            spectrum_one = {
                Tk: (by_rid[chosen_rid]["spec_true"], by_rid[chosen_rid]["preds"][Tk])
                for Tk in Ts_keys_show
            }
            save_spectrum_plot_combined(
                spectrum_one, Ts_to_show, args.T_round,
                os.path.join(args.out_dir, "energy_spectrum_all_T.png")
            )
            print(f"[Saved] energy_spectrum_all_T.png (one sample: {chosen_rid})")
        elif spectrum_cache:
            save_spectrum_plot_combined(
                spectrum_cache, Ts_to_show, args.T_round,
                os.path.join(args.out_dir, "energy_spectrum_all_T.png")
            )
            print("[Saved] energy_spectrum_all_T.png (no single sample for all T, used per-T cache)")

    # -----------------------------
    # Forward trajectory consistency figures
    # -----------------------------
    if args.forward_check and Ts_to_show:
        solver = FiniteDifferenceNSolver(
            n=args.N,
            dx=args.L / args.N,
            Re=args.Re,
            dt=args.dt,
            Lx=args.L,
            Ly=args.L
        )

        rng = np.random.default_rng(12345)
        fracs = [0.0, 0.25, 0.50, 0.75, 1.0]

        obs_dir = os.path.join(args.out_dir, "observations_forward_check")
        os.makedirs(obs_dir, exist_ok=True)
        obs_index = []

        for T in Ts_to_show:
            T_key = round(float(T), args.T_round)
            pool = forward_pool.get(T_key, [])
            if not pool:
                continue

            idxs = rng.choice(len(pool), size=min(args.forward_per_T, len(pool)), replace=False)

            for ex_i, idx in enumerate(idxs, start=1):
                fp, steps_T = pool[int(idx)]
                data = torch.load(fp, map_location="cpu")

                x0_raw = data["x0"].float()
                xT_raw = data["xT"].float()
                u0_raw = data.get("u0", None)
                v0_raw = data.get("v0", None)
                steps_T = int(data.get("steps", steps_T))

                # predict ω0
                xT_scaled = xT_raw / args.scale_factor
                t_norm = max(0.0, min(1.0, float(T_key) / args.T_max))
                t_channel = torch.full_like(xT_scaled, t_norm)
                inp = torch.cat([xT_scaled, t_channel], dim=1).to(device)

                with torch.no_grad():
                    pred_scaled = model(inp).detach().cpu()


                use_uv0 = (u0_raw is not None) and (v0_raw is not None)
                if use_uv0:
                    u0_phys = u0_raw.squeeze(0).squeeze(0).numpy()
                    v0_phys = v0_raw.squeeze(0).squeeze(0).numpy()
                    omega0_true_phys = solver.converter.vorticity_from_velocity(u0_phys, v0_phys)
                else:
                    omega0_true_phys = x0_raw.squeeze(0).squeeze(0).numpy()
                omega0_pred_phys = (pred_scaled * args.scale_factor).squeeze(0).squeeze(0).numpy()
                obs_phys = xT_raw.squeeze(0).squeeze(0).numpy()
                obs_scaled = xT_scaled.squeeze(0).squeeze(0).numpy()

                step_list = [int(round(f * steps_T)) for f in fracs]
                times = [f * float(T_key) for f in fracs]

                if use_uv0:
                    traj_true_phys = [forward_from_uv(solver, u0_phys, v0_phys, st) for st in step_list]
                else:
                    traj_true_phys = [forward_from_omega(solver, omega0_true_phys, st) for st in step_list]
                traj_pred_phys = [forward_from_omega(solver, omega0_pred_phys, st) for st in step_list]

                # plot in PHYSICAL units (consistent with DA results)
                traj_true = traj_true_phys
                traj_pred = traj_pred_phys
                obs_plot = obs_phys # Use physical observation for plotting

                out_path = os.path.join(args.out_dir, f"forward_trajectory_T_{T_key:g}_ex{ex_i}.png")
                save_forward_trajectory_figure(
                    traj_true=traj_true,
                    traj_pred=traj_pred,
                    times=times,
                    T=T_key,
                    out_path=out_path,
                    vlim=None,
                    obs_T=obs_plot
                )
                # Save trajectory arrays for DA/DL combined figure (plot_da_dl_forward_compare.py)
                npz_path = os.path.join(args.out_dir, f"forward_trajectory_data_T_{T_key:g}_ex{ex_i}.npz")
                np.savez(
                    npz_path,
                    traj_true_phys=np.array(traj_true_phys),
                    traj_pred_phys=np.array(traj_pred_phys),
                    times=np.array(times),
                    obs_phys=obs_phys,
                    T=T_key,
                    ex_i=ex_i,
                )

                # NEW: Save forward spectrum evolution (only for the first example to avoid clutter)
                if ex_i == 1:
                    spec_out_path = os.path.join(args.out_dir, f"forward_spectrum_evolution_T_{T_key:g}.png")
                    save_forward_spectrum_comparison(
                        traj_true_phys=traj_true_phys,
                        traj_pred_phys=traj_pred_phys,
                        times=times,
                        T_key=T_key,
                        out_path=spec_out_path
                    )
                    print(f"[Saved] forward spectrum evolution for T={T_key:g}")

                # Save observation data (including u0, v0 if available)
                obs_path = os.path.join(obs_dir, f"observation_T_{T_key:g}_ex{ex_i}.npz")
                
                save_dict = {
                    "xT_raw": obs_phys,
                    "xT_scaled": obs_scaled,
                    "T": float(T_key),
                    "steps": int(steps_T),
                    "t_norm": float(t_norm),
                }
                
                # Add u0, v0 if they exist
                if use_uv0:
                    save_dict["u0_raw"] = u0_phys
                    save_dict["v0_raw"] = v0_phys
                else:
                    # If not available in .pt, try to reconstruct them from x0 using solver?
                    # Or just save x0.
                    save_dict["x0_raw"] = omega0_true_phys

                np.savez(obs_path, **save_dict)
                obs_index.append((obs_path, T_key, int(steps_T), float(t_norm), fp))

        if obs_index:
            index_path = os.path.join(obs_dir, "observations_index.csv")
            with open(index_path, "w", encoding="utf-8") as f:
                f.write("file,T,steps,t_norm,source_pt\n")
                for obs_path, T_key, steps_T, t_norm, src in obs_index:
                    f.write(f"{obs_path},{T_key},{steps_T},{t_norm},{src}\n")
            print(f"[Saved] observation data for forward-check examples: {obs_dir}")

        print("[Saved] forward trajectory figures (paper layout)")

    print("\nDone.")
    print(f"All outputs are in: {args.out_dir}")


if __name__ == "__main__":
    main()
