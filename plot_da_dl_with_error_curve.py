#!/usr/bin/env python3
"""
Regenerate compare_forward_T_*.png with an added error curve:
- Bottom row: ||omega_hat - omega_true||_L2 vs t for DA (orange) and DL (blue)
- Uses 5-snapshot data already in the trajectory files
- All other rows identical to the original 3-row layout
"""
import os, re, glob, argparse
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


plt.rcParams.update({
    "axes.grid": False,
    "legend.frameon": False,
    "font.size": 11,
})


def find_da_folders(da_data_dir):
    folders = sorted(glob.glob(os.path.join(da_data_dir, "DA_DHIT_*")))
    out = []
    for folder in folders:
        name = os.path.basename(folder)
        try:
            parts = name.split("_")
            t_idx = parts.index("T")
            T_str = parts[t_idx + 1]
            ex_str = parts[t_idx + 2]
            ex_num = int(re.sub(r"^ex", "", ex_str, flags=re.I))
        except (ValueError, IndexError):
            continue
        out.append((folder, T_str, ex_num))
    return out


def load_da_trajectory(folder_path):
    """Load DA data, return 5-snapshot subsampled trajectories AND dense data for error curves."""
    mats = glob.glob(os.path.join(folder_path, "observation_*.mat"))
    if not mats:
        return None
    d = sio.loadmat(mats[0])
    omega_true_dense = d["omega_true"]   # (64, 64, Nt)
    omega_re_dense = d["omega_re"]
    T_final = float(d["T_final"].flat[0])
    time_vec = d["time_vec"].flatten()

    # Sign flip (same as original)
    corr = np.corrcoef(omega_re_dense[:, :, 0].flatten(),
                       omega_true_dense[:, :, 0].flatten())[0, 1]
    if corr < 0:
        omega_re_dense = -omega_re_dense
    else:
        omega_re_dense = -omega_re_dense
        omega_true_dense = -omega_true_dense

    # 5 snapshot indices
    ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    target_t = [r * T_final for r in ratios]
    idxs = [int(np.abs(time_vec - t).argmin()) for t in target_t]
    times_5 = [float(time_vec[i]) for i in idxs]
    traj_true = [omega_true_dense[:, :, i] for i in idxs]
    traj_da = [omega_re_dense[:, :, i] for i in idxs]
    obs_T = traj_true[-1]

    # Dense DA error curve
    Nt = omega_true_dense.shape[2]
    da_err_dense = np.zeros(Nt)
    for i in range(Nt):
        diff = omega_re_dense[:, :, i] - omega_true_dense[:, :, i]
        da_err_dense[i] = np.sqrt(np.mean(diff ** 2))

    return {
        "traj_true_5": traj_true,
        "traj_da_5": traj_da,
        "times_5": times_5,
        "T_final": T_final,
        "obs_T": obs_T,
        "time_dense": time_vec,
        "da_err_dense": da_err_dense,
        "omega_true_dense": omega_true_dense,
        "omega_re_dense": omega_re_dense,
    }


def load_dl_trajectory(dl_dir, T_str, ex_num):
    path = os.path.join(dl_dir, f"forward_trajectory_data_T_{T_str}_ex{ex_num}.npz")
    if not os.path.isfile(path):
        for p in glob.glob(os.path.join(dl_dir, "forward_trajectory_data_T_*.npz")):
            base = os.path.basename(p)
            if re.match(rf"forward_trajectory_data_T_[0-9.]+_ex{ex_num}\.npz", base):
                t_in_name = re.search(r"T_([0-9.]+)_", base)
                if t_in_name and abs(float(t_in_name.group(1)) - float(T_str)) < 1e-6:
                    path = p
                    break
    if not os.path.isfile(path):
        return None
    z = np.load(path, allow_pickle=True)
    n = len(z["times"])
    return {
        "traj_true": [z["traj_true_phys"][i] for i in range(n)],
        "traj_dl": [z["traj_pred_phys"][i] for i in range(n)],
        "times": list(z["times"]),
        "obs": z["obs_phys"],
    }


def save_figure_with_error_curve(da_data, dl_data, T_final, out_path):
    traj_true = da_data["traj_true_5"]
    traj_da = da_data["traj_da_5"]
    traj_dl = dl_data["traj_dl"]
    times = da_data["times_5"]
    obs_T = da_data["obs_T"]
    n_snap = len(traj_true)
    n_img_cols = n_snap + 1  # input + 5 snaps

    # Compute error curves (sparse: use 5 snapshots for DL, dense for DA)
    dl_err_sparse = np.array([
        np.sqrt(np.mean((traj_dl[i] - traj_true[i]) ** 2))
        for i in range(n_snap)
    ])
    da_err_sparse = np.array([
        np.sqrt(np.mean((traj_da[i] - traj_true[i]) ** 2))
        for i in range(n_snap)
    ])

    # Figure: 4 rows. Image rows 0-2 + error-curve row 3.
    # Per-cell compact colorbar style: each cell carries a short horizontal
    # colorbar under its image with ±vmax text on the sides (no extra cbar col).
    # A narrow gutter column sits between the ω_T input (col 0) and the
    # snapshot columns so the ω_0 row-label has room without overlapping ω_T.
    gutter_w = 0.18
    n_cols = n_img_cols + 1  # input + gutter + n_snap snapshots
    width_ratios = [1.0, gutter_w] + [1.0] * n_snap
    fig_w = 11.0 * (n_img_cols + gutter_w) / 6.0
    fig = plt.figure(figsize=(fig_w, 5.0 * 1.55))
    gs = GridSpec(
        4, n_cols, figure=fig,
        width_ratios=width_ratios,
        height_ratios=[1, 1, 1, 1.1],
        wspace=0.04, hspace=0.04,
    )

    def _fmt(v):
        av = abs(v)
        if av >= 10:
            return f"{v:.0f}"
        if av >= 1:
            return f"{v:.1f}"
        return f"{v:.2f}"

    def plot_cell(r, c, arr, label_left=None, title=None):
        vlim = float(np.percentile(np.abs(arr), 95)) or 1.0
        inner = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[r, c],
            height_ratios=[1.0, 0.05], hspace=0.04,
        )
        ax = fig.add_subplot(inner[0])
        ax.imshow(arr, origin="lower", cmap="RdBu_r",
                  vmin=-vlim, vmax=vlim, interpolation="nearest")
        if title is not None:
            ax.set_title(title, fontsize=11)
        if label_left is not None:
            ax.text(-0.03, 0.5, label_left, transform=ax.transAxes,
                    rotation=90, va='center', ha='right',
                    fontsize=11, fontweight="bold")
        ax.axis("off")
        footer = fig.add_subplot(inner[1])
        footer.axis("off")
        footer.set_xlim(0, 1); footer.set_ylim(0, 1)
        bar_l, bar_r = 0.22, 0.78
        cax = footer.inset_axes([bar_l, 0.05, bar_r - bar_l, 0.90])
        cb = fig.colorbar(ax.images[0], cax=cax,
                          orientation="horizontal", ticks=[])
        cb.outline.set_linewidth(0.3); cb.outline.set_edgecolor("black")
        footer.text(bar_l - 0.02, 0.5, _fmt(-vlim),
                    ha="right", va="center", fontsize=6)
        footer.text(bar_r + 0.02, 0.5, _fmt(vlim),
                    ha="left", va="center", fontsize=6)

    # Row 0: ω_T + true trajectory snapshots (snapshots start at col 2 after gutter)
    plot_cell(0, 0, obs_T, label_left=r"$\omega_T$")
    for j in range(n_snap):
        plot_cell(
            0, j + 2, traj_true[j],
            label_left=r"$\omega_0$" if j == 0 else None,
            title=None if j == 0 else rf"$t={times[j]:.3f}$",
        )

    # Row 1: DA reconstruction
    for j in range(n_snap):
        plot_cell(
            1, j + 2, traj_da[j],
            label_left=r"$\hat{\omega}_0^{\mathrm{DA}}$" if j == 0 else None,
        )

    # Row 2: DL reconstruction
    for j in range(n_snap):
        plot_cell(
            2, j + 2, traj_dl[j],
            label_left=r"$\hat{\omega}_0^{\mathrm{NN}}$" if j == 0 else None,
        )

    # Row 3: error curve — span the snapshot columns
    ax_err = fig.add_subplot(gs[3, 2:n_cols])
    ax_err.set_box_aspect(None)
    # Use DA dense curve
    ax_err.plot(da_data["time_dense"], da_data["da_err_dense"],
                '-', color='#ff7f0e', lw=2, label=r"$\hat{\omega}^{\mathrm{DA}}$")
    # DL only has 5 points — plot as markers connected by lines
    ax_err.plot(times, dl_err_sparse, 'o-', color='#1f77b4', ms=7, lw=2, label=r"$\hat{\omega}^{\mathrm{NN}}$")
    # Also scatter DA at the 5 snapshot times for reference
    ax_err.plot(times, da_err_sparse, 's', color='#ff7f0e', ms=7,
                markerfacecolor='none', markeredgewidth=2)

    ax_err.set_xlabel(r"$t$", fontsize=12)
    ax_err.set_ylabel(r"$\|\hat{\omega}(t)-\omega(t)\|_{L^2}$", fontsize=11)
    ax_err.legend(fontsize=10, loc='best', frameon=False)
    ax_err.set_xlim(0, T_final)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 1.0], pad=0.5)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_combined_png(image_paths, out_path):
    imgs = [mpimg.imread(p) for p in image_paths]
    widths = [im.shape[1] for im in imgs]
    max_w = max(widths)
    padded = []
    for im in imgs:
        if im.shape[1] == max_w:
            padded.append(im)
            continue
        pad_w = max_w - im.shape[1]
        left = pad_w // 2
        right = pad_w - left
        pad_spec = ((0, 0), (left, right), (0, 0))
        padded.append(np.pad(im, pad_spec, mode="constant", constant_values=1))
    combined = np.concatenate(padded, axis=0)
    fig_w = 12.0
    fig_h = fig_w * combined.shape[0] / combined.shape[1]
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(combined)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_compact_forward_summary(records, out_path):
    selected = [(0.5, 1), (1.2, 1), (2.0, 1)]
    snap_idx = [0, 2, 4]
    n_blocks = len(selected)
    fig = plt.figure(figsize=(12.0, 8.2))
    gs = GridSpec(
        3 * n_blocks,
        6,
        figure=fig,
        width_ratios=[0.18, 1, 1, 1, 0.045, 1.25],
        height_ratios=[1] * (3 * n_blocks),
        wspace=0.08,
        hspace=0.08,
    )

    for b, key in enumerate(selected):
        da_data, dl_data = records[key]
        T_final = da_data["T_final"]
        times = da_data["times_5"]
        rows = [
            (r"$\omega$", da_data["traj_true_5"]),
            (r"$\hat{\omega}^{\mathrm{DA}}$", da_data["traj_da_5"]),
            (r"$\hat{\omega}^{\mathrm{NN}}$", dl_data["traj_dl"]),
        ]
        vlims = [
            max(np.max(np.abs(x)) for x in rows[0][1]),
            max(np.max(np.abs(x)) for x in rows[1][1]),
            max(np.max(np.abs(x)) for x in rows[2][1]),
        ]
        true_sel = [rows[0][1][i] for i in snap_idx]
        da_sel = [rows[1][1][i] for i in snap_idx]
        nn_sel = [rows[2][1][i] for i in snap_idx]
        row_sel = [true_sel, da_sel, nn_sel]

        for r, (label, _) in enumerate(rows):
            rr = 3 * b + r
            ax_label = fig.add_subplot(gs[rr, 0])
            ax_label.axis("off")
            ax_label.text(
                0.5, 0.5, label, rotation=90, ha="center", va="center",
                fontsize=9, fontweight="bold",
            )
            last_im = None
            for j, idx in enumerate(snap_idx):
                ax = fig.add_subplot(gs[rr, 1 + j])
                last_im = ax.imshow(
                    row_sel[r][j],
                    origin="lower",
                    cmap="jet",
                    vmin=-vlims[r],
                    vmax=vlims[r],
                    interpolation="nearest",
                )
                if r == 0:
                    ax.set_title(rf"$t={times[idx]:.2f}$", fontsize=8, pad=2)
                ax.axis("off")
            cax = fig.add_subplot(gs[rr, 4])
            cb = fig.colorbar(last_im, cax=cax)
            cb.ax.tick_params(labelsize=5, pad=1)

        ax_err = fig.add_subplot(gs[3 * b:3 * b + 3, 5])
        dl_err_sparse = np.array([
            np.sqrt(np.mean((dl_data["traj_dl"][i] - da_data["traj_true_5"][i]) ** 2))
            for i in range(len(da_data["traj_true_5"]))
        ])
        da_err_sparse = np.array([
            np.sqrt(np.mean((da_data["traj_da_5"][i] - da_data["traj_true_5"][i]) ** 2))
            for i in range(len(da_data["traj_true_5"]))
        ])
        ax_err.plot(
            da_data["time_dense"], da_data["da_err_dense"],
            "-", color="#ff7f0e", lw=1.5, label=r"$\mathrm{DA}$",
        )
        ax_err.plot(
            times, dl_err_sparse,
            "o-", color="#1f77b4", ms=4, lw=1.5, label=r"$\mathrm{NN}$",
        )
        ax_err.plot(
            times, da_err_sparse,
            "s", color="#ff7f0e", ms=4, markerfacecolor="none", markeredgewidth=1.2,
        )
        ax_err.set_xlim(0, T_final)
        ax_err.set_title(rf"$T={T_final:g}$", fontsize=9, pad=2)
        ax_err.set_xlabel(r"$t$", fontsize=8)
        ax_err.set_ylabel(r"$\|\Delta\omega\|_2$", fontsize=8)
        ax_err.tick_params(labelsize=7)
        if b == 0:
            ax_err.legend(fontsize=7, frameon=False, loc="best")

    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_terminal_forward_summary(records, out_path):
    selected = [(0.5, 1), (1.2, 1), (2.0, 1)]
    fig = plt.figure(figsize=(7.1, 4.55))
    gs = GridSpec(
        4,
        4,
        figure=fig,
        width_ratios=[1, 1, 1, 0.035],
        height_ratios=[1, 1, 1, 0.48],
        wspace=0.12,
        hspace=0.035,
    )
    row_labels = [
        r"$\omega_T$",
        r"$\omega_T^{\mathrm{DA}}$",
        r"$\omega_T^{\mathrm{NN}}$",
    ]
    all_fields = []
    for key in selected:
        da_data, dl_data = records[key]
        all_fields.extend([
            da_data["traj_true_5"][-1],
            da_data["traj_da_5"][-1],
            dl_data["traj_dl"][-1],
        ])
    vlim = max(np.max(np.abs(x)) for x in all_fields)

    last_im = None
    for c, key in enumerate(selected):
        da_data, dl_data = records[key]
        T_final = da_data["T_final"]
        fields = [
            da_data["traj_true_5"][-1],
            da_data["traj_da_5"][-1],
            dl_data["traj_dl"][-1],
        ]
        for r, arr in enumerate(fields):
            ax = fig.add_subplot(gs[r, c])
            last_im = ax.imshow(
                arr,
                origin="lower",
                cmap="jet",
                vmin=-vlim,
                vmax=vlim,
                interpolation="nearest",
            )
            if r == 0:
                ax.set_title(rf"$T={T_final:g}$", fontsize=8, pad=1)
            if c == 0:
                ax.text(
                    -0.045,
                    0.5,
                    row_labels[r],
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold",
                )
            ax.axis("off")

        ax_err = fig.add_subplot(gs[3, c])
        times = da_data["times_5"]
        dl_err_sparse = np.array([
            np.sqrt(np.mean((dl_data["traj_dl"][i] - da_data["traj_true_5"][i]) ** 2))
            for i in range(len(da_data["traj_true_5"]))
        ])
        da_err_sparse = np.array([
            np.sqrt(np.mean((da_data["traj_da_5"][i] - da_data["traj_true_5"][i]) ** 2))
            for i in range(len(da_data["traj_true_5"]))
        ])
        ax_err.plot(
            da_data["time_dense"],
            da_data["da_err_dense"],
            "-",
            color="#ff7f0e",
            lw=1.4,
            label=r"$\mathrm{DA}$",
        )
        ax_err.plot(
            times,
            dl_err_sparse,
            "o-",
            color="#1f77b4",
            ms=3.5,
            lw=1.4,
            label=r"$\mathrm{NN}$",
        )
        ax_err.plot(
            times,
            da_err_sparse,
            "s",
            color="#ff7f0e",
            ms=3.5,
            markerfacecolor="none",
            markeredgewidth=1.0,
        )
        ax_err.set_xlim(0, T_final)
        ax_err.set_xticks([0, T_final / 2, T_final])
        ax_err.set_xticklabels(["0", rf"${T_final/2:g}$", rf"${T_final:g}$"])
        ax_err.set_xlabel(r"$t$", fontsize=7, labelpad=0)
        if c == 0:
            ax_err.set_ylabel(r"$\|\Delta\omega\|_2$", fontsize=7, labelpad=0)
        else:
            ax_err.tick_params(labelleft=False)
        ax_err.tick_params(labelsize=6.5, pad=1)
        ax_err.spines["top"].set_visible(False)
        ax_err.spines["right"].set_visible(False)

    cax = fig.add_subplot(gs[:3, 3])
    cb = fig.colorbar(last_im, cax=cax)
    cb.ax.tick_params(labelsize=6.5, pad=1)

    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--da_data_dir", type=str, required=True)
    parser.add_argument("--dl_dir", type=str, default="eval_outputs_phy")
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    da_list = find_da_folders(args.da_data_dir)
    n_ok = 0
    saved = {}
    records = {}
    for folder_path, T_str, ex_num in da_list:
        da_data = load_da_trajectory(folder_path)
        if da_data is None: continue
        dl_data = load_dl_trajectory(args.dl_dir, T_str, ex_num)
        if dl_data is None:
            print(f"  Skip T={T_str} ex{ex_num}: no DL data")
            continue
        if len(da_data["traj_true_5"]) != 5 or len(dl_data["traj_dl"]) != 5:
            continue

        out_name = f"compare_forward_T_{T_str}_ex{ex_num}.png"
        out_path = os.path.join(args.out_dir, out_name)
        save_figure_with_error_curve(da_data, dl_data, da_data["T_final"], out_path)
        print(f"  {out_name}")
        saved[(float(T_str), ex_num)] = out_path
        records[(float(T_str), ex_num)] = (da_data, dl_data)
        n_ok += 1
    selected = []
    for T in [0.5, 1.2, 2.0]:
        key = (T, 1)
        if key in saved:
            selected.append(saved[key])
    if len(selected) == 3:
        out_path = os.path.join(args.out_dir, "compare_forward_T_0.5_1.2_2_ex1_combined.png")
        save_combined_png(selected, out_path)
        print(f"  {os.path.basename(out_path)}")
    compact_keys = [(0.5, 1), (1.2, 1), (2.0, 1)]
    if all(k in records for k in compact_keys):
        out_path = os.path.join(args.out_dir, "compare_forward_T_0.5_1.2_2_ex1_compact.png")
        save_compact_forward_summary(records, out_path)
        print(f"  {os.path.basename(out_path)}")
        out_path = os.path.join(args.out_dir, "compare_forward_T_0.5_1.2_2_ex1_terminal_summary.png")
        save_terminal_forward_summary(records, out_path)
        print(f"  {os.path.basename(out_path)}")
    print(f"\nDone. {n_ok} figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
