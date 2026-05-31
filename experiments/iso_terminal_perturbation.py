"""
Iso-terminal perturbation experiment.
=====================================
Direct, visual test of the claim:

    "DA / physics-based inversion has a wide solution space — many distinct
     omega_0 candidates produce indistinguishable omega_T.  The NN's
     effective solution space is much narrower: it consistently picks the
     'real-looking' one even when adversarial alternatives are constructed
     to defeat the forward residual."

Construction.  For each test sample we:
  1. compute J_F^K, take its smallest-singular-vector v_min (the input-space
     direction that the forward map is least sensitive to);
  2. decode v_min into a real vorticity-field perturbation delta_omega_min,
     normalised to unit L2;
  3. for a sweep of epsilon in {0, 0.3, 1.0, 2.0, 4.0}, construct
        omega_tilde_0(eps)   = omega_0 + eps * ||omega_0|| * delta_omega_min
        omega_tilde_T(eps)   = curl_FD(advance(u0 + eps*||omega_0||*du_min,
                                                v0 + eps*||omega_0||*dv_min))
        omega_hat_0_NN(eps)  = G_theta(omega_tilde_T(eps))
  4. record four relative norms:
        drift_0          = ||omega_tilde_0 - omega_0||      / ||omega_0||  (= eps)
        residual_T       = ||omega_tilde_T - omega_T||      / ||omega_T||
        err_NN_to_true   = ||omega_hat_0_NN - omega_0||     / ||omega_0||
        err_NN_to_fake   = ||omega_hat_0_NN - omega_tilde_0||/||omega_0||

If the claim is true:
   * residual_T stays tiny across all eps   — DA-cannot-tell evidence
   * drift_0 grows linearly with eps        — but omega_0 differs visibly
   * err_NN_to_true stays small             — NN picks the real one
   * err_NN_to_fake tracks drift_0          — NN refuses the adversarial one

This script also runs the untrained NN as a control (separate output dir).

Outputs.  Per run directory:
  * visual_grid.png      5-column x 3-row visual: rows = (omega_tilde_0,
                         omega_tilde_T, NN(omega_tilde_T)); cols = epsilons.
  * epsilon_sweep.png    4-line plot of the four relative norms vs eps,
                         pooled mean +/- std across samples.
  * trajectories.csv     per (sample, eps) row of all four norms.
  * summary.json         pooled means/stds.
"""
import os, sys, glob, json, argparse, time, csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

sys.path.insert(0, '/workspace/ge')
from data_gen import FiniteDifferenceNSolver
from train_u_multiT_gradloss import SimpleUNet

# Reuse the helpers from the previous experiment.
from finite_time_singular_direction_filtering import (
    N, LX, LY, RE, DT, SCALE, T_MAX_NN, EPS_TINY,
    field_to_fourier_subspace, fourier_subspace_to_field,
    vorticity_perturbation_to_velocity, make_solver, rollout_uv,
    rollout_omega_from_uv, load_nn, neural_inverse, build_JF_K,
)


# ============================================================================
# Core construction
# ============================================================================
def find_v_min(J_F):
    """Return the smallest-singular-vector v_min (in input space).

    numpy SVD convention: J_F = U @ diag(S) @ Vt, columns of V are inputs.
    The smallest singular value is S[-1], and the corresponding input
    direction is Vt[-1, :].T.  Returned as a unit vector in R^{2|K|}.
    """
    U, S, Vt = np.linalg.svd(J_F, full_matrices=False)
    return Vt[-1, :].copy(), float(S[-1]), float(S[0])


def normalise_delta_omega(delta_omega):
    """Scale a field to unit L2 norm.  Returns (normed, original_norm)."""
    n = np.linalg.norm(delta_omega)
    return delta_omega / (n + EPS_TINY), float(n)


def iso_terminal_run(sample_file, model, modes, solver, T, n_steps, eps_list,
                     fd_eps=1e-3):
    """
    Single-sample procedure: build v_min, sweep epsilons, return per-eps
    results and the field tensors needed for the visual grid.
    """
    d = torch.load(sample_file, map_location='cpu', weights_only=False)
    u0 = d['u0'].squeeze().numpy().astype(np.float64)
    v0 = d['v0'].squeeze().numpy().astype(np.float64)
    omega_0 = solver.converter.vorticity_from_velocity(u0, v0)
    omega_T = rollout_omega_from_uv(solver, u0, v0, n_steps)

    # 1. Build J_F^K and get v_min.
    J_F = build_JF_K(solver, u0, v0, modes, n_steps, fd_eps)
    v_min, sigma_min, sigma_max = find_v_min(J_F)

    # 2. Decode v_min to a vorticity field, then normalise so ||delta_omega||=1.
    delta_omega_unit = fourier_subspace_to_field(v_min, modes)
    delta_omega_unit, _ = normalise_delta_omega(delta_omega_unit)

    # 3. Get the matching velocity-space perturbation (also unit-scaled).
    du_unit, dv_unit = vorticity_perturbation_to_velocity(solver, delta_omega_unit)

    norm_omega_0 = float(np.linalg.norm(omega_0))
    norm_omega_T = float(np.linalg.norm(omega_T))

    rows = []
    fields = []   # list of dicts (eps, omega_tilde_0, omega_tilde_T, omega_hat_NN)

    for eps in eps_list:
        alpha = eps * norm_omega_0          # ||alpha delta_omega|| = eps ||omega_0||
        omega_tilde_0 = omega_0 + alpha * delta_omega_unit
        # forward in velocity coordinates
        u_pert = u0 + alpha * du_unit
        v_pert = v0 + alpha * dv_unit
        omega_tilde_T = rollout_omega_from_uv(solver, u_pert, v_pert, n_steps)
        # NN inversion of the perturbed terminal
        omega_hat_NN = neural_inverse(model, omega_tilde_T, T)

        rows.append({
            'eps':            float(eps),
            'sigma_min':      sigma_min,
            'sigma_max':      sigma_max,
            'norm_omega_0':   norm_omega_0,
            'norm_omega_T':   norm_omega_T,
            'drift_0':        float(np.linalg.norm(omega_tilde_0 - omega_0) / norm_omega_0),
            'residual_T':     float(np.linalg.norm(omega_tilde_T - omega_T) / norm_omega_T),
            'err_NN_to_true': float(np.linalg.norm(omega_hat_NN - omega_0)        / norm_omega_0),
            'err_NN_to_fake': float(np.linalg.norm(omega_hat_NN - omega_tilde_0)  / norm_omega_0),
        })
        fields.append({
            'eps':           float(eps),
            'omega_tilde_0': omega_tilde_0,
            'omega_tilde_T': omega_tilde_T,
            'omega_hat_NN':  omega_hat_NN,
        })
    return rows, fields, omega_0, omega_T


# ============================================================================
# Plotting
# ============================================================================
def plot_visual_grid(out_path, fields, omega_0, omega_T, sample_name):
    """3 rows x N columns:
       row 0 -> tilde omega_0  (adversarial initial state)
       row 1 -> tilde omega_T  (its terminal image)
       row 2 -> G_theta(tilde omega_T)  (NN inversion)
       columns are epsilon values.

    Per-cell compact colorbar style: each cell has its own thin horizontal
    colorbar under the image with ±vmax labels on the sides.
    """
    n_eps = len(fields)
    nC = n_eps
    nR = 3
    row_labels = [r'$\tilde\omega_0$', r'$\tilde\omega_T$', r'$G_\theta(\tilde\omega_T)$']

    width_ratios = [0.28] + [1.0] * nC
    fig = plt.figure(figsize=(2.05 * nC + 0.55, 2.0 * nR + 0.25))
    outer = GridSpec(nR, 1 + nC, figure=fig,
                     width_ratios=width_ratios,
                     wspace=0.04, hspace=0.03)

    def _fmt(v):
        av = abs(v)
        if av >= 10:
            return f"{v:.0f}"
        if av >= 1:
            return f"{v:.1f}"
        return f"{v:.2f}"

    # Row labels on the left
    for r in range(nR):
        ax_label = fig.add_subplot(outer[r, 0])
        ax_label.axis("off")
        ax_label.text(0.5, 0.5, row_labels[r],
                      rotation=0, ha="center", va="center",
                      fontsize=13)

    for j, f in enumerate(fields):
        eps = f['eps']
        for r in range(nR):
            if r == 0:
                img = f['omega_tilde_0']
            elif r == 1:
                img = f['omega_tilde_T']
            else:
                img = f['omega_hat_NN']
            vlim = float(np.percentile(np.abs(img), 98)) or 1.0
            inner = GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[r, 1 + j],
                height_ratios=[1.0, 0.05], hspace=0.04,
            )
            ax = fig.add_subplot(inner[0])
            ax.imshow(img, cmap='RdBu_r',
                      vmin=-vlim, vmax=vlim, origin='lower',
                      interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            ax.axis("off")
            if r == 0:
                ax.set_title(rf"$\varepsilon={eps:.2g}$", fontsize=11, pad=3)
            footer = fig.add_subplot(inner[1])
            footer.axis("off")
            footer.set_xlim(0, 1); footer.set_ylim(0, 1)
            bar_l, bar_r = 0.22, 0.78
            cax = footer.inset_axes([bar_l, 0.05, bar_r - bar_l, 0.90])
            cb = fig.colorbar(ax.images[0], cax=cax,
                              orientation="horizontal", ticks=[])
            cb.outline.set_linewidth(0.3)
            cb.outline.set_edgecolor("black")
            footer.text(bar_l - 0.02, 0.5, _fmt(-vlim),
                        ha="right", va="center", fontsize=6)
            footer.text(bar_r + 0.02, 0.5, _fmt(vlim),
                        ha="left", va="center", fontsize=6)

    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def shell_spectrum(field, N=N):
    """Shell-averaged energy spectrum E(k) for k = 1 .. N/2."""
    F = np.fft.fft2(field) / (N*N)
    P = np.abs(F)**2
    kx = np.fft.fftfreq(N, d=1.0/N)
    ky = np.fft.fftfreq(N, d=1.0/N)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    KMAG = np.sqrt(KX**2 + KY**2)
    kbins = np.arange(0, N//2 + 1)
    Ek = np.zeros_like(kbins, dtype=float)
    for kb in kbins:
        mask = (KMAG >= kb - 0.5) & (KMAG < kb + 0.5)
        if mask.sum() == 0: continue
        Ek[kb] = P[mask].sum()
    return kbins, Ek


def _compute_spectral_data(omega_0, fields, omega_0_pool, eps_focus):
    """Shared computation for the two spectral diagnostic plots.

    Returns both the per-shell mean and the 95th percentile of Z(k) across
    the test ensemble.  The 95th percentile is the high-probability empirical
    envelope B_q(k) used in the spectral-admissibility discussion.
    """
    spec_stack = []
    kbins = None
    for om0 in omega_0_pool:
        kbins, Ek = shell_spectrum(om0)
        spec_stack.append(Ek)
    spec = np.array(spec_stack) if spec_stack else np.zeros((1, N//2 + 1))
    Ek_pool = spec.mean(axis=0)
    Ek_p95  = np.percentile(spec, 95, axis=0)
    _, Ek_true = shell_spectrum(omega_0)
    if eps_focus is None:
        eps_focus = fields[-2]['eps'] if len(fields) >= 2 else fields[-1]['eps']
    field_focus = next(f for f in fields if abs(f['eps'] - eps_focus) < 1e-9)
    # k_star: the wavenumber the adversarial perturbation injects into.
    delta_v_min_field = field_focus['omega_tilde_0'] - omega_0
    _, Ek_pert = shell_spectrum(delta_v_min_field)
    k_star = int(np.argmax(Ek_pert[1:]) + 1)
    return dict(kbins=kbins, Ek_pool=Ek_pool, Ek_p95=Ek_p95, Ek_true=Ek_true,
                field_focus=field_focus, k_star=k_star, eps_focus=eps_focus)


def plot_spectrum_at_eps(out_path, omega_0, fields, omega_0_pool, sample_name,
                         eps_focus=None):
    """Panel (c): shell-averaged spectrum with high-probability envelope B_q(k).

    Curves:
      * solid grey   : ensemble mean spectrum  <Z(k)>
      * dashed black : 95th-percentile envelope B_q(k) (q = 0.95)
      * shaded band  : region between <Z(k)> and B_q(k)
      * green        : true omega_0 spectrum
      * red dashed   : adversarial state omega_tilde_0 at the focus eps
      * blue dotted  : NN inversion G_theta(omega_tilde_T)
    """
    d = _compute_spectral_data(omega_0, fields, omega_0_pool, eps_focus)
    kbins   = d['kbins']
    Ek_mean = d['Ek_pool']
    Ek_p95  = d['Ek_p95']
    Ek_true = d['Ek_true']
    field_focus = d['field_focus']; eps_focus = d['eps_focus']
    _, Ek_fake = shell_spectrum(field_focus['omega_tilde_0'])
    _, Ek_NN   = shell_spectrum(field_focus['omega_hat_NN'])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.fill_between(kbins[1:], Ek_mean[1:], Ek_p95[1:],
                    color='gray', alpha=0.18, linewidth=0,
                    label=r'$[\langle Z\rangle,\,B_{q=0.95}]$')
    ax.loglog(kbins[1:], Ek_mean[1:], '-',  color='k', lw=2.2, alpha=0.7,
              label=r'$\langle Z(k)\rangle$')
    ax.loglog(kbins[1:], Ek_p95[1:],  '--', color='k', lw=2.6,
              label=r'$B_q(k)$')
    ax.loglog(kbins[1:], Ek_true[1:], '-',  color='g', lw=3.0,
              label=r'$Z[\omega_0]$')
    ax.loglog(kbins[1:], Ek_fake[1:], '--', color='r', lw=3.0,
              label=rf'$Z[\tilde\omega_0],\ \varepsilon={eps_focus:.2g}$')
    ax.loglog(kbins[1:], Ek_NN[1:],   ':',  color='b', lw=3.5,
              label=r'$Z[G_\theta(\tilde\omega_T)]$')
    ax.set_xlabel(r'$|k|$', fontsize=18)
    ax.set_ylabel(r'$Z(k)$', fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=13, loc='lower left', frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_energy_at_kstar(out_path, omega_0, fields, omega_0_pool, sample_name,
                         eps_focus=None):
    """Panel (d): Z(k_*) vs epsilon, with mean and 95th-percentile envelope."""
    d = _compute_spectral_data(omega_0, fields, omega_0_pool, eps_focus)
    Ek_pool = d['Ek_pool']; Ek_p95 = d['Ek_p95']
    Ek_true = d['Ek_true']; k_star = d['k_star']
    E_mean_kstar = float(Ek_pool[k_star])
    E_p95_kstar  = float(Ek_p95[k_star])
    E_true_kstar = float(Ek_true[k_star])
    eps_list = [f['eps'] for f in fields]
    E_fake_kstar, E_NN_kstar = [], []
    for f in fields:
        _, Ek_t0 = shell_spectrum(f['omega_tilde_0'])
        _, Ek_nn = shell_spectrum(f['omega_hat_NN'])
        E_fake_kstar.append(float(Ek_t0[k_star]))
        E_NN_kstar.append(float(Ek_nn[k_star]))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.axhline(E_mean_kstar, color='k', ls='-',  lw=2.2, alpha=0.7,
               label=r'$\langle Z(k_*)\rangle$')
    ax.axhline(E_p95_kstar,  color='k', ls='--', lw=2.6,
               label=r'$B_q(k_*)$')
    ax.axhline(E_true_kstar, color='g', ls=':',  lw=2.6,
               label=r'$Z[\omega_0](k_*)$')
    ax.plot(eps_list, E_fake_kstar, 'rs--', lw=3.0, ms=10,
            label=r'$Z[\tilde\omega_0](k_*)\propto\varepsilon^2$')
    ax.plot(eps_list, E_NN_kstar,   'bo-',  lw=3.0, ms=10,
            label=r'$Z[G_\theta(\tilde\omega_T)](k_*)$')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\varepsilon$', fontsize=18)
    ax.set_ylabel(rf'$Z(k_*\!=\!{k_star})$', fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=13, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_spectral_fingerprint(out_path, omega_0, fields, omega_T_pool, modes,
                              sample_name, eps_focus=None):
    """Backward-compatible combined plot — calls the two split functions and
    glues them into one PNG.  Kept for the export pathway; the manuscript
    figures now use the two separate plots."""
    base, ext = os.path.splitext(out_path)
    p1 = base + '_spectrum'       + ext
    p2 = base + '_energy_kstar'   + ext
    plot_spectrum_at_eps(p1, omega_0, fields, omega_T_pool, sample_name, eps_focus)
    plot_energy_at_kstar(p2, omega_0, fields, omega_T_pool, sample_name, eps_focus)


def plot_epsilon_sweep(out_path, all_rows, untrained_rows=None):
    """Pool across samples and plot the 4 (5 with untrained) relative norms vs eps."""
    eps_set = sorted({r['eps'] for r in all_rows})
    def collect(key, rows):
        return np.array([[r[key] for r in rows if r['eps'] == e] for e in eps_set])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    series = [
        ('drift_0',         'tab:blue',
         r'$\|\tilde\omega_0-\omega_0\|_2/\|\omega_0\|_2$'),
        ('residual_T',      'tab:red',
         r'$\|\tilde\omega_T-\omega_T\|_2/\|\omega_T\|_2$'),
        ('err_NN_to_true',  'tab:green',
         r'$\|G_\theta(\tilde\omega_T)-\omega_0\|_2/\|\omega_0\|_2$'),
        ('err_NN_to_fake',  'tab:purple',
         r'$\|G_\theta(\tilde\omega_T)-\tilde\omega_0\|_2/\|\omega_0\|_2$'),
    ]
    for key, color, label in series:
        arr = collect(key, all_rows)              # shape (n_eps, n_samples)
        mean = arr.mean(axis=1); std = arr.std(axis=1)
        ax.plot(eps_set, mean, 'o-', color=color, label=label, lw=3.0, ms=10)
        ax.fill_between(eps_set, mean-std, mean+std, color=color, alpha=0.15)

    if untrained_rows is not None:
        arr_u = collect('err_NN_to_true', untrained_rows)
        mean = arr_u.mean(axis=1); std = arr_u.std(axis=1)
        ax.plot(eps_set, mean, 's--', color='tab:olive', lw=3.0, ms=10, alpha=0.8,
                label=r'$\|G_\theta^{\mathrm{un}}(\tilde\omega_T)-\omega_0\|_2/\|\omega_0\|_2$')
        ax.fill_between(eps_set, mean-std, mean+std, color='tab:olive', alpha=0.10)

    ax.set_xlabel(r'$\varepsilon$', fontsize=18)
    ax.set_ylabel(r'$\|\cdot\|_2/\|\omega_0\|_2$', fontsize=18)
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=12, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def parse_modes(s):
    out = []
    for tok in s.split(';'):
        a, b = tok.split(',')
        out.append((int(a), int(b)))
    return out


def parse_eps(s):
    return [float(x) for x in s.split(',')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str,
        default='/workspace/ge/unet_turbulence_inverse_multiT_gradloss_with_initial.pth')
    p.add_argument('--data-dir', type=str,
        default='/workspace/ge/dataset_test_multiT_with_initial')
    p.add_argument('--T', type=float, default=1.0)
    p.add_argument('--num-samples', type=int, default=8)
    p.add_argument('--epsilons', type=str, default='0,0.3,1.0,2.0,4.0',
        help='comma-separated epsilon values (relative to ||omega_0||)')
    p.add_argument('--modes', type=str,
        default='1,0;0,1;1,1;2,1;3,2;4,4;6,4;8,8')
    p.add_argument('--fd-eps', type=float, default=1e-3,
        help='finite-difference epsilon for J_F^K construction')
    p.add_argument('--output-dir', type=str,
        default='/workspace/ge/results/iso_terminal_perturbation')
    p.add_argument('--tag', type=str, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--skip-untrained', action='store_true',
        help='Skip the untrained-NN control loop')
    p.add_argument('--num-visual', type=int, default=2,
        help='number of samples for which to render visual grids')
    args = p.parse_args()

    modes = parse_modes(args.modes)
    eps_list = parse_eps(args.epsilons)
    n_steps = int(round(args.T / DT))
    tag = args.tag or f'T{args.T:.1f}'
    out_dir = os.path.join(args.output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[config] T={args.T}  n_steps={n_steps}  d={2*len(modes)}  "
          f"eps={eps_list}")
    print(f"[config] output -> {out_dir}")

    solver = make_solver()
    model_tr = load_nn(args.ckpt, device='cpu', untrained=False)
    print(f"[model]  trained: {args.ckpt}")
    if not args.skip_untrained:
        model_un = load_nn(args.ckpt, device='cpu', untrained=True, seed=args.seed)
        print(f"[model]  untrained: random init (seed={args.seed})")

    t_id_str = f"T{int(round(args.T*1000)):04d}"
    all_files = sorted(glob.glob(os.path.join(args.data_dir, f'sim_*_{t_id_str}.pt')))
    sample_files = all_files[:args.num_samples]
    print(f"[data]   using {len(sample_files)} samples at T={args.T}")

    # Build pool of initial fields (omega_0) for the spectral-fingerprint
    # data-manifold envelope.  Use the full test pool.
    print(f"[data]   building initial-state pool from {len(all_files)} files ...")
    omega_0_pool = []
    for f in all_files:
        d = torch.load(f, map_location='cpu', weights_only=False)
        u0_p = d['u0'].squeeze().numpy().astype(np.float64)
        v0_p = d['v0'].squeeze().numpy().astype(np.float64)
        omega_0_pool.append(solver.converter.vorticity_from_velocity(u0_p, v0_p))

    rows_trained = []
    rows_untrained = []
    rendered = 0

    for i, f in enumerate(sample_files):
        t0 = time.time()
        rows, fields, om0, omT = iso_terminal_run(
            f, model_tr, modes, solver, args.T, n_steps, eps_list, args.fd_eps)
        for r in rows: r['sample_id'] = i
        rows_trained.extend(rows)
        # Visual grid + spectral fingerprint for first num_visual samples
        if rendered < args.num_visual:
            name = os.path.basename(f).replace('.pt', '')
            vg = os.path.join(out_dir, f'visual_grid_{name}.png')
            plot_visual_grid(vg, fields, om0, omT, name)
            print(f"  [render] {vg}")
            sf = os.path.join(out_dir, f'spectral_fingerprint_{name}.png')
            plot_spectral_fingerprint(sf, om0, fields, omega_0_pool, modes, name)
            print(f"  [render] {sf}")
            rendered += 1
        if not args.skip_untrained:
            rows_u, _, _, _ = iso_terminal_run(
                f, model_un, modes, solver, args.T, n_steps, eps_list, args.fd_eps)
            for r in rows_u: r['sample_id'] = i
            rows_untrained.extend(rows_u)
        dt = time.time() - t0
        sm = next(r for r in rows if r['eps'] == max(eps_list))
        print(f"[sample {i}] sigma_min={rows[0]['sigma_min']:.3e}  "
              f"@eps_max  residual_T={sm['residual_T']:.3e}  "
              f"drift_0={sm['drift_0']:.3f}  "
              f"err_NN_true={sm['err_NN_to_true']:.3f}  "
              f"err_NN_fake={sm['err_NN_to_fake']:.3f}  ({dt:.1f}s)")

    # CSV
    csv_path = os.path.join(out_dir, 'trajectories.csv')
    fieldnames = ['sample_id', 'eps', 'sigma_min', 'sigma_max',
                  'norm_omega_0', 'norm_omega_T',
                  'drift_0', 'residual_T', 'err_NN_to_true', 'err_NN_to_fake',
                  'model']
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_trained:
            r2 = {**r, 'model': 'trained'}; w.writerow({k: r2.get(k, '') for k in fieldnames})
        for r in rows_untrained:
            r2 = {**r, 'model': 'untrained'}; w.writerow({k: r2.get(k, '') for k in fieldnames})
    print(f"[csv]  {csv_path}")

    # Sweep plot
    sweep_path = os.path.join(out_dir, 'epsilon_sweep.png')
    plot_epsilon_sweep(sweep_path, rows_trained,
                       untrained_rows=None if args.skip_untrained else rows_untrained)
    print(f"[plot] {sweep_path}")

    # JSON summary (per epsilon)
    def stats(rows, key):
        eps_set = sorted({r['eps'] for r in rows})
        return {f'{e}': {
            'mean': float(np.mean([r[key] for r in rows if r['eps'] == e])),
            'std':  float(np.std( [r[key] for r in rows if r['eps'] == e])),
        } for e in eps_set}
    summary = {
        'config': {'T': args.T, 'n_steps': n_steps, 'eps_list': eps_list,
                   'modes': modes, 'num_samples': len(sample_files),
                   'fd_eps': args.fd_eps},
        'trained':   {k: stats(rows_trained,   k) for k in
                      ['drift_0', 'residual_T', 'err_NN_to_true', 'err_NN_to_fake']},
    }
    if not args.skip_untrained:
        summary['untrained'] = {k: stats(rows_untrained, k) for k in
                                ['drift_0', 'residual_T', 'err_NN_to_true', 'err_NN_to_fake']}
    with open(os.path.join(out_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f"[json] {os.path.join(out_dir, 'summary.json')}")


if __name__ == '__main__':
    main()
