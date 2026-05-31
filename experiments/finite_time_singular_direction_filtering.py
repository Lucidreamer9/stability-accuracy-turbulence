"""
Finite-time singular-direction filtering by the learned inverse.
============================================================
Mechanism experiment for the rewritten Section 2.3.

Hypothesis (geometric / operator-level):
  - The forward NS map's Fourier-projected tangent map J_F^K has a broad
    singular-value spectrum.  Small-sigma terminal singular directions
    are the ones that the formal inverse (J_F^K)^+ amplifies catastrophically.
  - Real turbulent data secants concentrate in the LARGE-sigma terminal
    subspace and avoid the small-sigma directions.
  - The learned neural inverse therefore inherits NO obligation to invert
    those small-sigma directions, and its tangent J_NN^K is gain-suppressed
    there.
  - Result: J_NN does not approximate the unstable pseudoinverse.  It is a
    data-regularized inverse: large gain only where the data has variance,
    small gain in the unstable directions that have low data support.

Pipeline (matches the data-generation pipeline exactly):
  - Loads (u0, v0) directly from each .pt file (these are the *actual*
    velocities the dataset's xT was advanced from).  Avoids the
    velocity_from_vorticity round-trip mismatch on the unperturbed orbit.
  - Perturbations on omega_0 are converted to (delta_u, delta_v) via the
    repo's velocity_from_vorticity (k2_modified Poisson + circdiff Biot-
    Savart).  Sanity check 14.2 measures the per-mode roundtrip error.
  - Centered finite differences with the numpy NS solver from data_gen
    (no autograd needed: J_F^K is at most ~16x16, FD is fast and avoids
    propagating any double-precision noise from the FFT through autograd).

Outputs go to results/finite_time_singular_direction_filtering/<tag>/.
"""
import os, sys, glob, json, argparse, time, math, csv
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspace/ge')
from data_gen import FiniteDifferenceNSolver
from train_u_multiT_gradloss import SimpleUNet


# ============================================================================
# Configuration constants tied to the data-generation pipeline.
# ============================================================================
N         = 64         # grid resolution
LX = LY   = 1.0
RE        = 1000.0
DT        = 1e-3
SCALE     = 20.0       # NN normalization: input = omega/20
T_MAX_NN  = 1.0        # NN time channel: t_norm = T / T_MAX
EPS_TINY  = 1e-30


# ============================================================================
# 1.  Helpers: Fourier subspace encoding / decoding (Hermitian-symmetric).
# ============================================================================
def field_to_fourier_subspace(omega, modes):
    """
    Project a real (N, N) field onto a list of Fourier modes K.

    Each mode k = (kx, ky) yields TWO real coordinates (Re, Im).
    Returns z in R^{2|K|}.

    Convention:  z[2j]   = Re fft2(omega)[kx_j % N, ky_j % N]
                 z[2j+1] = Im fft2(omega)[kx_j % N, ky_j % N]
    """
    omega_hat = np.fft.fft2(omega)
    z = np.zeros(2 * len(modes), dtype=np.float64)
    Nx, Ny = omega.shape
    for j, (kx, ky) in enumerate(modes):
        c = omega_hat[kx % Nx, ky % Ny]
        z[2*j  ] = c.real
        z[2*j+1] = c.imag
    return z


def fourier_subspace_to_field(delta_z, modes, shape=(N, N)):
    """
    Build a real-valued field whose Fourier coefficients match delta_z on
    the selected modes K and are zero elsewhere.

    Hermitian symmetry: for each k in K (with k != -k mod N), we set
        omega_hat[ k] = c
        omega_hat[-k] = c*
    which guarantees ifft2(omega_hat) is real.

    The returned delta_omega satisfies:
        fft2(delta_omega)[k]  = c
        fft2(delta_omega)[-k] = c*
    so the round-trip field_to_fourier_subspace is exact on K.
    """
    Nx, Ny = shape
    delta_hat = np.zeros((Nx, Ny), dtype=np.complex128)
    for j, (kx, ky) in enumerate(modes):
        c = delta_z[2*j] + 1j * delta_z[2*j+1]
        kxp = kx % Nx;   kyp = ky % Ny
        kxn = (-kx) % Nx; kyn = (-ky) % Ny
        delta_hat[kxp, kyp] += c
        if (kxp, kyp) != (kxn, kyn):
            delta_hat[kxn, kyn] += np.conj(c)
        else:
            # self-conjugate mode (e.g. (0,0) or (32,0) on N=64): coefficient
            # must be real.  Drop the imaginary part by symmetrising.
            delta_hat[kxp, kyp] = 2 * c.real
    delta_omega = np.fft.ifft2(delta_hat)
    return np.real(delta_omega)


def hermitian_imag_residue(delta_z, modes, shape=(N, N)):
    """Return max |Im part| of ifft2 of the constructed delta_hat (sanity)."""
    Nx, Ny = shape
    delta_hat = np.zeros((Nx, Ny), dtype=np.complex128)
    for j, (kx, ky) in enumerate(modes):
        c = delta_z[2*j] + 1j * delta_z[2*j+1]
        kxp = kx % Nx;  kyp = ky % Ny
        kxn = (-kx) % Nx; kyn = (-ky) % Ny
        delta_hat[kxp, kyp] += c
        if (kxp, kyp) != (kxn, kyn):
            delta_hat[kxn, kyn] += np.conj(c)
        else:
            delta_hat[kxp, kyp] = 2 * c.real
    return np.abs(np.imag(np.fft.ifft2(delta_hat))).max()


# ============================================================================
# 2.  Solver / NN wrappers.
# ============================================================================
def make_solver():
    return FiniteDifferenceNSolver(n=N, dx=LX/N, Re=RE, dt=DT, Lx=LX, Ly=LY)


def rollout_uv(solver, u, v, n_steps):
    """Advance (u, v) for n_steps without modification."""
    u = u.copy(); v = v.copy()
    for _ in range(n_steps):
        u, v = solver.forward_step(u, v)
    return u, v


def rollout_omega_from_uv(solver, u, v, n_steps):
    """(u, v) -> advance N steps -> curl -> omega_T."""
    uT, vT = rollout_uv(solver, u, v, n_steps)
    return solver.converter.vorticity_from_velocity(uT, vT)


def _build_circdiff_eigvals(solver):
    """
    Eigenvalues of the forward-diff circdiff operator used by data_gen.

    circdiff_x is implemented as np.roll(f, -1, axis=1) - f, i.e. f[j+1]-f[j].
    On the Fourier basis f[x] = exp(2*pi*i*k*x/N), this has eigenvalue
        alpha_x(k) = (exp(2*pi*i*k_x/N) - 1) / dx.
    Similarly for circdiff_y.  Both are complex.
    """
    N = solver.n
    kx = np.arange(N); ky = np.arange(N)
    ax = (np.exp(2j*np.pi*kx/N) - 1) / solver.dx
    ay = (np.exp(2j*np.pi*ky/N) - 1) / solver.dy
    AX, AY = np.meshgrid(ax, ay, indexing='ij')
    return AX, AY


def vorticity_perturbation_to_velocity(solver, delta_omega):
    """
    Stencil-consistent inverse: return (delta_u, delta_v) satisfying
        curl_FD(du, dv) = delta_omega   exactly,
        div_FD(du, dv)  = 0
    where curl_FD and div_FD are the *exact* operators used by
    `vorticity_from_velocity` and the solver's pressure projection.

    Derivation.  In Fourier with forward-diff symbol
        alpha_x = (exp(+2 pi i k_x / N) - 1) / dx,
        alpha_y = (exp(+2 pi i k_y / N) - 1) / dy,
    we have:
        - solver's div_FD eigenvalue = alpha_x du_hat + alpha_y dv_hat
        - solver's curl_FD (= vorticity_from_velocity) eigenvalue
          = beta_x dv_hat - beta_y du_hat,  where beta = conj(alpha)
        - alpha * beta = |alpha|^2 (real).
    Solving the two-equation system:
        du_hat = -alpha_y * domega_hat / (|alpha_x|^2 + |alpha_y|^2)
        dv_hat = +alpha_x * domega_hat / (|alpha_x|^2 + |alpha_y|^2)
    The denominator equals the solver's central-diff Laplacian symbol
    `k2_poisson` (= 4 sin^2(pi k_x / N)/dx^2 + 4 sin^2(pi k_y / N)/dy^2),
    so we reuse it.  Hermitian symmetry of (du_hat, dv_hat) is
    automatic because alpha(-k) = alpha(k)* and k2_poisson is real-even.

    Why this differs from `solver.converter.velocity_from_vorticity`:
        That helper uses the same Poisson denominator but writes
            du = circdiff_y(psi)/dy,  dv = -circdiff_x(psi)/dx,
        where the cdiff calls have shift [-1] (returning beta * psi).
        Composing with curl_FD (also beta-based) gives curl(du, dv)
        = -(beta_x^2 + beta_y^2) psi != omega in general — the residue
        is ~10-80% across our K (sanity check 14.2 route_A).
    """
    AX, AY = _build_circdiff_eigvals(solver)
    k2_pos = np.abs(AX)**2 + np.abs(AY)**2
    k2_pos[0, 0] = 1.0                               # avoid 0-division for mean mode
    dom_hat = np.fft.fft2(delta_omega)
    du_hat = -AY * dom_hat / k2_pos
    dv_hat =  AX * dom_hat / k2_pos
    du_hat[0, 0] = 0.0
    dv_hat[0, 0] = 0.0
    du = np.real(np.fft.ifft2(du_hat))
    dv = np.real(np.fft.ifft2(dv_hat))
    return du, dv


def load_nn(ckpt_path, device='cpu', untrained=False, seed=0):
    """Load NN from checkpoint, or return a freshly-initialised model.

    untrained=True is the control: random Xavier/He init, BatchNorm running
    stats at their default (mean=0, var=1), ReLU.  The network is still
    deterministic in eval(), but has had ZERO exposure to data.
    If the trained-NN's correlations (filter_ratio vs 1/sigma, filter_ratio
    vs data_support) persist on this model, they are architecture bias.
    If they vanish, they are genuinely learned from data.
    """
    torch.manual_seed(seed)
    model = SimpleUNet(n_channels=2, n_classes=1).to(device)
    if not untrained:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    model.eval()
    return model


def neural_inverse(model, omega_T_phys, T, device='cpu'):
    """
    Apply the learned inverse to a physical-units omega_T.

    Matches training:
        input  = (omega_T / SCALE, t_norm)
        output = omega_0_pred / SCALE

    Returns omega_0_pred in physical units.
    """
    with torch.no_grad():
        x = torch.from_numpy(omega_T_phys.astype(np.float32) / SCALE)
        x = x.view(1, 1, N, N).to(device)
        t_norm = max(0.0, min(1.0, T / T_MAX_NN))
        t_chan = torch.full_like(x, t_norm)
        inp = torch.cat([x, t_chan], dim=1)
        pred = model(inp)
    return pred[0, 0].cpu().numpy().astype(np.float64) * SCALE


# ============================================================================
# 3.  J_F^K via centered finite differences (velocity-form NS solver).
# ============================================================================
def build_JF_K(solver, u0, v0, modes, n_steps, eps):
    """
    J_F^K[:, j] = d zT^K / d (z0^K)_j  via centered FD on the velocity solver.

    Steps for each j:
      1. delta_z = eps e_j
      2. delta_omega_0 = fourier_subspace_to_field(delta_z, modes)
      3. delta_u, delta_v = vorticity_perturbation_to_velocity(delta_omega_0)
      4. omega_T_plus  = curl(advance(u0 + delta_u, v0 + delta_v))
         omega_T_minus = curl(advance(u0 - delta_u, v0 - delta_v))
      5. zT_plus  = field_to_fourier_subspace(omega_T_plus,  modes)
         zT_minus = field_to_fourier_subspace(omega_T_minus, modes)
      6. J_F[:, j] = (zT_plus - zT_minus) / (2 eps)
    """
    d = 2 * len(modes)
    J = np.zeros((d, d), dtype=np.float64)
    for j in range(d):
        e = np.zeros(d); e[j] = eps
        d_om = fourier_subspace_to_field(e, modes)
        du, dv = vorticity_perturbation_to_velocity(solver, d_om)
        omT_p = rollout_omega_from_uv(solver, u0 + du, v0 + dv, n_steps)
        omT_m = rollout_omega_from_uv(solver, u0 - du, v0 - dv, n_steps)
        zT_p = field_to_fourier_subspace(omT_p, modes)
        zT_m = field_to_fourier_subspace(omT_m, modes)
        J[:, j] = (zT_p - zT_m) / (2 * eps)
    return J


# ============================================================================
# 4.  J_NN^K via centered FD through the network.
# ============================================================================
def build_JNN_K(model, omega_T_base, T, modes, eps, device='cpu'):
    """
    J_NN^K[:, j] = d z0_hat^K / d (zT^K)_j via centered FD through neural net.

    Perturbations are added to omega_T_base (physical units), and the NN
    handles the /SCALE normalization internally.
    """
    d = 2 * len(modes)
    J = np.zeros((d, d), dtype=np.float64)
    for j in range(d):
        e = np.zeros(d); e[j] = eps
        d_omT = fourier_subspace_to_field(e, modes)
        om0_p = neural_inverse(model, omega_T_base + d_omT, T, device=device)
        om0_m = neural_inverse(model, omega_T_base - d_omT, T, device=device)
        z0_p = field_to_fourier_subspace(om0_p, modes)
        z0_m = field_to_fourier_subspace(om0_m, modes)
        J[:, j] = (z0_p - z0_m) / (2 * eps)
    return J


# ============================================================================
# 5.  Direction-wise analysis.
# ============================================================================
def per_direction_metrics(J_F, J_NN):
    """
    Returns dict keyed by:
        sigma, formal_gain, neural_gain, filter_ratio, alignment, U, S, Vt
    """
    U, S, Vt = np.linalg.svd(J_F, full_matrices=False)   # J_F = U diag(S) Vt
    d = len(S)
    formal_gain = 1.0 / (S + EPS_TINY)
    neural_resp = J_NN @ U                                # J_NN u_i for each i
    neural_gain = np.linalg.norm(neural_resp, axis=0)
    # alignment with formal-inverse direction v_i:
    align = np.zeros(d)
    for i in range(d):
        r = neural_resp[:, i]
        v = Vt[i, :]
        nr = np.linalg.norm(r); nv = np.linalg.norm(v)
        align[i] = float(np.dot(r, v) / (nr * nv + EPS_TINY))
    filter_ratio = neural_gain * S    # = neural_gain / formal_gain
    return {
        'sigma': S, 'formal_gain': formal_gain, 'neural_gain': neural_gain,
        'filter_ratio': filter_ratio, 'alignment': align,
        'U': U, 'S': S, 'Vt': Vt,
    }


def matrix_metrics(J_F, J_NN):
    JF_pinv = np.linalg.pinv(J_F)
    nF = np.linalg.norm(J_F, 'fro')
    nFi = np.linalg.norm(JF_pinv, 'fro')
    nNN = np.linalg.norm(J_NN, 'fro')
    align = float(np.sum(J_NN * JF_pinv) / (nNN * nFi + EPS_TINY))
    cond  = float(np.linalg.cond(J_F))
    return {
        'norm_J_F': float(nF),
        'norm_J_F_inv': float(nFi),
        'norm_J_NN': float(nNN),
        'matrix_alignment': align,
        'matrix_gain_ratio': float(nNN / (nFi + EPS_TINY)),
        'condition_number': cond,
    }


def data_support_on_directions(U, modes, omega_T_pool, n_pairs, rng):
    """
    For pairs of test trajectories, project terminal-secant onto U columns.

    Returns:
        data_support_i      = mean(alpha_i^2)
        data_support_frac_i = mean(alpha_i^2) / mean(||delta_zT||^2)
        per-pair alpha matrix shape (n_pairs, d)  for debug
    """
    M = len(omega_T_pool)
    d = U.shape[1]
    alphas = np.zeros((n_pairs, d))
    secant_norms2 = np.zeros(n_pairs)
    for p in range(n_pairs):
        a, b = rng.choice(M, size=2, replace=False)
        zTa = field_to_fourier_subspace(omega_T_pool[a], modes)
        zTb = field_to_fourier_subspace(omega_T_pool[b], modes)
        dz  = zTa - zTb
        alphas[p, :] = U.T @ dz
        secant_norms2[p] = float(dz @ dz)
    ds = (alphas**2).mean(axis=0)
    ds_frac = ds / (secant_norms2.mean() + EPS_TINY)
    return ds, ds_frac, alphas


# ============================================================================
# 6.  Plotting.
# ============================================================================
def plot_singular_spectrum(out_dir, samples_metrics, T):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for s, m in enumerate(samples_metrics):
        axs[0].semilogy(m['sigma'], 'o-', alpha=0.6, label=f's={s}')
        axs[1].plot(np.log(m['sigma'] + EPS_TINY) / T, 'o-', alpha=0.6,
                    label=f's={s}')
    axs[0].set_xlabel('singular index $i$'); axs[0].set_ylabel(r'$\sigma_i$')
    axs[0].set_title('Singular spectrum of $J_F^K$'); axs[0].grid(alpha=0.3)
    axs[1].set_xlabel('singular index $i$'); axs[1].set_ylabel(r'$\log\sigma_i / T$')
    axs[1].set_title('Finite-time exponent (not Lyapunov)'); axs[1].grid(alpha=0.3)
    if len(samples_metrics) <= 8:
        axs[0].legend(fontsize=7); axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'singular_spectrum.png'), dpi=150)
    plt.close(fig)


def plot_inverse_gain_vs_nn_gain(out_dir, samples_metrics):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, m in enumerate(samples_metrics):
        idx = np.arange(len(m['sigma']))
        ax.semilogy(idx, m['formal_gain'], 's-', color='#d62728', alpha=0.5,
                    label='formal $1/\\sigma_i$' if s == 0 else None)
        ax.semilogy(idx, m['neural_gain'], 'o-', color='#1f77b4', alpha=0.5,
                    label='neural $\\|J_{NN} u_i\\|$' if s == 0 else None)
    ax.set_xlabel('singular index $i$'); ax.set_ylabel('gain')
    ax.set_title('Formal inverse gain vs neural gain (per sample, all directions)')
    ax.grid(alpha=0.3, which='both'); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'inverse_gain_vs_nn_gain.png'), dpi=150)
    plt.close(fig)


def plot_filter_ratio_vs_singular_value(out_dir, samples_metrics):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, m in enumerate(samples_metrics):
        ax.loglog(m['formal_gain'], m['filter_ratio'], 'o', alpha=0.5)
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.set_xlabel(r'$1/\sigma_i$ (formal inverse gain)')
    ax.set_ylabel(r'filter ratio $= \|J_{NN} u_i\| \cdot \sigma_i$')
    ax.set_title('Filter ratio vs formal gain (1=NN matches inverse, <<1=NN suppresses)')
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'filter_ratio_vs_singular_value.png'), dpi=150)
    plt.close(fig)


def plot_data_support_vs_direction(out_dir, samples_metrics):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, m in enumerate(samples_metrics):
        idx = np.arange(len(m['sigma']))
        ax.semilogy(idx, m['data_support'] + EPS_TINY, 'o-', alpha=0.5,
                    label=f's={s}')
    ax.set_xlabel('singular index $i$ (sorted by decreasing $\\sigma$)')
    ax.set_ylabel(r'$\langle\alpha_i^2\rangle$ (data energy on $u_i$)')
    ax.set_title('Data support along terminal singular directions')
    ax.grid(alpha=0.3, which='both')
    if len(samples_metrics) <= 8: ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'data_support_vs_direction.png'), dpi=150)
    plt.close(fig)


def plot_filter_ratio_colored_by_data_support(out_dir, samples_metrics):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    fg, fr, ds = [], [], []
    for m in samples_metrics:
        fg.extend(m['formal_gain'])
        fr.extend(m['filter_ratio'])
        ds.extend(m['data_support_frac'])
    fg = np.array(fg); fr = np.array(fr); ds = np.array(ds)
    sc = ax.scatter(fg, fr, c=np.log10(ds + 1e-30), cmap='viridis', s=30, alpha=0.85)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r'$\log_{10}$ data-support fraction')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.set_xlabel(r'$1/\sigma_i$ (formal inverse gain)')
    ax.set_ylabel(r'filter ratio $= \|J_{NN}u_i\|\cdot\sigma_i$')
    ax.set_title('Key plot: filter ratio vs formal gain, coloured by data support')
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'filter_ratio_colored_by_data_support.png'),
                dpi=150)
    plt.close(fig)


def plot_direction_alignment(out_dir, samples_metrics):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, m in enumerate(samples_metrics):
        ax.plot(m['sigma'], m['alignment'], 'o', alpha=0.5)
    ax.set_xscale('log')
    ax.axhline( 1.0, color='gray', lw=0.5, ls=':')
    ax.axhline(-1.0, color='gray', lw=0.5, ls=':')
    ax.axhline( 0.0, color='gray', lw=0.5, ls='-')
    ax.set_xlabel(r'$\sigma_i$'); ax.set_ylabel(r'$\cos(J_{NN}u_i, v_i)$')
    ax.set_title('Direction alignment of NN response with formal inverse direction')
    ax.set_ylim(-1.1, 1.1); ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'direction_alignment_vs_singular_value.png'),
                dpi=150)
    plt.close(fig)


def plot_secant_summary(out_dir, secant_records):
    if not secant_records:
        return
    align_NN = np.array([r['secant_align_NN']  for r in secant_records])
    align_ph = np.array([r['secant_align_phys'] for r in secant_records])
    gain_NN  = np.array([r['secant_gain_NN']   for r in secant_records])
    gain_ph  = np.array([r['secant_gain_phys'] for r in secant_records])
    err_NN   = np.array([r['secant_rel_err_NN']  for r in secant_records])
    err_ph   = np.array([r['secant_rel_err_phys'] for r in secant_records])

    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    axs[0].hist([align_NN, align_ph], bins=20, label=['NN', 'phys $J_F^+$'])
    axs[0].set_title('Secant alignment with true $\\delta z_0$')
    axs[0].set_xlabel('cosine'); axs[0].legend(); axs[0].grid(alpha=0.3)

    axs[1].hist([np.log10(gain_NN+EPS_TINY), np.log10(gain_ph+EPS_TINY)],
                bins=20, label=['NN', 'phys'])
    axs[1].set_title('Secant gain $\\|pred\\|/\\|\\delta z_0\\|$ (log10)')
    axs[1].set_xlabel(r'$\log_{10}$ gain'); axs[1].legend(); axs[1].grid(alpha=0.3)

    axs[2].hist([np.log10(err_NN+EPS_TINY), np.log10(err_ph+EPS_TINY)],
                bins=20, label=['NN', 'phys'])
    axs[2].set_title('Secant relative error (log10)')
    axs[2].set_xlabel(r'$\log_{10}$ rel-err'); axs[2].legend(); axs[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'secant_inversion_summary.png'), dpi=150)
    plt.close(fig)


# ============================================================================
# 7.  Sanity checks.
# ============================================================================
def sanity_hermitian(modes, n_random=5, seed=0):
    rng = np.random.default_rng(seed)
    d = 2 * len(modes)
    residues = []
    for _ in range(n_random):
        z = rng.standard_normal(d)
        residues.append(hermitian_imag_residue(z, modes))
    return float(np.max(residues))


def sanity_omega_uv_roundtrip(solver, modes, eps=1e-3):
    """
    For each Fourier mode k in K, build a unit-norm vorticity perturbation,
    convert to (du, dv) via TWO routes, curl back, and report errors:

      route_A: solver.converter.velocity_from_vorticity (modified-k Poisson +
               circdiff Biot-Savart) — this is the OLD inverse and is
               INCONSISTENT with the FD curl on this grid.
      route_B: vorticity_perturbation_to_velocity (this experiment) —
               stencil-consistent: alpha-based inverse that exactly
               satisfies curl_FD(du, dv) = delta_omega.

    Route B should give residue at round-off (~1e-13).
    Route A is shown only to motivate the choice.
    """
    d = 2 * len(modes)
    rows = []
    for j in range(d):
        e = np.zeros(d); e[j] = eps
        d_om = fourier_subspace_to_field(e, modes)

        # route A (old, repo's velocity_from_vorticity)
        duA, dvA = solver.converter.velocity_from_vorticity(d_om)
        d_om_A = solver.converter.vorticity_from_velocity(duA, dvA)
        relA = np.linalg.norm(d_om_A - d_om) / (np.linalg.norm(d_om) + EPS_TINY)

        # route B (stencil-consistent)
        duB, dvB = vorticity_perturbation_to_velocity(solver, d_om)
        d_om_B = solver.converter.vorticity_from_velocity(duB, dvB)
        relB = np.linalg.norm(d_om_B - d_om) / (np.linalg.norm(d_om) + EPS_TINY)

        rows.append({'idx': j, 'rel_routeA_old': float(relA), 'rel_routeB_new': float(relB)})
    return rows


def sanity_forward_consistency(solver, sample_files, n_check=3):
    """Compare solver rollout xT vs dataset xT for a few samples."""
    rows = []
    for f in sample_files[:n_check]:
        d = torch.load(f, map_location='cpu', weights_only=False)
        u0 = d['u0'].squeeze().numpy().astype(np.float64)
        v0 = d['v0'].squeeze().numpy().astype(np.float64)
        xT_ds = d['xT'].squeeze().numpy().astype(np.float64)
        n_steps = int(d['steps'])
        omT_solver = rollout_omega_from_uv(solver, u0, v0, n_steps)
        rel = np.linalg.norm(omT_solver - xT_ds) / (np.linalg.norm(xT_ds) + EPS_TINY)
        rows.append({
            'file': os.path.basename(f),
            'n_steps': n_steps,
            'rel_to_dataset': float(rel),
        })
    return rows


# ============================================================================
# 8.  Main driver.
# ============================================================================
def parse_modes(s):
    out = []
    for tok in s.split(';'):
        a, b = tok.split(',')
        out.append((int(a), int(b)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str,
        default='/workspace/ge/unet_turbulence_inverse_multiT_gradloss_with_initial.pth')
    p.add_argument('--data-dir', type=str,
        default='/workspace/ge/dataset_test_multiT_with_initial')
    p.add_argument('--T', type=float, default=1.0,
        help='terminal horizon (must be one of the dataset T values)')
    p.add_argument('--num-samples', type=int, default=8)
    p.add_argument('--num-pairs',   type=int, default=64)
    p.add_argument('--epsilon',     type=float, default=1e-3)
    p.add_argument('--epsilon-check', action='store_true')
    p.add_argument('--modes', type=str,
        default='1,0;0,1;1,1;2,1;3,2;4,4;6,4;8,8')
    p.add_argument('--output-dir', type=str,
        default='/workspace/ge/results/finite_time_singular_direction_filtering')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--seed',   type=int, default=0)
    p.add_argument('--tag',    type=str, default=None,
        help='subdirectory tag; defaults to T-value')
    p.add_argument('--untrained', action='store_true',
        help='Control: use a randomly-initialised NN (no checkpoint load) '
             'to test whether the trained-NN correlations are architecture '
             'bias.  Should disrupt the filter_ratio <-> data_support coupling.')
    args = p.parse_args()

    modes = parse_modes(args.modes)
    n_steps = int(round(args.T / DT))
    tag = args.tag or f'T{args.T:.1f}_eps{args.epsilon:.0e}'
    out_dir = os.path.join(args.output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[config] T={args.T}  n_steps={n_steps}  modes={modes}  d={2*len(modes)}")
    print(f"[config] output -> {out_dir}")

    # build solver, load NN
    solver = make_solver()
    model  = load_nn(args.ckpt, device=args.device,
                     untrained=args.untrained, seed=args.seed)
    if args.untrained:
        print(f"[model]  UNTRAINED (random init, seed={args.seed}) "
              f"[control experiment, no checkpoint loaded]")
    else:
        print(f"[model]  loaded {args.ckpt}")

    # find samples at requested horizon
    t_id_str = f"T{int(round(args.T*1000)):04d}"
    sample_files = sorted(glob.glob(os.path.join(args.data_dir, f'sim_*_{t_id_str}.pt')))
    if len(sample_files) == 0:
        raise RuntimeError(f"No samples found at horizon {t_id_str} in {args.data_dir}")
    print(f"[data]   {len(sample_files)} samples available at T={args.T}")
    sample_files = sample_files[:args.num_samples]

    # Build a pool of terminal vorticities from the FULL test set for data
    # support analysis (need many pairs).
    pool_files = sorted(glob.glob(os.path.join(args.data_dir, f'sim_*_{t_id_str}.pt')))
    omega_T_pool = []
    for f in pool_files:
        d = torch.load(f, map_location='cpu', weights_only=False)
        omega_T_pool.append(d['xT'].squeeze().numpy().astype(np.float64))
    print(f"[data]   pool of {len(omega_T_pool)} terminal fields for secants")

    # ---- sanity checks ----
    print("\n[sanity 14.1] Hermitian symmetry: max imag residue after ifft2 ...")
    res = sanity_hermitian(modes, n_random=10, seed=args.seed)
    print(f"  max |Im part| = {res:.3e}  (should be ~1e-15)")

    print("\n[sanity 14.2] omega -> uv -> omega per-mode roundtrip ...")
    print(f"  {'mode':>10}     route_A_old        route_B_new (this expt)")
    rows = sanity_omega_uv_roundtrip(solver, modes, eps=args.epsilon)
    for j, r in enumerate(rows):
        kx, ky = modes[j // 2]
        part = 'Re' if (j % 2 == 0) else 'Im'
        print(f"  ({kx:2d},{ky:2d}) {part}    {r['rel_routeA_old']:14.3e}     {r['rel_routeB_new']:14.3e}")

    print("\n[sanity 14.4] forward rollout vs dataset xT ...")
    rows = sanity_forward_consistency(solver, sample_files, n_check=3)
    for r in rows:
        print(f"  {r['file']:32s}  steps={r['n_steps']}  "
              f"rel(rollout, dataset xT) = {r['rel_to_dataset']:.3e}")

    # ---- main per-sample loop ----
    rng = np.random.default_rng(args.seed)
    samples_metrics = []
    matrix_rows = []
    secant_records = []

    for s, f in enumerate(sample_files):
        t0 = time.time()
        d = torch.load(f, map_location='cpu', weights_only=False)
        u0 = d['u0'].squeeze().numpy().astype(np.float64)
        v0 = d['v0'].squeeze().numpy().astype(np.float64)
        omega_0 = solver.converter.vorticity_from_velocity(u0, v0)
        omega_T = rollout_omega_from_uv(solver, u0, v0, n_steps)

        J_F  = build_JF_K(solver, u0, v0, modes, n_steps, args.epsilon)
        J_NN = build_JNN_K(model, omega_T, args.T, modes, args.epsilon, args.device)

        m = per_direction_metrics(J_F, J_NN)
        ds, ds_frac, _ = data_support_on_directions(
            m['U'], modes, omega_T_pool, args.num_pairs, rng)
        m['data_support']      = ds
        m['data_support_frac'] = ds_frac

        mm = matrix_metrics(J_F, J_NN)
        mm['sample_id'] = s
        mm['file']      = os.path.basename(f)
        mm['num_modes'] = len(modes)
        mm['subspace_dim'] = 2 * len(modes)
        matrix_rows.append(mm)

        # Secant test (Section 11): use J_NN, J_F^+ on real data secants.
        JF_pinv = np.linalg.pinv(J_F)
        zT_s = field_to_fourier_subspace(omega_T, modes)
        z0_s = field_to_fourier_subspace(omega_0, modes)
        for p in range(min(args.num_pairs, len(omega_T_pool) - 1)):
            other = rng.choice(len(omega_T_pool))
            zT_b = field_to_fourier_subspace(omega_T_pool[other], modes)
            # pair the reference sample with another sample.  delta_z0 is
            # available only if we know the OTHER sample's omega_0 too.
            # Load it on the fly (cheap):
            d_b = torch.load(pool_files[other], map_location='cpu', weights_only=False)
            u0_b = d_b['u0'].squeeze().numpy().astype(np.float64)
            v0_b = d_b['v0'].squeeze().numpy().astype(np.float64)
            om0_b = solver.converter.vorticity_from_velocity(u0_b, v0_b)
            z0_b  = field_to_fourier_subspace(om0_b, modes)

            dz_T = zT_s - zT_b
            dz_0 = z0_s - z0_b
            pred_NN   = J_NN  @ dz_T
            pred_phys = JF_pinv @ dz_T
            n0 = np.linalg.norm(dz_0) + EPS_TINY
            secant_records.append({
                'pair_id': len(secant_records),
                'sample_a': s, 'sample_b': int(other),
                'reference_sample': s,
                'secant_align_NN':   float(pred_NN @ dz_0   / (np.linalg.norm(pred_NN)*n0   + EPS_TINY)),
                'secant_align_phys': float(pred_phys @ dz_0 / (np.linalg.norm(pred_phys)*n0 + EPS_TINY)),
                'secant_gain_NN':   float(np.linalg.norm(pred_NN)   / n0),
                'secant_gain_phys': float(np.linalg.norm(pred_phys) / n0),
                'secant_rel_err_NN':   float(np.linalg.norm(pred_NN   - dz_0) / n0),
                'secant_rel_err_phys': float(np.linalg.norm(pred_phys - dz_0) / n0),
                'norm_delta_zT': float(np.linalg.norm(dz_T)),
                'norm_delta_z0': float(n0),
            })

        samples_metrics.append(m)
        print(f"[sample {s}] cond(J_F^K)={mm['condition_number']:.3e}  "
              f"||J_F||={mm['norm_J_F']:.3e}  ||J_NN||={mm['norm_J_NN']:.3e}  "
              f"||J_F^+||={mm['norm_J_F_inv']:.3e}  ({time.time()-t0:.1f}s)")

    # ---- write CSVs ----
    sd_csv = os.path.join(out_dir, 'singular_direction_metrics.csv')
    with open(sd_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['sample_id', 'direction_index', 'sigma',
                    'finite_time_exponent', 'formal_inverse_gain', 'neural_gain',
                    'filter_ratio', 'direction_alignment',
                    'data_support', 'data_support_frac',
                    'condition_number_sample',
                    'norm_J_NN', 'norm_J_F_inv'])
        for s, m in enumerate(samples_metrics):
            cond = matrix_rows[s]['condition_number']
            for i in range(len(m['sigma'])):
                w.writerow([s, i, m['sigma'][i],
                            np.log(m['sigma'][i] + EPS_TINY) / args.T,
                            m['formal_gain'][i], m['neural_gain'][i],
                            m['filter_ratio'][i], m['alignment'][i],
                            m['data_support'][i], m['data_support_frac'][i],
                            cond, matrix_rows[s]['norm_J_NN'],
                            matrix_rows[s]['norm_J_F_inv']])
    print(f"[csv] {sd_csv}")

    mm_csv = os.path.join(out_dir, 'matrix_metrics.csv')
    with open(mm_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        keys = list(matrix_rows[0].keys())
        w.writerow(keys)
        for r in matrix_rows: w.writerow([r[k] for k in keys])
    print(f"[csv] {mm_csv}")

    sc_csv = os.path.join(out_dir, 'secant_metrics.csv')
    with open(sc_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        keys = list(secant_records[0].keys())
        w.writerow(keys)
        for r in secant_records: w.writerow([r[k] for k in keys])
    print(f"[csv] {sc_csv}")

    # ---- summary JSON ----
    def pooled(field):
        return np.concatenate([m[field] for m in samples_metrics])
    summary = {
        'config': {
            'T': args.T, 'n_steps': n_steps, 'epsilon': args.epsilon,
            'modes': modes, 'subspace_dim': 2*len(modes),
            'num_samples': args.num_samples, 'num_pairs': args.num_pairs,
            'pool_size': len(omega_T_pool),
        },
        'pooled_means': {k: float(np.mean(pooled(k))) for k in
            ['sigma', 'formal_gain', 'neural_gain', 'filter_ratio',
             'alignment', 'data_support', 'data_support_frac']},
        'pooled_stds': {k: float(np.std(pooled(k))) for k in
            ['sigma', 'formal_gain', 'neural_gain', 'filter_ratio',
             'alignment', 'data_support', 'data_support_frac']},
        'correlations': {
            'log(formal_gain) vs log(neural_gain)': float(np.corrcoef(
                np.log(pooled('formal_gain') + EPS_TINY),
                np.log(pooled('neural_gain') + EPS_TINY))[0, 1]),
            'log(formal_gain) vs filter_ratio': float(np.corrcoef(
                np.log(pooled('formal_gain') + EPS_TINY),
                pooled('filter_ratio'))[0, 1]),
            'data_support vs filter_ratio': float(np.corrcoef(
                pooled('data_support'), pooled('filter_ratio'))[0, 1]),
            'data_support vs alignment': float(np.corrcoef(
                pooled('data_support'), pooled('alignment'))[0, 1]),
        },
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2, default=lambda o: float(o))
    print(f"[json] {os.path.join(out_dir, 'summary.json')}")

    # ---- plots ----
    plot_singular_spectrum(out_dir, samples_metrics, args.T)
    plot_inverse_gain_vs_nn_gain(out_dir, samples_metrics)
    plot_filter_ratio_vs_singular_value(out_dir, samples_metrics)
    plot_data_support_vs_direction(out_dir, samples_metrics)
    plot_filter_ratio_colored_by_data_support(out_dir, samples_metrics)
    plot_direction_alignment(out_dir, samples_metrics)
    plot_secant_summary(out_dir, secant_records)
    print(f"[plots] saved to {out_dir}")

    # ---- assumptions banner ----
    readme = (
"Assumptions and conventions:\n"
" * Solver: numpy FiniteDifferenceNSolver from data_gen.py (velocity form).\n"
" * Time step dt=1e-3, Re=1000, grid 64x64 periodic, Lx=Ly=1.\n"
" * Forward map evaluated as (u0, v0) -> rollout -> curl -> omega_T.\n"
" * Vorticity perturbation -> (du, dv) via converter.velocity_from_vorticity\n"
"   (k^2-modified Poisson + circdiff Biot-Savart).  Per-mode roundtrip error\n"
"   is reported in the sanity check; this is the residue of the inconsistency\n"
"   between forward-diff curl and central-diff Laplacian on the FD grid.\n"
" * NN: 2-channel input (omega_T/20, T/T_max), 1-channel output omega_0/20.\n"
"   T_max = 1.0 (training convention).\n"
" * Fourier subspace K is GALERKIN-PROJECTED, not the restriction of the\n"
"   full operator to K.  J_F^K and J_NN^K are 2|K| x 2|K| real matrices.\n"
"   Singular directions of J_F^K are finite-time singular directions of\n"
"   the Fourier-projected tangent map.  They are LYAPUNOV-LIKE in spirit\n"
"   but are NOT covariant Lyapunov vectors.\n"
" * Centered FD with the configured epsilon.  Use --epsilon-check to sweep.\n"
)
    with open(os.path.join(out_dir, 'README.txt'), 'w') as fh:
        fh.write(readme)


if __name__ == '__main__':
    main()
