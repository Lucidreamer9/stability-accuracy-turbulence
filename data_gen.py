"""Core finite-difference Navier--Stokes solver and initial-condition sampler.

Defines:
  * FiniteDifferenceConverter: vorticity <-> velocity, modified-wavenumber
    Laplacian, central-shift / half-shift / circular-difference operators.
  * FiniteDifferenceLerayProjector: spectral Leray projection enforcing
    incompressibility.
  * FiniteDifferenceNSolver: one-step velocity solver combining advection,
    semi-implicit viscosity (Fourier symbol of the modified Laplacian),
    and pressure projection (Helmholtz / pseudo-spectral Poisson solve).
  * generate_initial_condition: random streamfunction sample with Gaussian
    spectral envelope centered at k_peak = 4.
"""
import numpy as np


# =============================================================================
# 1. Core solver classes
# =============================================================================
class FiniteDifferenceConverter:
    def __init__(self, n, dx, dy=None, Lx=None, Ly=None):
        self.n = n
        self.dx = dx
        self.dy = dy if dy is not None else dx
        self.Lx = Lx if Lx is not None else n * dx
        self.Ly = Ly if Ly is not None else n * self.dy
        self._setup_modified_laplacian()

    def _setup_modified_laplacian(self):
        nx_indices = np.arange(self.n)
        ny_indices = np.arange(self.n)
        kx_mod = 2 * np.sin(np.pi * nx_indices / self.Lx * self.dx) / self.dx
        ky_mod = 2 * np.sin(np.pi * ny_indices / self.Ly * self.dy) / self.dy
        self.kx_fd, self.ky_fd = np.meshgrid(kx_mod, ky_mod, indexing='ij')
        self.k2_fd = self.kx_fd**2 + self.ky_fd**2
        self.k2_poisson = self.k2_fd.copy()
        self.k2_poisson[0, 0] = np.inf
        self.k2_den = self.k2_fd.copy()
        self.k2_den[0, 0] = 0.0

    @staticmethod
    def circdiff(field, shift):
        dy_shift, dx_shift = shift
        if dx_shift != 0:
            return np.roll(field, -dx_shift, axis=1) - field
        if dy_shift != 0:
            return np.roll(field, -dy_shift, axis=0) - field
        return np.zeros_like(field)

    @staticmethod
    def shifthalf(field, shift):
        dy_shift, dx_shift = shift
        if dx_shift != 0:
            return 0.5 * (np.roll(field, -dx_shift, axis=1) + field)
        if dy_shift != 0:
            return 0.5 * (np.roll(field, -dy_shift, axis=0) + field)
        return field

    @staticmethod
    def laplacian(field, dx, dy):
        return (np.roll(field, -1, axis=0) + np.roll(field, 1, axis=0) - 2 * field) / (dx * dx) + \
               (np.roll(field, -1, axis=1) + np.roll(field, 1, axis=1) - 2 * field) / (dy * dy)

    def vorticity_from_velocity(self, u, v):
        dudy = self.circdiff(u, [0, -1]) / self.dy
        dvdx = self.circdiff(v, [-1, 0]) / self.dx
        return dvdx - dudy

    def velocity_from_vorticity(self, omega):
        omega_hat = np.fft.fft2(omega)
        psi_hat = omega_hat / self.k2_poisson
        psi_hat[0, 0] = 0.0
        psi = np.real(np.fft.ifft2(psi_hat))
        u = self.circdiff(psi, [0, -1]) / self.dy
        v = -self.circdiff(psi, [-1, 0]) / self.dx
        return u, v


class FiniteDifferenceLerayProjector:
    def __init__(self, n, dx, dy=None, Lx=None, Ly=None):
        self.n = n
        self.dx = dx
        self.dy = dy if dy is not None else dx
        self.Lx = Lx if Lx is not None else n * dx
        self.Ly = Ly if Ly is not None else n * self.dy
        self._setup_fd_projector()

    def _setup_fd_projector(self):
        kx = 2 * np.sin(np.pi * np.arange(self.n) / self.Lx * self.dx) / self.dx
        ky = 2 * np.sin(np.pi * np.arange(self.n) / self.Ly * self.dy) / self.dy
        kx, ky = np.meshgrid(kx, ky, indexing='ij')
        k2 = kx**2 + ky**2
        k2[0, 0] = 1.0
        self.Pxx = 1 - kx**2 / k2
        self.Pxy = -kx * ky / k2
        self.Pyx = -ky * kx / k2
        self.Pyy = 1 - ky**2 / k2
        self.Pxx[0, 0] = 1.0
        self.Pxy[0, 0] = 0.0
        self.Pyx[0, 0] = 0.0
        self.Pyy[0, 0] = 1.0

    def apply(self, u_field, v_field):
        u_hat = np.fft.fft2(u_field)
        v_hat = np.fft.fft2(v_field)
        u_proj_hat = self.Pxx * u_hat + self.Pxy * v_hat
        v_proj_hat = self.Pyx * u_hat + self.Pyy * v_hat
        u_proj = np.real(np.fft.ifft2(u_proj_hat))
        v_proj = np.real(np.fft.ifft2(v_proj_hat))
        return u_proj, v_proj


class FiniteDifferenceNSolver:
    """One-step 2D incompressible Navier--Stokes velocity solver.

    Uses circular finite differences for the nonlinear flux divergences,
    a semi-implicit half-step treatment of viscosity in Fourier space
    (modified-wavenumber Laplacian symbol), and a Helmholtz pressure
    projection (pseudo-spectral Poisson solve) to enforce incompressibility.
    """

    def __init__(self, n, dx, Re, dt, dy=None, Lx=None, Ly=None):
        self.n = n
        self.dx = dx
        self.dy = dy if dy is not None else dx
        self.Lx = Lx if Lx is not None else n * dx
        self.Ly = Ly if Ly is not None else n * self.dy
        self.Re = Re
        self.dt = dt
        self.nu = 1.0 / Re
        self.converter = FiniteDifferenceConverter(n, dx, self.dy, self.Lx, self.Ly)
        self.projector = FiniteDifferenceLerayProjector(n, dx, self.dy, self.Lx, self.Ly)
        self.k2_den = self.converter.k2_den

    def forward_step(self, u, v):
        cdiff = self.converter.circdiff
        shalf = self.converter.shifthalf
        lap = self.converter.laplacian
        dx, dy, dt, nu = self.dx, self.dy, self.dt, self.nu

        uv = shalf(u, [0, -1]) * shalf(v, [-1, 0])
        duu_dx = cdiff(shalf(u, [-1, 0])**2, [1, 0]) / dx
        duv_dy = cdiff(uv,                  [0, 1]) / dy
        duv_dx = cdiff(uv,                  [1, 0]) / dx
        dvv_dy = cdiff(shalf(v, [0, -1])**2, [0, 1]) / dy
        adv_u = -duu_dx - duv_dy
        adv_v = -duv_dx - dvv_dy

        u_rhs = adv_u + 0.5 * nu * lap(u, dx, dy)
        v_rhs = adv_v + 0.5 * nu * lap(v, dx, dy)
        denom = (1.0 + 0.5 * dt * nu * self.k2_den)
        u_star = np.real(np.fft.ifft2(np.fft.fft2(u + dt * u_rhs) / denom))
        v_star = np.real(np.fft.ifft2(np.fft.fft2(v + dt * v_rhs) / denom))

        div = cdiff(u_star, [1, 0]) / dx + cdiff(v_star, [0, 1]) / dy
        p = np.real(np.fft.ifft2(np.fft.fft2(div) / self.converter.k2_poisson)) / dt

        dpdx = cdiff(p, [-1, 0]) / dx
        dpdy = cdiff(p, [0, -1]) / dy
        u_new = u_star - dt * dpdx
        v_new = v_star - dt * dpdy
        return u_new, v_new


# =============================================================================
# 2. Initial condition (random streamfunction + central differences)
# =============================================================================
def generate_initial_condition(solver, n, Lx, Ly):
    """Sample a random streamfunction with Gaussian spectral envelope (k_peak=4)
    and decode to (u, v) via central differences.  Used as a raw seed; downstream
    pipelines typically apply warm-up steps and amplitude rescaling.
    """
    kx = np.fft.fftfreq(n, d=Lx / n)
    ky = np.fft.fftfreq(n, d=Ly / n)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)
    K[0, 0] = 1.0

    k_peak = 4.0
    spectrum = np.exp(-(K - k_peak)**2 / 2.0)

    noise_phys = np.random.randn(n, n)
    psi_hat = np.fft.fft2(noise_phys) * spectrum
    psi_hat[0, 0] = 0.0
    psi = np.real(np.fft.ifft2(psi_hat))

    dx = Lx / n
    dy = Ly / n
    u = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2 * dy)
    v = -(np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2 * dx)

    return u, v
