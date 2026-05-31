# DHIT_2D_DA — 2D Decaying Isotropic Turbulence with 4DVar Data Assimilation

A minimal MATLAB demo of strong-constraint 4D-Var data assimilation applied to
two-dimensional decaying isotropic turbulence. The forward model is an
incompressible Navier-Stokes solver on a periodic square (staggered grid,
modified wavenumbers, Crank-Nicolson diffusion, Adams-Bashforth advection); the
control variable is the initial velocity field `(u0, v0)`, optimized with
L-BFGS to fit observations of the terminal velocity field.

## Requirements

- MATLAB R2019b or newer (uses string arrays and `+` string concatenation).
- No additional toolboxes are required. The L-BFGS optimizer
  (`utils/fminlbfgs.m`) is bundled.

## Code layout

```
DHIT_2D_DA_opensource/
├── DHIT_2D_yexiao.m            % main entry point — run this in MATLAB
├── core/                       % NS solver + DA class
│   ├── DA_Case.m               % case object: IC, forward, adjoint, optimize
│   ├── F_simulation.m          % forward NS solver
│   ├── AD_simulation.m         % adjoint NS solver (returns gradient of cost)
│   ├── LF_simulation.m         % linearized forward solver (for validation)
│   └── Solver_*_*.m            % initialization / postprocessing helpers
└── utils/                      % math utilities
    ├── circdiff.m              % periodic finite difference
    ├── shifthalf.m             % staggered-grid half shift
    ├── decay_ic.m              % default decaying-isotropic initial condition
    ├── divergence_free_projection.m
    ├── inter_sparse_obs.m      % interpolate sparse observations
    └── fminlbfgs.m             % L-BFGS optimizer (writes iteration log)
```

## Quick start

The workflow has two phases, controlled by `Data_Assimilation.DA_flag` near
the top of `DHIT_2D_yexiao.m`.

### Phase 1 — generate truth and observations

1. Open `DHIT_2D_yexiao.m` and set:

   ```matlab
   Data_Assimilation.DA_flag = 1;
   ```

2. Run the script. It will:
   - Build a random divergence-free initial condition.
   - Run the forward NS solver from `t = 0` to `t = Parameters.T`.
   - Save the truth IC to `cases/DA_DHIT_1000_uv/InitialCondition.mat`.
   - Save observed velocity fields at `Data_Assimilation.DA_obs_tlist` to
     `cases/DA_DHIT_1000_uv/obs.dat`.

### Phase 2 — assimilate

1. In `DHIT_2D_yexiao.m`, set:

   ```matlab
   Data_Assimilation.DA_flag = 2;
   ```

2. Run the script. It will:
   - Load the truth IC (needed to set up grid spacing — only the IC structure
     is reused; the optimizer starts from a guess interpolated from the
     observation).
   - Load `obs.dat` and build an initial guess by interpolating the sparse
     terminal-time observation.
   - Run L-BFGS for `Data_Assimilation.iteration_number` iterations,
     minimizing `0.5 * sum((U(T) - U_obs(T))^2)` over the initial velocity.
   - Save the result to `cases/DA_DHIT_1000_uv/DA_results_uv.mat`.

A per-iteration log is written to `cases/DA_DHIT_1000_uv/uv.txt`.

## Key knobs

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `Parameters.Re` | 1000 | Reynolds number |
| `Parameters.Nx`, `Parameters.Ny` | 64 | grid resolution (periodic square) |
| `Parameters.Lx`, `Parameters.Ly` | 1 | domain size |
| `Parameters.T` | 0.2 | assimilation window length |
| `Parameters.dt` | 0.001 | time step |
| `Data_Assimilation.iteration_number` | 100 | max L-BFGS iterations |
| `Data_Assimilation.obs_gap` | 1 | observation stride (1 = dense) |
| `Data_Assimilation.DA_obs_tlist` | terminal step | observation time indices |
| `Data_Assimilation.lambda` | 0 | H1 regularizer weight on `(u0, v0)` |

For a fast smoke test, lower `Data_Assimilation.iteration_number` to 5 and
keep `Parameters.T = 0.2`.

## Validation hooks

Setting `validation_flag = 1` in the script enables three sanity checks that
verify the adjoint solver against the forward solver:

- `ValidateLinearizedForward` — finite-difference vs. linearized-tangent.
- `ValidateLinearizedForwardAdjoint` — inner-product duality identity.
- `ValidateAdjointGradient` — adjoint gradient vs. finite-difference gradient.

Each opens a figure and plots error vs. perturbation magnitude on log-log
axes; you should see slope ≈ 1 down to roundoff.

## License

Add the license that applies to this release (e.g. MIT or BSD-3-Clause)
before publishing.
