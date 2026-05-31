# Turbulent-flow inversion: data generation, training, and figure code

Reference implementation for the paper

> *A stability–accuracy trade-off in turbulent-flow inversion.*

This release contains:

1. **Data generation.** A 2D incompressible Navier–Stokes finite-difference
   solver and a script that uses it to generate the multi-horizon
   $(\omega_0,\omega_T)$ dataset.
2. **Model training.** A U-Net architecture with a composite loss
   (MSE + finite-difference gradient + shell-spectrum shape + high-band
   fraction + band-pass phase + total-energy) that learns the inverse map
   $\omega_T\mapsto\omega_0$.
3. **Evaluation and figure scripts.** The scripts that produce the
   trained-model outputs and the figure panels in the paper.

The **physics-based (data-assimilation) baseline** is not part of this
release — that pipeline is maintained separately by a collaborator and will
be linked from this repository once it is public. Figures that compare the
two methods read DA outputs from disk in a specific layout (see below); you
can either wait for the DA release or supply DA outputs that match the
expected format.

---

## ⚠ Hard-coded paths

Most scripts in this release embed **absolute paths** to the original
authors' working directories (e.g. `/export/yexiaohe/sdsu/ge`,
`/export/yexiaohe/sdsu/DA_DHIT_pngs_and_obs`). They are clearly visible at
the top of each script — either as `ROOT = Path("...")` constants or as
`argparse` defaults. Before running anything in step 3 below, **open each
script and edit the path constants** to point at your local copy.

We deliberately kept the paths as-is to make the original layout reproducible;
patching them with sentinels would obscure the data-flow that the paper
describes.


`experiments/` scripts `from data_gen import ...` and
`from train_u_multiT_gradloss import ...`, so **run them from the project
root** (e.g. `python experiments/iso_terminal_perturbation.py ...`) so the
top-level modules are on `sys.path`.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable PyTorch build is recommended for training; CPU works but is
slow.

---

## 1. Generate the training dataset

```bash
python data_gen_multiT.py
```

Produces `turbulence_dataset_multiT/sim_<seed>_T<t_id>.pt` and
`dataset_preview_multiT.png`. Edit the constants at the top of
`data_gen_multiT.py` to change configuration:

| Parameter | Default | Meaning |
|---|---|---|
| `NUM_SEEDS` | `10000` | random seeds attempted |
| `SAVE_DIR` | `turbulence_dataset_multiT` | output directory |
| `N` | `64` | grid resolution (periodic square) |
| `Lx, Ly` | `1.0, 1.0` | domain extent |
| `Re` | `1000.0` | Reynolds number |
| `dt` | `1e-3` | time step |
| `T_LIST` | `[0.2, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]` | horizons sampled |
| `WARMUP_STEPS` | `20` | warm-up steps before $\omega_0$ is recorded |

Each `.pt` sample stores `x0`, `xT`, `u0`, `v0`, and metadata
(`T_end`, `dt`, `Re`, `seed_id`, ...).

Train/test split used in the paper: 100 seeds for test, the rest for
train/val. Generate the full set first, then split by seed range yourself.

## 2. Train the inverse model

```bash
python train_u_multiT_gradloss.py
```

Default `DATA_DIR` is `turbulence_dataset_multiT_with_initial` — edit
`main()` to point at the directory created in step 1. Hyperparameters:

| Parameter | Value |
|---|---|
| `BATCH_SIZE` | `32` |
| `LR` | `1e-3` (Adam) |
| `EPOCHS` | `80` |
| `LAMBDA_GRAD` | `0.02` |
| `LAMBDA_SPEC` | `0.03` |
| `LAMBDA_HIFRAC` | `0.03` |
| `LAMBDA_PHASE` | `0.03` |
| `LAMBDA_ENERGY` | `0.05` |
| `K_MIN_RATIO`, `K_MAX_RATIO` | `0.4, 0.75` |
| `WARMUP_EPOCHS / ANNEAL_EPOCHS` | `20 / 20` |

Trained weights are saved to
`unet_turbulence_inverse_multiT_gradloss_with_initial.pth`. Per-epoch
validation snapshots are written to `train_results_multiT_gradloss/`.

Network (`SimpleUNet`): circular-padded $3{\times}3$ convolutions, encoder
channels $32{,}64{,}128{,}256$, bilinear upsampling with skip connections,
batch normalization, ReLU. Input: $(\omega_T/s, \tau)$ with $s=20$ and a
constant time channel encoding the horizon. Output: $\hat\omega_0/s$.

## 3. Reproduce the paper figures

### 3.0 Evaluate the trained model on the test set

```bash
python evaluate_reverse_turbulence.py \
  --model_path  unet_turbulence_inverse_multiT_gradloss_with_initial.pth \
  --test_dir    <your-test-data-dir> \
  --out_dir     eval_outputs_phy \
  --sample_T    0.2 0.5 0.8 1.2 1.6 2.0
```

This writes per-horizon `forward_trajectory_data_T_<t>_ex<i>.npz` files
consumed by the figure scripts below. Inspect the script header for the
full CLI surface.

### 3.1 Fig. 3 — Data-driven inversion gives stable initial-state recovery

| Panel | Script | Output |
|---|---|---|
| 3a | `make_results_31_initial_compare.py` | `eval_outputs/initial_error_NN_DA_full.png` |
| 3b | `make_results_31_spectrum_error.py` | `eval_outputs/spectrum_error_NN_DA_full.png` |
| 3c | `make_manuscript_31_figures.py` | `eval_outputs/sample_reconstructions_NN_DA_combined.png` |

3a and 3b additionally read DA outputs (see below); 3c also reads DA for
the right-hand columns of the side-by-side comparison.

### 3.2 Fig. 4 — Physics-based reconstruction favors forward consistency

| Panel | Script | Output |
|---|---|---|
| 4a | `make_results_32_forward_terminal_error.py` | `eval_outputs/da_dl_compare/forward_terminal_error_NN_DA.png` |
| 4b | `make_results_32_forward_terminal_error.py` | `eval_outputs/da_dl_compare/forward_terminal_spectrum_error_NN_DA.png` |
| 4c | `plot_da_dl_with_error_curve.py` | `eval_outputs/da_dl_compare/compare_forward_T_2_ex1.png` |

Both scripts compare NN and DA terminal-image quality and require the DA
outputs.

### 3.3 Fig. 5 — Adversarial probe of data-manifold restriction

```bash
python experiments/iso_terminal_perturbation.py \
  --ckpt       unet_turbulence_inverse_multiT_gradloss_with_initial.pth \
  --data-dir   <your-test-data-dir> \
  --T          1.0 \
  --num-samples 8 \
  --epsilons   0,0.2,0.5,1.0,2.0 \
  --output-dir results/iso_terminal_perturbation
```

Produces all four Fig. 5 panels:

| Panel | Output (under `--output-dir/<tag>`) |
|---|---|
| 5a | `visual_grid_sim_*.png` |
| 5b | `epsilon_sweep.png` |
| 5c | `spectral_fingerprint_*_spectrum.png` |
| 5d | `spectral_fingerprint_*_energy_kstar.png` |

It also writes per-sample CSVs and a `summary.json`. The script uses
`experiments/finite_time_singular_direction_filtering.py` for the
projected forward Jacobian, the smallest-singular-direction search, and
the neural-inverse wrapper.

---

