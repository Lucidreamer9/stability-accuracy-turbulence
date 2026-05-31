import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, desc=None: x

from data_gen import FiniteDifferenceNSolver, generate_initial_condition

warnings.filterwarnings("ignore")


def build_time_steps(t_list, dt):
    steps = {}
    for t in t_list:
        step = int(round(t / dt))
        if step <= 0:
            continue
        steps[step] = step * dt
    return dict(sorted(steps.items()))


def main():
    # Config
    NUM_SEEDS = 10000
    SAVE_DIR = "turbulence_dataset_multiT"

    N = 64
    Lx, Ly = 1.0, 1.0
    Re = 1000.0
    dt = 1e-3

    T_LIST = [0.2, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    WARMUP_STEPS = 20

    time_steps = build_time_steps(T_LIST, dt)
    if not time_steps:
        print("No valid T in T_LIST.")
        return

    max_steps = max(time_steps.keys())
    target_steps = list(time_steps.keys())
    target_set = set(target_steps)

    os.makedirs(SAVE_DIR, exist_ok=True)
    solver = FiniteDifferenceNSolver(n=N, dx=Lx / N, Re=Re, dt=dt, Lx=Lx, Ly=Ly)

    print("=== Multi-T dataset generator ===")
    print(f"Seeds: {NUM_SEEDS}")
    print(f"T targets: {sorted(time_steps.values())}")
    print(f"Max steps: {max_steps}")

    total_saved = 0
    seeds_with_any = 0

    pbar = tqdm(range(NUM_SEEDS), desc="Generating")
    for seed in pbar:
        try:
            np.random.seed(seed)
            u, v = generate_initial_condition(solver, N, Lx, Ly)

            for _ in range(WARMUP_STEPS):
                u, v = solver.forward_step(u, v)

            x0_omega = solver.converter.vorticity_from_velocity(u, v)
            std_val = np.std(x0_omega)
            if std_val < 1e-6:
                continue

            norm_factor = 10.0 / std_val
            u *= norm_factor
            v *= norm_factor
            x0_omega *= norm_factor

            # Save normalized initial velocity for forward-consistency checks
            u0 = u.copy()
            v0 = v.copy()
            u0_tensor = torch.from_numpy(u0).float().unsqueeze(0).unsqueeze(0)
            v0_tensor = torch.from_numpy(v0).float().unsqueeze(0).unsqueeze(0)

            saved_this_seed = 0

            for step in range(1, max_steps + 1):
                u, v = solver.forward_step(u, v)

                if step not in target_set:
                    continue

                xT_omega = solver.converter.vorticity_from_velocity(u, v)
                if np.isnan(np.sum(xT_omega)) or np.max(np.abs(xT_omega)) > 1e6:
                    continue

                t_actual = time_steps[step]
                t_id = int(round(t_actual * 1000))

                save_name = f"sim_{seed:05d}_T{t_id:04d}.pt"
                save_path = os.path.join(SAVE_DIR, save_name)

                x0_tensor = torch.from_numpy(x0_omega).float().unsqueeze(0).unsqueeze(0)
                xT_tensor = torch.from_numpy(xT_omega).float().unsqueeze(0).unsqueeze(0)

                torch.save(
                    {
                        "x0": x0_tensor,
                        "xT": xT_tensor,
                        "u0": u0_tensor,
                        "v0": v0_tensor,
                        "T_end": float(t_actual),
                        "t_id": int(t_id),
                        "dt": float(dt),
                        "steps": int(step),
                        "Re": float(Re),
                        "seed_id": int(seed),
                        "norm_factor": float(norm_factor),
                    },
                    save_path,
                )

                saved_this_seed += 1
                total_saved += 1

                if seed == 0 and saved_this_seed == 1:
                    limit = np.max(np.abs(x0_omega))
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(x0_omega, cmap="RdBu_r", origin="lower", vmin=-limit, vmax=limit)
                    plt.colorbar()
                    plt.title("Target x0 (T=0)")
                    plt.subplot(1, 2, 2)
                    plt.imshow(xT_omega, cmap="RdBu_r", origin="lower", vmin=-limit, vmax=limit)
                    plt.colorbar()
                    plt.title(f"Input xT (T={t_actual:.3f})")
                    plt.tight_layout()
                    plt.savefig("dataset_preview_multiT.png")
                    plt.close()

            if saved_this_seed > 0:
                seeds_with_any += 1

        except Exception as e:
            print(f"Error on seed {seed}: {e}")
            continue

    print("\nDone.")
    print(f"Seeds processed: {NUM_SEEDS}")
    print(f"Seeds with any saved samples: {seeds_with_any}")
    print(f"Total saved files: {total_saved}")
    print(f"Output dir: {SAVE_DIR}")


if __name__ == "__main__":
    main()
