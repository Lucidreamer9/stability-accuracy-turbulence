import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# =============================================================================
# 1. Dataset
# =============================================================================
class TurbulenceDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt files in {data_dir}")
        print(f"Found {len(self.files)} samples.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])

        x0 = data["x0"].squeeze(0)
        xT = data["xT"].squeeze(0)
        t_end = float(data.get("T_end", 0.0))

        scale_factor = 20.0
        x0 = x0 / scale_factor
        xT = xT / scale_factor

        T_MAX = 1.0
        t_norm = t_end / T_MAX if T_MAX > 0 else 0.0
        t_norm = max(0.0, min(1.0, t_norm))
        t_channel = torch.full_like(xT, t_norm)

        inp = torch.cat([xT, t_channel], dim=0)
        return inp, x0, t_end

# =============================================================================
# 2. Model
# =============================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, n_channels=2, n_classes=1):
        super(SimpleUNet, self).__init__()
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=1),
        )
        self.conv1 = DoubleConv(256, 128)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=1),
        )
        self.conv2 = DoubleConv(128, 64)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=1),
        )
        self.conv3 = DoubleConv(64, 32)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        x = torch.cat([x3, x], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv3(x)
        return self.outc(x)

# =============================================================================
# 3. Training
# =============================================================================
def gradient_loss(pred, target):
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(dx_pred, dx_target) + F.l1_loss(dy_pred, dy_target)


def build_shell_binning(n, device):
    k = torch.fft.fftfreq(n, d=1.0 / n, device=device)
    kx, ky = torch.meshgrid(k, k, indexing="ij")
    kk = torch.sqrt(kx ** 2 + ky ** 2)
    k_max = n // 2
    bins = torch.round(kk).to(torch.int64)
    mask = bins < k_max
    mask_flat = mask.flatten()
    bins_valid = bins[mask].flatten()
    bin_counts = torch.bincount(bins_valid.cpu(), minlength=k_max).float().to(device)
    bin_counts = torch.clamp(bin_counts, min=1.0)
    inv_k_sq = 1.0 / (kk ** 2 + 1e-8)
    inv_k_sq[kk == 0] = 0.0
    inv_k_sq_flat = inv_k_sq[mask].flatten()
    k_vals = torch.arange(k_max, device=device, dtype=torch.float32)
    return bins_valid, bin_counts, inv_k_sq_flat, k_vals, mask_flat


def build_band_mask(n, device, k_min_ratio, k_max_ratio):
    k = torch.fft.fftfreq(n, d=1.0 / n, device=device)
    kx, ky = torch.meshgrid(k, k, indexing="ij")
    kk = torch.sqrt(kx ** 2 + ky ** 2)
    k_max = n // 2
    k_low = k_min_ratio * k_max
    k_high = k_max_ratio * k_max
    return ((kk >= k_low) & (kk <= k_high)).float()


def shell_energy_spectrum(field, bins_valid, bin_counts, inv_k_sq_flat, k_vals, mask_flat):
    field_hat = torch.fft.fft2(field.squeeze(1), norm="ortho")
    flat = field_hat.abs().pow(2).reshape(field.shape[0], -1)[:, mask_flat]
    energy = 0.5 * flat * inv_k_sq_flat

    spec = torch.zeros((field.shape[0], k_vals.numel()), device=field.device)
    index = bins_valid.unsqueeze(0).expand(field.shape[0], -1)
    spec.scatter_add_(1, index, energy)
    spec = spec / bin_counts
    return spec


def spectral_losses(
    pred,
    target,
    bins_valid,
    bin_counts,
    inv_k_sq_flat,
    k_vals,
    mask_flat,
    k_min_ratio=0.4,
    k_max_ratio=0.75,
    eps=1e-8,
):
    spec_pred = shell_energy_spectrum(pred, bins_valid, bin_counts, inv_k_sq_flat, k_vals, mask_flat)
    spec_target = shell_energy_spectrum(target, bins_valid, bin_counts, inv_k_sq_flat, k_vals, mask_flat)

    k_top = k_vals.numel() - 1
    k_low = max(1, int(k_min_ratio * k_top))
    k_high = max(k_low, int(k_max_ratio * k_top))
    k_high = min(k_high, k_top)
    k_mask = (k_vals >= k_low) & (k_vals <= k_high)

    pred_total = spec_pred[:, 1:].sum(dim=1, keepdim=True)
    target_total = spec_target[:, 1:].sum(dim=1, keepdim=True)
    p_pred = spec_pred / (pred_total + eps)
    p_target = spec_target / (target_total + eps)

    shape_loss = F.l1_loss(torch.log(p_pred[:, k_mask] + eps), torch.log(p_target[:, k_mask] + eps))

    pred_hi = spec_pred[:, k_mask].sum(dim=1)
    target_hi = spec_target[:, k_mask].sum(dim=1)
    pred_frac = pred_hi / (pred_total.squeeze(1) + eps)
    target_frac = target_hi / (target_total.squeeze(1) + eps)
    frac_loss = F.l1_loss(pred_frac, target_frac)

    energy_loss = F.l1_loss(torch.log1p(pred_total.squeeze(1)), torch.log1p(target_total.squeeze(1)))

    return shape_loss, frac_loss, energy_loss


def bandpass_loss(pred, target, k_mask):
    pred_hat = torch.fft.fft2(pred.squeeze(1), norm="ortho")
    target_hat = torch.fft.fft2(target.squeeze(1), norm="ortho")
    pred_bp = torch.fft.ifft2(pred_hat * k_mask, norm="ortho").real
    target_bp = torch.fft.ifft2(target_hat * k_mask, norm="ortho").real
    return F.l1_loss(pred_bp, target_bp)


def visualize_results(model, loader, device, epoch, save_dir="train_results_multiT_gradloss"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    inputs, targets, t_vals = next(iter(loader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        preds = model(inputs)

    img_in = inputs[0, 0].cpu().numpy()
    img_target = targets[0, 0].cpu().numpy()
    img_pred = preds[0, 0].cpu().numpy()
    t_val = float(t_vals[0]) if hasattr(t_vals, "__len__") else float(t_vals)

    vmin, vmax = -1.0, 1.0
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    ax = axs[0]
    im = ax.imshow(img_in, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title(f"Input (T={t_val:.3f})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axs[1]
    im = ax.imshow(img_pred, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title("AI Prediction (t=0)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axs[2]
    im = ax.imshow(img_target, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title("Ground Truth (t=0)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Epoch {epoch} Visualization")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch:03d}.png"))
    plt.close()
    model.train()


def main():
    DATA_DIR = "turbulence_dataset_multiT_with_initial"
    BATCH_SIZE = 32
    LR = 1e-3
    EPOCHS = 80
    LAMBDA_GRAD = 0.02
    LAMBDA_SPEC = 0.03
    LAMBDA_HIFRAC = 0.03
    LAMBDA_PHASE = 0.03
    LAMBDA_ENERGY = 0.05
    K_MIN_RATIO = 0.4
    K_MAX_RATIO = 0.75
    WARMUP_EPOCHS = 20
    ANNEAL_EPOCHS = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    dataset = TurbulenceDataset(DATA_DIR)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    sample_in, _, _ = dataset[0]
    n = sample_in.shape[-1]
    bins_valid, bin_counts, inv_k_sq_flat, k_vals, mask_flat = build_shell_binning(n, DEVICE)
    band_mask = build_band_mask(n, DEVICE, K_MIN_RATIO, K_MAX_RATIO)

    model = SimpleUNet(n_channels=2, n_classes=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("=== Training U-Net (Multi-T) with gradient + spectral losses ===")
    loss_history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_grad = 0.0
        epoch_shape = 0.0
        epoch_frac = 0.0
        epoch_phase = 0.0
        epoch_energy = 0.0

        if epoch <= WARMUP_EPOCHS:
            ramp = 0.0
        elif epoch <= WARMUP_EPOCHS + ANNEAL_EPOCHS:
            ramp = (epoch - WARMUP_EPOCHS) / ANNEAL_EPOCHS
        else:
            ramp = 1.0

        for data, target, _ in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss_mse = criterion(output, target)
            loss_grad = gradient_loss(output, target)
            loss_shape, loss_frac, loss_energy = spectral_losses(
                output,
                target,
                bins_valid,
                bin_counts,
                inv_k_sq_flat,
                k_vals,
                mask_flat,
                k_min_ratio=K_MIN_RATIO,
                k_max_ratio=K_MAX_RATIO,
            )
            loss_phase = bandpass_loss(output, target, band_mask)

            loss = (
                loss_mse
                + LAMBDA_GRAD * loss_grad
                + ramp * LAMBDA_SPEC * loss_shape
                + ramp * LAMBDA_HIFRAC * loss_frac
                + ramp * LAMBDA_PHASE * loss_phase
                + ramp * LAMBDA_ENERGY * loss_energy
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += loss_mse.item()
            epoch_grad += loss_grad.item()
            epoch_shape += loss_shape.item()
            epoch_frac += loss_frac.item()
            epoch_phase += loss_phase.item()
            epoch_energy += loss_energy.item()

        avg_loss = epoch_loss / len(train_loader)
        avg_mse = epoch_mse / len(train_loader)
        avg_grad = epoch_grad / len(train_loader)
        avg_shape = epoch_shape / len(train_loader)
        avg_frac = epoch_frac / len(train_loader)
        avg_phase = epoch_phase / len(train_loader)
        avg_energy = epoch_energy / len(train_loader)
        loss_history.append(avg_loss)
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Loss: {avg_loss:.6f} (MSE: {avg_mse:.6f}, Grad: {avg_grad:.6f}, "
            f"SpecShape: {avg_shape:.6f}, HiFrac: {avg_frac:.6f}, "
            f"Phase: {avg_phase:.6f}, Energy: {avg_energy:.6f}, Ramp: {ramp:.2f})"
        )

        if epoch % 5 == 0 or epoch == 1:
            visualize_results(model, val_loader, DEVICE, epoch)

    torch.save(model.state_dict(), "unet_turbulence_inverse_multiT_gradloss_with_initial.pth")
    print("Model saved: unet_turbulence_inverse_multiT_gradloss_with_initial.pth")

    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss (MSE + Grad + SpecShape + HiFrac + Phase + Energy)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss_curve_multiT_gradloss.png")
    print("Training complete. Check train_results_multiT_gradloss/")


if __name__ == "__main__":
    main()
