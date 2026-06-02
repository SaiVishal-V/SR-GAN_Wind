"""Best/Median/Worst case 5-panel prediction comparison."""
import os, sys, argparse
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import compute_all_metrics, denormalize, NORM_MEAN, NORM_STD
from datasets.wind_dataset import WindSRDataset, get_temporal_split
from models.generator import SRResNet
from utils.checkpoint import load_checkpoint


def plot_5panel(lr, sr, hr, mask, title, save_path, vmin=0, vmax=20):
    """5-panel: LR, SR, GT, Error Map, Gradient Map."""
    sr_mps = denormalize(sr).squeeze().numpy()
    hr_mps = denormalize(hr).squeeze().numpy()
    lr_mps = denormalize(lr).squeeze().numpy()
    mask_np = mask.squeeze().numpy()

    error = np.abs(sr_mps - hr_mps) * mask_np
    error[mask_np == 0] = np.nan

    sr_mps[mask_np == 0] = np.nan
    hr_mps[mask_np == 0] = np.nan

    # Gradient magnitude
    sr_gy = np.diff(sr_mps, axis=0)
    sr_gx = np.diff(sr_mps, axis=1)
    hr_gy = np.diff(hr_mps, axis=0)
    hr_gx = np.diff(hr_mps, axis=1)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].imshow(lr_mps, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")
    axes[0].set_title("LR Input")

    axes[1].imshow(sr_mps, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")
    axes[1].set_title("SR Output")

    axes[2].imshow(hr_mps, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")
    axes[2].set_title("Ground Truth")

    im3 = axes[3].imshow(error, cmap="hot", vmin=0, vmax=3, origin="upper")
    axes[3].set_title("Error (m/s)")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    grad_err = np.sqrt(sr_gy[:-1,:]**2 + sr_gx[:,:-1]**2)[:sr_gy.shape[0]-1, :sr_gx.shape[1]-1]
    im4 = axes[4].imshow(grad_err, cmap="inferno", origin="upper")
    axes[4].set_title("SR Gradient Magnitude")
    plt.colorbar(im4, ax=axes[4], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/plots")
    parser.add_argument("--colab", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_cfg = config["model"]["generator"]
    generator = SRResNet(
        in_channels=config["model"]["in_channels"],
        num_features=gen_cfg["num_features"],
        num_residual_blocks=gen_cfg["num_residual_blocks"],
        scale_factor=config["model"]["scale_factor"],
    ).to(device)

    ckpt = load_checkpoint(args.checkpoint, device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    generator.eval()

    data_path = config["data"]["colab_path"] if args.colab else config["data"]["local_path"]
    import netCDF4
    ds = netCDF4.Dataset(data_path, "r")
    n_time = ds.variables[config["data"]["lr_variable"]].shape[0]
    ds.close()

    _, _, test_idx = get_temporal_split(n_time, 0.70, 0.15, 0.15)
    test_ds = WindSRDataset(nc_path=data_path, time_indices=test_idx, mode="full", cache=True)

    # Evaluate all test scenes
    rmses = []
    for i in range(len(test_ds)):
        s = test_ds[i]
        with torch.no_grad():
            sr = generator(s["lr"].unsqueeze(0).to(device))
        m = compute_all_metrics(sr.cpu(), s["hr"].unsqueeze(0), s["mask"].unsqueeze(0))
        rmses.append((i, m["rmse"]))

    rmses.sort(key=lambda x: x[1])
    best_i = rmses[0][0]
    median_i = rmses[len(rmses)//2][0]
    worst_i = rmses[-1][0]

    os.makedirs(args.output_dir, exist_ok=True)
    for idx, label in [(best_i, "best"), (median_i, "median"), (worst_i, "worst")]:
        s = test_ds[idx]
        with torch.no_grad():
            sr = generator(s["lr"].unsqueeze(0).to(device)).cpu().squeeze(0)
        m = compute_all_metrics(sr.unsqueeze(0), s["hr"].unsqueeze(0), s["mask"].unsqueeze(0))
        plot_5panel(s["lr"], sr, s["hr"], s["mask"],
                    f"{label.upper()} Case (RMSE={m['rmse']:.4f}, SSIM={m['ssim']:.4f})",
                    os.path.join(args.output_dir, f"prediction_{label}.png"))

    test_ds.close()


if __name__ == "__main__":
    main()
