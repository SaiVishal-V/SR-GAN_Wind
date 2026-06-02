"""Radially averaged power spectrum comparison."""
import os, sys, argparse
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datasets.wind_dataset import WindSRDataset, get_temporal_split
from models.generator import SRResNet
from utils.checkpoint import load_checkpoint
import torch.nn.functional as F


def radial_power_spectrum(field_2d):
    """Compute radially averaged power spectrum."""
    fft = np.fft.fft2(field_2d)
    power = np.abs(np.fft.fftshift(fft)) ** 2
    h, w = field_2d.shape
    cy, cx = h // 2, w // 2
    y, x = np.mgrid[:h, :w]
    r = np.sqrt((y - cy)**2 + (x - cx)**2).astype(int)
    max_r = min(cy, cx)
    profile = np.zeros(max_r)
    for ri in range(max_r):
        ring = power[r == ri]
        if len(ring) > 0:
            profile[ri] = ring.mean()
    return profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/plots")
    parser.add_argument("--colab", action="store_true")
    parser.add_argument("--n-scenes", type=int, default=10)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_cfg = config["model"]["generator"]
    generator = SRResNet(
        in_channels=config["model"]["in_channels"],
        num_features=gen_cfg["num_features"],
        num_residual_blocks=gen_cfg["num_residual_blocks"],
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
    test_ds = WindSRDataset(nc_path=data_path, time_indices=test_idx[:args.n_scenes],
                            mode="full", cache=True)

    gt_spectra, sr_spectra, bic_spectra = [], [], []
    for i in range(len(test_ds)):
        s = test_ds[i]
        mask = s["mask"].squeeze().numpy()
        hr = s["hr"].squeeze().numpy() * mask

        with torch.no_grad():
            sr = generator(s["lr"].unsqueeze(0).to(device)).cpu().squeeze().numpy() * mask

        bicubic = F.interpolate(s["lr"].unsqueeze(0), scale_factor=4,
                                mode="bicubic", align_corners=False).squeeze().numpy() * mask

        gt_spectra.append(radial_power_spectrum(hr))
        sr_spectra.append(radial_power_spectrum(sr))
        bic_spectra.append(radial_power_spectrum(bicubic))

    # Average spectra
    max_len = min(len(s) for s in gt_spectra)
    gt_avg = np.mean([s[:max_len] for s in gt_spectra], axis=0)
    sr_avg = np.mean([s[:max_len] for s in sr_spectra], axis=0)
    bic_avg = np.mean([s[:max_len] for s in bic_spectra], axis=0)

    freqs = np.arange(1, max_len)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(freqs, gt_avg[1:], "k-", linewidth=2, label="Ground Truth")
    ax.loglog(freqs, sr_avg[1:], "b-", linewidth=2, label="SR-GAN")
    ax.loglog(freqs, bic_avg[1:], "r--", linewidth=1.5, label="Bicubic")
    ax.set_xlabel("Spatial Frequency", fontsize=12)
    ax.set_ylabel("Power Spectral Density", fontsize=12)
    ax.set_title("Radially Averaged Power Spectrum Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "power_spectrum.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    test_ds.close()


if __name__ == "__main__":
    main()
