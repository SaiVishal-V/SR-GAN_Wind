"""Validation metrics over epochs: RMSE, SSIM, PSNR, correlation."""
import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_metrics(metrics_path, output_dir="outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)
    with open(metrics_path) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Validation Metrics", fontsize=16, fontweight="bold")

    configs = [
        ("rmse", "RMSE (normalized)", "tab:red", False),
        ("ssim", "SSIM", "tab:green", True),
        ("psnr", "PSNR (dB)", "tab:blue", True),
        ("correlation", "Correlation", "tab:purple", True),
    ]

    for ax, (key, title, color, higher_better) in zip(axes.flat, configs):
        vals = [h.get(key, None) for h in history]
        ema_vals = [h.get(f"ema_{key}", None) for h in history]
        valid = [(e, v) for e, v in zip(epochs, vals) if v is not None]
        if valid:
            es, vs = zip(*valid)
            ax.plot(es, vs, color=color, alpha=0.6, linewidth=1, label="Standard")
        valid_ema = [(e, v) for e, v in zip(epochs, ema_vals) if v is not None]
        if valid_ema:
            es, vs = zip(*valid_ema)
            ax.plot(es, vs, color=color, linewidth=2, linestyle="--", label="EMA")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "validation_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", default="outputs/plots")
    args = parser.parse_args()
    plot_metrics(args.metrics, args.output_dir)
