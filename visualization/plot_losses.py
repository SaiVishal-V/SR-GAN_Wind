"""Training loss curves: G loss, D loss, gradient loss, Laplacian loss."""
import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_losses(metrics_path, output_dir="outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)
    with open(metrics_path) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Loss Curves", fontsize=16, fontweight="bold")

    loss_keys = [
        ("pixel_loss", "Pixel Loss (L1/Charbonnier)", "tab:blue"),
        ("gradient_loss", "Gradient Loss", "tab:orange"),
        ("adversarial_loss", "Adversarial Loss (G)", "tab:red"),
        ("d_loss", "Discriminator Loss", "tab:purple"),
    ]

    for ax, (key, title, color) in zip(axes.flat, loss_keys):
        vals = [h.get(key, None) for h in history]
        valid = [(e, v) for e, v in zip(epochs, vals) if v is not None]
        if valid:
            es, vs = zip(*valid)
            ax.plot(es, vs, color=color, alpha=0.3, linewidth=0.5)
            # Smoothed
            if len(vs) > 5:
                smooth = np.convolve(vs, np.ones(5)/5, mode="valid")
                ax.plot(es[2:-2], smooth, color=color, linewidth=2, label="Smoothed")
            ax.legend()
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="Path to metrics_history.json")
    parser.add_argument("--output-dir", default="outputs/plots")
    args = parser.parse_args()
    plot_losses(args.metrics, args.output_dir)
