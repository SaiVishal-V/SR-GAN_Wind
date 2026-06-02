"""Gradient RMSE/MAE evolution and gradient magnitude histograms."""
import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", default="outputs/plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.metrics) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    grad_rmse = [h.get("gradient_rmse", None) for h in history]
    grad_mae = [h.get("gradient_mae", None) for h in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    valid_rmse = [(e, v) for e, v in zip(epochs, grad_rmse) if v is not None]
    valid_mae = [(e, v) for e, v in zip(epochs, grad_mae) if v is not None]
    if valid_rmse:
        es, vs = zip(*valid_rmse)
        ax.plot(es, vs, "b-", linewidth=2, label="Gradient RMSE")
    if valid_mae:
        es, vs = zip(*valid_mae)
        ax.plot(es, vs, "r--", linewidth=2, label="Gradient MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Error")
    ax.set_title("Gradient Preservation Metrics", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(args.output_dir, "gradient_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
