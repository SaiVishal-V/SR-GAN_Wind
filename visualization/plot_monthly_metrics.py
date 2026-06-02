"""Monthly metrics bar chart with error bars and 95% CI."""
import os, csv, argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to monthly_metrics.csv")
    parser.add_argument("--output-dir", default="outputs/plots")
    args = parser.parse_args()

    months, rmse_means, rmse_stds, ssim_means, ssim_stds = [], [], [], [], []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            months.append(row["month"])
            rmse_means.append(float(row["rmse_mps_mean"]))
            rmse_stds.append(float(row["rmse_mps_std"]))
            ssim_means.append(float(row["ssim_mean"]))
            ssim_stds.append(float(row["ssim_std"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Monthly Performance Metrics", fontsize=16, fontweight="bold")

    x = np.arange(len(months))
    ax1.bar(x, rmse_means, yerr=rmse_stds, capsize=4, color="steelblue", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(months, rotation=45, ha="right")
    ax1.set_ylabel("RMSE (m/s)")
    ax1.set_title("Monthly RMSE (mean ± std)")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, ssim_means, yerr=ssim_stds, capsize=4, color="seagreen", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(months, rotation=45, ha="right")
    ax2.set_ylabel("SSIM")
    ax2.set_title("Monthly SSIM (mean ± std)")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "monthly_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
