"""LR scheduler comparison visualization."""
import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/plots")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    epochs = args.epochs
    base_lr = args.lr

    # Simulate schedulers
    schedules = {}

    # Cosine
    param = [torch.nn.Parameter(torch.zeros(1))]
    opt = torch.optim.AdamW(param, lr=base_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    lrs = []
    for _ in range(epochs):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    schedules["Cosine Annealing"] = lrs

    # Warm Restarts
    opt2 = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt2, T_0=50, T_mult=2, eta_min=1e-6)
    lrs2 = []
    for _ in range(epochs):
        lrs2.append(opt2.param_groups[0]["lr"])
        opt2.step()
        sched2.step()
    schedules["Warm Restarts (T0=50, Tmult=2)"] = lrs2

    # OneCycleLR
    opt3 = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    sched3 = torch.optim.lr_scheduler.OneCycleLR(opt3, max_lr=base_lr*10, total_steps=epochs)
    lrs3 = []
    for _ in range(epochs):
        lrs3.append(opt3.param_groups[0]["lr"])
        opt3.step()
        sched3.step()
    schedules["OneCycleLR"] = lrs3

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, lrs in schedules.items():
        ax.plot(range(epochs), lrs, linewidth=2, label=name)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title("LR Scheduler Comparison", fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "scheduler_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
