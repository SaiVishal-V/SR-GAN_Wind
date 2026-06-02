"""
Ablation experiment runner.

Each experiment changes exactly ONE variable from baseline.
Auto-creates experiment directory with config_used.yaml and metrics.

Usage:
    python run_ablation.py --experiment E2 --config configs/config.yaml
    python run_ablation.py --experiment E3 --config configs/config.yaml --colab
"""
import argparse, copy, os, sys, yaml

sys.path.insert(0, os.path.dirname(__file__))

# Experiment definitions: each maps to ONE config override
EXPERIMENTS = {
    "E1": {"name": "baseline", "overrides": {}},
    "E2": {"name": "ema", "overrides": {"optimization.ema_enabled": True}},
    "E3": {"name": "laplacian_loss", "overrides": {"loss.laplacian_weight": 0.05}},
    "E4": {"name": "power_spectrum_eval", "overrides": {}},  # eval-only
    "E5": {"name": "spectral_loss", "overrides": {"loss.spectral_weight": 0.001}},
    "E6": {"name": "charbonnier", "overrides": {"loss.pixel_loss_type": "charbonnier"}},
    "E7": {"name": "warm_restarts", "overrides": {"training.scheduler.type": "warm_restarts"}},
    "E8": {"name": "sn_only_disc", "overrides": {"model.discriminator.use_instance_norm": False}},
}


def set_nested(d, key_path, value):
    """Set a nested dict value using dot notation."""
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def main():
    parser = argparse.ArgumentParser(description="Ablation Experiment Runner")
    parser.add_argument("--experiment", type=str, required=True, choices=list(EXPERIMENTS.keys()))
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--colab", action="store_true")
    parser.add_argument("--stage", type=str, default="all", choices=["pretrain", "gan", "all"])
    args = parser.parse_args()

    exp = EXPERIMENTS[args.experiment]
    exp_name = f"{args.experiment}_{exp['name']}"

    # Load and modify config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    for key_path, value in exp["overrides"].items():
        set_nested(config, key_path, value)

    config["experiment"]["name"] = exp_name

    # Create experiment directory
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    config["checkpoint"]["save_dir"] = os.path.join(exp_dir, "checkpoints")
    config["logging"]["tensorboard_dir"] = os.path.join(exp_dir, "tensorboard")

    # Save config used
    config_path = os.path.join(exp_dir, "config_used.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"{'='*60}")
    print(f"ABLATION EXPERIMENT: {exp_name}")
    print(f"{'='*60}")
    print(f"  Changed: {exp['overrides'] or 'None (baseline)'}")
    print(f"  Config:  {config_path}")
    print(f"  Output:  {exp_dir}")

    # Write modified config and run training
    temp_config = os.path.join(exp_dir, "config_used.yaml")
    train_cmd = [
        sys.executable, "training/train.py",
        "--config", temp_config,
        "--stage", args.stage,
    ]
    if args.colab:
        train_cmd.append("--colab")

    print(f"\n  Running: {' '.join(train_cmd)}")
    os.execv(sys.executable, train_cmd)


if __name__ == "__main__":
    main()
