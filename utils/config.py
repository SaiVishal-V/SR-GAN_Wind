"""
Configuration loader for WindGapGAN.

Loads YAML configs and supports CLI overrides via dot-notation:
    python train.py --config configs/default.yaml --data.nc_path /path/to/data.nc
"""

from __future__ import annotations

import argparse
import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

logger = logging.getLogger(__name__)


# ── Nested Config Dataclasses ──────────────────────────────────────────────


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    nc_path: Optional[str] = None
    target_variable: Optional[str] = None
    time_dim: Optional[str] = None
    lat_dim: Optional[str] = None
    lon_dim: Optional[str] = None
    sequence_length: int = 5
    patch_size: int = 64
    stride: int = 32
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    num_workers: int = 4
    normalize: bool = True
    norm_method: str = "auto"
    mask_variable: Optional[str] = None
    missing_values: Optional[list[float]] = None
    synthetic_mask_strategy: str = "mixed"
    synthetic_mask_ratio: float = 0.3


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    name: str = "unet"
    in_channels: int = 2
    out_channels: int = 1
    base_features: int = 32
    depth: int = 4
    dropout: float = 0.1
    use_batch_norm: bool = True
    convlstm_hidden: int = 64
    convlstm_layers: int = 2
    convlstm_kernel_size: int = 3
    discriminator_type: str = "spatial"
    disc_base_features: int = 64
    disc_depth: int = 3
    use_spectral_norm: bool = True
    conditional: bool = True


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    epochs: int = 200
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    early_stopping_patience: int = 30
    lr_generator: float = 1e-4
    lr_discriminator: float = 4e-4
    adversarial_weight: float = 0.01
    adversarial_ramp_epochs: int = 20
    gradient_loss_weight: float = 0.0


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""

    save_every: int = 50
    save_best_rmse: bool = True
    save_best_mae: bool = True
    save_best_corr: bool = True
    checkpoint_dir: str = "checkpoints"


@dataclass
class LoggingConfig:
    """Logging and experiment tracking configuration."""

    use_wandb: bool = False
    use_tensorboard: bool = True
    wandb_project: str = "windgapgan"
    wandb_entity: Optional[str] = None
    log_dir: str = "outputs/logs"
    log_every_n_steps: int = 10


@dataclass
class OutputConfig:
    """Output configuration."""

    output_dir: str = "outputs"
    save_predictions_netcdf: bool = True
    save_visualizations: bool = True
    save_metrics_csv: bool = True
    save_experiment_summary: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    run_classical_baselines: bool = True
    wind_speed_regimes: list[list[Optional[float]]] = field(
        default_factory=lambda: [[0, 5], [5, 10], [10, 15], [15, None]]
    )


@dataclass
class Config:
    """Master configuration for WindGapGAN."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 42
    device: str = "auto"

    def resolve_device(self) -> torch.device:
        """Resolve 'auto' to the best available device."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(self.device)


# ── Config Loading Utilities ───────────────────────────────────────────────


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively update base dict with override dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dot notation: 'data.nc_path' → d['data']['nc_path']."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    # Attempt type coercion
    raw = value
    if isinstance(raw, str):
        if raw.lower() == "true":
            raw = True
        elif raw.lower() == "false":
            raw = False
        elif raw.lower() == "null" or raw.lower() == "none":
            raw = None
        else:
            try:
                raw = int(raw)
            except ValueError:
                try:
                    raw = float(raw)
                except ValueError:
                    pass
    d[keys[-1]] = raw


def _dict_to_dataclass(cls: type, d: dict) -> Any:
    """Recursively convert a dict to a dataclass instance, ignoring unknown keys."""
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        return d

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for fname, ftype in field_types.items():
        if fname not in d:
            continue
        val = d[fname]
        # Resolve string type annotations to actual types
        if isinstance(ftype, str):
            # Handle forward references
            ftype_resolved = eval(ftype, {**globals(), "Optional": Optional})
        else:
            ftype_resolved = ftype

        if dataclasses.is_dataclass(ftype_resolved) and isinstance(val, dict):
            kwargs[fname] = _dict_to_dataclass(ftype_resolved, val)
        else:
            kwargs[fname] = val
    return cls(**kwargs)


def load_config(config_path: str | Path, cli_overrides: list[str] | None = None) -> Config:
    """
    Load configuration from a YAML file with optional CLI overrides.

    Args:
        config_path: Path to the YAML config file.
        cli_overrides: List of CLI override strings like ['--data.nc_path=/path/to/file'].

    Returns:
        Fully resolved Config dataclass.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # Apply CLI overrides
    if cli_overrides:
        for override in cli_overrides:
            if override.startswith("--"):
                override = override[2:]
            if "=" in override:
                key, value = override.split("=", 1)
            else:
                # Bare flag → treat as True
                key, value = override, "true"
            _set_nested(raw, key, value)

    # Convert to dataclass
    config = _dict_to_dataclass(Config, raw)

    logger.info("Configuration loaded from %s", config_path)
    return config


def parse_args_and_load_config() -> Config:
    """Parse command-line arguments and load configuration."""
    parser = argparse.ArgumentParser(
        description="WindGapGAN — Spatio-Temporal Gap Filling",
        add_help=True,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    # Capture all unknown args as CLI overrides
    args, unknown = parser.parse_known_args()

    config = load_config(args.config, cli_overrides=unknown)
    return config


def save_config(config: Config, save_path: str | Path) -> None:
    """Save configuration to a YAML file for reproducibility."""
    import dataclasses

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    def to_dict(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj):
            return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        elif isinstance(obj, list):
            return [to_dict(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    with open(save_path, "w") as f:
        yaml.dump(to_dict(config), f, default_flow_style=False, sort_keys=False)

    logger.info("Configuration saved to %s", save_path)
