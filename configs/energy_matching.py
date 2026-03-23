"""
Configuration for Energy Matching training pipeline.

Spec reference: tasks/todo.md §1, §10.5
"""

from dataclasses import dataclass, field

from configs.base import LangevinConfig


@dataclass
class EnergyMatchingConfig:
    """Energy Matching training configuration."""

    # ── Model architecture ──
    energy_dim: int = 1024
    energy_hidden_dims: list[int] = field(
        default_factory=lambda: [2048, 1024, 512]
    )
    norm_mode: str = "orthonorm"       # "orthonorm", "spectral_norm", "none"
    activation: str = "groupsort"      # "groupsort", "lipschitz_spline", "relu"
    ortho_n_iters: int = 15
    groupsort_size: int = 2
    spline_num_knots: int = 4

    # ── Training ──
    lr: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 64
    num_epochs: int = 100
    warmup_steps: int = 500

    # ── Loss function ──
    loss_type: str = "cosine_em"       # "energy_matching", "cosine_em", "weighted_em"
    magnitude_weight: float = 0.1      # Weight for magnitude term in cosine_em
    near_data_weight: float = 3.0      # Near-data emphasis for weighted_em
    t_min: float = 0.001              # Avoid exact noise (t=0)
    t_max: float = 0.999              # Avoid exact data (t=1)

    # ── Prior distribution ──
    # σ_prior = target_norm / √d ≈ 0.2051 / √1024 ≈ 0.00641
    prior_std: float = 0.00641
    target_norm: float = 0.2051        # Mean norm of SONAR embeddings

    # ── NCE warmstart ──
    nce_warmstart: bool = True
    nce_epochs: int = 10               # Epochs of NCE before switching to EM
    nce_loss_type: str = "simple"      # "full" (with log p_noise) or "simple"

    # ── Negative buffer ──
    buffer_size: int = 10000
    buffer_refresh_fraction: float = 0.05
    buffer_langevin_steps: int = 20
    buffer_langevin_lr: float = 0.01
    buffer_langevin_noise: float = 0.005
    buffer_seed_noise: float = 0.3     # Noise for data-seeded buffer init

    # ── Evaluation ──
    eval_every_epoch: int = 5
    eval_num_samples: int = 100
    eval_generate_samples: int = 500   # Number of samples to generate for eval
    eval_ode_steps: int = 100          # ODE steps for sample generation
    eval_noise_scales: list[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.2, 0.3]
    )

    # ── Langevin (for evaluation denoising) ──
    langevin: LangevinConfig = field(default_factory=lambda: LangevinConfig(
        lr=0.01,
        noise_scale=0.005,
        max_steps=100,
        target_norm=0.2051,
        method="pid",
    ))

    # ── Data ──
    num_train_sentences: int = 10000
    num_test_sentences: int = 500

    # ── Output ──
    output_dir: str = "experiments/02_energy_matching"
    checkpoint_dir: str = "experiments/02_energy_matching/checkpoints"

    # ── Logging ──
    wandb_project: str = "cebcm-energy-matching"
    use_wandb: bool = False
    log_every: int = 50
