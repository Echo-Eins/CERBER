"""
CEBCM Base Configuration.
All hyperparameters in one place as dataclasses.
"""

from dataclasses import dataclass, field


@dataclass
class SONARConfig:
    """SONAR encoder/decoder configuration."""
    encoder_name: str = "text_sonar_basic_encoder"
    decoder_name: str = "text_sonar_basic_decoder"
    tokenizer_name: str = "text_sonar_basic_encoder"
    embedding_dim: int = 1024
    default_lang: str = "eng_Latn"
    encode_batch_size: int = 64
    device: str = "cuda"  # "cuda" or "cpu"


@dataclass
class EBTConfig:
    """Energy-Based Transformer configuration."""
    dim: int = 1024
    hidden_dim: int = 2048
    # Pairwise Head
    pairwise_input_dim: int = 4096  # [q; c; q-c; q*c]
    # Chain Head
    chain_n_heads: int = 8
    chain_n_layers: int = 2
    chain_max_len: int = 20


@dataclass
class LangevinConfig:
    """Langevin Dynamics configuration."""
    lr: float = 0.01
    noise_scale: float = 0.005
    max_steps: int = 100
    energy_threshold: float = 0.5
    plateau_patience: int = 10
    plateau_delta: float = 1e-4
    cruise_ratio: float = 0.0  # 0.0 = pure Langevin, 0.7 = aggressive inertia
    momentum_beta: float = 0.9
    target_norm: float | None = None  # set from data statistics
    # Method selection: "overdamped", "pid", "underdamped"
    method: str = "pid"
    # PID-specific (arXiv:2511.12603)
    pid_kp: float = 1.0
    pid_ki: float = 0.3
    pid_kd: float = 0.1
    pid_integral_decay: float = 0.95
    # Underdamped-specific (GAUL, SIAM JUQ 2025)
    underdamped_friction: float = 0.5
    underdamped_mass: float = 1.0


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    lr: float = 5e-5  # Lowered from 1e-4 for orthonorm + learnable activations
    weight_decay: float = 0.01
    batch_size: int = 256
    num_negatives: int = 31
    temperature: float = 0.07
    total_epochs: int = 100
    gradient_penalty_lambda: float = 0.1
    # Architecture
    norm_mode: str = "orthonorm"    # "orthonorm" or "spectral_norm"
    activation: str = "groupsort"  # "groupsort", "lipschitz_spline", or "relu"
    # Curriculum
    easy_epochs_pct: float = 0.2
    medium_epochs_pct: float = 0.3
    hard_epochs_pct: float = 0.5
    # Focal-InfoNCE (for Stage 2+)
    focal_gamma: float = 2.0  # 0 = standard InfoNCE, 2 = strong focal


@dataclass
class Stage0Config:
    """Stage 0: SONAR validation experiment config."""
    sonar: SONARConfig = field(default_factory=SONARConfig)
    # Noise robustness (relative: 0.05 = 5% of embedding norm)
    noise_scales: list[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    )
    # Interpolation
    interpolation_steps: list[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    # Distribution analysis
    num_sentences: int = 1000
    # Output
    output_dir: str = "experiments/00_sonar_validation"


@dataclass
class Stage1Config:
    """Stage 1: Denoising PoC configuration."""
    sonar: SONARConfig = field(default_factory=SONARConfig)
    langevin: LangevinConfig = field(default_factory=lambda: LangevinConfig(
        lr=0.01,             # Increased: ∇E is O(1) with direction matching
        noise_scale=0.005,   # Small Langevin noise for exploration
        max_steps=100,
        target_norm=0.2051,
        method="pid",
        pid_kp=1.0,
        pid_ki=0.3,
        pid_kd=0.1,
        pid_integral_decay=0.95,
    ))
    # SimpleEnergy architecture
    energy_dim: int = 1024
    energy_hidden_dims: list[int] = field(
        default_factory=lambda: [2048, 1024, 512]  # Deeper: +1 layer for expressiveness
    )
    # Architecture choices
    norm_mode: str = "orthonorm"       # "orthonorm" (default) or "spectral_norm"
    activation: str = "groupsort"      # "groupsort" (default), "lipschitz_spline", "relu"
    ortho_n_iters: int = 15            # Bjorck iterations
    groupsort_size: int = 2            # Group size (2 = MaxMin)
    spline_num_knots: int = 4          # Knots for lipschitz_spline
    # Training
    lr: float = 1e-3  # Raised: cosine loss has O(1) gradients (was 5e-5 for MSE)
    warmup_steps: int = 500            # Linear warmup from 0 to lr
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 50
    # MDSM (Multi-Scale Denoising Score Matching) — replaces margin contrastive
    loss_type: str = "mdsm"            # "mdsm" (default) or "margin_contrastive" (legacy)
    mdsm_sigma_min: float = 0.01       # Minimum noise scale (fine structure)
    mdsm_sigma_max: float = 0.5        # Maximum noise scale (global structure)
    gradient_penalty_lambda: float = 0.0   # Disabled: orthonorm is already 1-Lipschitz
    margin: float = 1.0                # Only used with margin_contrastive loss
    # Noise for training negatives (only used with margin_contrastive loss)
    train_noise_scales: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3]
    )
    # Data
    num_train_sentences: int = 10000
    num_test_sentences: int = 500
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    # Evaluation
    eval_noise_scales: list[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.2, 0.3]
    )
    # Output
    output_dir: str = "experiments/01_denoising_poc"
    checkpoint_dir: str = "experiments/01_denoising_poc/checkpoints"
    # Logging
    wandb_project: str = "cebcm-stage1"
    use_wandb: bool = False
    log_every: int = 50
    eval_every_epoch: int = 5
