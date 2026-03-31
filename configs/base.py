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
    energy_threshold: float | None = None
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
    # Tamed gradient (Benko et al., AAAI 2025): grad / (1 + lr*||grad||)
    # Safety net for non-Lipschitz architectures (norm_mode=none)
    tamed: bool = False
    # Adaptive sigma schedule (NCSN-style annealing during Langevin)
    sigma_anneal: bool = True
    sigma_anneal_mode: str = "hybrid"  # "geometric", "adaptive", "hybrid"
    sigma_anneal_max: float = 0.3
    sigma_anneal_min: float = 0.01
    sigma_anneal_blend: float = 0.5  # for hybrid mode
    # Optional synchronized noise annealing during inference:
    # co-anneal Langevin noise with sigma schedule to avoid semantic mismatch.
    noise_anneal: bool = True
    noise_anneal_mode: str = "hybrid"  # "geometric", "adaptive", "hybrid"
    noise_anneal_max: float = 0.15
    noise_anneal_min: float = 0.0002
    noise_anneal_sync_with_sigma: bool = True


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
    # Critic architecture choices
    norm_mode: str = "orthonorm"       # "orthonorm" (default) or "spectral_norm"
    activation: str = "groupsort"      # "groupsort" (default), "lipschitz_spline", "relu"
    ortho_n_iters: int = 8             # Bjorck iterations
    ortho_schedule_enabled: bool = True
    ortho_schedule_iters: list[int] = field(default_factory=lambda: [8, 4, 2])
    ortho_schedule_boundaries: list[float] = field(default_factory=lambda: [0.34, 0.67])
    groupsort_size: int = 2            # Group size (2 = MaxMin)
    spline_num_knots: int = 4          # Knots for lipschitz_spline
    # Actor architecture choices (for actor_critic mode)
    actor_hidden_dims: list[int] = field(default_factory=lambda: [2048, 1024, 512])
    actor_norm_mode: str = "spectral_norm"  # "orthonorm", "spectral_norm", "none"
    actor_activation: str = "silu"          # "silu", "gelu", "relu", "groupsort", "lipschitz_spline"
    actor_lr: float = 3e-4
    actor_weight_decay: float = 0.01
    actor_step_size: float = 0.7            # train-time one-step update scale
    actor_eval_step_size: float = 0.7       # eval-time iterative update scale
    actor_steps_per_sample: int = 1         # unrolled actor steps in training
    actor_eval_steps: int = 4               # iterative actor-only steps at eval
    actor_tangent_projection: bool = True

    # Training
    lr: float = 1e-3  # σ-conditioned NCSN + σ²-weighted DSM (was 5e-5 for unconditioned MSE)
    warmup_steps: int = 500            # Linear warmup from 0 to lr
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 50
    # Objectives
    loss_type: str = "actor_critic"    # "actor_critic", "mdsm", or "margin_contrastive"
    # MDSM (Multi-Scale Denoising Score Matching)
    mdsm_sigma_min: float = 0.01       # Minimum noise scale (fine structure)
    mdsm_sigma_max: float = 0.5        # Maximum noise scale (global structure)
    mdsm_sigma_sampling: str = "loguniform"  # "loguniform" or "edm"
    mdsm_sigma_weighting: str = "sigma2"  # "sigma2", "uniform", "inv_sigma2"
    mdsm_directional: bool = True
    mdsm_magnitude_aux_weight: float = 0.05
    mdsm_cosine_eps: float = 1e-4
    mdsm_norm_floor: float = 1e-4
    mdsm_tangent_projection: bool = True
    mdsm_sigma_curriculum: bool = True
    mdsm_sigma_curriculum_start_min: float = 0.1
    mdsm_edm_p_mean: float = -1.2
    mdsm_edm_p_std: float = 1.2
    # Actor+Critic coupling losses
    actor_loss_direction_weight: float = 0.7
    actor_loss_vector_weight: float = 0.3
    actor_loss_magnitude_weight: float = 0.1
    actor_energy_weight: float = 0.05
    critic_margin_clean_actor: float = 0.05
    critic_margin_actor_noisy: float = 0.10
    critic_margin_clean_noisy: float = 0.20
    critic_loss_weight: float = 1.0
    critic_eval_langevin_steps: int = 30
    actor_use_sigma_curriculum: bool = True
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
    # Systems optimization
    seed: int = 42
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 2
    enable_amp: bool = True
    amp_dtype: str = "bf16"  # "bf16" or "fp16"
    mdsm_force_fp32: bool = True  # second-order MDSM path is numerically fragile in bf16/fp16
    enable_compile: bool = False
    compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    energy_scale_lr_multiplier: float = 5.0
    skip_non_finite_batches: bool = True
    max_consecutive_non_finite_batches: int = 20
    non_finite_lr_backoff: float = 0.99
    non_finite_backoff_streak_trigger: int = 5  # only backoff on sustained streaks, not every NaN
    # Logging
    wandb_project: str = "cebcm-stage1"
    use_wandb: bool = False
    log_every: int = 50
    eval_every_epoch: int = 5


@dataclass
class Stage1_5Config:
    """Stage 1.5: hybrid conditional critic + actor proposal/refinement."""
    sonar: SONARConfig = field(default_factory=SONARConfig)
    langevin: LangevinConfig = field(default_factory=lambda: LangevinConfig(
        lr=0.001,
        noise_scale=0.0002,
        max_steps=100,
        target_norm=0.2051,
        method="pid",
        pid_kp=1.0,
        pid_ki=0.3,
        pid_kd=0.1,
    ))

    # Architecture
    energy_dim: int = 1024
    energy_hidden_dims: list[int] = field(default_factory=lambda: [2048, 1024, 512])
    actor_hidden_dims: list[int] = field(default_factory=lambda: [2048, 1024, 512])
    critic_architecture: str = "radial_angular"  # "homogeneous" or "radial_angular"
    angular_hidden_dims: list[int] = field(default_factory=lambda: [2048, 1024, 512])
    radial_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    norm_mode: str = "none"
    activation: str = "silu"
    angular_norm_mode: str = "none"
    angular_activation: str = "silu"
    radial_norm_mode: str = "none"
    radial_activation: str = "silu"
    actor_norm_mode: str = "none"   # "orthonorm", "spectral_norm", "none"
    actor_activation: str = "silu"       # "silu", "gelu", "relu", "groupsort", "lipschitz_spline"
    ortho_n_iters: int = 2
    ortho_schedule_enabled: bool = False
    ortho_schedule_iters: list[int] = field(default_factory=lambda: [2])
    ortho_schedule_boundaries: list[float] = field(default_factory=list)
    twin_aggregate: str = "softmax"  # "max", "mean", or "softmax"
    twin_softmax_temperature: float = 0.10
    sigma_head_weighting_enabled: bool = True
    angular_weight_low_sigma: float = 0.45
    angular_weight_high_sigma: float = 0.75
    head_weight_power: float = 1.0

    # Optimizer
    critic_lr: float = 1e-4
    actor_lr: float = 5e-5
    prior_critic_lr: float = 1e-4
    weight_decay: float = 0.01
    clip_grad_norm: float = 1.0
    critic_steps_per_actor: int = 2
    critic_step_mode: str = "joint"  # "alternating" or "joint" (same-batch two-head update)

    # LR scheduler: "none", "cosine_warmup", "cosine_warm_restarts"
    lr_scheduler: str = "none"
    lr_warmup_epochs: int = 5           # linear warmup from lr_warmup_start_factor to 1.0
    lr_warmup_start_factor: float = 0.1 # initial LR multiplier during warmup
    lr_min_factor: float = 0.01         # minimum LR as fraction of base LR
    lr_restart_period: int = 10         # T_0 for CosineAnnealingWarmRestarts (epochs)
    lr_restart_mult: int = 2            # T_mult — each restart period grows by this factor
    lr_plateau_patience: int = 0        # 0=disabled; >0: boost LR if no improvement for N epochs
    lr_plateau_boost: float = 3.0       # multiply LR by this factor on plateau detection

    # Loss weights
    lambda_mdsm: float = 0.5
    lambda_mdsm_angular: float = 0.5
    lambda_mdsm_radial: float = 0.2
    mdsm_warmup_epochs: int = 5  # Epochs of ranking-focused warmup before full MDSM
    lambda_rank: float = 1.0
    lambda_rank_angular: float = 1.0
    lambda_rank_radial: float = 0.5
    lambda_nce: float = 0.10
    lambda_cql: float = 0.1
    lambda_shell: float = 0.1
    lambda_geo: float = 1.0
    lambda_align: float = 0.1
    lambda_bc_reg: float = 0.3
    lambda_prior: float = 0.1
    lambda_prior_nce: float = 0.05
    lambda_actor_barrier: float = 0.1
    lambda_actor_descent: float = 0.1
    # P0: Clean-minimum penalty — prevents sub-clean attractors
    lambda_clean_min: float = 0.2
    lambda_clean_min_angular: float = 0.0
    lambda_clean_min_radial: float = 0.2
    clean_min_margin: float = 0.1
    # P0: Explicit gradient direction loss — teaches critic WHERE to point
    lambda_direction: float = 0.1
    lambda_direction_angular: float = 0.1
    lambda_direction_radial: float = 0.0
    direction_num_samples: int = 1
    # Head specialization regularizers
    lambda_angular_purity: float = 0.05
    lambda_radial_purity: float = 0.05
    lambda_head_corr: float = 0.02
    head_corr_target: float = 0.20
    # P1: Support/manifold proximity penalty — kNN to retrieval bank
    lambda_support: float = 0.0
    support_k: int = 5
    support_threshold_percentile: float = 95.0  # auto-calibrate from bank distances
    # P1: Enhanced multi-negative contrastive — in-batch cross-negatives
    lambda_inbatch_nce: float = 0.0
    inbatch_nce_temperature: float = 0.07
    use_inbatch_negatives: bool = False

    # Feature flags
    use_cql: bool = False
    use_nce: bool = False
    use_bc: bool = True
    use_grad_align: bool = True
    use_prior_critic: bool = False
    use_prior_nce: bool = False
    use_gradient_penalty: bool = False
    use_shell_barrier: bool = False
    use_clean_min_penalty: bool = True
    use_direction_loss: bool = True
    route_cd_to_radial_only: bool = True
    route_energy_floor_to_radial_only: bool = True
    route_direction_to_angular_only: bool = True
    route_clean_min_to_radial_only: bool = True
    use_support_penalty: bool = False
    # Energy scale calibration
    energy_scale_trainable: bool = False
    energy_scale_init_log: float = 0.0
    energy_scale_lr_multiplier: float = 1.0
    lambda_energy_scale_reg: float = 0.0
    # Energy scale regularization (prevent unbounded energy growth)
    use_energy_reg: bool = True
    lambda_energy_reg: float = 0.01
    energy_reg_universal: bool = False  # True = regularize e_pos, e_actor, e_hard; False = only e_pos
    energy_reg_actor_weight: float = 2.0  # relative weight for actor/hard vs clean in universal mode
    # Energy floor: softplus penalty for E < -threshold (kills spurious wells)
    use_energy_floor: bool = False
    lambda_energy_floor: float = 0.1
    energy_floor_threshold: float = 5.0   # penalize E < -5
    energy_floor_relative_to_clean: bool = True
    energy_floor_margin: float = 0.15
    energy_floor_sharpness: float = 2.0   # softplus sharpness (1=gentle, 5≈relu)
    energy_floor_num_random: int = 64     # random sphere points to probe for spurious wells
    energy_floor_adversarial_steps: int = 0  # 0=random only, >0=gradient descent steps to find wells
    energy_floor_adversarial_lr: float = 0.01  # step size for adversarial well-finding
    # Contrastive divergence: run Langevin in critic loop, push up energy at endpoints
    use_cd: bool = True
    lambda_cd: float = 0.1
    cd_relative_to_clean: bool = True
    cd_margin: float = 0.15
    cd_num_samples: int = 32             # number of particles to run
    cd_num_steps: int = 10               # Langevin steps per particle
    cd_lr: float = 0.01                  # Langevin step size for CD
    cd_noise_scale: float = 0.001        # Langevin noise for CD (low = more adversarial)
    # Interpolated gradient penalty (WGAN-GP style) — prevents spurious wells in clean-to-hard corridor
    use_interp_gp: bool = False
    lambda_interp_gp: float = 0.1

    # Regularization params
    cql_noise_scale: float = 0.5
    nce_temperature: float = 0.07
    nce_num_random_negatives: int = 4
    gradient_penalty_lambda: float = 0.05
    shell_barrier_margin: float = 0.1

    # Noise and MDSM
    sigma_curriculum_start: float = 0.01
    sigma_curriculum_end: float = 0.3
    sigma_min: float = 0.01
    sigma_max: float = 0.3
    sigma_sampling: str = "loguniform"
    sigma_weighting: str = "sigma2"
    edm_p_mean: float = -1.2
    edm_p_std: float = 1.2
    # MDSM numeric safety / low-sigma handling.
    # target_mode:
    #   - "standard": classic target = (noisy-clean)/sigma_eff_sq
    #   - "logspace": computes inverse sigma_eff_sq in log-space to avoid hard dead-zones
    mdsm_target_mode: str = "logspace"
    # sigma_eff floor mode for standard target:
    #   - "constant": fixed absolute floor
    #   - "adaptive": floor scales with ||x||^2 to keep relative geometry
    #   - "none": no explicit floor (not recommended unless logspace mode is used)
    mdsm_sigma_eff_floor_mode: str = "adaptive"
    mdsm_sigma_eff_floor: float = 1e-10
    mdsm_sigma_floor: float = 1e-8
    mdsm_inv_sigma2_clip: float = 1e8
    mdsm_weight_floor: float = 1e-8
    mdsm_tangent_projection: bool = True
    mdsm_directional: bool = True
    mdsm_magnitude_aux_weight: float = 0.05
    mdsm_cosine_eps: float = 1e-6
    mdsm_norm_floor: float = 1e-6
    mdsm_force_fp32: bool = True
    mdsm_gradient_checkpointing: bool = True

    # Ranking margins
    critic_margin_clean_actor: float = 0.1
    critic_margin_actor_noisy: float = 0.05
    critic_margin_clean_noisy: float = 0.15
    rank_normalize_by_std: bool = False
    rank_std_floor: float = 1e-2
    actor_energy_margin_pos: float = 0.05
    actor_energy_margin_hard: float = 0.05
    actor_barrier_normalize_by_std: bool = False

    # Actor/inference rollout
    actor_step_size: float = 0.5
    actor_tangent_projection: bool = True
    actor_seed_mix_query: float = 0.5
    actor_seed_noise_scale: float = 1.0
    actor_eval_steps: int = 3
    critic_eval_langevin_steps: int = 20
    eval_langevin_batch_size: int = 16
    eval_noise_scales: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.3])

    # Retrieval conditioning
    retrieval_bank_size: int = 4096
    retrieval_topk_pos: int = 8
    retrieval_hard_start: int = 8
    retrieval_hard_end: int = 32
    retrieval_self_sim_exclude: float = 0.9995
    retrieval_min_pos_similarity: float = 0.15
    retrieval_strict_index_exclusion: bool = True

    # AMP
    amp_enabled: bool = True
    amp_dtype: str = "bf16"
    enable_compile: bool = False
    compile_mode: str = "reduce-overhead"
    compile_fullgraph: bool = False

    # Training
    seed: int = 42
    num_epochs: int = 50
    batch_size: int = 32
    num_workers: int = 4
    log_every: int = 50
    checkpoint_every_epochs: int = 10
    rolling_checkpoint_name: str = "latest_epoch.pt"
    eval_num_samples: int = 128
    eval_every_epochs: int = 2
    langevin_tangent_noise: bool = True

    # Kill criteria thresholds
    min_cosine_improvement: float = 0.06
    min_cosine_success_rate: float = 0.70
    min_geodesic_improvement: float = 0.02
    min_l2_improvement: float = 0.005
    min_energy_success_rate: float = 0.65
    max_clean_min_violation_rate: float = 0.05
    clean_min_violation_mode: str = "margin"  # "margin" or "strict"
    min_step_norm: float = 1e-3

    # Stability
    skip_non_finite_batches: bool = True
    non_finite_backoff_streak_trigger: int = 3
    non_finite_lr_backoff: float = 0.5
    max_consecutive_non_finite_batches: int = 10
    guard_loss_spikes: bool = True
    loss_spike_factor: float = 20.0
    loss_spike_warmup_steps: int = 25
    param_finite_check_interval: int = 50

    # Paths
    train_data_path: str = "data/wikitext_sonar_10k.pt"
    val_data_path: str = ""
    output_dir: str = "experiments/03_Stage_1.5"
    checkpoint_dir: str = "experiments/03_Stage_1.5/checkpoints"
    logs_dir: str = "experiments/03_Stage_1.5/logs"
