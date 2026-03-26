# Stage 1.5: Hybrid Actor-Critic with SOTA Stabilization

## Overview

Stage 1.5 implements the fixed actor-critic training pipeline with all P0-P2 corrections from research3.md.

**Key fixes:**
- ✅ Removed `actor_energy_loss` contradiction (P0)
- ✅ Added MDSM to critic for gradient validity (P0)
- ✅ Hybrid critic: E_cond(q,v) + λ*E_prior(v) (P1)
- ✅ Alternating training: 2 critic steps : 1 actor step (P1)
- ✅ CQL regularization for OOD prevention (P2)
- ✅ BC regularization for embedding anchor (P2)
- ✅ Gradient penalty for smooth landscape (P2)
- ✅ Shell barrier for norm control (P2)
- ✅ Comprehensive metrics telemetry (P3)

## Directory Structure

```
experiments/03_Stage_1.5/
├── README.md              # This file
├── checkpoints/           # Saved model checkpoints
│   ├── checkpoint_epoch_1.pt
│   ├── checkpoint_epoch_2.pt
│   ├── ...
│   └── best.pt            # Best checkpoint by composite score
├── logs/                  # Training logs
│   ├── training_metrics.jsonl    # Per-epoch metrics (JSONL)
│   └── training_summary.json     # Final summary
└── (runtime artifacts)
```

## Usage

### Start New Training

```bash
python experiments/01_denoising_poc/train_stage1_5.py \
  --config configs/stage1_5_config.json
```

### Resume from Checkpoint

```bash
python experiments/01_denoising_poc/train_stage1_5.py \
  --config configs/stage1_5_config.json \
  --resume experiments/03_Stage_1.5/checkpoints/checkpoint_epoch_10.pt
```

## Configuration

Edit `configs/stage1_5_config.json` for hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 50 | Training epochs |
| `batch_size` | 64 | Batch size |
| `critic_lr` | 1e-4 | Critic learning rate |
| `actor_lr` | 5e-5 | Actor learning rate |
| `critic_steps_per_actor` | 2 | Alternating ratio |
| `lambda_mdsm` | 1.0 | MDSM loss weight |
| `lambda_rank` | 0.25 | Ranking loss weight |
| `lambda_cql` | 0.1 | CQL regularization weight |
| `use_cql` | true | Enable CQL |
| `use_gradient_penalty` | false | Enable gradient penalty |
| `use_shell_barrier` | false | Enable shell barrier |

## Composite Score

Checkpoints are selected by composite score:

```python
composite_score = (
    cosine_improvement * 0.4 +          # Primary semantic metric
    geodesic_improvement * 0.3 +        # Geometry-aware metric
    clean_min_violation_gate * 0.2 +    # 1.0 if violation < 5%
    energy_success_rate * 0.1           # Energy descent rate
)
```

## Kill Criteria

Training stops automatically if:
- `cosine_improvement < 0.05`
- `energy_success_rate < 0.5`
- `clean_min_violation_rate > 0.1`
- `geodesic_improvement < 0.01`

## Metrics Tracked

Per-epoch metrics (saved to `logs/training_metrics.jsonl`):

**Losses:**
- `critic_loss`, `actor_loss`
- `mdsm_loss`, `ranking_loss`, `cql_loss`
- `gp_loss` (gradient penalty), `shell_loss`
- `bc_loss` (behavior cloning)

**Energy Statistics:**
- `e_clean_mean`, `e_actor_mean`, `e_noisy_mean`
- `critic_gap_clean_actor`, `critic_gap_clean_noisy`

**Stability Metrics:**
- `clean_min_violation_rate`
- `grad_norm_mean`, `grad_norm_max`

## What Stage 1.5 Does

Stage 1.5 is an **improved denoising autoencoder** for SONAR embeddings:

**Input:** Noisy embedding `v_noisy`
**Output:** Refined embedding `v_clean`

**Capabilities:**
- ✅ Denoising: Restore corrupted vectors
- ✅ Quality scoring: E(q, v) for ranking candidates
- ✅ Gradient-based refinement via Langevin dynamics
- ✅ OOD detection via CQL + shell barrier

**NOT Capabilities:**
- ❌ Autoregressive generation
- ❌ Sequence modeling
- ❌ New vector synthesis from scratch

For autoregressive capabilities, Stage 2 (IPP + Chain Head) is required.

## References

- research3.md: Full audit and design rationale
- tasks/todo.md: Implementation checklist
- CEBCM_Technical_Specification.md: Full architecture spec

## Checkpoint Format

```python
checkpoint = {
    "epoch": int,
    "global_step": int,
    "critic": state_dict,
    "actor": state_dict,
    "optimizer": state_dict,
    "scaler": state_dict,
    "config": dataclass_dict,
    "train_metrics": dict,
    "eval_metrics": dict,
    "composite_score": float,
}
```
