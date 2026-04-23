# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CERBER implements CEBCM (Concept-Driven Energy-Based Coding Machine) — an energy-based architecture for multi-step reasoning in SONAR latent space (1024-dim, norm ≈ 0.2051). The active work is Stage 4: autoregressive chain generation (`experiments/13_chain_generator/`).

## Commands

```bash
# Install (CUDA 12.8+ / DGX target)
pip install -e .
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Train chain generator (main active experiment)
python experiments/13_chain_generator/train_chain_generator.py \
  --config configs/chain_generator_no_sadt_df_ema_soft_ss.json

# Train FF variant
python experiments/13_chain_generator/train_chain_generator_ff.py \
  --config configs/chain_generator_ff_config.json

# Run tests
pytest tests/

# Data preparation
python scripts/prepare_hotpotqa.py
python scripts/prepare_sonar_sequences.py

# GUI monitor
python cerber_gui/app.py
```

## Architecture

### SONAR Space Geometry
All vectors live on a 1024-dim hypersphere with `target_norm=0.2051`. This is ~156× smaller than standard transformer residual-stream scale (`sqrt(1024)=32`). The model bridges this via `_to_residual_space()` (×156 up) and `output_proj` (learned down-projection, xavier gain=0.01).

### Chain Generator (`cebcm/models/chain_generator.py`)
Autoregressive transformer decoder: `start_token → [DecoderBlock × 6] → output_proj → SONAR vector`.

Key components:
- **DecoderBlock**: pre-norm with `AdaRMSNorm` (DiT-style AdaLN-Zero for diffusion conditioning), `CausalRoPESelfAttention`, cross-attention to context, `SwiGLUFFN`
- **LayerScale**: per-channel gates (init 1e-4) on every residual branch
- **residual_norm_clamp**: soft L2 clamp on sublayer outputs before residual add (bf16 overflow protection)
- **adaln_scale_clamp**: tanh bound on AdaLN modulation scales (prevents SwiGLU quadratic explosion)
- **sphere_project**: projects raw output onto SONAR hypersphere with NaN fallback to e₀

### Training Pipeline (`experiments/13_chain_generator/train_chain_generator.py`)
Composite loss: `L = λ_step·L_tf + λ_ans·L_ans + λ_roll·L_roll + λ_df·L_df`

- **L_tf**: teacher-forced step prediction (cosine + MSE)
- **L_roll**: free-run rollout prediction (gradient detached per step — only 1-step deep)
- **L_df**: Diffusion Forcing denoising (v-prediction, Min-SNR-γ weights, logSNR sampling)
- **Horizon schedule**: `system1_epochs` at `target_steps=1`, then curriculum `[[2,2],[3,2],[4,10],[5,10]]`

### Config System
JSON configs in `configs/` map to `ChainGeneratorConfig` dataclass (line 16 of chain_generator.py). Training params are in the `"training"` section. The config is passed as a plain dict (`cfg`) through most training functions.

## Critical Knowledge (from tasks/lessons.md)

### Numerical Safety
- **bf16 max = 65504**. FFN outputs can overflow if unconstrained. `residual_norm_clamp=256` is the safety valve.
- **AdaLN scale must be bounded** (`adaln_scale_clamp=1.0`). Unbounded scale + SwiGLU = quadratic explosion → NaN at horizon transitions.
- **Min-SNR weights must stay in fp32** entirely. bf16 has no denormals; `clamp(min=1e-8)` is a no-op when the value underflows to 0.
- **NaN × 0 ≠ 0 in autograd**. Use `torch.where(mask, term, zeros)` not multiplication by mask.
- **NaN replacement**: replace with GT (not zero). `cos(0, target) = 0 → loss=1.0` is a "loss bomb".

### Weight Decay
- LayerScale params (`.ls_*`) are 1D but NOT biases — they need weight decay (ndim≤1 heuristic was wrong, fixed in `_build_wd_param_groups`)
- `adaln_modulation` Linear weight (2D) correctly gets decay

### Exposure Bias
The fundamental unsolved problem. Train roll_cos ≈ 0.90 but eval roll_cos_last ≈ 0.62 at `target_steps=5`. Gap grows ~0.03 per chain step. SS (0.30), noisy TF (0.002), and DF (0.15) all showed negligible improvement (<0.01 on roll_cos_last). Root cause: `generate()` detaches at every step (line ~1570), so the model never receives gradient for cascading multi-step drift.

### Horizon Transitions
System1→System2 is the most dangerous point. "Cold positions" (never seen during System1) have untrained attention patterns. Always complete all warmups BEFORE the transition. `horizon_warmup_steps=200` with `horizon_warmup_factor=0.1` ramps LR. Adam second-moment is optionally scaled down (`horizon_adam_sq_scale`).

## Key Files

| File | Role |
|------|------|
| `cebcm/models/chain_generator.py` | Model: ChainGenerator, DecoderBlock, AdaRMSNorm, SwiGLUFFN |
| `experiments/13_chain_generator/train_chain_generator.py` | Training loop, all losses, probes, eval |
| `configs/chain_generator_no_sadt_df_ema_soft_ss.json` | Active training config |
| `configs/base.py` | Central dataclass definitions |
| `tasks/lessons.md` | **READ THIS FIRST** — accumulated debugging wisdom, prevents repeat mistakes |
| `tasks/todo.md` | Current task tracker |
| `CEBCM_Technical_Specification.md` | Full architecture spec (v1.4) |
| `IMPLEMENTATION_PLAN.md` | Stage roadmap and design decisions |

## Agent Directives

### 1. Plan First
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately. ONLY SOTA techniques!
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- One tack per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update tasks/lessons.md with the pattern
- Write rules for yourself that prevent the same mistake
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- Skip this for simple, obvious fixes — don't over-engineer

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them

## Task Management
1. **Plan First**: Write plan to tasks/todo.md with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to tasks/todo.md
6. **Capture Lessons**: Update tasks/lessons.md after corrections
7. **Check project docs**: Always check IMPLEMENTATION_PLAN.md, CEBCM_Technical_Specification.md, and tasks/lessons.md before planning non-trivial tasks

## Core Principles

**Simplicity First**: Make every change as simple as possible. Impact minimal code.
**No Laziness**: Find root causes. No temporary fixes. Senior developer standards, SOTA techniques!
