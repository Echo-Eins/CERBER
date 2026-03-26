# Stage 1.5: Hybrid Actor + Twin Critic

## What this stage trains

Stage 1.5 trains:
- conditional **twin critics** `E(q, v)` with MDSM + ranking (+ optional NCE/CQL/prior terms),
- a proposal/refinement **actor** that predicts a latent update,
- optional short-run Langevin refinement during eval.

Current train script:
- `experiments/01_denoising_poc/train_stage1_5.py`

## Run

```bash
bash experiments/03_Stage_1.5/run.sh
```

Resume:

```bash
bash experiments/03_Stage_1.5/run.sh --resume experiments/03_Stage_1.5/checkpoints/epoch_10.pt
```

## Current default profile (from `configs/stage1_5_config.json`)

- `batch_size = 32`
- `critic_steps_per_actor = 2`
- `norm_mode = orthonorm`, `ortho_n_iters = 4`
- `ortho_schedule_enabled = true`, `ortho_schedule_iters = [4,2,1]`
- `twin_aggregate = softmax`, `twin_softmax_temperature = 0.10`
- `retrieval_bank_size = 2048`
- `eval_every_epochs = 2`, `eval_num_samples = 64`
- `critic_eval_langevin_steps = 10`
- `eval_langevin_batch_size = 16`
- `mdsm_gradient_checkpointing = true`
- `enable_compile = false` (opt-in)
- `param_finite_check_interval = 50`

## Logging semantics

Console training line now reports:
- `rank_loss`: hinge ranking objective value,
- `rank_success`: joint ordering success `E(clean) < E(actor) < E(hard)`,
- `rank(c<a)`, `rank(a<h)`, `rank(c<h)`: per-inequality pass rates,
- `viol`: clean-minimum violation rate (`E(actor) < E(clean)`).

Per-epoch metrics are stored in:
- `experiments/03_Stage_1.5/logs/training_metrics.jsonl`
- `experiments/03_Stage_1.5/logs/training_summary.json`

Schema note:
- epochs with skipped eval (`eval_every_epochs`) write:
  - `"eval_ran": false`
  - `"kill_criteria": {"status": "not_evaluated", ...}`

## Kill criteria and best checkpoint

Source of truth:
- `cebcm/training/kill_criteria.py`

`best.pt` is selected by composite score **only on epochs where eval actually ran**.

Important:
- kill criteria are strict quality gates for evaluation/reporting and checkpoint scoring,
- they do **not** hard-stop the training loop automatically at the moment.

## Checkpoint keys

Stage 1.5 checkpoints store:
- `critic1_state`, `critic2_state`, `actor_state`
- optional `prior_state`
- `opt_c_state`, `opt_a_state`, `scaler_state`
- `train_metrics`, `eval_metrics`, `best_score`, `config`
