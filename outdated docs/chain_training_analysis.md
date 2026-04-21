# Chain Generator (Exp 13/14) — Deep Training Analysis

## Executive Summary

I reviewed the training logs (E0–E16), the prior analysis ([09_04_26_train_analysis.txt](file:///c:/Coding/Python/CERBER/09_04_26_train_analysis.txt)), and the full codebase: [train_chain_generator.py](file:///c:/Coding/Python/CERBER/experiments/13_chain_generator/train_chain_generator.py), [chain_generator.py](file:///c:/Coding/Python/CERBER/cebcm/models/chain_generator.py), and [chain_generator_config.json](file:///c:/Coding/Python/CERBER/configs/chain_generator_config.json).

**The prior analysis correctly identifies 5 issues.** I confirm all of them and found **3 additional bugs** not covered in that document.

---

## Training Dynamics Summary

| Epoch | Phase | `tf_cos` | `roll_cos` | Gap | `rank_acc` |
|-------|-------|----------|------------|-----|------------|
| E0 | System1 (1 step) | 0.682 | 0.682 | **0.0** | 0.17 |
| E5 | System1 (1 step) | 0.727 | 0.727 | **0.0** | 0.50 |
| E10 | System1 (1 step) | 0.744 | 0.744 | **0.0** | 0.61 |
| E11 | System2 (2 steps) | 0.808 | 0.734 | **7.4** | 0.72 |
| E12 | System2 (4 steps) | 0.771 | 0.674 | **9.7** | 0.77 |
| E13 | System2 (6 steps) | 0.716 | 0.627 | **8.9** | 0.80 |
| E14 | System2 (8 steps) | 0.732 | 0.637 | **9.5** | 0.81 |
| E15 | System2 (10 steps) | 0.750 | 0.657 | **9.3** | 0.83 |
| E16 | System2 (12 steps) | 0.753 | 0.678 | **7.5** | 0.93 |

> [!IMPORTANT]
> The gap opens **instantly** at E11 when `target_steps` jumps from 1→2 and widens to ~8–10 points. This is the hallmark of **exposure bias**: teacher-forced [forward()](file:///c:/Coding/Python/CERBER/cebcm/models/chain_generator.py#48-86) always feeds ground truth, while [generate()](file:///c:/Coding/Python/CERBER/cebcm/models/chain_generator.py#294-511) feeds its own predictions and drifts.

---

## Confirmed Issues (from prior analysis)

### 1. ✅ Gradient flow through L_roll is shallow and noisy
The autoregressive loop in [generate()](file:///c:/Coding/Python/CERBER/cebcm/models/chain_generator.py#294-511) (lines 383–432) builds a deep stochastic graph through `torch.cat`, [_sphere_project](file:///c:/Coding/Python/CERBER/cebcm/models/chain_generator.py#234-236), and noise injections. Effective gradient signal degrades exponentially with chain depth. **Confirmed by code review.**

### 2. ✅ Noise level 0.005 is cosmetic  
Config: `free_run_noise_std = 0.005`, `free_run_temperature = 1.05` → effective `noise_std = 0.00525`.
After [_sphere_project](file:///c:/Coding/Python/CERBER/cebcm/models/chain_generator.py#234-236) (norm=0.2051), angular perturbation ≈ 1.5° → cosine deviation ≈ 0.0003. This is **300× smaller** than the 7–10% tf/roll gap. The rollout distribution is virtually identical to teacher forcing.

### 3. ✅ L_roll magnitude is deceptively balanced  
`lambda_roll = 1.0` appears fine, but L_roll's gradient is noisy/shallow, so the optimizer mostly optimizes step-1 via L_roll while deeper steps get degraded signal. The gap widens from 7.4% → 9.7% despite equal weights.

### 4. ✅ Scheduled sampling is absent  
[forward()](file:///c:/Coding/Python/CERBER/cebcm/models/chain_generator.py#48-86) (line 282) is pure teacher forcing: `decoder_input = torch.cat([start, v_target_chain[:, :-1, :]], dim=1)`. The model **never** sees its own predictions during the forward pass.

### 5. ✅ `repeat_ban` `torch.where` corrupts training gradients  
Lines 412–429: `torch.where(mask3, candidate, next_vec)` introduces a piecewise gradient with a second `randn` sample. This fires during training (retries=2, threshold=0.995).

---

## NEW Issues Discovered

### 6. 🐛 [select_training_targets](file:///c:/Coding/Python/CERBER/experiments/13_chain_generator/train_chain_generator.py#148-169) suffix alignment causes target/prediction misalignment at System2 transition

```python
# train_chain_generator.py:148-168
def select_training_targets(chains, chain_lens, target_steps):
    for i in range(bsz):
        li = max(1, min(int(chain_lens[i].item()), full_len))
        ti = min(steps, li)
        start = li - ti  # ← suffix-aligned
        targets[i, :ti] = chains[i, start:li]
```

This selects the **last** `target_steps` tokens from each chain (suffix-aligned, ending at the answer). But `model.generate()` always **starts from scratch** (from `start_token`), generating from position 0. The model is asked to match the *end* of the chain when generating from the *beginning*.

**For System1 (steps=1):** This is fine — one step = just the last token (answer).

**For System2 (steps=4):** The targets are `chain[li-4 : li]` (the last 4 reasoning+answer steps), but the model generates starting from position 0 with no prior context. The model must produce step "li-4" as its first output while having no hidden state from steps 0..li-5. **This is an impossible task** — the model is being asked to generate late-chain vectors from cold start.

> [!CAUTION]
> This is likely the **primary cause** of the instant gap at E11. It's not just exposure bias — the targets themselves become misaligned with what the model can actually produce from generation start.

### 7. 🐛 `L_ans` (answer loss) double-counts gradients with `L_step`

```python
# train_chain_generator.py:260-265
tf_final = _gather_last_valid(v_tf, valid_lens)
tgt_final = _gather_last_valid(chains, valid_lens)
ans_cos = (1.0 - F.cosine_similarity(tf_final, tgt_final, dim=-1)).mean()
l_ans = w_cos * ans_cos + w_mse * ans_mse
```

`L_step` already computes the masked cosine loss over ALL positions including the last one. Then `L_ans` re-applies a separate loss on the **exact same last position**. With `lambda_step = lambda_ans = 1.0`, the answer position receives **2× the gradient** of every other position.

**Effect:** The model is over-optimizing the final answer token at the expense of intermediate reasoning steps. This is visible in E12 where `loss_ans` is only 0.094 while `loss_step` and `loss_roll` are ~0.24 and ~0.32 — the answer is already very close, but intermediate steps suffer.

### 8. 🐛 Horizon ramp is too aggressive — jumps 2→4→6→8→10→12 in single-epoch steps

```python
# train_chain_generator.py:136-145
def get_chain_steps(epoch, cfg):
    if epoch < s1_epochs:  # epochs 0-9 → 1 step
        return 1
    ramp_progress = min(1.0, (epoch - s1_epochs) / max(ramp_epochs, 1))
    return max(1, int(1 + ramp_progress * (max_steps - 1)))
```

With `system1_epochs=10` and `system2_ramp_epochs=10`:
- E10: 1 step, E11: 2, E12: 4, E13: 6, E14: 8, E15: 10, E16: 12, ...

The model jumps from 1→2 steps at E11 (tolerable), then to **4 steps at E12** — doubling the chain length in a single epoch. Combined with Issue #6 (suffix alignment), the model must suddenly predict 4-step suffix chains cold, which is a shock. The logs confirm: E12 `roll_cos` crashes from 0.734 to 0.674.

A **linear** ramp (1→2→3→4→...) would be much gentler. The current [int()](file:///c:/Coding/Python/CERBER/cebcm/visualization/energy_landscape.py#454-471) truncation creates step-function jumps.

---

## Ranked Fix Priority

| # | Fix | Impact | Effort |
|---|-----|--------|--------|
| **6** | Fix [select_training_targets](file:///c:/Coding/Python/CERBER/experiments/13_chain_generator/train_chain_generator.py#148-169) — use **prefix-aligned** targets starting from position 0, matching what [generate()](file:///c:/Coding/Python/CERBER/cebcm/models/chain_generator.py#294-511) actually produces | **Critical** | Medium |
| **4** | Implement scheduled sampling in [forward()](file:///c:/Coding/Python/CERBER/cebcm/models/chain_generator.py#48-86) (ss_prob ramp) | **High** | Medium |
| **3** | Detach `next_vec` before appending to chain in [generate()](file:///c:/Coding/Python/CERBER/cebcm/models/chain_generator.py#294-511) to prevent gradient corruption via noisy autoregressive path | **High** | Trivial |
| **5** | Disable `repeat_ban` during training (`max_retries=0`) | **High** | Trivial |
| **2** | Raise `free_run_noise_std` to 0.04–0.07 | **Medium** | Trivial |
| **7** | Remove `L_ans` or reduce `lambda_ans` to 0.0 since the last step is already included in `L_step` | **Medium** | Trivial |
| **8** | Use linear step ramp: `return max(1, 1 + (epoch - s1_epochs))` | **Medium** | Trivial |
| **1** | Add per-step cosine diagnostics (step-0 vs step-N) | **Diagnostic** | Low |

---

## Key Insight

The prior analysis correctly identifies the **symptom** (exposure bias gap) but misses that [select_training_targets](file:///c:/Coding/Python/CERBER/experiments/13_chain_generator/train_chain_generator.py#148-169) with suffix alignment makes the problem far worse than standard exposure bias. In a normal seq2seq model, exposure bias degrades quality gradually. Here, the targets are **structurally misaligned** with the generation starting point, making the gap open instantly and catastrophically at E11.

**Fix #6 alone should dramatically reduce the gap**, because the model would then be asked to predict steps 0..N from position 0, which is exactly what [generate()](file:///c:/Coding/Python/CERBER/cebcm/models/chain_generator.py#294-511) does.
