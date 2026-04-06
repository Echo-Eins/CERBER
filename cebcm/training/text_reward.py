"""
REINFORCE-style text supervision loss for CEBM autoregressor.

Since SONAR decoder is non-differentiable, we use a REINFORCE-style approach:
1. Decode v_predicted → text via SONAR decoder (no grad)
2. Re-encode decoded text back through SONAR encoder (no grad)
3. Compute reward = similarity between re-encoded text and target encoding
4. Use reward as REINFORCE signal to train the model

The key insight: if decode(v) produces text that, when re-encoded, matches
the target embedding, then v is a good answer vector. This catches cases
where vectors are close in cosine but decode to different text.

VRAM note: Requires SONAR decoder (~3.3GB) in the training loop.
To save VRAM, call compute_text_reward() only every N batches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from cebcm.models.sonar_wrapper import SONARWrapper


def compute_text_reward(
    v_predicted: Tensor,
    v_target: Tensor,
    target_texts: list[str],
    sonar: SONARWrapper,
    lang: str = "eng_Latn",
    use_bleu: bool = False,
) -> tuple[Tensor, dict[str, float]]:
    """
    Compute text-level reward for predicted vectors.

    Args:
        v_predicted:  [B, D] model output vectors (from Langevin)
        v_target:     [B, D] target answer embeddings (from dataset)
        target_texts: Ground truth answer strings (from dataset)
        sonar:        SONAR wrapper with encoder + decoder
        lang:         Language code for SONAR
        use_bleu:     Also compute BLEU score bonus

    Returns:
        (reward [B], metrics dict)
    """
    B = v_predicted.shape[0]
    device = v_predicted.device

    with torch.no_grad():
        # 1. Decode predicted vectors to text
        pred_texts = sonar.decode_safe(v_predicted, lang=lang)

        # 2. Re-encode predicted texts back through SONAR
        # Filter out decode failures
        valid_mask = torch.ones(B, dtype=torch.bool, device=device)
        clean_texts = []
        for i, t in enumerate(pred_texts):
            if t.startswith("[OOM]") or t.startswith("[ERROR"):
                valid_mask[i] = False
                clean_texts.append("")  # placeholder
            else:
                clean_texts.append(t)

        n_valid = int(valid_mask.sum().item())

        if n_valid == 0:
            # All decodes failed — return zero reward
            return torch.zeros(B, device=device), {
                "text_reward_mean": 0.0,
                "text_reward_std": 0.0,
                "decode_success_rate": 0.0,
                "roundtrip_cos_mean": 0.0,
            }

        # Encode only valid predicted texts
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        valid_pred_texts = [clean_texts[i] for i in valid_indices.tolist()]
        valid_target_texts = [target_texts[i] for i in valid_indices.tolist()]

        v_reencoded = sonar.encode(valid_pred_texts, lang=lang).to(device)  # [N_valid, D]
        v_target_enc = sonar.encode(valid_target_texts, lang=lang).to(device)  # [N_valid, D]

        # 3. Round-trip cosine: does re-encoded predicted text match target?
        roundtrip_cos = F.cosine_similarity(v_reencoded, v_target_enc, dim=-1)  # [N_valid]

        # 4. Self-consistency cosine: does re-encoded text match original vector?
        # This catches OOD drift — if decode→re-encode changes the vector,
        # the original vector was in a bad region.
        v_pred_valid = v_predicted[valid_indices]
        self_cos = F.cosine_similarity(v_reencoded, v_pred_valid, dim=-1)  # [N_valid]

        # 5. Combined reward: roundtrip quality + self-consistency
        reward_valid = 0.7 * roundtrip_cos + 0.3 * self_cos

        # 6. BLEU score bonus (optional, more expensive)
        bleu_bonus = torch.zeros_like(reward_valid)
        if use_bleu:
            bleu_scores = _compute_bleu_batch(valid_pred_texts, valid_target_texts)
            bleu_bonus = torch.tensor(bleu_scores, device=device, dtype=reward_valid.dtype)
            reward_valid = reward_valid + 0.2 * bleu_bonus

        # Scatter rewards back to full batch
        reward = torch.zeros(B, device=device)
        reward[valid_indices] = reward_valid

    metrics = {
        "text_reward_mean": reward.mean().item(),
        "text_reward_std": reward.std().item(),
        "decode_success_rate": n_valid / B,
        "roundtrip_cos_mean": roundtrip_cos.mean().item(),
        "self_cos_mean": self_cos.mean().item(),
    }
    if use_bleu:
        metrics["bleu_mean"] = bleu_bonus.mean().item()

    return reward, metrics


def compute_reinforce_loss(
    v_predicted: Tensor,
    v_target: Tensor,
    reward: Tensor,
    baseline: float | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """
    REINFORCE loss: use text reward as training signal.

    Uses reward-weighted cosine similarity as the differentiable proxy.
    The gradient flows through v_predicted → Langevin → critic parameters.

    Math:
        advantage = reward - baseline
        loss = -mean(advantage · cos(v_predicted, v_target))

    When advantage > 0 (good reward): minimizing loss → maximize cosine
        → critic adjusts so Langevin converges closer to v_target.
    When advantage < 0 (bad reward): minimizing loss → decrease cosine
        → critic adjusts to explore away from current trajectory.

    Args:
        v_predicted: [B, D] model output vectors (requires grad through Langevin)
        v_target:    [B, D] target answer vectors
        reward:      [B] text reward (no grad, from compute_text_reward)
        baseline:    Optional baseline for variance reduction (running mean of rewards)

    Returns:
        (loss scalar, metrics dict)
    """
    reward_detached = reward.detach()

    # Baseline subtraction for variance reduction
    if baseline is not None:
        advantage = reward_detached - baseline
    else:
        advantage = reward_detached - reward_detached.mean()

    # Reward-weighted cosine: proper differentiable proxy with correct gradient direction
    cos_sim = F.cosine_similarity(v_predicted, v_target.detach(), dim=-1)  # [B]
    loss = -(advantage * cos_sim).mean()

    metrics = {
        "reinforce_loss": loss.item(),
        "advantage_mean": advantage.mean().item(),
        "advantage_std": advantage.std().item(),
        "reward_cos_mean": cos_sim.mean().item(),
        "reward_baseline": baseline if baseline is not None else reward_detached.mean().item(),
    }
    return loss, metrics


class RewardBaseline:
    """Exponential moving average baseline for REINFORCE variance reduction."""

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self._value: float | None = None

    @property
    def value(self) -> float | None:
        return self._value

    def update(self, reward_mean: float) -> float:
        """Update baseline and return current value."""
        if self._value is None:
            self._value = reward_mean
        else:
            self._value = self.decay * self._value + (1 - self.decay) * reward_mean
        return self._value


def _compute_bleu_batch(
    predictions: list[str],
    references: list[str],
) -> list[float]:
    """Compute sentence-level BLEU scores."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        return [0.0] * len(predictions)

    smoother = SmoothingFunction().method1
    scores = []
    for pred, ref in zip(predictions, references):
        try:
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            if not pred_tokens or not ref_tokens:
                scores.append(0.0)
                continue
            score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoother)
            scores.append(float(score))
        except Exception:
            scores.append(0.0)
    return scores
