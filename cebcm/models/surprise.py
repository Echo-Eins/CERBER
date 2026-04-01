"""
Surprise Predictor: SSM-based next-vector prediction for surprise scoring.

Predicts the next SONAR vector in a sequence. The prediction error
(surprise score) determines which vectors become Global Tokens — visible
to all positions through direct attention in the ContextEncoder.

Surprise Score:
  S_t = (1 - cos_sim(V̂_t, V_t)) / 2    ∈ [0, 1]

  - S ≈ 0 → vector is predictable, routine information
  - S > θ → vector is surprising, new critical information → GLOBAL TOKEN

Training: fully self-supervised (no labels needed).
  Loss = MSE(V̂_t, V_t) + λ_cos · (1 - cos_sim(V̂_t, V_t))

After training, the SurprisePredictor is FROZEN forever.
The rest of the pipeline adapts to its surprise scores.

Architecture:
  SSMBackbone (2 layers, d_state=64) → Linear prediction head → V̂

At inference:
  - Incremental mode: one vector per step, O(1) per update
  - Returns surprise score + updates internal state

Spec reference: §9.5 (Google Titans-inspired)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cebcm.models.ssm import SSMConfig, SSMBackbone


@dataclass
class SurpriseConfig:
    """Configuration for SurprisePredictor."""
    d_model: int = 1024  # SONAR embedding dim
    ssm_d_state: int = 64  # SSM state dimension
    ssm_d_conv: int = 4  # Conv width
    ssm_expand: int = 2  # Expansion factor
    ssm_n_layers: int = 2  # SSM depth
    ssm_dropout: float = 0.0  # No dropout (simple model)
    # Prediction head
    pred_hidden: int = 1024  # Hidden dim of prediction MLP
    # Training
    cos_loss_weight: float = 0.5  # Weight of cosine loss vs MSE
    # Surprise threshold
    threshold_mode: str = "percentile"  # "percentile" or "fixed"
    threshold_percentile: float = 95.0  # Top 5% = global tokens
    threshold_fixed: float = 0.3  # Fixed threshold if mode="fixed"


class SurprisePredictor(nn.Module):
    """
    SSM-based next-vector predictor for surprise scoring.

    Given [V₁, ..., V_{t-1}], predicts V̂_t. The prediction error
    is the surprise score, used to flag Global Tokens.
    """

    def __init__(self, cfg: SurpriseConfig):
        super().__init__()
        self.cfg = cfg

        # SSM backbone for sequence processing
        ssm_cfg = SSMConfig(
            d_model=cfg.d_model,
            d_state=cfg.ssm_d_state,
            d_conv=cfg.ssm_d_conv,
            expand=cfg.ssm_expand,
            n_layers=cfg.ssm_n_layers,
            dropout=cfg.ssm_dropout,
        )
        self.ssm = SSMBackbone(ssm_cfg)

        # Prediction head: hidden state → predicted next vector
        self.pred_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.pred_hidden),
            nn.GELU(),
            nn.Linear(cfg.pred_hidden, cfg.d_model),
        )

        # Incremental inference state
        self._ssm_states: list[tuple[Tensor, Tensor]] | None = None
        self._last_hidden: Tensor | None = None

    def predict_next(self, sequence: Tensor) -> Tensor:
        """
        Predict next vector at each position in the sequence.

        Args:
            sequence: [B, L, D] input SONAR vectors

        Returns:
            predictions: [B, L, D] predicted next vector at each position
                         predictions[:, t] ≈ sequence[:, t+1]
        """
        hidden = self.ssm(sequence)  # [B, L, D]
        return self.pred_head(hidden)  # [B, L, D]

    def compute_surprise(self, sequence: Tensor) -> Tensor:
        """
        Compute surprise scores for each position (except first).

        Args:
            sequence: [B, L, D] input SONAR vectors

        Returns:
            surprise: [B, L-1] surprise scores ∈ [0, 1]
        """
        predictions = self.predict_next(sequence[:, :-1])  # [B, L-1, D]
        actual = sequence[:, 1:]  # [B, L-1, D]

        cos_sim = F.cosine_similarity(predictions, actual, dim=-1)  # [B, L-1]
        surprise = (1.0 - cos_sim) / 2.0  # Normalize to [0, 1]
        return surprise

    def compute_loss(
            self, sequence: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Self-supervised training loss: predict next vector.

        Args:
            sequence: [B, L, D] sequence of SONAR vectors

        Returns:
            loss: scalar training loss
            metrics: diagnostic values
        """
        predictions = self.predict_next(sequence[:, :-1])  # [B, L-1, D]
        targets = sequence[:, 1:]  # [B, L-1, D]

        loss_mse = F.mse_loss(predictions, targets)
        loss_cos = (1.0 - F.cosine_similarity(predictions, targets, dim=-1)).mean()
        loss = loss_mse + self.cfg.cos_loss_weight * loss_cos

        with torch.no_grad():
            surprise = (1.0 - F.cosine_similarity(predictions, targets, dim=-1)) / 2.0
            metrics = {
                "surprise_loss": loss.item(),
                "surprise_mse": loss_mse.item(),
                "surprise_cos": loss_cos.item(),
                "surprise_mean": surprise.mean().item(),
                "surprise_std": surprise.std().item(),
                "surprise_max": surprise.max().item(),
            }

        return loss, metrics

    def get_threshold(self, surprise_scores: Tensor) -> Tensor:
        """
        Compute the surprise threshold for flagging Global Tokens.

        Args:
            surprise_scores: [B, L] surprise values

        Returns:
            threshold: [B] per-batch threshold
        """
        if self.cfg.threshold_mode == "percentile":
            pct = self.cfg.threshold_percentile / 100.0
            return torch.quantile(surprise_scores.float(), pct, dim=-1)
        else:
            return torch.full(
                (surprise_scores.shape[0],),
                self.cfg.threshold_fixed,
                device=surprise_scores.device,
            )

    def flag_globals(self, surprise_scores: Tensor) -> Tensor:
        """
        Flag which positions should be Global Tokens.

        Args:
            surprise_scores: [B, L] surprise values

        Returns:
            is_global: [B, L] boolean mask
        """
        threshold = self.get_threshold(surprise_scores)
        return surprise_scores > threshold.unsqueeze(-1)

    # ---- Incremental inference ----

    def reset_state(self) -> None:
        """Reset incremental state for new dialogue."""
        self._ssm_states = None
        self._last_hidden = None

    def step(self, v_new: Tensor) -> tuple[float, Tensor]:
        """
        Incremental inference: process one new vector.

        Call this for each new SONAR vector in the dialogue.
        Returns the surprise score for this vector and the predicted next.

        Args:
            v_new: [B, D] new SONAR vector

        Returns:
            surprise: float — surprise score for v_new
            v_predicted: [B, D] — prediction for the NEXT vector
        """
        # Compute surprise: compare previous prediction with actual
        if self._last_hidden is not None:
            v_pred_prev = self.pred_head(self._last_hidden)
            cos_sim = F.cosine_similarity(v_pred_prev, v_new, dim=-1).mean().item()
            surprise = (1.0 - cos_sim) / 2.0
        else:
            surprise = 0.5  # No prediction for first vector

        # Update SSM state
        hidden, self._ssm_states = self.ssm.step(v_new, self._ssm_states)
        self._last_hidden = hidden

        # Predict next vector
        v_predicted = self.pred_head(hidden)

        return surprise, v_predicted

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())