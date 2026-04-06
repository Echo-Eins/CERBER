"""
Conditional Angular Energy Critic — semantic QA evaluation on the unit sphere.

Key architectural insight: decompose energy into angular (semantic) and radial
(manifold) components.  This critic handles the ANGULAR part — it operates on
normalized vectors, so its gradient is *tangential* to the sphere by
construction (the Jacobian of normalization is the tangent-plane projector).

Features fed to the MLP:
    [q̂, v̂, q̂−v̂, q̂∗v̂, ctx̂, ctx̂∗v̂, cos(q,v), cos(ctx,v), θ(q,v), σ_embed]

where  q̂ = v_query / ‖v_query‖ ,  etc.

Why normalization matters:
  ∂v̂/∂v = (I − v̂v̂ᵀ) / ‖v‖        (Jacobian of L2-normalization)
  ⇒  ∇_v E_angular  is always ⊥ v̂   (tangential to the sphere)
  The radial guard (separate module) handles ‖v‖ → target_norm.
  Together, angular + radial gradients are ORTHOGONAL — zero interference.

Training losses (all used during training):
    1. Focal-InfoNCE contrastive: E(q, answer, ctx) < E(q, distractor, ctx)
    2. Direction loss (gradient supervision): -∇E should point toward target
    3. Cosine reconstruction (metric only, no gradient into landscape)

Design rules from lessons.md applied:
  • Zero-init output layer (stable E ≈ 0 at start)
  • No learnable energy scale (lessons: "NEVER use learnable global energy scale")
  • Energy clamping to prevent OOD explosion
  • Direction loss bounded [0, 2] — safer than MDSM (no Hessian-vector products)
  • Direction loss with warmup (lessons: "NEVER combine 2nd-order and 1st-order
    losses from start")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── σ embedding ─────────────────────────────────────────────────
_SIGMA_EMBED_FREQS = 4
_SIGMA_EMBED_DIM = _SIGMA_EMBED_FREQS * 2  # sin + cos = 8


@dataclass
class ConditionalAngularCriticConfig:
    """Configuration for the Conditional Angular Energy Critic."""

    dim: int = 1024
    """Embedding dimension (SONAR = 1024)."""

    hidden_dims: list[int] = field(default_factory=lambda: [2048, 1024, 512])
    """Hidden layer sizes of the energy MLP."""

    activation: str = "silu"
    """Activation function: silu | gelu | relu."""

    dropout: float = 0.1
    """Dropout between hidden layers."""

    use_layer_norm: bool = True
    """LayerNorm after each hidden layer (stabilises contrastive training)."""

    energy_output_clamp: float = 50.0
    """Symmetric clamp on output energy.  Prevents OOD explosions
    (lessons: E_predicted=38 when unclamped).  With τ=0.07 the useful
    logit range is ±50/0.07 ≈ ±714 — more than enough for softmax."""

    # ── Context integration ──
    include_context: bool = True
    """If True, v_context features are part of the input.  Set False
    for context-free evaluation (falls back to v_query as context)."""

    # ── Contrastive loss ──
    temperature: float = 0.07
    """InfoNCE temperature τ."""

    focal_gamma: float = 2.0
    """Focal-InfoNCE γ  (0 = standard InfoNCE)."""

    num_negatives: int = 7
    """Number of negatives per positive in contrastive loss."""


class ConditionalAngularCritic(nn.Module):
    """
    Context-aware angular energy critic for QA on the unit sphere.

    E(v_query, v_candidate, v_context, σ) → scalar

    Lower energy ⇔ v_candidate is a better answer to v_query given context.
    Gradient ∇_{v_candidate} E is always **tangential** to the sphere at
    v_candidate (guaranteed by normalization Jacobian).

    Parameters: ~13 M at default settings.
    """

    def __init__(self, cfg: ConditionalAngularCriticConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = ConditionalAngularCriticConfig()
        self.cfg = cfg

        # Input dimension depends on whether context is included
        #   With context: [q̂, v̂, q̂−v̂, q̂∗v̂, ctx̂, ctx̂∗v̂, cos_qv, cos_cv, θ_qv, σ]
        #                  4×D + 2×D + 3 scalars + 8 σ = 6D + 11
        #   Without:      [q̂, v̂, q̂−v̂, q̂∗v̂, cos_qv, θ_qv, σ]
        #                  4×D + 2 + 8 = 4D + 10
        if cfg.include_context:
            input_dim = cfg.dim * 6 + 3 + _SIGMA_EMBED_DIM
        else:
            input_dim = cfg.dim * 4 + 2 + _SIGMA_EMBED_DIM

        # Sinusoidal σ frequencies
        freqs = torch.arange(1, _SIGMA_EMBED_FREQS + 1, dtype=torch.float32) * math.pi
        self.register_buffer("_sigma_freqs", freqs)

        # ── Build MLP ────────────────────────────────────────────
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in cfg.hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if cfg.use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(_make_activation(cfg.activation))
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            prev_dim = h_dim

        # Final projection → scalar energy
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

        # Zero-init output layer for stable start (E ≈ 0 everywhere).
        # From lessons.md: "Always zero-init output layer of energy networks."
        with torch.no_grad():
            out_layer = self.net[-1]
            nn.init.zeros_(out_layer.weight)
            if out_layer.bias is not None:
                nn.init.zeros_(out_layer.bias)

    # ── helpers ──────────────────────────────────────────────────

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _make_activation(name: str) -> nn.Module:  # pragma: no cover – alias
        return _make_activation(name)

    def _embed_sigma(self, sigma: Tensor) -> Tensor:
        """Sinusoidal embedding of noise level σ.  [B, 1] → [B, 8]."""
        log_sigma = torch.log(sigma.clamp(min=1e-8))
        args = log_sigma * self._sigma_freqs          # [B, 4]
        return torch.cat([args.sin(), args.cos()], dim=-1)  # [B, 8]

    def _estimate_sigma(self, v_query: Tensor, v_candidate: Tensor) -> Tensor:
        """Estimate relative noise level from query–candidate distance."""
        with torch.no_grad():
            dist = (v_candidate - v_query).norm(dim=-1, keepdim=True)
            q_norm = v_query.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            return dist / q_norm

    # ── feature construction ─────────────────────────────────────

    def _build_features(
        self,
        v_query: Tensor,
        v_candidate: Tensor,
        v_context: Tensor | None,
        sigma: Tensor,
    ) -> Tensor:
        """
        Build angular feature vector on the unit sphere.

        All directional vectors are L2-normalised ⇒ the MLP sees only
        semantic *directions*, never magnitudes.  The chain-rule through
        ``F.normalize`` projects the output gradient onto the tangent plane
        automatically.
        """
        _EPS = 1e-6  # clamp for acos stability

        q_hat = F.normalize(v_query, dim=-1)       # [B, D]
        v_hat = F.normalize(v_candidate, dim=-1)    # [B, D]

        diff_qv = q_hat - v_hat                     # [B, D]
        prod_qv = q_hat * v_hat                      # [B, D]

        # Scalar summaries (detach-safe: used for info, gradient flows
        # through the vector components above)
        cos_qv = (q_hat * v_hat).sum(dim=-1, keepdim=True)  # [B, 1]
        cos_qv = cos_qv.clamp(-1.0 + _EPS, 1.0 - _EPS)
        theta_qv = torch.acos(cos_qv)                        # [B, 1]

        sigma_emb = self._embed_sigma(sigma)                  # [B, 8]

        if self.cfg.include_context and v_context is not None:
            ctx_hat = F.normalize(v_context, dim=-1)          # [B, D]
            prod_cv = ctx_hat * v_hat                          # [B, D]
            cos_cv = (ctx_hat * v_hat).sum(dim=-1, keepdim=True)  # [B, 1]
            cos_cv = cos_cv.clamp(-1.0 + _EPS, 1.0 - _EPS)
            return torch.cat([
                q_hat, v_hat, diff_qv, prod_qv,
                ctx_hat, prod_cv,
                cos_qv, cos_cv, theta_qv,
                sigma_emb,
            ], dim=-1)
        else:
            return torch.cat([
                q_hat, v_hat, diff_qv, prod_qv,
                cos_qv, theta_qv,
                sigma_emb,
            ], dim=-1)

    # ── forward ──────────────────────────────────────────────────

    def forward(
        self,
        v_query: Tensor,
        v_candidate: Tensor,
        v_context: Tensor | None = None,
        sigma: Tensor | None = None,
    ) -> Tensor:
        """
        Compute conditional angular energy.

        Args:
            v_query:     [B, D] question embedding
            v_candidate: [B, D] candidate answer embedding
            v_context:   [B, D] context (mean of reasoning steps).
                         Falls back to v_query when None.
            sigma:       [B, 1] noise level.  Auto-estimated if None.

        Returns:
            [B] energy scalars.  Lower = better answer.
        """
        if v_context is None:
            v_context = v_query

        if sigma is None:
            sigma = self._estimate_sigma(v_query, v_candidate)
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)

        features = self._build_features(v_query, v_candidate, v_context, sigma)
        raw = self.net(features).squeeze(-1)             # [B]

        return raw.clamp(-self.cfg.energy_output_clamp,
                         self.cfg.energy_output_clamp)

    # ── gradient ─────────────────────────────────────────────────

    def energy_and_grad(
        self,
        v_query: Tensor,
        v_candidate: Tensor,
        v_context: Tensor | None = None,
        sigma: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute energy **and** ∇_{v_candidate} E.

        The returned gradient is tangential to the unit sphere at
        v_candidate (enforced by the normalisation Jacobian, not by
        explicit projection).

        Returns:
            (energy [B],  grad [B, D])
        """
        if sigma is None:
            sigma = self._estimate_sigma(v_query, v_candidate)

        v_req = v_candidate.detach().requires_grad_(True)
        energy = self.forward(v_query, v_req, v_context=v_context,
                              sigma=sigma.detach())
        grad = torch.autograd.grad(energy.sum(), v_req,
                                   create_graph=False)[0]
        return energy.detach(), grad.detach()

    # ── training losses ──────────────────────────────────────────

    def compute_contrastive_loss(
        self,
        v_query: Tensor,       # [B, D]
        v_positive: Tensor,    # [B, D]
        v_negatives: Tensor,   # [B, N, D]
        v_context: Tensor | None = None,  # [B, D]
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Focal-InfoNCE contrastive loss.

        E(q, answer, ctx) should be *lower* than E(q, distractor, ctx).
        """
        B, N, D = v_negatives.shape
        tau = self.cfg.temperature
        gamma = self.cfg.focal_gamma

        # Positive energy  [B]
        E_pos = self.forward(v_query, v_positive, v_context=v_context)

        # Negative energies  [B, N]
        v_q_exp = v_query.unsqueeze(1).expand_as(v_negatives).reshape(B * N, D)
        v_neg_flat = v_negatives.reshape(B * N, D)
        v_ctx_exp = None
        if v_context is not None:
            v_ctx_exp = v_context.unsqueeze(1).expand(B, N, D).reshape(B * N, D)
        E_neg = self.forward(v_q_exp, v_neg_flat,
                             v_context=v_ctx_exp).reshape(B, N)

        # Logits: lower energy ⇒ higher logit
        logit_pos = -E_pos / tau                 # [B]
        logits_neg = -E_neg / tau                 # [B, N]
        all_logits = torch.cat([logit_pos.unsqueeze(1), logits_neg], dim=1)  # [B, 1+N]

        if gamma > 0:
            # Focal weighting: (1−p)^γ  — upweight hard negatives
            with torch.no_grad():
                probs = F.softmax(all_logits, dim=1)
                focal_w = (1.0 - probs).pow(gamma)
                focal_w = focal_w / focal_w.sum(dim=1, keepdim=True) * (1 + N)

            log_probs = F.log_softmax(all_logits, dim=1)
            loss = -(focal_w[:, 0] * log_probs[:, 0]).mean()
        else:
            labels = torch.zeros(B, dtype=torch.long, device=v_query.device)
            loss = F.cross_entropy(all_logits, labels)

        with torch.no_grad():
            rank_acc = (E_pos.unsqueeze(1) < E_neg).float().mean().item()
            energy_gap = (E_neg.mean(dim=1) - E_pos).mean().item()

        metrics = {
            "contrastive_loss": loss.item(),
            "rank_acc": rank_acc,
            "E_pos_mean": E_pos.mean().item(),
            "E_neg_mean": E_neg.mean().item(),
            "energy_gap": energy_gap,
        }
        return loss, metrics

    def compute_direction_loss(
        self,
        v_query: Tensor,           # [B, D]
        v_noisy: Tensor,           # [B, D]
        v_target: Tensor,          # [B, D]
        v_context: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Gradient direction supervision: −∇E should point toward v_target.

            L_dir = 1 − cos(−∇E(q, v_noisy, ctx),  v_target − v_noisy)

        Bounded ∈ [0, 2]  (safer than MDSM which has unbounded Hessian-vector
        products).  Requires ``create_graph=True`` for second-order backprop.

        From lessons.md: "direction_loss (cosine) preferred over MDSM (MSE);
        Phase 1.5 (+ direction_loss λ=0.3): cosine success 60.55%."
        """
        v_req = v_noisy.detach().requires_grad_(True)
        sigma = self._estimate_sigma(v_query, v_req)
        energy = self.forward(v_query, v_req, v_context=v_context,
                              sigma=sigma.detach())
        grad = torch.autograd.grad(energy.sum(), v_req,
                                   create_graph=True)[0]

        # Target direction (full-space, not normalised)
        target_dir = (v_target - v_noisy).detach()

        # Skip degenerate samples where target_dir ≈ 0
        target_norm = target_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Cosine between −∇E and target direction
        cos_sim = ((-grad / grad_norm) * (target_dir / target_norm)).sum(dim=-1)
        loss = (1.0 - cos_sim).mean()

        return loss, {
            "direction_loss": loss.item(),
            "direction_cos": cos_sim.mean().item(),
        }

    def compute_path_contrastive_loss(
        self,
        v_query: Tensor,           # [B, D]
        v_answer: Tensor,          # [B, D]
        v_context: Tensor | None = None,
        num_waypoints: int = 5,
        waypoint_noise: float = 0.02,
        margin: float = 0.1,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Path-contrastive loss: enforce monotonically decreasing energy
        along the geodesic from v_query → v_answer.

        For waypoints at t_i along the SLERP path (t=0 at query, t=1 at answer):
            E(q, v_{t_i}) < E(q, v_{t_j})  when t_i > t_j

        This is **1st-order only** (no create_graph) and covers the actual
        Langevin navigation path, unlike direction_loss which only covers a
        neighborhood of v_answer.

        NOT the same as interp_gp (lessons: Phase 2e):
          - interp_gp penalised ||∇E||² → flattened gradients → killed Langevin
          - This enforces energy ORDERING → creates monotonic descent → gradients
            naturally point toward answer

        Implemented as softplus margin loss on adjacent waypoint pairs.
        """
        B, D = v_query.shape
        _EPS = 1e-6

        # Waypoint positions along geodesic: t ∈ (0, 1]
        # Skip t=0 (query itself), include t=1 (answer)
        t_values = torch.linspace(
            1.0 / num_waypoints, 1.0, num_waypoints,
            device=v_query.device, dtype=v_query.dtype,
        )  # e.g. [0.2, 0.4, 0.6, 0.8, 1.0] for num_waypoints=5

        # SLERP on unit sphere
        q_hat = F.normalize(v_query, dim=-1)     # [B, D]
        a_hat = F.normalize(v_answer, dim=-1)     # [B, D]

        cos_angle = (q_hat * a_hat).sum(dim=-1, keepdim=True)       # [B, 1]
        cos_angle = cos_angle.clamp(-1.0 + _EPS, 1.0 - _EPS)
        angle = torch.acos(cos_angle)                                # [B, 1]
        sin_angle = angle.sin().clamp(min=_EPS)                      # [B, 1]

        # Compute energies at each waypoint
        energies = []
        for t in t_values:
            # SLERP: w_t = sin((1-t)θ)/sin(θ) · q̂ + sin(tθ)/sin(θ) · â
            w_q = torch.sin((1.0 - t) * angle) / sin_angle  # [B, 1]
            w_a = torch.sin(t * angle) / sin_angle           # [B, 1]
            w_t = w_q * q_hat + w_a * a_hat                  # [B, D]

            # Add small tangent-space noise for robustness
            if waypoint_noise > 0:
                noise = torch.randn_like(w_t)
                # Project noise to tangent space at w_t
                w_t_hat = F.normalize(w_t, dim=-1)
                noise = noise - (noise * w_t_hat).sum(dim=-1, keepdim=True) * w_t_hat
                w_t = w_t + waypoint_noise * noise
                w_t = F.normalize(w_t, dim=-1)

            e = self.forward(v_query, w_t, v_context=v_context)  # [B]
            energies.append(e)

        # Softplus margin loss on adjacent pairs: E(closer) < E(farther)
        total_loss = torch.zeros(1, device=v_query.device, dtype=v_query.dtype)
        violation_count = 0.0
        num_pairs = 0

        for i in range(1, len(energies)):
            e_near = energies[i]      # higher t = closer to answer → lower energy
            e_far = energies[i - 1]   # lower t = farther from answer → higher energy
            # softplus(e_near - e_far + margin): 0 when e_near << e_far
            pair_loss = F.softplus(e_near - e_far + margin)
            total_loss = total_loss + pair_loss.mean()

            with torch.no_grad():
                violation_count += (e_near > e_far).float().mean().item()
            num_pairs += 1

        total_loss = total_loss / max(num_pairs, 1)

        with torch.no_grad():
            # Energy at answer (last waypoint) vs energy at first waypoint
            e_answer = energies[-1].mean().item()
            e_first = energies[0].mean().item()
            path_gap = e_first - e_answer  # positive = correct ordering

        metrics = {
            "path_loss": total_loss.item(),
            "path_violations": violation_count / max(num_pairs, 1),
            "path_energy_gap": path_gap,
            "path_E_answer": e_answer,
            "path_E_first": e_first,
        }
        return total_loss, metrics

    def compute_cosine_loss(
        self,
        v_predicted: Tensor,   # [B, D]
        v_target: Tensor,      # [B, D]
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Cosine reconstruction metric.  No gradient into the energy landscape
        — purely informational (how close did Langevin get?).
        """
        cos_sim = F.cosine_similarity(v_predicted, v_target, dim=-1)
        loss = (1.0 - cos_sim).mean()
        return loss, {
            "cos_sim_mean": cos_sim.mean().item(),
            "cos_sim_min": cos_sim.min().item(),
            "cos_sim_max": cos_sim.max().item(),
        }


# ── private helpers ──────────────────────────────────────────────

def _make_activation(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")
