from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cebcm.models.chain_head import _apply_rope, _build_rope_cache


@dataclass
class ChainGeneratorConfig:
    """Configuration for autoregressive chain generation in SONAR space."""

    d_model: int = 1024
    n_heads: int = 8
    n_layers: int = 6
    dim_feedforward: int = 4096
    max_chain_len: int = 20
    dropout: float = 0.1
    target_norm: float = 0.2051

    # Supervised teacher-forcing step loss.
    loss_cosine_weight: float = 1.0
    loss_mse_weight: float = 0.1


class CrossAttention(nn.Module):
    """Multi-head cross-attention over context bank [B, K, D]."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, x: Tensor, context: Tensor, context_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: [B, L, D]
            context: [B, K, D]
            context_mask: optional [B, K] bool/float (1=valid, 0=masked)
        """
        bsz, seq_len, d_model = x.shape
        ctx_len = context.shape[1]

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(bsz, ctx_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(bsz, ctx_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if context_mask is not None:
            cm = context_mask.to(device=x.device)
            if cm.shape != (bsz, ctx_len):
                raise ValueError(
                    f"context_mask shape {tuple(cm.shape)} does not match context shape {(bsz, ctx_len)}"
                )
            # SDPA additive mask: 0 for valid, -inf for invalid.
            invalid = cm <= 0
            attn_mask = torch.zeros((bsz, self.n_heads, seq_len, ctx_len), device=x.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(invalid[:, None, None, :], float("-inf"))

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            scale=self.head_dim ** -0.5,
        )

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        return self.out_proj(out)


class CausalRoPESelfAttention(nn.Module):
    """Causal self-attention with RoPE."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 32,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

        rope_cache = _build_rope_cache(max_seq_len, self.head_dim, torch.device("cpu"))
        self.register_buffer("_rope_cache", rope_cache, persistent=False)
        self._cached_seq_len = max_seq_len

    def _get_rope(self, seq_len: int, device: torch.device) -> Tensor:
        if seq_len <= self._cached_seq_len:
            return self._rope_cache[:seq_len].to(device)
        rope = _build_rope_cache(seq_len, self.head_dim, device)
        self._rope_cache = rope
        self._cached_seq_len = seq_len
        return rope

    def forward(self, x: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        rope = self._get_rope(seq_len, x.device)
        q = _apply_rope(q, rope)
        k = _apply_rope(k, rope)

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=True,
            scale=self.head_dim ** -0.5,
        )

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(out)


class DecoderBlock(nn.Module):
    """Pre-norm decoder block: self-attn -> cross-attn -> ffn."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        max_seq_len: int = 32,
    ):
        super().__init__()

        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = CausalRoPESelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, context: Tensor, context_mask: Tensor | None = None) -> Tensor:
        x = x + self.self_attn(self.norm_self(x))
        x = x + self.cross_attn(self.norm_cross(x), context, context_mask=context_mask)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class ChainGenerator(nn.Module):
    """Autoregressive transformer decoder in SONAR embedding space."""

    def __init__(self, cfg: ChainGeneratorConfig | None = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else ChainGeneratorConfig()

        # Initialize start_token to have standard normal variance (norm ≈ sqrt(D) ≈ 32)
        # to correctly match the scaled residual stream magnitude.
        self.start_token = nn.Parameter(torch.randn(1, 1, self.cfg.d_model))

        max_seq = self.cfg.max_chain_len + 1
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=self.cfg.d_model,
                    n_heads=self.cfg.n_heads,
                    dim_feedforward=self.cfg.dim_feedforward,
                    dropout=self.cfg.dropout,
                    max_seq_len=max_seq,
                )
                for _ in range(self.cfg.n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(self.cfg.d_model)
        self.output_proj = nn.Linear(self.cfg.d_model, self.cfg.d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.01)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def _safe_normalize(self, v: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
        """Numerically safe normalization that prevents NaN gradients.

        F.normalize with default eps=1e-12 can produce exploding gradients
        when input norm approaches zero (especially in bfloat16). We use a
        larger eps and clamp the norm to prevent this.
        """
        v_float = v.float()
        norms = v_float.norm(dim=dim, keepdim=True).clamp(min=eps)
        return (v_float / norms).to(dtype=v.dtype)

    def _sphere_project(self, v: Tensor) -> Tensor:
        return self._safe_normalize(v, dim=-1) * self.cfg.target_norm

    def _to_residual_space(self, sonar_vectors: Tensor) -> Tensor:
        """
        SONAR embeddings have norm ~0.2 (variance ~4e-5). If fed directly into 
        the residual stream or cross-attn, they are completely obliterated by 
        LayerNorm-scaled self-attention updates (norm ~32, variance ~1.0).
        This scales them up to naturally match standard Transformer variance.
        """
        import math
        scale = math.sqrt(self.cfg.d_model) / self.cfg.target_norm
        return sonar_vectors * scale

    def _prepare_context(
        self,
        v_query: Tensor,
        v_context_bank: Tensor | None,
        context_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        bsz, d_model = v_query.shape

        if v_context_bank is None:
            context = v_query.unsqueeze(1)
            context = self._to_residual_space(context)
            mask = torch.ones((bsz, 1), device=v_query.device, dtype=torch.bool)
            return context, mask

        context = v_context_bank
        context = self._to_residual_space(context)
        if context.dim() == 2:
            context = context.unsqueeze(1)
        if context.dim() != 3:
            raise ValueError(f"v_context_bank must have shape [B,K,D] or [B,D], got {tuple(context.shape)}")
        if context.shape[0] != bsz or context.shape[2] != d_model:
            raise ValueError(
                f"v_context_bank shape {tuple(context.shape)} incompatible with v_query {(bsz, d_model)}"
            )

        if context_mask is None:
            mask = torch.ones((bsz, context.shape[1]), device=v_query.device, dtype=torch.bool)
        else:
            mask = context_mask.to(device=v_query.device)
            if mask.shape != (bsz, context.shape[1]):
                raise ValueError(
                    f"context_mask shape {tuple(mask.shape)} does not match context shape {(bsz, context.shape[1])}"
                )
            mask = mask > 0
        return context, mask

    def forward(
        self,
        v_query: Tensor,
        v_target_chain: Tensor,
        v_context_bank: Tensor | None = None,
        context_mask: Tensor | None = None,
        scheduled_sampling_prob: float = 0.0,
    ) -> Tensor:
        """Teacher-forced forward pass with optional scheduled sampling.

        When scheduled_sampling_prob > 0 and training, each position (except
        the first) independently uses the model's own prediction instead of
        ground truth with probability ``scheduled_sampling_prob``.
        This bridges the teacher-forcing / free-run distribution gap.
        """
        bsz, num_steps, _ = v_target_chain.shape
        context, ctx_mask = self._prepare_context(v_query, v_context_bank, context_mask)

        ss_prob = float(scheduled_sampling_prob)
        use_ss = self.training and ss_prob > 0.0 and num_steps > 1

        if not use_ss:
            # Pure teacher forcing (original path).
            start = self.start_token.expand(bsz, -1, -1)
            scaled_target = self._to_residual_space(v_target_chain)
            decoder_input = torch.cat([start, scaled_target[:, :-1, :]], dim=1)

            x = decoder_input
            for layer in self.layers:
                x = layer(x, context, context_mask=ctx_mask)

            x = self.final_norm(x)
            v_pred = self.output_proj(x)
            return v_pred

        # ── Scheduled sampling: step-by-step with token mixing ──
        seq = self.start_token.expand(bsz, -1, -1)  # [B, 1, D]
        preds: list[Tensor] = []

        for t in range(num_steps):
            x = seq
            for layer in self.layers:
                x = layer(x, context, context_mask=ctx_mask)

            x = self.final_norm(x)
            raw = self.output_proj(x[:, -1:, :])  # [B, 1, D]
            pred_t = self._sphere_project(raw)
            preds.append(raw)

            if t < num_steps - 1:
                # Decide per-sample: use own prediction or ground truth.
                # Generate random threshold for each sample in batch.
                rand_vals = torch.rand(bsz, 1, 1, device=x.device)
                use_pred = rand_vals < ss_prob
                
                # Context sequence must hold residual-scaled vectors
                scaled_gt = self._to_residual_space(v_target_chain[:, t : t + 1, :])
                # DETACH the prediction being used as context to prevent recursive BPTT
                # across Transformer layers!
                scaled_noisy_pred = self._to_residual_space(pred_t).detach()
                next_vec = torch.where(use_pred, scaled_noisy_pred, scaled_gt)
                
                seq = torch.cat([seq, next_vec], dim=1)

        return torch.cat(preds, dim=1)

    def generate(
        self,
        v_query: Tensor,
        num_steps: int = 1,
        v_context_bank: Tensor | None = None,
        context_mask: Tensor | None = None,
        temperature: float = 1.0,
        latent_noise_std: float = 0.0,
        start_noise_std: float = 0.0,
        repeat_penalty: float = 0.0,
        repeat_cos_threshold: float = 0.98,
        repeat_ban_threshold: float = 0.995,
        repeat_ban_max_retries: int = 4,
        energy_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
        target_vec: Tensor | None = None,
        stagnation_patience: int = 0,
        stagnation_delta_energy: float = 1e-4,
        stagnation_delta_cos: float = 1e-4,
        convergence_cos: float = 0.0,
        convergence_window: int = 2,
        chain_prefix: Tensor | None = None,
        return_info: bool = False,
        num_candidates: int = 1,
        oracle_guide: Tensor | None = None,
        oracle_max_retries: int = 0,
        oracle_prob: float = 0.0,
    ) -> Tensor | tuple[Tensor, dict[str, float | int | bool]]:
        """
        Autoregressive generation with adaptive stopping and resume support.

        Stopping modes (checked in order):
          1. convergence_cos > 0: stop when the last `convergence_window`
             consecutive outputs all have pairwise cosine > convergence_cos.
             This is the SONAR-space equivalent of EOS — the model has
             converged to its answer and keeps repeating it.
          2. stagnation_patience > 0: stop when energy/cosine metrics are flat.
          3. num_steps reached (hard cap).

        Resume mode:
          If chain_prefix is provided [B, P, D], generation resumes from that
          chain instead of starting from start_token. This enables the
          "append final vector and keep going" workflow.

        Returns:
            chain [B, T, D] or (chain, info) when return_info=True.
        """
        bsz, _ = v_query.shape
        context, ctx_mask = self._prepare_context(v_query, v_context_bank, context_mask)

        temp = max(float(temperature), 1e-4)
        latent_noise_std = max(float(latent_noise_std), 0.0)
        start_noise_std = max(float(start_noise_std), 0.0)
        repeat_penalty = max(float(repeat_penalty), 0.0)
        repeat_cos_threshold = float(repeat_cos_threshold)
        repeat_ban_threshold = float(repeat_ban_threshold)
        repeat_ban_max_retries = max(int(repeat_ban_max_retries), 0)
        stagnation_patience = max(int(stagnation_patience), 0)
        convergence_cos = float(convergence_cos)
        convergence_window = max(2, int(convergence_window))

        # Resume from prefix or start fresh.
        if chain_prefix is not None:
            if chain_prefix.dim() == 2:
                chain_prefix = chain_prefix.unsqueeze(0)  # [1, P, D]
            if chain_prefix.shape[0] == 1 and bsz > 1:
                chain_prefix = chain_prefix.expand(bsz, -1, -1)
            start = self.start_token.expand(bsz, -1, -1)
            chain = torch.cat([start, chain_prefix], dim=1)
            # Pre-seed generated list with prefix vectors for anti-loop checks.
            generated: list[Tensor] = [
                chain_prefix[:, i : i + 1, :] for i in range(chain_prefix.shape[1])
            ]
        else:
            chain = self.start_token.expand(bsz, -1, -1)
            if start_noise_std > 0.0:
                chain = chain + start_noise_std * torch.randn_like(chain)
            generated = []

        energy_hist: list[float] = []
        cos_hist: list[float] = []
        raw_norms: list[float] = []
        early_stop = False
        early_stop_step = -1
        early_stop_reason = ""
        repeat_resamples = 0

        t_target = None
        if target_vec is not None:
            t_target = target_vec
            if t_target.dim() == 1:
                t_target = t_target.unsqueeze(0)
            t_target = t_target.to(device=v_query.device)

        steps = max(1, int(num_steps))
        for step_idx in range(steps):
            x = chain
            for layer in self.layers:
                x = layer(x, context, context_mask=ctx_mask)

            x = self.final_norm(x)
            raw_next = self.output_proj(x[:, -1:, :])

            raw_norms.append(float(raw_next.detach().norm(dim=-1).mean().item()))

            # Repeat penalty: ONLY at inference. During training, this pushes
            # raw_next toward zero norm, causing F.normalize gradient explosion
            # in bfloat16 (the primary NaN collapse trigger at E13+).
            # The penalty provides no useful gradient (history is detached), and
            # Diversity must be handled by scheduled sampling/noise in training.
            if generated and repeat_penalty > 0.0 and not self.training:
                hist = torch.cat(generated, dim=1).detach()
                cos_hist_tensor = (
                    self._safe_normalize(raw_next.expand(-1, hist.shape[1], -1), dim=-1)
                    * self._safe_normalize(hist, dim=-1)
                ).sum(dim=-1)
                max_cos, max_idx = cos_hist_tensor.max(dim=1)
                over = (max_cos - repeat_cos_threshold).clamp(min=0.0)
                if torch.any(over > 0):
                    repel = hist[torch.arange(bsz, device=hist.device), max_idx].unsqueeze(1)
                    raw_next = raw_next - repeat_penalty * over.view(bsz, 1, 1) * repel

            # Scale noise by temperature: higher temp → more noise, lower temp → less.
            noise_std = latent_noise_std * temp

            clean_next_vec = self._sphere_project(raw_next)

            # Oracle-guided DAgger with probability decay.
            # oracle_prob < 1.0 means some samples skip oracle guidance entirely,
            # forcing the model to learn robust generation without oracle dependency.
            # This prevents val roll_cos_last degradation caused by train-only
            # oracle reliance (oracle is disabled at eval since self.training=False).
            import random as _random
            use_oracle = (
                oracle_guide is not None
                and oracle_max_retries > 0
                and self.training
                and noise_std > 0.0
                and _random.random() < oracle_prob
            )
            if use_oracle:
                t_idx = min(step_idx, oracle_guide.shape[1] - 1)
                t_step = oracle_guide[:, t_idx, :]  # [B, D]

                best_cand = clean_next_vec.clone()
                best_cos = (
                    self._safe_normalize(best_cand.squeeze(1), dim=-1)
                    * self._safe_normalize(t_step, dim=-1)
                ).sum(dim=-1)
                jitter_std = max(noise_std, 0.01)

                for _ in range(oracle_max_retries):
                    cand_noisy = raw_next + jitter_std * torch.randn_like(raw_next)
                    cand_proj = self._sphere_project(cand_noisy)
                    cand_cos = (
                        self._safe_normalize(cand_proj.squeeze(1), dim=-1)
                        * self._safe_normalize(t_step, dim=-1)
                    ).sum(dim=-1)

                    improved = cand_cos > best_cos
                    if improved.any():
                        best_cos = torch.where(improved, cand_cos, best_cos)
                        best_cand = torch.where(improved.view(bsz, 1, 1), cand_proj, best_cand)

                next_vec_for_chain = best_cand

            elif num_candidates > 1 and energy_fn is not None and noise_std > 0.0:
                best_noisy_vecs = []
                for b in range(bsz):
                    raw_b = raw_next[b : b + 1]  # [1, 1, D]
                    raw_k = raw_b.expand(int(num_candidates), -1, -1).clone()
                    raw_k_noisy = raw_k + noise_std * torch.randn_like(raw_k)
                    cand_vecs = self._sphere_project(raw_k_noisy)  # [K, 1, D]
                    
                    q_b = v_query[b : b + 1].expand(int(num_candidates), -1)  # [K, D]
                    with torch.no_grad():
                        e_vals = energy_fn(q_b, cand_vecs.squeeze(1))  # [K]
                    
                    best_idx = e_vals.argmin()
                    best_noisy_vecs.append(cand_vecs[best_idx : best_idx + 1])
                
                next_vec_for_chain = torch.cat(best_noisy_vecs, dim=0)
            elif noise_std > 0.0:
                raw_next_noisy = raw_next + noise_std * torch.randn_like(raw_next)
                next_vec_for_chain = self._sphere_project(raw_next_noisy)
            else:
                next_vec_for_chain = clean_next_vec

            # Repeat-ban is inference-only. In training it creates a second
            # train/eval mismatch and adds stochastic resampling to the rollout
            # context while the loss still supervises the clean raw prediction.
            if (
                generated
                and repeat_ban_threshold < 1.0
                and repeat_ban_max_retries > 0
                and not self.training
            ):
                hist = torch.cat(generated, dim=1)
                jitter_std = max(noise_std, 0.01)
                for _ in range(repeat_ban_max_retries):
                    cos_to_hist = (
                        self._safe_normalize(next_vec_for_chain.expand(-1, hist.shape[1], -1), dim=-1)
                        * self._safe_normalize(hist, dim=-1)
                    ).sum(dim=-1)
                    max_cos = cos_to_hist.max(dim=1).values
                    repeat_mask = max_cos > repeat_ban_threshold
                    if not torch.any(repeat_mask):
                        break

                    candidate_noisy = raw_next + jitter_std * torch.randn_like(raw_next)
                    candidate = self._sphere_project(candidate_noisy)
                    mask3 = repeat_mask.view(bsz, 1, 1)
                    next_vec_for_chain = torch.where(mask3, candidate, next_vec_for_chain)
                    repeat_resamples += int(repeat_mask.sum().item())

            # Mathematical Fix: The model's TRUE prediction is the clean vector.
            # We must evaluate the loss (and report metrics) on the clean vector.
            # In contrast, the context chain gets the NOISY / RE-ROLLED vector.
            # If we evaluated the noisy vector during training, we'd penalize random
            # noise it couldn't predict.
            if self.training:
                # Training: track the raw unprojected logit to compute MSE gradients.
                generated.append(raw_next)
            else:
                # Inference: track the ACTUAL vector chosen by the Critic/Repeat-Ban!
                generated.append(next_vec_for_chain)

            # NaN guard: if generated vector contains NaN, replace with the
            # previous valid vector (or start_token projection). This prevents
            # a single NaN from cascading through the entire chain.
            if torch.isnan(next_vec_for_chain).any():
                if len(generated) >= 2:
                    next_vec_for_chain = generated[-2].detach().clone()
                    next_vec_for_chain = self._sphere_project(next_vec_for_chain)
                else:
                    next_vec_for_chain = self._sphere_project(
                        torch.randn(bsz, 1, self.cfg.d_model, device=v_query.device)
                    )
                # Also fix the recorded generated vector
                if self.training:
                    generated[-1] = next_vec_for_chain / self.cfg.target_norm  # undo sphere project for raw logit scale
                else:
                    generated[-1] = next_vec_for_chain

            # Fix #3: Detach before appending so backward() through L_roll
            # only goes one step deep, not through the entire autoregressive chain.
            # Scale the next_vec_for_chain UP to residual space before appending!
            scaled_next = self._to_residual_space(next_vec_for_chain)
            chain = torch.cat([chain, scaled_next.detach()], dim=1)

            # ── Convergence check (SONAR-space EOS) ──
            # If the last W outputs are all mutually similar (cos > threshold),
            # the model has converged to its answer. This is trained via
            # answer-repeat padding: [steps..., answer, answer, answer].
            if convergence_cos > 0 and len(generated) >= convergence_window:
                window = torch.cat(generated[-convergence_window:], dim=1)  # [B, W, D]
                # Check all consecutive pairs in window.
                w1 = window[:, :-1, :]   # [B, W-1, D]
                w2 = window[:, 1:, :]    # [B, W-1, D]
                pair_cos = (self._safe_normalize(w1, dim=-1) * self._safe_normalize(w2, dim=-1)).sum(dim=-1)
                # Converged if ALL pairs exceed threshold (mean across batch).
                min_pair_cos = pair_cos.min(dim=1).values.mean().item()
                if min_pair_cos > convergence_cos:
                    early_stop = True
                    early_stop_step = step_idx + 1
                    early_stop_reason = "convergence"
                    break

            if energy_fn is not None:
                try:
                    e_val = energy_fn(v_query, clean_next_vec.squeeze(1))
                    if torch.is_tensor(e_val):
                        energy_hist.append(float(e_val.detach().mean().item()))
                    else:
                        energy_hist.append(float(e_val))
                except Exception as exc:
                    import warnings
                    if step_idx == 0:
                        warnings.warn(f"energy_fn failed at step 0: {exc}", stacklevel=2)

            if t_target is not None:
                cos_val = (
                    self._safe_normalize(clean_next_vec.squeeze(1), dim=-1)
                    * self._safe_normalize(t_target, dim=-1)
                ).sum(dim=-1)
                cos_hist.append(float(cos_val.mean().item()))

            if stagnation_patience > 0 and step_idx + 1 >= (stagnation_patience + 1):
                checks = []
                if len(energy_hist) > stagnation_patience:
                    de = abs(energy_hist[-1] - energy_hist[-1 - stagnation_patience])
                    checks.append(de <= stagnation_delta_energy)
                if len(cos_hist) > stagnation_patience:
                    dc = abs(cos_hist[-1] - cos_hist[-1 - stagnation_patience])
                    checks.append(dc <= stagnation_delta_cos)
                stagnated = len(checks) > 0 and all(checks)
                if stagnated:
                    early_stop = True
                    early_stop_step = step_idx + 1
                    early_stop_reason = "stagnation"
                    break

        # Only return newly generated steps (exclude prefix).
        prefix_len = chain_prefix.shape[1] if chain_prefix is not None else 0
        new_generated = generated[prefix_len:]
        chain_out = torch.cat(new_generated, dim=1) if new_generated else torch.zeros(
            bsz, 0, v_query.shape[-1], device=v_query.device
        )

        if not return_info:
            return chain_out

        raw_norm_mean = sum(raw_norms) / len(raw_norms) if raw_norms else 0.0

        info: dict[str, float | int | bool | str] = {
            "early_stop": early_stop,
            "early_stop_step": int(early_stop_step),
            "early_stop_reason": early_stop_reason,
            "repeat_resamples": int(repeat_resamples),
            "steps_generated": int(chain_out.shape[1]),
            "prefix_len": int(prefix_len),
            "temperature": float(temp),
            "latent_noise_std": float(latent_noise_std),
            "start_noise_std": float(start_noise_std),
            "repeat_penalty": float(repeat_penalty),
            "raw_norm_mean": float(raw_norm_mean),
        }
        if energy_hist:
            info["energy_final"] = float(energy_hist[-1])
        if cos_hist:
            info["cos_final"] = float(cos_hist[-1])

        return chain_out, info

    def beam_generate(
        self,
        v_query: Tensor,
        num_steps: int = 1,
        beam_width: int = 4,
        num_candidates: int = 4,
        v_context_bank: Tensor | None = None,
        context_mask: Tensor | None = None,
        temperature: float = 1.0,
        noise_std: float = 0.05,
        energy_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> Tensor:
        """
        Energy-Guided Beam Search for Autoregressive Generation.

        Maintains `beam_width` active reasoning trajectories. At each step, expands
        each trajectory with `num_candidates` noisy extensions, evaluates them
        using `energy_fn`, and prunes back to `beam_width` by cumulative energy.
        """
        bsz, _ = v_query.shape
        if energy_fn is None:
            raise ValueError("beam_generate requires energy_fn to score paths")

        context, ctx_mask = self._prepare_context(v_query, v_context_bank, context_mask)
        temp = max(float(temperature), 1e-4)
        n_std = max(float(noise_std) * temp, 0.0)

        # active_chains: [B, W, t, D]. Starts at t=1 (start_token only)
        active_chains = self.start_token.expand(bsz, 1, 1, -1)  # [B, 1, 1, D]
        active_energies = torch.zeros(bsz, 1, device=v_query.device)  # [B, 1]
        
        W = 1
        steps = max(1, int(num_steps))
        K = max(1, int(num_candidates))
        
        for step_idx in range(steps):
            B_W = bsz * W
            flat_chains = active_chains.reshape(B_W, -1, self.cfg.d_model)
            
            flat_context = context.repeat_interleave(W, dim=0)
            flat_ctx_mask = ctx_mask.repeat_interleave(W, dim=0)

            x = flat_chains
            for layer in self.layers:
                x = layer(x, flat_context, context_mask=flat_ctx_mask)

            x = self.final_norm(x)
            raw_next = self.output_proj(x[:, -1:, :])  # [B*W, 1, D]

            new_W = W * K
            raw_k = raw_next.reshape(bsz, W, 1, self.cfg.d_model).unsqueeze(2).expand(-1, -1, K, -1, -1).clone()
            
            if n_std > 0.0:
                raw_k_noisy = raw_k + n_std * torch.randn_like(raw_k)
            else:
                raw_k_noisy = raw_k
                
            cand_vecs = self._sphere_project(raw_k_noisy)  # [B, W, K, 1, D]

            flat_cands = cand_vecs.reshape(bsz * new_W, self.cfg.d_model)
            flat_qs = v_query.repeat_interleave(new_W, dim=0)
            
            with torch.no_grad():
                step_energies = energy_fn(flat_qs, flat_cands)  # [B*W*K]
            
            step_energies = step_energies.reshape(bsz, W, K)
            
            # Cumulative energy over the chain path
            cum_energies = active_energies.unsqueeze(2) + step_energies  # [B, W, K]
            cum_energies_flat = cum_energies.reshape(bsz, new_W)
            
            next_W = min(int(beam_width), new_W)
            top_energies, top_indices = torch.topk(cum_energies_flat, next_W, dim=1, largest=False)  # [B, next_W]
            
            reconstructed_cands = cand_vecs.reshape(bsz, new_W, 1, self.cfg.d_model)
            t = active_chains.shape[2]
            history = active_chains.unsqueeze(2).expand(-1, -1, K, -1, -1).reshape(bsz, new_W, t, self.cfg.d_model)
            
            next_chains = []
            for b in range(bsz):
                idx = top_indices[b]
                b_hist = history[b, idx]  # [next_W, t, D]
                b_cand = reconstructed_cands[b, idx]  # [next_W, 1, D]
                b_new = torch.cat([b_hist, b_cand], dim=1)  # [next_W, t+1, D]
                next_chains.append(b_new)
                
            active_chains = torch.stack(next_chains, dim=0)  # [B, next_W, t+1, D]
            active_energies = top_energies
            W = next_W
            
        # Return best chain per batch item (idx 0), excluding start_token (pos 0)
        best_chains = active_chains[:, 0, 1:, :]  # [B, num_steps, D]
        return best_chains

    def compute_loss(
        self,
        v_query: Tensor,
        v_target_chain: Tensor,
        loss_mask: Tensor | None = None,
        v_context_bank: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Masked teacher-forcing loss."""
        v_pred = self.forward(
            v_query,
            v_target_chain,
            v_context_bank=v_context_bank,
            context_mask=context_mask,
        )

        bsz, num_steps, _ = v_pred.shape
        if loss_mask is None:
            mask = torch.ones(bsz, num_steps, device=v_pred.device, dtype=v_pred.dtype)
        else:
            mask = loss_mask.to(device=v_pred.device, dtype=v_pred.dtype)
            if mask.shape != (bsz, num_steps):
                raise ValueError(
                    f"loss_mask shape {tuple(mask.shape)} does not match (B,N)=({bsz},{num_steps})"
                )

        mask_bool = mask > 0
        mask_sum = mask.sum().clamp(min=1.0)

        v_pred_f = v_pred.float()
        v_target_f = v_target_chain.float()
        cos_sim = (
            self._safe_normalize(v_pred_f, dim=-1)
            * self._safe_normalize(v_target_f, dim=-1)
        ).sum(dim=-1)
        cos_term = torch.where(mask_bool, 1.0 - cos_sim, torch.zeros_like(cos_sim))
        cos_loss = cos_term.sum() / mask_sum

        mse_per_step = (v_pred_f - v_target_f).pow(2).sum(dim=-1)
        mse_term = torch.where(mask_bool, mse_per_step, torch.zeros_like(mse_per_step))
        mse_loss = mse_term.sum() / mask_sum

        loss = self.cfg.loss_cosine_weight * cos_loss + self.cfg.loss_mse_weight * mse_loss

        with torch.no_grad():
            valid_per_step = mask.sum(dim=0).clamp(min=1.0)
            sample_valid_counts = mask.sum(dim=1).long().clamp(min=1)
            last_idx = (sample_valid_counts - 1).clamp(min=0)
            cos_masked = torch.where(mask_bool, cos_sim, torch.zeros_like(cos_sim))
            pred_norm = v_pred.norm(dim=-1)
            pred_norm_masked = torch.where(mask_bool, pred_norm, torch.zeros_like(pred_norm))
            cos_last = cos_sim.gather(1, last_idx.unsqueeze(1)).squeeze(1).mean().item()
            cos_first = cos_sim[:, 0].mean().item()

            metrics = {
                "loss": float(loss.item()),
                "cos_loss": float(cos_loss.item()),
                "mse_loss": float(mse_loss.item()),
                "cos_sim_mean": float((cos_masked.sum() / mask_sum).item()),
                "cos_sim_last": float(cos_last),
                "cos_sim_first": float(cos_first),
                "pred_norm_mean": float((pred_norm_masked.sum() / mask_sum).item()),
                "valid_tokens": float(mask_sum.item()),
                "valid_tokens_per_sample": float(sample_valid_counts.float().mean().item()),
                "cos_step0": float((
                    torch.where(mask_bool[:, 0], cos_sim[:, 0], torch.zeros_like(cos_sim[:, 0])).sum()
                    / mask[:, 0].sum().clamp(min=1.0)
                ).item()),
                "cos_step_last_masked": float(cos_last),
                "cos_step_mean_masked": float((cos_masked.sum() / mask_sum).item()),
                "valid_steps": float(mask_sum.item()),
                "valid_steps_per_sample": float(sample_valid_counts.float().mean().item()),
            }

        return loss, metrics

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
