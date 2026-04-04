"""
Inference Diagnostics — comprehensive inference analysis backend.

Provides:
  1. Dual checkpoint loading (pairwise + chain head)
  2. System 1/2 inference with full diagnostics
  3. Attention capture at every chain evaluation step (System 2)
  4. Energy landscape with Langevin trajectory
  5. Numerical metrics export (JSON)
  6. SONAR text encode/decode for chat interface
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch import Tensor


# ─── Preset test questions ──────────────────────────────────────────

PRESET_QUESTIONS = [
    {"label": "Simple factual", "text": "The capital of France is Paris."},
    {"label": "Complex reasoning", "text": "If all mammals are warm-blooded and whales are mammals, then whales are warm-blooded."},
    {"label": "Negation", "text": "The sun does not rise in the west."},
    {"label": "Analogy", "text": "A doctor is to a hospital as a teacher is to a school."},
    {"label": "Contradiction", "text": "The object is both completely red and completely blue at the same time."},
    {"label": "Temporal", "text": "World War II ended before the first moon landing."},
    {"label": "Causal", "text": "Rain causes the ground to become wet."},
    {"label": "Abstract", "text": "Freedom is more valuable than comfort."},
]


# ─── Data Classes ───────────────────────────────────────────────────

@dataclass
class InferenceResult:
    """All diagnostics from a single inference run."""
    mode: str  # "system1" or "system2"
    # Input/Output vectors
    v_query: Tensor | None = None
    v_init: Tensor | None = None
    v_final: Tensor | None = None
    v_target: Tensor | None = None
    # Text (if SONAR available)
    input_text: str | None = None
    output_text: str | None = None
    target_text: str | None = None
    # Core metrics
    cos_init: float = 0.0
    cos_final: float = 0.0
    delta_cos: float = 0.0
    energy_init: float = 0.0
    energy_final: float = 0.0
    num_steps: int = 0
    # Trajectories
    energy_trajectory: list[float] = field(default_factory=list)
    cos_trajectory: list[float] = field(default_factory=list)
    v_trajectory: list[Tensor] = field(default_factory=list)
    # System 2 specifics
    chain_energies: list[float] = field(default_factory=list)
    backtrack_count: int = 0
    # Attention snapshots: list of {"step": int, "maps": list[Tensor], "chain_len": int}
    attention_snapshots: list[dict] = field(default_factory=list)
    # Timing
    elapsed_ms: float = 0.0


@dataclass
class DiagnosticsState:
    """Holds loaded models and last result."""
    pairwise_model: nn.Module | None = None
    pairwise_type: str = "simple"
    pairwise_checkpoint: dict | None = None
    chain_head: nn.Module | None = None
    chain_head_info: dict | None = None
    sonar: Any = None
    device: str = "cpu"
    last_result: InferenceResult | None = None
    sys1_result: InferenceResult | None = None  # For "both" mode comparison


# Module-level state
_state = DiagnosticsState()


# ─── Model Loading ───────���──────────────────────────────────────────

def _load_pairwise_from_checkpoint(ckpt_path: str, device: torch.device):
    """Standalone pairwise model loader (no circular import with app.py)."""
    from cebcm.models.energy import SimpleEnergy
    from cebcm.models.energy_decomposed import AngularEnergyCritic, RadialEnergyCritic

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) or {}
    model_state = ckpt.get("model_state_dict") or ckpt.get("model") or {}

    # Detect model type
    model_type = ckpt.get("model_type", "simple")

    # Stage 1.5 twin critics
    if "critic1_state" in ckpt:
        critic_arch = str(cfg.get("critic_architecture", "homogeneous"))
        dim = cfg.get("energy_dim", 1024)

        if critic_arch == "radial_angular":
            c1 = AngularEnergyCritic(
                dim=dim,
                hidden_dims=cfg.get("angular_hidden_dims", [2048, 1024, 512]),
                norm_mode=cfg.get("angular_norm_mode", "none"),
                activation=cfg.get("angular_activation", "silu"),
            ).to(device)
            c1.load_state_dict(ckpt["critic1_state"])
            c1.eval()
            return c1, "angular", ckpt
        else:
            c1 = SimpleEnergy(
                dim=dim,
                hidden_dims=cfg.get("energy_hidden_dims", [2048, 1024, 512]),
                norm_mode=cfg.get("norm_mode", "none"),
                activation=cfg.get("activation", "silu"),
            ).to(device)
            c1.load_state_dict(ckpt["critic1_state"])
            c1.eval()
            return c1, "simple_twin", ckpt

    # SimpleEnergy or unconditional
    if model_state:
        dim = cfg.get("dim", cfg.get("energy_dim", 1024))
        hidden = cfg.get("hidden_dims", cfg.get("energy_hidden_dims", [2048, 1024, 512]))
        model = SimpleEnergy(
            dim=dim,
            hidden_dims=hidden,
            norm_mode=cfg.get("norm_mode", "none"),
            activation=cfg.get("activation", "silu"),
        ).to(device)
        model.load_state_dict(model_state)
        model.eval()
        return model, model_type, ckpt

    raise KeyError(f"Cannot find model state. Keys: {list(ckpt.keys())}")


def load_pairwise_model(ckpt_path: str, device: str = "auto") -> str:
    """Load pairwise energy model from checkpoint."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _state.device = device

    path = Path(ckpt_path)
    if not path.exists():
        return f"File not found: {ckpt_path}"

    dev = torch.device(device)
    model, model_type, checkpoint = _load_pairwise_from_checkpoint(str(path), dev)

    _state.pairwise_model = model
    _state.pairwise_type = model_type
    _state.pairwise_checkpoint = checkpoint

    n_params = sum(p.numel() for p in model.parameters()) if isinstance(model, nn.Module) else 0
    cfg = checkpoint.get("config", {}) or {}
    return (
        f"Pairwise model loaded: {model_type}, {n_params:,} params\n"
        f"Device: {device}\n"
        f"Config: {json.dumps({k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))}, indent=2)[:500]}"
    )


def load_chain_head_model(ckpt_path: str, device: str = "auto") -> str:
    """Load Chain Head from checkpoint."""
    from cerber_gui.chain_diagnostics import load_chain_head_from_checkpoint

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _state.device = device

    path = Path(ckpt_path)
    if not path.exists():
        return f"File not found: {ckpt_path}"

    dev = torch.device(device)
    model, info = load_chain_head_from_checkpoint(str(path), dev)
    model.eval()

    _state.chain_head = model
    _state.chain_head_info = info

    p = info["params"]
    ep = info["epoch"]
    ra = info["best_rank_acc"]
    vm = info.get("val_metrics") or {}
    vm_str = json.dumps({k: round(v, 4) for k, v in vm.items()})[:400]
    return (
        f"Chain Head loaded: {p:,} params\n"
        f"Epoch: {ep}, Best rank_acc: {ra}\n"
        f"Val metrics: {vm_str}"
    )


def load_sonar(device: str = "auto") -> str:
    """Lazy-load SONAR encoder/decoder."""
    if _state.sonar is not None:
        return "SONAR already loaded."
    try:
        from cebcm.models.sonar_wrapper import SONARWrapper
        if device == "auto":
            device = _state.device
        _state.sonar = SONARWrapper(device=device)
        return f"SONAR loaded on {device}"
    except Exception as e:
        return f"SONAR load failed: {e}"


def get_state() -> DiagnosticsState:
    """Access module state."""
    return _state


# ─── Inference Core ─────────────────────────────────────────────────

def _extract_attention(model: nn.Module, chain: Tensor) -> list[Tensor]:
    """Extract attention weights from Chain Head layers."""
    from cerber_gui.chain_diagnostics import extract_attention_weights
    return extract_attention_weights(model, chain)


@torch.no_grad()
def _compute_pairwise_energy(v_query: Tensor, v: Tensor) -> float:
    """Compute pairwise energy scalar."""
    if _state.pairwise_model is None:
        return 0.0
    E = _state.pairwise_model(v_query, v)
    return E.mean().item()


def run_system1_diagnostics(
    v_query: Tensor,
    v_init: Tensor,
    v_target: Tensor | None = None,
    max_steps: int = 50,
    lr: float = 0.01,
    noise_scale: float = 0.005,
    target_norm: float = 0.2051,
) -> InferenceResult:
    """Run System 1 with full trajectory tracking."""
    from cebcm.inference.langevin import run_langevin

    if _state.pairwise_model is None:
        raise RuntimeError("Pairwise model not loaded")

    dev = torch.device(_state.device)
    v_query = v_query.to(dev)
    v_init = v_init.to(dev)
    if v_target is not None:
        v_target = v_target.to(dev)

    t0 = time.time()

    result = run_langevin(
        method="pid",
        energy_fn=_state.pairwise_model,
        v_query=v_query,
        v_init=v_init,
        max_steps=max_steps,
        lr=lr,
        noise_scale=noise_scale,
        target_norm=target_norm,
        v_target=v_target,
        track_vectors=True,
    )

    elapsed = (time.time() - t0) * 1000

    cos_init = F.cosine_similarity(v_init, v_target, dim=-1).mean().item() if v_target is not None else 0.0
    cos_final = F.cosine_similarity(result.v_final, v_target, dim=-1).mean().item() if v_target is not None else 0.0

    ir = InferenceResult(
        mode="system1",
        v_query=v_query.cpu(),
        v_init=v_init.cpu(),
        v_final=result.v_final.cpu(),
        v_target=v_target.cpu() if v_target is not None else None,
        cos_init=cos_init,
        cos_final=cos_final,
        delta_cos=cos_final - cos_init,
        energy_init=result.trajectory[0] if result.trajectory else 0.0,
        energy_final=result.trajectory[-1] if result.trajectory else 0.0,
        num_steps=result.num_steps,
        energy_trajectory=result.trajectory,
        cos_trajectory=result.cos_trajectory,
        v_trajectory=[v.cpu() for v in result.v_trajectory] if result.v_trajectory else [],
        elapsed_ms=elapsed,
    )
    _state.last_result = ir
    return ir


def run_system2_diagnostics(
    v_query: Tensor,
    v_init: Tensor,
    v_target: Tensor | None = None,
    max_steps: int = 200,
    lr: float = 0.01,
    noise_scale: float = 0.005,
    target_norm: float = 0.2051,
    chain_eval_every: int = 5,
    backtrack_patience: int = 30,
    max_chain_len: int = 20,
) -> InferenceResult:
    """
    Run System 2 with attention capture at every chain evaluation.

    Custom loop matching system_switching.run_system2() but with added diagnostics:
    - Full v_trajectory tracking
    - Attention snapshot at each chain eval step
    """
    if _state.pairwise_model is None:
        raise RuntimeError("Pairwise model not loaded")
    if _state.chain_head is None:
        raise RuntimeError("Chain Head not loaded")

    dev = torch.device(_state.device)
    v_query = v_query.to(dev)
    v_init = v_init.to(dev)
    if v_target is not None:
        v_target = v_target.to(dev)

    pairwise_fn = _state.pairwise_model
    chain_head = _state.chain_head

    t0 = time.time()

    # State
    v_current = v_init.clone()
    v_best = v_current.clone()
    best_chain_energy = float("inf")

    chain_buffer: list[Tensor] = [v_init.detach().clone()]
    energy_trajectory: list[float] = []
    chain_energies: list[float] = []
    cos_trajectory: list[float] = []
    v_trajectory: list[Tensor] = [v_init.detach().cpu().clone()]
    attention_snapshots: list[dict] = []
    backtrack_count = 0
    steps_since_improve = 0
    just_backtracked = False

    for step in range(max_steps):
        # Langevin step using pairwise energy
        v_current = v_current.detach().requires_grad_(True)
        E_pair = pairwise_fn(v_query, v_current)
        grad = torch.autograd.grad(E_pair.sum(), v_current)[0]

        with torch.no_grad():
            noise = noise_scale * torch.randn_like(v_current)
            v_current = v_current - lr * grad + noise
            if target_norm is not None:
                v_current = F.normalize(v_current, dim=-1) * target_norm

        energy_trajectory.append(E_pair.mean().item())
        v_trajectory.append(v_current.detach().cpu().clone())

        if v_target is not None:
            with torch.no_grad():
                cos = F.cosine_similarity(v_current, v_target, dim=-1).mean().item()
                cos_trajectory.append(cos)

        # Chain validation every N steps
        if (step + 1) % chain_eval_every == 0:
            if not just_backtracked:
                chain_buffer.append(v_current.detach().clone())
            just_backtracked = False

            if len(chain_buffer) > max_chain_len:
                chain_buffer = chain_buffer[-max_chain_len:]

            if len(chain_buffer) >= 3:
                chain_tensor = torch.stack(chain_buffer, dim=1)  # [B, L, D]
                with torch.no_grad():
                    E_chain = chain_head(chain_tensor).mean().item()
                chain_energies.append(E_chain)

                # Extract attention at this evaluation point
                try:
                    attn_maps = _extract_attention(chain_head, chain_tensor)
                    attention_snapshots.append({
                        "step": step + 1,
                        "maps": [m.cpu() for m in attn_maps],
                        "chain_len": chain_tensor.shape[1],
                        "chain_energy": E_chain,
                    })
                except Exception:
                    pass  # Don't break inference for attention extraction failures

                if E_chain < best_chain_energy:
                    best_chain_energy = E_chain
                    v_best = v_current.detach().clone()
                    steps_since_improve = 0
                else:
                    steps_since_improve += chain_eval_every

                if steps_since_improve >= backtrack_patience:
                    v_current = v_best.clone()
                    backtrack_count += 1
                    steps_since_improve = 0
                    just_backtracked = True
                    chain_buffer = chain_buffer[:max(3, len(chain_buffer) // 2)]
                    chain_buffer.append(v_best.detach().clone())

    elapsed = (time.time() - t0) * 1000

    cos_init = F.cosine_similarity(v_init, v_target, dim=-1).mean().item() if v_target is not None else 0.0
    cos_final = F.cosine_similarity(v_best, v_target, dim=-1).mean().item() if v_target is not None else 0.0

    ir = InferenceResult(
        mode="system2",
        v_query=v_query.cpu(),
        v_init=v_init.cpu(),
        v_final=v_best.cpu(),
        v_target=v_target.cpu() if v_target is not None else None,
        cos_init=cos_init,
        cos_final=cos_final,
        delta_cos=cos_final - cos_init,
        energy_init=energy_trajectory[0] if energy_trajectory else 0.0,
        energy_final=energy_trajectory[-1] if energy_trajectory else 0.0,
        num_steps=max_steps,
        energy_trajectory=energy_trajectory,
        cos_trajectory=cos_trajectory,
        v_trajectory=v_trajectory,
        chain_energies=chain_energies,
        backtrack_count=backtrack_count,
        attention_snapshots=attention_snapshots,
        elapsed_ms=elapsed,
    )
    _state.last_result = ir
    return ir


def run_inference(
    mode: str = "system2",
    data_path: str = "data/squad_sequences.pt",
    seq_idx: int = 0,
    noise_pct: float = 5.0,
    max_steps: int = 200,
    lr: float = 0.01,
    noise_scale: float = 0.005,
    target_norm: float = 0.2051,
    chain_eval_every: int = 5,
    backtrack_patience: int = 30,
    max_chain_len: int = 20,
) -> InferenceResult:
    """Run inference on a sample from SONAR data."""
    dev = torch.device(_state.device)

    # Load data
    raw = torch.load(data_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "sequences" in raw:
        seqs = raw["sequences"]
        seq = seqs[seq_idx % len(seqs)]
        # Use first vector as query, second as target
        v_query = seq[0:1].to(dev)  # [1, D]
        v_target = seq[1:2].to(dev) if seq.shape[0] > 1 else v_query.clone()
    elif isinstance(raw, dict) and "vectors" in raw:
        vectors = raw["vectors"]
        v_query = vectors[seq_idx % len(vectors)].unsqueeze(0).to(dev)
        v_target = vectors[(seq_idx + 1) % len(vectors)].unsqueeze(0).to(dev)
    else:
        raise ValueError(f"Unknown data format: {type(raw)}")

    # Add noise to create init
    noise_rel = noise_pct / 100.0
    v_noisy = v_query + noise_rel * v_query.norm() * torch.randn_like(v_query)
    if target_norm:
        v_noisy = F.normalize(v_noisy, dim=-1) * target_norm

    if mode == "system1":
        return run_system1_diagnostics(
            v_query=v_query, v_init=v_noisy, v_target=v_target,
            max_steps=max_steps, lr=lr, noise_scale=noise_scale,
            target_norm=target_norm,
        )
    elif mode == "both":
        # Run both systems, store sys1 result in _state for comparison
        sys1 = run_system1_diagnostics(
            v_query=v_query, v_init=v_noisy.clone(), v_target=v_target,
            max_steps=min(max_steps, 50), lr=lr, noise_scale=noise_scale,
            target_norm=target_norm,
        )
        _state.sys1_result = sys1
        sys2 = run_system2_diagnostics(
            v_query=v_query, v_init=v_noisy.clone(), v_target=v_target,
            max_steps=max_steps, lr=lr, noise_scale=noise_scale,
            target_norm=target_norm, chain_eval_every=chain_eval_every,
            backtrack_patience=backtrack_patience, max_chain_len=max_chain_len,
        )
        return sys2
    else:
        return run_system2_diagnostics(
            v_query=v_query, v_init=v_noisy, v_target=v_target,
            max_steps=max_steps, lr=lr, noise_scale=noise_scale,
            target_norm=target_norm, chain_eval_every=chain_eval_every,
            backtrack_patience=backtrack_patience, max_chain_len=max_chain_len,
        )


def run_text_inference(
    text: str,
    mode: str = "system2",
    noise_pct: float = 5.0,
    max_steps: int = 200,
    lr: float = 0.01,
    noise_scale: float = 0.005,
    target_norm: float = 0.2051,
    chain_eval_every: int = 5,
    backtrack_patience: int = 30,
    max_chain_len: int = 20,
) -> InferenceResult:
    """Encode text with SONAR, run inference, decode output."""
    if _state.sonar is None:
        raise RuntimeError("SONAR not loaded. Call load_sonar() first.")

    dev = torch.device(_state.device)

    # Encode
    v_clean = _state.sonar.encode([text]).to(dev)  # [1, D]
    v_query = v_clean.clone()
    v_target = v_clean.clone()

    # Add noise
    noise_rel = noise_pct / 100.0
    v_noisy = v_clean + noise_rel * v_clean.norm() * torch.randn_like(v_clean)
    if target_norm:
        v_noisy = F.normalize(v_noisy, dim=-1) * target_norm

    if mode == "system1":
        result = run_system1_diagnostics(
            v_query=v_query, v_init=v_noisy, v_target=v_target,
            max_steps=max_steps, lr=lr, noise_scale=noise_scale,
            target_norm=target_norm,
        )
    else:
        result = run_system2_diagnostics(
            v_query=v_query, v_init=v_noisy, v_target=v_target,
            max_steps=max_steps, lr=lr, noise_scale=noise_scale,
            target_norm=target_norm, chain_eval_every=chain_eval_every,
            backtrack_patience=backtrack_patience, max_chain_len=max_chain_len,
        )

    result.input_text = text

    # Decode output
    try:
        result.output_text = _state.sonar.decode(result.v_final.to(dev))[0]
        result.target_text = text
    except Exception as e:
        result.output_text = f"[decode error: {e}]"

    return result


# ─── Plotting Functions ─────────────────────────────────────────────

def create_energy_trajectory_plot(result: InferenceResult) -> go.Figure:
    """Energy over Langevin steps, with chain energy overlay for System 2."""
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]],
    )

    # Pairwise energy
    steps = list(range(len(result.energy_trajectory)))
    fig.add_trace(
        go.Scatter(
            x=steps, y=result.energy_trajectory,
            mode="lines", name="Pairwise Energy",
            line=dict(color="#2196F3", width=2),
        ),
        secondary_y=False,
    )

    # Chain energies (System 2)
    if result.chain_energies:
        chain_steps = [(i + 1) * 5 for i in range(len(result.chain_energies))]
        # Adjust if we know actual steps from attention_snapshots
        if result.attention_snapshots:
            chain_steps = [s["step"] for s in result.attention_snapshots[:len(result.chain_energies)]]
        fig.add_trace(
            go.Scatter(
                x=chain_steps, y=result.chain_energies,
                mode="lines+markers", name="Chain Energy",
                line=dict(color="#FF5722", width=2),
                marker=dict(size=6),
            ),
            secondary_y=True,
        )

    # Overlay System 1 if available (from "both" mode)
    if _state.sys1_result is not None and result.mode == "system2":
        s1 = _state.sys1_result
        if s1.energy_trajectory:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(s1.energy_trajectory))),
                    y=s1.energy_trajectory,
                    mode="lines", name="System 1 Energy",
                    line=dict(color="#4CAF50", width=2, dash="dash"),
                ),
                secondary_y=False,
            )

    fig.update_layout(
        title=f"Energy Trajectory ({result.mode}, {result.num_steps} steps)",
        xaxis_title="Step",
        template="plotly_dark",
        height=400,
    )
    fig.update_yaxes(title_text="Pairwise Energy", secondary_y=False)
    fig.update_yaxes(title_text="Chain Energy", secondary_y=True)
    return fig


def create_cosine_trajectory_plot(result: InferenceResult) -> go.Figure:
    """Cosine similarity to target over steps."""
    fig = go.Figure()

    if result.cos_trajectory:
        steps = list(range(len(result.cos_trajectory)))
        fig.add_trace(go.Scatter(
            x=steps, y=result.cos_trajectory,
            mode="lines", name="cos(v_current, v_target)",
            line=dict(color="#4CAF50", width=2),
        ))

        # Add init and final markers
        fig.add_hline(y=result.cos_init, line_dash="dash",
                      line_color="gray", annotation_text=f"init: {result.cos_init:.4f}")
        fig.add_hline(y=result.cos_final, line_dash="dash",
                      line_color="#FF9800", annotation_text=f"final: {result.cos_final:.4f}")

    # Overlay System 1 if available
    if _state.sys1_result is not None and result.mode == "system2":
        s1 = _state.sys1_result
        if s1.cos_trajectory:
            fig.add_trace(go.Scatter(
                x=list(range(len(s1.cos_trajectory))),
                y=s1.cos_trajectory,
                mode="lines", name="System 1",
                line=dict(color="#2196F3", width=2, dash="dash"),
            ))

    fig.update_layout(
        title=f"Cosine Similarity (delta={result.delta_cos:+.4f})",
        xaxis_title="Step",
        yaxis_title="Cosine Similarity",
        template="plotly_dark",
        height=400,
    )
    return fig


def create_attention_animation(result: InferenceResult, layer_idx: int = 0) -> go.Figure:
    """Animated heatmap showing attention evolution across chain evaluation steps."""
    if not result.attention_snapshots:
        fig = go.Figure()
        fig.update_layout(title="No attention snapshots (System 1 or no chain evals)")
        return fig

    frames = []
    for i, snap in enumerate(result.attention_snapshots):
        maps = snap["maps"]
        if layer_idx >= len(maps):
            continue
        attn = maps[layer_idx]  # [H, L, L]
        avg_attn = attn.mean(dim=0).numpy()  # [L, L] average over heads

        frames.append(go.Frame(
            data=[go.Heatmap(
                z=avg_attn,
                colorscale="Viridis",
                zmin=0, zmax=float(avg_attn.max()),
            )],
            name=f"step_{snap['step']}",
            layout=go.Layout(
                title=f"Attention (Layer {layer_idx}, Step {snap['step']}, Chain len={snap['chain_len']}, E={snap.get('chain_energy', 0):.4f})",
            ),
        ))

    if not frames:
        fig = go.Figure()
        fig.update_layout(title="No valid attention frames")
        return fig

    # Initial frame
    first_snap = result.attention_snapshots[0]
    first_attn = first_snap["maps"][layer_idx].mean(dim=0).numpy()

    fig = go.Figure(
        data=[go.Heatmap(z=first_attn, colorscale="Viridis", zmin=0, zmax=float(first_attn.max()))],
        frames=frames,
        layout=go.Layout(
            title=f"Attention Evolution (Layer {layer_idx})",
            template="plotly_dark",
            height=500,
            width=600,
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.15, x=0.5, xanchor="center",
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, {"frame": {"duration": 500, "redraw": True},
                                      "fromcurrent": True}]),
                    dict(label="Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}]),
                ],
            )],
            sliders=[dict(
                active=0,
                steps=[dict(args=[[f.name], {"frame": {"duration": 0, "redraw": True},
                                              "mode": "immediate"}],
                            label=f.name.replace("step_", ""),
                            method="animate")
                       for f in frames],
                x=0.1, len=0.8,
                y=-0.05,
                currentvalue=dict(prefix="Step: "),
            )],
        ),
    )
    return fig


def create_attention_grid(
    result: InferenceResult,
    snapshot_idx: int = -1,
    layer_idx: int = 0,
) -> go.Figure:
    """Grid of all attention heads at a specific chain evaluation step."""
    if not result.attention_snapshots:
        fig = go.Figure()
        fig.update_layout(title="No attention snapshots available")
        return fig

    snap = result.attention_snapshots[snapshot_idx]
    if layer_idx >= len(snap["maps"]):
        fig = go.Figure()
        fig.update_layout(title=f"Layer {layer_idx} not available")
        return fig

    attn = snap["maps"][layer_idx]  # [H, L, L]
    n_heads = attn.shape[0]
    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"Head {h}" for h in range(n_heads)],
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
    )

    for h in range(n_heads):
        r = h // cols + 1
        c = h % cols + 1
        fig.add_trace(
            go.Heatmap(
                z=attn[h].numpy(),
                colorscale="Viridis",
                showscale=(h == 0),
                zmin=0, zmax=float(attn[h].max()),
            ),
            row=r, col=c,
        )

    fig.update_layout(
        title=f"All Heads (Layer {layer_idx}, Step {snap['step']}, E={snap.get('chain_energy', 0):.4f})",
        template="plotly_dark",
        height=250 * rows,
        width=250 * cols,
    )
    return fig


def create_landscape_with_trajectory(result: InferenceResult) -> go.Figure:
    """3D energy landscape in the (v_query, v_init, v_final) plane with trajectory."""
    if _state.pairwise_model is None or result.v_query is None:
        fig = go.Figure()
        fig.update_layout(title="No pairwise model or query vector")
        return fig

    from cerber_gui.landscape_3d import scan_energy_landscape_3d, create_surface_plot

    dev = torch.device(_state.device)
    v_clean = result.v_target.to(dev) if result.v_target is not None else result.v_query.to(dev)
    v_noisy = result.v_init.to(dev)
    v_denoised = result.v_final.to(dev)

    # Build trajectory list for landscape overlay
    traj = None
    if result.v_trajectory:
        traj = [v.to(dev) for v in result.v_trajectory[::max(1, len(result.v_trajectory) // 50)]]

    landscape = scan_energy_landscape_3d(
        energy_fn=_state.pairwise_model,
        v_clean=v_clean,
        v_noisy=v_noisy,
        grid_size=40,
        range_factor=1.5,
        v_denoised=v_denoised,
        trajectory=traj,
        model_type=_state.pairwise_type,
        batch_size=100,
    )

    fig = create_surface_plot(
        landscape,
        title="Energy Landscape + Langevin Trajectory",
        colorscale="Inferno",
    )
    return fig


def create_contour_with_trajectory(result: InferenceResult) -> go.Figure:
    """2D contour plot with Langevin trajectory overlay."""
    if _state.pairwise_model is None or result.v_query is None:
        fig = go.Figure()
        fig.update_layout(title="No pairwise model or query vector")
        return fig

    from cerber_gui.landscape_3d import scan_energy_landscape_3d, create_contour_plot

    dev = torch.device(_state.device)
    v_clean = result.v_target.to(dev) if result.v_target is not None else result.v_query.to(dev)
    v_noisy = result.v_init.to(dev)
    v_denoised = result.v_final.to(dev)

    traj = None
    if result.v_trajectory:
        traj = [v.to(dev) for v in result.v_trajectory[::max(1, len(result.v_trajectory) // 50)]]

    landscape = scan_energy_landscape_3d(
        energy_fn=_state.pairwise_model,
        v_clean=v_clean,
        v_noisy=v_noisy,
        grid_size=40,
        range_factor=1.5,
        v_denoised=v_denoised,
        trajectory=traj,
        model_type=_state.pairwise_type,
        batch_size=100,
    )

    fig = create_contour_plot(
        landscape,
        title="Energy Contour + Trajectory",
        colorscale="Inferno",
        show_trajectory=True,
    )
    return fig


# ─── Metrics Export ─────────────────────────────────────────────────

def create_metrics_summary(result: InferenceResult) -> dict:
    """Flat dict of all numerical metrics for JSON export."""
    metrics = {
        "mode": result.mode,
        "cos_init": round(result.cos_init, 6),
        "cos_final": round(result.cos_final, 6),
        "delta_cos": round(result.delta_cos, 6),
        "energy_init": round(result.energy_init, 6),
        "energy_final": round(result.energy_final, 6),
        "energy_reduction_pct": round(
            (result.energy_init - result.energy_final) / max(abs(result.energy_init), 1e-8) * 100, 2
        ),
        "num_steps": result.num_steps,
        "elapsed_ms": round(result.elapsed_ms, 1),
        "v_final_norm": round(result.v_final.norm().item(), 6) if result.v_final is not None else None,
        "v_query_norm": round(result.v_query.norm().item(), 6) if result.v_query is not None else None,
        "v_init_norm": round(result.v_init.norm().item(), 6) if result.v_init is not None else None,
    }

    # System 2 extras
    if result.chain_energies:
        metrics["chain_energy_final"] = round(result.chain_energies[-1], 6)
        metrics["chain_energy_best"] = round(min(result.chain_energies), 6)
        metrics["chain_energy_init"] = round(result.chain_energies[0], 6)
        metrics["chain_evals"] = len(result.chain_energies)
    metrics["backtrack_count"] = result.backtrack_count
    metrics["attention_snapshots"] = len(result.attention_snapshots)

    # Energy trajectory stats
    if result.energy_trajectory:
        traj = result.energy_trajectory
        metrics["energy_mean"] = round(np.mean(traj), 6)
        metrics["energy_std"] = round(np.std(traj), 6)
        metrics["energy_min"] = round(min(traj), 6)
        metrics["energy_max"] = round(max(traj), 6)

    # Cosine trajectory stats
    if result.cos_trajectory:
        ct = result.cos_trajectory
        metrics["cos_mean"] = round(np.mean(ct), 6)
        metrics["cos_max"] = round(max(ct), 6)
        metrics["cos_min"] = round(min(ct), 6)

    # Text
    if result.input_text:
        metrics["input_text"] = result.input_text
    if result.output_text:
        metrics["output_text"] = result.output_text

    return metrics


def format_metrics_markdown(result: InferenceResult) -> str:
    """Format metrics as readable Markdown."""
    m = create_metrics_summary(result)

    lines = [
        f"### Inference Result ({m['mode']})",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Mode | {m['mode']} |",
        f"| Steps | {m['num_steps']} |",
        f"| Time | {m['elapsed_ms']:.1f} ms |",
        f"| cos(init, target) | {m['cos_init']:.6f} |",
        f"| cos(final, target) | {m['cos_final']:.6f} |",
        f"| Delta cos | {m['delta_cos']:+.6f} |",
        f"| Energy init | {m['energy_init']:.6f} |",
        f"| Energy final | {m['energy_final']:.6f} |",
        f"| Energy reduction | {m['energy_reduction_pct']:.1f}% |",
        f"| v_final norm | {m.get('v_final_norm', 'N/A')} |",
    ]

    if m.get("chain_energy_final") is not None:
        lines.extend([
            f"| Chain energy (final) | {m['chain_energy_final']:.6f} |",
            f"| Chain energy (best) | {m['chain_energy_best']:.6f} |",
            f"| Chain evals | {m['chain_evals']} |",
            f"| Backtracks | {m['backtrack_count']} |",
            f"| Attention snapshots | {m['attention_snapshots']} |",
        ])

    if result.input_text:
        lines.extend([
            "",
            f"**Input**: {result.input_text}",
            f"**Output**: {result.output_text}",
        ])

    # Show System 1 comparison if available (from "both" mode)
    if _state.sys1_result is not None and result.mode == "system2":
        s1 = _state.sys1_result
        lines.extend([
            "",
            "---",
            "#### System 1 Comparison (Fast Shot)",
            f"| cos(final, target) | {s1.cos_final:.6f} |",
            f"| Delta cos | {s1.delta_cos:+.6f} |",
            f"| Steps | {s1.num_steps} |",
            f"| Time | {s1.elapsed_ms:.1f} ms |",
        ])

    return "\n".join(lines)


def export_metrics_json(result: InferenceResult, path: str | None = None) -> str:
    """Export metrics to JSON. Returns JSON string."""
    m = create_metrics_summary(result)
    json_str = json.dumps(m, indent=2, ensure_ascii=False)
    if path:
        Path(path).write_text(json_str)
    return json_str


def export_metrics_csv(result: InferenceResult, path: str | None = None) -> str:
    """Export per-step trajectory data as CSV for analysis."""
    lines = ["step,pairwise_energy,cosine_to_target,chain_energy"]
    chain_step_map = {}
    if result.attention_snapshots:
        for i, snap in enumerate(result.attention_snapshots):
            if i < len(result.chain_energies):
                chain_step_map[snap["step"]] = result.chain_energies[i]

    for i in range(len(result.energy_trajectory)):
        e = result.energy_trajectory[i]
        c = result.cos_trajectory[i] if i < len(result.cos_trajectory) else ""
        ce = chain_step_map.get(i + 1, "")
        lines.append(f"{i},{e:.6f},{c},{ce}")

    csv_str = "\n".join(lines)
    if path:
        Path(path).write_text(csv_str)
    return csv_str
