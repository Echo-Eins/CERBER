"""
Chain Head Diagnostics — visualization and analysis tools.

Provides:
  1. Attention heatmaps: what does each head attend to?
  2. Energy perturbation analysis: how sensitive is E to position swaps?
  3. Per-type ranking analysis: which negatives fool the model?
  4. Chain energy trajectory: energy as chain grows step by step
  5. Train/val overfitting diagnostics from training logs
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch import Tensor

from cebcm.data.sequence_loading import load_sonar_sequences
from cebcm.models.chain_head import ChainHeadConfig, EBTChainHead


# ─── Attention Extraction ─────────────────────────────────────────────

def extract_attention_weights(
    model: EBTChainHead,
    chain: Tensor,  # [1, L, D]
) -> list[Tensor]:
    """
    Extract attention weights from all layers using forward hooks.

    Returns list of [H, L+1, L+1] tensors (one per layer).
    L+1 because CLS token is prepended.
    """
    attention_maps = []

    def make_hook(storage):
        def hook_fn(module, input, output):
            # Re-run the attention computation to capture weights
            # Works with RoPE: apply rotary to Q/K before computing scores
            from cebcm.models.chain_head import _apply_rope
            x = input[0]  # [B, L, D]
            B, L, _ = x.shape
            q = module.q_proj(x).view(B, L, module.n_heads, module.head_dim).transpose(1, 2)
            k = module.k_proj(x).view(B, L, module.n_heads, module.head_dim).transpose(1, 2)
            # Apply RoPE rotation to Q and K
            rope = module._get_rope(L, x.device)
            q = _apply_rope(q, rope)
            k = _apply_rope(k, rope)
            scale = module.head_dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            storage.append(attn[0].detach().cpu())  # [H, L, L]
        return hook_fn

    hooks = []
    for layer in model.layers:
        hooks.append(layer.attn.register_forward_hook(make_hook(attention_maps)))

    with torch.no_grad():
        model(chain)

    for h in hooks:
        h.remove()

    return attention_maps


def create_attention_heatmap(
    attention_maps: list[Tensor],
    layer_idx: int = 0,
    head_idx: int | None = None,
) -> go.Figure:
    """
    Create attention heatmap for a specific layer/head.

    If head_idx is None, averages across all heads.
    """
    attn = attention_maps[layer_idx]  # [H, L, L]

    if head_idx is not None:
        matrix = attn[head_idx].numpy()
        title = f"Layer {layer_idx}, Head {head_idx}"
    else:
        matrix = attn.mean(dim=0).numpy()
        title = f"Layer {layer_idx}, Average across {attn.shape[0]} heads"

    L = matrix.shape[0]
    labels = ["CLS"] + [f"V{i}" for i in range(L - 1)]

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        colorscale="Viridis",
        text=np.round(matrix, 3),
        texttemplate="%{text}" if L <= 15 else "",
        hovertemplate="From %{y} to %{x}: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Key (attended to)",
        yaxis_title="Query (attending from)",
        yaxis_autorange="reversed",
        width=700,
        height=600,
    )
    return fig


def create_all_heads_heatmap(
    attention_maps: list[Tensor],
    layer_idx: int = 0,
) -> go.Figure:
    """Create grid of all attention heads for a layer."""
    attn = attention_maps[layer_idx]  # [H, L, L]
    n_heads = attn.shape[0]
    L = attn.shape[1]

    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols

    labels = ["CLS"] + [f"V{i}" for i in range(L - 1)]

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"Head {i}" for i in range(n_heads)],
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
    )

    for i in range(n_heads):
        r = i // cols + 1
        c = i % cols + 1
        fig.add_trace(
            go.Heatmap(
                z=attn[i].numpy(),
                x=labels,
                y=labels,
                colorscale="Viridis",
                showscale=(i == 0),
                hovertemplate="From %{y} to %{x}: %{z:.4f}<extra></extra>",
            ),
            row=r, col=c,
        )
        fig.update_yaxes(autorange="reversed", row=r, col=c)

    fig.update_layout(
        title=f"Layer {layer_idx} — All {n_heads} Attention Heads",
        width=900,
        height=250 * rows,
    )
    return fig


# ─── Energy Perturbation Analysis ─────────────────────────────────────

def compute_swap_sensitivity(
    model: EBTChainHead,
    chain: Tensor,  # [1, L, D]
    device: torch.device,
) -> dict:
    """
    Measure how much energy changes when swapping each adjacent pair.

    If model learned ordering, swapping positions (i, i+1) should
    increase energy. If energy stays flat or decreases, the model
    doesn't understand order at that position.

    Returns dict with per-position swap deltas and the original energy.
    """
    model.eval()
    chain = chain.to(device)
    L = chain.shape[1]

    with torch.no_grad():
        E_original = model(chain).item()

    deltas = []
    for i in range(L - 1):
        swapped = chain.clone()
        swapped[0, i], swapped[0, i + 1] = chain[0, i + 1].clone(), chain[0, i].clone()
        with torch.no_grad():
            E_swapped = model(swapped).item()
        deltas.append(E_swapped - E_original)

    return {
        "E_original": E_original,
        "swap_deltas": deltas,
        "positions": list(range(L - 1)),
    }


def create_swap_sensitivity_plot(swap_data: dict) -> go.Figure:
    """Visualize energy change per adjacent swap."""
    deltas = swap_data["swap_deltas"]
    positions = swap_data["positions"]

    colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in deltas]

    fig = go.Figure(data=go.Bar(
        x=[f"{i}<->{i+1}" for i in positions],
        y=deltas,
        marker_color=colors,
        hovertemplate="Swap %{x}: dE=%{y:.6f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"Energy change per adjacent swap (E_original={swap_data['E_original']:.6f})",
        xaxis_title="Swapped positions",
        yaxis_title="Energy delta (positive = model detects swap)",
        width=800,
        height=400,
    )
    return fig


# ─── Chain Growth Energy Trajectory ───────────────────────────────────

def compute_chain_growth_energy(
    model: EBTChainHead,
    chain: Tensor,  # [1, L, D]
    device: torch.device,
) -> dict:
    """
    Compute energy as chain grows from length 3 to L.

    For a coherent chain, energy should stay low or decrease as more
    consistent evidence accumulates. Rising energy = chain degrading.
    """
    model.eval()
    chain = chain.to(device)
    L = chain.shape[1]

    energies = []
    lengths = []
    for l in range(3, L + 1):
        sub = chain[:, :l, :]
        with torch.no_grad():
            e = model(sub).item()
        energies.append(e)
        lengths.append(l)

    return {"lengths": lengths, "energies": energies}


def create_chain_growth_plot(growth_data: dict) -> go.Figure:
    """Visualize energy as chain grows."""
    fig = go.Figure(data=go.Scatter(
        x=growth_data["lengths"],
        y=growth_data["energies"],
        mode="lines+markers",
        marker=dict(size=8),
        line=dict(width=2),
        hovertemplate="Chain length %{x}: E=%{y:.6f}<extra></extra>",
    ))
    fig.update_layout(
        title="Energy vs Chain Length (coherent chain should stay low)",
        xaxis_title="Chain length",
        yaxis_title="Energy",
        width=800,
        height=400,
    )
    return fig


# ─── Positive vs Negative Comparison ─────────────────────────────────

def compute_pos_neg_comparison(
    model: EBTChainHead,
    chain: Tensor,  # [1, L, D]
    device: torch.device,
    n_trials: int = 20,
    target_norm: float = 0.2051,
) -> dict:
    """
    Compare energy of original chain vs various perturbations.

    Tests: adjacent-swap, full-shuffle, truncation, random replacement.
    """
    model.eval()
    chain = chain.to(device)
    L = chain.shape[1]

    with torch.no_grad():
        E_pos = model(chain).item()

    results = {"positive": E_pos, "negatives": {}}

    # Adjacent swaps
    adj_Es = []
    for _ in range(min(n_trials, L - 1)):
        i = torch.randint(0, L - 1, (1,)).item()
        swapped = chain.clone()
        swapped[0, i], swapped[0, i + 1] = chain[0, i + 1].clone(), chain[0, i].clone()
        with torch.no_grad():
            adj_Es.append(model(swapped).item())
    results["negatives"]["adj_swap"] = adj_Es

    # Full shuffle
    shuf_Es = []
    for _ in range(n_trials):
        perm = torch.randperm(L)
        shuffled = chain[:, perm, :]
        with torch.no_grad():
            shuf_Es.append(model(shuffled).item())
    results["negatives"]["full_shuffle"] = shuf_Es

    # Truncated (remove middle)
    trunc_Es = []
    for _ in range(n_trials):
        drop = torch.randint(1, max(2, L - 1), (1,)).item()
        mask = torch.ones(L, dtype=torch.bool)
        mask[drop] = False
        truncated = chain[:, mask, :]
        with torch.no_grad():
            trunc_Es.append(model(truncated).item())
    results["negatives"]["truncated"] = trunc_Es

    # Random replacement
    replace_Es = []
    for _ in range(n_trials):
        idx = torch.randint(0, L, (1,)).item()
        replaced = chain.clone()
        replaced[0, idx] = F.normalize(torch.randn(1024, device=device), dim=-1) * target_norm
        with torch.no_grad():
            replace_Es.append(model(replaced).item())
    results["negatives"]["random_replace"] = replace_Es

    return results


def create_pos_neg_violin(comparison: dict) -> go.Figure:
    """Violin plot of positive energy vs negative type energies."""
    fig = go.Figure()

    # Positive (single line)
    fig.add_hline(
        y=comparison["positive"],
        line_dash="dash", line_color="green", line_width=2,
        annotation_text=f"Positive: {comparison['positive']:.4f}",
        annotation_position="top right",
    )

    colors = {
        "adj_swap": "#e74c3c",
        "full_shuffle": "#3498db",
        "truncated": "#f39c12",
        "random_replace": "#9b59b6",
    }

    for neg_type, energies in comparison["negatives"].items():
        fig.add_trace(go.Violin(
            y=energies,
            name=neg_type,
            box_visible=True,
            meanline_visible=True,
            marker_color=colors.get(neg_type, "#95a5a6"),
            hovertemplate=f"{neg_type}: E=%{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        title="Positive vs Negative Energy Distributions (higher = model detects corruption)",
        yaxis_title="Energy",
        showlegend=True,
        width=800,
        height=500,
    )
    return fig


# ─── Training Log Analysis ────────────────────────────────────────────

def parse_training_log(log_text: str) -> dict:
    """
    Parse Phase A training log text into structured data.

    Extracts per-epoch train and val metrics.
    """
    import re

    epochs = {"train": [], "val": []}
    current_section = None
    current_epoch = None
    current_metrics = {}

    for line in log_text.strip().split("\n"):
        line = line.strip()

        # Detect epoch/section
        m = re.match(r"Epoch (\d+) (train|val)", line)
        if m:
            if current_section and current_metrics:
                epochs[current_section].append({
                    "epoch": current_epoch, **current_metrics
                })
            current_epoch = int(m.group(1))
            current_section = m.group(2)
            current_metrics = {}
            continue

        # Parse metric lines
        m = re.match(r"(\w+):\s*([-\d.]+)", line)
        if m and current_section:
            current_metrics[m.group(1)] = float(m.group(2))

    # Final flush
    if current_section and current_metrics:
        epochs[current_section].append({
            "epoch": current_epoch, **current_metrics
        })

    return epochs


def create_overfitting_plot(parsed_log: dict) -> go.Figure:
    """
    Visualize train vs val metrics to diagnose overfitting.

    Key signals:
    - Train rank_acc rising while val falls = overfitting
    - Train-val gap for adj_swap = model memorizing patterns
    """
    train = parsed_log.get("train", [])
    val = parsed_log.get("val", [])

    if not train or not val:
        fig = go.Figure()
        fig.update_layout(title="No training data found in log")
        return fig

    metrics_to_plot = [
        ("chain_rank_acc", "Rank Accuracy"),
        ("chain_acc_adj_swap", "Adj-Swap Accuracy"),
        ("chain_acc_wrong_conclusion", "Wrong Conclusion Accuracy"),
        ("chain_energy_gap", "Energy Gap"),
    ]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[m[1] for m in metrics_to_plot],
        horizontal_spacing=0.12,
        vertical_spacing=0.12,
    )

    for idx, (key, label) in enumerate(metrics_to_plot):
        r = idx // 2 + 1
        c = idx % 2 + 1

        train_vals = [e.get(key, 0) for e in train]
        val_vals = [e.get(key, 0) for e in val]
        train_epochs = [e["epoch"] for e in train]
        val_epochs = [e["epoch"] for e in val]

        fig.add_trace(
            go.Scatter(
                x=train_epochs, y=train_vals,
                mode="lines+markers", name=f"Train {label}",
                line=dict(color="#3498db"),
                showlegend=(idx == 0),
                legendgroup="train",
                hovertemplate=f"Epoch %{{x}}: {label}=%{{y:.4f}}<extra>Train</extra>",
            ),
            row=r, col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=val_epochs, y=val_vals,
                mode="lines+markers", name=f"Val {label}",
                line=dict(color="#e74c3c"),
                showlegend=(idx == 0),
                legendgroup="val",
                hovertemplate=f"Epoch %{{x}}: {label}=%{{y:.4f}}<extra>Val</extra>",
            ),
            row=r, col=c,
        )

        # Add 0.5 line for adj_swap (random baseline)
        if "adj_swap" in key:
            fig.add_hline(y=0.5, line_dash="dot", line_color="gray", row=r, col=c)

    fig.update_layout(
        title="Train vs Val Metrics (divergence = overfitting)",
        width=1000,
        height=700,
    )
    return fig


def create_energy_scale_plot(parsed_log: dict) -> go.Figure:
    """Track energy magnitude over training to detect explosion."""
    train = parsed_log.get("train", [])
    if not train:
        fig = go.Figure()
        fig.update_layout(title="No training data")
        return fig

    epochs = [e["epoch"] for e in train]
    e_pos = [e.get("chain_E_pos_mean", 0) for e in train]
    e_neg = [e.get("chain_E_neg_mean", 0) for e in train]
    gap = [e.get("chain_energy_gap", 0) for e in train]
    grad_p = [e.get("chain_grad_penalty", 0) for e in train]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Energy Magnitude", "Gap & Gradient Penalty"],
    )

    fig.add_trace(go.Scatter(x=epochs, y=e_pos, name="E_pos", line=dict(color="#2ecc71")), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=e_neg, name="E_neg", line=dict(color="#e74c3c")), row=1, col=1)
    fig.add_hline(y=1.0, line_dash="dash", line_color="orange", annotation_text="norm_margin=1.0", row=1, col=1)

    fig.add_trace(go.Scatter(x=epochs, y=gap, name="Energy Gap", line=dict(color="#3498db")), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=grad_p, name="Grad Penalty", line=dict(color="#9b59b6")), row=1, col=2)

    fig.update_layout(title="Energy Scale Monitoring", width=1000, height=400)
    return fig


# ─── Model Loading Helper ─────────────────────────────────────────────

def load_chain_head_from_checkpoint(
    ckpt_path: str,
    device: str = "cpu",
) -> tuple[EBTChainHead, dict]:
    """Load Chain Head model and return (model, checkpoint_info)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    if "chain_head" in cfg_dict:
        cfg_dict = cfg_dict["chain_head"]

    cfg = ChainHeadConfig(
        d_model=cfg_dict.get("d_model", 1024),
        n_heads=cfg_dict.get("n_heads", 8),
        n_layers=cfg_dict.get("n_layers", 2),
        dim_feedforward=cfg_dict.get("dim_feedforward", 2048),
        max_chain_len=cfg_dict.get("max_chain_len", 200),
        dropout=0.0,
        energy_hidden=cfg_dict.get("energy_hidden", 512),
    )
    model = EBTChainHead(cfg)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif "chain_head" in ckpt:
        model.load_state_dict(ckpt["chain_head"])
    else:
        raise KeyError(f"No chain head state found. Keys: {list(ckpt.keys())}")
    model = model.to(device)
    model.eval()

    info = {
        "params": model.num_params,
        "epoch": ckpt.get("epoch", "?"),
        "best_rank_acc": ckpt.get("best_rank_acc", "?"),
        "val_metrics": ckpt.get("val_metrics", {}),
    }
    return model, info


def make_sample_chain(
    data_path: str,
    chain_len: int = 10,
    seq_idx: int = 0,
    device: str = "cpu",
) -> Tensor:
    """Load a sample chain from SONAR data for visualization."""
    sequences, _, _, _ = load_sonar_sequences(
        data_path,
        max_seq_len=64,
        min_seq_len=2,
    )
    if not sequences:
        raise ValueError(f"No valid sequences found in dataset: {data_path}")

    seq = sequences[seq_idx % len(sequences)]
    chain_len = min(chain_len, seq.shape[0])
    chain = seq[:chain_len].unsqueeze(0).to(device)  # [1, L, D]
    return chain
