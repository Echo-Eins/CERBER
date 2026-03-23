"""
Stage 1: Denoising PoC — Train SimpleEnergy on noise reconstruction task.

Pipeline:
    SONAR encoder (frozen) → [SimpleEnergy] → Langevin → SONAR decoder (frozen)
                               ^^^TRAINS^^^

Training loop (MDSM — Multi-Scale Denoising Score Matching):
    1. Sample V_clean from pre-encoded WikiText vectors
    2. σ ~ LogUniform(σ_min, σ_max) — continuous noise scale
    3. V_noisy = V_clean + σ·||V_clean||·ε  (relative Gaussian noise)
    4. Loss = 1 - cos(∇_V E(V_clean, V_noisy, σ), V_clean - V_noisy)
       Direction matching: train ∇E to point from noisy toward clean.
       The energy function is σ-conditioned (NCSN-style) via sinusoidal
       embedding of log(σ) — this gives the network extra context about
       the noise regime, but cosine loss is used because 1-Lipschitz
       hidden layers cap ||∇E|| far below the raw target score magnitude.

    This trains the energy gradient direction at ALL noise scales.
    Langevin dynamics then follows these gradients for refinement.

Architecture:
    - OrthoLinear layers (Bjorck orthonormalization, all σ_i ≈ 1)
    - GroupSort activation (1-Lipschitz, no gradient attenuation)
    - 4 layers: 4096→2048→1024→512→1

Usage:
    # 1. First encode the dataset (one-time):
    python -m cebcm.data.encode_dataset --num_sentences 10500 --output data/wikitext_sonar_10k.pt

    # 2. Train (MDSM with PID Langevin, orthonorm+GroupSort):
    python experiments/01_denoising_poc/train.py --data data/wikitext_sonar_10k.pt

    # 3. Train with legacy margin contrastive + spectral norm:
    python experiments/01_denoising_poc/train.py --data data/wikitext_sonar_10k.pt --loss margin_contrastive --norm spectral_norm --activation relu

    # 4. Train with wandb:
    python experiments/01_denoising_poc/train.py --data data/wikitext_sonar_10k.pt --wandb

Spec reference: §13.1, IMPLEMENTATION_PLAN §4.1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.base import Stage1Config
from cebcm.models.energy import SimpleEnergy
from cebcm.training.losses import (
    multiscale_dsm_loss,
    margin_contrastive_loss,
    gradient_penalty,
)
from cebcm.inference.langevin import (
    langevin_dynamics,
    pid_langevin_dynamics,
    underdamped_langevin_dynamics,
    run_langevin,
)
from cebcm.data.dataset import SONARVectorDataset


def add_relative_noise(v: torch.Tensor, scale: float) -> torch.Tensor:
    """Add Gaussian noise relative to embedding norm: noise = randn * scale * ||V||."""
    norms = v.norm(dim=-1, keepdim=True)
    noise = torch.randn_like(v) * scale * norms
    return v + noise


def train_epoch_mdsm(
    model: SimpleEnergy,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    config: Stage1Config,
    device: torch.device,
    epoch: int,
    global_step: int,
) -> tuple[dict, int]:
    """Train one epoch with MDSM loss. Returns (metrics, new_global_step)."""
    model.train()
    total_loss = 0.0
    total_dsm = 0.0
    total_gp = 0.0
    num_batches = 0

    for batch_idx, v_clean in enumerate(dataloader):
        v_clean = v_clean.to(device)

        # Warmup: linearly increase LR from 0 to target
        if global_step < config.warmup_steps:
            warmup_factor = global_step / config.warmup_steps
            for pg in optimizer.param_groups:
                pg["lr"] = config.lr * warmup_factor

        # MDSM loss: learn correct score at all noise scales
        loss_dsm = multiscale_dsm_loss(
            energy_fn=model,
            v_clean=v_clean,
            sigma_min=config.mdsm_sigma_min,
            sigma_max=config.mdsm_sigma_max,
            relative_noise=True,
        )

        # Optional gradient penalty (lighter with orthonorm — already Lipschitz)
        loss_gp = torch.tensor(0.0, device=device)
        if config.gradient_penalty_lambda > 0:
            noise_scale = 0.1  # fixed scale for GP sampling
            v_noisy = add_relative_noise(v_clean, noise_scale)
            loss_gp = gradient_penalty(model, v_clean, v_noisy)

        loss = loss_dsm + config.gradient_penalty_lambda * loss_gp

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_dsm += loss_dsm.item()
        total_gp += loss_gp.item()
        num_batches += 1
        global_step += 1

        if config.log_every > 0 and (batch_idx + 1) % config.log_every == 0:
            e_scale = model.log_energy_scale.exp().item()
            print(
                f"  [{batch_idx + 1}/{len(dataloader)}] "
                f"loss={loss.item():.4f} DSM={loss_dsm.item():.4f} "
                f"GP={loss_gp.item():.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f} "
                f"E_scale={e_scale:.1f}"
            )

    return {
        "loss": total_loss / max(num_batches, 1),
        "dsm_loss": total_dsm / max(num_batches, 1),
        "gradient_penalty": total_gp / max(num_batches, 1),
    }, global_step


def train_epoch_contrastive(
    model: SimpleEnergy,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: Stage1Config,
    device: torch.device,
    epoch: int,
) -> dict:
    """Train one epoch with legacy margin contrastive loss. Returns metrics."""
    model.train()
    total_loss = 0.0
    total_e_pos = 0.0
    total_e_neg = 0.0
    total_gp = 0.0
    num_batches = 0

    for batch_idx, v_orig in enumerate(dataloader):
        v_orig = v_orig.to(device)

        scale_idx = torch.randint(0, len(config.train_noise_scales), (1,)).item()
        noise_scale = config.train_noise_scales[scale_idx]

        e_pos = model(v_orig, v_orig)
        v_noisy = add_relative_noise(v_orig, noise_scale)
        e_neg = model(v_orig, v_noisy)

        loss_contrastive = margin_contrastive_loss(e_pos, e_neg, margin=config.margin)
        gp = gradient_penalty(model, v_orig, v_noisy)
        loss = loss_contrastive + config.gradient_penalty_lambda * gp

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_e_pos += e_pos.mean().item()
        total_e_neg += e_neg.mean().item()
        total_gp += gp.item()
        num_batches += 1

        if config.log_every > 0 and (batch_idx + 1) % config.log_every == 0:
            print(
                f"  [{batch_idx + 1}/{len(dataloader)}] "
                f"loss={loss.item():.4f} E_pos={e_pos.mean().item():.4f} "
                f"E_neg={e_neg.mean().item():.4f} GP={gp.item():.4f}"
            )

    return {
        "loss": total_loss / max(num_batches, 1),
        "e_pos_mean": total_e_pos / max(num_batches, 1),
        "e_neg_mean": total_e_neg / max(num_batches, 1),
        "gradient_penalty": total_gp / max(num_batches, 1),
        "energy_gap": (total_e_neg - total_e_pos) / max(num_batches, 1),
    }


def evaluate_denoising(
    model: SimpleEnergy,
    dataset: SONARVectorDataset,
    config: Stage1Config,
    device: torch.device,
    num_samples: int = 100,
) -> dict:
    """
    Evaluate denoising quality using the configured Langevin method.
    Run Langevin dynamics on noisy vectors, measure if they get closer to originals.
    """
    model.eval()
    results = {}

    indices = torch.randperm(len(dataset))[:num_samples]

    # Method-specific kwargs
    method = config.langevin.method
    method_kwargs = {}
    if method == "pid":
        method_kwargs = dict(
            kp=config.langevin.pid_kp,
            ki=config.langevin.pid_ki,
            kd=config.langevin.pid_kd,
            integral_decay=config.langevin.pid_integral_decay,
        )
    elif method == "underdamped":
        method_kwargs = dict(
            friction=config.langevin.underdamped_friction,
            mass=config.langevin.underdamped_mass,
        )
    elif method == "overdamped":
        method_kwargs = dict(momentum_beta=config.langevin.momentum_beta)

    for noise_scale in config.eval_noise_scales:
        cos_before_list = []
        cos_after_list = []
        energy_before_list = []
        energy_after_list = []

        for idx in indices:
            v_orig = dataset[idx.item()].unsqueeze(0).to(device)
            v_noisy = add_relative_noise(v_orig, noise_scale)

            cos_before = F.cosine_similarity(v_orig, v_noisy, dim=-1).item()

            result = run_langevin(
                method=method,
                energy_fn=model,
                v_query=v_orig,
                v_init=v_noisy,
                lr=config.langevin.lr,
                noise_scale=config.langevin.noise_scale,
                max_steps=config.langevin.max_steps,
                target_norm=config.langevin.target_norm,
                energy_threshold=config.langevin.energy_threshold,
                plateau_patience=config.langevin.plateau_patience,
                plateau_delta=config.langevin.plateau_delta,
                v_target=v_orig,
                **method_kwargs,
            )

            cos_after = F.cosine_similarity(v_orig, result.v_final, dim=-1).item()

            e_before = model(v_orig, v_noisy).item()
            e_after = model(v_orig, result.v_final).item()

            cos_before_list.append(cos_before)
            cos_after_list.append(cos_after)
            energy_before_list.append(e_before)
            energy_after_list.append(e_after)

        cos_before_mean = sum(cos_before_list) / len(cos_before_list)
        cos_after_mean = sum(cos_after_list) / len(cos_after_list)
        improvement = cos_after_mean - cos_before_mean

        results[f"noise_{noise_scale}"] = {
            "cos_before_mean": cos_before_mean,
            "cos_after_mean": cos_after_mean,
            "improvement": improvement,
            "energy_before_mean": sum(energy_before_list) / len(energy_before_list),
            "energy_after_mean": sum(energy_after_list) / len(energy_after_list),
            "success_rate": sum(
                1 for a, b in zip(cos_after_list, cos_before_list) if a > b
            ) / len(cos_after_list),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Train SimpleEnergy for denoising")
    parser.add_argument("--data", type=str, required=True, help="Path to encoded .pt dataset")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    # Architecture flags
    parser.add_argument("--loss", type=str, default=None,
                        choices=["mdsm", "margin_contrastive"],
                        help="Loss type (default: mdsm)")
    parser.add_argument("--norm", type=str, default=None,
                        choices=["orthonorm", "spectral_norm", "none"],
                        help="Normalization (default: orthonorm)")
    parser.add_argument("--activation", type=str, default=None,
                        choices=["groupsort", "lipschitz_spline", "relu"],
                        help="Activation (default: groupsort)")
    parser.add_argument("--langevin_method", type=str, default=None,
                        choices=["overdamped", "pid", "underdamped"],
                        help="Langevin method (default: pid)")
    args = parser.parse_args()

    config = Stage1Config()
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.loss is not None:
        config.loss_type = args.loss
    if args.norm is not None:
        config.norm_mode = args.norm
    if args.activation is not None:
        config.activation = args.activation
    if args.langevin_method is not None:
        config.langevin.method = args.langevin_method
    config.use_wandb = args.wandb

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ============================================================
    # Data
    # ============================================================
    print(f"Loading dataset from {args.data}...")
    full_dataset = SONARVectorDataset(args.data)
    print(f"  Total vectors: {len(full_dataset)}, dim: {full_dataset.embeddings.shape[1]}")

    n_test = min(config.num_test_sentences, len(full_dataset) // 10)
    n_train = len(full_dataset) - n_test
    train_dataset = full_dataset.subset(0, n_train)
    test_dataset = full_dataset.subset(n_train, len(full_dataset))
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )

    # ============================================================
    # Model
    # ============================================================
    model = SimpleEnergy(
        dim=config.energy_dim,
        hidden_dims=config.energy_hidden_dims,
        norm_mode=config.norm_mode,
        activation=config.activation,
        ortho_n_iters=config.ortho_n_iters,
        groupsort_size=config.groupsort_size,
        spline_num_knots=config.spline_num_knots,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"SimpleEnergy: {num_params:,} parameters")
    print(f"  Architecture: norm={config.norm_mode}, activation={config.activation}")
    print(f"  Hidden dims: {config.energy_hidden_dims}")
    print(f"  Loss: {config.loss_type}")
    print(f"  Langevin: {config.langevin.method}")

    # Cosine direction loss is scale-invariant: gradient magnitude doesn't matter,
    # only direction. Single param group with uniform LR suffices.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=config.lr * 0.1
    )

    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        print(f"  Resumed at epoch {start_epoch}")

    # ============================================================
    # Wandb
    # ============================================================
    wandb_run = None
    if config.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=config.wandb_project,
            config={
                "lr": config.lr,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "loss_type": config.loss_type,
                "norm_mode": config.norm_mode,
                "activation": config.activation,
                "energy_hidden_dims": config.energy_hidden_dims,
                "mdsm_sigma_min": config.mdsm_sigma_min,
                "mdsm_sigma_max": config.mdsm_sigma_max,
                "langevin_method": config.langevin.method,
                "langevin_lr": config.langevin.lr,
                "langevin_steps": config.langevin.max_steps,
                "target_norm": config.langevin.target_norm,
                "num_params": num_params,
            },
        )

    # ============================================================
    # Output dirs
    # ============================================================
    output_dir = Path(config.output_dir)
    checkpoint_dir = Path(config.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Training loop
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Training SimpleEnergy for {config.num_epochs} epochs")
    print(f"  Batch size: {config.batch_size}")
    print(f"  LR: {config.lr} (warmup: {config.warmup_steps} steps)")
    print(f"  Loss: {config.loss_type}")
    if config.loss_type == "mdsm":
        print(f"  MDSM σ range: [{config.mdsm_sigma_min}, {config.mdsm_sigma_max}]")
    else:
        print(f"  Margin: {config.margin}")
        print(f"  Train noise scales: {config.train_noise_scales}")
    print(f"  Eval noise scales: {config.eval_noise_scales}")
    print(f"  Langevin: method={config.langevin.method}, lr={config.langevin.lr}, steps={config.langevin.max_steps}")
    print(f"  Target norm: {config.langevin.target_norm}")
    print(f"{'='*60}\n")

    all_metrics: list[dict] = []
    best_improvement = -float("inf")
    start_time = time.time()

    for epoch in range(start_epoch, config.num_epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch + 1}/{config.num_epochs}")

        # Train
        if config.loss_type == "mdsm":
            train_metrics, global_step = train_epoch_mdsm(
                model, train_loader, optimizer, scheduler,
                config, device, epoch, global_step,
            )
        else:
            train_metrics = train_epoch_contrastive(
                model, train_loader, optimizer, config, device, epoch,
            )

        # Step scheduler after warmup completes
        if global_step >= config.warmup_steps:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        metrics_str = " ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
        print(f"  {metrics_str} lr={optimizer.param_groups[0]['lr']:.6f} ({epoch_time:.1f}s)")

        # Evaluate periodically
        eval_metrics = None
        if (epoch + 1) % config.eval_every_epoch == 0 or epoch == config.num_epochs - 1:
            print("  Evaluating denoising...")
            eval_metrics = evaluate_denoising(
                model, test_dataset, config, device, num_samples=100
            )
            for noise_key, metrics in eval_metrics.items():
                print(
                    f"    {noise_key}: "
                    f"cos_before={metrics['cos_before_mean']:.4f} -> "
                    f"cos_after={metrics['cos_after_mean']:.4f} "
                    f"(d={metrics['improvement']:+.4f}, "
                    f"success={metrics['success_rate']:.0%})"
                )

            avg_improvement = sum(
                m["improvement"] for m in eval_metrics.values()
            ) / len(eval_metrics)
            if avg_improvement > best_improvement:
                best_improvement = avg_improvement
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "metrics": eval_metrics,
                        "config": {
                            "energy_dim": config.energy_dim,
                            "energy_hidden_dims": config.energy_hidden_dims,
                            "norm_mode": config.norm_mode,
                            "activation": config.activation,
                            "loss_type": config.loss_type,
                        },
                    },
                    checkpoint_dir / "best.pt",
                )
                print(f"    * New best (avg improvement: {avg_improvement:+.4f})")

        # Log
        epoch_record = {"epoch": epoch + 1, "train": train_metrics}
        if eval_metrics is not None:
            epoch_record["eval"] = eval_metrics
        all_metrics.append(epoch_record)

        if wandb_run is not None:
            log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
            log_dict["train/lr"] = optimizer.param_groups[0]["lr"]
            if eval_metrics is not None:
                for noise_key, metrics in eval_metrics.items():
                    for mk, mv in metrics.items():
                        log_dict[f"eval/{noise_key}/{mk}"] = mv
            wandb_run.log(log_dict, step=epoch + 1)

        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                },
                checkpoint_dir / f"epoch_{epoch + 1:03d}.pt",
            )

    # ============================================================
    # Save final results
    # ============================================================
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")

    print("\nFinal evaluation on test set...")
    model.eval()
    final_eval = evaluate_denoising(model, test_dataset, config, device, num_samples=min(200, len(test_dataset)))
    for noise_key, metrics in final_eval.items():
        print(
            f"  {noise_key}: "
            f"cos_before={metrics['cos_before_mean']:.4f} -> "
            f"cos_after={metrics['cos_after_mean']:.4f} "
            f"(d={metrics['improvement']:+.4f}, "
            f"success={metrics['success_rate']:.0%})"
        )

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": config.num_epochs - 1,
            "global_step": global_step,
            "final_eval": final_eval,
            "config": {
                "energy_dim": config.energy_dim,
                "energy_hidden_dims": config.energy_hidden_dims,
                "norm_mode": config.norm_mode,
                "activation": config.activation,
                "loss_type": config.loss_type,
            },
        },
        checkpoint_dir / "final.pt",
    )

    summary = {
        "total_time_seconds": total_time,
        "num_epochs": config.num_epochs,
        "best_improvement": best_improvement,
        "final_eval": final_eval,
        "config": {
            "lr": config.lr,
            "batch_size": config.batch_size,
            "loss_type": config.loss_type,
            "norm_mode": config.norm_mode,
            "activation": config.activation,
            "energy_hidden_dims": config.energy_hidden_dims,
            "mdsm_sigma_min": config.mdsm_sigma_min,
            "mdsm_sigma_max": config.mdsm_sigma_max,
            "langevin_method": config.langevin.method,
            "langevin_lr": config.langevin.lr,
            "langevin_steps": config.langevin.max_steps,
            "target_norm": config.langevin.target_norm,
        },
        "epoch_metrics": all_metrics,
    }
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved to {output_dir / 'training_metrics.json'}")

    # Kill criterion check
    print(f"\n{'='*60}")
    print("KILL CRITERION CHECK")
    print(f"{'='*60}")
    all_pass = True
    for noise_key, metrics in final_eval.items():
        passed = metrics["improvement"] > 0 and metrics["success_rate"] > 0.5
        status = "PASS" if passed else "FAIL"
        print(f"  {noise_key}: improvement={metrics['improvement']:+.4f}, success={metrics['success_rate']:.0%} -> {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  VERDICT: Stage 1 PASSED — energy function works for denoising")
    else:
        print("\n  VERDICT: Stage 1 FAILED — denoised vectors not closer to originals")
        print("    -> Energy function does not work. Review architecture or training.")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
