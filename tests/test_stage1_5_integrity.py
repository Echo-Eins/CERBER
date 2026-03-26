from __future__ import annotations

import importlib.util
import math
import tempfile
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from configs.base import Stage1_5Config
from cebcm.training.kill_criteria import (
    ConditionalThresholds,
    summarize_conditional_eval,
)


def _load_stage15_train_module():
    root = Path(__file__).resolve().parent.parent
    module_path = root / "experiments" / "01_denoising_poc" / "train_stage1_5.py"
    spec = importlib.util.spec_from_file_location("train_stage1_5_mod", module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_STAGE15 = _load_stage15_train_module()
retrieve_pos_hard = _STAGE15.retrieve_pos_hard
_not_evaluated_kill_stub = _STAGE15._not_evaluated_kill_stub
resolve_ortho_n_iters = _STAGE15.resolve_ortho_n_iters
_validate_stage15_config = _STAGE15._validate_stage15_config


def test_retrieve_pos_hard_strict_index_exclusion() -> None:
    # Query vectors are exactly present in the bank at indices 0 and 1.
    q = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    bank = torch.tensor(
        [
            [1.0, 0.0, 0.0],   # self of q[0]
            [0.0, 1.0, 0.0],   # self of q[1]
            [0.7, 0.7, 0.0],
            [0.1, 0.0, 0.9],
        ],
        dtype=torch.float32,
    )
    bank_n = F.normalize(bank, dim=-1)
    q_indices = torch.tensor([0, 1], dtype=torch.long)
    bank_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    pos, hard = retrieve_pos_hard(
        q=q,
        bank=bank,
        bank_n=bank_n,
        topk_pos=4,
        hard_start=1,
        hard_end=4,
        self_sim_exclude=0.9999,
        q_indices=q_indices,
        bank_indices=bank_indices,
        strict_index_exclusion=True,
    )

    # Returned positives/hards must not be identical to self vectors.
    assert not torch.allclose(pos[0], q[0])
    assert not torch.allclose(pos[1], q[1])
    assert not torch.allclose(hard[0], q[0])
    assert not torch.allclose(hard[1], q[1])


def test_conditional_kill_criteria_regression() -> None:
    thresholds = ConditionalThresholds(
        min_cos_improvement=0.05,
        min_cos_success_rate=0.60,
        min_geodesic_improvement=0.01,
        min_l2_improvement=0.0,
        min_energy_success_rate=0.55,
        max_clean_min_violation=0.10,
        min_step_norm=1e-6,
    )
    eval_metrics = {
        "noise_0.05": {
            "improvement": 0.08,
            "success_rate": 0.75,
            "geodesic_improvement": 0.03,
            "l2_improvement": 0.02,
            "energy_success_rate": 0.70,
            "clean_min_violation_rate": 0.02,
            "step_norm_mean": 0.10,
        },
        "noise_0.1": {
            "improvement": 0.06,
            "success_rate": 0.70,
            "geodesic_improvement": 0.02,
            "l2_improvement": 0.01,
            "energy_success_rate": 0.68,
            "clean_min_violation_rate": 0.04,
            "step_norm_mean": 0.08,
        },
    }
    report = summarize_conditional_eval(eval_metrics, thresholds)
    assert report["passed"] is True
    assert math.isfinite(report["score"])
    assert report["aggregate"]["noise_pass_rate"] == 1.0


def test_conditional_kill_criteria_empty_schema_stable() -> None:
    report = summarize_conditional_eval({}, ConditionalThresholds())
    assert report["status"] == "invalid_eval"
    assert report["passed"] is False
    assert math.isinf(report["score"]) and report["score"] < 0
    assert isinstance(report["global_gates"], dict)
    assert isinstance(report["aggregate"], dict)
    assert isinstance(report["per_noise"], dict)


def test_conditional_kill_criteria_cos_success_alias() -> None:
    thresholds = ConditionalThresholds(
        min_cos_improvement=0.0,
        min_cos_success_rate=0.7,
        min_geodesic_improvement=0.0,
        min_l2_improvement=0.0,
        min_energy_success_rate=0.0,
        max_clean_min_violation=1.0,
        min_step_norm=0.0,
    )
    eval_metrics = {
        "noise_0.05": {
            "improvement": 0.1,
            "cos_success_rate": 0.8,
            "geodesic_improvement": 0.1,
            "l2_improvement": 0.1,
            "energy_success_rate": 0.8,
            "clean_min_violation_rate": 0.0,
            "step_norm_mean": 0.1,
        }
    }
    report = summarize_conditional_eval(eval_metrics, thresholds)
    assert report["passed"] is True
    assert report["per_noise"]["noise_0.05"]["metrics"]["cos_success_rate"] == 0.8


def test_not_evaluated_kill_stub_schema() -> None:
    stub = _not_evaluated_kill_stub()
    assert stub["status"] == "not_evaluated"
    assert stub["score"] is None
    assert stub["passed"] is None
    assert isinstance(stub["global_gates"], dict)
    assert isinstance(stub["aggregate"], dict)
    assert isinstance(stub["per_noise"], dict)


def test_stage15_config_rejects_negative_lambda_rank() -> None:
    cfg = {
        "num_epochs": 1,
        "batch_size": 2,
        "critic_lr": 1e-4,
        "actor_lr": 1e-4,
        "prior_critic_lr": 1e-4,
        "clip_grad_norm": 1.0,
        "loss_spike_factor": 10.0,
        "sigma_curriculum_end": 0.5,
        "sigma_max": 0.5,
        "nce_temperature": 0.07,
        "actor_step_size": 1.0,
        "eval_every_epochs": 1,
        "lambda_rank": -0.1,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(cfg, f)
        tmp = f.name
    try:
        try:
            _STAGE15.load_config(tmp)
            raised = False
        except ValueError as exc:
            raised = "lambda_rank" in str(exc)
        assert raised is True
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_ortho_schedule_resolution() -> None:
    cfg = Stage1_5Config(
        ortho_n_iters=4,
        ortho_schedule_enabled=True,
        ortho_schedule_iters=[4, 2, 1],
        ortho_schedule_boundaries=[0.34, 0.67],
        num_epochs=50,
    )
    assert resolve_ortho_n_iters(cfg, epoch_idx=0, total_epochs=50) == 4
    assert resolve_ortho_n_iters(cfg, epoch_idx=20, total_epochs=50) == 2
    assert resolve_ortho_n_iters(cfg, epoch_idx=40, total_epochs=50) == 1


def test_stage15_config_rejects_bad_ortho_schedule_shape() -> None:
    cfg = Stage1_5Config(
        ortho_schedule_enabled=True,
        ortho_schedule_iters=[4, 2],
        ortho_schedule_boundaries=[0.3, 0.7],  # invalid: expected len=1
    )
    with torch.no_grad():
        try:
            _validate_stage15_config(cfg)
            raised = False
        except ValueError as exc:
            raised = "ortho_schedule_boundaries" in str(exc)
    assert raised is True
