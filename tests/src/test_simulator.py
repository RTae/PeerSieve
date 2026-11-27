from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from experiments.phase1.summarize import (
    export_system_metrics_json,
    format_system_markdown_report,
    summarize_system_directory,
)
from src.simulator import (
    ExperimentConfig,
    learning_rate_for_round,
    resolve_device,
    run_experiment,
    should_eval,
)


def test_learning_rate_schedule_and_eval_cadence() -> None:
    cfg_const = ExperimentConfig(lr=0.02, lr_schedule="constant", rounds=10)
    assert learning_rate_for_round(cfg_const, 0) == 0.02
    assert learning_rate_for_round(cfg_const, 9) == 0.02

    cfg_cos = ExperimentConfig(lr=0.02, lr_schedule="cosine", rounds=10)
    assert math.isclose(learning_rate_for_round(cfg_cos, 0), 0.02)
    assert math.isclose(learning_rate_for_round(cfg_cos, 9), 0.001)
    assert learning_rate_for_round(cfg_cos, 4) < learning_rate_for_round(cfg_cos, 1)

    with pytest.raises(ValueError, match="Unknown lr_schedule"):
        learning_rate_for_round(ExperimentConfig(lr_schedule="linear"), 0)

    assert should_eval(0, 10, 0) is False
    assert should_eval(9, 10, 0) is True
    assert should_eval(0, 10, 5) is True
    assert should_eval(4, 10, 5) is True
    assert should_eval(3, 10, 5) is False
    assert should_eval(9, 10, 5) is True
    assert resolve_device("cpu") == torch.device("cpu")


def test_end_to_end_synthetic_model_poison_shows_sign_effect() -> None:
    """Synthetic DFL smoke test checking that descent defends and ascent degrades."""
    base = ExperimentConfig(
        dataset="synthetic",
        n_nodes=10,
        samples_per_node=200,
        alpha_dir=5.0,
        rounds=6,
        topology="random",
        er_p=0.5,
        byzantine_frac=0.3,
        attack="model_poison",
        aggregator="trustgraph",
        device="cpu",
        eval_every=3,
        seed=11,
        verbose=False,
    )
    res_neg = run_experiment(ExperimentConfig(**{**base.__dict__, "val_grad_sign": "descent"}))
    res_pos = run_experiment(ExperimentConfig(**{**base.__dict__, "val_grad_sign": "ascent"}))

    s_neg = res_neg["summary"]
    s_pos = res_pos["summary"]
    assert math.isfinite(float(s_neg["final_benign_test_acc_mean"]))
    assert s_neg["had_nan"] is False
    assert float(s_neg["mean_benign_score"]) > 0.0
    assert float(s_neg["mean_byz_score"]) < 0.0
    assert float(s_neg["byz_accept_cumulative"]) == 0.0

    assert float(s_pos["mean_benign_score"]) < 0.0
    assert float(s_pos["mean_byz_score"]) > 0.0
    assert float(s_pos["byz_accept_cumulative"]) > 0.8
    assert float(s_neg["final_consensus_test_acc"]) > float(s_pos["final_consensus_test_acc"]) + 0.15


def test_simulator_logs_system_cost_metrics(tmp_path: Path) -> None:
    out = run_experiment(
        ExperimentConfig(
            dataset="synthetic",
            n_nodes=8,
            samples_per_node=40,
            rounds=4,
            model_name="cnn2x2_gn",
            topology="random",
            er_p=0.5,
            byzantine_frac=0.25,
            attack="model_poison",
            aggregator="trustgraph",
            val_grad_sign="descent",
            trustgraph_clip_norm=True,
            device="cpu",
            eval_every=4,
            seed=42,
            output_dir=str(tmp_path / "run_mp" / "seed_42"),
            verbose=False,
        )
    )
    sm = out["summary"]["system_metrics"]
    assert sm["payload_fp32_mb"] > 0.0
    assert sm["total_transfers"] > 0
    assert sm["total_rejected_transfers"] == sm["honest_rejected_transfers"] + sm["byz_rejected_transfers"]
    assert 0.0 <= float(sm["wasted_transfer_pct"]) <= 100.0
    assert sm["batched_recv_buf_max_mib"] >= sm["batched_recv_buf_mean_mib"] > 0.0
    assert set(sm["round_time_by_link"].keys()) == {"10gbps", "1gbps", "100mbps"}
    assert sm["train_sec_per_peer"]["repeats"] == 4
    assert sm["val_grad_sec_per_peer"]["repeats"] == 4
    assert sm["agg_sec_per_peer"]["repeats"] == 4
    assert "repeat_transfers_after_first_rej" in sm["byz_repeat_rejections"]
    assert "isolated_single_rejections" in sm["honest_rejection_streaks"]
    assert all("wasted_transfer_rate" in rec and "train_sec_per_peer" in rec for rec in out["history"])

    sys_rows = summarize_system_directory(tmp_path)
    assert len(sys_rows) == 1
    md = format_system_markdown_report(sys_rows)
    assert "System Cost Summary" in md and "model_poison" in md
    exported = export_system_metrics_json(tmp_path, tmp_path / "system_metrics.json", job_id="123", commit="abc")
    assert exported["total_runs"] == 1 and len(exported["runs"][0]["curves"]["round_wasted_pct"]) == 4
