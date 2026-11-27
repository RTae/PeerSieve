from __future__ import annotations

import json
import math
from pathlib import Path

from experiments.phase1.summarize import (
    _format_mean_std,
    format_markdown_report,
    method_label,
    summarize_directory,
    summarize_system_directory,
)
from src.simulator import ExperimentConfig, run_experiment


def test_method_label_and_mean_std_formatting() -> None:
    label = method_label(
        {
            "aggregator": "trustgraph",
            "val_grad_sign": "descent",
            "trustgraph_use_relu": False,
            "trustgraph_metric": "euclidean",
            "trustgraph_clip_norm": True,
            "trustgraph_clip_mode": "model",
            "trustgraph_delta_mode": "receiver",
            "trustgraph_score_threshold": 0.02,
            "alpha": 0.3,
            "val_split": "iid",
            "grad_clip_norm": 10.0,
            "byz_local_epochs": 2,
            "samples_per_node": 2500,
        }
    )
    assert label == "trustgraph(descent,norelu,euclidean,mclip,rxdelta,t=0.02,a=0.3,val=iid)[gc=10,bep=2,spn=2500]"
    assert _format_mean_std(91.234, 0.5, seeds=1) == "91.23"
    assert _format_mean_std(91.234, 0.5, seeds=3) == "91.23 ± 0.50"


def test_summarize_directory_deduplicates_and_counts_nans(tmp_path: Path) -> None:
    def _write_summary(rel: str, seed: int, acc: float, cons: float, fpr: float, nan_flag: bool) -> None:
        p = tmp_path / rel / "summary.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "topology": "random",
                "attack": "model_poison",
                "byzantine_frac": 0.3,
                "aggregator": "trustgraph",
                "val_grad_sign": "descent",
                "seed": seed,
            },
            "final_benign_test_acc_mean": acc,
            "final_consensus_test_acc": cons,
            "fpr_cumulative": fpr,
            "byz_accept_cumulative": 0.1,
            "mean_benign_score": float("nan") if nan_flag else 0.05,
            "mean_byz_score": float("nan") if nan_flag else -0.05,
            "had_nan": nan_flag,
        }
        p.write_text(json.dumps(payload), encoding="utf-8")

    _write_summary("a/seed_42", 42, 0.40, 0.50, 0.45, False)
    _write_summary("b/seed_42", 42, 0.99, 0.99, 0.01, False)
    _write_summary("a/seed_43", 43, 0.10, 0.10, 0.95, True)

    rows = summarize_directory(tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert r.seeds == 2
    assert math.isclose(r.acc_mean, 25.0, abs_tol=1e-6)
    assert math.isclose(r.cons_mean, 30.0, abs_tol=1e-6)
    assert r.nan_count == 1
    md = format_markdown_report(rows, tmp_path, "Test Summary")
    assert "| 1/2 |" in md


def test_summarize_system_directory_separates_models(tmp_path: Path) -> None:
    for model_name in ("cnn2x2", "cnn2x2_gn"):
        run_experiment(
            ExperimentConfig(
                dataset="synthetic",
                n_nodes=6,
                samples_per_node=30,
                rounds=2,
                model_name=model_name,
                topology="ring",
                byzantine_frac=0.0,
                attack="none",
                device="cpu",
                eval_every=2,
                seed=7,
                output_dir=str(tmp_path / model_name / "seed_7"),
                verbose=False,
            )
        )
    sys_rows = summarize_system_directory(tmp_path)
    assert len(sys_rows) == 2
    assert {r.model for r in sys_rows} == {"cnn2x2", "cnn2x2_gn"}
