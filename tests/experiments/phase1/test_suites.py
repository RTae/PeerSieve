from __future__ import annotations

from dataclasses import asdict

import pytest

from experiments.phase1.suites import build_suite_runs
from experiments.phase1.summarize import method_label
from src.simulator import ExperimentConfig


def test_paper_cnn_10k_suite_count_and_sharding() -> None:
    runs_cnn = build_suite_runs("paper_cnn_10k", ExperimentConfig())
    assert len(runs_cnn) == 31
    names = [name for name, _ in runs_cnn]
    assert len(names) == len(set(names))
    sharded = [[name for idx, (name, _) in enumerate(runs_cnn) if idx % 4 == s] for s in range(4)]
    assert sum(len(x) for x in sharded) == 31
    assert set().union(*sharded) == set(names)


def test_phase1_suite_counts_and_uniqueness() -> None:
    runs_smoke = build_suite_runs("smoke", ExperimentConfig())
    assert len(runs_smoke) == 4
    assert all(cfg.aggregator == "trustgraph" for _, cfg in runs_smoke)

    pm50 = build_suite_runs("paper_resnet18_50k", ExperimentConfig())
    assert len(pm50) == 14
    assert len({name for name, _ in pm50}) == 14
    assert all(
        cfg.aggregator == "trustgraph" and cfg.samples_per_node == 2500 and cfg.rounds == 25 and cfg.grad_clip_norm == 10.0
        for _, cfg in pm50
    )

    for suite_name, expected_count in (
        ("fig3a_attacks", 4),
        ("table1_topologies", 3),
        ("fig3b_byz_sweep", 5),
        ("sec5_5_ablations", 5),
    ):
        section_runs = build_suite_runs(suite_name, ExperimentConfig())
        assert len(section_runs) == expected_count
        assert len({name for name, _ in section_runs}) == expected_count
        keys = {
            (cfg.topology, cfg.attack, int(round(cfg.byzantine_frac * 100)), method_label(asdict(cfg)))
            for _, cfg in section_runs
        }
        assert len(keys) == expected_count

    for suite_name in ("fig3b_fpr_rxdelta", "fig3b_fpr_rxdelta_tau"):
        suite_runs = build_suite_runs(suite_name, ExperimentConfig())
        assert len(suite_runs) == 8
        assert len({name for name, _ in suite_runs}) == 8
        assert all(cfg.trustgraph_delta_mode == "receiver" for _, cfg in suite_runs)
        keys = {
            (cfg.topology, cfg.attack, int(round(cfg.byzantine_frac * 100)), method_label(asdict(cfg)))
            for _, cfg in suite_runs
        }
        assert len(keys) == 8

    with pytest.raises(ValueError, match="Unknown suite"):
        build_suite_runs("nonexistent_suite", ExperimentConfig())
