from __future__ import annotations

import math

import networkx as nx

from src.metrics import MetricsTracker, compute_rejection_streaks, summarize_timings


def test_compute_rejection_streaks_and_summarize_timings() -> None:
    assert compute_rejection_streaks([]) == (0, 0, 0, 0)
    assert compute_rejection_streaks([False, False, False]) == (0, 0, 0, 0)
    assert compute_rejection_streaks([True, False, True, True, True, False, True]) == (5, 3, 4, 2)

    assert summarize_timings([]) == {"median": 0.0, "min": 0.0, "max": 0.0, "repeats": 0}
    stats = summarize_timings([0.3, 0.1, 0.2])
    assert stats == {"median": 0.2, "min": 0.1, "max": 0.3, "repeats": 3}


def test_metrics_tracker_counts_fpr_transport_and_nan_summary() -> None:
    graph = nx.complete_graph(3)
    tracker = MetricsTracker(graph=graph, benign_ids=[0, 1], byzantine_ids={2}, param_count=1000, rounds=2)

    reports_r0 = [
        {
            "rank": 0,
            "self_fallback": False,
            "byz_weight_share": 0.0,
            "edges": [
                {"sender_id": 1, "zero_frac": 0.0, "score": 0.5},
                {"sender_id": 2, "zero_frac": 1.0, "score": -0.6},
            ],
        },
        {
            "rank": 1,
            "self_fallback": True,
            "byz_weight_share": 0.0,
            "edges": [
                {"sender_id": 0, "zero_frac": 1.0, "score": -0.1},
                {"sender_id": 2, "zero_frac": 1.0, "score": -0.5},
            ],
        },
    ]
    stats_r0 = tracker.summarize_benign_round(
        benign_reports=reports_r0,
        agg_sec_per_peer=0.01,
        d2h_sec_per_peer=0.02,
        net_sec_per_peer=0.5,
        h2d_sec_per_peer=0.03,
        wire_bytes_recv_per_peer=8000.0,
    )
    rec0 = tracker.record_round(
        round_idx=0,
        train_sec_per_peer=0.4,
        val_grad_sec_per_peer=0.05,
        agg_stats=stats_r0,
        eval_metrics={"benign_test_acc_mean": 0.8, "benign_test_acc_std": 0.02, "consensus_test_acc": 0.85},
        round_wall_sec=1.0,
    )
    assert math.isclose(float(rec0["fpr_round"]), 0.5)
    assert math.isclose(float(rec0["byz_accept_rate"]), 0.0)
    assert math.isclose(float(rec0["wasted_transfer_rate"]), 0.75)

    # Round 1 injects a NaN benign score to verify finalize_summary promotes had_nan.
    reports_r1 = [
        {
            "rank": 0,
            "self_fallback": False,
            "byz_weight_share": 0.0,
            "edges": [
                {"sender_id": 1, "zero_frac": 0.0, "score": float("nan")},
                {"sender_id": 2, "zero_frac": 1.0, "score": -0.4},
            ],
        },
        {
            "rank": 1,
            "self_fallback": False,
            "byz_weight_share": 0.0,
            "edges": [
                {"sender_id": 0, "zero_frac": 0.0, "score": 0.2},
                {"sender_id": 2, "zero_frac": 1.0, "score": -0.4},
            ],
        },
    ]
    stats_r1 = tracker.summarize_benign_round(
        benign_reports=reports_r1,
        agg_sec_per_peer=0.01,
        d2h_sec_per_peer=0.02,
        net_sec_per_peer=0.5,
        h2d_sec_per_peer=0.03,
        wire_bytes_recv_per_peer=8000.0,
    )
    tracker.record_round(
        round_idx=1,
        train_sec_per_peer=0.4,
        val_grad_sec_per_peer=0.05,
        agg_stats=stats_r1,
        eval_metrics={"benign_test_acc_mean": 0.9, "benign_test_acc_std": 0.01, "consensus_test_acc": 0.92},
        round_wall_sec=1.0,
    )

    summary = tracker.finalize_summary(
        config_dict={"seed": 42},
        resolved_device="cpu",
        topo_info={"n_nodes": 3},
        total_wall_sec=2.0,
        had_nan=False,
    )
    assert summary["had_nan"] is True
    assert math.isclose(float(summary["fpr_cumulative"]), 0.25)
    assert math.isclose(float(summary["byz_accept_cumulative"]), 0.0)
    sm = summary["system_metrics"]
    assert sm["byz_repeat_rejections"]["repeat_transfers_after_first_rej"] == 2
    assert sm["measured_transport_sec"]["net_sec_per_peer"]["median"] == 0.5
