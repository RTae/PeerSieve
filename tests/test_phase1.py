"""Unit and offline integration tests for Phase 1 TrustGraph-DFL reproduction."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.phase1.run import build_suite_runs
from experiments.phase1.summarize import format_markdown_report, summarize_directory
from src.aggregators import (
    AggregatorConfig,
    aggregate_trustgraph,
)
from src.attacks import AttackConfig, apply_round_attack
from src.data import apply_label_flip, make_synthetic_cifar_tensors, partition_dirichlet_balanced
from src.models import (
    build_model,
    compute_validation_gradient,
    model_to_vector,
    train_local_epoch,
    vector_to_model,
)
from src.simulator import ExperimentConfig, run_experiment
from src.topology import build_topology, select_byzantine_nodes, topology_summary


def test_balanced_dirichlet_partition_exact_counts_and_non_iid() -> None:
    x_tr, y_tr, x_te, y_te = make_synthetic_cifar_tensors(n_train=10000, n_test=200, seed=1)
    part = partition_dirichlet_balanced(
        x_tr, y_tr, x_te, y_te, n_nodes=20, samples_per_node=500, alpha_dir=0.5, val_frac=0.1, seed=42
    )
    assert len(part.nodes) == 20
    entropies = []
    for i, nd in enumerate(part.nodes):
        assert nd.x_train.shape == (450, 3, 32, 32)
        assert nd.x_val.shape == (50, 3, 32, 32)
        assert sum(part.class_counts[i]) == 500
        probs = np.array(part.class_counts[i], dtype=np.float64) / 500.0
        probs = probs[probs > 0]
        entropies.append(-float(np.sum(probs * np.log(probs))))

    # Uniform 10-class entropy is ln(10) ~= 2.3026; Dir(0.5) must be substantially more skewed.
    assert np.mean(entropies) < 1.95
    total_per_class = np.sum(np.array(part.class_counts), axis=0)
    assert np.all(total_per_class == 1000)


def test_iid_val_split_is_balanced_per_node() -> None:
    x_tr, y_tr, x_te, y_te = make_synthetic_cifar_tensors(n_train=10000, n_test=200, seed=2)
    part = partition_dirichlet_balanced(
        x_tr,
        y_tr,
        x_te,
        y_te,
        n_nodes=20,
        samples_per_node=500,
        alpha_dir=0.5,
        val_frac=0.1,
        val_split="iid",
        seed=42,
    )
    for nd in part.nodes:
        assert nd.x_train.shape == (450, 3, 32, 32)
        assert nd.x_val.shape == (50, 3, 32, 32)
        counts = torch.bincount(nd.y_val, minlength=10)
        assert torch.all(counts == 5)


def test_label_flip_pair_0_1() -> None:
    y = torch.tensor([0, 1, 2, 3, 0, 1, 9])
    flipped = apply_label_flip(y, mode="pair_0_1")
    assert torch.equal(flipped, torch.tensor([1, 0, 2, 3, 1, 0, 9]))


def test_topologies_connected_and_degree() -> None:
    for name in ("ring", "random", "small_world"):
        g = build_topology(name, n_nodes=20, seed=7)
        byz = select_byzantine_nodes(20, 0.3, seed=7)
        info = topology_summary(g, byz)
        assert info["graph_connected"] is True
        assert info["n_byzantine"] == 6
        if name == "ring":
            assert info["degree_min"] == 2 and info["degree_max"] == 2


def test_trustgraph_sign_direction_and_self_fallback() -> None:
    val_grad = torch.tensor([1.0, 2.0, -1.0])
    self_post = torch.tensor([10.0, 10.0, 10.0])
    honest_update = -0.1 * val_grad
    mal_update = +1.0 * val_grad  # sign-flipped poisoning
    nbr_posts = {1: self_post + honest_update, 2: self_post + mal_update}
    nbr_updates = {1: honest_update, 2: mal_update}

    # Corrected descent direction (-val_grad): honest gets weight 1, malicious gets 0.
    cfg_neg = AggregatorConfig(name="trustgraph", val_grad_sign="descent_neg", alpha=0.5)
    res_neg = aggregate_trustgraph(self_post, val_grad, [1, 2], nbr_posts, nbr_updates, cfg_neg)
    assert res_neg.scores[1] > 0.99 and res_neg.scores[2] < -0.99
    assert res_neg.neighbor_weights[1] > 0.99 and res_neg.neighbor_weights[2] == 0.0
    assert res_neg.neighbor_zero_weight_frac == {1: 0.0, 2: 1.0}

    # Literal paper sign (+val_grad): inverted! Malicious gets weight 1, honest gets 0.
    cfg_pos = AggregatorConfig(name="trustgraph", val_grad_sign="paper_pos", alpha=0.5)
    res_pos = aggregate_trustgraph(self_post, val_grad, [1, 2], nbr_posts, nbr_updates, cfg_pos)
    assert res_pos.scores[1] < -0.99 and res_pos.scores[2] > 0.99
    assert res_pos.neighbor_weights[1] == 0.0 and res_pos.neighbor_weights[2] > 0.99

    # All-negative fallback: node keeps self_post unchanged (not 0.5 * self_post).
    res_fb = aggregate_trustgraph(
        self_post, val_grad, [2], {2: nbr_posts[2]}, {2: nbr_updates[2]}, cfg_neg
    )
    assert res_fb.self_fallback is True
    assert torch.allclose(res_fb.new_model, self_post)


def test_model_poison_and_noise_attacks() -> None:
    prev = [torch.ones(4) for _ in range(4)]
    updates = [torch.tensor([0.1, -0.1, 0.2, 0.0]), torch.tensor([0.3, -0.3, 0.0, 0.2]), torch.zeros(4), torch.zeros(4)]
    posts = [prev[i] + updates[i] for i in range(4)]
    gen = torch.Generator().manual_seed(0)

    apply_round_attack(
        AttackConfig(name="model_poison", poison_lambda=10.0),
        byzantine_ids={2, 3},
        benign_ids=[0, 1],
        prev_models=prev,
        post_models=posts,
        updates=updates,
        generator=gen,
    )
    avg_ben = 0.5 * (updates[0] + updates[1])
    assert torch.allclose(updates[2], -10.0 * avg_ben)
    assert torch.allclose(posts[2], prev[0] + updates[2])


def test_validation_gradient_matches_exact_mean_across_batches() -> None:
    torch.manual_seed(0)
    model = build_model("cnn2x2")
    x = torch.randn(10, 3, 32, 32)
    y = torch.randint(0, 10, (10,))
    g_full = compute_validation_gradient(model, x, y, batch_size=64)
    g_batched = compute_validation_gradient(model, x, y, batch_size=4)
    assert torch.allclose(g_full, g_batched, atol=1e-5)
    assert model_to_vector(model).numel() > 100_000


def test_vector_to_model_does_not_alias_source_vector() -> None:
    torch.manual_seed(0)
    model = build_model("cnn2x2")
    v0 = model_to_vector(model)
    v0_copy = v0.clone()
    vector_to_model(v0, model)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    train_local_epoch(model, x, y)
    v1 = model_to_vector(model)
    assert torch.equal(v0, v0_copy)
    assert float((v1 - v0).norm().item()) > 0.0


def test_end_to_end_synthetic_model_poison_shows_sign_effect() -> None:
    """Fast synthetic DFL smoke test confirming descent_neg defends while paper_pos degrades."""
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
    res_neg = run_experiment(ExperimentConfig(**{**base.__dict__, "val_grad_sign": "descent_neg"}))
    res_pos = run_experiment(ExperimentConfig(**{**base.__dict__, "val_grad_sign": "paper_pos"}))

    s_neg = res_neg["summary"]
    s_pos = res_pos["summary"]
    assert math.isfinite(float(s_neg["final_benign_test_acc_mean"]))
    assert s_neg["had_nan"] is False
    assert float(s_neg["mean_benign_score"]) > 0.0
    assert float(s_neg["mean_byz_score"]) < 0.0
    assert float(s_neg["byz_accept_cumulative"]) == 0.0

    # With literal paper sign (+nabla F), cosine signs flip and Byzantine peers get accepted.
    assert float(s_pos["mean_benign_score"]) < 0.0
    assert float(s_pos["mean_byz_score"]) > 0.0
    assert float(s_pos["byz_accept_cumulative"]) > 0.8
    assert float(s_neg["final_consensus_test_acc"]) > float(s_pos["final_consensus_test_acc"]) + 0.15


def test_paper_cnn_10k_suite_count_and_sharding() -> None:
    runs_cnn = build_suite_runs("paper_cnn_10k", ExperimentConfig())
    assert len(runs_cnn) == 31
    names = [name for name, _ in runs_cnn]
    assert len(names) == len(set(names))
    sharded = [[name for idx, (name, _) in enumerate(runs_cnn) if idx % 4 == s] for s in range(4)]
    assert sum(len(x) for x in sharded) == 31
    assert set().union(*sharded) == set(names)


def test_trustgraph_clip_norm_caps_large_updates_and_preserves_small() -> None:
    self_prev = torch.tensor([1.0, 2.0, 3.0])
    self_update = torch.tensor([0.3, 0.4, 0.0])  # norm r_i = 0.5
    self_post = self_prev + self_update
    val_grad = -self_update

    u_huge = 100.0 * self_update
    u_small = 0.5 * self_update
    cfg = AggregatorConfig(name="trustgraph", val_grad_sign="descent_neg", trustgraph_clip_norm=True, alpha=0.5)

    res_huge = aggregate_trustgraph(
        self_post,
        val_grad,
        [1],
        {1: self_prev + u_huge},
        {1: u_huge},
        cfg,
        self_prev=self_prev,
        self_update=self_update,
    )
    delta_huge = res_huge.new_model - self_prev
    assert float(delta_huge.norm().item()) <= float(self_update.norm().item()) + 1e-5
    assert torch.allclose(delta_huge, self_update, atol=1e-5)

    res_small = aggregate_trustgraph(
        self_post,
        val_grad,
        [2],
        {2: self_prev + u_small},
        {2: u_small},
        cfg,
        self_prev=self_prev,
        self_update=self_update,
    )
    assert torch.allclose(res_small.new_model - self_prev, 0.75 * self_update, atol=1e-5)


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
                "val_grad_sign": "descent_neg",
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

    # Duplicate seed 42 in two subfolders + second seed 43 with NaN.
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


def test_model_space_clip_and_stabilized_models() -> None:
    from src.models import augment_cifar_batch

    # Model-space clip caps large displacement (theta_j - theta_i^t) to clip_factor * r_i.
    self_prev = torch.tensor([10.0, 0.0, 0.0])
    self_update = torch.tensor([0.3, 0.4, 0.0])  # r_i = 0.5
    self_post = self_prev + self_update
    val_grad = -self_update
    nbr_far = self_prev + torch.tensor([100.0, 0.0, 0.0])
    nbr_upd = torch.tensor([0.3, 0.4, 0.0])  # positively aligned direction

    cfg = AggregatorConfig(
        name="trustgraph",
        val_grad_sign="descent_neg",
        trustgraph_clip_norm=True,
        trustgraph_clip_mode="model",
        trustgraph_clip_factor=2.0,
        trustgraph_score_threshold=0.02,
        alpha=0.5,
    )
    res = aggregate_trustgraph(
        self_post,
        val_grad,
        [1],
        {1: nbr_far},
        {1: nbr_upd},
        cfg,
        self_prev=self_prev,
        self_update=self_update,
    )
    # Clipped neighbor is self_prev + [1.0, 0, 0] (norm 2 * 0.5 = 1.0).
    expected = 0.5 * self_post + 0.5 * (self_prev + torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(res.new_model, expected, atol=1e-5)

    # GroupNorm CNN & ResNet-18 (offline scratch init) preserve frozen BN stats in train().
    gn_model = build_model("cnn2x2_gn")
    xb = augment_cifar_batch(torch.randn(4, 3, 32, 32), generator=torch.Generator().manual_seed(0))
    assert xb.shape == (4, 3, 32, 32)
    assert gn_model(xb).shape == (4, 10)

    rn = build_model("resnet18_scratch")
    rn.train()
    for mod in rn.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            assert mod.training is False
    assert rn(xb).shape == (4, 10)


def test_phase1_suite_counts_and_sharding() -> None:
    from dataclasses import asdict
    from experiments.phase1.summarize import method_label

    runs_cnn = build_suite_runs("paper_cnn_10k", ExperimentConfig())
    assert len(runs_cnn) == 31
    assert len({name for name, _ in runs_cnn}) == 31

    runs_smoke = build_suite_runs("smoke", ExperimentConfig())
    assert len(runs_smoke) == 4
    assert all(cfg.aggregator == "trustgraph" for _, cfg in runs_smoke)

    pm50 = build_suite_runs("paper_resnet18_50k", ExperimentConfig())
    assert len(pm50) == 14
    assert len({name for name, _ in pm50}) == 14
    assert all(cfg.aggregator == "trustgraph" and cfg.samples_per_node == 2500 and cfg.rounds == 25 and cfg.grad_clip_norm == 10.0 for _, cfg in pm50)

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


def test_trustgraph_receiver_anchored_delta_mode() -> None:
    self_prev = torch.tensor([0.0, 0.0, 0.0])
    self_update = torch.tensor([0.2, 0.0, 0.0])
    self_post = self_prev + self_update
    # Receiver wants to move along +x (val_grad = [-1, 0, 0] -> descent_neg = [+1, 0, 0]).
    val_grad = torch.tensor([-1.0, 0.0, 0.0])
    # Neighbor j ALREADY moved to x=+1.0 in prior rounds, and took a noisy tiny step [-0.05, 0.1, 0] this round.
    nbr_post = torch.tensor([0.95, 0.1, 0.0])
    nbr_sender_update = torch.tensor([-0.05, 0.1, 0.0])

    cfg_sender = AggregatorConfig(name="trustgraph", val_grad_sign="descent_neg", trustgraph_delta_mode="sender")
    res_s = aggregate_trustgraph(
        self_post, val_grad, [1], {1: nbr_post}, {1: nbr_sender_update}, cfg_sender, self_prev=self_prev, self_update=self_update
    )
    assert res_s.scores[1] < 0.0 and res_s.neighbor_weights[1] == 0.0

    cfg_rx = AggregatorConfig(name="trustgraph", val_grad_sign="descent_neg", trustgraph_delta_mode="receiver")
    res_r = aggregate_trustgraph(
        self_post, val_grad, [1], {1: nbr_post}, {1: nbr_sender_update}, cfg_rx, self_prev=self_prev, self_update=self_update
    )
    assert res_r.scores[1] > 0.99 and res_r.neighbor_weights[1] > 0.99


def test_validation_gradient_weight_decay_matches_l2_term() -> None:
    torch.manual_seed(0)
    model = build_model("cnn2x2")
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    g0 = compute_validation_gradient(model, x, y, batch_size=8, weight_decay=0.0)
    g_wd = compute_validation_gradient(model, x, y, batch_size=8, weight_decay=1e-2)
    theta = model_to_vector(model)
    assert torch.allclose(g_wd - g0, 1e-2 * theta, atol=1e-6)


def test_local_grad_clip_bounds_step_magnitude() -> None:
    from src.models import train_local_epoch

    torch.manual_seed(0)
    m_unclipped = build_model("cnn2x2_gn")
    m_clipped = build_model("cnn2x2_gn")
    m_clipped.load_state_dict(m_unclipped.state_dict())
    before = model_to_vector(m_unclipped)

    # Large input magnitudes produce huge unclipped gradients.
    x = torch.randn(16, 3, 32, 32) * 50.0
    y = torch.randint(0, 10, (16,))
    train_local_epoch(m_unclipped, x, y, lr=0.05, momentum=0.0, weight_decay=0.0, batch_size=16, epochs=1, grad_clip_norm=0.0)
    train_local_epoch(m_clipped, x, y, lr=0.05, momentum=0.0, weight_decay=0.0, batch_size=16, epochs=1, grad_clip_norm=1.0)

    upd_unclipped = float(torch.linalg.vector_norm(model_to_vector(m_unclipped) - before).item())
    upd_clipped = float(torch.linalg.vector_norm(model_to_vector(m_clipped) - before).item())
    assert upd_clipped <= 0.05 * 1.0 + 1e-4
    assert upd_unclipped > upd_clipped * 5.0


def test_simulator_logs_system_cost_metrics() -> None:
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
            val_grad_sign="descent_neg",
            trustgraph_clip_norm=True,
            device="cpu",
            eval_every=4,
            seed=42,
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
