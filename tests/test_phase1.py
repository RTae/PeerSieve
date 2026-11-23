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
    aggregate_balance,
    aggregate_fedavg,
    aggregate_krum,
    aggregate_trimmed_mean,
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


def test_balance_krum_trimmed_mean_and_fedavg() -> None:
    self_post = torch.tensor([10.0, 0.0, 0.0, 0.0])
    nbr_posts = {
        1: torch.tensor([10.1, 0.05, 0.0, 0.0]),
        2: torch.tensor([9.95, -0.05, 0.0, 0.0]),
        3: torch.tensor([10.05, 0.02, 0.0, 0.0]),
        4: torch.tensor([-50.0, 50.0, 50.0, 50.0]),
    }

    res_bal = aggregate_balance(
        self_post, [1, 2, 3, 4], nbr_posts, round_idx=0, total_rounds=200, cfg=AggregatorConfig(name="balance")
    )
    assert res_bal.neighbor_weights[4] == 0.0
    assert all(res_bal.neighbor_weights[j] > 0.0 for j in (1, 2, 3))

    res_krum = aggregate_krum(
        0, self_post, [1, 2, 3, 4], nbr_posts, byzantine_frac=0.25, cfg=AggregatorConfig(name="krum")
    )
    assert res_krum.neighbor_weights[4] == 0.0

    # Coord 0 sorted: [-50, 9.95, 10.0, 10.05, 10.1] -> mean(9.95, 10.0, 10.05) = 10.0
    # Coord 1 sorted: [-0.05, 0.0, 0.02, 0.05, 50.0] -> mean(0.0, 0.02, 0.05) = 0.07 / 3
    res_tm = aggregate_trimmed_mean(
        self_post, [1, 2, 3, 4], nbr_posts, cfg=AggregatorConfig(name="trimmed_mean", trim_ratio=0.3)
    )
    assert torch.allclose(res_tm.new_model, torch.tensor([10.0, 0.07 / 3.0, 0.0, 0.0]), atol=1e-4)

    # Ring degree-2 edge case must not crash for Krum or Trimmed Mean.
    ring_posts = {1: nbr_posts[1], 4: nbr_posts[4]}
    res_ring_k = aggregate_krum(0, self_post, [1, 4], ring_posts, byzantine_frac=0.5, cfg=AggregatorConfig(name="krum"))
    res_ring_tm = aggregate_trimmed_mean(self_post, [1, 4], ring_posts, cfg=AggregatorConfig(name="trimmed_mean"))
    assert torch.isfinite(res_ring_k.new_model).all()
    assert torch.isfinite(res_ring_tm.new_model).all()

    res_fa = aggregate_fedavg(self_post, [1, 2], {1: nbr_posts[1], 2: nbr_posts[2]})
    expected = (self_post + nbr_posts[1] + nbr_posts[2]) / 3.0
    assert torch.allclose(res_fa.new_model, expected)


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


def test_full_and_step_a_suite_count_and_sharding() -> None:
    runs_full = build_suite_runs("full", ExperimentConfig())
    runs_alias = build_suite_runs("step_a", ExperimentConfig())
    assert len(runs_full) == 75
    assert [n for n, _ in runs_full] == [n for n, _ in runs_alias]
    names = [name for name, _ in runs_full]
    assert len(names) == len(set(names))
    sharded = [[name for idx, (name, _) in enumerate(runs_full) if idx % 4 == s] for s in range(4)]
    assert sum(len(x) for x in sharded) == 75
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


def test_stabilized_ab_suite_count_and_sharding() -> None:
    runs = build_suite_runs("stabilized_ab", ExperimentConfig(grad_clip_norm=1.0))
    assert len(runs) == 110
    assert all(cfg.grad_clip_norm == 1.0 for _, cfg in runs)
    names = [name for name, _ in runs]
    assert len(names) == len(set(names))
    sharded = [[name for idx, (name, _) in enumerate(runs) if idx % 8 == s] for s in range(8)]
    assert sum(len(x) for x in sharded) == 110
    assert set().union(*sharded) == set(names)

    calib = build_suite_runs("paper_match_calib", ExperimentConfig())
    assert len(calib) == 16
    assert len({name for name, _ in calib}) == 16
    assert all(cfg.grad_clip_norm == 1.0 for _, cfg in calib)

    pm = build_suite_runs("paper_match", ExperimentConfig())
    assert len(pm) == 50
    assert len({name for name, _ in pm}) == 50
    assert all(cfg.grad_clip_norm == 10.0 for _, cfg in pm)

    diag = build_suite_runs("fpr_ceiling_diag", ExperimentConfig())
    assert len(diag) == 16
    assert len({name for name, _ in diag}) == 16
    assert sum(1 for _, cfg in diag if cfg.samples_per_node == 500) == 8
    assert sum(1 for _, cfg in diag if cfg.samples_per_node == 2500) == 8

    pm50 = build_suite_runs("paper_match_50k", ExperimentConfig())
    assert len(pm50) == 50
    assert len({name for name, _ in pm50}) == 50
    assert all(cfg.samples_per_node == 2500 and cfg.rounds == 25 and cfg.grad_clip_norm == 10.0 for _, cfg in pm50)

    l1 = build_suite_runs("fpr_lever1", ExperimentConfig())
    assert len(l1) == 8
    assert len({name for name, _ in l1}) == 8
    assert all(cfg.trustgraph_delta_mode == "receiver" for _, cfg in l1)

    from dataclasses import asdict
    from experiments.phase1.summarize import method_label

    l1b = build_suite_runs("fpr_lever1b", ExperimentConfig())
    assert len(l1b) == 8
    assert len({name for name, _ in l1b}) == 8
    assert all(cfg.trustgraph_delta_mode == "receiver" for _, cfg in l1b)
    l1b_keys = {
        (cfg.topology, cfg.attack, int(round(cfg.byzantine_frac * 100)), method_label(asdict(cfg)))
        for _, cfg in l1b
    }
    assert len(l1b_keys) == 8

    l2 = build_suite_runs("fpr_lever2", ExperimentConfig())
    assert len(l2) == 8
    assert len({name for name, _ in l2}) == 8
    assert all(cfg.trustgraph_delta_mode == "sender" for _, cfg in l2)
    l2_keys = {
        (cfg.topology, cfg.attack, int(round(cfg.byzantine_frac * 100)), method_label(asdict(cfg)))
        for _, cfg in l2
    }
    assert len(l2_keys) == 8

    l3 = build_suite_runs("fpr_lever3", ExperimentConfig())
    assert len(l3) == 8
    assert len({name for name, _ in l3}) == 8
    assert all(cfg.trustgraph_delta_mode == "sender" for _, cfg in l3)
    l3_keys = {
        (cfg.topology, cfg.attack, int(round(cfg.byzantine_frac * 100)), method_label(asdict(cfg)))
        for _, cfg in l3
    }
    assert len(l3_keys) == 8


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


def test_train_local_epoch_max_steps_caps_updates() -> None:
    from src.models import train_local_epoch

    torch.manual_seed(0)
    m_full = build_model("cnn2x2_gn")
    m_cap = build_model("cnn2x2_gn")
    m_cap.load_state_dict(m_full.state_dict())
    m_exact = build_model("cnn2x2_gn")
    m_exact.load_state_dict(m_full.state_dict())

    x = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 10, (32,))

    # With batch_size=8 and 32 samples, a full epoch takes 4 steps; max_steps=1 stops after 1 batch.
    train_local_epoch(m_full, x, y, lr=0.01, momentum=0.0, weight_decay=0.0, batch_size=8, epochs=1, max_steps=0, generator=torch.Generator().manual_seed(7))
    torch.manual_seed(123)
    train_local_epoch(m_cap, x, y, lr=0.01, momentum=0.0, weight_decay=0.0, batch_size=8, epochs=1, max_steps=1, generator=torch.Generator().manual_seed(7))

    # Manually replicate the exact first batch of the same seeded permutation (matching dropout seed).
    perm = torch.randperm(32, generator=torch.Generator().manual_seed(7))
    torch.manual_seed(123)
    m_exact.train()
    opt = torch.optim.SGD(m_exact.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
    opt.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(m_exact(x[perm[:8]]), y[perm[:8]]).backward()
    opt.step()

    assert torch.allclose(model_to_vector(m_cap), model_to_vector(m_exact), atol=1e-6)
    assert not torch.allclose(model_to_vector(m_cap), model_to_vector(m_full), atol=1e-4)


def test_trustgraph_score_scopes() -> None:
    from src.models import ParamSlice, model_param_slices

    rn = build_model("resnet18_scratch")
    slices = model_param_slices(rn)
    assert slices[-1].end == model_to_vector(rn).numel()
    assert sum(1 for s in slices if s.ndim >= 2) == 21

    # Construct 3 weight matrices where fc has 100x larger norm and opposite sign (-1),
    # while conv1 and layer4 are perfectly aligned (+1).
    synth_slices = [
        ParamSlice("backbone.conv1.weight", 0, 2, 4),
        ParamSlice("backbone.layer4.0.conv1.weight", 2, 4, 4),
        ParamSlice("backbone.fc.weight", 4, 6, 2),
    ]
    self_prev = torch.zeros(6)
    self_upd = torch.tensor([1.0, 0.0, 1.0, 0.0, 100.0, 0.0])
    self_post = self_prev + self_upd
    val_grad = -self_upd  # descent_neg -> ref_vec = +self_upd
    nbr_upd = torch.tensor([1.0, 0.0, 1.0, 0.0, -100.0, 0.0])
    nbr_post = self_prev + nbr_upd

    scores_by_scope: dict[str, float] = {}
    for scope in ("global", "layer_mean", "backbone_only", "head_only"):
        cfg = AggregatorConfig(name="trustgraph", val_grad_sign="descent_neg", trustgraph_score_scope=scope)
        res = aggregate_trustgraph(
            self_post,
            val_grad,
            [1],
            {1: nbr_post},
            {1: nbr_upd},
            cfg,
            self_prev=self_prev,
            self_update=self_upd,
            param_slices=synth_slices,
        )
        scores_by_scope[scope] = res.scores[1]

    assert scores_by_scope["global"] < -0.99
    assert math.isclose(scores_by_scope["layer_mean"], 1.0 / 3.0, abs_tol=1e-6)
    assert math.isclose(scores_by_scope["backbone_only"], 1.0, abs_tol=1e-6)
    assert scores_by_scope["head_only"] < -0.99

    # End-to-end synthetic run with layer_mean scope.
    out = run_experiment(
        ExperimentConfig(
            dataset="synthetic",
            n_nodes=6,
            samples_per_node=40,
            rounds=2,
            model_name="cnn2x2_gn",
            attack="label_flip_all",
            byzantine_frac=0.33,
            aggregator="trustgraph",
            trustgraph_score_scope="layer_mean",
            device="cpu",
            verbose=False,
        )
    )
    assert out["summary"]["had_nan"] is False
