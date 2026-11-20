"""Unit and offline integration tests for Phase 1 TrustGraph-DFL reproduction."""

from __future__ import annotations

import math

import numpy as np
import torch

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
from src.models import build_model, compute_validation_gradient, model_to_vector
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
    # Honest descent update points opposite to +nabla F (val_grad).
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

    # All-negative fallback: node keeps self_post unchanged (not 0.5 * self_post!).
    res_fb = aggregate_trustgraph(
        self_post, val_grad, [2], {2: nbr_posts[2]}, {2: nbr_updates[2]}, cfg_neg
    )
    assert res_fb.self_fallback is True
    assert torch.allclose(res_fb.new_model, self_post)


def test_balance_krum_trimmed_mean_and_fedavg() -> None:
    self_post = torch.tensor([10.0, 0.0, 0.0, 0.0])
    nbr_posts = {
        1: torch.tensor([10.1, 0.05, 0.0, 0.0]),  # benign
        2: torch.tensor([9.95, -0.05, 0.0, 0.0]),  # benign
        3: torch.tensor([10.05, 0.02, 0.0, 0.0]),  # benign
        4: torch.tensor([-50.0, 50.0, 50.0, 50.0]),  # malicious outlier
    }

    # BALANCE rejects outlier 4 and keeps 1, 2, 3.
    res_bal = aggregate_balance(
        self_post, [1, 2, 3, 4], nbr_posts, round_idx=0, total_rounds=200, cfg=AggregatorConfig(name="balance")
    )
    assert res_bal.neighbor_weights[4] == 0.0
    assert all(res_bal.neighbor_weights[j] > 0.0 for j in (1, 2, 3))

    # Multi-Krum with 25% Byzantine rejects outlier 4.
    res_krum = aggregate_krum(
        0, self_post, [1, 2, 3, 4], nbr_posts, byzantine_frac=0.25, cfg=AggregatorConfig(name="krum")
    )
    assert res_krum.neighbor_weights[4] == 0.0

    # Trimmed mean removes extreme outlier coordinate-wise.
    # Coord 0 sorted: [-50, 9.95, 10.0, 10.05, 10.1] -> mean(9.95, 10.0, 10.05) = 10.0
    # Coord 1 sorted: [-0.05, 0.0, 0.02, 0.05, 50.0] -> mean(0.0, 0.02, 0.05) = 0.07 / 3
    res_tm = aggregate_trimmed_mean(
        0, self_post, [1, 2, 3, 4], nbr_posts, cfg=AggregatorConfig(name="trimmed_mean", trim_ratio=0.3)
    )
    assert torch.allclose(res_tm.new_model, torch.tensor([10.0, 0.07 / 3.0, 0.0, 0.0]), atol=1e-4)

    # Ring degree-2 edge case must not crash for Krum or Trimmed Mean.
    ring_posts = {1: nbr_posts[1], 4: nbr_posts[4]}
    res_ring_k = aggregate_krum(0, self_post, [1, 4], ring_posts, byzantine_frac=0.5, cfg=AggregatorConfig(name="krum"))
    res_ring_tm = aggregate_trimmed_mean(0, self_post, [1, 4], ring_posts, cfg=AggregatorConfig(name="trimmed_mean"))
    assert torch.isfinite(res_ring_k.new_model).all()
    assert torch.isfinite(res_ring_tm.new_model).all()

    # FedAvg matches exact mean.
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
    from src.models import train_local_epoch, vector_to_model

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
    assert float(s_neg["mean_benign_score"]) > 0.0
    assert float(s_neg["mean_byz_score"]) < 0.0
    assert float(s_neg["byz_accept_cumulative"]) == 0.0

    # With literal paper sign (+nabla F), cosine signs flip and Byzantine peers get accepted.
    assert float(s_pos["mean_benign_score"]) < 0.0
    assert float(s_pos["mean_byz_score"]) > 0.0
    assert float(s_pos["byz_accept_cumulative"]) > 0.8
    assert float(s_neg["final_consensus_test_acc"]) > float(s_pos["final_consensus_test_acc"]) + 0.15


def test_step_a_suite_count_and_sharding() -> None:
    from experiments.phase1.run import _suite_runs

    runs = _suite_runs("step_a", ExperimentConfig())
    assert len(runs) == 75
    names = [name for name, _ in runs]
    assert len(names) == len(set(names))
    sharded = [
        [name for idx, (name, _) in enumerate(runs) if idx % 4 == s]
        for s in range(4)
    ]
    assert sum(len(x) for x in sharded) == 75
    assert set().union(*sharded) == set(names)


def test_trustgraph_clip_norm_caps_large_updates_and_preserves_small() -> None:
    self_prev = torch.tensor([1.0, 2.0, 3.0])
    self_update = torch.tensor([0.3, 0.4, 0.0])  # norm r_i = 0.5
    self_post = self_prev + self_update
    val_grad = -self_update  # aligned with self_update

    # Neighbor 1: huge norm (50.0) aligned positively -> must be clipped to norm 0.5.
    u_huge = 100.0 * self_update
    # Neighbor 2: small norm (0.25 < r_i) aligned positively -> must remain unclipped.
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
    # 0.5 * self_update + 0.5 * (0.5 * self_update) = 0.75 * self_update
    assert torch.allclose(res_small.new_model - self_prev, 0.75 * self_update, atol=1e-5)
