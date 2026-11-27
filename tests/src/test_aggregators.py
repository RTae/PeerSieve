from __future__ import annotations

import pytest
import torch

from src.aggregators import AggregatorConfig, aggregate_trustgraph


def test_aggregator_config_validate_rejects_invalid_fields() -> None:
    AggregatorConfig().validate()
    with pytest.raises(ValueError, match="Unknown aggregator"):
        AggregatorConfig(name="fedavg").validate()
    with pytest.raises(ValueError, match="Unknown val_grad_sign"):
        AggregatorConfig(val_grad_sign="invalid_sign").validate()
    with pytest.raises(ValueError, match="Unknown trustgraph_metric"):
        AggregatorConfig(trustgraph_metric="l1").validate()
    with pytest.raises(ValueError, match="Unknown trustgraph_clip_mode"):
        AggregatorConfig(trustgraph_clip_mode="layer").validate()
    with pytest.raises(ValueError, match="Unknown trustgraph_delta_mode"):
        AggregatorConfig(trustgraph_delta_mode="global").validate()


def test_trustgraph_sign_direction_and_self_fallback() -> None:
    val_grad = torch.tensor([1.0, 2.0, -1.0])
    self_post = torch.tensor([10.0, 10.0, 10.0])
    benign_update = -0.1 * val_grad
    mal_update = +1.0 * val_grad
    nbr_posts = {1: self_post + benign_update, 2: self_post + mal_update}
    nbr_updates = {1: benign_update, 2: mal_update}

    cfg_neg = AggregatorConfig(name="trustgraph", val_grad_sign="descent", alpha=0.5)
    res_neg = aggregate_trustgraph(self_post, val_grad, [1, 2], nbr_posts, nbr_updates, cfg_neg)
    assert res_neg.scores[1] > 0.99 and res_neg.scores[2] < -0.99
    assert res_neg.neighbor_weights[1] > 0.99 and res_neg.neighbor_weights[2] == 0.0
    assert res_neg.neighbor_zero_weight_frac == {1: 0.0, 2: 1.0}

    cfg_pos = AggregatorConfig(name="trustgraph", val_grad_sign="ascent", alpha=0.5)
    res_pos = aggregate_trustgraph(self_post, val_grad, [1, 2], nbr_posts, nbr_updates, cfg_pos)
    assert res_pos.scores[1] < -0.99 and res_pos.scores[2] > 0.99
    assert res_pos.neighbor_weights[1] == 0.0 and res_pos.neighbor_weights[2] > 0.99

    res_fb = aggregate_trustgraph(
        self_post, val_grad, [2], {2: nbr_posts[2]}, {2: nbr_updates[2]}, cfg_neg
    )
    assert res_fb.self_fallback is True
    assert torch.allclose(res_fb.new_model, self_post)


def test_trustgraph_empty_neighbors_and_zero_norm_scores() -> None:
    val_grad = torch.tensor([1.0, 0.0, 0.0])
    self_post = torch.tensor([3.0, 4.0, 5.0])
    cfg = AggregatorConfig(val_grad_sign="descent")

    res_empty = aggregate_trustgraph(self_post, val_grad, [], {}, {}, cfg)
    assert res_empty.self_fallback is True
    assert torch.allclose(res_empty.new_model, self_post)

    zero_upd = torch.zeros(3)
    res_zero = aggregate_trustgraph(self_post, val_grad, [1], {1: self_post.clone()}, {1: zero_upd}, cfg)
    assert res_zero.scores[1] == 0.0
    assert res_zero.self_fallback is True


def test_trustgraph_threshold_no_relu_and_euclidean_variants() -> None:
    val_grad = torch.tensor([-1.0, 0.0])
    self_post = torch.tensor([0.0, 0.0])
    # Neighbor 1 has cosine ~0.6; neighbor 2 has cosine ~-0.8.
    u_pos = torch.tensor([0.6, 0.8])
    u_neg = torch.tensor([-0.8, 0.6])
    posts = {1: self_post + u_pos, 2: self_post + u_neg}
    updates = {1: u_pos, 2: u_neg}

    cfg_thresh = AggregatorConfig(val_grad_sign="descent", trustgraph_score_threshold=0.7)
    res_thresh = aggregate_trustgraph(self_post, val_grad, [1, 2], posts, updates, cfg_thresh)
    assert res_thresh.self_fallback is True

    cfg_norelu = AggregatorConfig(val_grad_sign="descent", trustgraph_use_relu=False)
    res_norelu = aggregate_trustgraph(self_post, val_grad, [1, 2], posts, updates, cfg_norelu)
    assert res_norelu.self_fallback is False
    assert res_norelu.neighbor_weights[1] > res_norelu.neighbor_weights[2] > 0.0

    cfg_euc = AggregatorConfig(val_grad_sign="descent", trustgraph_metric="euclidean")
    res_euc = aggregate_trustgraph(self_post, val_grad, [1, 2], posts, updates, cfg_euc)
    assert res_euc.self_fallback is False
    assert res_euc.neighbor_weights[1] > res_euc.neighbor_weights[2] > 0.0


def test_trustgraph_clip_norm_caps_large_updates_and_preserves_small() -> None:
    self_prev = torch.tensor([1.0, 2.0, 3.0])
    self_update = torch.tensor([0.3, 0.4, 0.0])
    self_post = self_prev + self_update
    val_grad = -self_update

    u_huge = 100.0 * self_update
    u_small = 0.5 * self_update
    cfg = AggregatorConfig(name="trustgraph", val_grad_sign="descent", trustgraph_clip_norm=True, alpha=0.5)

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


def test_trustgraph_model_space_clip_caps_displacement() -> None:
    self_prev = torch.tensor([10.0, 0.0, 0.0])
    self_update = torch.tensor([0.3, 0.4, 0.0])
    self_post = self_prev + self_update
    val_grad = -self_update
    nbr_far = self_prev + torch.tensor([100.0, 0.0, 0.0])
    nbr_upd = torch.tensor([0.3, 0.4, 0.0])

    cfg = AggregatorConfig(
        name="trustgraph",
        val_grad_sign="descent",
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
    expected = 0.5 * self_post + 0.5 * (self_prev + torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(res.new_model, expected, atol=1e-5)


def test_trustgraph_receiver_anchored_delta_mode() -> None:
    self_prev = torch.tensor([0.0, 0.0, 0.0])
    self_update = torch.tensor([0.2, 0.0, 0.0])
    self_post = self_prev + self_update
    val_grad = torch.tensor([-1.0, 0.0, 0.0])
    nbr_post = torch.tensor([0.95, 0.1, 0.0])
    nbr_sender_update = torch.tensor([-0.05, 0.1, 0.0])

    cfg_sender = AggregatorConfig(name="trustgraph", val_grad_sign="descent", trustgraph_delta_mode="sender")
    res_s = aggregate_trustgraph(
        self_post, val_grad, [1], {1: nbr_post}, {1: nbr_sender_update}, cfg_sender, self_prev=self_prev, self_update=self_update
    )
    assert res_s.scores[1] < 0.0 and res_s.neighbor_weights[1] == 0.0

    cfg_rx = AggregatorConfig(name="trustgraph", val_grad_sign="descent", trustgraph_delta_mode="receiver")
    res_r = aggregate_trustgraph(
        self_post, val_grad, [1], {1: nbr_post}, {1: nbr_sender_update}, cfg_rx, self_prev=self_prev, self_update=self_update
    )
    assert res_r.scores[1] > 0.99 and res_r.neighbor_weights[1] > 0.99

    with pytest.raises(ValueError, match="self_prev is required"):
        aggregate_trustgraph(self_post, val_grad, [1], {1: nbr_post}, {1: nbr_sender_update}, cfg_rx)
