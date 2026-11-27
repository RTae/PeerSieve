from __future__ import annotations

from pathlib import Path
import threading

import pytest
import torch

from src.distributed import _resolve_distributed_rank, run_distributed_peer
from src.models import build_model, model_to_vector
from src.simulator import ExperimentConfig, prepare_dataset, seed_everything
from src.topology import build_topology, select_byzantine_nodes
from src.transport import TcpPeerTransport


def test_resolve_distributed_rank_torchrun_and_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID"):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_LOCALID", "1")
    assert _resolve_distributed_rank() == (3, 4, 1)

    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert _resolve_distributed_rank() == (2, 8, 0)


def test_distributed_peers_end_to_end_socketpair(tmp_path: Path) -> None:
    world_size = 4
    cfg = ExperimentConfig(
        dataset="synthetic",
        n_nodes=world_size,
        samples_per_node=200,
        alpha_dir=5.0,
        val_split="iid",
        rounds=4,
        model_name="cnn2x2",
        topology="ring",
        byzantine_frac=0.25,
        attack="model_poison",
        aggregator="trustgraph",
        val_grad_sign="descent",
        trustgraph_clip_norm=True,
        device="cpu",
        eval_every=4,
        seed=11,
        output_dir=str(tmp_path / "dist_run"),
        verbose=False,
    )
    seed_everything(cfg.seed)
    preloaded = prepare_dataset(cfg)
    shared_init = model_to_vector(build_model(cfg.model_name))
    graph = build_topology("ring", n_nodes=world_size, seed=cfg.seed)
    param_count = int(shared_init.numel())
    transports = TcpPeerTransport.build_socketpair_group(
        graph_neighbors={r: list(graph.neighbors(r)) for r in range(world_size)},
        param_count=param_count,
        device=torch.device("cpu"),
        send_both_vectors=True,
    )

    results: list[dict[str, object] | None] = [None] * world_size
    errors: list[BaseException] = []

    def _peer_worker(r: int) -> None:
        try:
            results[r] = run_distributed_peer(
                cfg=cfg,
                rank=r,
                world_size=world_size,
                local_rank=0,
                transport=transports[r],
                preloaded_data=preloaded,
                init_vec=shared_init,
            )
        except BaseException as exc:
            errors.append(exc)

    try:
        threads = [threading.Thread(target=_peer_worker, args=(r,)) for r in range(world_size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60.0)
    finally:
        for tr in transports:
            tr.close()

    assert not errors, f"Distributed peer errors: {errors}"
    assert results[0] is not None
    assert all(results[r] is None for r in range(1, world_size))
    summary = results[0]["summary"]
    assert summary["execution_mode"] == "distributed"
    assert summary["world_size"] == world_size
    assert summary["had_nan"] is False
    assert (tmp_path / "dist_run" / "summary.json").exists()
    assert (tmp_path / "dist_run" / "history.jsonl").exists()
    sm = summary["system_metrics"]
    assert "measured_transport_sec" in sm
    assert sm["measured_transport_sec"]["net_sec_per_peer"]["repeats"] == 4
    assert sm["measured_transport_sec"]["wire_mb_per_peer_round"] > 0.0
    assert summary["send_both_vectors"] is True
    assert summary["wire_bytes_per_neighbor"] == param_count * 8
    assert float(summary["mean_benign_score"]) > 0.0
    assert float(summary["mean_byz_score"]) < 0.0
    assert float(summary["byz_accept_cumulative"]) == 0.0


def test_distributed_rxdelta_wire_bytes_and_adjacent_byzantine_stability(tmp_path: Path) -> None:
    """Check that receiver-anchored mode sends one vector and adjacent attackers stay finite when rank 0 is Byzantine."""
    world_size = 4
    chosen_seed = next(
        s for s in range(1, 200) if 0 in select_byzantine_nodes(world_size, 0.5, seed=s + 1000)
    )
    cfg = ExperimentConfig(
        dataset="synthetic",
        n_nodes=world_size,
        samples_per_node=80,
        alpha_dir=5.0,
        val_split="iid",
        rounds=4,
        model_name="cnn2x2",
        topology="ring",
        byzantine_frac=0.5,
        attack="model_poison",
        aggregator="trustgraph",
        val_grad_sign="descent",
        trustgraph_clip_norm=True,
        trustgraph_clip_mode="model",
        trustgraph_delta_mode="receiver",
        device="cpu",
        eval_every=4,
        seed=chosen_seed,
        output_dir=str(tmp_path / "dist_rxdelta"),
        verbose=False,
    )
    seed_everything(cfg.seed)
    preloaded = prepare_dataset(cfg)
    shared_init = model_to_vector(build_model(cfg.model_name))
    graph = build_topology("ring", n_nodes=world_size, seed=cfg.seed)
    param_count = int(shared_init.numel())
    transports = TcpPeerTransport.build_socketpair_group(
        graph_neighbors={r: list(graph.neighbors(r)) for r in range(world_size)},
        param_count=param_count,
        device=torch.device("cpu"),
        send_both_vectors=False,
    )

    results: list[dict[str, object] | None] = [None] * world_size
    errors: list[BaseException] = []

    def _peer_worker(r: int) -> None:
        try:
            results[r] = run_distributed_peer(
                cfg=cfg,
                rank=r,
                world_size=world_size,
                local_rank=0,
                transport=transports[r],
                preloaded_data=preloaded,
                init_vec=shared_init,
            )
        except BaseException as exc:
            errors.append(exc)

    try:
        threads = [threading.Thread(target=_peer_worker, args=(r,)) for r in range(world_size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60.0)
    finally:
        for tr in transports:
            tr.close()

    assert not errors, f"Distributed peer errors: {errors}"
    assert results[0] is not None
    summary = results[0]["summary"]
    assert 0 in summary["topology"]["byzantine_ids"]
    assert summary["had_nan"] is False
    assert summary["send_both_vectors"] is False
    assert summary["wire_bytes_per_neighbor"] == param_count * 4
    expected_wire_mb = round((2 * param_count * 4) / 1e6, 2)
    assert summary["system_metrics"]["measured_transport_sec"]["wire_mb_per_peer_round"] == expected_wire_mb
