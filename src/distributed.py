from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from src.attacks import label_flip_mode_for_attack
from src.data import PartitionedDataset, apply_label_flip
from src.metrics import MetricsTracker
from src.models import build_model, evaluate_accuracy, model_to_vector, vector_to_model
from src.peer import PeerNode
from src.simulator import (
    ExperimentConfig,
    learning_rate_for_round,
    prepare_dataset,
    resolve_device,
    seed_everything,
    should_eval,
)
from src.topology import build_topology, select_byzantine_nodes, topology_summary
from src.transport import TcpPeerTransport, probe_interface_info


def _resolve_distributed_rank() -> tuple[int, int, int]:
    """Read distributed process rank metadata from torchrun or Slurm environment variables.

    Returns:
        Tuple of (rank, world_size, local_rank) integers.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        return rank, world_size, local_rank
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    local_rank = int(os.environ.get("SLURM_LOCALID", str(rank)))
    return rank, world_size, local_rank


def _resolve_rank_device(requested: str, local_rank: int) -> torch.device:
    """Select the compute device for this rank and bind CUDA ranks by local_rank.

    Args:
        requested: Device string ('auto', 'cpu', 'cuda', or 'mps').
        local_rank: Node-local rank used to pick a visible CUDA GPU.

    Returns:
        Resolved torch.device for this peer process.
    """
    base = resolve_device(requested)
    if base.type == "cuda":
        n_gpu = max(1, torch.cuda.device_count())
        gpu_idx = local_rank % n_gpu
        torch.cuda.set_device(gpu_idx)
        return torch.device(f"cuda:{gpu_idx}")
    return base


def _register_and_discover_peers(
    rank: int,
    world_size: int,
    coord_dir: Path,
    iface: str | None,
    base_port: int,
    device: torch.device,
    timeout_sec: float = 90.0,
) -> list[dict[str, object]]:
    """Publish this peer's TCP endpoint into a shared directory and wait for all ranks.

    Args:
        rank: Global peer rank in [0, world_size - 1].
        world_size: Total number of participating peers.
        coord_dir: Shared filesystem directory used for peer endpoint files.
        iface: Optional Linux network interface name (defaults to loopback).
        base_port: Base TCP port; rank listens on base_port + rank.
        device: Compute device assigned to this peer.
        timeout_sec: Maximum seconds to wait for all peers to register.

    Returns:
        Ordered list of peer endpoint dictionaries for ranks 0..world_size-1.
    """
    coord_dir.mkdir(parents=True, exist_ok=True)
    net_info = probe_interface_info(iface)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    my_record: dict[str, object] = {
        "rank": rank,
        "local_rank": int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", str(rank)))),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "port": base_port + rank,
        "device": str(device),
        "gpu_name": gpu_name,
        **net_info,
    }
    tmp_p = coord_dir / f"peer_{rank}.tmp"
    fin_p = coord_dir / f"peer_{rank}.json"
    tmp_p.write_text(json.dumps(my_record), encoding="utf-8")
    tmp_p.replace(fin_p)

    max_checks = max(1, int(timeout_sec / 0.25))
    for _ in range(max_checks):
        if all((coord_dir / f"peer_{r}.json").exists() for r in range(world_size)):
            return [
                json.loads((coord_dir / f"peer_{r}.json").read_text(encoding="utf-8"))
                for r in range(world_size)
            ]
        time.sleep(0.25)
    raise RuntimeError(f"Rank {rank} timed out waiting for {world_size} peer endpoints in {coord_dir}.")


def run_distributed_peer(
    cfg: ExperimentConfig,
    rank: int,
    world_size: int,
    local_rank: int,
    transport: TcpPeerTransport | None = None,
    preloaded_data: PartitionedDataset | None = None,
    init_vec: torch.Tensor | None = None,
    iface: str | None = None,
    base_port: int = 29600,
    coord_dir: Path | None = None,
) -> dict[str, object] | None:
    """Execute one decentralized FL peer process (one rank = one PeerNode).

    Args:
        cfg: Base experiment configuration (n_nodes is overridden with world_size).
        rank: Global peer rank in [0, world_size - 1].
        world_size: Total number of peer processes.
        local_rank: Node-local rank used for GPU assignment.
        transport: Optional pre-built TcpPeerTransport (used by offline socketpair tests).
        preloaded_data: Optional pre-partitioned dataset shared across threads in tests.
        init_vec: Optional shared initial parameter vector to bypass per-rank model init RNG.
        iface: Optional Linux network interface name for TCP sockets.
        base_port: Base TCP port for peer listeners.
        coord_dir: Optional coordination directory for endpoint discovery.

    Returns:
        Dictionary with 'summary' and 'history' on rank 0, or None on non-zero ranks.
    """
    run_cfg = replace(cfg, n_nodes=world_size)
    agg_cfg = run_cfg.aggregator_config()
    atk_cfg = run_cfg.attack_config()
    if init_vec is None:
        seed_everything(run_cfg.seed)
    device = _resolve_rank_device(run_cfg.device, local_rank)

    data = preloaded_data if preloaded_data is not None else prepare_dataset(run_cfg)
    graph = build_topology(
        name=run_cfg.topology,
        n_nodes=world_size,
        er_p=run_cfg.er_p,
        ws_k=run_cfg.ws_k,
        ws_p=run_cfg.ws_p,
        seed=run_cfg.seed,
    )
    byz_frac_eff = 0.0 if run_cfg.attack == "none" else run_cfg.byzantine_frac
    byzantine_ids = select_byzantine_nodes(world_size, byz_frac_eff, seed=run_cfg.seed + 1000)
    benign_ids = [i for i in range(world_size) if i not in byzantine_ids]
    topo_info = topology_summary(graph, byzantine_ids)

    neighbors = list(graph.neighbors(rank))
    is_byz = rank in byzantine_ids
    shard = data.nodes[rank]
    flip_mode = label_flip_mode_for_attack(run_cfg.attack)
    y_train_local = apply_label_flip(shard.y_train, mode=flip_mode) if (flip_mode and is_byz) else shard.y_train

    model = build_model(run_cfg.model_name, num_classes=data.num_classes).to(device)
    peer_init_vec = init_vec.clone().to(device) if init_vec is not None else model_to_vector(model)
    param_count = int(peer_init_vec.numel())

    peer = PeerNode(
        peer_id=rank,
        neighbors=neighbors,
        is_byzantine=is_byz,
        model=model,
        init_vec=peer_init_vec,
        x_train=shard.x_train.to(device),
        y_train=y_train_local.to(device),
        x_val=shard.x_val.to(device),
        y_val=shard.y_val.to(device),
        device=device,
    )
    x_test = data.x_test.to(device)
    y_test = data.y_test.to(device)

    owns_transport = transport is None
    peer_endpoints: list[dict[str, object]] = []
    if transport is None:
        out_base = Path(run_cfg.output_dir) if run_cfg.output_dir else Path("outputs/distributed_tmp")
        run_nonce = (
            os.environ.get("DIST_RUN_ID")
            or os.environ.get("SLURM_JOB_ID")
            or os.environ.get("TORCHELASTIC_RUN_ID")
            or f"port{base_port}"
        )
        cdir = coord_dir if coord_dir is not None else out_base / f"coord_{run_nonce}"
        peer_endpoints = _register_and_discover_peers(
            rank=rank,
            world_size=world_size,
            coord_dir=cdir,
            iface=iface,
            base_port=base_port,
            device=device,
        )
        send_both = run_cfg.trustgraph_delta_mode != "receiver"
        transport = TcpPeerTransport.connect_tcp(
            rank=rank,
            world_size=world_size,
            neighbors=neighbors,
            param_count=param_count,
            device=device,
            peers=peer_endpoints,
            send_both_vectors=send_both,
        )

    out_path = Path(run_cfg.output_dir) if (rank == 0 and run_cfg.output_dir) else None
    history_file = None
    tracker = MetricsTracker(graph, benign_ids, byzantine_ids, param_count, run_cfg.rounds) if rank == 0 else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": asdict(run_cfg),
                    "execution_mode": "distributed",
                    "world_size": world_size,
                    "resolved_device": str(device),
                    "param_count": param_count,
                    "topology": topo_info,
                    "peers": peer_endpoints,
                },
                f,
                indent=2,
            )
        history_file = open(out_path / "history.jsonl", "w", encoding="utf-8")

    train_gen = torch.Generator(device="cpu").manual_seed(run_cfg.seed + 2000 + rank)
    attack_gen = torch.Generator(device="cpu").manual_seed(run_cfg.seed + 3000 + rank)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    had_nan = False
    t_run_start = time.perf_counter()

    try:
        transport.barrier()
        for r in range(run_cfg.rounds):
            t_round_start = time.perf_counter()
            lr_now = learning_rate_for_round(run_cfg, r)
            node_epochs = (
                run_cfg.byz_local_epochs if (is_byz and run_cfg.byz_local_epochs > 0) else run_cfg.local_epochs
            )
            step = peer.local_step(
                lr=lr_now,
                momentum=run_cfg.momentum,
                weight_decay=run_cfg.weight_decay,
                batch_size=run_cfg.batch_size,
                epochs=node_epochs,
                augment=run_cfg.augment,
                grad_clip_norm=run_cfg.grad_clip_norm,
                val_grad_weight_decay=run_cfg.val_grad_weight_decay,
                train_gen=train_gen,
                skip_training=False,
            )
            payload = peer.prepare_decentralized_payload(step, atk_cfg, attack_gen)
            received, tx_stats = transport.exchange(rank, neighbors, payload)
            agg_res, agg_sec = peer.aggregate_received(step, received, agg_cfg, run_cfg.attack)

            local_finite = peer.has_finite_weights()
            do_eval = should_eval(r, run_cfg.rounds, run_cfg.eval_every)
            local_eval: tuple[float, float] | None = None
            if do_eval and not is_byz:
                local_eval = peer.evaluate(x_test, y_test, run_cfg.eval_batch_size)

            edge_report: list[dict[str, object]] = []
            byz_w_share = 0.0
            self_fallback = False
            if agg_res is not None:
                self_fallback = agg_res.self_fallback
                total_w = sum(agg_res.neighbor_weights.values())
                byz_w = sum(agg_res.neighbor_weights[j] for j in neighbors if j in byzantine_ids)
                byz_w_share = (byz_w / total_w) if total_w > 0.0 else 0.0
                for j in neighbors:
                    edge_report.append(
                        {
                            "sender_id": j,
                            "zero_frac": agg_res.neighbor_zero_weight_frac[j],
                            "score": agg_res.scores.get(j),
                        }
                    )

            peer_round_msg: dict[str, object] = {
                "rank": rank,
                "is_byzantine": is_byz,
                "finite": local_finite,
                "train_sec": step.train_sec,
                "val_grad_sec": step.val_grad_sec,
                "agg_sec": agg_sec,
                "d2h_sec": tx_stats.d2h_sec,
                "net_sec": tx_stats.net_sec,
                "h2d_sec": tx_stats.h2d_sec,
                "wire_bytes_recv": tx_stats.bytes_recv,
                "self_fallback": self_fallback,
                "byz_weight_share": byz_w_share,
                "edges": edge_report,
                "eval_acc": local_eval[0] if local_eval is not None else None,
                "eval_loss": local_eval[1] if local_eval is not None else None,
            }
            gathered_msgs = transport.gather_json(peer_round_msg)
            gathered_vecs = transport.gather_tensors(peer.model_vec) if do_eval else None

            if rank == 0 and tracker is not None and gathered_msgs is not None:
                ben_msgs = [m for m in gathered_msgs if not m["is_byzantine"]]
                if any(not bool(m["finite"]) for m in ben_msgs):
                    had_nan = True

                n_ben = max(len(ben_msgs), 1)
                train_mean = sum(float(m["train_sec"]) for m in ben_msgs) / n_ben
                val_mean = sum(float(m["val_grad_sec"]) for m in ben_msgs) / n_ben
                agg_stats = tracker.summarize_benign_round(
                    benign_reports=ben_msgs,
                    agg_sec_per_peer=sum(float(m["agg_sec"]) for m in ben_msgs) / n_ben,
                    d2h_sec_per_peer=sum(float(m["d2h_sec"]) for m in ben_msgs) / n_ben,
                    net_sec_per_peer=sum(float(m["net_sec"]) for m in ben_msgs) / n_ben,
                    h2d_sec_per_peer=sum(float(m["h2d_sec"]) for m in ben_msgs) / n_ben,
                    wire_bytes_recv_per_peer=sum(float(m["wire_bytes_recv"]) for m in ben_msgs) / n_ben,
                )

                eval_metrics: dict[str, float] | None = None
                if do_eval and gathered_vecs is not None:
                    accs = [float(m["eval_acc"]) for m in ben_msgs if m["eval_acc"] is not None]
                    losses = [float(m["eval_loss"]) for m in ben_msgs if m["eval_loss"] is not None]
                    cons_vec = torch.stack([gathered_vecs[i] for i in benign_ids], dim=0).mean(dim=0).to(device)
                    vector_to_model(cons_vec, peer.model)
                    cons_acc, cons_loss = evaluate_accuracy(peer.model, x_test, y_test, batch_size=run_cfg.eval_batch_size)
                    eval_metrics = {
                        "benign_test_acc_mean": float(np.mean(accs)),
                        "benign_test_acc_std": float(np.std(accs)),
                        "benign_test_acc_min": float(np.min(accs)),
                        "benign_test_acc_max": float(np.max(accs)),
                        "benign_test_loss_mean": float(np.mean(losses)),
                        "consensus_test_acc": float(cons_acc),
                        "consensus_test_loss": float(cons_loss),
                    }

                rec = tracker.record_round(
                    round_idx=r,
                    train_sec_per_peer=train_mean,
                    val_grad_sec_per_peer=val_mean,
                    agg_stats=agg_stats,
                    eval_metrics=eval_metrics,
                    round_wall_sec=time.perf_counter() - t_round_start,
                )
                if history_file is not None:
                    history_file.write(json.dumps(rec) + "\n")
                    history_file.flush()
                if run_cfg.verbose and do_eval:
                    print(
                        f"[dist r={r + 1:3d}/{run_cfg.rounds}] "
                        f"acc={100.0 * float(rec['benign_test_acc_mean']):5.2f}% "
                        f"(cons={100.0 * float(rec['consensus_test_acc']):5.2f}%) "
                        f"fpr={100.0 * float(rec['fpr_round']):5.1f}% "
                        f"byz_acc={100.0 * float(rec['byz_accept_rate']):5.1f}% "
                        f"net={float(rec.get('net_sec_per_peer', 0.0)):.3f}s "
                        f"({float(rec['round_wall_sec']):.2f}s)",
                        flush=True,
                    )

        local_peak_alloc = (
            round(float(torch.cuda.max_memory_allocated(device)) / float(1024 ** 2), 2)
            if device.type == "cuda"
            else None
        )
        local_peak_res = (
            round(float(torch.cuda.max_memory_reserved(device)) / float(1024 ** 2), 2)
            if device.type == "cuda"
            else None
        )
        mem_reports = transport.gather_json(
            {"rank": rank, "peak_gpu_alloc_mib": local_peak_alloc, "peak_gpu_reserved_mib": local_peak_res}
        )

        if rank == 0 and tracker is not None:
            allocs = [float(m["peak_gpu_alloc_mib"]) for m in (mem_reports or []) if m["peak_gpu_alloc_mib"] is not None]
            resvs = [float(m["peak_gpu_reserved_mib"]) for m in (mem_reports or []) if m["peak_gpu_reserved_mib"] is not None]
            summary = tracker.finalize_summary(
                config_dict=asdict(run_cfg),
                resolved_device=str(device),
                topo_info=topo_info,
                total_wall_sec=time.perf_counter() - t_run_start,
                had_nan=had_nan,
                peak_gpu_alloc_mib=round(max(allocs), 2) if allocs else None,
                peak_gpu_reserved_mib=round(max(resvs), 2) if resvs else None,
            )
            summary["execution_mode"] = "distributed"
            summary["world_size"] = world_size
            summary["send_both_vectors"] = bool(transport.send_both_vectors)
            summary["wire_bytes_per_neighbor"] = int(param_count * 4 * (2 if transport.send_both_vectors else 1))
            summary["peers"] = peer_endpoints
            if out_path is not None:
                with open(out_path / "summary.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
            return {"summary": summary, "history": tracker.history}
        return None
    finally:
        if history_file is not None:
            history_file.close()
        if owns_transport and transport is not None:
            transport.close()


def run_distributed_experiment(
    cfg: ExperimentConfig,
    iface: str | None = None,
    base_port: int = 29600,
) -> dict[str, object] | None:
    """Entry point for srun or torchrun distributed DFL execution.

    Args:
        cfg: Base experiment configuration.
        iface: Optional Linux network interface name for TCP transport.
        base_port: Base TCP port for peer listeners.

    Returns:
        Dictionary with 'summary' and 'history' on rank 0, or None on other ranks.
    """
    rank, world_size, local_rank = _resolve_distributed_rank()
    return run_distributed_peer(
        cfg=cfg,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        iface=iface,
        base_port=base_port,
    )
