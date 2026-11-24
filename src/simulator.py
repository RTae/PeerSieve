"""Synchronous single-process DFL runner built on PeerNode, LocalTransport, and MetricsTracker."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn

from src.aggregators import AggregatorConfig
from src.attacks import AttackConfig, apply_round_attack, label_flip_mode_for_attack
from src.data import (
    PartitionedDataset,
    apply_label_flip,
    load_cifar10_tensors,
    make_synthetic_cifar_tensors,
    partition_dirichlet_balanced,
)
from src.metrics import MetricsTracker, RoundAggregationStats
from src.models import build_model, evaluate_accuracy, model_to_vector, vector_to_model
from src.peer import LocalStepResult, PeerNode, PeerPayload, sync_device
from src.topology import build_topology, select_byzantine_nodes, topology_summary
from src.transport import LocalTransport


@dataclass
class ExperimentConfig:
    # Dataset and partitioning (TrustGraph-DFL Section 5.1)
    dataset: str = "cifar10"  # cifar10 or synthetic
    data_dir: str = "data"
    download_data: bool = True
    n_nodes: int = 20
    samples_per_node: int = 500
    alpha_dir: float = 0.5
    val_frac: float = 0.1
    val_split: str = "local"  # local (paper) or iid (balanced validation)
    # Model and local training (TrustGraph-DFL Section 5.1)
    model_name: str = "cnn2x2"
    rounds: int = 200
    local_epochs: int = 1
    lr: float = 0.01
    lr_schedule: str = "constant"  # constant or cosine
    momentum: float = 0.9
    weight_decay: float = 1e-4
    batch_size: int = 64
    augment: bool = False
    grad_clip_norm: float = 0.0
    val_grad_weight_decay: bool = False
    byz_local_epochs: int = 0
    # Topology
    topology: str = "random"  # ring, random, or small_world
    er_p: float = 0.3
    ws_k: int = 4
    ws_p: float = 0.3
    # Byzantine setup
    byzantine_frac: float = 0.3
    attack: str = "label_flip"  # none, label_flip, label_flip_all, model_poison, or random_noise
    poison_lambda: float = 10.0
    noise_sigma: float = 0.5
    # Aggregator
    aggregator: str = "trustgraph"
    alpha: float = 0.5
    val_grad_sign: str = "descent_neg"  # descent_neg or paper_pos
    trustgraph_use_relu: bool = True
    trustgraph_metric: str = "cosine"
    trustgraph_score_threshold: float = 0.0
    trustgraph_clip_norm: bool = False
    trustgraph_clip_mode: str = "update"
    trustgraph_clip_factor: float = 2.0
    trustgraph_delta_mode: str = "sender"
    # Execution and logging
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda, or mps
    eval_every: int = 5
    eval_batch_size: int = 512
    output_dir: str | None = None
    verbose: bool = True

    def aggregator_config(self) -> AggregatorConfig:
        cfg = AggregatorConfig(
            name=self.aggregator,
            alpha=self.alpha,
            val_grad_sign=self.val_grad_sign,
            trustgraph_use_relu=self.trustgraph_use_relu,
            trustgraph_metric=self.trustgraph_metric,
            trustgraph_score_threshold=self.trustgraph_score_threshold,
            trustgraph_clip_norm=self.trustgraph_clip_norm,
            trustgraph_clip_mode=self.trustgraph_clip_mode,
            trustgraph_clip_factor=self.trustgraph_clip_factor,
            trustgraph_delta_mode=self.trustgraph_delta_mode,
        )
        cfg.validate()
        return cfg

    def attack_config(self) -> AttackConfig:
        cfg = AttackConfig(
            name=self.attack,
            poison_lambda=self.poison_lambda,
            noise_sigma=self.noise_sigma,
        )
        cfg.validate()
        return cfg


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dataset(cfg: ExperimentConfig) -> PartitionedDataset:
    if cfg.dataset == "cifar10":
        x_tr, y_tr, x_te, y_te = load_cifar10_tensors(cfg.data_dir, download=cfg.download_data)
    elif cfg.dataset == "synthetic":
        n_total = cfg.n_nodes * cfg.samples_per_node
        x_tr, y_tr, x_te, y_te = make_synthetic_cifar_tensors(
            n_train=max(n_total, 2000),
            n_test=1000,
            seed=cfg.seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset!r}")

    return partition_dirichlet_balanced(
        x_train=x_tr,
        y_train=y_tr,
        x_test=x_te,
        y_test=y_te,
        n_nodes=cfg.n_nodes,
        samples_per_node=cfg.samples_per_node,
        alpha_dir=cfg.alpha_dir,
        val_frac=cfg.val_frac,
        val_split=cfg.val_split,
        seed=cfg.seed,
    )


def learning_rate_for_round(cfg: ExperimentConfig, round_idx: int) -> float:
    if cfg.lr_schedule == "constant":
        return cfg.lr
    if cfg.lr_schedule == "cosine":
        lr_min = 0.05 * cfg.lr
        frac = round_idx / max(cfg.rounds - 1, 1)
        return lr_min + 0.5 * (cfg.lr - lr_min) * (1.0 + math.cos(math.pi * frac))
    raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule!r}")


def should_eval(round_idx: int, total_rounds: int, eval_every: int) -> bool:
    if eval_every <= 0:
        return round_idx == total_rounds - 1
    return (round_idx == 0) or ((round_idx + 1) % eval_every == 0) or (round_idx == total_rounds - 1)


def _evaluate_benign_peers(
    peers: list[PeerNode],
    benign_ids: list[int],
    work_model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    eval_batch_size: int,
) -> dict[str, float]:
    """Evaluate all benign peer models and their consensus average on the held-out test set."""
    accs: list[float] = []
    losses: list[float] = []
    for i in benign_ids:
        acc, loss = peers[i].evaluate(x_test, y_test, eval_batch_size)
        accs.append(acc)
        losses.append(loss)

    consensus_vec = torch.stack([peers[i].model_vec for i in benign_ids], dim=0).mean(dim=0)
    vector_to_model(consensus_vec, work_model)
    cons_acc, cons_loss = evaluate_accuracy(work_model, x_test, y_test, batch_size=eval_batch_size)
    return {
        "benign_test_acc_mean": float(np.mean(accs)),
        "benign_test_acc_std": float(np.std(accs)),
        "benign_test_acc_min": float(np.min(accs)),
        "benign_test_acc_max": float(np.max(accs)),
        "benign_test_loss_mean": float(np.mean(losses)),
        "consensus_test_acc": float(cons_acc),
        "consensus_test_loss": float(cons_loss),
    }


def run_experiment(
    cfg: ExperimentConfig,
    preloaded_data: PartitionedDataset | None = None,
) -> dict[str, object]:
    """Run one complete synchronous DFL experiment over PeerNode and LocalTransport."""
    agg_cfg = cfg.aggregator_config()
    atk_cfg = cfg.attack_config()
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    data = preloaded_data if preloaded_data is not None else prepare_dataset(cfg)
    graph = build_topology(
        name=cfg.topology,
        n_nodes=cfg.n_nodes,
        er_p=cfg.er_p,
        ws_k=cfg.ws_k,
        ws_p=cfg.ws_p,
        seed=cfg.seed,
    )
    byz_frac_eff = 0.0 if cfg.attack == "none" else cfg.byzantine_frac
    byzantine_ids = select_byzantine_nodes(cfg.n_nodes, byz_frac_eff, seed=cfg.seed + 1000)
    benign_ids = [i for i in range(cfg.n_nodes) if i not in byzantine_ids]
    topo_info = topology_summary(graph, byzantine_ids)

    work_model = build_model(cfg.model_name, num_classes=data.num_classes).to(device)
    init_vec = model_to_vector(work_model)
    param_count = int(init_vec.numel())

    flip_mode = label_flip_mode_for_attack(cfg.attack)
    peers: list[PeerNode] = []
    for i, nd in enumerate(data.nodes):
        is_byz = i in byzantine_ids
        y_i = apply_label_flip(nd.y_train, mode=flip_mode) if (flip_mode and is_byz) else nd.y_train
        peers.append(
            PeerNode(
                peer_id=i,
                neighbors=list(graph.neighbors(i)),
                is_byzantine=is_byz,
                model=work_model,
                init_vec=init_vec,
                x_train=nd.x_train.to(device),
                y_train=y_i.to(device),
                x_val=nd.x_val.to(device),
                y_val=nd.y_val.to(device),
                device=device,
            )
        )

    x_test = data.x_test.to(device)
    y_test = data.y_test.to(device)
    transport = LocalTransport()
    tracker = MetricsTracker(graph, benign_ids, byzantine_ids, param_count, cfg.rounds)

    train_gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 2000)
    attack_gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 3000)

    out_path = Path(cfg.output_dir) if cfg.output_dir else None
    history_file = None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": asdict(cfg),
                    "execution_mode": "local",
                    "resolved_device": str(device),
                    "param_count": param_count,
                    "topology": topo_info,
                },
                f,
                indent=2,
            )
        history_file = open(out_path / "history.jsonl", "w", encoding="utf-8")

    had_nan = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_run_start = time.perf_counter()

    try:
        for r in range(cfg.rounds):
            t_round_start = time.perf_counter()
            lr_now = learning_rate_for_round(cfg, r)

            steps: list[LocalStepResult] = []
            benign_train_sec = 0.0
            benign_val_sec = 0.0
            n_benign_seen = 0
            for i, peer in enumerate(peers):
                skip = peer.is_byzantine and cfg.attack in {"model_poison", "random_noise"}
                node_epochs = (
                    cfg.byz_local_epochs if (peer.is_byzantine and cfg.byz_local_epochs > 0) else cfg.local_epochs
                )
                step = peer.local_step(
                    lr=lr_now,
                    momentum=cfg.momentum,
                    weight_decay=cfg.weight_decay,
                    batch_size=cfg.batch_size,
                    epochs=node_epochs,
                    augment=cfg.augment,
                    grad_clip_norm=cfg.grad_clip_norm,
                    val_grad_weight_decay=cfg.val_grad_weight_decay,
                    train_gen=train_gen,
                    skip_training=skip,
                )
                steps.append(step)
                if not peer.is_byzantine:
                    benign_train_sec += step.train_sec
                    benign_val_sec += step.val_grad_sec
                    n_benign_seen += 1

            prev_models = [s.prev_model for s in steps]
            post_models = [s.post_model for s in steps]
            updates = [s.update for s in steps]
            apply_round_attack(
                attack_cfg=atk_cfg,
                byzantine_ids=byzantine_ids,
                benign_ids=benign_ids,
                prev_models=prev_models,
                post_models=post_models,
                updates=updates,
                generator=attack_gen,
            )

            outgoing = {
                i: PeerPayload(sender_id=i, post_model=post_models[i], update=updates[i])
                for i in range(cfg.n_nodes)
            }
            transport.stage_round(outgoing)

            benign_zero = 0.0
            benign_pairs = 0
            byz_accept = 0.0
            byz_pairs = 0
            byz_weight_shares: list[float] = []
            benign_scores: list[float] = []
            byz_scores: list[float] = []
            self_fallbacks = 0

            sync_device(device)
            t_agg0 = time.perf_counter()
            for i in benign_ids:
                peer = peers[i]
                received, _ = transport.exchange(i, peer.neighbors, outgoing[i])
                res, _ = peer.aggregate_received(steps[i], received, agg_cfg, cfg.attack)
                assert res is not None
                if res.self_fallback:
                    self_fallbacks += 1
                total_nbr_w = sum(res.neighbor_weights.values())
                byz_w = 0.0
                for j in peer.neighbors:
                    z_frac = res.neighbor_zero_weight_frac[j]
                    tracker.record_edge_outcome(i, j, z_frac >= 0.5)
                    if j in byzantine_ids:
                        byz_pairs += 1
                        byz_accept += 1.0 - z_frac
                        byz_w += res.neighbor_weights[j]
                        if j in res.scores:
                            byz_scores.append(res.scores[j])
                    else:
                        benign_pairs += 1
                        benign_zero += z_frac
                        if j in res.scores:
                            benign_scores.append(res.scores[j])
                byz_weight_shares.append((byz_w / total_nbr_w) if total_nbr_w > 0.0 else 0.0)
            sync_device(device)
            agg_sec_per_peer = (time.perf_counter() - t_agg0) / max(len(benign_ids), 1)

            if byzantine_ids:
                if cfg.attack in {"label_flip", "label_flip_all"}:
                    for m in byzantine_ids:
                        received, _ = transport.exchange(m, peers[m].neighbors, outgoing[m])
                        peers[m].aggregate_received(steps[m], received, agg_cfg, cfg.attack)
                else:
                    benign_next_mean = torch.stack([peers[i].model_vec for i in benign_ids], dim=0).mean(dim=0)
                    for m in byzantine_ids:
                        peers[m].model_vec = benign_next_mean.clone()

            if not had_nan and any(not peers[i].has_finite_weights() for i in benign_ids):
                had_nan = True

            agg_stats = RoundAggregationStats(
                benign_zero=benign_zero,
                benign_pairs=benign_pairs,
                byz_accept=byz_accept,
                byz_pairs=byz_pairs,
                byz_weight_share=float(np.mean(byz_weight_shares)) if byz_weight_shares else 0.0,
                self_fallback_rate=self_fallbacks / max(len(benign_ids), 1),
                benign_score_mean=float(np.mean(benign_scores)) if benign_scores else None,
                byz_score_mean=float(np.mean(byz_scores)) if byz_scores else None,
                agg_sec_per_peer=agg_sec_per_peer,
            )

            do_eval = should_eval(r, cfg.rounds, cfg.eval_every)
            eval_metrics = (
                _evaluate_benign_peers(peers, benign_ids, work_model, x_test, y_test, cfg.eval_batch_size)
                if do_eval
                else None
            )
            denom = max(n_benign_seen, 1)
            record = tracker.record_round(
                round_idx=r,
                train_sec_per_peer=benign_train_sec / denom,
                val_grad_sec_per_peer=benign_val_sec / denom,
                agg_stats=agg_stats,
                eval_metrics=eval_metrics,
                round_wall_sec=time.perf_counter() - t_round_start,
            )
            if history_file is not None:
                history_file.write(json.dumps(record) + "\n")
                history_file.flush()

            if cfg.verbose and do_eval:
                score_str = ""
                if record["benign_score_mean"] is not None:
                    b_s = float(record["benign_score_mean"])
                    m_s = record["byz_score_mean"]
                    m_txt = f"{float(m_s):+.3f}" if m_s is not None else "n/a"
                    score_str = f" cos(ben={b_s:+.3f},byz={m_txt})"
                print(
                    f"[r={r + 1:3d}/{cfg.rounds}] "
                    f"acc={100.0 * float(record['benign_test_acc_mean']):5.2f}% "
                    f"(cons={100.0 * float(record['consensus_test_acc']):5.2f}%) "
                    f"fpr={100.0 * float(record['fpr_round']):5.1f}% "
                    f"byz_acc={100.0 * float(record['byz_accept_rate']):5.1f}%"
                    f"{score_str} "
                    f"({float(record['round_wall_sec']):.2f}s)",
                    flush=True,
                )
    finally:
        if history_file is not None:
            history_file.close()
        transport.close()

    peak_alloc = (
        round(float(torch.cuda.max_memory_allocated(device)) / float(1024 ** 2), 2)
        if device.type == "cuda"
        else None
    )
    peak_resv = (
        round(float(torch.cuda.max_memory_reserved(device)) / float(1024 ** 2), 2)
        if device.type == "cuda"
        else None
    )
    summary = tracker.finalize_summary(
        config_dict=asdict(cfg),
        resolved_device=str(device),
        topo_info=topo_info,
        total_wall_sec=time.perf_counter() - t_run_start,
        had_nan=had_nan,
        peak_gpu_alloc_mib=peak_alloc,
        peak_gpu_reserved_mib=peak_resv,
    )
    summary["execution_mode"] = "local"
    if out_path is not None:
        with open(out_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return {"summary": summary, "history": tracker.history}
