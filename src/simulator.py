"""Synchronous decentralized FL simulator for Phase 1 experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from src.aggregators import AggregatorConfig, aggregate_node
from src.attacks import AttackConfig, apply_round_attack, label_flip_mode_for_attack
from src.data import (
    PartitionedDataset,
    apply_label_flip,
    load_cifar10_tensors,
    make_synthetic_cifar_tensors,
    partition_dirichlet_balanced,
)
from src.models import (
    ParamSlice,
    build_model,
    compute_validation_gradient,
    evaluate_accuracy,
    model_param_slices,
    model_to_vector,
    train_local_epoch,
    vector_to_model,
)
from src.topology import build_topology, select_byzantine_nodes, topology_summary


@dataclass
class ExperimentConfig:
    # Dataset and partitioning (TrustGraph-DFL Section 5.1)
    dataset: str = "cifar10"  # cifar10 | synthetic
    data_dir: str = "data"
    download_data: bool = True
    n_nodes: int = 20
    samples_per_node: int = 500
    alpha_dir: float = 0.5
    val_frac: float = 0.1
    val_split: str = "local"  # local (paper) | iid (diagnostic)
    # Model and local training (TrustGraph-DFL Section 5.1)
    model_name: str = "cnn2x2"
    rounds: int = 200
    local_epochs: int = 1
    lr: float = 0.01
    lr_schedule: str = "constant"  # constant | cosine
    momentum: float = 0.9
    weight_decay: float = 1e-4
    batch_size: int = 64
    augment: bool = False
    grad_clip_norm: float = 0.0
    val_grad_weight_decay: bool = False
    byz_local_epochs: int = 0
    max_local_steps: int = 0
    # Topology
    topology: str = "random"  # ring | random | small_world
    er_p: float = 0.3
    ws_k: int = 4
    ws_p: float = 0.3
    # Byzantine setup
    byzantine_frac: float = 0.3
    attack: str = "label_flip"  # none | label_flip | label_flip_all | model_poison | random_noise
    poison_lambda: float = 10.0
    noise_sigma: float = 0.5
    # Aggregator
    aggregator: str = "trustgraph"  # fedavg | trustgraph | balance | krum | trimmed_mean
    alpha: float = 0.5
    val_grad_sign: str = "descent_neg"  # descent_neg | paper_pos
    trustgraph_use_relu: bool = True
    trustgraph_metric: str = "cosine"
    trustgraph_score_threshold: float = 0.0
    trustgraph_clip_norm: bool = False
    trustgraph_clip_mode: str = "update"
    trustgraph_clip_factor: float = 2.0
    trustgraph_delta_mode: str = "sender"
    trustgraph_score_scope: str = "global"
    trustgraph_ema_beta: float = 0.0
    balance_gamma: float = 0.3
    balance_kappa: float = 1.0
    krum_mode: str = "multi"
    trim_ratio: float = 0.3
    # Execution and logging
    seed: int = 42
    device: str = "auto"  # auto | cpu | cuda | mps
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
            trustgraph_score_scope=self.trustgraph_score_scope,
            trustgraph_ema_beta=self.trustgraph_ema_beta,
            balance_gamma=self.balance_gamma,
            balance_kappa=self.balance_kappa,
            krum_mode=self.krum_mode,
            trim_ratio=self.trim_ratio,
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


@dataclass
class _DeviceData:
    x_train: list[torch.Tensor]
    y_train: list[torch.Tensor]
    x_val: list[torch.Tensor]
    y_val: list[torch.Tensor]
    x_test: torch.Tensor
    y_test: torch.Tensor


@dataclass
class _AggregationMetrics:
    benign_zero: float
    benign_pairs: int
    byz_accept: float
    byz_pairs: int
    byz_weight_share: float
    self_fallback_rate: float
    benign_score_mean: float | None
    byz_score_mean: float | None


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


def _stage_device_data(
    data: PartitionedDataset,
    attack: str,
    byzantine_ids: set[int],
    device: torch.device,
) -> _DeviceData:
    """Move partitioned peer tensors to the target device and apply label flips if configured."""
    flip_mode = label_flip_mode_for_attack(attack)
    x_tr, y_tr, x_va, y_va = [], [], [], []
    for i, nd in enumerate(data.nodes):
        y_i = apply_label_flip(nd.y_train, mode=flip_mode) if (flip_mode and i in byzantine_ids) else nd.y_train
        x_tr.append(nd.x_train.to(device))
        y_tr.append(y_i.to(device))
        x_va.append(nd.x_val.to(device))
        y_va.append(nd.y_val.to(device))
    return _DeviceData(
        x_train=x_tr,
        y_train=y_tr,
        x_val=x_va,
        y_val=y_va,
        x_test=data.x_test.to(device),
        y_test=data.y_test.to(device),
    )


def _round_lr(cfg: ExperimentConfig, round_idx: int) -> float:
    if cfg.lr_schedule == "constant":
        return cfg.lr
    if cfg.lr_schedule == "cosine":
        lr_min = 0.05 * cfg.lr
        frac = round_idx / max(cfg.rounds - 1, 1)
        return lr_min + 0.5 * (cfg.lr - lr_min) * (1.0 + math.cos(math.pi * frac))
    raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule!r}")


def _local_training_phase(
    cfg: ExperimentConfig,
    prev_models: list[torch.Tensor],
    work_model: nn.Module,
    dev_data: _DeviceData,
    byzantine_ids: set[int],
    train_gen: torch.Generator,
    device: torch.device,
    round_idx: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[int, torch.Tensor]]:
    """Run Phase 1 local SGD and pre-update validation gradient computation."""
    post_models: list[torch.Tensor] = [torch.empty(0, device=device) for _ in range(cfg.n_nodes)]
    updates: list[torch.Tensor] = [torch.empty(0, device=device) for _ in range(cfg.n_nodes)]
    val_grads: dict[int, torch.Tensor] = {}
    lr_now = _round_lr(cfg, round_idx)

    for i in range(cfg.n_nodes):
        if i in byzantine_ids and cfg.attack in {"model_poison", "random_noise"}:
            continue
        vector_to_model(prev_models[i], work_model)
        if i not in byzantine_ids and cfg.aggregator == "trustgraph":
            # Algorithm 1 line 8: compute validation gradient at theta_i^t before local update.
            val_grads[i] = compute_validation_gradient(
                work_model,
                dev_data.x_val[i],
                dev_data.y_val[i],
                batch_size=cfg.batch_size,
                weight_decay=(cfg.weight_decay if cfg.val_grad_weight_decay else 0.0),
            )
        node_epochs = (
            cfg.byz_local_epochs
            if (i in byzantine_ids and cfg.byz_local_epochs > 0)
            else cfg.local_epochs
        )
        train_local_epoch(
            work_model,
            dev_data.x_train[i],
            dev_data.y_train[i],
            lr=lr_now,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            batch_size=cfg.batch_size,
            epochs=node_epochs,
            augment=cfg.augment,
            grad_clip_norm=cfg.grad_clip_norm,
            max_steps=cfg.max_local_steps,
            generator=train_gen,
        )
        post_vec = model_to_vector(work_model)
        post_models[i] = post_vec
        updates[i] = post_vec - prev_models[i]
    return post_models, updates, val_grads


def _aggregation_phase(
    cfg: ExperimentConfig,
    agg_cfg: AggregatorConfig,
    graph: nx.Graph,
    benign_ids: list[int],
    byzantine_ids: set[int],
    prev_models: list[torch.Tensor],
    post_models: list[torch.Tensor],
    updates: list[torch.Tensor],
    val_grads: dict[int, torch.Tensor],
    round_idx: int,
    byz_frac_eff: float,
    device: torch.device,
    param_slices: list[ParamSlice] | None = None,
    ema_scores_by_node: dict[int, dict[int, float]] | None = None,
) -> tuple[list[torch.Tensor], _AggregationMetrics]:
    """Run Phase 2/3 scoring and neighborhood aggregation across all benign peers."""
    next_models: list[torch.Tensor] = [vec.clone() for vec in prev_models]
    benign_zero = 0.0
    benign_pairs = 0
    byz_accept = 0.0
    byz_pairs = 0
    byz_weight_shares: list[float] = []
    benign_scores: list[float] = []
    byz_scores: list[float] = []
    self_fallbacks = 0

    for i in benign_ids:
        nbrs = list(graph.neighbors(i))
        nbr_posts = {j: post_models[j] for j in nbrs}
        nbr_updates = {j: updates[j] for j in nbrs}
        vg = val_grads.get(i, torch.empty(0, device=device))
        node_ema = ema_scores_by_node[i] if ema_scores_by_node is not None else None

        res = aggregate_node(
            self_id=i,
            self_post=post_models[i],
            val_grad=vg,
            neighbor_ids=nbrs,
            neighbor_posts=nbr_posts,
            neighbor_updates=nbr_updates,
            round_idx=round_idx,
            total_rounds=cfg.rounds,
            byzantine_frac=byz_frac_eff,
            cfg=agg_cfg,
            self_prev=prev_models[i],
            self_update=updates[i],
            param_slices=param_slices,
            ema_scores=node_ema,
        )
        if node_ema is not None and cfg.aggregator == "trustgraph" and cfg.trustgraph_ema_beta > 0.0:
            for j, sc in res.scores.items():
                node_ema[j] = sc
        next_models[i] = res.new_model
        if res.self_fallback:
            self_fallbacks += 1

        total_nbr_w = sum(res.neighbor_weights.values())
        byz_w = 0.0
        for j in nbrs:
            z_frac = res.neighbor_zero_weight_frac[j]
            if j in byzantine_ids:
                byz_pairs += 1
                byz_accept += 1.0 - z_frac
                byz_w += res.neighbor_weights[j]
                if cfg.aggregator == "trustgraph" and j in res.scores:
                    byz_scores.append(res.scores[j])
            else:
                benign_pairs += 1
                benign_zero += z_frac
                if cfg.aggregator == "trustgraph" and j in res.scores:
                    benign_scores.append(res.scores[j])
        byz_weight_shares.append((byz_w / total_nbr_w) if total_nbr_w > 0.0 else 0.0)

    if byzantine_ids:
        if cfg.attack in {"label_flip", "label_flip_all"}:
            for m in byzantine_ids:
                nbrs = list(graph.neighbors(m))
                next_models[m] = torch.stack(
                    [post_models[m]] + [post_models[j] for j in nbrs], dim=0
                ).mean(dim=0)
        else:
            benign_next_mean = torch.stack([next_models[i] for i in benign_ids], dim=0).mean(dim=0)
            for m in byzantine_ids:
                next_models[m] = benign_next_mean.clone()

    metrics = _AggregationMetrics(
        benign_zero=benign_zero,
        benign_pairs=benign_pairs,
        byz_accept=byz_accept,
        byz_pairs=byz_pairs,
        byz_weight_share=float(np.mean(byz_weight_shares)) if byz_weight_shares else 0.0,
        self_fallback_rate=self_fallbacks / max(len(benign_ids), 1),
        benign_score_mean=float(np.mean(benign_scores)) if benign_scores else None,
        byz_score_mean=float(np.mean(byz_scores)) if byz_scores else None,
    )
    return next_models, metrics


def _evaluate_benign_peers(
    node_models: list[torch.Tensor],
    benign_ids: list[int],
    work_model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    eval_batch_size: int,
) -> dict[str, float]:
    """Evaluate all benign peer models plus their consensus average on the clean test set."""
    accs: list[float] = []
    losses: list[float] = []
    for i in benign_ids:
        vector_to_model(node_models[i], work_model)
        acc, loss = evaluate_accuracy(work_model, x_test, y_test, batch_size=eval_batch_size)
        accs.append(acc)
        losses.append(loss)

    consensus_vec = torch.stack([node_models[i] for i in benign_ids], dim=0).mean(dim=0)
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


def _should_eval(round_idx: int, total_rounds: int, eval_every: int) -> bool:
    if eval_every <= 0:
        return round_idx == total_rounds - 1
    return (round_idx == 0) or ((round_idx + 1) % eval_every == 0) or (round_idx == total_rounds - 1)


def _has_nonfinite_models(node_models: list[torch.Tensor], benign_ids: list[int]) -> bool:
    return any(not bool(torch.isfinite(node_models[i]).all().item()) for i in benign_ids)


def run_experiment(
    cfg: ExperimentConfig,
    preloaded_data: PartitionedDataset | None = None,
) -> dict[str, object]:
    """Run one complete synchronous DFL experiment and return summary + history."""
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

    dev_data = _stage_device_data(data, cfg.attack, byzantine_ids, device)
    work_model = build_model(cfg.model_name, num_classes=data.num_classes).to(device)
    param_slices = model_param_slices(work_model)
    init_vec = model_to_vector(work_model)
    param_count = int(init_vec.numel())
    node_models: list[torch.Tensor] = [init_vec.clone() for _ in range(cfg.n_nodes)]

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
                    "resolved_device": str(device),
                    "param_count": param_count,
                    "topology": topo_info,
                },
                f,
                indent=2,
            )
        history_file = open(out_path / "history.jsonl", "w", encoding="utf-8")

    history: list[dict[str, object]] = []
    ema_scores_by_node: dict[int, dict[int, float]] = {i: {} for i in benign_ids}
    cum_benign_zero = 0.0
    cum_benign_pairs = 0
    cum_byz_accept = 0.0
    cum_byz_pairs = 0
    had_nan = False
    t_run_start = time.perf_counter()

    try:
        for r in range(cfg.rounds):
            t_round_start = time.perf_counter()
            prev_models = [vec.clone() for vec in node_models]

            post_models, updates, val_grads = _local_training_phase(
                cfg, prev_models, work_model, dev_data, byzantine_ids, train_gen, device, round_idx=r
            )
            apply_round_attack(
                attack_cfg=atk_cfg,
                byzantine_ids=byzantine_ids,
                benign_ids=benign_ids,
                prev_models=prev_models,
                post_models=post_models,
                updates=updates,
                generator=attack_gen,
            )
            node_models, agg_m = _aggregation_phase(
                cfg=cfg,
                agg_cfg=agg_cfg,
                graph=graph,
                benign_ids=benign_ids,
                byzantine_ids=byzantine_ids,
                prev_models=prev_models,
                post_models=post_models,
                updates=updates,
                val_grads=val_grads,
                round_idx=r,
                byz_frac_eff=byz_frac_eff,
                device=device,
                param_slices=param_slices,
                ema_scores_by_node=ema_scores_by_node,
            )
            if not had_nan and _has_nonfinite_models(node_models, benign_ids):
                had_nan = True

            cum_benign_zero += agg_m.benign_zero
            cum_benign_pairs += agg_m.benign_pairs
            cum_byz_accept += agg_m.byz_accept
            cum_byz_pairs += agg_m.byz_pairs

            record: dict[str, object] = {
                "round": r,
                "fpr_round": (agg_m.benign_zero / agg_m.benign_pairs) if agg_m.benign_pairs > 0 else 0.0,
                "fpr_cumulative": (cum_benign_zero / cum_benign_pairs) if cum_benign_pairs > 0 else 0.0,
                "byz_accept_rate": (agg_m.byz_accept / agg_m.byz_pairs) if agg_m.byz_pairs > 0 else 0.0,
                "byz_weight_share": agg_m.byz_weight_share,
                "self_fallback_rate": agg_m.self_fallback_rate,
                "benign_score_mean": agg_m.benign_score_mean,
                "byz_score_mean": agg_m.byz_score_mean,
            }

            do_eval = _should_eval(r, cfg.rounds, cfg.eval_every)
            if do_eval:
                record.update(
                    _evaluate_benign_peers(
                        node_models,
                        benign_ids,
                        work_model,
                        dev_data.x_test,
                        dev_data.y_test,
                        cfg.eval_batch_size,
                    )
                )

            record["round_wall_sec"] = time.perf_counter() - t_round_start
            history.append(record)
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

    eval_records = [rec for rec in history if "benign_test_acc_mean" in rec]
    final_eval = eval_records[-1]
    best_eval = max(eval_records, key=lambda rec: float(rec["benign_test_acc_mean"]))

    ben_scores = [float(r["benign_score_mean"]) for r in history if r["benign_score_mean"] is not None]
    byz_scores = [float(r["byz_score_mean"]) for r in history if r["byz_score_mean"] is not None]
    mean_ben = float(np.mean(ben_scores)) if ben_scores else None
    mean_byz = float(np.mean(byz_scores)) if byz_scores else None
    if mean_ben is not None and math.isnan(mean_ben):
        had_nan = True

    summary: dict[str, object] = {
        "config": asdict(cfg),
        "resolved_device": str(device),
        "param_count": param_count,
        "topology": topo_info,
        "total_wall_sec": time.perf_counter() - t_run_start,
        "had_nan": had_nan,
        "final_benign_test_acc_mean": final_eval["benign_test_acc_mean"],
        "final_benign_test_acc_std": final_eval["benign_test_acc_std"],
        "final_consensus_test_acc": final_eval["consensus_test_acc"],
        "best_benign_test_acc_mean": best_eval["benign_test_acc_mean"],
        "best_round": best_eval["round"],
        "fpr_cumulative": (cum_benign_zero / cum_benign_pairs) if cum_benign_pairs > 0 else 0.0,
        "fpr_final_round": history[-1]["fpr_round"],
        "byz_accept_cumulative": (cum_byz_accept / cum_byz_pairs) if cum_byz_pairs > 0 else 0.0,
        "mean_benign_score": mean_ben,
        "mean_byz_score": mean_byz,
    }

    if out_path is not None:
        with open(out_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return {"summary": summary, "history": history}
