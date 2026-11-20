"""Synchronous decentralized FL simulator for Phase 1 experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

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
    build_model,
    compute_validation_gradient,
    evaluate_accuracy,
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
    momentum: float = 0.9
    weight_decay: float = 1e-4
    batch_size: int = 64
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

    # Move partitioned tensors onto target device (10k CIFAR-10 images ~122 MB in FP32).
    flip_mode = label_flip_mode_for_attack(cfg.attack)
    node_x_tr: list[torch.Tensor] = []
    node_y_tr: list[torch.Tensor] = []
    node_x_val: list[torch.Tensor] = []
    node_y_val: list[torch.Tensor] = []
    for i in range(cfg.n_nodes):
        nd = data.nodes[i]
        y_tr_i = apply_label_flip(nd.y_train, mode=flip_mode) if (flip_mode and i in byzantine_ids) else nd.y_train
        node_x_tr.append(nd.x_train.to(device))
        node_y_tr.append(y_tr_i.to(device))
        node_x_val.append(nd.x_val.to(device))
        node_y_val.append(nd.y_val.to(device))
    x_test = data.x_test.to(device)
    y_test = data.y_test.to(device)

    # Shared scratch model + per-node parameter vectors (all initialized to same theta^0).
    work_model = build_model(cfg.model_name, num_classes=data.num_classes).to(device)
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
    cum_benign_zero = 0.0
    cum_benign_pairs = 0
    cum_byz_accept = 0.0
    cum_byz_pairs = 0
    t_run_start = time.perf_counter()

    try:
        for r in range(cfg.rounds):
            t_round_start = time.perf_counter()
            prev_models = [vec.clone() for vec in node_models]
            post_models: list[torch.Tensor] = [torch.empty(0, device=device) for _ in range(cfg.n_nodes)]
            updates: list[torch.Tensor] = [torch.empty(0, device=device) for _ in range(cfg.n_nodes)]
            val_grads: dict[int, torch.Tensor] = {}

            # Phase 1: Local training (benign peers, plus data-poisoning Byzantine peers).
            for i in range(cfg.n_nodes):
                if i in byzantine_ids and cfg.attack in {"model_poison", "random_noise"}:
                    continue
                vector_to_model(prev_models[i], work_model)
                if i not in byzantine_ids and cfg.aggregator == "trustgraph":
                    # Algorithm 1 line 8: compute validation gradient at theta_i^t before update.
                    val_grads[i] = compute_validation_gradient(
                        work_model, node_x_val[i], node_y_val[i], batch_size=cfg.batch_size
                    )
                train_local_epoch(
                    work_model,
                    node_x_tr[i],
                    node_y_tr[i],
                    lr=cfg.lr,
                    momentum=cfg.momentum,
                    weight_decay=cfg.weight_decay,
                    batch_size=cfg.batch_size,
                    epochs=cfg.local_epochs,
                    generator=train_gen,
                )
                post_vec = model_to_vector(work_model)
                post_models[i] = post_vec
                updates[i] = post_vec - prev_models[i]

            # Craft round-level Byzantine payloads (model_poison / random_noise).
            apply_round_attack(
                attack_cfg=atk_cfg,
                byzantine_ids=byzantine_ids,
                benign_ids=benign_ids,
                prev_models=prev_models,
                post_models=post_models,
                updates=updates,
                generator=attack_gen,
            )

            # Phase 2 & 3: Consistency scoring and neighborhood aggregation at each benign peer.
            next_models: list[torch.Tensor] = [vec.clone() for vec in prev_models]
            round_benign_zero = 0.0
            round_benign_pairs = 0
            round_byz_accept = 0.0
            round_byz_pairs = 0
            byz_weight_shares: list[float] = []
            benign_cosines: list[float] = []
            byz_cosines: list[float] = []
            self_fallbacks = 0

            for i in benign_ids:
                nbrs = list(graph.neighbors(i))
                nbr_posts = {j: post_models[j] for j in nbrs}
                nbr_updates = {j: updates[j] for j in nbrs}
                vg = val_grads.get(i, torch.empty(0, device=device))

                res = aggregate_node(
                    self_id=i,
                    self_post=post_models[i],
                    val_grad=vg,
                    neighbor_ids=nbrs,
                    neighbor_posts=nbr_posts,
                    neighbor_updates=nbr_updates,
                    round_idx=r,
                    total_rounds=cfg.rounds,
                    byzantine_frac=byz_frac_eff,
                    cfg=agg_cfg,
                    self_prev=prev_models[i],
                    self_update=updates[i],
                )
                next_models[i] = res.new_model
                if res.self_fallback:
                    self_fallbacks += 1

                total_nbr_w = sum(res.neighbor_weights.values())
                byz_w = 0.0
                for j in nbrs:
                    z_frac = res.neighbor_zero_weight_frac[j]
                    if j in byzantine_ids:
                        round_byz_pairs += 1
                        round_byz_accept += 1.0 - z_frac
                        byz_w += res.neighbor_weights[j]
                        if cfg.aggregator == "trustgraph" and j in res.scores:
                            byz_cosines.append(res.scores[j])
                    else:
                        round_benign_pairs += 1
                        round_benign_zero += z_frac
                        if cfg.aggregator == "trustgraph" and j in res.scores:
                            benign_cosines.append(res.scores[j])
                if total_nbr_w > 0.0:
                    byz_weight_shares.append(byz_w / total_nbr_w)
                else:
                    byz_weight_shares.append(0.0)

            # Update Byzantine nodes' local model state for data-poisoning attacks (or sync benign mean).
            if byzantine_ids:
                if cfg.attack in {"label_flip", "label_flip_all"}:
                    for m in byzantine_ids:
                        nbrs = list(graph.neighbors(m))
                        nbr_posts = {j: post_models[j] for j in nbrs}
                        # Data-poisoning peers stay synced via simple neighborhood average.
                        next_models[m] = torch.stack(
                            [post_models[m]] + [nbr_posts[j] for j in nbrs], dim=0
                        ).mean(dim=0)
                else:
                    benign_next_mean = torch.stack([next_models[i] for i in benign_ids], dim=0).mean(dim=0)
                    for m in byzantine_ids:
                        next_models[m] = benign_next_mean.clone()

            node_models = next_models

            cum_benign_zero += round_benign_zero
            cum_benign_pairs += round_benign_pairs
            cum_byz_accept += round_byz_accept
            cum_byz_pairs += round_byz_pairs

            record: dict[str, object] = {
                "round": r,
                "fpr_round": (round_benign_zero / round_benign_pairs) if round_benign_pairs > 0 else 0.0,
                "fpr_cumulative": (cum_benign_zero / cum_benign_pairs) if cum_benign_pairs > 0 else 0.0,
                "byz_accept_rate": (round_byz_accept / round_byz_pairs) if round_byz_pairs > 0 else 0.0,
                "byz_weight_share": float(np.mean(byz_weight_shares)) if byz_weight_shares else 0.0,
                "self_fallback_rate": self_fallbacks / max(len(benign_ids), 1),
                "benign_score_mean": float(np.mean(benign_cosines)) if benign_cosines else None,
                "byz_score_mean": float(np.mean(byz_cosines)) if byz_cosines else None,
            }

            if cfg.eval_every <= 0:
                should_eval = r == cfg.rounds - 1
            else:
                should_eval = (r == 0) or ((r + 1) % cfg.eval_every == 0) or (r == cfg.rounds - 1)
            if should_eval:
                accs: list[float] = []
                losses: list[float] = []
                for i in benign_ids:
                    vector_to_model(node_models[i], work_model)
                    acc, loss = evaluate_accuracy(work_model, x_test, y_test, batch_size=cfg.eval_batch_size)
                    accs.append(acc)
                    losses.append(loss)
                # Consensus benign model accuracy (average of benign models).
                consensus_vec = torch.stack([node_models[i] for i in benign_ids], dim=0).mean(dim=0)
                vector_to_model(consensus_vec, work_model)
                cons_acc, cons_loss = evaluate_accuracy(
                    work_model, x_test, y_test, batch_size=cfg.eval_batch_size
                )
                record.update(
                    {
                        "benign_test_acc_mean": float(np.mean(accs)),
                        "benign_test_acc_std": float(np.std(accs)),
                        "benign_test_acc_min": float(np.min(accs)),
                        "benign_test_acc_max": float(np.max(accs)),
                        "benign_test_loss_mean": float(np.mean(losses)),
                        "consensus_test_acc": float(cons_acc),
                        "consensus_test_loss": float(cons_loss),
                    }
                )

            record["round_wall_sec"] = time.perf_counter() - t_round_start
            history.append(record)
            if history_file is not None:
                history_file.write(json.dumps(record) + "\n")
                history_file.flush()

            if cfg.verbose and should_eval:
                score_str = ""
                if record["benign_score_mean"] is not None:
                    b_s = record["benign_score_mean"]
                    m_s = record["byz_score_mean"]
                    m_txt = f"{m_s:+.3f}" if m_s is not None else "n/a"
                    score_str = f" cos(ben={b_s:+.3f},byz={m_txt})"
                print(
                    f"[r={r + 1:3d}/{cfg.rounds}] "
                    f"acc={100.0 * record['benign_test_acc_mean']:5.2f}% "
                    f"(cons={100.0 * record['consensus_test_acc']:5.2f}%) "
                    f"fpr={100.0 * record['fpr_round']:5.1f}% "
                    f"byz_acc={100.0 * record['byz_accept_rate']:5.1f}%"
                    f"{score_str} "
                    f"({record['round_wall_sec']:.2f}s)"
                )
    finally:
        if history_file is not None:
            history_file.close()

    eval_records = [rec for rec in history if "benign_test_acc_mean" in rec]
    final_eval = eval_records[-1]
    best_eval = max(eval_records, key=lambda rec: float(rec["benign_test_acc_mean"]))

    summary: dict[str, object] = {
        "config": asdict(cfg),
        "resolved_device": str(device),
        "param_count": param_count,
        "topology": topo_info,
        "total_wall_sec": time.perf_counter() - t_run_start,
        "final_benign_test_acc_mean": final_eval["benign_test_acc_mean"],
        "final_benign_test_acc_std": final_eval["benign_test_acc_std"],
        "final_consensus_test_acc": final_eval["consensus_test_acc"],
        "best_benign_test_acc_mean": best_eval["benign_test_acc_mean"],
        "best_round": best_eval["round"],
        "fpr_cumulative": (cum_benign_zero / cum_benign_pairs) if cum_benign_pairs > 0 else 0.0,
        "fpr_final_round": history[-1]["fpr_round"],
        "byz_accept_cumulative": (cum_byz_accept / cum_byz_pairs) if cum_byz_pairs > 0 else 0.0,
        "mean_benign_score": (
            float(np.mean([r["benign_score_mean"] for r in history if r["benign_score_mean"] is not None]))
            if any(r["benign_score_mean"] is not None for r in history)
            else None
        ),
        "mean_byz_score": (
            float(np.mean([r["byz_score_mean"] for r in history if r["byz_score_mean"] is not None]))
            if any(r["byz_score_mean"] is not None for r in history)
            else None
        ),
    }

    if out_path is not None:
        with open(out_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return {"summary": summary, "history": history}
