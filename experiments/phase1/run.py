"""CLI entry point for Phase 1 TrustGraph-DFL experiments (local, suite, distributed, transfer)."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from experiments.phase1.suites import SUITES, build_suite_runs
from src.distributed import run_distributed_experiment
from src.simulator import ExperimentConfig, prepare_dataset, run_experiment
from src.transport import run_transfer_benchmark

__all__ = ["SUITES", "build_suite_runs", "parse_args", "main"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run local, sharded suite, distributed, or TCP transfer experiments.")
    p.add_argument(
        "--mode",
        choices=["single", "suite", "distributed", "transfer_bench"],
        default="single",
    )
    p.add_argument("--suite", choices=sorted(SUITES), default="smoke")
    p.add_argument("--dataset", choices=["cifar10", "synthetic"], default="cifar10")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--n-nodes", type=int, default=20)
    p.add_argument("--samples-per-node", type=int, default=500)
    p.add_argument("--alpha-dir", type=float, default=0.5)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--val-split", choices=["local", "iid"], default="local")
    p.add_argument(
        "--model",
        default="cnn2x2",
        choices=["cnn2x2", "cnn2x2_gn", "resnet18_pretrained", "resnet18_scratch"],
    )
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--grad-clip-norm", type=float, default=0.0)
    p.add_argument("--val-grad-weight-decay", action="store_true")
    p.add_argument("--byz-local-epochs", type=int, default=0)
    p.add_argument("--topology", choices=["ring", "random", "small_world"], default="random")
    p.add_argument("--er-p", type=float, default=0.3)
    p.add_argument("--ws-k", type=int, default=4)
    p.add_argument("--ws-p", type=float, default=0.3)
    p.add_argument("--byzantine-frac", type=float, default=0.3)
    p.add_argument(
        "--attack",
        choices=["none", "label_flip", "label_flip_all", "model_poison", "random_noise"],
        default="label_flip",
    )
    p.add_argument("--poison-lambda", type=float, default=10.0)
    p.add_argument("--noise-sigma", type=float, default=0.5)
    p.add_argument("--aggregator", choices=["trustgraph"], default="trustgraph")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--val-grad-sign", choices=["descent_neg", "paper_pos"], default="descent_neg")
    p.add_argument("--no-relu", action="store_true")
    p.add_argument("--trustgraph-metric", choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--trustgraph-score-threshold", type=float, default=0.0)
    p.add_argument("--trustgraph-clip-norm", action="store_true")
    p.add_argument("--trustgraph-clip-mode", choices=["update", "model"], default="update")
    p.add_argument("--trustgraph-clip-factor", type=float, default=2.0)
    p.add_argument("--trustgraph-delta-mode", choices=["sender", "receiver"], default="sender")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--skip-existing", action="store_true", help="Skip runs that already have summary.json.")
    p.add_argument("--output-root", default="outputs/phase1")
    p.add_argument("--quiet", action="store_true")
    # Distributed and TCP transport options
    p.add_argument("--dist-iface", default=None, help="Network interface for distributed TCP transport (default loopback).")
    p.add_argument("--dist-port", type=int, default=29600, help="Base TCP port for distributed peer sockets.")
    p.add_argument("--bench-role", choices=["auto", "server", "client"], default="auto")
    p.add_argument("--bench-host", default="127.0.0.1")
    p.add_argument("--bench-port", type=int, default=29501)
    p.add_argument("--bench-payload-bytes", type=int, default=11181642 * 4)
    p.add_argument("--bench-repeats", type=int, default=25)
    p.add_argument("--bench-burst-repeats", type=int, default=10)
    p.add_argument("--bench-out", default=None)
    return p.parse_args()


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        download_data=not args.no_download,
        n_nodes=args.n_nodes,
        samples_per_node=args.samples_per_node,
        alpha_dir=args.alpha_dir,
        val_frac=args.val_frac,
        val_split=args.val_split,
        model_name=args.model,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        augment=args.augment,
        grad_clip_norm=args.grad_clip_norm,
        val_grad_weight_decay=args.val_grad_weight_decay,
        byz_local_epochs=args.byz_local_epochs,
        topology=args.topology,
        er_p=args.er_p,
        ws_k=args.ws_k,
        ws_p=args.ws_p,
        byzantine_frac=args.byzantine_frac,
        attack=args.attack,
        poison_lambda=args.poison_lambda,
        noise_sigma=args.noise_sigma,
        aggregator=args.aggregator,
        alpha=args.alpha,
        val_grad_sign=args.val_grad_sign,
        trustgraph_use_relu=not args.no_relu,
        trustgraph_metric=args.trustgraph_metric,
        trustgraph_score_threshold=args.trustgraph_score_threshold,
        trustgraph_clip_norm=args.trustgraph_clip_norm,
        trustgraph_clip_mode=args.trustgraph_clip_mode,
        trustgraph_clip_factor=args.trustgraph_clip_factor,
        trustgraph_delta_mode=args.trustgraph_delta_mode,
        device=args.device,
        eval_every=args.eval_every,
        verbose=not args.quiet,
    )


def _run_tag(args: argparse.Namespace) -> str:
    delta_suffix = "_rxdelta" if args.trustgraph_delta_mode == "receiver" else ""
    tag = (
        f"{args.dataset}_{args.model}_{args.topology}_{args.attack}_byz{int(round(args.byzantine_frac * 100))}"
        f"_{args.aggregator}_{args.val_grad_sign}{delta_suffix}"
    )
    return tag


def _run_single(args: argparse.Namespace, base_cfg: ExperimentConfig, out_root: Path) -> None:
    tag = _run_tag(args)
    for seed in args.seeds:
        out_dir = out_root / "single" / tag / f"seed_{seed}"
        if args.skip_existing and (out_dir / "summary.json").exists():
            print(f"=== Skipping completed {tag} (seed={seed}) ===", flush=True)
            continue
        cfg = replace(base_cfg, seed=seed, output_dir=str(out_dir))
        print(f"=== Running {tag} (seed={seed}) -> {out_dir} ===", flush=True)
        res = run_experiment(cfg)
        s = res["summary"]
        print(
            f"DONE {tag} seed={seed}: final_acc={100.0 * float(s['final_benign_test_acc_mean']):.2f}% "
            f"best_acc={100.0 * float(s['best_benign_test_acc_mean']):.2f}% "
            f"fpr_cum={100.0 * float(s['fpr_cumulative']):.2f}% "
            f"wall={float(s['total_wall_sec']):.1f}s",
            flush=True,
        )


def _run_suite(args: argparse.Namespace, base_cfg: ExperimentConfig, out_root: Path) -> None:
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"Invalid shard {args.shard_index}/{args.num_shards}")

    suite_root = out_root / args.suite
    suite_root.mkdir(parents=True, exist_ok=True)

    pending_jobs: list[tuple[int, str, ExperimentConfig]] = []
    skipped = 0
    for seed in args.seeds:
        seed_base = replace(base_cfg, seed=seed)
        for run_name, cfg in build_suite_runs(args.suite, seed_base):
            out_dir = suite_root / run_name / f"seed_{seed}"
            if args.skip_existing and (out_dir / "summary.json").exists():
                skipped += 1
                continue
            pending_jobs.append((seed, run_name, cfg))

    shard_jobs = [job for idx, job in enumerate(pending_jobs) if idx % args.num_shards == args.shard_index]
    print(
        f"=== Suite '{args.suite}': shard {args.shard_index}/{args.num_shards} "
        f"running {len(shard_jobs)}/{len(pending_jobs)} pending jobs (skipped {skipped} completed) ===",
        flush=True,
    )

    summaries: list[dict[str, object]] = []
    data_cache: dict[tuple[int, str, float, int, int, float], object] = {}
    for job_idx, (seed, run_name, cfg) in enumerate(shard_jobs, start=1):
        cache_key = (seed, cfg.val_split, cfg.alpha_dir, cfg.n_nodes, cfg.samples_per_node, cfg.val_frac)
        if cache_key not in data_cache:
            data_cache[cache_key] = prepare_dataset(cfg)
        out_dir = suite_root / run_name / f"seed_{seed}"
        run_cfg = replace(cfg, output_dir=str(out_dir))
        print(
            f"\n=== [{args.suite} shard {args.shard_index} ({job_idx}/{len(shard_jobs)})] "
            f"{run_name} (seed={seed}) ===",
            flush=True,
        )
        res = run_experiment(run_cfg, preloaded_data=data_cache[cache_key])
        s = dict(res["summary"])
        s["run_name"] = run_name
        summaries.append(s)

    summary_name = (
        "suite_summaries.json"
        if args.num_shards == 1
        else f"suite_summaries_shard{args.shard_index}.json"
    )
    with open(suite_root / summary_name, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(f"\n=== Suite '{args.suite}' shard {args.shard_index}/{args.num_shards} summary ===", flush=True)
    print(
        f"{'Run':44s} {'Seed':>5s} {'FinalAcc':>9s} {'ConsAcc':>9s} "
        f"{'FPR(cum)':>9s} {'ByzAcc':>8s} {'CosBen':>8s} {'CosByz':>8s}",
        flush=True,
    )
    for s in summaries:
        cb = f"{float(s['mean_benign_score']):+.3f}" if s["mean_benign_score"] is not None else "   n/a"
        cm = f"{float(s['mean_byz_score']):+.3f}" if s["mean_byz_score"] is not None else "   n/a"
        print(
            f"{str(s['run_name']):44s} {int(s['config']['seed']):5d} "
            f"{100.0 * float(s['final_benign_test_acc_mean']):8.2f}% "
            f"{100.0 * float(s['final_consensus_test_acc']):8.2f}% "
            f"{100.0 * float(s['fpr_cumulative']):8.2f}% "
            f"{100.0 * float(s['byz_accept_cumulative']):7.2f}% "
            f"{cb:>8s} {cm:>8s}",
            flush=True,
        )


def _run_distributed(args: argparse.Namespace, base_cfg: ExperimentConfig, out_root: Path) -> None:
    tag = _run_tag(args)
    seed = args.seeds[0]
    out_dir = out_root / "distributed" / tag / f"seed_{seed}"
    cfg = replace(base_cfg, seed=seed, output_dir=str(out_dir))
    res = run_distributed_experiment(cfg, iface=args.dist_iface, base_port=args.dist_port)
    if res is not None:
        s = res["summary"]
        print(
            f"DONE distributed {tag} seed={seed}: final_acc={100.0 * float(s['final_benign_test_acc_mean']):.2f}% "
            f"cons_acc={100.0 * float(s['final_consensus_test_acc']):.2f}% "
            f"fpr_cum={100.0 * float(s['fpr_cumulative']):.2f}% "
            f"wall={float(s['total_wall_sec']):.1f}s",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if args.mode == "transfer_bench":
        run_transfer_benchmark(
            role=args.bench_role,
            host=args.bench_host,
            port=args.bench_port,
            payload_bytes=args.bench_payload_bytes,
            single_repeats=args.bench_repeats,
            burst_repeats=args.bench_burst_repeats,
            out_path=args.bench_out,
        )
        return
    base_cfg = _config_from_args(args)
    out_root = Path(args.output_root)
    if args.mode == "single":
        _run_single(args, base_cfg, out_root)
    elif args.mode == "distributed":
        _run_distributed(args, base_cfg, out_root)
    else:
        _run_suite(args, base_cfg, out_root)


if __name__ == "__main__":
    main()
