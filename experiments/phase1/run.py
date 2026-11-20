"""CLI runner for Phase 1 (TrustGraph-DFL CIFAR-10 reproduction)."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from src.simulator import ExperimentConfig, prepare_dataset, run_experiment


def _suite_runs(suite: str, base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Build named experiment configs for a Phase 1 suite."""
    runs: list[tuple[str, ExperimentConfig]] = []

    if suite == "smoke":
        smoke_base = replace(
            base,
            topology="random",
            byzantine_frac=0.3,
            eval_every=base.eval_every if base.eval_every <= 0 else min(base.eval_every, base.rounds),
        )
        for atk in ("label_flip", "model_poison"):
            for tag, overrides in [
                ("fedavg", {"aggregator": "fedavg"}),
                ("trimmed_mean", {"aggregator": "trimmed_mean"}),
                ("krum", {"aggregator": "krum"}),
                ("balance", {"aggregator": "balance"}),
                ("tg_descent_neg", {"aggregator": "trustgraph", "val_grad_sign": "descent_neg"}),
                ("tg_paper_pos", {"aggregator": "trustgraph", "val_grad_sign": "paper_pos"}),
            ]:
                runs.append((f"{atk}__{tag}", replace(smoke_base, attack=atk, **overrides)))
        return runs

    if suite == "sign_check":
        for atk in ("none", "label_flip", "model_poison", "random_noise"):
            for sign in ("descent_neg", "paper_pos"):
                runs.append(
                    (
                        f"{atk}__tg_{sign}",
                        replace(
                            base,
                            topology="random",
                            byzantine_frac=0.3,
                            attack=atk,
                            aggregator="trustgraph",
                            val_grad_sign=sign,
                        ),
                    )
                )
        return runs

    if suite == "main_attacks":
        methods = [
            ("fedavg", {"aggregator": "fedavg"}),
            ("trimmed_mean", {"aggregator": "trimmed_mean"}),
            ("krum", {"aggregator": "krum"}),
            ("balance", {"aggregator": "balance"}),
            ("tg_descent_neg", {"aggregator": "trustgraph", "val_grad_sign": "descent_neg"}),
            ("tg_paper_pos", {"aggregator": "trustgraph", "val_grad_sign": "paper_pos"}),
        ]
        for atk in ("none", "label_flip", "model_poison", "random_noise"):
            for tag, overrides in methods:
                runs.append(
                    (
                        f"{atk}__{tag}",
                        replace(base, topology="random", byzantine_frac=0.3, attack=atk, **overrides),
                    )
                )
        return runs

    if suite == "topologies":
        methods = [
            ("fedavg", {"aggregator": "fedavg"}),
            ("trimmed_mean", {"aggregator": "trimmed_mean"}),
            ("krum", {"aggregator": "krum"}),
            ("balance", {"aggregator": "balance"}),
            ("tg_descent_neg", {"aggregator": "trustgraph", "val_grad_sign": "descent_neg"}),
            ("tg_paper_pos", {"aggregator": "trustgraph", "val_grad_sign": "paper_pos"}),
        ]
        for topo in ("ring", "random", "small_world"):
            for tag, overrides in methods:
                runs.append(
                    (
                        f"{topo}__{tag}",
                        replace(base, topology=topo, byzantine_frac=0.3, attack=base.attack, **overrides),
                    )
                )
        return runs

    if suite == "fpr_sweep":
        methods = [
            ("tg_descent_neg", {"aggregator": "trustgraph", "val_grad_sign": "descent_neg"}),
            ("tg_paper_pos", {"aggregator": "trustgraph", "val_grad_sign": "paper_pos"}),
            ("balance", {"aggregator": "balance"}),
            ("krum", {"aggregator": "krum"}),
            ("trimmed_mean", {"aggregator": "trimmed_mean"}),
        ]
        for frac in (0.1, 0.2, 0.3, 0.4, 0.5):
            pct = int(round(frac * 100))
            for tag, overrides in methods:
                runs.append(
                    (
                        f"byz{pct}__{tag}",
                        replace(
                            base,
                            topology="random",
                            byzantine_frac=frac,
                            attack=base.attack,
                            **overrides,
                        ),
                    )
                )
        return runs

    if suite == "ablations":
        abl_base = replace(base, topology="random", byzantine_frac=0.3, attack="model_poison", aggregator="trustgraph")
        runs.extend(
            [
                ("full_descent_neg_a05", replace(abl_base, val_grad_sign="descent_neg", alpha=0.5)),
                ("no_relu_descent_neg", replace(abl_base, val_grad_sign="descent_neg", trustgraph_use_relu=False)),
                ("euclidean", replace(abl_base, val_grad_sign="descent_neg", trustgraph_metric="euclidean")),
                ("alpha_03", replace(abl_base, val_grad_sign="descent_neg", alpha=0.3)),
                ("alpha_07", replace(abl_base, val_grad_sign="descent_neg", alpha=0.7)),
            ]
        )
        return runs

    if suite == "step_a":
        methods_all = [
            ("fedavg", {"aggregator": "fedavg"}),
            ("trimmed_mean", {"aggregator": "trimmed_mean"}),
            ("krum", {"aggregator": "krum"}),
            ("balance", {"aggregator": "balance"}),
            ("tg_descent_neg", {"aggregator": "trustgraph", "val_grad_sign": "descent_neg"}),
            ("tg_paper_pos", {"aggregator": "trustgraph", "val_grad_sign": "paper_pos"}),
        ]
        # 1) main_attacks (24 runs): random ER(0.3), 30% Byzantine, 4 attacks x 6 methods
        for atk in ("none", "label_flip", "model_poison", "random_noise"):
            for tag, overrides in methods_all:
                runs.append(
                    (
                        f"main_attacks/{atk}__{tag}",
                        replace(base, topology="random", byzantine_frac=0.3, attack=atk, **overrides),
                    )
                )
        # 2) topologies (12 runs): ring & small_world, label_flip, 30% Byzantine x 6 methods
        for topo in ("ring", "small_world"):
            for tag, overrides in methods_all:
                runs.append(
                    (
                        f"topologies/{topo}__label_flip__{tag}",
                        replace(base, topology=topo, byzantine_frac=0.3, attack="label_flip", **overrides),
                    )
                )
        # 3) fpr_sweep (25 runs): random, label_flip, 10%-50% Byzantine x 5 defenses
        methods_fpr = [
            ("tg_descent_neg", {"aggregator": "trustgraph", "val_grad_sign": "descent_neg"}),
            ("tg_paper_pos", {"aggregator": "trustgraph", "val_grad_sign": "paper_pos"}),
            ("balance", {"aggregator": "balance"}),
            ("krum", {"aggregator": "krum"}),
            ("trimmed_mean", {"aggregator": "trimmed_mean"}),
        ]
        for frac in (0.1, 0.2, 0.3, 0.4, 0.5):
            pct = int(round(frac * 100))
            for tag, overrides in methods_fpr:
                runs.append(
                    (
                        f"fpr_sweep/byz{pct}__{tag}",
                        replace(
                            base,
                            topology="random",
                            byzantine_frac=frac,
                            attack="label_flip",
                            **overrides,
                        ),
                    )
                )
        # 4) ablations & diagnostics (12 runs)
        tg_mp = replace(
            base,
            topology="random",
            byzantine_frac=0.3,
            attack="model_poison",
            aggregator="trustgraph",
            val_grad_sign="descent_neg",
        )
        runs.extend(
            [
                ("ablations/no_relu__model_poison", replace(tg_mp, trustgraph_use_relu=False)),
                ("ablations/euclidean__model_poison", replace(tg_mp, trustgraph_metric="euclidean")),
                ("ablations/alpha03__model_poison", replace(tg_mp, alpha=0.3)),
                ("ablations/alpha07__model_poison", replace(tg_mp, alpha=0.7)),
                (
                    "diagnostics/valiid__label_flip__tg_descent_neg",
                    replace(tg_mp, attack="label_flip", val_split="iid"),
                ),
                (
                    "diagnostics/valiid__model_poison__tg_descent_neg",
                    replace(tg_mp, attack="model_poison", val_split="iid"),
                ),
                (
                    "diagnostics/bal_g015__label_flip",
                    replace(
                        base,
                        topology="random",
                        byzantine_frac=0.3,
                        attack="label_flip",
                        aggregator="balance",
                        balance_gamma=0.15,
                    ),
                ),
                (
                    "diagnostics/bal_g015__model_poison",
                    replace(
                        base,
                        topology="random",
                        byzantine_frac=0.3,
                        attack="model_poison",
                        aggregator="balance",
                        balance_gamma=0.15,
                    ),
                ),
                (
                    "diagnostics/bal_g015__random_noise",
                    replace(
                        base,
                        topology="random",
                        byzantine_frac=0.3,
                        attack="random_noise",
                        aggregator="balance",
                        balance_gamma=0.15,
                    ),
                ),
                (
                    "diagnostics/label_flip_all__fedavg",
                    replace(base, topology="random", byzantine_frac=0.3, attack="label_flip_all", aggregator="fedavg"),
                ),
                (
                    "diagnostics/label_flip_all__balance_g015",
                    replace(
                        base,
                        topology="random",
                        byzantine_frac=0.3,
                        attack="label_flip_all",
                        aggregator="balance",
                        balance_gamma=0.15,
                    ),
                ),
                (
                    "diagnostics/label_flip_all__tg_descent_neg",
                    replace(tg_mp, attack="label_flip_all"),
                ),
                (
                    "diagnostics/clip_norm__model_poison__tg_descent_neg",
                    replace(tg_mp, attack="model_poison", trustgraph_clip_norm=True),
                ),
                (
                    "diagnostics/clip_norm__random_noise__tg_descent_neg",
                    replace(tg_mp, attack="random_noise", trustgraph_clip_norm=True),
                ),
            ]
        )
        return runs

    raise ValueError(f"Unknown suite: {suite!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1: TrustGraph-DFL reproduction runner.")
    p.add_argument("--mode", choices=["single", "suite"], default="single")
    p.add_argument(
        "--suite",
        choices=["smoke", "sign_check", "main_attacks", "topologies", "fpr_sweep", "ablations", "step_a"],
        default="smoke",
    )
    p.add_argument("--dataset", choices=["cifar10", "synthetic"], default="cifar10")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--n-nodes", type=int, default=20)
    p.add_argument("--samples-per-node", type=int, default=500)
    p.add_argument("--alpha-dir", type=float, default=0.5)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--val-split", choices=["local", "iid"], default="local")
    p.add_argument("--model", default="cnn2x2", choices=["cnn2x2", "lenet2x2"])
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
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
    p.add_argument(
        "--aggregator",
        choices=["fedavg", "trustgraph", "balance", "krum", "trimmed_mean"],
        default="trustgraph",
    )
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--val-grad-sign", choices=["descent_neg", "paper_pos"], default="descent_neg")
    p.add_argument("--no-relu", action="store_true")
    p.add_argument("--trustgraph-metric", choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--trustgraph-clip-norm", action="store_true")
    p.add_argument("--balance-gamma", type=float, default=0.3)
    p.add_argument("--balance-kappa", type=float, default=1.0)
    p.add_argument("--krum-mode", choices=["multi", "single"], default="multi")
    p.add_argument("--trim-ratio", type=float, default=0.3)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--skip-existing", action="store_true", help="Skip runs that already have summary.json.")
    p.add_argument("--output-root", default="outputs/phase1")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = ExperimentConfig(
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
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
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
        trustgraph_clip_norm=args.trustgraph_clip_norm,
        balance_gamma=args.balance_gamma,
        balance_kappa=args.balance_kappa,
        krum_mode=args.krum_mode,
        trim_ratio=args.trim_ratio,
        device=args.device,
        eval_every=args.eval_every,
        verbose=not args.quiet,
    )

    out_root = Path(args.output_root)
    if args.mode == "single":
        for seed in args.seeds:
            tag = (
                f"{args.dataset}_{args.topology}_{args.attack}_byz{int(round(args.byzantine_frac * 100))}"
                f"_{args.aggregator}"
            )
            if args.aggregator == "trustgraph":
                tag += f"_{args.val_grad_sign}"
            out_dir = out_root / "single" / tag / f"seed_{seed}"
            cfg = replace(base_cfg, seed=seed, output_dir=str(out_dir))
            print(f"=== Running {tag} (seed={seed}) -> {out_dir} ===")
            res = run_experiment(cfg)
            s = res["summary"]
            print(
                f"DONE {tag} seed={seed}: final_acc={100.0 * s['final_benign_test_acc_mean']:.2f}% "
                f"best_acc={100.0 * s['best_benign_test_acc_mean']:.2f}% "
                f"fpr_cum={100.0 * s['fpr_cumulative']:.2f}% "
                f"wall={s['total_wall_sec']:.1f}s"
            )
        return

    # Suite mode: cache partitioned dataset per (seed, val_split) to avoid reloading CIFAR-10 repeatedly.
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"Invalid shard {args.shard_index}/{args.num_shards}")
    suite_root = out_root / args.suite
    suite_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    data_cache: dict[tuple[int, str], object] = {}

    all_jobs: list[tuple[int, str, ExperimentConfig]] = []
    skipped = 0
    for seed in args.seeds:
        seed_base = replace(base_cfg, seed=seed)
        for run_name, cfg in _suite_runs(args.suite, seed_base):
            out_dir = suite_root / run_name / f"seed_{seed}"
            if args.skip_existing and (out_dir / "summary.json").exists():
                skipped += 1
                continue
            all_jobs.append((seed, run_name, cfg))

    shard_jobs = [job for idx, job in enumerate(all_jobs) if idx % args.num_shards == args.shard_index]
    print(
        f"=== Suite '{args.suite}': shard {args.shard_index}/{args.num_shards} "
        f"running {len(shard_jobs)}/{len(all_jobs)} pending jobs (skipped {skipped} completed) ===",
        flush=True,
    )

    for job_idx, (seed, run_name, cfg) in enumerate(shard_jobs, start=1):
        cache_key = (seed, cfg.val_split)
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

    print(f"\n=== Suite '{args.suite}' shard {args.shard_index}/{args.num_shards} summary ===")
    print(f"{'Run':44s} {'Seed':>5s} {'FinalAcc':>9s} {'ConsAcc':>9s} {'FPR(cum)':>9s} {'ByzAcc':>8s} {'CosBen':>8s} {'CosByz':>8s}")
    for s in summaries:
        cb = f"{s['mean_benign_score']:+.3f}" if s["mean_benign_score"] is not None else "   n/a"
        cm = f"{s['mean_byz_score']:+.3f}" if s["mean_byz_score"] is not None else "   n/a"
        print(
            f"{s['run_name']:44s} {s['config']['seed']:5d} "
            f"{100.0 * s['final_benign_test_acc_mean']:8.2f}% "
            f"{100.0 * s['final_consensus_test_acc']:8.2f}% "
            f"{100.0 * s['fpr_cumulative']:8.2f}% "
            f"{100.0 * s['byz_accept_cumulative']:7.2f}% "
            f"{cb:>8s} {cm:>8s}"
        )


if __name__ == "__main__":
    main()
