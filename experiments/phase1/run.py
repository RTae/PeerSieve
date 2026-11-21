"""CLI runner for Phase 1 (TrustGraph-DFL CIFAR-10 reproduction)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import replace
import json
from pathlib import Path

from src.simulator import ExperimentConfig, prepare_dataset, run_experiment

MethodSpec = tuple[str, dict[str, object]]

ALL_METHODS: tuple[MethodSpec, ...] = (
    ("fedavg", {"aggregator": "fedavg"}),
    ("trimmed_mean", {"aggregator": "trimmed_mean"}),
    ("krum", {"aggregator": "krum"}),
    ("balance", {"aggregator": "balance"}),
    ("tg_descent_neg", {"aggregator": "trustgraph", "val_grad_sign": "descent_neg"}),
    ("tg_paper_pos", {"aggregator": "trustgraph", "val_grad_sign": "paper_pos"}),
)

DEFENSE_METHODS: tuple[MethodSpec, ...] = (
    ("tg_descent_neg", {"aggregator": "trustgraph", "val_grad_sign": "descent_neg"}),
    ("tg_paper_pos", {"aggregator": "trustgraph", "val_grad_sign": "paper_pos"}),
    ("balance", {"aggregator": "balance"}),
    ("krum", {"aggregator": "krum"}),
    ("trimmed_mean", {"aggregator": "trimmed_mean"}),
)

CNN_GN_BALANCE_GAMMA = 0.15
RESNET18_BALANCE_GAMMA = 0.015


def _build_smoke(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    smoke_base = replace(
        base,
        topology="random",
        byzantine_frac=0.3,
        eval_every=base.eval_every if base.eval_every <= 0 else min(base.eval_every, base.rounds),
    )
    return [
        (f"{atk}__{tag}", replace(smoke_base, attack=atk, **overrides))
        for atk in ("label_flip", "model_poison")
        for tag, overrides in ALL_METHODS
    ]


def _build_sign_check(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    return [
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
        for atk in ("none", "label_flip", "model_poison", "random_noise")
        for sign in ("descent_neg", "paper_pos")
    ]


def _build_main_attacks(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    return [
        (f"{atk}__{tag}", replace(base, topology="random", byzantine_frac=0.3, attack=atk, **overrides))
        for atk in ("none", "label_flip", "model_poison", "random_noise")
        for tag, overrides in ALL_METHODS
    ]


def _build_topologies(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    return [
        (f"{topo}__{tag}", replace(base, topology=topo, byzantine_frac=0.3, attack=base.attack, **overrides))
        for topo in ("ring", "random", "small_world")
        for tag, overrides in ALL_METHODS
    ]


def _build_fpr_sweep(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    runs: list[tuple[str, ExperimentConfig]] = []
    for frac in (0.1, 0.2, 0.3, 0.4, 0.5):
        pct = int(round(frac * 100))
        for tag, overrides in DEFENSE_METHODS:
            runs.append(
                (
                    f"byz{pct}__{tag}",
                    replace(base, topology="random", byzantine_frac=frac, attack=base.attack, **overrides),
                )
            )
    return runs


def _build_ablations(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    abl_base = replace(
        base,
        topology="random",
        byzantine_frac=0.3,
        attack="model_poison",
        aggregator="trustgraph",
        val_grad_sign="descent_neg",
    )
    return [
        ("full_descent_neg_a05", replace(abl_base, alpha=0.5)),
        ("no_relu_descent_neg", replace(abl_base, trustgraph_use_relu=False)),
        ("euclidean", replace(abl_base, trustgraph_metric="euclidean")),
        ("alpha_03", replace(abl_base, alpha=0.3)),
        ("alpha_07", replace(abl_base, alpha=0.7)),
    ]


def _build_full_phase1(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Complete 75-config Phase 1 reproduction, ablation, and diagnostic matrix."""
    runs: list[tuple[str, ExperimentConfig]] = []

    # 1) Main attacks (24): random ER(0.3), 30% Byzantine, 4 attacks x 6 methods
    for name, cfg in _build_main_attacks(base):
        runs.append((f"main_attacks/{name}", cfg))

    # 2) Additional topologies (12): ring & small_world under label_flip, 30% Byzantine
    for topo in ("ring", "small_world"):
        for tag, overrides in ALL_METHODS:
            runs.append(
                (
                    f"topologies/{topo}__label_flip__{tag}",
                    replace(base, topology=topo, byzantine_frac=0.3, attack="label_flip", **overrides),
                )
            )

    # 3) FPR sweep (25): random ER(0.3), label_flip, 10%-50% Byzantine x 5 defenses
    for name, cfg in _build_fpr_sweep(replace(base, attack="label_flip")):
        runs.append((f"fpr_sweep/{name}", cfg))

    # 4) Ablations & diagnostics (14)
    rand_base = replace(base, topology="random", byzantine_frac=0.3)
    tg_mp = replace(rand_base, attack="model_poison", aggregator="trustgraph", val_grad_sign="descent_neg")
    bal_g015 = replace(rand_base, aggregator="balance", balance_gamma=0.15)
    runs.extend(
        [
            ("ablations/no_relu__model_poison", replace(tg_mp, trustgraph_use_relu=False)),
            ("ablations/euclidean__model_poison", replace(tg_mp, trustgraph_metric="euclidean")),
            ("ablations/alpha03__model_poison", replace(tg_mp, alpha=0.3)),
            ("ablations/alpha07__model_poison", replace(tg_mp, alpha=0.7)),
            ("diagnostics/valiid__label_flip__tg_descent_neg", replace(tg_mp, attack="label_flip", val_split="iid")),
            ("diagnostics/valiid__model_poison__tg_descent_neg", replace(tg_mp, attack="model_poison", val_split="iid")),
            ("diagnostics/bal_g015__label_flip", replace(bal_g015, attack="label_flip")),
            ("diagnostics/bal_g015__model_poison", replace(bal_g015, attack="model_poison")),
            ("diagnostics/bal_g015__random_noise", replace(bal_g015, attack="random_noise")),
            ("diagnostics/label_flip_all__fedavg", replace(rand_base, attack="label_flip_all", aggregator="fedavg")),
            ("diagnostics/label_flip_all__balance_g015", replace(bal_g015, attack="label_flip_all")),
            ("diagnostics/label_flip_all__tg_descent_neg", replace(tg_mp, attack="label_flip_all")),
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


def _build_stabilized_ab(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Option C (110 runs): Part A stabilized 2-conv CNN + Part B pretrained ResNet-18."""
    runs: list[tuple[str, ExperimentConfig]] = []
    tg_stab_overrides: dict[str, object] = {
        "aggregator": "trustgraph",
        "val_grad_sign": "descent_neg",
        "trustgraph_clip_norm": True,
        "trustgraph_clip_mode": "model",
        "trustgraph_clip_factor": 2.0,
        "trustgraph_score_threshold": 0.02,
    }
    tg_raw_overrides: dict[str, object] = {
        "aggregator": "trustgraph",
        "val_grad_sign": "descent_neg",
        "trustgraph_clip_norm": False,
        "trustgraph_score_threshold": 0.0,
    }

    # --- Part A: Stabilized 2-conv 2-FC CNN (GroupNorm + Aug + Cosine LR, 200 rounds) ---
    base_a = replace(
        base,
        model_name="cnn2x2_gn",
        rounds=200,
        lr=0.01,
        lr_schedule="cosine",
        augment=True,
        topology="random",
        byzantine_frac=0.3,
    )
    methods_a6: tuple[MethodSpec, ...] = (
        ("fedavg", {"aggregator": "fedavg"}),
        ("trimmed_mean", {"aggregator": "trimmed_mean"}),
        ("krum", {"aggregator": "krum"}),
        ("balance", {"aggregator": "balance", "balance_gamma": CNN_GN_BALANCE_GAMMA}),
        ("tg_raw", tg_raw_overrides),
        ("tg_stab", tg_stab_overrides),
    )
    methods_a5: tuple[MethodSpec, ...] = (
        ("fedavg", {"aggregator": "fedavg"}),
        ("trimmed_mean", {"aggregator": "trimmed_mean"}),
        ("krum", {"aggregator": "krum"}),
        ("balance", {"aggregator": "balance", "balance_gamma": CNN_GN_BALANCE_GAMMA}),
        ("tg_stab", tg_stab_overrides),
    )
    methods_a_fpr: tuple[MethodSpec, ...] = (
        ("tg_stab", tg_stab_overrides),
        ("balance", {"aggregator": "balance", "balance_gamma": CNN_GN_BALANCE_GAMMA}),
        ("krum", {"aggregator": "krum"}),
        ("trimmed_mean", {"aggregator": "trimmed_mean"}),
    )

    # A1 (24): alpha_dir=0.5 main attacks
    for atk in ("none", "label_flip_all", "model_poison", "random_noise"):
        for tag, overrides in methods_a6:
            runs.append((f"part_a/dir05/{atk}__{tag}", replace(base_a, alpha_dir=0.5, attack=atk, **overrides)))

    # A2 (20): alpha_dir=5.0 main attacks
    for atk in ("none", "label_flip_all", "model_poison", "random_noise"):
        for tag, overrides in methods_a5:
            runs.append((f"part_a/dir50/{atk}__{tag}", replace(base_a, alpha_dir=5.0, attack=atk, **overrides)))

    # A3 (16): alpha_dir=5.0 FPR sweep under label_flip_all (30% already in A2)
    for frac in (0.1, 0.2, 0.4, 0.5):
        pct = int(round(frac * 100))
        for tag, overrides in methods_a_fpr:
            runs.append(
                (
                    f"part_a/fpr_dir50/byz{pct}__{tag}",
                    replace(base_a, alpha_dir=5.0, attack="label_flip_all", byzantine_frac=frac, **overrides),
                )
            )

    # --- Part B: Pretrained ResNet-18 fine-tuned on 10k CIFAR-10 (alpha_dir=0.5, 50 rounds) ---
    base_b = replace(
        base,
        model_name="resnet18_pretrained",
        rounds=50,
        lr=0.005,
        lr_schedule="cosine",
        augment=True,
        alpha_dir=0.5,
        topology="random",
        byzantine_frac=0.3,
        eval_batch_size=256,
    )
    methods_b6: tuple[MethodSpec, ...] = (
        ("fedavg", {"aggregator": "fedavg"}),
        ("trimmed_mean", {"aggregator": "trimmed_mean"}),
        ("krum", {"aggregator": "krum"}),
        ("balance", {"aggregator": "balance", "balance_gamma": RESNET18_BALANCE_GAMMA}),
        ("tg_raw", tg_raw_overrides),
        ("tg_stab", tg_stab_overrides),
    )
    methods_b5: tuple[MethodSpec, ...] = (
        ("fedavg", {"aggregator": "fedavg"}),
        ("trimmed_mean", {"aggregator": "trimmed_mean"}),
        ("krum", {"aggregator": "krum"}),
        ("balance", {"aggregator": "balance", "balance_gamma": RESNET18_BALANCE_GAMMA}),
        ("tg_stab", tg_stab_overrides),
    )
    methods_b_fpr: tuple[MethodSpec, ...] = (
        ("tg_stab", tg_stab_overrides),
        ("balance", {"aggregator": "balance", "balance_gamma": RESNET18_BALANCE_GAMMA}),
        ("krum", {"aggregator": "krum"}),
        ("trimmed_mean", {"aggregator": "trimmed_mean"}),
    )

    # B1 (24): main attacks on random ER(0.3), 30% Byzantine
    for atk in ("none", "label_flip_all", "model_poison", "random_noise"):
        for tag, overrides in methods_b6:
            runs.append((f"part_b/main/{atk}__{tag}", replace(base_b, attack=atk, **overrides)))

    # B2 (10): ring & small_world topologies under label_flip_all
    for topo in ("ring", "small_world"):
        for tag, overrides in methods_b5:
            runs.append(
                (
                    f"part_b/topologies/{topo}__label_flip_all__{tag}",
                    replace(base_b, topology=topo, attack="label_flip_all", **overrides),
                )
            )

    # B3 (16): FPR sweep 10/20/40/50% under label_flip_all (30% already in B1)
    for frac in (0.1, 0.2, 0.4, 0.5):
        pct = int(round(frac * 100))
        for tag, overrides in methods_b_fpr:
            runs.append(
                (
                    f"part_b/fpr/byz{pct}__{tag}",
                    replace(base_b, attack="label_flip_all", byzantine_frac=frac, **overrides),
                )
            )

    return runs


def _build_paper_match_calib(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Phase 1c calibration (16 runs, 2/GPU on 8x A100): grad_clip, low-FPR validation, & BALANCE gamma."""
    base_b = replace(
        base,
        model_name="resnet18_pretrained",
        rounds=50,
        lr=0.005,
        lr_schedule="cosine",
        augment=True,
        grad_clip_norm=1.0,
        alpha_dir=0.5,
        topology="random",
        byzantine_frac=0.3,
        eval_batch_size=256,
    )
    tg_stab: dict[str, object] = {
        "aggregator": "trustgraph",
        "val_grad_sign": "descent_neg",
        "trustgraph_clip_norm": True,
        "trustgraph_clip_mode": "model",
        "trustgraph_clip_factor": 2.0,
        "trustgraph_score_threshold": 0.02,
    }
    tg_stab_t01: dict[str, object] = {**tg_stab, "trustgraph_score_threshold": 0.01}
    tg_raw: dict[str, object] = {
        "aggregator": "trustgraph",
        "val_grad_sign": "descent_neg",
        "trustgraph_clip_norm": False,
        "trustgraph_score_threshold": 0.0,
    }

    return [
        # 1-2: Confirm grad_clip_norm=1.0 fixes isolated-peer NaN under model_poison (local val, dir=0.5)
        ("calib/local_dir05/model_poison__tg_stab", replace(base_b, attack="model_poison", **tg_stab)),
        ("calib/local_dir05/model_poison__tg_raw", replace(base_b, attack="model_poison", **tg_raw)),
        # 3-9: Stratified balanced validation (val_split=iid) with non-IID dir=0.5 training shards
        ("calib/iidval_dir05/none__tg_stab", replace(base_b, val_split="iid", attack="none", **tg_stab)),
        (
            "calib/iidval_dir05/label_flip_all_byz30__tg_stab",
            replace(base_b, val_split="iid", attack="label_flip_all", **tg_stab),
        ),
        (
            "calib/iidval_dir05/label_flip_all_byz50__tg_stab",
            replace(base_b, val_split="iid", attack="label_flip_all", byzantine_frac=0.5, **tg_stab),
        ),
        (
            "calib/iidval_dir05/model_poison_byz30__tg_stab",
            replace(base_b, val_split="iid", attack="model_poison", **tg_stab),
        ),
        (
            "calib/iidval_dir05/random_noise_byz30__tg_stab",
            replace(base_b, val_split="iid", attack="random_noise", **tg_stab),
        ),
        (
            "calib/iidval_dir05/random_noise_byz30__tg_stab_t01",
            replace(base_b, val_split="iid", attack="random_noise", **tg_stab_t01),
        ),
        (
            "calib/iidval_dir05/label_flip_all_byz50__tg_stab_t01",
            replace(base_b, val_split="iid", attack="label_flip_all", byzantine_frac=0.5, **tg_stab_t01),
        ),
        # 10-11: Local validation under milder heterogeneity (dir=5.0) & larger shard (20k -> 100 val/node)
        (
            "calib/local_dir50/label_flip_all_byz50__tg_stab",
            replace(base_b, alpha_dir=5.0, attack="label_flip_all", byzantine_frac=0.5, **tg_stab),
        ),
        (
            "calib/local_dir05_20k/label_flip_all_byz30__tg_stab",
            replace(base_b, samples_per_node=1000, attack="label_flip_all", **tg_stab),
        ),
        # 12-16: BALANCE gamma calibration on pretrained ResNet-18 with grad_clip_norm=1.0
        ("calib/balance/none__bal_g03", replace(base_b, attack="none", aggregator="balance", balance_gamma=0.03)),
        (
            "calib/balance/label_flip_all__bal_g03",
            replace(base_b, attack="label_flip_all", aggregator="balance", balance_gamma=0.03),
        ),
        ("calib/balance/none__bal_g05", replace(base_b, attack="none", aggregator="balance", balance_gamma=0.05)),
        (
            "calib/balance/label_flip_all__bal_g05",
            replace(base_b, attack="label_flip_all", aggregator="balance", balance_gamma=0.05),
        ),
        (
            "calib/balance/model_poison__bal_g05",
            replace(base_b, attack="model_poison", aggregator="balance", balance_gamma=0.05),
        ),
    ]


SUITES: dict[str, Callable[[ExperimentConfig], list[tuple[str, ExperimentConfig]]]] = {
    "smoke": _build_smoke,
    "sign_check": _build_sign_check,
    "main_attacks": _build_main_attacks,
    "topologies": _build_topologies,
    "fpr_sweep": _build_fpr_sweep,
    "ablations": _build_ablations,
    "full": _build_full_phase1,
    "step_a": _build_full_phase1,  # backwards-compatible alias
    "stabilized_ab": _build_stabilized_ab,
    "paper_match_calib": _build_paper_match_calib,
}


def build_suite_runs(suite: str, base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Return the ordered list of (run_name, config) entries for a named Phase 1 suite."""
    if suite not in SUITES:
        raise ValueError(f"Unknown suite {suite!r}; expected one of {sorted(SUITES)}.")
    return SUITES[suite](base)


# Backwards-compatible name used by existing tests/scripts.
_suite_runs = build_suite_runs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1: TrustGraph-DFL reproduction runner.")
    p.add_argument("--mode", choices=["single", "suite"], default="single")
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
        choices=["cnn2x2", "cnn2x2_gn", "lenet2x2", "resnet18_pretrained", "resnet18_scratch"],
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
    p.add_argument("--trustgraph-score-threshold", type=float, default=0.0)
    p.add_argument("--trustgraph-clip-norm", action="store_true")
    p.add_argument("--trustgraph-clip-mode", choices=["update", "model"], default="update")
    p.add_argument("--trustgraph-clip-factor", type=float, default=2.0)
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
        balance_gamma=args.balance_gamma,
        balance_kappa=args.balance_kappa,
        krum_mode=args.krum_mode,
        trim_ratio=args.trim_ratio,
        device=args.device,
        eval_every=args.eval_every,
        verbose=not args.quiet,
    )


def _run_single(args: argparse.Namespace, base_cfg: ExperimentConfig, out_root: Path) -> None:
    tag = (
        f"{args.dataset}_{args.topology}_{args.attack}_byz{int(round(args.byzantine_frac * 100))}"
        f"_{args.aggregator}"
    )
    if args.aggregator == "trustgraph":
        tag += f"_{args.val_grad_sign}"

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


def main() -> None:
    args = parse_args()
    base_cfg = _config_from_args(args)
    out_root = Path(args.output_root)
    if args.mode == "single":
        _run_single(args, base_cfg, out_root)
    else:
        _run_suite(args, base_cfg, out_root)


if __name__ == "__main__":
    main()
