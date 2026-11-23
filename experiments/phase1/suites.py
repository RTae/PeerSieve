"""Named Phase 1 TrustGraph-DFL experiment suite builders."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from src.simulator import ExperimentConfig

MethodSpec = tuple[str, dict[str, object]]

ALL_LITERAL_METHODS: tuple[MethodSpec, ...] = (
    ("fedavg", {"aggregator": "fedavg"}),
    ("trimmed_mean", {"aggregator": "trimmed_mean"}),
    ("krum", {"aggregator": "krum"}),
    ("balance", {"aggregator": "balance"}),
    ("tg_descent_neg", {"aggregator": "trustgraph", "val_grad_sign": "descent_neg"}),
    ("tg_paper_pos", {"aggregator": "trustgraph", "val_grad_sign": "paper_pos"}),
)

DEFENSE_LITERAL_METHODS: tuple[MethodSpec, ...] = (
    ("tg_descent_neg", {"aggregator": "trustgraph", "val_grad_sign": "descent_neg"}),
    ("tg_paper_pos", {"aggregator": "trustgraph", "val_grad_sign": "paper_pos"}),
    ("balance", {"aggregator": "balance"}),
    ("krum", {"aggregator": "krum"}),
    ("trimmed_mean", {"aggregator": "trimmed_mean"}),
)


def _base_50k(base: ExperimentConfig, **overrides: object) -> ExperimentConfig:
    """Return shared full-50k CIFAR-10 ResNet-18 configuration."""
    defaults: dict[str, object] = {
        "model_name": "resnet18_pretrained",
        "samples_per_node": 2500,
        "rounds": 25,
        "lr": 0.005,
        "lr_schedule": "cosine",
        "augment": True,
        "grad_clip_norm": 10.0,
        "alpha_dir": 0.5,
        "val_split": "iid",
        "topology": "random",
        "byzantine_frac": 0.3,
        "eval_batch_size": 256,
    }
    defaults.update(overrides)
    return replace(base, **defaults)


def _tg_50k_overrides(**overrides: object) -> dict[str, object]:
    """Return default stabilized 50k TrustGraph overrides."""
    cfg: dict[str, object] = {
        "aggregator": "trustgraph",
        "val_grad_sign": "descent_neg",
        "trustgraph_clip_norm": True,
        "trustgraph_clip_mode": "model",
        "trustgraph_clip_factor": 2.0,
        "trustgraph_score_threshold": 0.003,
        "trustgraph_delta_mode": "sender",
        "trustgraph_score_scope": "global",
    }
    cfg.update(overrides)
    return cfg


def _build_smoke(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Fast 12-run smoke suite (2 attacks x 6 methods)."""
    smoke_base = replace(
        base,
        topology="random",
        byzantine_frac=0.3,
        eval_every=base.eval_every if base.eval_every <= 0 else min(base.eval_every, base.rounds),
    )
    return [
        (f"{atk}__{tag}", replace(smoke_base, attack=atk, **overrides))
        for atk in ("label_flip", "model_poison")
        for tag, overrides in ALL_LITERAL_METHODS
    ]


def _build_literal_full(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Literal 75-config TrustGraph-DFL audit suite (10k CIFAR-10, 2-conv CNN, 200 rounds)."""
    runs: list[tuple[str, ExperimentConfig]] = []

    # 1) Main attacks (24): random ER(0.3), 30% Byzantine, 4 attacks x 6 methods
    for atk in ("none", "label_flip", "model_poison", "random_noise"):
        for tag, overrides in ALL_LITERAL_METHODS:
            runs.append(
                (f"main_attacks/{atk}__{tag}", replace(base, topology="random", byzantine_frac=0.3, attack=atk, **overrides))
            )

    # 2) Additional topologies (12): ring & small_world under label_flip, 30% Byzantine
    for topo in ("ring", "small_world"):
        for tag, overrides in ALL_LITERAL_METHODS:
            runs.append(
                (
                    f"topologies/{topo}__label_flip__{tag}",
                    replace(base, topology=topo, byzantine_frac=0.3, attack="label_flip", **overrides),
                )
            )

    # 3) FPR sweep (25): random ER(0.3), label_flip, 10%-50% Byzantine x 5 defenses
    for frac in (0.1, 0.2, 0.3, 0.4, 0.5):
        pct = int(round(frac * 100))
        for tag, overrides in DEFENSE_LITERAL_METHODS:
            runs.append(
                (
                    f"fpr_sweep/byz{pct}__{tag}",
                    replace(base, topology="random", byzantine_frac=frac, attack="label_flip", **overrides),
                )
            )

    # 4) Ablations & targeted diagnostics (14)
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
            ("diagnostics/clip_norm__model_poison__tg_descent_neg", replace(tg_mp, attack="model_poison", trustgraph_clip_norm=True)),
            ("diagnostics/clip_norm__random_noise__tg_descent_neg", replace(tg_mp, attack="random_noise", trustgraph_clip_norm=True)),
        ]
    )
    return runs


def _build_paper_match_50k(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Locked strong full-50k CIFAR-10 TrustGraph-DFL baseline (50 runs/seed)."""
    base_50k = _base_50k(base)
    tg_strong = _tg_50k_overrides()
    bal_cal: dict[str, object] = {"aggregator": "balance", "balance_gamma": 0.04}
    methods_5: tuple[MethodSpec, ...] = (
        ("fedavg", {"aggregator": "fedavg"}),
        ("trimmed_mean", {"aggregator": "trimmed_mean"}),
        ("krum", {"aggregator": "krum"}),
        ("balance", bal_cal),
        ("tg_strong", tg_strong),
    )
    methods_fpr: tuple[MethodSpec, ...] = (
        ("tg_strong", tg_strong),
        ("balance", bal_cal),
        ("krum", {"aggregator": "krum"}),
        ("trimmed_mean", {"aggregator": "trimmed_mean"}),
    )

    runs: list[tuple[str, ExperimentConfig]] = []
    # Panel 1 (20): Fig. 2 & Fig. 3(a) main attacks on Random ER (p=0.3, 30% Byzantine)
    for atk in ("none", "label_flip_all", "model_poison", "random_noise"):
        bep = 2 if atk == "label_flip_all" else 0
        for tag, overrides in methods_5:
            runs.append((f"main/{atk}__{tag}", replace(base_50k, attack=atk, byz_local_epochs=bep, **overrides)))

    # Panel 2 (10): Table 1 Ring & Small-World topologies under label_flip_all (30% Byzantine, bep=2)
    for topo in ("ring", "small_world"):
        for tag, overrides in methods_5:
            runs.append(
                (
                    f"topologies/{topo}__label_flip_all__{tag}",
                    replace(base_50k, topology=topo, attack="label_flip_all", byz_local_epochs=2, **overrides),
                )
            )

    # Panel 3 (16): Fig. 3(b) FPR sweep 10/20/40/50% Byzantine under label_flip_all (30% in Panel 1)
    for frac in (0.1, 0.2, 0.4, 0.5):
        pct = int(round(frac * 100))
        for tag, overrides in methods_fpr:
            runs.append(
                (
                    f"fpr/byz{pct}__{tag}",
                    replace(base_50k, attack="label_flip_all", byz_local_epochs=2, byzantine_frac=frac, **overrides),
                )
            )

    # Panel 4 (4): Section 5.5 ablations under model_poison (30% Byzantine)
    tg_mp = replace(base_50k, attack="model_poison", **tg_strong)
    runs.extend(
        [
            ("ablations/no_relu__model_poison", replace(tg_mp, trustgraph_use_relu=False)),
            ("ablations/euclidean__model_poison", replace(tg_mp, trustgraph_metric="euclidean", trustgraph_score_threshold=0.0)),
            ("ablations/alpha03__model_poison", replace(tg_mp, alpha=0.3)),
            ("ablations/alpha07__model_poison", replace(tg_mp, alpha=0.7)),
        ]
    )
    return runs


def _build_fpr_lever1(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Lever 1 (8 runs): receiver-anchored displacement (theta_j^{t+1/2} - theta_i^t)."""
    cfg = _base_50k(base, **_tg_50k_overrides(trustgraph_delta_mode="receiver"))
    return [
        ("lever1/none__iidval", replace(cfg, attack="none", val_split="iid")),
        ("lever1/none__local", replace(cfg, attack="none", val_split="local")),
        ("lever1/label_flip_all_byz30__iidval", replace(cfg, attack="label_flip_all", byz_local_epochs=2, val_split="iid")),
        ("lever1/label_flip_all_byz50__iidval", replace(cfg, attack="label_flip_all", byz_local_epochs=2, byzantine_frac=0.5, val_split="iid")),
        ("lever1/label_flip_all_byz50__local", replace(cfg, attack="label_flip_all", byz_local_epochs=2, byzantine_frac=0.5, val_split="local")),
        ("lever1/label_flip_all_byz50_bep1__iidval", replace(cfg, attack="label_flip_all", byz_local_epochs=0, byzantine_frac=0.5, val_split="iid")),
        ("lever1/model_poison__iidval", replace(cfg, attack="model_poison", val_split="iid")),
        ("lever1/random_noise__iidval", replace(cfg, attack="random_noise", val_split="iid")),
    ]


def _four_attack_grid(prefix: str, tag: str, cfg: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    return [
        (f"{prefix}/{tag}__none", replace(cfg, attack="none")),
        (f"{prefix}/{tag}__label_flip_all_byz30", replace(cfg, attack="label_flip_all", byz_local_epochs=2)),
        (f"{prefix}/{tag}__model_poison", replace(cfg, attack="model_poison")),
        (f"{prefix}/{tag}__random_noise", replace(cfg, attack="random_noise")),
    ]


def _build_fpr_lever1b(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Lever 1b (8 runs): tau in {0.02, 0.035} sweep under receiver-anchored displacement."""
    base_rx = _base_50k(base, **_tg_50k_overrides(trustgraph_delta_mode="receiver"))
    runs: list[tuple[str, ExperimentConfig]] = []
    for tau, tag in ((0.02, "t020"), (0.035, "t035")):
        runs.extend(_four_attack_grid("lever1b", tag, replace(base_rx, trustgraph_score_threshold=tau)))
    return runs


def _build_fpr_lever2(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Lever 2 (8 runs): fewer local SGD steps per round in strict sender mode."""
    base_sender = _base_50k(base, **_tg_50k_overrides())
    runs: list[tuple[str, ExperimentConfig]] = []
    runs.extend(_four_attack_grid("lever2", "bs256_r25", replace(base_sender, batch_size=256, rounds=25, max_local_steps=0, eval_every=5)))
    runs.extend(_four_attack_grid("lever2", "steps4_r100", replace(base_sender, batch_size=64, rounds=100, max_local_steps=4, eval_every=10)))
    return runs


def _build_fpr_lever3(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Lever 3 (8 runs): layer-mean and subspace (backbone_only / head_only) cosine in sender mode."""
    base_sender = _base_50k(base, **_tg_50k_overrides())
    lm = replace(base_sender, trustgraph_score_scope="layer_mean")
    bb = replace(base_sender, trustgraph_score_scope="backbone_only")
    hd = replace(base_sender, trustgraph_score_scope="head_only")
    return _four_attack_grid("lever3", "layer_mean", lm) + [
        ("lever3/backbone_only__none", replace(bb, attack="none")),
        ("lever3/backbone_only__label_flip_all_byz30", replace(bb, attack="label_flip_all", byz_local_epochs=2)),
        ("lever3/backbone_only__model_poison", replace(bb, attack="model_poison")),
        ("lever3/head_only__label_flip_all_byz30", replace(hd, attack="label_flip_all", byz_local_epochs=2)),
    ]


def _build_fpr_lever4(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Lever 4 (8 runs): multi-round zero-initialized EMA on consistency scores (beta in {0.7, 0.85})."""
    base_sender = _base_50k(base, **_tg_50k_overrides())
    runs: list[tuple[str, ExperimentConfig]] = []
    for beta, tag in ((0.7, "ema70"), (0.85, "ema85")):
        runs.extend(_four_attack_grid("lever4", tag, replace(base_sender, trustgraph_ema_beta=beta)))
    return runs


SUITES: dict[str, Callable[[ExperimentConfig], list[tuple[str, ExperimentConfig]]]] = {
    "smoke": _build_smoke,
    "full": _build_literal_full,
    "literal": _build_literal_full,
    "step_a": _build_literal_full,
    "paper_match_50k": _build_paper_match_50k,
    "baseline_50k": _build_paper_match_50k,
    "fpr_lever1": _build_fpr_lever1,
    "fpr_lever1b": _build_fpr_lever1b,
    "fpr_lever2": _build_fpr_lever2,
    "fpr_lever3": _build_fpr_lever3,
    "fpr_lever4": _build_fpr_lever4,
}


def build_suite_runs(suite: str, base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Return the ordered list of (run_name, config) entries for a named Phase 1 suite."""
    if suite not in SUITES:
        raise ValueError(f"Unknown suite {suite!r}; expected one of {sorted(SUITES)}.")
    return SUITES[suite](base)


_suite_runs = build_suite_runs
