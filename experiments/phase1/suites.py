"""Named Phase 1 TrustGraph-DFL experiment suites aligned with TrustGraph-DFL.pdf."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from src.simulator import ExperimentConfig

MethodSpec = tuple[str, dict[str, object]]

TRUSTGRAPH_SIGN_VARIANTS: tuple[MethodSpec, ...] = (
    ("tg_descent_neg", {"aggregator": "trustgraph", "val_grad_sign": "descent_neg"}),
    ("tg_paper_pos", {"aggregator": "trustgraph", "val_grad_sign": "paper_pos"}),
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
    """Return default stabilized 50k TrustGraph-DFL overrides."""
    cfg: dict[str, object] = {
        "aggregator": "trustgraph",
        "val_grad_sign": "descent_neg",
        "trustgraph_clip_norm": True,
        "trustgraph_clip_mode": "model",
        "trustgraph_clip_factor": 2.0,
        "trustgraph_score_threshold": 0.003,
        "trustgraph_delta_mode": "sender",
    }
    cfg.update(overrides)
    return cfg


def _build_smoke(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Fast 4-run TrustGraph-DFL sanity suite (2 attacks x 2 gradient signs)."""
    smoke_base = replace(
        base,
        topology="random",
        byzantine_frac=0.3,
        eval_every=base.eval_every if base.eval_every <= 0 else min(base.eval_every, base.rounds),
    )
    return [
        (f"{atk}__{tag}", replace(smoke_base, attack=atk, **overrides))
        for atk in ("label_flip", "model_poison")
        for tag, overrides in TRUSTGRAPH_SIGN_VARIANTS
    ]


def _build_paper_cnn_10k(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Literal Section 5 audit (31 configs: 10k CIFAR-10, 2-conv CNN, 200 rounds)."""
    runs: list[tuple[str, ExperimentConfig]] = []

    # 1) Fig. 3a main attacks (8): random ER(0.3), 30% Byzantine, 4 attacks x 2 TrustGraph signs
    for atk in ("none", "label_flip", "model_poison", "random_noise"):
        for tag, overrides in TRUSTGRAPH_SIGN_VARIANTS:
            runs.append(
                (f"fig3a_attacks/{atk}__{tag}", replace(base, topology="random", byzantine_frac=0.3, attack=atk, **overrides))
            )

    # 2) Table 1 topologies (4): ring & small_world under label_flip, 30% Byzantine
    for topo in ("ring", "small_world"):
        for tag, overrides in TRUSTGRAPH_SIGN_VARIANTS:
            runs.append(
                (
                    f"table1_topologies/{topo}__label_flip__{tag}",
                    replace(base, topology=topo, byzantine_frac=0.3, attack="label_flip", **overrides),
                )
            )

    # 3) Fig. 3b Byzantine sweep (10): random ER(0.3), label_flip, 10%-50% Byzantine x 2 signs
    for frac in (0.1, 0.2, 0.3, 0.4, 0.5):
        pct = int(round(frac * 100))
        for tag, overrides in TRUSTGRAPH_SIGN_VARIANTS:
            runs.append(
                (
                    f"fig3b_byz_sweep/byz{pct}__{tag}",
                    replace(base, topology="random", byzantine_frac=frac, attack="label_flip", **overrides),
                )
            )

    # 4) Section 5.5 ablations & targeted TrustGraph-DFL diagnostics (9)
    rand_base = replace(base, topology="random", byzantine_frac=0.3)
    tg_mp = replace(rand_base, attack="model_poison", aggregator="trustgraph", val_grad_sign="descent_neg")
    runs.extend(
        [
            ("sec5_5_ablations/no_relu__model_poison", replace(tg_mp, trustgraph_use_relu=False)),
            ("sec5_5_ablations/euclidean__model_poison", replace(tg_mp, trustgraph_metric="euclidean")),
            ("sec5_5_ablations/alpha03__model_poison", replace(tg_mp, alpha=0.3)),
            ("sec5_5_ablations/alpha07__model_poison", replace(tg_mp, alpha=0.7)),
            ("diagnostics/valiid__label_flip__tg_descent_neg", replace(tg_mp, attack="label_flip", val_split="iid")),
            ("diagnostics/valiid__model_poison__tg_descent_neg", replace(tg_mp, attack="model_poison", val_split="iid")),
            ("diagnostics/label_flip_all__tg_descent_neg", replace(tg_mp, attack="label_flip_all")),
            ("diagnostics/clip_norm__model_poison__tg_descent_neg", replace(tg_mp, attack="model_poison", trustgraph_clip_norm=True)),
            ("diagnostics/clip_norm__random_noise__tg_descent_neg", replace(tg_mp, attack="random_noise", trustgraph_clip_norm=True)),
        ]
    )
    return runs


def _build_fig3a_attacks(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Fig. 2 / Fig. 3a (4 runs): clean + 30% Byzantine attacks on 50k ResNet-18."""
    base_50k = _base_50k(base)
    tg_strong = _tg_50k_overrides()
    return [
        (
            f"fig3a_attacks/{atk}__tg_strong",
            replace(base_50k, attack=atk, byz_local_epochs=(2 if atk == "label_flip_all" else 0), **tg_strong),
        )
        for atk in ("none", "label_flip_all", "model_poison", "random_noise")
    ]


def _build_table1_topologies(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Table 1 (3 runs): Ring, Random, Small-World under 30% label_flip_all on 50k ResNet-18."""
    base_50k = _base_50k(base)
    tg_strong = _tg_50k_overrides()
    return [
        (
            f"table1_topologies/{topo}__label_flip_all__tg_strong",
            replace(base_50k, topology=topo, attack="label_flip_all", byz_local_epochs=2, **tg_strong),
        )
        for topo in ("ring", "random", "small_world")
    ]


def _build_fig3b_byz_sweep(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Fig. 3b (5 runs): 10% to 50% Byzantine stress sweep under label_flip_all on 50k ResNet-18."""
    base_50k = _base_50k(base)
    tg_strong = _tg_50k_overrides()
    return [
        (
            f"fig3b_byz_sweep/byz{int(round(frac * 100))}__tg_strong",
            replace(base_50k, attack="label_flip_all", byz_local_epochs=2, byzantine_frac=frac, **tg_strong),
        )
        for frac in (0.1, 0.2, 0.3, 0.4, 0.5)
    ]


def _build_sec5_5_ablations(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Section 5.5 (5 runs): full alpha=0.5, alpha=0.3, alpha=0.7, Euclidean, No ReLU under model_poison."""
    base_50k = _base_50k(base)
    tg_mp = replace(base_50k, attack="model_poison", **_tg_50k_overrides())
    return [
        ("sec5_5_ablations/full_alpha05__model_poison", tg_mp),
        ("sec5_5_ablations/alpha03__model_poison", replace(tg_mp, alpha=0.3)),
        ("sec5_5_ablations/alpha07__model_poison", replace(tg_mp, alpha=0.7)),
        ("sec5_5_ablations/euclidean__model_poison", replace(tg_mp, trustgraph_metric="euclidean", trustgraph_score_threshold=0.0)),
        ("sec5_5_ablations/no_relu__model_poison", replace(tg_mp, trustgraph_use_relu=False)),
    ]


def _build_paper_resnet18_50k(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Full 50k ResNet-18 evaluation (14 deduplicated runs: Fig. 3a, Table 1, Fig. 3b, Section 5.5)."""
    runs: list[tuple[str, ExperimentConfig]] = []
    runs.extend(_build_fig3a_attacks(base))
    runs.extend(r for r in _build_table1_topologies(base) if r[1].topology != "random")
    runs.extend(r for r in _build_fig3b_byz_sweep(base) if int(round(r[1].byzantine_frac * 100)) != 30)
    runs.extend(r for r in _build_sec5_5_ablations(base) if not r[0].startswith("sec5_5_ablations/full_alpha05"))
    return runs


def _four_attack_grid(prefix: str, tag: str, cfg: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    return [
        (f"{prefix}/{tag}__none", replace(cfg, attack="none")),
        (f"{prefix}/{tag}__label_flip_all_byz30", replace(cfg, attack="label_flip_all", byz_local_epochs=2)),
        (f"{prefix}/{tag}__model_poison", replace(cfg, attack="model_poison")),
        (f"{prefix}/{tag}__random_noise", replace(cfg, attack="random_noise")),
    ]


def _build_fig3b_fpr_rxdelta(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Fig. 3b FPR diagnostic (8 runs): receiver-anchored delta (theta_j^{t+1/2} - theta_i^t)."""
    cfg = _base_50k(base, **_tg_50k_overrides(trustgraph_delta_mode="receiver"))
    return [
        ("fig3b_fpr_rxdelta/none__iidval", replace(cfg, attack="none", val_split="iid")),
        ("fig3b_fpr_rxdelta/none__local", replace(cfg, attack="none", val_split="local")),
        ("fig3b_fpr_rxdelta/label_flip_all_byz30__iidval", replace(cfg, attack="label_flip_all", byz_local_epochs=2, val_split="iid")),
        ("fig3b_fpr_rxdelta/label_flip_all_byz50__iidval", replace(cfg, attack="label_flip_all", byz_local_epochs=2, byzantine_frac=0.5, val_split="iid")),
        ("fig3b_fpr_rxdelta/label_flip_all_byz50__local", replace(cfg, attack="label_flip_all", byz_local_epochs=2, byzantine_frac=0.5, val_split="local")),
        ("fig3b_fpr_rxdelta/label_flip_all_byz50_bep1__iidval", replace(cfg, attack="label_flip_all", byz_local_epochs=0, byzantine_frac=0.5, val_split="iid")),
        ("fig3b_fpr_rxdelta/model_poison__iidval", replace(cfg, attack="model_poison", val_split="iid")),
        ("fig3b_fpr_rxdelta/random_noise__iidval", replace(cfg, attack="random_noise", val_split="iid")),
    ]


def _build_fig3b_fpr_rxdelta_tau(base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Fig. 3b FPR diagnostic (8 runs): tau in {0.02, 0.035} sweep under receiver-anchored delta."""
    base_rx = _base_50k(base, **_tg_50k_overrides(trustgraph_delta_mode="receiver"))
    runs: list[tuple[str, ExperimentConfig]] = []
    for tau, tag in ((0.02, "t020"), (0.035, "t035")):
        runs.extend(_four_attack_grid("fig3b_fpr_rxdelta_tau", tag, replace(base_rx, trustgraph_score_threshold=tau)))
    return runs


SUITES: dict[str, Callable[[ExperimentConfig], list[tuple[str, ExperimentConfig]]]] = {
    "smoke": _build_smoke,
    "paper_cnn_10k": _build_paper_cnn_10k,
    "paper_resnet18_50k": _build_paper_resnet18_50k,
    "fig3a_attacks": _build_fig3a_attacks,
    "table1_topologies": _build_table1_topologies,
    "fig3b_byz_sweep": _build_fig3b_byz_sweep,
    "sec5_5_ablations": _build_sec5_5_ablations,
    "fig3b_fpr_rxdelta": _build_fig3b_fpr_rxdelta,
    "fig3b_fpr_rxdelta_tau": _build_fig3b_fpr_rxdelta_tau,
}


def build_suite_runs(suite: str, base: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    """Return the ordered list of (run_name, config) entries for a named Phase 1 suite."""
    if suite not in SUITES:
        raise ValueError(f"Unknown suite {suite!r}; expected one of {sorted(SUITES)}.")
    return SUITES[suite](base)


_suite_runs = build_suite_runs
