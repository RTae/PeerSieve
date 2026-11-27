import json
from pathlib import Path
from typing import Annotated, Literal

import matplotlib.pyplot as plt
import numpy as np
import typer

from experiments.phase1 import figroles as fr
from experiments.phase1 import figstyle as fs
from src.logging_utils import configure_logging, get_logger

logger = get_logger()
app = typer.Typer(
    name="peersieve-plot",
    help="Generate publication 300 dpi PNG and header-free PDF figures from committed JSON datasets.",
    add_completion=False,
    pretty_exceptions_enable=False,
)


def _parse_variant_key(label: str) -> str | None:
    """Map a method summary string to its canonical figure variant key.

    Args:
        label: Method label string from phase1_paper_match_50k_3seed.json.

    Returns:
        Variant key ('tg_stab', 'tg_a03', 'tg_a07', 'tg_euclidean', 'tg_no_relu'), or None.
    """
    if label.startswith("trustgraph(descent,mclip,t=0.003,val=iid)["):
        return "tg_stab"
    if label.startswith("trustgraph(descent,mclip,t=0.003,a=0.3,val=iid)"):
        return "tg_a03"
    if label.startswith("trustgraph(descent,mclip,t=0.003,a=0.7,val=iid)"):
        return "tg_a07"
    if label.startswith("trustgraph(descent,euclidean"):
        return "tg_euclidean"
    if label.startswith("trustgraph(descent,norelu"):
        return "tg_no_relu"
    return None


def _index_summary_rows(rows: list[dict[str, object]]) -> dict[tuple[str, str, int, str], dict[str, object]]:
    """Index 50k summary rows by (topology, attack, byz_pct, variant_key).

    Args:
        rows: List of summary row dictionaries from phase1_paper_match_50k_3seed.json.

    Returns:
        Lookup dictionary keyed by (topology, attack, byz_pct, variant_key).
    """
    out: dict[tuple[str, str, int, str], dict[str, object]] = {}
    for r in rows:
        mk = _parse_variant_key(str(r["method"]))
        if mk is None:
            continue
        out[(str(r["topology"]), str(r["attack"]), int(r["byz_pct"]), mk)] = r
    return out


def _select_run_by_suffix(runs: list[dict[str, object]], suffix: str) -> dict[str, object]:
    """Return the first run record whose run_name ends with suffix.

    Args:
        runs: List of run dictionaries from phase1_fpr_rxdelta.json.
        suffix: Trailing run_name path suffix to match.

    Returns:
        Matching run dictionary.
    """
    for r in runs:
        if str(r["run_name"]).endswith(suffix):
            return r
    raise KeyError(f"Run ending with {suffix!r} not found.")


def _select_baseline_row(data: dict[str, object], attack: str) -> dict[str, object]:
    """Return the 3-seed 30% Byzantine baseline row matching attack from rxdelta JSON data.

    Args:
        data: Parsed phase1_fpr_rxdelta.json dictionary.
        attack: Attack name to match at 30% Byzantine fraction.

    Returns:
        Matching baseline summary row dictionary.
    """
    for r in data["baseline_50k_3seed"]["rows"]:
        if str(r["attack"]) == attack and int(r["byz_pct"]) == 30:
            return r
    raise KeyError(f"Baseline 3-seed row for attack={attack!r} not found.")


def plot_attack_accuracy(
    rows_idx: dict[tuple[str, str, int, str], dict[str, object]],
    data: dict[str, object],
    out_dir: Path,
) -> None:
    """Plot convergence and final accuracy across all four attack regimes (Figure 1).

    Args:
        rows_idx: Indexed 50k summary rows from _index_summary_rows().
        data: Full parsed phase1_paper_match_50k_3seed.json dataset.
        out_dir: Output directory for PNG and PDF figures.

    Returns:
        None.
    """
    curves = data["curves"]
    fig, (ax_conv, ax_bar) = plt.subplots(1, 2, figsize=(15.4, 6.4))
    fs.light_grid(ax_conv, "y")
    fs.light_grid(ax_bar, "y")

    logger.info("Fig. 1(a) TrustGraph-DFL Consensus Convergence Across Attacks (3 seeds)")
    for atk in fr.ATTACK_ORDER:
        c = curves[f"{atk}__tg_strong"]
        rounds = np.array(c["eval_rounds"], dtype=float)
        mean = np.array(c["cons_mean"], dtype=float)
        std = np.array(c["cons_std"], dtype=float)
        col = fr.ATTACK_COLOR[atk]
        ax_conv.plot(
            rounds,
            mean,
            label=fr.ATTACK_LABEL[atk],
            color=col,
            linestyle=fr.ATTACK_LINESTYLE[atk],
            marker=fr.ATTACK_MARKER[atk],
            markersize=7.5,
            linewidth=2.6,
            zorder=4 if atk == "none" else 5,
        )
        ax_conv.fill_between(rounds, np.clip(mean - std, 0, 100), np.clip(mean + std, 0, 100), color=col, alpha=0.14)
        pts = " ".join(f"r{int(r)}={v:.2f}" for r, v in zip(rounds, mean))
        logger.info("  {:16s}: {} (final +/- {:.2f})", atk, pts, std[-1])

    ax_conv.set_title("(a) Consensus Convergence Over Training", pad=10, fontweight="bold")
    ax_conv.set_xlabel("Communication Round")
    ax_conv.set_ylabel("Consensus Test Accuracy (%)")
    ax_conv.set_xlim(0.5, 25.5)
    ax_conv.set_xticks([1, 5, 10, 15, 20, 25])
    ax_conv.set_ylim(35, 98)
    ax_conv.legend(loc="lower right", ncol=1)

    short_labels = ["No Attack\n(0% Byz)", "Label Flip\n($E_b=2$)", "Model Poison\n($\\lambda=10$)", "Random Noise\n($\\sigma=0.5$)"]
    x = np.arange(len(fr.ATTACK_ORDER), dtype=float)
    w = 0.34
    peer_m = [float(rows_idx[("random", a, 30, "tg_stab")]["acc_mean"]) for a in fr.ATTACK_ORDER]
    peer_s = [float(rows_idx[("random", a, 30, "tg_stab")]["acc_std"]) for a in fr.ATTACK_ORDER]
    cons_m = [float(rows_idx[("random", a, 30, "tg_stab")]["cons_mean"]) for a in fr.ATTACK_ORDER]
    cons_s = [float(rows_idx[("random", a, 30, "tg_stab")]["cons_std"]) for a in fr.ATTACK_ORDER]

    logger.info("Fig. 1(b) TrustGraph-DFL Per-Peer vs. Consensus Accuracy (%)")
    for a, pm, ps, cm, cs in zip(fr.ATTACK_ORDER, peer_m, peer_s, cons_m, cons_s):
        logger.info("  {:16s}: peer={:.2f}+/-{:.2f}%  cons={:.2f}+/-{:.2f}%", a, pm, ps, cm, cs)

    b1 = ax_bar.bar(
        x - w / 2, peer_m, width=w, yerr=peer_s, capsize=4,
        color=fs.PALETTE["blue_secondary"], edgecolor="black", linewidth=1.6,
        error_kw={"elinewidth": 1.6, "ecolor": "black"}, label="Per-Peer Mean", zorder=3,
    )
    b2 = ax_bar.bar(
        x + w / 2, cons_m, width=w, yerr=cons_s, capsize=4,
        color=fs.PALETTE["blue_main"], edgecolor="black", linewidth=1.6,
        error_kw={"elinewidth": 1.6, "ecolor": "black"}, label="Consensus Average", zorder=3,
    )
    for b, m, s in zip(b1, peer_m, peer_s):
        ax_bar.text(b.get_x() + b.get_width() / 2.0, m + s + 1.0, f"{m:.1f}", ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=fs.PALETTE["blue_secondary"])
    for b, m, s in zip(b2, cons_m, cons_s):
        ax_bar.text(b.get_x() + b.get_width() / 2.0, m + s + 1.0, f"{m:.1f}", ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=fs.PALETTE["blue_main"])

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(short_labels)
    ax_bar.set_ylabel("Final Test Accuracy (%)")
    ax_bar.set_title("(b) Final Accuracy by Attack Regime", pad=10, fontweight="bold")
    ax_bar.set_ylim(60, 108)
    ax_bar.legend(loc="upper center", ncol=2, columnspacing=1.4)

    top = fs.header(
        fig,
        "TrustGraph-DFL Robustness Across Attack Regimes (Full 50k CIFAR-10)",
        "Pretrained ResNet-18 (2,500/node, Dir alpha=0.5, iid val=250, random ER p=0.3, 30% Byz, 25 rounds, 3 seeds; 0 NaNs)",
    )
    bot = fs.footnote(
        fig,
        "Stabilized TrustGraph-DFL (descent, mclip 2*r_i, tau=0.003, gc=10) sustains 91.5-92.9% consensus and 88.9-91.4% per-peer accuracy across all regimes.",
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=top - 0.08, bottom=bot + 0.90 / 6.4, wspace=0.22)
    fs.save(fig, out_dir, "fig1-attacks-50k")


def plot_topology_and_byzantine_sweep(
    rows_idx: dict[tuple[str, str, int, str], dict[str, object]],
    out_dir: Path,
) -> None:
    """Plot topology accuracy comparison and 10%-50% Byzantine stress sweep (Figure 2).

    Args:
        rows_idx: Indexed 50k summary rows from _index_summary_rows().
        out_dir: Output directory for PNG and PDF figures.

    Returns:
        None.
    """
    fig, (ax_topo, ax_sweep) = plt.subplots(1, 2, figsize=(15.4, 6.4))
    fs.light_grid(ax_topo, "y")
    fs.light_grid(ax_sweep, "y")

    topos = [("ring", "Ring\n(deg = 2)"), ("random", "Random\n(ER p = 0.3)"), ("small_world", "Small-World\n(WS k=4, p=0.3)")]
    x = np.arange(len(topos), dtype=float)
    w = 0.34
    p_m = [float(rows_idx[(tp, "label_flip_all", 30, "tg_stab")]["acc_mean"]) for tp, _ in topos]
    p_s = [float(rows_idx[(tp, "label_flip_all", 30, "tg_stab")]["acc_std"]) for tp, _ in topos]
    c_m = [float(rows_idx[(tp, "label_flip_all", 30, "tg_stab")]["cons_mean"]) for tp, _ in topos]
    c_s = [float(rows_idx[(tp, "label_flip_all", 30, "tg_stab")]["cons_std"]) for tp, _ in topos]

    logger.info("Fig. 2(a) TrustGraph-DFL Topologies under label_flip_all (30% Byz, bep=2, 3 seeds)")
    for (tp, _), pm, ps, cm, cs in zip(topos, p_m, p_s, c_m, c_s):
        logger.info("  {:12s}: peer={:.2f}+/-{:.2f}%  cons={:.2f}+/-{:.2f}%", tp, pm, ps, cm, cs)

    b1 = ax_topo.bar(
        x - w / 2, p_m, width=w, yerr=p_s, capsize=4,
        color=fs.PALETTE["blue_secondary"], edgecolor="black", linewidth=1.6,
        error_kw={"elinewidth": 1.6, "ecolor": "black"}, label="Per-Peer Mean", zorder=3,
    )
    b2 = ax_topo.bar(
        x + w / 2, c_m, width=w, yerr=c_s, capsize=4,
        color=fs.PALETTE["blue_main"], edgecolor="black", linewidth=1.6,
        error_kw={"elinewidth": 1.6, "ecolor": "black"}, label="Consensus Average", zorder=3,
    )
    for b, m, s in zip(b1, p_m, p_s):
        ax_topo.text(b.get_x() + b.get_width() / 2.0, m + s + 1.0, f"{m:.1f}", ha="center", va="bottom", fontsize=12.0, fontweight="bold", color=fs.PALETTE["blue_secondary"])
    for b, m, s in zip(b2, c_m, c_s):
        ax_topo.text(b.get_x() + b.get_width() / 2.0, m + s + 1.0, f"{m:.1f}", ha="center", va="bottom", fontsize=12.0, fontweight="bold", color=fs.PALETTE["blue_main"])

    ax_topo.set_xticks(x)
    ax_topo.set_xticklabels([lbl for _, lbl in topos])
    ax_topo.set_ylabel("Final Test Accuracy (%)")
    ax_topo.set_title("(a) Topology Robustness (30% Label Flip)", pad=10, fontweight="bold")
    ax_topo.set_ylim(55, 108)
    ax_topo.legend(loc="upper center", ncol=2, columnspacing=1.4)

    fracs = [10, 20, 30, 40, 50]
    cons_m = np.array([float(rows_idx[("random", "label_flip_all", f, "tg_stab")]["cons_mean"]) for f in fracs])
    cons_sd = np.array([float(rows_idx[("random", "label_flip_all", f, "tg_stab")]["cons_std"]) for f in fracs])
    peer_m = np.array([float(rows_idx[("random", "label_flip_all", f, "tg_stab")]["acc_mean"]) for f in fracs])
    peer_sd = np.array([float(rows_idx[("random", "label_flip_all", f, "tg_stab")]["acc_std"]) for f in fracs])
    fpr_m = np.array([float(rows_idx[("random", "label_flip_all", f, "tg_stab")]["fpr_mean"]) for f in fracs])
    fpr_sd = np.array([float(rows_idx[("random", "label_flip_all", f, "tg_stab")]["fpr_std"]) for f in fracs])

    logger.info("Fig. 2(b) TrustGraph-DFL Byzantine Stress Sweep (10%-50%, label_flip_all, bep=2, 3 seeds)")
    for f, cm, cs, pm, ps, fm, fs_ in zip(fracs, cons_m, cons_sd, peer_m, peer_sd, fpr_m, fpr_sd):
        logger.info("  byz={}%: cons={:.2f}+/-{:.2f}%  peer={:.2f}+/-{:.2f}%  fpr={:.2f}+/-{:.2f}%", f, cm, cs, pm, ps, fm, fs_)

    for arr, sd, lbl, col, ls, mk in (
        (cons_m, cons_sd, "Consensus Acc (%)", fs.PALETTE["blue_main"], "-", "o"),
        (peer_m, peer_sd, "Per-Peer Mean Acc (%)", fs.PALETTE["blue_secondary"], "--", "s"),
        (fpr_m, fpr_sd, "Cumulative FPR (%)", fs.PALETTE["teal"], "-.", "D"),
    ):
        ax_sweep.plot(fracs, arr, label=lbl, color=col, linestyle=ls, marker=mk, markersize=8.0, linewidth=2.6, zorder=4)
        ax_sweep.fill_between(fracs, np.clip(arr - sd, 0, 100), np.clip(arr + sd, 0, 100), color=col, alpha=0.14)

    ax_sweep.set_title("(b) Accuracy & FPR vs. Byzantine Fraction", pad=10, fontweight="bold")
    ax_sweep.set_xlabel("Byzantine Peer Fraction (%)")
    ax_sweep.set_ylabel("Percentage (%)")
    ax_sweep.set_xticks(fracs)
    ax_sweep.set_xlim(7, 53)
    ax_sweep.set_ylim(25, 102)
    ax_sweep.legend(loc="center right", ncol=1)

    top = fs.header(
        fig,
        "TrustGraph-DFL Topology Robustness and Byzantine Stress Sweep (Full 50k CIFAR-10)",
        "Pretrained ResNet-18 (2,500/node, Dir alpha=0.5, iid val=250, label_flip_all E_b=2, 25 rounds, 3 seeds)",
    )
    bot = fs.footnote(
        fig,
        "TrustGraph-DFL reaches 91.7-92.4% consensus across all three topologies and holds 89.5% peer / 92.0% consensus at 50% Byzantine while FPR drops to 36.2%.",
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=top - 0.08, bottom=bot + 0.90 / 6.4, wspace=0.22)
    fs.save(fig, out_dir, "fig2-topologies-and-stress-50k")


def plot_ablations_and_fpr_dynamics(
    rows_idx: dict[tuple[str, str, int, str], dict[str, object]],
    data: dict[str, object],
    out_dir: Path,
) -> None:
    """Plot component ablations under model poisoning and per-round FPR trajectory (Figure 3).

    Args:
        rows_idx: Indexed 50k summary rows from _index_summary_rows().
        data: Full parsed phase1_paper_match_50k_3seed.json dataset.
        out_dir: Output directory for PNG and PDF figures.

    Returns:
        None.
    """
    fig, (ax_abl, ax_dyn) = plt.subplots(1, 2, figsize=(14.8, 6.4))
    fs.light_grid(ax_abl, "y")
    fs.light_grid(ax_dyn, "y")

    abl_keys = [
        ("tg_stab", "Full\n($\\alpha=0.5$)"),
        ("tg_a03", "$\\alpha=0.3$"),
        ("tg_a07", "$\\alpha=0.7$"),
        ("tg_euclidean", "Euclidean\nMetric"),
        ("tg_no_relu", "No ReLU\nZeroing"),
    ]
    x = np.arange(len(abl_keys), dtype=float)
    means = [float(rows_idx[("random", "model_poison", 30, k)]["cons_mean"]) for k, _ in abl_keys]
    stds = [float(rows_idx[("random", "model_poison", 30, k)]["cons_std"]) for k, _ in abl_keys]
    colors = [fr.ABLATION_COLOR[k] for k, _ in abl_keys]

    logger.info("Fig. 3(a) Section 5.5 Ablations under model_poison (3 seeds)")
    bars = ax_abl.bar(
        x, means, width=0.58, yerr=stds, capsize=5,
        color=colors, edgecolor="black", linewidth=1.8,
        error_kw={"elinewidth": 1.8, "ecolor": "black"}, zorder=3,
    )
    for b, (k, _), m, s in zip(bars, abl_keys, means, stds):
        ax_abl.text(
            b.get_x() + b.get_width() / 2.0,
            min(m + s + 1.8, 102.0),
            f"{m:.1f}%",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=fs.INK,
        )
        logger.info("  {:14s}: cons={:.2f} +/- {:.2f}%", k, m, s)

    ax_abl.set_xticks(x)
    ax_abl.set_xticklabels([lbl for _, lbl in abl_keys])
    ax_abl.set_ylabel("Final Consensus Test Accuracy (%)")
    ax_abl.set_title("(a) Section 5.5 Ablations (Model Poisoning)", pad=10, fontweight="bold")
    ax_abl.set_ylim(0, 108)

    curves = data["curves"]
    logger.info("Fig. 3(b) Per-Round FPR Trajectory for TrustGraph-DFL")
    for atk in fr.ATTACK_ORDER:
        c = curves[f"{atk}__tg_strong"]
        rds = np.array(c["fpr_rounds"], dtype=float)
        fm = np.array(c["fpr_mean"], dtype=float)
        fsd = np.array(c["fpr_std"], dtype=float)
        col = fr.ATTACK_COLOR[atk]
        ax_dyn.plot(
            rds, fm,
            label=fr.ATTACK_LABEL[atk],
            color=col,
            linestyle=fr.ATTACK_LINESTYLE[atk],
            marker=fr.ATTACK_MARKER[atk],
            markevery=[0, 4, 9, 14, 19, 24],
            markersize=7,
            linewidth=2.4,
        )
        ax_dyn.fill_between(rds, np.clip(fm - fsd, 0, 100), np.clip(fm + fsd, 0, 100), color=col, alpha=0.12)
        logger.info("  {:16s}: r1={:.1f}% r5={:.1f}% r15={:.1f}% r25={:.1f}%", atk, fm[0], fm[4], fm[14], fm[-1])

    ax_dyn.set_title("(b) Per-Round FPR Climbs Near Convergence", pad=10, fontweight="bold")
    ax_dyn.set_xlabel("Communication Round")
    ax_dyn.set_ylabel("Per-Round FPR (%)")
    ax_dyn.set_xlim(0.5, 25.5)
    ax_dyn.set_xticks([1, 5, 10, 15, 20, 25])
    ax_dyn.set_ylim(0, 75)
    ax_dyn.legend(loc="lower right", ncol=1)

    top = fs.header(
        fig,
        "TrustGraph-DFL Ablations and Per-Round FPR Dynamics (Full 50k CIFAR-10)",
        "Pretrained ResNet-18 (2,500/node, Dir alpha=0.5, random ER p=0.3, 30% Byzantine, 25 rounds, 3 seeds)",
    )
    bot = fs.footnote(
        fig,
        "Left: Removing ReLU drops consensus by 35.9 pp; Euclidean drops 9.0 pp. Right: FPR starts at 0% at round 1 and climbs near convergence.",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=top - 0.08, bottom=bot + 0.90 / 6.4, wspace=0.22)
    fs.save(fig, out_dir, "fig3-ablations-and-fpr-dynamics-50k")


def plot_receiver_anchored_fpr(data: dict[str, object], out_dir: Path) -> None:
    """Plot sender vs. receiver-anchored (rxdelta) FPR trade-off and trajectory (Figure 4).

    Args:
        data: Full parsed phase1_fpr_rxdelta.json dataset.
        out_dir: Output directory for PNG and PDF figures.

    Returns:
        None.
    """
    runs: list[dict[str, object]] = data["runs"]
    base_mp = _select_baseline_row(data, "model_poison")
    base_curve = data["baseline_50k_3seed"]["curves"]["model_poison__tg_strong"]

    rx_t003_mp = _select_run_by_suffix(runs, "fig3b_fpr_rxdelta/model_poison__iidval")
    rx_t020_mp = _select_run_by_suffix(runs, "fig3b_fpr_rxdelta_tau/t020__model_poison")
    rx_t035_mp = _select_run_by_suffix(runs, "fig3b_fpr_rxdelta_tau/t035__model_poison")

    fig, (ax_trade, ax_traj) = plt.subplots(1, 2, figsize=(15.6, 6.5))
    fs.light_grid(ax_trade, "y")
    fs.light_grid(ax_traj, "y")

    ax_trade.axvspan(3.0, 9.0, color=fs.PALETTE["green_1"], alpha=0.65, zorder=1)
    ax_trade.text(6.0, 48.5, "Paper\n3-9%\ntarget", ha="center", va="top", fontsize=11.5, fontweight="bold", color=fs.INK)

    base_pt = (float(base_mp["fpr_mean"]), float(base_mp["byz_accept_mean"]))
    rx_pts = [
        (r"$\tau=0.003$", 100.0 * float(rx_t003_mp["fpr_cumulative"]), 100.0 * float(rx_t003_mp["byz_accept_cumulative"]), (10, 5)),
        (r"$\tau=0.020$", 100.0 * float(rx_t020_mp["fpr_cumulative"]), 100.0 * float(rx_t020_mp["byz_accept_cumulative"]), (10, 5)),
        (r"$\tau=0.035$", 100.0 * float(rx_t035_mp["fpr_cumulative"]), 100.0 * float(rx_t035_mp["byz_accept_cumulative"]), (-6, -22)),
    ]

    logger.info("Fig. 4(a) Sender vs. rxdelta Trade-off on model_poison (30% Byz, 50k CIFAR-10)")
    logger.info("  Base sender (3-seed)     : FPR={:5.2f}%, ByzAcc={:5.2f}%", base_pt[0], base_pt[1])

    ax_trade.scatter(
        [base_pt[0]], [base_pt[1]],
        s=140, color=fs.PALETTE["gray"], edgecolor="black", linewidth=1.5, marker="o", zorder=5,
        label="Base sender (3-seed)",
    )
    ax_trade.annotate(
        "Base sender\n(3-seed)", xy=base_pt, xytext=(-56, 22), textcoords="offset points",
        fontsize=11.5, fontweight="bold", color=fs.INK,
        arrowprops={"arrowstyle": "-", "color": fs.PALETTE["gray"], "lw": 1.2},
    )

    ax_trade.plot(
        [p[1] for p in rx_pts], [p[2] for p in rx_pts],
        color=fs.PALETTE["blue_main"], linestyle="-", marker="s", markersize=9.0, linewidth=2.8, zorder=6,
        label="rxdelta (receiver-anchored)",
    )
    for lbl, x_v, y_v, off in rx_pts:
        logger.info("  rxdelta {:16s} : FPR={:5.2f}%, ByzAcc={:5.2f}%", lbl, x_v, y_v)
        ax_trade.annotate(lbl, xy=(x_v, y_v), xytext=off, textcoords="offset points", fontsize=12.0, color=fs.PALETTE["blue_main"], fontweight="bold")

    ax_trade.set_title("(a) FPR vs. Byzantine Acceptance (Model Poisoning)", pad=10, fontweight="bold")
    ax_trade.set_xlabel("Cumulative FPR (%)")
    ax_trade.set_ylabel("Byzantine Acceptance Rate (%)")
    ax_trade.set_xlim(1, 52)
    ax_trade.set_ylim(20, 50)
    ax_trade.legend(loc="lower left", ncol=1)

    ax_traj.axhspan(3.0, 9.0, color=fs.PALETTE["green_1"], alpha=0.65, zorder=1)
    rounds_b = np.array(base_curve["fpr_rounds"], dtype=float)
    fpr_b_m = np.array(base_curve["fpr_mean"], dtype=float)
    fpr_b_s = np.array(base_curve["fpr_std"], dtype=float)

    def _extract_fpr_curve(run: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
        """Extract (1-based round indices, per-round FPR %) arrays from a run history.

        Args:
            run: Run dictionary containing a 'history' list of per-round records.

        Returns:
            Tuple of (rounds_array, fpr_percent_array) as NumPy float arrays.
        """
        h = run["history"]
        return np.array([float(x["round"]) + 1.0 for x in h]), np.array([100.0 * float(x["fpr_round"]) for x in h])

    r_rx003, fpr_rx003 = _extract_fpr_curve(rx_t003_mp)
    r_rx020, fpr_rx020 = _extract_fpr_curve(rx_t020_mp)
    r_rx035, fpr_rx035 = _extract_fpr_curve(rx_t035_mp)
    marks = [0, 4, 9, 14, 19, 24]

    ax_traj.plot(
        rounds_b, fpr_b_m, label="Base sender (3-seed)",
        color=fs.PALETTE["gray"], linestyle="-", marker="o", markevery=marks, markersize=7.5, linewidth=2.4, zorder=3,
    )
    ax_traj.fill_between(rounds_b, np.clip(fpr_b_m - fpr_b_s, 0, 100), np.clip(fpr_b_m + fpr_b_s, 0, 100), color=fs.PALETTE["gray"], alpha=0.14)
    ax_traj.plot(
        r_rx035, fpr_rx035, label=r"rxdelta ($\tau=0.035$)",
        color=fs.PALETTE["teal"], linestyle="--", marker="D", markevery=marks, markersize=7.0, linewidth=2.4, zorder=4,
    )
    ax_traj.plot(
        r_rx020, fpr_rx020, label=r"rxdelta ($\tau=0.020$)",
        color=fs.PALETTE["blue_secondary"], linestyle="-.", marker="^", markevery=marks, markersize=7.5, linewidth=2.4, zorder=5,
    )
    ax_traj.plot(
        r_rx003, fpr_rx003, label=r"rxdelta ($\tau=0.003$)",
        color=fs.PALETTE["blue_main"], linestyle="-", marker="s", markevery=marks, markersize=7.5, linewidth=2.8, zorder=6,
    )

    logger.info("Fig. 4(b) Per-Round FPR Trajectory on model_poison (r=1, 5, 10, 15, 25)")
    for name, arr in (
        ("Base sender (3-seed)", fpr_b_m),
        ("rxdelta (tau=0.035)", fpr_rx035),
        ("rxdelta (tau=0.020)", fpr_rx020),
        ("rxdelta (tau=0.003)", fpr_rx003),
    ):
        logger.info("  {:24s}: r1={:5.1f}% r5={:5.1f}% r10={:5.1f}% r15={:5.1f}% r25={:5.1f}%", name, arr[0], arr[4], arr[9], arr[14], arr[-1])

    ax_traj.set_title("(b) Per-Round FPR Over Training", pad=10, fontweight="bold")
    ax_traj.set_xlabel("Communication Round")
    ax_traj.set_ylabel("Per-Round FPR (%)")
    ax_traj.set_xlim(0.5, 25.5)
    ax_traj.set_xticks([1, 5, 10, 15, 20, 25])
    ax_traj.set_ylim(-3, 84)
    ax_traj.legend(loc="upper left", ncol=1)

    top = fs.header(
        fig,
        "Receiver-Anchored Scoring (rxdelta) on TrustGraph-DFL (Full 50k CIFAR-10)",
        "Pretrained ResNet-18 (2,500/node, Dir alpha=0.5, iid val=250, random ER p=0.3, 30% model poisoning, 25 rounds)",
    )
    bot = fs.footnote(
        fig,
        "rxdelta (tau=0.003) brings FPR from 43.9% to 7.4% (inside the 3-9% paper target); tau=0.035 beats sender on both FPR (25.7%) and Byz accept (31.7%).",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=top - 0.08, bottom=bot + 0.90 / 6.5, wspace=0.22)
    fs.save(fig, out_dir, "fig4-fpr-rxdelta-50k")


def plot_v100_bandwidth_waste(data: dict[str, object], out_dir: Path) -> None:
    """Plot real multi-node V100 received vs. wasted wire traffic per round (Figure 5).

    Args:
        data: Full parsed phase1_v100_distributed_2node.json dataset.
        out_dir: Output directory for PNG and PDF figures.

    Returns:
        None.
    """
    runs_by_label = {str(r["label"]): r for r in data["runs"]}
    hw = data.get("hardware", {})
    gpu_name = str(hw.get("gpu", "Tesla V100-PCIE-16GB"))
    iface = str(hw.get("interface", "enp68s0f0"))
    speed = hw.get("link_speed_mbps", 1000)
    mtu = hw.get("mtu", 1500)
    nodes = "+".join(str(n) for n in data.get("nodes", ["ja3", "ja4"]))

    def _extract_wire_mb(specs: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
        """Extract per-peer useful, false-positive waste, and attacker waste wire MB arrays.

        Args:
            specs: Ordered list of (run_label, tick_label) tuples.

        Returns:
            Tuple of (useful_mb, fp_waste_mb, byz_waste_mb, waste_pcts).
        """
        use_l, fp_l, byz_l, wp_l = [], [], [], []
        for key, _ in specs:
            r = runs_by_label[key]
            sm = r["system_metrics"]
            wire_mb = float(r["wire_bytes_per_neighbor"]) / 1e6
            deg = float(sm["mean_benign_degree"])
            tot_tx = max(int(sm["total_transfers"]), 1)
            hon_rej = int(sm["honest_rejected_transfers"])
            byz_rej = int(sm["byz_rejected_transfers"])
            tot_rej = int(sm["total_rejected_transfers"])
            kept = tot_tx - tot_rej
            rx_mb = wire_mb * deg
            u_mb = rx_mb * kept / tot_tx
            fp_mb = rx_mb * hon_rej / tot_tx
            bz_mb = rx_mb * byz_rej / tot_tx
            wp = 100.0 * tot_rej / tot_tx
            use_l.append(u_mb)
            fp_l.append(fp_mb)
            byz_l.append(bz_mb)
            wp_l.append(wp)
            logger.info(
                "  {:24s}: rx={:6.2f} MB | useful={:6.2f} | fp_waste={:5.2f} | byz_waste={:5.2f} ({:5.1f}% wasted)",
                key,
                rx_mb,
                u_mb,
                fp_mb,
                bz_mb,
                wp,
            )
        return np.array(use_l), np.array(fp_l), np.array(byz_l), wp_l

    fig, (ax_4p, ax_2p) = plt.subplots(1, 2, figsize=(16.4, 6.6), gridspec_kw={"width_ratios": [1.45, 1.0]})
    fs.light_grid(ax_4p, "y")
    fs.light_grid(ax_2p, "y")

    specs_4p = [
        ("4p_none_sender", "No Attack\n(sender)"),
        ("4p_label_flip_sender", "Label Flip\n(sender)"),
        ("4p_model_poison_sender", "Model Poison\n(sender)"),
        ("4p_random_noise_sender", "Random Noise\n(sender)"),
        ("4p_model_poison_rxdelta", "Model Poison\n(rxdelta)"),
    ]
    logger.info("Fig. 5(a) Real 4-Peer V100 Wire Receive Traffic per Peer / Round (MB)")
    u4, f4, b4, w4 = _extract_wire_mb(specs_4p)
    x4 = np.arange(len(specs_4p), dtype=float)
    bw4 = 0.54
    ax_4p.bar(x4, u4, width=bw4, color=fs.PALETTE["blue_main"], edgecolor="black", linewidth=1.6, label="Accepted (Useful)", zorder=3)
    ax_4p.bar(x4, f4, bottom=u4, width=bw4, color=fs.PALETTE["red_2"], edgecolor="black", linewidth=1.6, label="Wasted (False Positive)", zorder=3)
    ax_4p.bar(x4, b4, bottom=u4 + f4, width=bw4, color=fs.PALETTE["red_strong"], edgecolor="black", linewidth=1.6, label="Wasted (Blocked Attacker)", zorder=3)
    tot4 = u4 + f4 + b4
    for xi, tot, wp in zip(x4, tot4, w4):
        ax_4p.text(xi, tot + 7.0, f"{wp:.1f}%\nwasted", ha="center", va="bottom", fontsize=11.0, fontweight="bold", color=fs.PALETTE["red_strong"])
    ax_4p.set_xticks(x4)
    ax_4p.set_xticklabels([lbl for _, lbl in specs_4p], fontsize=12.0)
    ax_4p.set_ylabel("Received Wire Traffic / Round (MB)")
    ax_4p.set_title("(a) 4 Peers Across 2 V100 Nodes (Degree 3)", pad=10, fontweight="bold")
    ax_4p.set_ylim(0, max(tot4) * 1.44)
    ax_4p.legend(loc="upper center", ncol=3, fontsize=10.5, columnspacing=1.0, handletextpad=0.5)

    specs_2p = [
        ("2p_none_sender", "No Attack\n(sender)"),
        ("2p_model_poison_sender", "Model Poison\n(sender)"),
        ("2p_model_poison_rxdelta", "Model Poison\n(rxdelta)"),
    ]
    logger.info("Fig. 5(b) Real 2-Peer V100 Wire Receive Traffic per Peer / Round (MB)")
    u2, f2, b2, w2 = _extract_wire_mb(specs_2p)
    x2 = np.arange(len(specs_2p), dtype=float)
    bw2 = 0.50
    ax_2p.bar(x2, u2, width=bw2, color=fs.PALETTE["blue_main"], edgecolor="black", linewidth=1.6, label="Accepted (Useful)", zorder=3)
    ax_2p.bar(x2, f2, bottom=u2, width=bw2, color=fs.PALETTE["red_2"], edgecolor="black", linewidth=1.6, label="Wasted (False Positive)", zorder=3)
    ax_2p.bar(x2, b2, bottom=u2 + f2, width=bw2, color=fs.PALETTE["red_strong"], edgecolor="black", linewidth=1.6, label="Wasted (Blocked Attacker)", zorder=3)
    tot2 = u2 + f2 + b2
    for xi, tot, wp in zip(x2, tot2, w2):
        ax_2p.text(xi, tot + 2.5, f"{wp:.1f}%\nwasted", ha="center", va="bottom", fontsize=11.0, fontweight="bold", color=fs.PALETTE["red_strong"])
    ax_2p.set_xticks(x2)
    ax_2p.set_xticklabels([lbl for _, lbl in specs_2p], fontsize=12.0)
    ax_2p.set_ylabel("Received Wire Traffic / Round (MB)")
    ax_2p.set_title("(b) Dedicated 1 Peer / V100 (Degree 1)", pad=10, fontweight="bold")
    ax_2p.set_ylim(0, max(tot2) * 1.48)
    ax_2p.legend(loc="upper right", ncol=1, fontsize=10.5)

    top = fs.header(
        fig,
        "Real Multi-Node V100 Received vs. Wasted FP32 Wire Traffic",
        f"2x {gpu_name} ({nodes}, {iface} {speed} Mb/s MTU {mtu}, ResNet-18, 2,500 images/peer, 25 rounds, seed 42)",
    )
    bot = fs.footnote(
        fig,
        "Left: 4 peers discard 34.3-52.4% of real TCP wire bytes (268.4 MB/round in sender, 134.2 MB in rxdelta). Right: 2 peers discard 26% (no attack) to 100% (model poison).",
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=top - 0.08, bottom=bot + 0.90 / 6.6, wspace=0.20)
    fs.save(fig, out_dir, "fig5-v100-bandwidth-waste")


def plot_v100_distributed_ethernet(data: dict[str, object], out_dir: Path) -> None:
    """Plot real multi-node V100 training compute vs. 1 Gb/s Ethernet transfer times (Figure 6).

    Args:
        data: Full parsed phase1_v100_distributed_2node.json dataset.
        out_dir: Output directory for PNG and PDF figures.

    Returns:
        None.
    """
    runs_by_label = {str(r["label"]): r for r in data["runs"]}
    hw = data.get("hardware", {})
    gpu_name = str(hw.get("gpu", "Tesla V100-PCIE-16GB"))
    iface = str(hw.get("interface", "enp68s0f0"))
    speed = hw.get("link_speed_mbps", 1000)
    mtu = hw.get("mtu", 1500)
    nodes = "+".join(str(n) for n in data.get("nodes", ["ja3", "ja4"]))

    def _extract_bars(specs: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], list[float]]:
        """Extract compute, useful transfer, wasted transfer, comm share, and waste % arrays.

        Args:
            specs: Ordered list of (run_label, tick_label) tuples.

        Returns:
            Tuple of (compute_arr, useful_tx_arr, wasted_tx_arr, comm_shares, waste_pcts).
        """
        comps, tx_use, tx_was, shares, wastes = [], [], [], [], []
        for key, _ in specs:
            r = runs_by_label[key]
            sm = r["system_metrics"]
            mt = sm["measured_transport_sec"]
            c = (
                float(sm["train_sec_per_peer"]["median"])
                + float(sm["val_grad_sec_per_peer"]["median"])
                + float(sm["agg_sec_per_peer"]["median"])
            )
            tx = (
                float(mt["d2h_sec_per_peer"]["median"])
                + float(mt["net_sec_per_peer"]["median"])
                + float(mt["h2d_sec_per_peer"]["median"])
            )
            wp = float(sm["wasted_transfer_pct"])
            u = tx * (1.0 - wp / 100.0)
            w = tx * (wp / 100.0)
            tot = c + tx
            sh = 100.0 * tx / max(tot, 1e-9)
            comps.append(c)
            tx_use.append(u)
            tx_was.append(w)
            shares.append(sh)
            wastes.append(wp)
            logger.info(
                "  {:24s}: compute={:.3f}s | tx={:.3f}s (net={:.3f}s, use={:.3f}s, waste={:.3f}s) | total={:.3f}s ({:.1f}% comm, {:.1f}% wasted)",
                key,
                c,
                tx,
                float(mt["net_sec_per_peer"]["median"]),
                u,
                w,
                tot,
                sh,
                wp,
            )
        return np.array(comps), np.array(tx_use), np.array(tx_was), shares, wastes

    fig, (ax_2p, ax_4p) = plt.subplots(1, 2, figsize=(16.4, 6.6), gridspec_kw={"width_ratios": [1.0, 1.45]})
    fs.light_grid(ax_2p, "y")
    fs.light_grid(ax_4p, "y")

    specs_2p = [
        ("2p_none_sender", "No Attack\n(sender, 89.5 MB)"),
        ("2p_model_poison_sender", "Model Poison\n(sender, 89.5 MB)"),
        ("2p_model_poison_rxdelta", "Model Poison\n(rxdelta, 44.7 MB)"),
    ]
    logger.info("Fig. 6(a) Real 2-Peer V100 Distributed Training (1 Dedicated V100 + NIC / Peer)")
    c2, u2, w2, sh2, _ = _extract_bars(specs_2p)
    x2 = np.arange(len(specs_2p), dtype=float)
    bw2 = 0.50
    ax_2p.bar(x2, c2, width=bw2, color=fs.PALETTE["gray"], edgecolor="black", linewidth=1.6, label="Measured V100 Compute", zorder=3)
    ax_2p.bar(x2, u2, bottom=c2, width=bw2, color=fs.PALETTE["blue_main"], edgecolor="black", linewidth=1.6, label="Useful 1 Gb/s Transfer", zorder=3)
    ax_2p.bar(x2, w2, bottom=c2 + u2, width=bw2, color=fs.PALETTE["red_strong"], edgecolor="black", linewidth=1.6, label="Wasted 1 Gb/s Transfer", zorder=3)
    t2 = c2 + u2 + w2
    for xi, tot, sh in zip(x2, t2, sh2):
        ax_2p.text(xi, tot + 0.05, f"{tot:.2f}s\n({sh:.1f}% comm)", ha="center", va="bottom", fontsize=11.0, fontweight="bold", color=fs.INK)
    ax_2p.set_xticks(x2)
    ax_2p.set_xticklabels([lbl for _, lbl in specs_2p], fontsize=12.0)
    ax_2p.set_ylabel("Measured Per-Peer Round Time (seconds)")
    ax_2p.set_title("(a) Dedicated 1 Peer / V100 (Degree 1)", pad=10, fontweight="bold")
    ax_2p.set_ylim(0, max(t2) * 1.46)
    ax_2p.legend(loc="upper right", ncol=1, fontsize=10.5)

    specs_4p = [
        ("4p_none_sender", "No Attack\n(sender)"),
        ("4p_label_flip_sender", "Label Flip\n(sender)"),
        ("4p_model_poison_sender", "Model Poison\n(sender)"),
        ("4p_random_noise_sender", "Random Noise\n(sender)"),
        ("4p_model_poison_rxdelta", "Model Poison\n(rxdelta)"),
    ]
    logger.info("Fig. 6(b) Real 4-Peer V100 Distributed Training (2 Peers / V100, Degree 3)")
    c4, u4, w4, sh4, _ = _extract_bars(specs_4p)
    x4 = np.arange(len(specs_4p), dtype=float)
    bw4 = 0.54
    ax_4p.bar(x4, c4, width=bw4, color=fs.PALETTE["gray"], edgecolor="black", linewidth=1.6, label="Measured V100 Compute", zorder=3)
    ax_4p.bar(x4, u4, bottom=c4, width=bw4, color=fs.PALETTE["blue_main"], edgecolor="black", linewidth=1.6, label="Useful 1 Gb/s Transfer", zorder=3)
    ax_4p.bar(x4, w4, bottom=c4 + u4, width=bw4, color=fs.PALETTE["red_strong"], edgecolor="black", linewidth=1.6, label="Wasted 1 Gb/s Transfer", zorder=3)
    t4 = c4 + u4 + w4
    for xi, tot, sh in zip(x4, t4, sh4):
        ax_4p.text(xi, tot + 0.14, f"{tot:.2f}s\n({sh:.1f}% comm)", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=fs.INK)
    ax_4p.set_xticks(x4)
    ax_4p.set_xticklabels([lbl for _, lbl in specs_4p], fontsize=12.0)
    ax_4p.set_ylabel("Measured Per-Peer Round Time (seconds)")
    ax_4p.set_title("(b) 4 Peers Across 2 V100 Nodes (Degree 3)", pad=10, fontweight="bold")
    ax_4p.set_ylim(0, max(t4) * 1.44)
    ax_4p.legend(loc="upper center", ncol=3, fontsize=10.5, columnspacing=1.2, handletextpad=0.5)

    top = fs.header(
        fig,
        "Real Multi-Node V100 Distributed Training Over 1 Gb/s Ethernet",
        f"2x {gpu_name} ({nodes}, {iface} {speed} Mb/s MTU {mtu}, ResNet-18, 2,500 images/peer, 25 rounds, seed 42)",
    )
    bot = fs.footnote(
        fig,
        "Left: 1 dedicated V100 and NIC per peer (degree 1) spends 29-46% of each round on Ethernet. Right: 4 peers (degree 3) spend 60% in sender (3.13 s) and 43% in rxdelta (1.57 s).",
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=top - 0.08, bottom=bot + 0.90 / 6.6, wspace=0.20)
    fs.save(fig, out_dir, "fig6-v100-distributed-ethernet")


@app.command()
def plot_cli(
    data_50k: Annotated[Path, typer.Option(help="Committed 50k 3-seed reproduction JSON dataset.")] = Path("report/data/phase1_paper_match_50k_3seed.json"),
    data_rxdelta: Annotated[Path, typer.Option(help="Committed receiver-anchored FPR JSON dataset.")] = Path("report/data/phase1_fpr_rxdelta.json"),
    data_v100: Annotated[Path, typer.Option(help="Committed real multi-node V100 Ethernet JSON dataset.")] = Path("report/data/phase1_v100_distributed_2node.json"),
    out_dir: Annotated[Path, typer.Option(help="Output directory for PNG and header-free PDF figures.")] = Path("report/figures"),
    log_level: Annotated[Literal["DEBUG", "INFO", "WARNING", "ERROR"], typer.Option(help="Minimum loguru log level.")] = "INFO",
) -> None:
    """Load committed JSON datasets and generate all PNG and header-free PDF figures."""
    configure_logging(level=log_level)

    parsed_50k = json.loads(data_50k.read_text(encoding="utf-8"))
    parsed_rxdelta = json.loads(data_rxdelta.read_text(encoding="utf-8"))
    rows_idx = _index_summary_rows(parsed_50k["rows"])

    fs.apply("compact")
    plot_attack_accuracy(rows_idx, parsed_50k, out_dir)
    plot_topology_and_byzantine_sweep(rows_idx, out_dir)
    plot_ablations_and_fpr_dynamics(rows_idx, parsed_50k, out_dir)
    plot_receiver_anchored_fpr(parsed_rxdelta, out_dir)

    if data_v100.exists():
        parsed_v100 = json.loads(data_v100.read_text(encoding="utf-8"))
        plot_v100_bandwidth_waste(parsed_v100, out_dir)
        plot_v100_distributed_ethernet(parsed_v100, out_dir)


def main() -> None:
    """Invoke the Typer CLI application for generating publication figures.

    Returns:
        None.
    """
    app()


if __name__ == "__main__":
    main()
