"""Generate all final TrustGraph-DFL Phase 1 figures from committed JSON data.

Follows the figures4papers house style via ``figstyle.py`` (unedited skill copy)
and fixed series roles in ``figroles.py``. Prints every plotted number to stdout
and writes 300 dpi PNGs (with header/footnote) plus header-free cropped PDFs to
``report/figures/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.phase1 import figroles as fr
from experiments.phase1 import figstyle as fs


def _method_key(label: str) -> str | None:
    if label.startswith("trustgraph(descent_neg,mclip,t=0.003,val=iid)["):
        return "tg_stab"
    if label.startswith("trustgraph(descent_neg,mclip,t=0.003,a=0.3,val=iid)"):
        return "tg_a03"
    if label.startswith("trustgraph(descent_neg,mclip,t=0.003,a=0.7,val=iid)"):
        return "tg_a07"
    if label.startswith("trustgraph(descent_neg,euclidean"):
        return "tg_euclidean"
    if label.startswith("trustgraph(descent_neg,norelu"):
        return "tg_no_relu"
    return None


def _index_rows(rows: list[dict[str, object]]) -> dict[tuple[str, str, int, str], dict[str, object]]:
    out: dict[tuple[str, str, int, str], dict[str, object]] = {}
    for r in rows:
        mk = _method_key(str(r["method"]))
        if mk is None:
            continue
        out[(str(r["topology"]), str(r["attack"]), int(r["byz_pct"]), mk)] = r
    return out


def _find_run(runs: list[dict[str, object]], suffix: str) -> dict[str, object]:
    for r in runs:
        if str(r["run_name"]).endswith(suffix):
            return r
    raise KeyError(f"Run ending with {suffix!r} not found.")


def _find_baseline_row(data: dict[str, object], attack: str) -> dict[str, object]:
    for r in data["baseline_50k_3seed"]["rows"]:
        if str(r["attack"]) == attack and int(r["byz_pct"]) == 30:
            return r
    raise KeyError(f"Baseline 3-seed row for attack={attack!r} not found.")


def plot_fig1_attacks(
    rows_idx: dict[tuple[str, str, int, str], dict[str, object]],
    data: dict[str, object],
    out_dir: Path,
) -> None:
    """Fig. 1: TrustGraph-DFL convergence and final accuracy across all 4 attack regimes."""
    curves = data["curves"]
    fig, (ax_conv, ax_bar) = plt.subplots(1, 2, figsize=(15.4, 6.4))
    fs.light_grid(ax_conv, "y")
    fs.light_grid(ax_bar, "y")

    print("\n=== Fig. 1 (50k) TrustGraph-DFL Consensus Convergence Across Attacks (3 seeds) ===")
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
        print(f"  {atk:16s}: {pts} (final +/- {std[-1]:.2f})")

    ax_conv.set_title("(a) Consensus Convergence Over Training", pad=10, fontweight="bold")
    ax_conv.set_xlabel("Communication Round")
    ax_conv.set_ylabel("Consensus Test Accuracy (%)")
    ax_conv.set_xlim(0.5, 25.5)
    ax_conv.set_xticks([1, 5, 10, 15, 20, 25])
    ax_conv.set_ylim(35, 98)
    ax_conv.legend(loc="lower right", ncol=1)

    short_labels = ["Clean\n(None)", "Label Flip\n($E_b=2$)", "Model Poison\n($\\lambda=10$)", "Random Noise\n($\\sigma=0.5$)"]
    x = np.arange(len(fr.ATTACK_ORDER), dtype=float)
    w = 0.34
    peer_m = [float(rows_idx[("random", a, 30, "tg_stab")]["acc_mean"]) for a in fr.ATTACK_ORDER]
    peer_s = [float(rows_idx[("random", a, 30, "tg_stab")]["acc_std"]) for a in fr.ATTACK_ORDER]
    cons_m = [float(rows_idx[("random", a, 30, "tg_stab")]["cons_mean"]) for a in fr.ATTACK_ORDER]
    cons_s = [float(rows_idx[("random", a, 30, "tg_stab")]["cons_std"]) for a in fr.ATTACK_ORDER]

    print("\n=== Fig. 1(b) TrustGraph-DFL Per-Peer vs. Consensus Accuracy (%) ===")
    for a, pm, ps, cm, cs in zip(fr.ATTACK_ORDER, peer_m, peer_s, cons_m, cons_s):
        print(f"  {a:16s}: peer={pm:.2f}+/-{ps:.2f}%  cons={cm:.2f}+/-{cs:.2f}%")

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
        "Stabilized TrustGraph-DFL (descent_neg, mclip 2*r_i, tau=0.003, gc=10) sustains 91.5-92.9% consensus and 88.9-91.4% per-peer accuracy across all regimes.",
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=top - 0.08, bottom=bot + 0.90 / 6.4, wspace=0.22)
    fs.save(fig, out_dir, "fig1-attacks-50k")


def plot_fig2_topologies_and_stress(
    rows_idx: dict[tuple[str, str, int, str], dict[str, object]],
    out_dir: Path,
) -> None:
    """Fig. 2: Topology robustness (Table 1) and 10%-50% Byzantine stress sweep."""
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

    print("\n=== Fig. 2(a) TrustGraph-DFL Topologies under label_flip_all (30% Byz, bep=2, 3 seeds) ===")
    for (tp, _), pm, ps, cm, cs in zip(topos, p_m, p_s, c_m, c_s):
        print(f"  {tp:12s}: peer={pm:.2f}+/-{ps:.2f}%  cons={cm:.2f}+/-{cs:.2f}%")

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

    print("\n=== Fig. 2(b) TrustGraph-DFL Byzantine Stress Sweep (10%-50%, label_flip_all, bep=2, 3 seeds) ===")
    for f, cm, cs, pm, ps, fm, fs_ in zip(fracs, cons_m, cons_sd, peer_m, peer_sd, fpr_m, fpr_sd):
        print(f"  byz={f}%: cons={cm:.2f}+/-{cs:.2f}%  peer={pm:.2f}+/-{ps:.2f}%  fpr={fm:.2f}+/-{fs_:.2f}%")

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


def plot_fig3_ablations_and_fpr_dynamics(
    rows_idx: dict[tuple[str, str, int, str], dict[str, object]],
    data: dict[str, object],
    out_dir: Path,
) -> None:
    """Fig. 3: Section 5.5 ablations (model_poison) + per-round FPR trajectory."""
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

    print("\n=== Fig. 3(a) (50k) Section 5.5 Ablations under model_poison (3 seeds) ===")
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
        print(f"  {k:14s}: cons={m:.2f} +/- {s:.2f}%")

    ax_abl.set_xticks(x)
    ax_abl.set_xticklabels([lbl for _, lbl in abl_keys])
    ax_abl.set_ylabel("Final Consensus Test Accuracy (%)")
    ax_abl.set_title("(a) Section 5.5 Ablations (Model Poisoning)", pad=10, fontweight="bold")
    ax_abl.set_ylim(0, 108)

    curves = data["curves"]
    print("\n=== Fig. 3(b) (50k) Per-Round FPR Trajectory for TrustGraph-DFL ===")
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
        print(f"  {atk:16s}: r1={fm[0]:.1f}% r5={fm[4]:.1f}% r15={fm[14]:.1f}% r25={fm[-1]:.1f}%")

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


def plot_fig4_fpr_rxdelta(data: dict[str, object], out_dir: Path) -> None:
    """Fig. 4: 2-panel comparison of sender vs. rxdelta on full 50k CIFAR-10 (model_poison)."""
    runs: list[dict[str, object]] = data["runs"]
    base_mp = _find_baseline_row(data, "model_poison")
    base_curve = data["baseline_50k_3seed"]["curves"]["model_poison__tg_strong"]

    l1_mp = _find_run(runs, "lever1/model_poison__iidval")
    l1b_t020_mp = _find_run(runs, "lever1b/t020__model_poison")
    l1b_t035_mp = _find_run(runs, "lever1b/t035__model_poison")

    fig, (ax_trade, ax_traj) = plt.subplots(1, 2, figsize=(15.6, 6.5))
    fs.light_grid(ax_trade, "y")
    fs.light_grid(ax_traj, "y")

    ax_trade.axvspan(3.0, 9.0, color=fs.PALETTE["green_1"], alpha=0.65, zorder=1)
    ax_trade.text(6.0, 48.5, "Paper\n3–9%\ntarget", ha="center", va="top", fontsize=11.5, fontweight="bold", color=fs.INK)

    base_pt = (float(base_mp["fpr_mean"]), float(base_mp["byz_accept_mean"]))
    rx_pts = [
        (r"$\tau=0.003$", 100.0 * float(l1_mp["fpr_cumulative"]), 100.0 * float(l1_mp["byz_accept_cumulative"]), (10, 5)),
        (r"$\tau=0.020$", 100.0 * float(l1b_t020_mp["fpr_cumulative"]), 100.0 * float(l1b_t020_mp["byz_accept_cumulative"]), (10, 5)),
        (r"$\tau=0.035$", 100.0 * float(l1b_t035_mp["fpr_cumulative"]), 100.0 * float(l1b_t035_mp["byz_accept_cumulative"]), (-6, -22)),
    ]

    print("\n=== Fig. 4(a) Sender vs. rxdelta Trade-off on model_poison (30% Byz, 50k CIFAR-10) ===")
    print(f"  Base sender (3-seed)     : FPR={base_pt[0]:5.2f}%, ByzAcc={base_pt[1]:5.2f}%")

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
        print(f"  rxdelta {lbl:16s} : FPR={x_v:5.2f}%, ByzAcc={y_v:5.2f}%")
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

    def _traj(run: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
        h = run["history"]
        return np.array([float(x["round"]) + 1.0 for x in h]), np.array([100.0 * float(x["fpr_round"]) for x in h])

    r_l1, fpr_l1 = _traj(l1_mp)
    r_l1b_020, fpr_l1b_020 = _traj(l1b_t020_mp)
    r_l1b_035, fpr_l1b_035 = _traj(l1b_t035_mp)
    marks = [0, 4, 9, 14, 19, 24]

    ax_traj.plot(
        rounds_b, fpr_b_m, label="Base sender (3-seed)",
        color=fs.PALETTE["gray"], linestyle="-", marker="o", markevery=marks, markersize=7.5, linewidth=2.4, zorder=3,
    )
    ax_traj.fill_between(rounds_b, np.clip(fpr_b_m - fpr_b_s, 0, 100), np.clip(fpr_b_m + fpr_b_s, 0, 100), color=fs.PALETTE["gray"], alpha=0.14)
    ax_traj.plot(
        r_l1b_035, fpr_l1b_035, label=r"rxdelta ($\tau=0.035$)",
        color=fs.PALETTE["teal"], linestyle="--", marker="D", markevery=marks, markersize=7.0, linewidth=2.4, zorder=4,
    )
    ax_traj.plot(
        r_l1b_020, fpr_l1b_020, label=r"rxdelta ($\tau=0.020$)",
        color=fs.PALETTE["blue_secondary"], linestyle="-.", marker="^", markevery=marks, markersize=7.5, linewidth=2.4, zorder=5,
    )
    ax_traj.plot(
        r_l1, fpr_l1, label=r"rxdelta ($\tau=0.003$)",
        color=fs.PALETTE["blue_main"], linestyle="-", marker="s", markevery=marks, markersize=7.5, linewidth=2.8, zorder=6,
    )

    print("\n=== Fig. 4(b) Per-Round FPR Trajectory on model_poison (r=1, 5, 10, 15, 25) ===")
    for name, arr in (
        ("Base sender (3-seed)", fpr_b_m),
        ("rxdelta (tau=0.035)", fpr_l1b_035),
        ("rxdelta (tau=0.020)", fpr_l1b_020),
        ("rxdelta (tau=0.003)", fpr_l1),
    ):
        print(f"  {name:24s}: r1={arr[0]:5.1f}% r5={arr[4]:5.1f}% r10={arr[9]:5.1f}% r15={arr[14]:5.1f}% r25={arr[-1]:5.1f}%")

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


def main() -> None:
    p = argparse.ArgumentParser(description="Draw all final TrustGraph-DFL figures from committed JSON data.")
    p.add_argument("--data-50k", default="report/data/phase1_paper_match_50k_3seed.json")
    p.add_argument("--data-rxdelta", default="report/data/phase1_fpr_rxdelta.json")
    p.add_argument("--out-dir", default="report/figures")
    args = p.parse_args()

    data_50k = json.loads(Path(args.data_50k).read_text(encoding="utf-8"))
    data_rxdelta = json.loads(Path(args.data_rxdelta).read_text(encoding="utf-8"))
    rows_idx = _index_rows(data_50k["rows"])
    out_dir = Path(args.out_dir)

    fs.apply("compact")
    plot_fig1_attacks(rows_idx, data_50k, out_dir)
    plot_fig2_topologies_and_stress(rows_idx, out_dir)
    plot_fig3_ablations_and_fpr_dynamics(rows_idx, data_50k, out_dir)
    plot_fig4_fpr_rxdelta(data_rxdelta, out_dir)


if __name__ == "__main__":
    main()
