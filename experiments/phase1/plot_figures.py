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

    short_labels = ["No Attack\n(0% Byz)", "Label Flip\n($E_b=2$)", "Model Poison\n($\\lambda=10$)", "Random Noise\n($\\sigma=0.5$)"]
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

    rx_t003_mp = _find_run(runs, "fig3b_fpr_rxdelta/model_poison__iidval")
    rx_t020_mp = _find_run(runs, "fig3b_fpr_rxdelta_tau/t020__model_poison")
    rx_t035_mp = _find_run(runs, "fig3b_fpr_rxdelta_tau/t035__model_poison")

    fig, (ax_trade, ax_traj) = plt.subplots(1, 2, figsize=(15.6, 6.5))
    fs.light_grid(ax_trade, "y")
    fs.light_grid(ax_traj, "y")

    ax_trade.axvspan(3.0, 9.0, color=fs.PALETTE["green_1"], alpha=0.65, zorder=1)
    ax_trade.text(6.0, 48.5, "Paper\n3–9%\ntarget", ha="center", va="top", fontsize=11.5, fontweight="bold", color=fs.INK)

    base_pt = (float(base_mp["fpr_mean"]), float(base_mp["byz_accept_mean"]))
    rx_pts = [
        (r"$\tau=0.003$", 100.0 * float(rx_t003_mp["fpr_cumulative"]), 100.0 * float(rx_t003_mp["byz_accept_cumulative"]), (10, 5)),
        (r"$\tau=0.020$", 100.0 * float(rx_t020_mp["fpr_cumulative"]), 100.0 * float(rx_t020_mp["byz_accept_cumulative"]), (10, 5)),
        (r"$\tau=0.035$", 100.0 * float(rx_t035_mp["fpr_cumulative"]), 100.0 * float(rx_t035_mp["byz_accept_cumulative"]), (-6, -22)),
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

    r_rx003, fpr_rx003 = _traj(rx_t003_mp)
    r_rx020, fpr_rx020 = _traj(rx_t020_mp)
    r_rx035, fpr_rx035 = _traj(rx_t035_mp)
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

    print("\n=== Fig. 4(b) Per-Round FPR Trajectory on model_poison (r=1, 5, 10, 15, 25) ===")
    for name, arr in (
        ("Base sender (3-seed)", fpr_b_m),
        ("rxdelta (tau=0.035)", fpr_rx035),
        ("rxdelta (tau=0.020)", fpr_rx020),
        ("rxdelta (tau=0.003)", fpr_rx003),
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


def _select_run(
    runs: list[dict[str, object]],
    topology: str,
    attack: str,
    byz_pct: int,
    method_prefix: str = "trustgraph(descent_neg,mclip,t=0.003,val=iid)[",
) -> dict[str, object]:
    for r in runs:
        if (
            str(r["topology"]) == topology
            and str(r["attack"]) == attack
            and int(r["byz_pct"]) == byz_pct
            and str(r["method"]).startswith(method_prefix)
        ):
            return r
    raise KeyError(f"Run not found for ({topology}, {attack}, byz={byz_pct}).")


def plot_bandwidth_and_round_time(data: dict[str, object], out_dir: Path) -> None:
    """Plot received vs. wasted FP32 bandwidth across attacks and round time by link speed."""
    runs: list[dict[str, object]] = data["runs"]
    fig, (ax_bw, ax_time) = plt.subplots(1, 2, figsize=(15.6, 6.5))
    fs.light_grid(ax_bw, "y")
    fs.light_grid(ax_time, "y")

    short_labels = ["No Attack\n(0% Byz)", "Label Flip\n($E_b=2$)", "Model Poison\n($\\lambda=10$)", "Random Noise\n($\\sigma=0.5$)"]
    x = np.arange(len(fr.ATTACK_ORDER), dtype=float)
    useful = []
    w_hon = []
    w_byz = []
    w_pct = []
    print("\n=== Fig. 5(a) Per-Peer Receive Bandwidth per Round (MB, 50k ResNet-18, Random ER p=0.3) ===")
    for atk in fr.ATTACK_ORDER:
        byz_p = 30
        r = _select_run(runs, "random", atk, byz_p)
        sm = r["system_metrics"]
        u = float(sm["useful_mb_per_peer_round"])
        wh = float(sm["wasted_honest_mb_per_peer_round"])
        wb = float(sm["wasted_byz_mb_per_peer_round"])
        wp = float(sm["wasted_transfer_pct"])
        useful.append(u)
        w_hon.append(wh)
        w_byz.append(wb)
        w_pct.append(wp)
        print(f"  {atk:16s}: rx={u + wh + wb:6.1f} MB | useful={u:5.1f} | wasted_honest={wh:5.1f} | wasted_byz={wb:5.1f} ({wp:5.1f}% wasted)")

    useful_a = np.array(useful)
    w_hon_a = np.array(w_hon)
    w_byz_a = np.array(w_byz)
    bar_w = 0.54
    ax_bw.bar(x, useful_a, width=bar_w, color=fs.PALETTE["blue_main"], edgecolor="black", linewidth=1.6, label="Accepted (Useful)", zorder=3)
    ax_bw.bar(x, w_hon_a, bottom=useful_a, width=bar_w, color=fs.PALETTE["red_2"], edgecolor="black", linewidth=1.6, label="Wasted (Rejected Honest)", zorder=3)
    ax_bw.bar(x, w_byz_a, bottom=useful_a + w_hon_a, width=bar_w, color=fs.PALETTE["red_strong"], edgecolor="black", linewidth=1.6, label="Wasted (Rejected Byzantine)", zorder=3)

    totals = useful_a + w_hon_a + w_byz_a
    for xi, tot, wp in zip(x, totals, w_pct):
        ax_bw.text(xi, tot + 6.0, f"{wp:.1f}%\nwasted", ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=fs.PALETTE["red_strong"])

    ax_bw.set_xticks(x)
    ax_bw.set_xticklabels(short_labels)
    ax_bw.set_ylabel("Received FP32 Payload per Peer / Round (MB)")
    ax_bw.set_title("(a) Useful vs. Wasted Receive Traffic per Round", pad=10, fontweight="bold")
    ax_bw.set_ylim(0, max(totals) * 1.38)
    ax_bw.legend(loc="upper center", ncol=1, fontsize=11.0)

    # Panel (b): Per-peer round time breakdown on model_poison across link speeds
    mp = _select_run(runs, "random", "model_poison", 30)
    sm_mp = mp["system_metrics"]
    t_comp = float(sm_mp["train_sec_per_peer"]["median"]) + float(sm_mp["val_grad_sec_per_peer"]["median"]) + float(sm_mp["agg_sec_per_peer"]["median"])
    links = [("10gbps", "10 Gb/s\nDatacenter"), ("1gbps", "1 Gb/s\nEthernet"), ("100mbps", "100 Mb/s\nWAN / Edge")]
    xl = np.arange(len(links), dtype=float)
    comp_vals = []
    useful_c = []
    wasted_c = []
    comm_shares = []

    print("\n=== Fig. 5(b) Per-Peer Round Time Breakdown on model_poison (30% Byz, Random) ===")
    for key, _ in links:
        blk = sm_mp["round_time_by_link"][key]
        c_tot = float(blk["comm_sec"])
        c_w = float(blk["wasted_comm_sec"])
        c_u = max(0.0, c_tot - c_w)
        comp_vals.append(t_comp)
        useful_c.append(c_u)
        wasted_c.append(c_w)
        comm_shares.append(float(blk["comm_share_pct"]))
        print(f"  {key:8s}: compute+val+agg={t_comp:.3f}s | useful_comm={c_u:.3f}s | wasted_comm={c_w:.3f}s | total={t_comp + c_tot:.3f}s ({blk['comm_share_pct']:.1f}% comm)")

    comp_a = np.array(comp_vals)
    uc_a = np.array(useful_c)
    wc_a = np.array(wasted_c)
    ax_time.bar(xl, comp_a, width=0.52, color=fs.PALETTE["gray"], edgecolor="black", linewidth=1.6, label="Local Train + Val Grad + Agg", zorder=3)
    ax_time.bar(xl, uc_a, bottom=comp_a, width=0.52, color=fs.PALETTE["blue_main"], edgecolor="black", linewidth=1.6, label="Useful FP32 Transfer", zorder=3)
    ax_time.bar(xl, wc_a, bottom=comp_a + uc_a, width=0.52, color=fs.PALETTE["red_strong"], edgecolor="black", linewidth=1.6, label="Wasted FP32 Transfer", zorder=3)

    tots_t = comp_a + uc_a + wc_a
    for xi, tot, sh in zip(xl, tots_t, comm_shares):
        ax_time.text(xi, tot + max(tots_t) * 0.02, f"{tot:.1f}s\n({sh:.0f}% comm)", ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=fs.INK)

    ax_time.set_xticks(xl)
    ax_time.set_xticklabels([lbl for _, lbl in links])
    ax_time.set_ylabel("Per-Peer Round Time (seconds)")
    ax_time.set_title("(b) Round Time Breakdown (30% Model Poisoning)", pad=10, fontweight="bold")
    ax_time.set_ylim(0, max(tots_t) * 1.28)
    ax_time.legend(loc="upper left", ncol=1, fontsize=11.0)

    top = fs.header(
        fig,
        "Batched FP32 TrustGraph-DFL Communication Waste and Round-Time Bottleneck",
        "Pretrained ResNet-18 (11.18M FP32 params = 44.7 MB/update, full 50k CIFAR-10, 20 peers, 25 rounds, seed 42)",
    )
    bot = fs.footnote(
        fig,
        "Left: 41-58% of received FP32 updates receive zero trust weight and are discarded. Right: Uncompressed FP32 transfer dominates round time at 1 Gb/s and 100 Mb/s.",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=top - 0.08, bottom=bot + 0.90 / 6.5, wspace=0.22)
    fs.save(fig, out_dir, "fig5-bandwidth-and-round-time-50k")


def plot_receive_memory_and_rejection_streaks(data: dict[str, object], out_dir: Path) -> None:
    """Plot batched O(d) receive-buffer memory across topologies and repeated rejection streaks."""
    runs: list[dict[str, object]] = data["runs"]
    fig, (ax_mem, ax_str) = plt.subplots(1, 2, figsize=(15.6, 6.5))
    fs.light_grid(ax_mem, "y")
    fs.light_grid(ax_str, "y")

    topos = [("ring", "Ring\n(deg = 2)"), ("small_world", "Small-World\n(deg = 4)"), ("random", "Random\n(ER p = 0.3)")]
    xt = np.arange(len(topos), dtype=float)
    w = 0.34
    mean_bufs = []
    max_bufs = []
    payload_mib = 0.0
    print("\n=== Fig. 6(a) Batched FP32 Receive-Buffer Memory Across Topologies (MiB) ===")
    for tp, _ in topos:
        r = _select_run(runs, tp, "label_flip_all", 30)
        sm = r["system_metrics"]
        payload_mib = float(sm["payload_fp32_mib"])
        mb = float(sm["batched_recv_buf_mean_mib"])
        xb = float(sm["batched_recv_buf_max_mib"])
        mean_bufs.append(mb)
        max_bufs.append(xb)
        print(f"  {tp:12s}: mean_deg={sm['mean_benign_degree']:.1f} ({mb:.1f} MiB) | max_deg={sm['max_benign_degree']} ({xb:.1f} MiB) | single_payload={payload_mib:.1f} MiB")

    b1 = ax_mem.bar(xt - w / 2, mean_bufs, width=w, color=fs.PALETTE["blue_secondary"], edgecolor="black", linewidth=1.6, label="Mean-Degree Peer Buffer", zorder=3)
    b2 = ax_mem.bar(xt + w / 2, max_bufs, width=w, color=fs.PALETTE["blue_main"], edgecolor="black", linewidth=1.6, label="Max-Degree Peer Buffer", zorder=3)
    ax_mem.axhline(payload_mib, color=fs.PALETTE["teal"], linestyle="--", linewidth=2.4, label=f"1-Update Reference ({payload_mib:.1f} MiB)", zorder=4)

    for b, v in zip(b1, mean_bufs):
        ax_mem.text(b.get_x() + b.get_width() / 2.0, v + 10.0, f"{v:.0f}", ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=fs.PALETTE["blue_secondary"])
    for b, v in zip(b2, max_bufs):
        ax_mem.text(b.get_x() + b.get_width() / 2.0, v + 10.0, f"{v:.0f}", ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=fs.PALETTE["blue_main"])

    ax_mem.set_xticks(xt)
    ax_mem.set_xticklabels([lbl for _, lbl in topos])
    ax_mem.set_ylabel("Simultaneous FP32 Receive Buffers (MiB)")
    ax_mem.set_title("(a) Batched Receive Memory Scales Linearly With Degree", pad=10, fontweight="bold")
    ax_mem.set_ylim(0, max(max_bufs) * 1.32)
    ax_mem.legend(loc="upper left", ncol=1, fontsize=11.0)

    # Panel (b): Repeated rejections on Byzantine edges vs. honest single-round vs. >=3 consecutive rejections
    atks = [("label_flip_all", "Label Flip\n($E_b=2$)"), ("model_poison", "Model Poison\n($\\lambda=10$)"), ("random_noise", "Random Noise\n($\\sigma=0.5$)")]
    xa = np.arange(len(atks), dtype=float)
    w3 = 0.25
    byz_ge3 = []
    hon_ge1 = []
    hon_ge3 = []
    print("\n=== Fig. 6(b) Repeated Rejection Streaks Across 25 Rounds (Random ER p=0.3, 30% Byz) ===")
    for atk, _ in atks:
        r = _select_run(runs, "random", atk, 30)
        sm = r["system_metrics"]
        bg3 = float(sm["byz_repeat_rejections"]["frac_edges_with_streak_ge3_pct"])
        hg1 = float(sm["honest_rejection_streaks"]["frac_edges_with_ge1_rej_pct"])
        hg3 = float(sm["honest_rejection_streaks"]["frac_edges_with_streak_ge3_pct"])
        byz_ge3.append(bg3)
        hon_ge1.append(hg1)
        hon_ge3.append(hg3)
        print(
            f"  {atk:16s}: byz_mean_rej={sm['byz_repeat_rejections']['mean_rejected_rounds']:.1f}/25 (streak>=3: {bg3:.1f}%) | "
            f"hon_ge1={hg1:.1f}% | hon_streak>=3={hg3:.1f}%"
        )

    s1 = ax_str.bar(xa - w3, byz_ge3, width=w3, color=fs.PALETTE["red_strong"], edgecolor="black", linewidth=1.5, label="Byzantine Edges: >=3 Consec. Rej", zorder=3)
    s2 = ax_str.bar(xa, hon_ge1, width=w3, color=fs.PALETTE["blue_secondary"], edgecolor="black", linewidth=1.5, label="Honest Edges: >=1 Rejection", zorder=3)
    s3 = ax_str.bar(xa + w3, hon_ge3, width=w3, color=fs.PALETTE["teal"], edgecolor="black", linewidth=1.5, label="Honest Edges: >=3 Consec. Rej", zorder=3)

    for bars, vals, col in ((s1, byz_ge3, fs.PALETTE["red_strong"]), (s2, hon_ge1, fs.PALETTE["blue_secondary"]), (s3, hon_ge3, fs.PALETTE["teal"])):
        for b, v in zip(bars, vals):
            ax_str.text(b.get_x() + b.get_width() / 2.0, v + 2.0, f"{v:.0f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=col)

    ax_str.set_xticks(xa)
    ax_str.set_xticklabels([lbl for _, lbl in atks])
    ax_str.set_ylabel("Share of Directed Peer Edges (%)")
    ax_str.set_title("(b) Repeat Rejections: Byzantine vs. Honest Edges", pad=10, fontweight="bold")
    ax_str.set_ylim(0, 132)
    ax_str.legend(loc="upper center", ncol=1, fontsize=10.5)

    top = fs.header(
        fig,
        "Receive-Buffer Memory Growth and Repeated-Transfer Behavior in TrustGraph-DFL",
        "Pretrained ResNet-18 (42.7 MiB/update, full 50k CIFAR-10, 20 peers, 25 rounds, seed 42)",
    )
    bot = fs.footnote(
        fig,
        "Left: Batched FP32 receive buffers grow linearly with peer degree (up to 10x a single update on Random graphs). Right: Single-rejection banning would hit nearly all honest edges.",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=top - 0.08, bottom=bot + 0.90 / 6.5, wspace=0.22)
    fs.save(fig, out_dir, "fig6-memory-and-rejection-streaks-50k")


def main() -> None:
    p = argparse.ArgumentParser(description="Draw all final TrustGraph-DFL figures from committed JSON data.")
    p.add_argument("--data-50k", default="report/data/phase1_paper_match_50k_3seed.json")
    p.add_argument("--data-rxdelta", default="report/data/phase1_fpr_rxdelta.json")
    p.add_argument("--data-system", default="report/data/phase1_system_metrics_50k.json")
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

    system_data_path = Path(args.data_system)
    if system_data_path.exists():
        data_system = json.loads(system_data_path.read_text(encoding="utf-8"))
        plot_bandwidth_and_round_time(data_system, out_dir)
        plot_receive_memory_and_rejection_streaks(data_system, out_dir)


if __name__ == "__main__":
    main()
