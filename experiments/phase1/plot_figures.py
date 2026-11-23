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

from matplotlib.colors import LinearSegmentedColormap
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
    if label.startswith("balance("):
        return "balance"
    if label.startswith("krum"):
        return "krum"
    if label.startswith("trimmed_mean"):
        return "trimmed_mean"
    if label.startswith("fedavg"):
        return "fedavg"
    return None


def _curve_tag(m: str) -> str:
    return "tg_strong" if m == "tg_stab" else m


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


def plot_fig2_convergence(data: dict[str, object], out_dir: Path) -> None:
    """Fig. 2: Consensus accuracy convergence across three attacks on full 50k CIFAR-10."""
    curves = data["curves"]
    attacks = [
        ("model_poison", "(a) Model Poisoning ($\\lambda=10$)"),
        ("random_noise", "(b) Random Noise ($\\sigma=0.5$)"),
        ("label_flip_all", "(c) Label Flipping (All Classes, $E_b=2$)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 6.7), sharey=True)
    print("\n=== Fig. 2 (50k) Convergence Curves (3-seed mean +/- std consensus acc %) ===")
    for ax, (atk, title) in zip(axes, attacks):
        fs.light_grid(ax, "y")
        print(f"\n[{atk}]")
        for m in fr.METHOD_ORDER:
            c = curves[f"{atk}__{_curve_tag(m)}"]
            rounds = np.array(c["eval_rounds"], dtype=float)
            mean = np.array(c["cons_mean"], dtype=float)
            std = np.array(c["cons_std"], dtype=float)
            color = fr.SERIES_COLOR[m]
            lw = 3.0 if m == "tg_stab" else 2.2
            zorder = 5 if m == "tg_stab" else 3
            ax.plot(
                rounds,
                mean,
                label=fr.SERIES_LABEL[m],
                color=color,
                linestyle=fr.SERIES_LINESTYLE[m],
                marker=fr.SERIES_MARKER[m],
                markersize=8.0 if m == "tg_stab" else 7.0,
                linewidth=lw,
                zorder=zorder,
            )
            ax.fill_between(
                rounds,
                np.clip(mean - std, 0.0, 100.0),
                np.clip(mean + std, 0.0, 100.0),
                color=color,
                alpha=0.14,
                zorder=zorder - 1,
            )
            pts = " ".join(f"r{int(r)}={v:.2f}" for r, v in zip(rounds, mean))
            print(f"  {fr.SERIES_LABEL[m]:18s}: {pts} (final +/- {std[-1]:.2f})")

        ax.set_title(title, pad=10, fontweight="bold")
        ax.set_xlabel("Communication Round")
        ax.set_xlim(0, 26)
        ax.set_xticks([1, 5, 10, 15, 20, 25])
        ax.set_ylim(2, 99)

    axes[0].set_ylabel("Consensus Test Accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    top = fs.header(
        fig,
        "Consensus Convergence Under 30% Byzantine Attacks (Full 50k CIFAR-10)",
        f"Pretrained ResNet-18 (2,500 samples/node, Dir alpha=0.5, random ER p=0.3, 25 rounds; mean +/- std over 3 seeds, n={data['total_runs']} runs, 0 NaNs)",
    )
    bot = fs.footnote(
        fig,
        "TrustGraph (stab) = descent_neg (-nabla F_val), model-space clip 2*r_i, tau=0.003, 250 stratified val/node; BALANCE gamma=0.04; grad clip=10.0.",
    )
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, top - 0.015), ncol=5, columnspacing=1.5, handlelength=2.4)
    fig.subplots_adjust(left=0.06, right=0.985, top=top - 0.165, bottom=bot + 0.85 / 6.7, wspace=0.12)
    fs.save(fig, out_dir, "fig2-convergence-50k")


def plot_fig3a_heatmap_and_bars(
    rows_idx: dict[tuple[str, str, int, str], dict[str, object]],
    out_dir: Path,
) -> None:
    """Fig. 3(a): Annotated attack heatmap + grouped bar comparison."""
    attacks = [
        ("none", "Clean\n(None)"),
        ("label_flip_all", "Label Flip\n($E_b=2$)"),
        ("model_poison", "Model Poison\n($\\lambda=10$)"),
        ("random_noise", "Random Noise\n($\\sigma=0.5$)"),
    ]
    fig, (ax_hm, ax_bar) = plt.subplots(1, 2, figsize=(16.8, 6.7), gridspec_kw={"width_ratios": [1.05, 1.25]})

    mat = np.zeros((len(fr.METHOD_ORDER), len(attacks)), dtype=float)
    std_mat = np.zeros_like(mat)
    print("\n=== Fig. 3(a) (50k) Final Consensus Accuracy by Attack (3-seed mean +/- std %) ===")
    for i, m in enumerate(fr.METHOD_ORDER):
        parts = []
        for j, (atk, _) in enumerate(attacks):
            r = rows_idx[("random", atk, 30, m)]
            mat[i, j] = float(r["cons_mean"])
            std_mat[i, j] = float(r["cons_std"])
            parts.append(f"{atk}={mat[i, j]:.2f}+/-{std_mat[i, j]:.2f}")
        print(f"  {fr.SERIES_LABEL[m]:18s}: " + "  ".join(parts))

    cmap = LinearSegmentedColormap.from_list(
        "f4p_blue_red",
        [fs.PALETTE["red_1"], "#F7F7F7", fs.PALETTE["green_1"], fs.PALETTE["blue_secondary"], fs.PALETTE["blue_main"]],
    )
    ax_hm.imshow(mat, cmap=cmap, vmin=10.0, vmax=95.0, aspect="auto")
    ax_hm.set_xticks(np.arange(len(attacks)))
    ax_hm.set_xticklabels([lbl for _, lbl in attacks])
    ax_hm.set_yticks(np.arange(len(fr.METHOD_ORDER)))
    ax_hm.set_yticklabels([fr.SERIES_LABEL[m] for m in fr.METHOD_ORDER])
    ax_hm.set_title("(a) Consensus Accuracy Heatmap (%)", pad=10, fontweight="bold")
    ax_hm.set_xticks(np.arange(-0.5, len(attacks), 1), minor=True)
    ax_hm.set_yticks(np.arange(-0.5, len(fr.METHOD_ORDER), 1), minor=True)
    ax_hm.grid(which="minor", color="white", linewidth=2.0)
    ax_hm.tick_params(which="minor", bottom=False, left=False)
    for i in range(len(fr.METHOD_ORDER)):
        for j in range(len(attacks)):
            val = mat[i, j]
            sd = std_mat[i, j]
            ax_hm.text(
                j,
                i,
                f"{val:.1f}%\n±{sd:.1f}",
                ha="center",
                va="center",
                fontsize=12.5,
                fontweight="bold" if i == 0 else "normal",
                color="white" if val >= 68.0 else fs.INK,
            )

    fs.light_grid(ax_bar, "y")
    x = np.arange(len(attacks), dtype=float)
    n_m = len(fr.METHOD_ORDER)
    width = 0.155
    offsets = (np.arange(n_m) - (n_m - 1) / 2.0) * width
    for idx, m in enumerate(fr.METHOD_ORDER):
        means = mat[idx]
        stds = std_mat[idx]
        bars = ax_bar.bar(
            x + offsets[idx],
            means,
            width=width * 0.92,
            yerr=stds,
            capsize=3.2,
            color=fr.SERIES_COLOR[m],
            edgecolor="black",
            linewidth=1.5,
            error_kw={"elinewidth": 1.5, "ecolor": "black"},
            label=fr.SERIES_LABEL[m],
            zorder=3,
        )
        if m == "tg_stab":
            for b, val in zip(bars, means):
                ax_bar.text(
                    b.get_x() + b.get_width() / 2.0,
                    val + 2.0,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=12.0,
                    fontweight="bold",
                    color=fr.SERIES_COLOR[m],
                )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([lbl for _, lbl in attacks])
    ax_bar.set_ylabel("Final Consensus Test Accuracy (%)")
    ax_bar.set_title("(b) Attack Comparison With 3-Seed Uncertainty", pad=10, fontweight="bold")
    ax_bar.set_ylim(0, 108)
    ax_bar.set_xlim(-0.55, len(attacks) - 0.45)

    handles, labels = ax_bar.get_legend_handles_labels()
    top = fs.header(
        fig,
        "Defense Comparison Across Attack Regimes (Full 50k CIFAR-10, 30% Byzantine)",
        "Pretrained ResNet-18 (2,500 samples/node, Dir alpha=0.5, random ER p=0.3, 25 rounds, 3 seeds; 0/150 NaNs)",
    )
    bot = fs.footnote(
        fig,
        "TrustGraph (stab) is the sole defense sustaining 91.5-92.9% consensus across every regime; Krum collapses on model_poison (37.1%) and random_noise (30.4%).",
    )
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, top - 0.015), ncol=5, columnspacing=1.5, handlelength=1.8)
    fig.subplots_adjust(left=0.11, right=0.985, top=top - 0.165, bottom=bot + 0.95 / 6.7, wspace=0.22)
    fs.save(fig, out_dir, "fig3a-attack-heatmap-50k")


def plot_table1_topologies(rows_idx: dict[tuple[str, str, int, str], dict[str, object]], out_dir: Path) -> None:
    """Table 1 visual: Per-peer & consensus accuracy across Ring, Random, and Small-World."""
    topos = [("ring", "Ring\n(deg = 2)"), ("random", "Random\n(ER p = 0.3)"), ("small_world", "Small-World\n(WS k=4, p=0.3)")]
    fig, (ax_peer, ax_cons) = plt.subplots(1, 2, figsize=(15.0, 6.7), sharey=True)
    fs.light_grid(ax_peer, "y")
    fs.light_grid(ax_cons, "y")

    x = np.arange(len(topos), dtype=float)
    n_m = len(fr.METHOD_ORDER)
    width = 0.155
    offsets = (np.arange(n_m) - (n_m - 1) / 2.0) * width

    print("\n=== Table 1 (50k) Topologies under label_flip_all (30% Byz, bep=2, 3 seeds) ===")
    for idx, m in enumerate(fr.METHOD_ORDER):
        p_m = [float(rows_idx[(tp, "label_flip_all", 30, m)]["acc_mean"]) for tp, _ in topos]
        p_s = [float(rows_idx[(tp, "label_flip_all", 30, m)]["acc_std"]) for tp, _ in topos]
        c_m = [float(rows_idx[(tp, "label_flip_all", 30, m)]["cons_mean"]) for tp, _ in topos]
        c_s = [float(rows_idx[(tp, "label_flip_all", 30, m)]["cons_std"]) for tp, _ in topos]

        b1 = ax_peer.bar(
            x + offsets[idx], p_m, width=width * 0.92, yerr=p_s, capsize=3.2,
            color=fr.SERIES_COLOR[m], edgecolor="black", linewidth=1.5,
            error_kw={"elinewidth": 1.5, "ecolor": "black"}, label=fr.SERIES_LABEL[m], zorder=3,
        )
        b2 = ax_cons.bar(
            x + offsets[idx], c_m, width=width * 0.92, yerr=c_s, capsize=3.2,
            color=fr.SERIES_COLOR[m], edgecolor="black", linewidth=1.5,
            error_kw={"elinewidth": 1.5, "ecolor": "black"}, zorder=3,
        )
        if m == "tg_stab":
            for b, val in zip(b1, p_m):
                ax_peer.text(b.get_x() + b.get_width() / 2.0, val + 1.8, f"{val:.1f}", ha="center", va="bottom", fontsize=12.0, fontweight="bold", color=fr.SERIES_COLOR[m])
            for b, val in zip(b2, c_m):
                ax_cons.text(b.get_x() + b.get_width() / 2.0, val + 1.2, f"{val:.1f}", ha="center", va="bottom", fontsize=12.0, fontweight="bold", color=fr.SERIES_COLOR[m])

        txt = "  ".join(f"{tp}: peer={pm:.2f}+/-{ps:.2f} cons={cm:.2f}+/-{cs:.2f}" for (tp, _), pm, ps, cm, cs in zip(topos, p_m, p_s, c_m, c_s))
        print(f"  {fr.SERIES_LABEL[m]:18s}: {txt}")

    for ax, ttl in ((ax_peer, "(a) Per-Peer Mean Benign Accuracy (%)"), (ax_cons, "(b) Consensus Model Accuracy (%)")):
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in topos])
        ax.set_title(ttl, pad=10, fontweight="bold")
        ax.set_ylim(50, 100)
        ax.set_xlim(-0.55, len(topos) - 0.45)
    ax_peer.set_ylabel("Test Accuracy (%)")

    handles, labels = ax_peer.get_legend_handles_labels()
    top = fs.header(
        fig,
        "Topology Robustness Under Label Flipping (Full 50k CIFAR-10, 30% Byzantine)",
        "Pretrained ResNet-18 (2,500 samples/node, Dir alpha=0.5, E_b=2, 25 rounds; mean +/- std over 3 seeds)",
    )
    bot = fs.footnote(
        fig,
        "TrustGraph (stab) achieves the highest per-peer accuracy on all three topologies (84.7% Ring, 89.3% Random, 89.5% Small-World) and 91.7-92.4% consensus.",
    )
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, top - 0.015), ncol=5, columnspacing=1.5, handlelength=1.8)
    fig.subplots_adjust(left=0.07, right=0.985, top=top - 0.165, bottom=bot + 0.95 / 6.7, wspace=0.14)
    fs.save(fig, out_dir, "fig-table1-topologies-50k")


def plot_fig3b_stress_fpr(rows_idx: dict[tuple[str, str, int, str], dict[str, object]], out_dir: Path) -> None:
    """Fig. 3(b): Per-peer accuracy and honest FPR across 10% -> 50% Byzantine peers."""
    fracs = [10, 20, 30, 40, 50]
    fig, (ax_acc, ax_fpr) = plt.subplots(1, 2, figsize=(14.0, 6.7))
    fs.light_grid(ax_acc, "y")
    fs.light_grid(ax_fpr, "y")

    print("\n=== Fig. 3(b) (50k) Byzantine Stress Sweep under label_flip_all (bep=2, 3 seeds) ===")
    for m in fr.DEFENSE_ORDER:
        acc_m = np.array([float(rows_idx[("random", "label_flip_all", f, m)]["acc_mean"]) for f in fracs])
        acc_s = np.array([float(rows_idx[("random", "label_flip_all", f, m)]["acc_std"]) for f in fracs])
        fpr_m = np.array([float(rows_idx[("random", "label_flip_all", f, m)]["fpr_mean"]) for f in fracs])
        fpr_s = np.array([float(rows_idx[("random", "label_flip_all", f, m)]["fpr_std"]) for f in fracs])

        color = fr.SERIES_COLOR[m]
        kw = dict(
            color=color,
            linestyle=fr.SERIES_LINESTYLE[m],
            marker=fr.SERIES_MARKER[m],
            linewidth=3.0 if m == "tg_stab" else 2.3,
            markersize=8.5 if m == "tg_stab" else 7.5,
            label=fr.SERIES_LABEL[m],
            zorder=5 if m == "tg_stab" else 3,
        )
        ax_acc.plot(fracs, acc_m, **kw)
        ax_acc.fill_between(fracs, np.clip(acc_m - acc_s, 0, 100), np.clip(acc_m + acc_s, 0, 100), color=color, alpha=0.14)
        ax_fpr.plot(fracs, fpr_m, **kw)
        ax_fpr.fill_between(fracs, np.clip(fpr_m - fpr_s, 0, 100), np.clip(fpr_m + fpr_s, 0, 100), color=color, alpha=0.14)

        print(f"  {fr.SERIES_LABEL[m]:14s} PeerAcc: " + ", ".join(f"{f}%:{v:.1f}+/-{s:.1f}" for f, v, s in zip(fracs, acc_m, acc_s)))
        print(f"  {fr.SERIES_LABEL[m]:14s} FPR    : " + ", ".join(f"{f}%:{v:.1f}+/-{s:.1f}" for f, v, s in zip(fracs, fpr_m, fpr_s)))

    ax_acc.set_title("(a) Per-Peer Mean Accuracy vs. Byzantine Fraction", pad=10, fontweight="bold")
    ax_acc.set_xlabel("Byzantine Peer Fraction (%)")
    ax_acc.set_ylabel("Per-Peer Benign Test Accuracy (%)")
    ax_acc.set_xticks(fracs)
    ax_acc.set_xlim(7, 53)
    ax_acc.set_ylim(58, 96)

    ax_fpr.set_title("(b) Cumulative Honest FPR vs. Byzantine Fraction", pad=10, fontweight="bold")
    ax_fpr.set_xlabel("Byzantine Peer Fraction (%)")
    ax_fpr.set_ylabel("Honest False Positive Rate (%)")
    ax_fpr.set_xticks(fracs)
    ax_fpr.set_xlim(7, 53)
    ax_fpr.set_ylim(-4, 60)

    handles, labels = ax_acc.get_legend_handles_labels()
    top = fs.header(
        fig,
        "Byzantine Stress Sweep Under All-Class Label Flipping (Full 50k CIFAR-10)",
        "Pretrained ResNet-18 (2,500 samples/node, Dir alpha=0.5, random ER p=0.3, E_b=2, 25 rounds; mean +/- std over 3 seeds)",
    )
    bot = fs.footnote(
        fig,
        "At 50% Byzantine, TrustGraph (stab) leads per-peer accuracy at 89.5% +/- 0.4% (92.0% consensus) while honest FPR decreases from 43.1% to 36.2%.",
    )
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, top - 0.015), ncol=4, columnspacing=1.6)
    fig.subplots_adjust(left=0.075, right=0.985, top=top - 0.165, bottom=bot + 0.85 / 6.7, wspace=0.22)
    fs.save(fig, out_dir, "fig3b-stress-and-fpr-50k")


def plot_fig4_ablations_and_fpr_dynamics(
    rows_idx: dict[tuple[str, str, int, str], dict[str, object]],
    data: dict[str, object],
    out_dir: Path,
) -> None:
    """Fig. 4: Section 5.5 ablations (model_poison) + per-round honest FPR trajectory."""
    fig, (ax_abl, ax_dyn) = plt.subplots(1, 2, figsize=(14.4, 6.4))
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
    colors = [fr.SERIES_COLOR[k] for k, _ in abl_keys]

    print("\n=== Fig. 4(a) (50k) Section 5.5 Ablations under model_poison (3 seeds) ===")
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
    atk_specs = [
        ("none", "Clean (No Attack)", fs.PALETTE["gray"], "-", "o"),
        ("label_flip_all", "Label Flipping", fs.PALETTE["blue_main"], "-", "s"),
        ("model_poison", "Model Poisoning", fs.PALETTE["red_strong"], "--", "^"),
        ("random_noise", "Random Noise", fs.PALETTE["teal"], "-.", "D"),
    ]
    print("\n=== Fig. 4(b) (50k) Per-Round Honest FPR Trajectory for TrustGraph (stab) ===")
    for atk, lbl, col, ls, mk in atk_specs:
        c = curves[f"{atk}__tg_strong"]
        rds = np.array(c["fpr_rounds"], dtype=float)
        fm = np.array(c["fpr_mean"], dtype=float)
        fsd = np.array(c["fpr_std"], dtype=float)
        ax_dyn.plot(rds, fm, label=lbl, color=col, linestyle=ls, marker=mk, markevery=[0, 4, 9, 14, 19, 24], markersize=7, linewidth=2.4)
        ax_dyn.fill_between(rds, np.clip(fm - fsd, 0, 100), np.clip(fm + fsd, 0, 100), color=col, alpha=0.12)
        print(f"  {atk:14s}: r1={fm[0]:.1f}% r5={fm[4]:.1f}% r15={fm[14]:.1f}% r25={fm[-1]:.1f}%")

    ax_dyn.set_title("(b) Per-Round Honest FPR Climbs Near Convergence", pad=10, fontweight="bold")
    ax_dyn.set_xlabel("Communication Round")
    ax_dyn.set_ylabel("Per-Round Honest FPR (%)")
    ax_dyn.set_xlim(0.5, 25.5)
    ax_dyn.set_xticks([1, 5, 10, 15, 20, 25])
    ax_dyn.set_ylim(0, 75)
    ax_dyn.legend(loc="lower right", ncol=1)

    top = fs.header(
        fig,
        "TrustGraph Ablations and Per-Round Honest FPR Dynamics (Full 50k CIFAR-10)",
        "Pretrained ResNet-18 (2,500 samples/node, Dir alpha=0.5, random ER p=0.3, 30% Byzantine, 25 rounds, 3 seeds)",
    )
    bot = fs.footnote(
        fig,
        "Left: Removing ReLU drops consensus by 35.9 pp; Euclidean drops 9.0 pp. Right: Honest FPR starts at 0-3% at round 1 and climbs near convergence.",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=top - 0.08, bottom=bot + 0.90 / 6.4, wspace=0.22)
    fs.save(fig, out_dir, "fig4-ablations-and-fpr-dynamics-50k")


def plot_fig5_fpr_levers(data: dict[str, object], out_dir: Path) -> None:
    """Fig. 5: 2-panel comparison of Levers 1-4 on full 50k CIFAR-10 (model_poison)."""
    runs: list[dict[str, object]] = data["runs"]
    base_mp = _find_baseline_row(data, "model_poison")
    base_curve = data["baseline_50k_3seed"]["curves"]["model_poison__tg_strong"]

    l1_mp = _find_run(runs, "lever1/model_poison__iidval")
    l1b_t020_mp = _find_run(runs, "lever1b/t020__model_poison")
    l1b_t035_mp = _find_run(runs, "lever1b/t035__model_poison")
    l2_bs256_mp = _find_run(runs, "lever2/bs256_r25__model_poison")
    l2_steps4_mp = _find_run(runs, "lever2/steps4_r100__model_poison")
    l3_layer_mp = _find_run(runs, "lever3/layer_mean__model_poison")
    l3_bb_mp = _find_run(runs, "lever3/backbone_only__model_poison")
    l4_e70_mp = _find_run(runs, "lever4/ema70__model_poison")
    l4_e85_mp = _find_run(runs, "lever4/ema85__model_poison")

    fig, (ax_trade, ax_traj) = plt.subplots(1, 2, figsize=(15.6, 6.5))
    fs.light_grid(ax_trade, "y")
    fs.light_grid(ax_traj, "y")

    ax_trade.axvspan(3.0, 9.0, color=fs.PALETTE["green_1"], alpha=0.65, zorder=1)
    ax_trade.text(6.0, 48.0, "Paper\n3–9%\ntarget", ha="center", va="top", fontsize=11.5, fontweight="bold", color=fs.INK)

    base_pt = (float(base_mp["fpr_mean"]), float(base_mp["byz_accept_mean"]))
    rx_pts = [
        (r"$ \tau=0.003 $", 100.0 * float(l1_mp["fpr_cumulative"]), 100.0 * float(l1_mp["byz_accept_cumulative"]), (8, 5)),
        (r"$ \tau=0.020 $", 100.0 * float(l1b_t020_mp["fpr_cumulative"]), 100.0 * float(l1b_t020_mp["byz_accept_cumulative"]), (8, 4)),
        (r"$ \tau=0.035 $", 100.0 * float(l1b_t035_mp["fpr_cumulative"]), 100.0 * float(l1b_t035_mp["byz_accept_cumulative"]), (-2, 10)),
    ]
    ema_pts = [
        (r"$ \beta=0.70 $", 100.0 * float(l4_e70_mp["fpr_cumulative"]), 100.0 * float(l4_e70_mp["byz_accept_cumulative"]), (8, -2)),
        (r"$ \beta=0.85 $", 100.0 * float(l4_e85_mp["fpr_cumulative"]), 100.0 * float(l4_e85_mp["byz_accept_cumulative"]), (8, -4)),
    ]
    l2_pts = [
        ("L2: bs=256", 100.0 * float(l2_bs256_mp["fpr_cumulative"]), 100.0 * float(l2_bs256_mp["byz_accept_cumulative"]), (-36, -24)),
        ("L2: steps=4", 100.0 * float(l2_steps4_mp["fpr_cumulative"]), 100.0 * float(l2_steps4_mp["byz_accept_cumulative"]), (-10, -24)),
    ]
    l3_pts = [
        ("L3: layer_mean", 100.0 * float(l3_layer_mp["fpr_cumulative"]), 100.0 * float(l3_layer_mp["byz_accept_cumulative"]), (-108, -6)),
        ("L3: backbone", 100.0 * float(l3_bb_mp["fpr_cumulative"]), 100.0 * float(l3_bb_mp["byz_accept_cumulative"]), (-108, 14)),
    ]

    print("\n=== Fig. 5(a) Four-Lever Trade-off on model_poison (30% Byz, 50k CIFAR-10) ===")
    print(f"  Base sender (3-seed)     : FPR={base_pt[0]:5.2f}%, ByzAcc={base_pt[1]:5.2f}%")

    ax_trade.scatter(
        [base_pt[0]], [base_pt[1]],
        s=130, color=fs.PALETTE["gray"], edgecolor="black", linewidth=1.4, marker="o", zorder=5,
        label="Base sender (3-seed)",
    )
    ax_trade.annotate(
        "Base sender\n(3-seed)", xy=base_pt, xytext=(-58, 22), textcoords="offset points",
        fontsize=11.5, fontweight="bold", color=fs.INK,
        arrowprops={"arrowstyle": "-", "color": fs.PALETTE["gray"], "lw": 1.2},
    )

    ax_trade.plot(
        [p[1] for p in rx_pts], [p[2] for p in rx_pts],
        color=fs.PALETTE["blue_main"], linestyle="-", marker="s", markersize=8.5, linewidth=2.6, zorder=6,
        label="L1/L1b rxdelta",
    )
    for lbl, x_v, y_v, off in rx_pts:
        print(f"  L1/1b rxdelta {lbl:10s} : FPR={x_v:5.2f}%, ByzAcc={y_v:5.2f}%")
        ax_trade.annotate(lbl.strip(), xy=(x_v, y_v), xytext=off, textcoords="offset points", fontsize=11.5, color=fs.PALETTE["blue_main"], fontweight="bold")
    ax_trade.text(16.0, 44.0, "L1/L1b: rxdelta", fontsize=12.5, fontweight="bold", color=fs.PALETTE["blue_main"])

    ax_trade.plot(
        [p[1] for p in ema_pts], [p[2] for p in ema_pts],
        color=fs.PALETTE["teal"], linestyle="--", marker="D", markersize=8.0, linewidth=2.5, zorder=5,
        label="L4 score EMA (sender)",
    )
    for lbl, x_v, y_v, off in ema_pts:
        print(f"  L4 EMA {lbl:17s} : FPR={x_v:5.2f}%, ByzAcc={y_v:5.2f}%")
        ax_trade.annotate(lbl.strip(), xy=(x_v, y_v), xytext=off, textcoords="offset points", fontsize=11.5, color=fs.PALETTE["teal"], fontweight="bold")
    ax_trade.text(11.5, 22.0, "L4: score EMA\n(sender)", fontsize=12.0, fontweight="bold", color=fs.PALETTE["teal"])

    for lbl, x_v, y_v, off in l2_pts:
        print(f"  {lbl:24s} : FPR={x_v:5.2f}%, ByzAcc={y_v:5.2f}%")
    ax_trade.scatter(
        [p[1] for p in l2_pts], [p[2] for p in l2_pts],
        s=95, color="white", edgecolor=fs.PALETTE["red_strong"], linewidth=2.0, marker="^", zorder=4,
    )
    for lbl, x_v, y_v, off in l2_pts:
        ax_trade.annotate(
            lbl, xy=(x_v, y_v), xytext=off, textcoords="offset points", fontsize=10.5,
            color=fs.PALETTE["red_strong"], arrowprops={"arrowstyle": "-", "color": fs.PALETTE["red_strong"], "lw": 0.9},
        )

    for lbl, x_v, y_v, off in l3_pts:
        print(f"  {lbl:24s} : FPR={x_v:5.2f}%, ByzAcc={y_v:5.2f}%")
    ax_trade.scatter(
        [p[1] for p in l3_pts], [p[2] for p in l3_pts],
        s=90, color="white", edgecolor=fs.PALETTE["violet"], linewidth=2.0, marker="P", zorder=4,
    )
    for lbl, x_v, y_v, off in l3_pts:
        ax_trade.annotate(
            lbl, xy=(x_v, y_v), xytext=off, textcoords="offset points", fontsize=10.5,
            color=fs.PALETTE["violet"], arrowprops={"arrowstyle": "-", "color": fs.PALETTE["violet"], "lw": 0.9},
        )

    ax_trade.set_title("(a) FPR vs. Byzantine Acceptance (Model Poisoning)", pad=10, fontweight="bold")
    ax_trade.set_xlabel("Cumulative Honest FPR (%)")
    ax_trade.set_ylabel("Byzantine Acceptance Rate (%)")
    ax_trade.set_xlim(1, 52)
    ax_trade.set_ylim(10, 50)

    ax_traj.axhspan(3.0, 9.0, color=fs.PALETTE["green_1"], alpha=0.65, zorder=1)
    rounds_b = np.array(base_curve["fpr_rounds"], dtype=float)
    fpr_b_m = np.array(base_curve["fpr_mean"], dtype=float)
    fpr_b_s = np.array(base_curve["fpr_std"], dtype=float)

    def _traj(run: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
        h = run["history"]
        return np.array([float(x["round"]) + 1.0 for x in h]), np.array([100.0 * float(x["fpr_round"]) for x in h])

    r_l1, fpr_l1 = _traj(l1_mp)
    r_l1b, fpr_l1b = _traj(l1b_t020_mp)
    r_l4, fpr_l4 = _traj(l4_e85_mp)
    marks = [0, 4, 9, 14, 19, 24]

    ax_traj.plot(
        rounds_b, fpr_b_m, label="Base sender (3-seed)",
        color=fs.PALETTE["gray"], linestyle="-", marker="o", markevery=marks, markersize=7.5, linewidth=2.4, zorder=3,
    )
    ax_traj.fill_between(rounds_b, np.clip(fpr_b_m - fpr_b_s, 0, 100), np.clip(fpr_b_m + fpr_b_s, 0, 100), color=fs.PALETTE["gray"], alpha=0.14)
    ax_traj.plot(
        r_l4, fpr_l4, label=r"L4: EMA ($\beta=0.85$, sender)",
        color=fs.PALETTE["teal"], linestyle="--", marker="D", markevery=marks, markersize=7.0, linewidth=2.4, zorder=4,
    )
    ax_traj.plot(
        r_l1b, fpr_l1b, label=r"L1b: rxdelta ($\tau=0.020$)",
        color=fs.PALETTE["blue_secondary"], linestyle="-.", marker="^", markevery=marks, markersize=7.5, linewidth=2.4, zorder=5,
    )
    ax_traj.plot(
        r_l1, fpr_l1, label=r"L1: rxdelta ($\tau=0.003$)",
        color=fs.PALETTE["blue_main"], linestyle="-", marker="s", markevery=marks, markersize=7.5, linewidth=2.8, zorder=6,
    )

    print("\n=== Fig. 5(b) Per-Round Honest FPR Trajectory on model_poison (r=1, 5, 10, 15, 25) ===")
    for name, arr in (
        ("Base sender (3-seed)", fpr_b_m),
        ("L4 EMA (beta=0.85)", fpr_l4),
        ("L1b rxdelta (tau=0.020)", fpr_l1b),
        ("L1 rxdelta (tau=0.003)", fpr_l1),
    ):
        print(f"  {name:24s}: r1={arr[0]:5.1f}% r5={arr[4]:5.1f}% r10={arr[9]:5.1f}% r15={arr[14]:5.1f}% r25={arr[-1]:5.1f}%")

    ax_traj.set_title("(b) Per-Round Honest FPR Over Training", pad=10, fontweight="bold")
    ax_traj.set_xlabel("Communication Round")
    ax_traj.set_ylabel("Per-Round Honest FPR (%)")
    ax_traj.set_xlim(0.5, 25.5)
    ax_traj.set_xticks([1, 5, 10, 15, 20, 25])
    ax_traj.set_ylim(-3, 84)
    ax_traj.legend(loc="upper left", ncol=1)

    top = fs.header(
        fig,
        "Four-Lever Honest FPR Diagnostic on TrustGraph-DFL (Full 50k CIFAR-10)",
        "Pretrained ResNet-18 (2,500/node, Dir alpha=0.5, iid val=250, random ER p=0.3, 30% model poisoning, 25 rounds)",
    )
    bot = fs.footnote(
        fig,
        "L1 (rxdelta, tau=0.003) brings FPR into 7.4% (3-9% target band); L4 (EMA beta=0.85) halves sender FPR to 21.1% and cuts MP Byz accept to 14.3%.",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=top - 0.08, bottom=bot + 0.90 / 6.5, wspace=0.22)
    fs.save(fig, out_dir, "fig5-fpr-levers-50k")


def main() -> None:
    p = argparse.ArgumentParser(description="Draw all final Phase 1 TrustGraph-DFL figures from committed JSON data.")
    p.add_argument("--data-50k", default="report/data/phase1_paper_match_50k_3seed.json")
    p.add_argument("--data-levers", default="report/data/phase1_fpr_levers.json")
    p.add_argument("--out-dir", default="report/figures")
    args = p.parse_args()

    data_50k = json.loads(Path(args.data_50k).read_text(encoding="utf-8"))
    data_levers = json.loads(Path(args.data_levers).read_text(encoding="utf-8"))
    rows_idx = _index_rows(data_50k["rows"])
    out_dir = Path(args.out_dir)

    fs.apply("compact")
    plot_fig2_convergence(data_50k, out_dir)
    plot_fig3a_heatmap_and_bars(rows_idx, out_dir)
    plot_table1_topologies(rows_idx, out_dir)
    plot_fig3b_stress_fpr(rows_idx, out_dir)
    plot_fig4_ablations_and_fpr_dynamics(rows_idx, data_50k, out_dir)
    plot_fig5_fpr_levers(data_levers, out_dir)


if __name__ == "__main__":
    main()
