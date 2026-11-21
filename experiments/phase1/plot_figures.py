"""Generate publication-style Phase 1c figures from committed 3-seed JSON data.

Uses the figures4papers house style via ``figstyle.py`` (unedited skill copy)
and fixed project roles in ``figroles.py``. Prints every plotted number to stdout
and writes 300 dpi PNGs (with header/footnote) plus header-free PDFs.
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
    if label.startswith("trustgraph(descent_neg,mclip,t=0.02)["):
        return "tg_stab"
    if label.startswith("trustgraph(descent_neg,mclip,t=0.02,a=0.3)"):
        return "tg_a03"
    if label.startswith("trustgraph(descent_neg,mclip,t=0.02,a=0.7)"):
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


def _index_rows(rows: list[dict[str, object]]) -> dict[tuple[str, str, int, str], dict[str, object]]:
    out: dict[tuple[str, str, int, str], dict[str, object]] = {}
    for r in rows:
        mk = _method_key(str(r["method"]))
        if mk is None:
            continue
        key = (str(r["topology"]), str(r["attack"]), int(r["byz_pct"]), mk)
        out[key] = r
    return out


def plot_fig2_convergence(data: dict[str, object], out_dir: Path) -> None:
    """Fig. 2: Consensus accuracy convergence curves across three attacks (3 seeds)."""
    curves = data["curves"]
    attacks = [
        ("model_poison", "(a) Model Poisoning ($\\lambda=10$)"),
        ("random_noise", "(b) Random Noise ($\\sigma=0.5$)"),
        ("label_flip_all", "(c) Label Flipping (All Classes)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 6.7), sharey=True)
    print("\n=== Fig. 2 Convergence Curves (3-seed mean +/- std consensus acc %) ===")
    for ax, (atk, title) in zip(axes, attacks):
        fs.light_grid(ax, "y")
        print(f"\n[{atk}]")
        for m in fr.METHOD_ORDER:
            c = curves[f"{atk}__{m}"]
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
                markersize=7.5 if m == "tg_stab" else 6.5,
                linewidth=lw,
                zorder=zorder,
            )
            ax.fill_between(rounds, np.clip(mean - std, 0.0, 100.0), np.clip(mean + std, 0.0, 100.0), color=color, alpha=0.14, zorder=zorder - 1)
            print(f"  {fr.SERIES_LABEL[m]:18s}: r1={mean[0]:.2f} r10={mean[2]:.2f} r30={mean[6]:.2f} r50={mean[-1]:.2f} +/- {std[-1]:.2f}")

        ax.set_title(title, pad=10, fontweight="bold")
        ax.set_xlabel("Communication Round")
        ax.set_xlim(-1, 52)
        ax.set_xticks([1, 10, 20, 30, 40, 50])
        ax.set_ylim(2, 98)

    axes[0].set_ylabel("Consensus Test Accuracy (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    top = fs.header(
        fig,
        "Consensus Convergence Under 30% Byzantine Attacks",
        f"Pretrained ResNet-18 on 10k CIFAR-10 (Dirichlet alpha=0.5, random ER p=0.3, grad clip 10.0; mean +/- std over 3 seeds, n={data['total_runs']} total runs)",
    )
    bot = fs.footnote(
        fig,
        "TrustGraph (stab) = descent_neg (-nabla F_val), model-space clip 2*r_i, cosine threshold tau=0.02; BALANCE calibrated to gamma=0.025.",
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, top - 0.015),
        ncol=5,
        columnspacing=1.5,
        handlelength=2.4,
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=top - 0.165, bottom=bot + 0.85 / 6.7, wspace=0.12)
    fs.save(fig, out_dir, "fig2-convergence")


def plot_fig3a_attacks(rows_idx: dict[tuple[str, str, int, str], dict[str, object]], data: dict[str, object], out_dir: Path) -> None:
    """Fig. 3(a): Final consensus accuracy across all four attack regimes (grouped bars)."""
    attacks = [
        ("none", "Clean\n(No Attack)"),
        ("label_flip_all", "Label Flip\n(All Classes)"),
        ("model_poison", "Model Poison\n($\\lambda=10$)"),
        ("random_noise", "Random Noise\n($\\sigma=0.5$)"),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    fs.light_grid(ax, "y")

    x = np.arange(len(attacks), dtype=float)
    n_m = len(fr.METHOD_ORDER)
    width = 0.155
    offsets = (np.arange(n_m) - (n_m - 1) / 2.0) * width

    print("\n=== Fig. 3(a) Final Consensus Accuracy by Attack (3-seed mean +/- std %) ===")
    for idx, m in enumerate(fr.METHOD_ORDER):
        means = [float(rows_idx[("random", atk, 30, m)]["cons_mean"]) for atk, _ in attacks]
        stds = [float(rows_idx[("random", atk, 30, m)]["cons_std"]) for atk, _ in attacks]
        bars = ax.bar(
            x + offsets[idx],
            means,
            width=width * 0.92,
            yerr=stds,
            capsize=3.5,
            color=fr.SERIES_COLOR[m],
            edgecolor="black",
            linewidth=1.6,
            error_kw={"elinewidth": 1.6, "ecolor": "black"},
            label=fr.SERIES_LABEL[m],
            zorder=3,
        )
        # Annotate TrustGraph (stab) bars so exact values stand out cleanly.
        if m == "tg_stab":
            for b, val in zip(bars, means):
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    val + 2.2,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=12.5,
                    fontweight="bold",
                    color=fr.SERIES_COLOR[m],
                )
        row_str = "  ".join(f"{atk}={v:.2f}+/-{s:.2f}" for (atk, _), v, s in zip(attacks, means, stds))
        print(f"  {fr.SERIES_LABEL[m]:18s}: {row_str}")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in attacks])
    ax.set_ylabel("Final Consensus Test Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.55, len(attacks) - 0.45)

    top = fs.header(
        fig,
        "Defense Comparison Across Attack Types (30% Byzantine)",
        "Pretrained ResNet-18 on 10k CIFAR-10 (Dirichlet alpha=0.5, random ER p=0.3, 50 rounds, 3 seeds; 0/150 NaNs)",
    )
    bot = fs.footnote(
        fig,
        "Only TrustGraph (stab) stays above 87% across all four regimes; Krum fails on 2/3 model_poison seeds (std=36.8%) and random_noise.",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=5, columnspacing=1.2, handlelength=1.6)
    fig.subplots_adjust(left=0.09, right=0.98, top=top - 0.13, bottom=bot + 0.95 / 6.4)
    fs.save(fig, out_dir, "fig3a-attack-comparison")


def plot_fig3b_stress_fpr(rows_idx: dict[tuple[str, str, int, str], dict[str, object]], out_dir: Path) -> None:
    """Fig. 3(b): Consensus accuracy and honest FPR vs. Byzantine fraction (10% -> 50%)."""
    fracs = [10, 20, 30, 40, 50]
    fig, (ax_acc, ax_fpr) = plt.subplots(1, 2, figsize=(13.8, 6.7))
    fs.light_grid(ax_acc, "y")
    fs.light_grid(ax_fpr, "y")

    print("\n=== Fig. 3(b) Byzantine Stress Sweep under label_flip_all (3-seed mean +/- std %) ===")
    for m in fr.DEFENSE_ORDER:
        acc_m = np.array([float(rows_idx[("random", "label_flip_all", f, m)]["cons_mean"]) for f in fracs])
        acc_s = np.array([float(rows_idx[("random", "label_flip_all", f, m)]["cons_std"]) for f in fracs])
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

        acc_txt = ", ".join(f"{f}%:{v:.1f}+/-{s:.1f}" for f, v, s in zip(fracs, acc_m, acc_s))
        fpr_txt = ", ".join(f"{f}%:{v:.1f}+/-{s:.1f}" for f, v, s in zip(fracs, fpr_m, fpr_s))
        print(f"  {fr.SERIES_LABEL[m]:14s} ConsAcc: {acc_txt}")
        print(f"  {fr.SERIES_LABEL[m]:14s} FPR    : {fpr_txt}")

    ax_acc.set_title("(a) Consensus Accuracy vs. Byzantine Fraction", pad=10, fontweight="bold")
    ax_acc.set_xlabel("Byzantine Peer Fraction (%)")
    ax_acc.set_ylabel("Final Consensus Test Accuracy (%)")
    ax_acc.set_xticks(fracs)
    ax_acc.set_xlim(7, 53)
    ax_acc.set_ylim(10, 98)

    ax_fpr.set_title("(b) Cumulative Honest FPR vs. Byzantine Fraction", pad=10, fontweight="bold")
    ax_fpr.set_xlabel("Byzantine Peer Fraction (%)")
    ax_fpr.set_ylabel("Honest False Positive Rate (%)")
    ax_fpr.set_xticks(fracs)
    ax_fpr.set_xlim(7, 53)
    ax_fpr.set_ylim(-4, 82)

    handles, labels = ax_acc.get_legend_handles_labels()
    top = fs.header(
        fig,
        "Byzantine Stress Sweep Under All-Class Label Flipping (10% to 50%)",
        "Pretrained ResNet-18 on 10k CIFAR-10 (Dirichlet alpha=0.5, random ER p=0.3, 50 rounds; mean +/- std over 3 seeds)",
    )
    bot = fs.footnote(
        fig,
        "At 50% Byzantine, TrustGraph (stab) holds 86.4% consensus with invariant FPR (~51%), while Krum collapses to 20.0% (71.3% FPR).",
    )
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, top - 0.015), ncol=4, columnspacing=1.6)
    fig.subplots_adjust(left=0.075, right=0.985, top=top - 0.165, bottom=bot + 0.85 / 6.7, wspace=0.22)
    fs.save(fig, out_dir, "fig3b-stress-and-fpr")


def plot_fig4_ablations_and_fpr_dynamics(
    rows_idx: dict[tuple[str, str, int, str], dict[str, object]],
    data: dict[str, object],
    out_dir: Path,
) -> None:
    """Fig. 4: Section 5.5 ablations (model_poison) + per-round honest FPR trajectory."""
    fig, (ax_abl, ax_dyn) = plt.subplots(1, 2, figsize=(14.2, 6.2))
    fs.light_grid(ax_abl, "y")
    fs.light_grid(ax_dyn, "y")

    # Panel (a): Section 5.5 ablations under model_poison
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

    print("\n=== Fig. 4(a) Section 5.5 Ablations under model_poison (3-seed mean +/- std %) ===")
    bars = ax_abl.bar(
        x,
        means,
        width=0.58,
        yerr=stds,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=1.8,
        error_kw={"elinewidth": 1.8, "ecolor": "black"},
        zorder=3,
    )
    for b, (k, _), m, s in zip(bars, abl_keys, means, stds):
        ax_abl.text(
            b.get_x() + b.get_width() / 2.0,
            m + s + 1.8,
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
    ax_abl.set_ylim(0, 105)

    # Panel (b): Round-by-round honest FPR trajectory for tg_stab across attacks
    curves = data["curves"]
    atk_specs = [
        ("none", "Clean (No Attack)", fs.PALETTE["gray"], "-", "o"),
        ("label_flip_all", "Label Flipping", fs.PALETTE["blue_main"], "-", "s"),
        ("model_poison", "Model Poisoning", fs.PALETTE["red_strong"], "--", "^"),
        ("random_noise", "Random Noise", fs.PALETTE["teal"], "-.", "D"),
    ]
    print("\n=== Fig. 4(b) Round-by-Round Honest FPR Trajectory for TrustGraph (stab) ===")
    for atk, lbl, col, ls, mk in atk_specs:
        c = curves[f"{atk}__tg_stab"]
        rds = np.array(c["fpr_rounds"], dtype=float)
        fm = np.array(c["fpr_mean"], dtype=float)
        fsd = np.array(c["fpr_std"], dtype=float)
        ax_dyn.plot(rds, fm, label=lbl, color=col, linestyle=ls, marker=mk, markevery=[0, 4, 9, 19, 29, 39, 49], markersize=7, linewidth=2.4)
        ax_dyn.fill_between(rds, np.clip(fm - fsd, 0, 100), np.clip(fm + fsd, 0, 100), color=col, alpha=0.12)
        print(f"  {atk:14s}: r1={fm[0]:.1f}% r10={fm[9]:.1f}% r30={fm[29]:.1f}% r50={fm[-1]:.1f}%")

    ax_dyn.set_title("(b) Per-Round Honest FPR Climbs Near Convergence", pad=10, fontweight="bold")
    ax_dyn.set_xlabel("Communication Round")
    ax_dyn.set_ylabel("Per-Round Honest FPR (%)")
    ax_dyn.set_xlim(0, 51)
    ax_dyn.set_xticks([1, 10, 20, 30, 40, 50])
    ax_dyn.set_ylim(0, 82)
    ax_dyn.legend(loc="lower right", ncol=1)

    top = fs.header(
        fig,
        "TrustGraph Component Ablations and Honest FPR Dynamics",
        "Pretrained ResNet-18 on 10k CIFAR-10 (Dirichlet alpha=0.5, random ER p=0.3, 30% Byzantine, 3 seeds)",
    )
    bot = fs.footnote(
        fig,
        "Left: ReLU zeroing & cosine direction are essential under model poisoning. Right: Late-round FPR (>60%) motivates PeerSieve's recoverable EMA probation.",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=top - 0.08, bottom=bot + 0.90 / 6.2, wspace=0.22)
    fs.save(fig, out_dir, "fig4-ablations-and-fpr-dynamics")


def main() -> None:
    p = argparse.ArgumentParser(description="Draw Phase 1c figures from committed JSON data.")
    p.add_argument("--data", default="report/data/phase1_paper_match_3seed.json")
    p.add_argument("--out-dir", default="report/figures")
    args = p.parse_args()

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    rows_idx = _index_rows(data["rows"])
    out_dir = Path(args.out_dir)

    fs.apply("compact")
    plot_fig2_convergence(data, out_dir)
    plot_fig3a_attacks(rows_idx, data, out_dir)
    plot_fig3b_stress_fpr(rows_idx, out_dir)
    plot_fig4_ablations_and_fpr_dynamics(rows_idx, data, out_dir)


if __name__ == "__main__":
    main()
