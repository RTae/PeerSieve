"""Generate the Phase 1 four-lever FPR diagnostic figure from committed JSON data.

Follows the figures4papers house style via ``figstyle.py`` (unedited skill copy).
Prints every plotted number to stdout and exports both a 300 dpi PNG (with header
and footnote) and a header-free cropped PDF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.phase1 import figstyle as fs


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


def plot_fig5_fpr_levers(data: dict[str, object], out_dir: Path) -> None:
    """Draw 2-panel comparison of Levers 1-4 on full 50k CIFAR-10 (model_poison)."""
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

    # Target honest FPR band claimed in TrustGraph-DFL Fig. 3(b) (~3-9%)
    ax_trade.axvspan(3.0, 9.0, color=fs.PALETTE["green_1"], alpha=0.65, zorder=1)
    ax_trade.text(6.0, 48.0, "Paper\n3–9%\ntarget", ha="center", va="top", fontsize=11.5, fontweight="bold", color=fs.INK)

    # Panel (a): Honest FPR vs. Byzantine Acceptance on model_poison (30% Byz)
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

    # Base sender marker
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

    # Lever 1 / 1b rxdelta frontier
    ax_trade.plot(
        [p[1] for p in rx_pts], [p[2] for p in rx_pts],
        color=fs.PALETTE["blue_main"], linestyle="-", marker="s", markersize=8.5, linewidth=2.6, zorder=6,
        label="L1/L1b rxdelta",
    )
    for lbl, x_v, y_v, off in rx_pts:
        print(f"  L1/1b rxdelta {lbl:10s} : FPR={x_v:5.2f}%, ByzAcc={y_v:5.2f}%")
        ax_trade.annotate(lbl.strip(), xy=(x_v, y_v), xytext=off, textcoords="offset points", fontsize=11.5, color=fs.PALETTE["blue_main"], fontweight="bold")
    ax_trade.text(16.0, 44.0, "L1/L1b: rxdelta", fontsize=12.5, fontweight="bold", color=fs.PALETTE["blue_main"])

    # Lever 4 EMA frontier
    ax_trade.plot(
        [p[1] for p in ema_pts], [p[2] for p in ema_pts],
        color=fs.PALETTE["teal"], linestyle="--", marker="D", markersize=8.0, linewidth=2.5, zorder=5,
        label="L4 score EMA (sender)",
    )
    for lbl, x_v, y_v, off in ema_pts:
        print(f"  L4 EMA {lbl:17s} : FPR={x_v:5.2f}%, ByzAcc={y_v:5.2f}%")
        ax_trade.annotate(lbl.strip(), xy=(x_v, y_v), xytext=off, textcoords="offset points", fontsize=11.5, color=fs.PALETTE["teal"], fontweight="bold")
    ax_trade.text(11.5, 22.0, "L4: score EMA\n(sender)", fontsize=12.0, fontweight="bold", color=fs.PALETTE["teal"])

    # Lever 2 & Lever 3 ruled-out single-round clusters
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

    # Panel (b): Per-round Honest FPR trajectory on model_poison (r=1..25)
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
    p = argparse.ArgumentParser(description="Draw Phase 1 four-lever FPR diagnostic figure.")
    p.add_argument("--data", default="report/data/phase1_fpr_levers.json")
    p.add_argument("--out-dir", default="report/figures_50k")
    args = p.parse_args()

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    fs.apply("compact")
    plot_fig5_fpr_levers(data, Path(args.out_dir))


if __name__ == "__main__":
    main()
