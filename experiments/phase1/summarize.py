"""Summarize Phase 1 experiment outputs into Markdown tables."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def _method_label(cfg: dict[str, object]) -> str:
    agg = str(cfg["aggregator"])
    val_split = str(cfg.get("val_split", "local"))
    if agg == "trustgraph":
        sign = str(cfg.get("val_grad_sign", "descent_neg"))
        suffix = ""
        if not cfg.get("trustgraph_use_relu", True):
            suffix += ",norelu"
        if cfg.get("trustgraph_metric", "cosine") != "cosine":
            suffix += f",{cfg['trustgraph_metric']}"
        if float(cfg.get("alpha", 0.5)) != 0.5:
            suffix += f",a={cfg['alpha']}"
        if val_split != "local":
            suffix += f",val={val_split}"
        return f"trustgraph({sign}{suffix})"
    if agg == "balance":
        gamma = float(cfg.get("balance_gamma", 0.3))
        if abs(gamma - 0.3) > 1e-9:
            return f"balance(g={gamma:g})"
    return agg


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize Phase 1 summary.json files.")
    p.add_argument("root", help="Directory containing Phase 1 outputs (searched recursively for summary.json).")
    p.add_argument("--out", default=None, help="Optional Markdown file path to write the table (e.g. report/phase1_smoke.md).")
    p.add_argument("--title", default="Phase 1 Experiment Summary", help="Heading when writing a Markdown report.")
    args = p.parse_args()

    files = sorted(Path(args.root).rglob("summary.json"))
    if not files:
        print(f"No summary.json files found under {args.root}")
        return

    grouped: dict[tuple[str, str, int, str], list[dict[str, object]]] = defaultdict(list)
    seen_seed: set[tuple[str, str, int, str, int]] = set()
    for path in files:
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        cfg = s["config"]
        key = (
            str(cfg["topology"]),
            str(cfg["attack"]),
            int(round(float(cfg["byzantine_frac"]) * 100)),
            _method_label(cfg),
        )
        dedup_key = (*key, int(cfg["seed"]))
        if dedup_key in seen_seed:
            continue
        seen_seed.add(dedup_key)
        grouped[key].append(s)

    lines = [
        f"# {args.title}",
        "",
        f"Source outputs: `{args.root}`",
        "",
        "| Topology | Attack | Byz% | Method | Seeds | Final Acc (%) | Cons Acc (%) | FPR (%) | Byz Accept (%) | Cos(Ben/Byz) |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (topo, atk, byz_pct, method), rows in sorted(grouped.items()):
        accs = [100.0 * float(r["final_benign_test_acc_mean"]) for r in rows]
        cons = [100.0 * float(r["final_consensus_test_acc"]) for r in rows]
        fprs = [100.0 * float(r["fpr_cumulative"]) for r in rows]
        byz_acc = [100.0 * float(r["byz_accept_cumulative"]) for r in rows]
        ben_cos = [float(r["mean_benign_score"]) for r in rows if r.get("mean_benign_score") is not None]
        byz_cos = [float(r["mean_byz_score"]) for r in rows if r.get("mean_byz_score") is not None]
        cos_txt = (
            f"{np.mean(ben_cos):+.3f} / {np.mean(byz_cos):+.3f}"
            if ben_cos and byz_cos
            else (f"{np.mean(ben_cos):+.3f} / n/a" if ben_cos else "n/a")
        )
        acc_txt = f"{np.mean(accs):.2f} ± {np.std(accs):.2f}" if len(accs) > 1 else f"{accs[0]:.2f}"
        cons_txt = f"{np.mean(cons):.2f}"
        fpr_txt = f"{np.mean(fprs):.2f}"
        byz_txt = f"{np.mean(byz_acc):.2f}"
        lines.append(
            f"| {topo} | {atk} | {byz_pct} | {method} | {len(rows)} | "
            f"{acc_txt} | {cons_txt} | {fpr_txt} | {byz_txt} | {cos_txt} |"
        )

    output = "\n".join(lines) + "\n"
    print(output, end="")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
