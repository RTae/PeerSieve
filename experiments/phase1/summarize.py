"""Summarize Phase 1 experiment outputs into Markdown tables."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


@dataclass
class SummaryRow:
    model: str
    alpha_dir: float
    topology: str
    attack: str
    byz_pct: int
    method: str
    seeds: int
    acc_mean: float
    acc_std: float
    cons_mean: float
    cons_std: float
    fpr_mean: float
    fpr_std: float
    byz_accept_mean: float
    ben_cos_mean: float | None
    byz_cos_mean: float | None
    nan_count: int


def method_label(cfg: dict[str, object]) -> str:
    """Format a canonical display label distinguishing aggregator variants."""
    agg = str(cfg["aggregator"])
    val_split = str(cfg.get("val_split", "local"))
    extra: list[str] = []
    gc = float(cfg.get("grad_clip_norm", 0.0))
    if gc > 0.0:
        extra.append(f"gc={gc:g}")
    if cfg.get("val_grad_weight_decay", False):
        extra.append("wdmatch")
    if "weight_decay" in cfg and float(cfg["weight_decay"]) == 0.0:
        extra.append("wd=0")
    if "momentum" in cfg and float(cfg["momentum"]) == 0.0:
        extra.append("mom=0")
    bep = int(cfg.get("byz_local_epochs", 0))
    if bep > 0:
        extra.append(f"bep={bep}")
    bs = int(cfg.get("batch_size", 64))
    if bs != 64:
        extra.append(f"bs={bs}")
    spn = int(cfg.get("samples_per_node", 500))
    if spn != 500:
        extra.append(f"spn={spn}")
    suffix = f"[{','.join(extra)}]" if extra else ""
    if agg == "trustgraph":
        sign = str(cfg.get("val_grad_sign", "descent_neg"))
        parts = [sign]
        if not cfg.get("trustgraph_use_relu", True):
            parts.append("norelu")
        if cfg.get("trustgraph_metric", "cosine") != "cosine":
            parts.append(str(cfg["trustgraph_metric"]))
        if cfg.get("trustgraph_clip_norm", False):
            clip_mode = str(cfg.get("trustgraph_clip_mode", "update"))
            parts.append("mclip" if clip_mode == "model" else "clip")
        if str(cfg.get("trustgraph_delta_mode", "sender")) == "receiver":
            parts.append("rxdelta")
        thresh = float(cfg.get("trustgraph_score_threshold", 0.0))
        if abs(thresh) > 1e-9:
            parts.append(f"t={thresh:g}")
        if float(cfg.get("alpha", 0.5)) != 0.5:
            parts.append(f"a={cfg['alpha']}")
        if val_split != "local":
            parts.append(f"val={val_split}")
        return f"trustgraph({','.join(parts)}){suffix}"
    return f"{agg}{suffix}"


# Backwards-compatible alias.
_method_label = method_label


def _is_nan_run(summary: dict[str, object]) -> bool:
    if bool(summary.get("had_nan", False)):
        return True
    ben = summary.get("mean_benign_score")
    return ben is not None and math.isnan(float(ben))


def summarize_directory(root: str | Path) -> list[SummaryRow]:
    """Load and aggregate all summary.json files under root, deduplicating identical (config, seed)."""
    files = sorted(Path(root).rglob("summary.json"))
    grouped: dict[tuple[str, float, str, str, int, str], list[dict[str, object]]] = defaultdict(list)
    seen_seed: set[tuple[str, float, str, str, int, str, int]] = set()

    for path in files:
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        cfg = s["config"]
        key = (
            str(cfg.get("model_name", "cnn2x2")),
            float(cfg.get("alpha_dir", 0.5)),
            str(cfg["topology"]),
            str(cfg["attack"]),
            int(round(float(cfg["byzantine_frac"]) * 100)),
            method_label(cfg),
        )
        dedup_key = (*key, int(cfg["seed"]))
        if dedup_key in seen_seed:
            continue
        seen_seed.add(dedup_key)
        grouped[key].append(s)

    rows_out: list[SummaryRow] = []
    for (model, alpha_dir, topo, atk, byz_pct, method), items in sorted(grouped.items()):
        accs = [100.0 * float(r["final_benign_test_acc_mean"]) for r in items]
        cons = [100.0 * float(r["final_consensus_test_acc"]) for r in items]
        fprs = [100.0 * float(r["fpr_cumulative"]) for r in items]
        byz_acc = [100.0 * float(r["byz_accept_cumulative"]) for r in items]
        ben_cos = [float(r["mean_benign_score"]) for r in items if r.get("mean_benign_score") is not None]
        byz_cos = [float(r["mean_byz_score"]) for r in items if r.get("mean_byz_score") is not None]
        nan_count = sum(1 for r in items if _is_nan_run(r))

        rows_out.append(
            SummaryRow(
                model=model,
                alpha_dir=alpha_dir,
                topology=topo,
                attack=atk,
                byz_pct=byz_pct,
                method=method,
                seeds=len(items),
                acc_mean=float(np.mean(accs)),
                acc_std=float(np.std(accs)),
                cons_mean=float(np.mean(cons)),
                cons_std=float(np.std(cons)),
                fpr_mean=float(np.mean(fprs)),
                fpr_std=float(np.std(fprs)),
                byz_accept_mean=float(np.mean(byz_acc)),
                ben_cos_mean=float(np.mean(ben_cos)) if ben_cos else None,
                byz_cos_mean=float(np.mean(byz_cos)) if byz_cos else None,
                nan_count=nan_count,
            )
        )
    return rows_out


def _fmt_mean_std(mean: float, std: float, seeds: int) -> str:
    return f"{mean:.2f} ± {std:.2f}" if seeds > 1 else f"{mean:.2f}"


def format_markdown_report(rows: list[SummaryRow], root: str | Path, title: str) -> str:
    """Render aggregated SummaryRows as a GitHub-flavored Markdown report."""
    multi_context = len({(r.model, r.alpha_dir) for r in rows}) > 1
    if multi_context:
        header = "| Model | Dir | Topology | Attack | Byz% | Method | Seeds | Final Acc (%) | Cons Acc (%) | FPR (%) | Byz Accept (%) | Cos(Ben/Byz) | NaNs |"
        sep = "|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"
    else:
        header = "| Topology | Attack | Byz% | Method | Seeds | Final Acc (%) | Cons Acc (%) | FPR (%) | Byz Accept (%) | Cos(Ben/Byz) | NaNs |"
        sep = "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"

    lines = [
        f"# {title}",
        "",
        f"Source outputs: `{root}`",
        "",
        header,
        sep,
    ]
    for r in rows:
        if r.ben_cos_mean is not None and r.byz_cos_mean is not None:
            cos_txt = f"{r.ben_cos_mean:+.3f} / {r.byz_cos_mean:+.3f}"
        elif r.ben_cos_mean is not None:
            cos_txt = f"{r.ben_cos_mean:+.3f} / n/a"
        else:
            cos_txt = "n/a"
        prefix = f"| {r.model} | {r.alpha_dir:g} " if multi_context else ""
        lines.append(
            f"{prefix}| {r.topology} | {r.attack} | {r.byz_pct} | {r.method} | {r.seeds} | "
            f"{_fmt_mean_std(r.acc_mean, r.acc_std, r.seeds)} | "
            f"{_fmt_mean_std(r.cons_mean, r.cons_std, r.seeds)} | "
            f"{_fmt_mean_std(r.fpr_mean, r.fpr_std, r.seeds)} | "
            f"{r.byz_accept_mean:.2f} | {cos_txt} | {r.nan_count}/{r.seeds} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize Phase 1 summary.json files.")
    p.add_argument("root", help="Directory containing Phase 1 outputs (searched recursively for summary.json).")
    p.add_argument("--out", default=None, help="Optional Markdown file path to write the table.")
    p.add_argument("--title", default="Phase 1 Experiment Summary", help="Heading for the Markdown report.")
    args = p.parse_args()

    rows = summarize_directory(args.root)
    if not rows:
        print(f"No summary.json files found under {args.root}")
        return

    output = format_markdown_report(rows, root=args.root, title=args.title)
    print(output, end="")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
