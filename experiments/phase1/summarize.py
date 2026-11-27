from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import typer

from src.logging_utils import configure_logging, get_logger

logger = get_logger()
app = typer.Typer(
    name="peersieve-summarize",
    help="Aggregate summary.json files into Markdown accuracy and system-cost tables.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@dataclass
class SummaryRow:
    """Multi-seed accuracy, FPR, Byzantine acceptance, and NaN counts for one config."""

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
    """Format a canonical display label distinguishing aggregator and training variants.

    Args:
        cfg: Serialized ExperimentConfig dictionary from summary.json.

    Returns:
        Compact method string encoding scoring sign, clipping, threshold, and key overrides.
    """
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


def _has_nan_result(summary: dict[str, object]) -> bool:
    """Check whether a run summary recorded NaN weights or a NaN benign cosine score.

    Args:
        summary: Parsed summary.json dictionary.

    Returns:
        True if the run had NaN values, False otherwise.
    """
    if bool(summary.get("had_nan", False)):
        return True
    ben = summary.get("mean_benign_score")
    return ben is not None and math.isnan(float(ben))


def summarize_directory(root: str | Path) -> list[SummaryRow]:
    """Load and aggregate all summary.json files under root, deduplicating (config, seed).

    Args:
        root: Directory tree searched recursively for summary.json files.

    Returns:
        Sorted list of SummaryRow objects aggregated across unique seeds per configuration.
    """
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
        nan_count = sum(1 for r in items if _has_nan_result(r))

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


def _format_mean_std(mean: float, std: float, seeds: int) -> str:
    """Format mean +/- std when multiple seeds exist, otherwise format mean only.

    Args:
        mean: Mean metric value.
        std: Standard deviation across seeds.
        seeds: Number of seeds averaged.

    Returns:
        Formatted string with two decimal places.
    """
    return f"{mean:.2f} ± {std:.2f}" if seeds > 1 else f"{mean:.2f}"


@dataclass
class SystemSummaryRow:
    """Aggregated bandwidth, memory, compute, and rejection-streak metrics for one config."""

    model: str
    alpha_dir: float
    topology: str
    attack: str
    byz_pct: int
    method: str
    seeds: int
    mean_deg: float
    max_deg: int
    rx_mb: float
    useful_mb: float
    wasted_mb: float
    wasted_byz_mb: float
    wasted_benign_mb: float
    wasted_pct: float
    recv_buf_mean_mib: float
    recv_buf_max_mib: float
    peak_gpu_alloc_mib: float | None
    train_sec: float
    val_sec: float
    agg_sec: float
    comm_1gbps_sec: float
    comm_1gbps_share_pct: float
    byz_mean_rej_rounds: float
    byz_repeat_gb: float
    byz_streak_ge3_pct: float
    benign_ge1_pct: float
    benign_streak_ge3_pct: float


def summarize_system_directory(root: str | Path) -> list[SystemSummaryRow]:
    """Aggregate system_metrics across all summary.json files under root.

    Args:
        root: Directory tree searched recursively for summary.json files with system_metrics.

    Returns:
        Sorted list of SystemSummaryRow objects grouped by model, Dirichlet alpha, topology, attack, and method.
    """
    files = sorted(Path(root).rglob("summary.json"))
    grouped: dict[tuple[str, float, str, str, int, str], list[dict[str, object]]] = defaultdict(list)
    seen_seed: set[tuple[str, float, str, str, int, str, int]] = set()

    for path in files:
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        if "system_metrics" not in s:
            continue
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

    rows_out: list[SystemSummaryRow] = []
    for (model, alpha_dir, topo, atk, byz_pct, method), items in sorted(grouped.items()):
        sms = [r["system_metrics"] for r in items]
        gpu_peaks = [float(sm["peak_gpu_alloc_mib"]) for sm in sms if sm.get("peak_gpu_alloc_mib") is not None]

        def _benign_streak(sm: dict[str, object], field: str) -> float:
            """Read a benign rejection-streak metric supporting both benign and legacy JSON keys.

            Args:
                sm: system_metrics dictionary from summary.json.
                field: Streak metric key inside the rejection-streaks sub-dictionary.

            Returns:
                Float value of the requested streak metric.
            """
            block = sm.get("benign_rejection_streaks") or sm["honest_rejection_streaks"]
            return float(block[field])

        rows_out.append(
            SystemSummaryRow(
                model=model,
                alpha_dir=alpha_dir,
                topology=topo,
                attack=atk,
                byz_pct=byz_pct,
                method=method,
                seeds=len(items),
                mean_deg=float(np.mean([float(sm["mean_benign_degree"]) for sm in sms])),
                max_deg=max(int(sm["max_benign_degree"]) for sm in sms),
                rx_mb=float(np.mean([float(sm["rx_mb_per_peer_round"]) for sm in sms])),
                useful_mb=float(np.mean([float(sm["useful_mb_per_peer_round"]) for sm in sms])),
                wasted_mb=float(np.mean([float(sm["wasted_mb_per_peer_round"]) for sm in sms])),
                wasted_byz_mb=float(np.mean([float(sm["wasted_byz_mb_per_peer_round"]) for sm in sms])),
                wasted_benign_mb=float(
                    np.mean(
                        [
                            float(sm.get("wasted_benign_mb_per_peer_round", sm["wasted_honest_mb_per_peer_round"]))
                            for sm in sms
                        ]
                    )
                ),
                wasted_pct=float(np.mean([float(sm["wasted_transfer_pct"]) for sm in sms])),
                recv_buf_mean_mib=float(np.mean([float(sm["batched_recv_buf_mean_mib"]) for sm in sms])),
                recv_buf_max_mib=float(np.mean([float(sm["batched_recv_buf_max_mib"]) for sm in sms])),
                peak_gpu_alloc_mib=float(np.mean(gpu_peaks)) if gpu_peaks else None,
                train_sec=float(np.mean([float(sm["train_sec_per_peer"]["median"]) for sm in sms])),
                val_sec=float(np.mean([float(sm["val_grad_sec_per_peer"]["median"]) for sm in sms])),
                agg_sec=float(np.mean([float(sm["agg_sec_per_peer"]["median"]) for sm in sms])),
                comm_1gbps_sec=float(np.mean([float(sm["round_time_by_link"]["1gbps"]["comm_sec"]) for sm in sms])),
                comm_1gbps_share_pct=float(np.mean([float(sm["round_time_by_link"]["1gbps"]["comm_share_pct"]) for sm in sms])),
                byz_mean_rej_rounds=float(np.mean([float(sm["byz_repeat_rejections"]["mean_rejected_rounds"]) for sm in sms])),
                byz_repeat_gb=float(np.mean([float(sm["byz_repeat_rejections"]["repeat_wasted_gb_after_first_rej"]) for sm in sms])),
                byz_streak_ge3_pct=float(np.mean([float(sm["byz_repeat_rejections"]["frac_edges_with_streak_ge3_pct"]) for sm in sms])),
                benign_ge1_pct=float(np.mean([_benign_streak(sm, "frac_edges_with_ge1_rej_pct") for sm in sms])),
                benign_streak_ge3_pct=float(np.mean([_benign_streak(sm, "frac_edges_with_streak_ge3_pct") for sm in sms])),
            )
        )
    return rows_out


def format_system_markdown_report(rows: list[SystemSummaryRow], title: str = "System Cost Summary") -> str:
    """Render aggregated SystemSummaryRow entries as a Markdown table.

    Args:
        rows: List of SystemSummaryRow objects from summarize_system_directory().
        title: Section heading placed above the Markdown table.

    Returns:
        Formatted Markdown string ending with a newline.
    """
    lines = [
        f"## {title}",
        "",
        "| Topology | Attack | Byz% | Method | Mean/Max Degree | Rx MB/round | Wasted MB/round (%) | Recv Buf Mean/Max (MiB) | Train/Val/Agg (s) | 1 Gb/s Comm (s, %) | Attacker Blocked (rounds) | Benign >=3 FP Streak (%) |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.topology} | {r.attack} | {r.byz_pct} | {r.method} | {r.mean_deg:.1f}/{r.max_deg} | "
            f"{r.rx_mb:.1f} | {r.wasted_mb:.1f} ({r.wasted_pct:.1f}%) | "
            f"{r.recv_buf_mean_mib:.0f}/{r.recv_buf_max_mib:.0f} | "
            f"{r.train_sec:.2f}/{r.val_sec:.3f}/{r.agg_sec:.4f} | "
            f"{r.comm_1gbps_sec:.2f} ({r.comm_1gbps_share_pct:.1f}%) | "
            f"{r.byz_mean_rej_rounds:.1f} | {r.benign_streak_ge3_pct:.1f} |"
        )
    return "\n".join(lines) + "\n"


def export_system_metrics_json(
    root: str | Path,
    out_path: str | Path,
    job_id: str | None = None,
    commit: str | None = None,
    transfer_bench_json: str | Path | None = None,
) -> dict[str, object]:
    """Export all summary.json system_metrics and per-round curves to one JSON dataset.

    Args:
        root: Directory tree containing completed run outputs.
        out_path: Destination JSON file path.
        job_id: Optional Slurm job ID metadata string.
        commit: Optional Git commit hash metadata string.
        transfer_bench_json: Optional measured inter-node TCP benchmark JSON file to embed.

    Returns:
        Exported JSON payload dictionary that was written to out_path.
    """
    root_p = Path(root)
    files = sorted(root_p.rglob("summary.json"))
    runs: list[dict[str, object]] = []
    for path in files:
        s = json.loads(path.read_text(encoding="utf-8"))
        if "system_metrics" not in s:
            continue
        rel_name = str(path.parent.relative_to(root_p))
        hist_path = path.parent / "history.jsonl"
        round_wasted: list[float] = []
        round_fpr: list[float] = []
        round_byz_acc: list[float] = []
        if hist_path.exists():
            for line in hist_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                round_wasted.append(round(100.0 * float(rec.get("wasted_transfer_rate", 0.0)), 2))
                round_fpr.append(round(100.0 * float(rec.get("fpr_round", 0.0)), 2))
                round_byz_acc.append(round(100.0 * float(rec.get("byz_accept_rate", 0.0)), 2))
        cfg = s["config"]
        runs.append(
            {
                "run_name": rel_name,
                "execution_mode": s.get("execution_mode", "local"),
                "model": cfg.get("model_name", "cnn2x2"),
                "topology": cfg["topology"],
                "attack": cfg["attack"],
                "byz_pct": int(round(float(cfg["byzantine_frac"]) * 100)),
                "method": method_label(cfg),
                "seed": int(cfg["seed"]),
                "param_count": int(s["param_count"]),
                "had_nan": bool(s.get("had_nan", False)),
                "final_benign_test_acc_mean": round(100.0 * float(s["final_benign_test_acc_mean"]), 2),
                "final_consensus_test_acc": round(100.0 * float(s["final_consensus_test_acc"]), 2),
                "fpr_cumulative_pct": round(100.0 * float(s["fpr_cumulative"]), 2),
                "byz_accept_cumulative_pct": round(100.0 * float(s["byz_accept_cumulative"]), 2),
                "total_wall_sec": round(float(s["total_wall_sec"]), 2),
                "system_metrics": s["system_metrics"],
                "curves": {
                    "round_wasted_pct": round_wasted,
                    "round_fpr_pct": round_fpr,
                    "round_byz_accept_pct": round_byz_acc,
                },
            }
        )
    walls = [float(r["total_wall_sec"]) for r in runs]
    payload: dict[str, object] = {
        "description": "Phase 1 CIFAR-10 ResNet-18 batched FP32 TrustGraph-DFL system-cost metrics",
        "source_root": str(root_p),
        "job_id": job_id,
        "commit": commit,
        "total_runs": len(runs),
        "nan_runs": sum(1 for r in runs if r["had_nan"]),
        "timing": {
            "count": len(walls),
            "median_sec": round(float(np.median(walls)), 2) if walls else 0.0,
            "min_sec": round(float(np.min(walls)), 2) if walls else 0.0,
            "max_sec": round(float(np.max(walls)), 2) if walls else 0.0,
        },
        "runs": runs,
    }
    if transfer_bench_json is not None:
        payload["measured_cluster_transfer"] = json.loads(Path(transfer_bench_json).read_text(encoding="utf-8"))
    dst = Path(out_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def format_markdown_report(rows: list[SummaryRow], root: str | Path, title: str) -> str:
    """Render aggregated SummaryRow entries as a GitHub-flavored Markdown report.

    Args:
        rows: List of SummaryRow objects from summarize_directory().
        root: Source output directory displayed in the report header.
        title: Top-level Markdown heading for the report.

    Returns:
        Formatted Markdown report string ending with a newline.
    """
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
            f"{_format_mean_std(r.acc_mean, r.acc_std, r.seeds)} | "
            f"{_format_mean_std(r.cons_mean, r.cons_std, r.seeds)} | "
            f"{_format_mean_std(r.fpr_mean, r.fpr_std, r.seeds)} | "
            f"{r.byz_accept_mean:.2f} | {cos_txt} | {r.nan_count}/{r.seeds} |"
        )
    return "\n".join(lines) + "\n"


@app.command()
def summarize_cli(
    root: Annotated[Path, typer.Argument(help="Directory searched recursively for summary.json files.")],
    out: Annotated[Path | None, typer.Option(help="Optional Markdown file path to save the summary tables.")] = None,
    title: Annotated[str, typer.Option(help="Top-level Markdown heading for the report.")] = "Phase 1 Experiment Summary",
    export_system_json: Annotated[Path | None, typer.Option(help="Optional JSON path to export aggregated system metrics.")] = None,
    transfer_bench_json: Annotated[Path | None, typer.Option(help="Optional measured TCP transfer JSON to embed in exported JSON.")] = None,
    job_id: Annotated[str | None, typer.Option(help="Optional Slurm job ID recorded in exported JSON metadata.")] = None,
    commit: Annotated[str | None, typer.Option(help="Optional git commit hash recorded in exported JSON metadata.")] = None,
    log_level: Annotated[Literal["DEBUG", "INFO", "WARNING", "ERROR"], typer.Option(help="Minimum loguru log level.")] = "INFO",
) -> None:
    """Summarize completed summary.json files into Markdown tables and optional JSON.

    Args:
        root: Directory searched recursively for summary.json files.
        out: Optional path to write the Markdown report.
        title: Heading for the Markdown report.
        export_system_json: Optional output path for aggregated system metrics JSON.
        transfer_bench_json: Optional measured TCP benchmark JSON to embed.
        job_id: Optional Slurm job ID metadata.
        commit: Optional git commit hash metadata.
        log_level: Minimum loguru log level.

    Returns:
        None.
    """
    configure_logging(level=log_level)

    rows = summarize_directory(root)
    if not rows:
        logger.warning("No summary.json files found under {}", root)
        return

    output = format_markdown_report(rows, root=root, title=title)
    sys_rows = summarize_system_directory(root)
    if sys_rows:
        output += "\n" + format_system_markdown_report(sys_rows)
    logger.info("\n{}", output.rstrip())
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output, encoding="utf-8")
        logger.success("Wrote Markdown summary to {}", out)
    if export_system_json is not None:
        exported = export_system_metrics_json(
            root,
            export_system_json,
            job_id=job_id,
            commit=commit,
            transfer_bench_json=transfer_bench_json,
        )
        logger.success("Exported {} runs to {}", exported["total_runs"], export_system_json)


def main() -> None:
    """Invoke the Typer CLI application for summarizing completed runs.

    Returns:
        None.
    """
    app()


if __name__ == "__main__":
    main()
