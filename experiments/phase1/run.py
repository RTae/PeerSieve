from dataclasses import fields, replace
import json
from pathlib import Path
from typing import Annotated, Literal

import typer

from experiments.phase1.suites import SUITES, build_suite_runs
from src.distributed import run_distributed_experiment
from src.logging_utils import configure_logging, get_logger
from src.simulator import ExperimentConfig, prepare_dataset, run_experiment
from src.transport import run_transfer_benchmark

__all__ = ["PRESETS", "SUITES", "app", "build_suite_runs", "main"]

logger = get_logger()

PresetChoice = Literal["cnn_10k", "resnet18_50k"]
DatasetChoice = Literal["cifar10", "synthetic"]
ValSplitChoice = Literal["local", "iid"]
ModelChoice = Literal["cnn2x2", "cnn2x2_gn", "resnet18_pretrained", "resnet18_scratch"]
TopologyChoice = Literal["ring", "random", "small_world"]
AttackChoice = Literal["none", "label_flip", "label_flip_all", "model_poison", "random_noise"]
ValGradSignChoice = Literal["descent", "ascent"]
DeltaModeChoice = Literal["sender", "receiver"]
DeviceChoice = Literal["auto", "cpu", "cuda", "mps"]
LogLevelChoice = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
BenchRoleChoice = Literal["auto", "server", "client"]

PRESETS: dict[str, dict[str, object]] = {
    "cnn_10k": {},
    "resnet18_50k": {
        "model_name": "resnet18_pretrained",
        "samples_per_node": 2500,
        "rounds": 25,
        "lr": 0.005,
        "lr_schedule": "cosine",
        "augment": True,
        "grad_clip_norm": 10.0,
        "val_split": "iid",
        "val_grad_sign": "descent",
        "trustgraph_clip_norm": True,
        "trustgraph_clip_mode": "model",
        "trustgraph_clip_factor": 2.0,
        "trustgraph_score_threshold": 0.003,
        "eval_batch_size": 256,
    },
}

_CONFIG_FIELDS: dict[str, type] = {
    f.name: type(f.default) for f in fields(ExperimentConfig) if f.default is not None
}

app = typer.Typer(
    name="peersieve-run",
    help="Run TrustGraph-DFL single experiments, named benchmark suites, multi-process TCP runs, or link benchmarks.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


def _parse_seeds(raw_seeds: list[str]) -> list[int]:
    """Parse repeated or comma-separated random seed strings into integers.

    Args:
        raw_seeds: List of seed strings such as ['42,43,44'] or ['42', '43'].

    Returns:
        Flat list of integer random seeds in input order.
    """
    parsed: list[int] = []
    for item in raw_seeds:
        for part in item.split(","):
            token = part.strip()
            if not token:
                continue
            try:
                parsed.append(int(token))
            except ValueError as exc:
                raise typer.BadParameter(f"Invalid integer seed {token!r}.") from exc
    if not parsed:
        raise typer.BadParameter("At least one integer seed is required.")
    return parsed


def _coerce_value(key: str, raw: str, target_type: type) -> object:
    """Convert a CLI string value into the target Python scalar type of an ExperimentConfig field.

    Args:
        key: ExperimentConfig field name.
        raw: Raw string value from --set key=value.
        target_type: Expected Python type (bool, int, float, or str).

    Returns:
        Coerced scalar value matching target_type.
    """
    if target_type is bool:
        low = raw.strip().lower()
        if low in {"1", "true", "yes", "on"}:
            return True
        if low in {"0", "false", "no", "off"}:
            return False
        raise typer.BadParameter(f"Field {key!r} expects a boolean (true/false), got {raw!r}.")
    try:
        return target_type(raw)
    except ValueError as exc:
        raise typer.BadParameter(f"Field {key!r} expects {target_type.__name__}, got {raw!r}.") from exc


def _resolve_config(
    preset: PresetChoice,
    config_file: Path | None,
    direct_overrides: dict[str, object],
    set_items: list[str],
    quiet: bool,
) -> ExperimentConfig:
    """Build an ExperimentConfig by layering preset, optional JSON file, flags, and --set overrides.

    Args:
        preset: Base hyperparameter preset ('cnn_10k' or 'resnet18_50k').
        config_file: Optional JSON file path (plain dict or saved config.json).
        direct_overrides: Explicit CLI flag overrides (keys with None values are ignored).
        set_items: List of 'key=value' overrides targeting any ExperimentConfig field.
        quiet: Suppress INFO logs on stderr when True.

    Returns:
        Resolved ExperimentConfig ready for training.
    """
    values: dict[str, object] = dict(PRESETS[preset])

    if config_file is not None:
        raw_json = json.loads(config_file.read_text(encoding="utf-8"))
        cfg_dict = raw_json.get("config", raw_json) if isinstance(raw_json, dict) else {}
        if not isinstance(cfg_dict, dict):
            raise typer.BadParameter(f"Config file {config_file} must contain a JSON object.")
        for k, v in cfg_dict.items():
            if k in _CONFIG_FIELDS:
                values[k] = v

    for k, v in direct_overrides.items():
        if v is not None:
            values[k] = v

    for item in set_items:
        if "=" not in item:
            raise typer.BadParameter(f"Expected --set key=value, got {item!r}.")
        key, raw_val = item.split("=", 1)
        key = key.strip().replace("-", "_")
        if key not in _CONFIG_FIELDS:
            valid = ", ".join(sorted(_CONFIG_FIELDS))
            raise typer.BadParameter(f"Unknown config field {key!r}. Valid fields: {valid}")
        values[key] = _coerce_value(key, raw_val.strip(), _CONFIG_FIELDS[key])

    values["verbose"] = not quiet
    return replace(ExperimentConfig(), **values)


def _run_tag(cfg: ExperimentConfig) -> str:
    """Build a unique directory tag from dataset, model, topology, attack, and scoring mode.

    Args:
        cfg: Populated ExperimentConfig for the run.

    Returns:
        Filesystem-safe tag string for single and distributed output folders.
    """
    delta_suffix = "_rxdelta" if cfg.trustgraph_delta_mode == "receiver" else ""
    return (
        f"{cfg.dataset}_{cfg.model_name}_{cfg.topology}_{cfg.attack}"
        f"_byz{int(round(cfg.byzantine_frac * 100))}_{cfg.aggregator}_{cfg.val_grad_sign}{delta_suffix}"
    )


@app.command("list-suites")
def list_suites() -> None:
    """List all registered named experiment suites and their run counts per seed."""
    configure_logging()
    base = ExperimentConfig()
    logger.info("Registered experiment suites ({} total):", len(SUITES))
    for name in sorted(SUITES):
        n_runs = len(build_suite_runs(name, base))
        logger.info("  {:24s} ({:2d} runs / seed)", name, n_runs)


@app.command("single")
def single_cmd(
    preset: Annotated[PresetChoice, typer.Option(help="Base hyperparameter preset ('cnn_10k' or 'resnet18_50k').", rich_help_panel="Config & Overrides")] = "resnet18_50k",
    config: Annotated[Path | None, typer.Option(help="Optional JSON config file (or saved config.json) loaded over the preset.", rich_help_panel="Config & Overrides")] = None,
    set_override: Annotated[list[str] | None, typer.Option("--set", help="Override any ExperimentConfig field as key=value (repeatable, e.g. --set er_p=1.0).", rich_help_panel="Config & Overrides")] = None,
    dataset: Annotated[DatasetChoice | None, typer.Option(help="Dataset ('cifar10' or 'synthetic').", rich_help_panel="Core Options")] = None,
    model: Annotated[ModelChoice | None, typer.Option(help="Model architecture.", rich_help_panel="Core Options")] = None,
    n_nodes: Annotated[int | None, typer.Option(min=2, help="Number of decentralized peers.", rich_help_panel="Core Options")] = None,
    samples_per_node: Annotated[int | None, typer.Option(min=10, help="Train + validation samples per peer.", rich_help_panel="Core Options")] = None,
    rounds: Annotated[int | None, typer.Option(min=1, help="Number of communication rounds.", rich_help_panel="Core Options")] = None,
    topology: Annotated[TopologyChoice | None, typer.Option(help="Graph topology ('ring', 'random', 'small_world').", rich_help_panel="Core Options")] = None,
    attack: Annotated[AttackChoice | None, typer.Option(help="Byzantine attack mode.", rich_help_panel="Core Options")] = None,
    byzantine_frac: Annotated[float | None, typer.Option(min=0.0, max=1.0, help="Fraction of Byzantine peers.", rich_help_panel="Core Options")] = None,
    val_split: Annotated[ValSplitChoice | None, typer.Option(help="Validation split ('local' or 'iid').", rich_help_panel="Core Options")] = None,
    val_grad_sign: Annotated[ValGradSignChoice | None, typer.Option(help="Validation direction: 'descent' (-grad) or 'ascent' (+grad).", rich_help_panel="Core Options")] = None,
    delta_mode: Annotated[DeltaModeChoice | None, typer.Option("--delta-mode", "--trustgraph-delta-mode", help="Scoring anchor: 'sender' (2 vectors) or 'receiver' (rxdelta, 1 vector).", rich_help_panel="Core Options")] = None,
    no_download: Annotated[bool, typer.Option("--no-download", help="Disable automatic CIFAR-10 download.", rich_help_panel="Execution & Logging")] = False,
    seeds: Annotated[list[str] | None, typer.Option("--seed", "--seeds", help="Random seed(s); repeat or comma-separate (default: 42).", rich_help_panel="Execution & Logging")] = None,
    device: Annotated[DeviceChoice | None, typer.Option(help="Compute device ('auto', 'cpu', 'cuda', or 'mps').", rich_help_panel="Execution & Logging")] = None,
    eval_every: Annotated[int | None, typer.Option(help="Evaluate test accuracy every N rounds (0 = final round only).", rich_help_panel="Execution & Logging")] = None,
    skip_existing: Annotated[bool, typer.Option("--skip-existing", help="Skip seeds that already have summary.json.", rich_help_panel="Execution & Logging")] = False,
    output_root: Annotated[Path, typer.Option(help="Root directory for single/<tag>/seed_<id> outputs.", rich_help_panel="Execution & Logging")] = Path("outputs/phase1"),
    log_level: Annotated[LogLevelChoice, typer.Option(help="Minimum loguru log level.", rich_help_panel="Execution & Logging")] = "INFO",
    log_file: Annotated[Path | None, typer.Option(help="Optional extra log file capturing all CLI logs.", rich_help_panel="Execution & Logging")] = None,
    quiet: Annotated[bool, typer.Option("--quiet", help="Suppress INFO logs on stderr (per-run run.log still writes INFO).", rich_help_panel="Execution & Logging")] = False,
) -> None:
    """Run one custom TrustGraph-DFL experiment in a single process across one or more seeds."""
    configure_logging(level=log_level, log_file=log_file, quiet=quiet)
    base_cfg = _resolve_config(
        preset=preset,
        config_file=config,
        direct_overrides={
            "dataset": dataset,
            "model_name": model,
            "n_nodes": n_nodes,
            "samples_per_node": samples_per_node,
            "rounds": rounds,
            "topology": topology,
            "attack": attack,
            "byzantine_frac": byzantine_frac,
            "val_split": val_split,
            "val_grad_sign": val_grad_sign,
            "trustgraph_delta_mode": delta_mode,
            "download_data": False if no_download else None,
            "device": device,
            "eval_every": eval_every,
        },
        set_items=set_override or [],
        quiet=quiet,
    )
    tag = _run_tag(base_cfg)
    for seed in _parse_seeds(seeds or ["42"]):
        out_dir = output_root / "single" / tag / f"seed_{seed}"
        if skip_existing and (out_dir / "summary.json").exists():
            logger.info("Skipping completed {} (seed={})", tag, seed)
            continue
        cfg = replace(base_cfg, seed=seed, output_dir=str(out_dir))
        logger.info("Running {} (seed={}) -> {}", tag, seed, out_dir)
        res = run_experiment(cfg)
        s = res["summary"]
        logger.success(
            "DONE {} seed={}: final_acc={:.2f}% best_acc={:.2f}% fpr_cum={:.2f}% wall={:.1f}s",
            tag,
            seed,
            100.0 * float(s["final_benign_test_acc_mean"]),
            100.0 * float(s["best_benign_test_acc_mean"]),
            100.0 * float(s["fpr_cumulative"]),
            float(s["total_wall_sec"]),
        )


@app.command("suite")
def suite_cmd(
    suite: Annotated[str, typer.Option(help="Named suite from suites.py (run 'list-suites' to see all names).", rich_help_panel="Suite Selection")] = "smoke",
    seeds: Annotated[list[str] | None, typer.Option("--seed", "--seeds", help="Random seed(s); repeat or comma-separate (default: 42).", rich_help_panel="Suite Selection")] = None,
    shard_index: Annotated[int, typer.Option(min=0, help="Zero-based worker shard index for multi-GPU runs.", rich_help_panel="Sharding")] = 0,
    num_shards: Annotated[int, typer.Option(min=1, help="Total number of parallel worker shards.", rich_help_panel="Sharding")] = 1,
    skip_existing: Annotated[bool, typer.Option("--skip-existing", help="Skip runs that already have summary.json.", rich_help_panel="Sharding")] = False,
    dataset: Annotated[DatasetChoice, typer.Option(help="Dataset used by the suite.", rich_help_panel="Base Overrides")] = "cifar10",
    data_dir: Annotated[str, typer.Option(help="Directory storing or receiving CIFAR-10 batches.", rich_help_panel="Base Overrides")] = "data",
    no_download: Annotated[bool, typer.Option("--no-download", help="Disable automatic CIFAR-10 download.", rich_help_panel="Base Overrides")] = False,
    rounds: Annotated[int, typer.Option(min=1, help="Base communication rounds (50k ResNet-18 suites pin 25).", rich_help_panel="Base Overrides")] = 200,
    eval_every: Annotated[int, typer.Option(help="Evaluate test accuracy every N rounds (0 = final round only).", rich_help_panel="Base Overrides")] = 5,
    device: Annotated[DeviceChoice, typer.Option(help="Compute device ('auto', 'cpu', 'cuda', or 'mps').", rich_help_panel="Execution & Logging")] = "auto",
    output_root: Annotated[Path, typer.Option(help="Root directory for <suite>/<run>/seed_<id> outputs.", rich_help_panel="Execution & Logging")] = Path("outputs/phase1"),
    log_level: Annotated[LogLevelChoice, typer.Option(help="Minimum loguru log level.", rich_help_panel="Execution & Logging")] = "INFO",
    log_file: Annotated[Path | None, typer.Option(help="Optional extra log file capturing all CLI logs.", rich_help_panel="Execution & Logging")] = None,
    quiet: Annotated[bool, typer.Option("--quiet", help="Suppress INFO logs on stderr.", rich_help_panel="Execution & Logging")] = False,
) -> None:
    """Run one worker shard of a named benchmark suite across one or more random seeds."""
    configure_logging(level=log_level, log_file=log_file, quiet=quiet)
    if suite not in SUITES:
        raise typer.BadParameter(f"Unknown suite {suite!r}. Choose one of: {', '.join(sorted(SUITES))}")
    if not (0 <= shard_index < num_shards):
        raise typer.BadParameter(f"Invalid shard {shard_index}/{num_shards}: expected 0 <= shard_index < num_shards.")

    base_cfg = ExperimentConfig(
        dataset=dataset,
        data_dir=data_dir,
        download_data=not no_download,
        rounds=rounds,
        eval_every=eval_every,
        device=device,
        verbose=not quiet,
    )
    suite_root = output_root / suite
    suite_root.mkdir(parents=True, exist_ok=True)

    pending_jobs: list[tuple[int, str, ExperimentConfig]] = []
    skipped = 0
    for seed in _parse_seeds(seeds or ["42"]):
        seed_base = replace(base_cfg, seed=seed)
        for run_name, cfg in build_suite_runs(suite, seed_base):
            out_dir = suite_root / run_name / f"seed_{seed}"
            if skip_existing and (out_dir / "summary.json").exists():
                skipped += 1
                continue
            pending_jobs.append((seed, run_name, cfg))

    shard_jobs = [job for idx, job in enumerate(pending_jobs) if idx % num_shards == shard_index]
    logger.info(
        "Suite '{}': shard {}/{} running {}/{} pending jobs (skipped {} completed)",
        suite,
        shard_index,
        num_shards,
        len(shard_jobs),
        len(pending_jobs),
        skipped,
    )

    summaries: list[dict[str, object]] = []
    data_cache: dict[tuple[int, str, float, int, int, float], object] = {}
    for job_idx, (seed, run_name, cfg) in enumerate(shard_jobs, start=1):
        cache_key = (seed, cfg.val_split, cfg.alpha_dir, cfg.n_nodes, cfg.samples_per_node, cfg.val_frac)
        if cache_key not in data_cache:
            data_cache[cache_key] = prepare_dataset(cfg)
        out_dir = suite_root / run_name / f"seed_{seed}"
        run_cfg = replace(cfg, output_dir=str(out_dir))
        logger.info("[{} shard {} ({}/{})] {} (seed={})", suite, shard_index, job_idx, len(shard_jobs), run_name, seed)
        res = run_experiment(run_cfg, preloaded_data=data_cache[cache_key])
        s = dict(res["summary"])
        s["run_name"] = run_name
        summaries.append(s)

    summary_name = "suite_summaries.json" if num_shards == 1 else f"suite_summaries_shard{shard_index}.json"
    with open(suite_root / summary_name, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    logger.info("Suite '{}' shard {}/{} summary", suite, shard_index, num_shards)
    logger.info("{:44s} {:>5s} {:>9s} {:>9s} {:>9s} {:>8s} {:>8s} {:>8s}", "Run", "Seed", "FinalAcc", "ConsAcc", "FPR(cum)", "ByzAcc", "CosBen", "CosByz")
    for s in summaries:
        cb = f"{float(s['mean_benign_score']):+.3f}" if s["mean_benign_score"] is not None else "   n/a"
        cm = f"{float(s['mean_byz_score']):+.3f}" if s["mean_byz_score"] is not None else "   n/a"
        logger.info(
            "{:44s} {:5d} {:8.2f}% {:8.2f}% {:8.2f}% {:7.2f}% {:>8s} {:>8s}",
            str(s["run_name"]),
            int(s["config"]["seed"]),
            100.0 * float(s["final_benign_test_acc_mean"]),
            100.0 * float(s["final_consensus_test_acc"]),
            100.0 * float(s["fpr_cumulative"]),
            100.0 * float(s["byz_accept_cumulative"]),
            cb,
            cm,
        )


@app.command("distributed")
def distributed_cmd(
    preset: Annotated[PresetChoice, typer.Option(help="Base hyperparameter preset ('cnn_10k' or 'resnet18_50k').", rich_help_panel="Config & Overrides")] = "resnet18_50k",
    config: Annotated[Path | None, typer.Option(help="Optional JSON config file loaded over the preset.", rich_help_panel="Config & Overrides")] = None,
    set_override: Annotated[list[str] | None, typer.Option("--set", help="Override any ExperimentConfig field as key=value (repeatable, e.g. --set er_p=1.0).", rich_help_panel="Config & Overrides")] = None,
    dataset: Annotated[DatasetChoice | None, typer.Option(help="Dataset ('cifar10' or 'synthetic').", rich_help_panel="Core Options")] = None,
    model: Annotated[ModelChoice | None, typer.Option(help="Model architecture.", rich_help_panel="Core Options")] = None,
    n_nodes: Annotated[int | None, typer.Option(min=2, help="Logical peer count (overridden by torchrun/Slurm WORLD_SIZE).", rich_help_panel="Core Options")] = None,
    samples_per_node: Annotated[int | None, typer.Option(min=10, help="Train + validation samples per peer.", rich_help_panel="Core Options")] = None,
    rounds: Annotated[int | None, typer.Option(min=1, help="Number of communication rounds.", rich_help_panel="Core Options")] = None,
    topology: Annotated[TopologyChoice | None, typer.Option(help="Graph topology ('ring', 'random', 'small_world').", rich_help_panel="Core Options")] = None,
    attack: Annotated[AttackChoice | None, typer.Option(help="Byzantine attack mode.", rich_help_panel="Core Options")] = None,
    byzantine_frac: Annotated[float | None, typer.Option(min=0.0, max=1.0, help="Fraction of Byzantine peers.", rich_help_panel="Core Options")] = None,
    val_split: Annotated[ValSplitChoice | None, typer.Option(help="Validation split ('local' or 'iid').", rich_help_panel="Core Options")] = None,
    val_grad_sign: Annotated[ValGradSignChoice | None, typer.Option(help="Validation direction: 'descent' (-grad) or 'ascent' (+grad).", rich_help_panel="Core Options")] = None,
    delta_mode: Annotated[DeltaModeChoice | None, typer.Option("--delta-mode", "--trustgraph-delta-mode", help="Wire/scoring anchor: 'sender' (89.45 MB/nbr) or 'receiver' (44.73 MB/nbr).", rich_help_panel="Core Options")] = None,
    dist_iface: Annotated[str | None, typer.Option(help="Linux NIC name for inter-node TCP (e.g. 'enp68s0f0'; default loopback).", rich_help_panel="Distributed TCP")] = None,
    dist_port: Annotated[int, typer.Option(min=1024, max=65535, help="Base TCP port; rank r listens on dist_port + r.", rich_help_panel="Distributed TCP")] = 29600,
    no_download: Annotated[bool, typer.Option("--no-download", help="Disable automatic CIFAR-10 download.", rich_help_panel="Execution & Logging")] = False,
    seed: Annotated[int, typer.Option("--seed", "--seeds", help="Random seed for this distributed run.", rich_help_panel="Execution & Logging")] = 42,
    device: Annotated[DeviceChoice | None, typer.Option(help="Compute device ('auto', 'cpu', 'cuda', or 'mps').", rich_help_panel="Execution & Logging")] = None,
    eval_every: Annotated[int | None, typer.Option(help="Evaluate test accuracy every N rounds.", rich_help_panel="Execution & Logging")] = None,
    output_root: Annotated[Path, typer.Option(help="Root directory where rank 0 saves distributed/<tag>/seed_<id>.", rich_help_panel="Execution & Logging")] = Path("outputs/phase1"),
    log_level: Annotated[LogLevelChoice, typer.Option(help="Minimum loguru log level.", rich_help_panel="Execution & Logging")] = "INFO",
    log_file: Annotated[Path | None, typer.Option(help="Optional extra log file capturing all CLI logs.", rich_help_panel="Execution & Logging")] = None,
    quiet: Annotated[bool, typer.Option("--quiet", help="Suppress INFO logs on stderr.", rich_help_panel="Execution & Logging")] = False,
) -> None:
    """Launch this process's peer rank in a multi-process TCP DFL run (torchrun or Slurm srun)."""
    configure_logging(level=log_level, log_file=log_file, quiet=quiet)
    base_cfg = _resolve_config(
        preset=preset,
        config_file=config,
        direct_overrides={
            "dataset": dataset,
            "model_name": model,
            "n_nodes": n_nodes,
            "samples_per_node": samples_per_node,
            "rounds": rounds,
            "topology": topology,
            "attack": attack,
            "byzantine_frac": byzantine_frac,
            "val_split": val_split,
            "val_grad_sign": val_grad_sign,
            "trustgraph_delta_mode": delta_mode,
            "download_data": False if no_download else None,
            "device": device,
            "eval_every": eval_every,
        },
        set_items=set_override or [],
        quiet=quiet,
    )
    tag = _run_tag(base_cfg)
    out_dir = output_root / "distributed" / tag / f"seed_{seed}"
    cfg = replace(base_cfg, seed=seed, output_dir=str(out_dir))
    res = run_distributed_experiment(cfg, iface=dist_iface, base_port=dist_port)
    if res is not None:
        s = res["summary"]
        logger.success(
            "DONE distributed {} seed={}: final_acc={:.2f}% cons_acc={:.2f}% fpr_cum={:.2f}% wall={:.1f}s",
            tag,
            seed,
            100.0 * float(s["final_benign_test_acc_mean"]),
            100.0 * float(s["final_consensus_test_acc"]),
            100.0 * float(s["fpr_cumulative"]),
            float(s["total_wall_sec"]),
        )


@app.command("transfer-bench")
def transfer_bench_cmd(
    role: Annotated[BenchRoleChoice, typer.Option("--role", "--bench-role", help="Process role ('auto' uses SLURM_PROCID==0 as server).")] = "auto",
    host: Annotated[str, typer.Option("--host", "--bench-host", help="Server hostname or IPv4 address for the client to dial.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "--bench-port", min=1024, max=65535, help="TCP port bound by the server.")] = 29501,
    payload_bytes: Annotated[int, typer.Option("--payload-bytes", "--bench-payload-bytes", min=1024, help="Single FP32 model payload size in bytes (ResNet-18 = 44,726,568).")] = 11181642 * 4,
    repeats: Annotated[int, typer.Option("--repeats", "--bench-repeats", min=1, help="Measured repeats for single-update transfers.")] = 25,
    burst_repeats: Annotated[int, typer.Option("--burst-repeats", "--bench-burst-repeats", min=1, help="Measured repeats for multi-neighbor bursts.")] = 10,
    out: Annotated[Path | None, typer.Option("--out", "--bench-out", help="Optional JSON path where the client saves benchmark timings.")] = None,
    log_level: Annotated[LogLevelChoice, typer.Option(help="Minimum loguru log level.")] = "INFO",
    log_file: Annotated[Path | None, typer.Option(help="Optional extra log file path.")] = None,
) -> None:
    """Measure point-to-point TCP transfer goodput for ResNet-18 FP32 payload sizes."""
    configure_logging(level=log_level, log_file=log_file)
    run_transfer_benchmark(
        role=role,
        host=host,
        port=port,
        payload_bytes=payload_bytes,
        single_repeats=repeats,
        burst_repeats=burst_repeats,
        out_path=out,
    )


def main() -> None:
    """Run the Typer CLI for experiments.

    Returns:
        None.
    """
    app()


if __name__ == "__main__":
    main()
