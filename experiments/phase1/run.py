from dataclasses import replace
import json
from pathlib import Path
from typing import Annotated, Literal

import typer

from experiments.phase1.suites import SUITES, build_suite_runs
from src.distributed import run_distributed_experiment
from src.logging_utils import configure_logging, get_logger
from src.simulator import ExperimentConfig, prepare_dataset, run_experiment
from src.transport import run_transfer_benchmark

__all__ = ["SUITES", "app", "build_suite_runs", "main"]

logger = get_logger()

DatasetChoice = Literal["cifar10", "synthetic"]
ValSplitChoice = Literal["local", "iid"]
ModelChoice = Literal["cnn2x2", "cnn2x2_gn", "resnet18_pretrained", "resnet18_scratch"]
LrScheduleChoice = Literal["constant", "cosine"]
TopologyChoice = Literal["ring", "random", "small_world"]
AttackChoice = Literal["none", "label_flip", "label_flip_all", "model_poison", "random_noise"]
AggregatorChoice = Literal["trustgraph"]
ValGradSignChoice = Literal["descent_neg", "paper_pos"]
MetricChoice = Literal["cosine", "euclidean"]
ClipModeChoice = Literal["update", "model"]
DeltaModeChoice = Literal["sender", "receiver"]
DeviceChoice = Literal["auto", "cpu", "cuda", "mps"]
LogLevelChoice = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
BenchRoleChoice = Literal["auto", "server", "client"]

app = typer.Typer(
    name="peersieve-run",
    help="Run TrustGraph-DFL single experiments, named benchmark suites, multi-process TCP runs, or link benchmarks.",
    no_args_is_help=True,
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


def _build_experiment_config(
    *,
    dataset: DatasetChoice,
    data_dir: str,
    no_download: bool,
    n_nodes: int,
    samples_per_node: int,
    alpha_dir: float,
    val_frac: float,
    val_split: ValSplitChoice,
    model: ModelChoice,
    rounds: int,
    local_epochs: int,
    lr: float,
    lr_schedule: LrScheduleChoice,
    momentum: float,
    weight_decay: float,
    batch_size: int,
    augment: bool,
    grad_clip_norm: float,
    val_grad_weight_decay: bool,
    byz_local_epochs: int,
    topology: TopologyChoice,
    er_p: float,
    ws_k: int,
    ws_p: float,
    byzantine_frac: float,
    attack: AttackChoice,
    poison_lambda: float,
    noise_sigma: float,
    aggregator: AggregatorChoice,
    alpha: float,
    val_grad_sign: ValGradSignChoice,
    no_relu: bool,
    trustgraph_metric: MetricChoice,
    trustgraph_score_threshold: float,
    trustgraph_clip_norm: bool,
    trustgraph_clip_mode: ClipModeChoice,
    trustgraph_clip_factor: float,
    trustgraph_delta_mode: DeltaModeChoice,
    device: DeviceChoice,
    eval_every: int,
    quiet: bool,
) -> ExperimentConfig:
    """Construct an ExperimentConfig from typed CLI parameters.

    Args:
        dataset: Dataset name ('cifar10' or 'synthetic').
        data_dir: Directory storing or receiving CIFAR-10 files.
        no_download: Disable automatic dataset download when True.
        n_nodes: Number of decentralized peers.
        samples_per_node: Total train plus validation images per peer.
        alpha_dir: Dirichlet non-IID concentration parameter.
        val_frac: Fraction of each peer's samples held out for validation.
        val_split: Validation split mode ('local' or 'iid').
        model: Model architecture key.
        rounds: Total decentralized communication rounds.
        local_epochs: Local SGD epochs per round for benign peers.
        lr: Initial SGD learning rate.
        lr_schedule: Learning rate schedule ('constant' or 'cosine').
        momentum: SGD momentum factor.
        weight_decay: L2 weight decay applied during local training.
        batch_size: Mini-batch size for local SGD.
        augment: Enable random crop and horizontal flip augmentation.
        grad_clip_norm: Local gradient clipping norm threshold (0 disables).
        val_grad_weight_decay: Include L2 decay term in validation gradient.
        byz_local_epochs: Local epochs for label-flipping attackers (0 uses local_epochs).
        topology: Communication graph family.
        er_p: Erdos-Renyi edge probability for random graphs.
        ws_k: Nearest-neighbor degree for Watts-Strogatz graphs.
        ws_p: Rewiring probability for Watts-Strogatz graphs.
        byzantine_frac: Fraction of Byzantine peers in [0, 1].
        attack: Byzantine attack type.
        poison_lambda: Scaling multiplier for model_poison attacks.
        noise_sigma: Standard deviation for random_noise attacks.
        aggregator: Aggregation rule key.
        alpha: Self-weight parameter in TrustGraph-DFL.
        val_grad_sign: Validation gradient sign ('descent_neg' or 'paper_pos').
        no_relu: Disable ReLU zeroing of negative trust scores when True.
        trustgraph_metric: Neighbor similarity metric ('cosine' or 'euclidean').
        trustgraph_score_threshold: Minimum score threshold tau before ReLU.
        trustgraph_clip_norm: Enable norm clipping of neighbor updates/displacements.
        trustgraph_clip_mode: Clipping reference ('update' or 'model').
        trustgraph_clip_factor: Multiplier relative to receiver's own update norm.
        trustgraph_delta_mode: Scoring reference ('sender' or 'receiver').
        device: Target compute device.
        eval_every: Evaluate test accuracy every N rounds (0 evaluates final round only).
        quiet: Suppress INFO logs on stderr when True.

    Returns:
        Populated ExperimentConfig instance.
    """
    return ExperimentConfig(
        dataset=dataset,
        data_dir=data_dir,
        download_data=not no_download,
        n_nodes=n_nodes,
        samples_per_node=samples_per_node,
        alpha_dir=alpha_dir,
        val_frac=val_frac,
        val_split=val_split,
        model_name=model,
        rounds=rounds,
        local_epochs=local_epochs,
        lr=lr,
        lr_schedule=lr_schedule,
        momentum=momentum,
        weight_decay=weight_decay,
        batch_size=batch_size,
        augment=augment,
        grad_clip_norm=grad_clip_norm,
        val_grad_weight_decay=val_grad_weight_decay,
        byz_local_epochs=byz_local_epochs,
        topology=topology,
        er_p=er_p,
        ws_k=ws_k,
        ws_p=ws_p,
        byzantine_frac=byzantine_frac,
        attack=attack,
        poison_lambda=poison_lambda,
        noise_sigma=noise_sigma,
        aggregator=aggregator,
        alpha=alpha,
        val_grad_sign=val_grad_sign,
        trustgraph_use_relu=not no_relu,
        trustgraph_metric=trustgraph_metric,
        trustgraph_score_threshold=trustgraph_score_threshold,
        trustgraph_clip_norm=trustgraph_clip_norm,
        trustgraph_clip_mode=trustgraph_clip_mode,
        trustgraph_clip_factor=trustgraph_clip_factor,
        trustgraph_delta_mode=trustgraph_delta_mode,
        device=device,
        eval_every=eval_every,
        verbose=not quiet,
    )


@app.command("list-suites")
def list_suites() -> None:
    """List all registered named experiment suites and their run counts per seed.

    Returns:
        None.
    """
    configure_logging()
    base = ExperimentConfig()
    logger.info("Registered experiment suites ({} total):", len(SUITES))
    for name in sorted(SUITES):
        n_runs = len(build_suite_runs(name, base))
        logger.info("  {:24s} ({:2d} runs / seed)", name, n_runs)


@app.command("single")
def single_cmd(
    dataset: Annotated[DatasetChoice, typer.Option(help="Dataset to train on ('cifar10' or 'synthetic').", rich_help_panel="Model & Data")] = "cifar10",
    data_dir: Annotated[str, typer.Option(help="Directory storing or receiving CIFAR-10 batches.", rich_help_panel="Model & Data")] = "data",
    no_download: Annotated[bool, typer.Option("--no-download", help="Disable automatic CIFAR-10 download (required on offline compute nodes).", rich_help_panel="Model & Data")] = False,
    model: Annotated[ModelChoice, typer.Option(help="Model architecture to train.", rich_help_panel="Model & Data")] = "cnn2x2",
    n_nodes: Annotated[int, typer.Option(min=2, help="Number of decentralized peers in the graph.", rich_help_panel="Model & Data")] = 20,
    samples_per_node: Annotated[int, typer.Option(min=10, help="Total train + validation samples assigned to each peer.", rich_help_panel="Model & Data")] = 500,
    alpha_dir: Annotated[float, typer.Option(help="Dirichlet non-IID concentration parameter (lower = more skew).", rich_help_panel="Model & Data")] = 0.5,
    val_frac: Annotated[float, typer.Option(help="Fraction of each peer's samples reserved for validation.", rich_help_panel="Model & Data")] = 0.1,
    val_split: Annotated[ValSplitChoice, typer.Option(help="Validation split: 'local' (non-IID shard) or 'iid' (balanced).", rich_help_panel="Model & Data")] = "local",
    rounds: Annotated[int, typer.Option(min=1, help="Number of decentralized communication rounds.", rich_help_panel="Training")] = 200,
    local_epochs: Annotated[int, typer.Option(min=1, help="Local SGD epochs per round on benign peers.", rich_help_panel="Training")] = 1,
    lr: Annotated[float, typer.Option(help="Initial local SGD learning rate.", rich_help_panel="Training")] = 0.01,
    lr_schedule: Annotated[LrScheduleChoice, typer.Option(help="Learning rate schedule across rounds.", rich_help_panel="Training")] = "constant",
    momentum: Annotated[float, typer.Option(help="SGD momentum factor.", rich_help_panel="Training")] = 0.9,
    weight_decay: Annotated[float, typer.Option(help="L2 weight decay during local SGD.", rich_help_panel="Training")] = 1e-4,
    batch_size: Annotated[int, typer.Option(min=1, help="Mini-batch size for local SGD.", rich_help_panel="Training")] = 64,
    augment: Annotated[bool, typer.Option("--augment", help="Enable CIFAR-10 random crop and horizontal flip augmentation.", rich_help_panel="Training")] = False,
    grad_clip_norm: Annotated[float, typer.Option(help="Clip local gradient L2 norm to this cap (0 disables).", rich_help_panel="Training")] = 0.0,
    val_grad_weight_decay: Annotated[bool, typer.Option("--val-grad-weight-decay", help="Add L2 weight decay term to validation gradients.", rich_help_panel="Training")] = False,
    byz_local_epochs: Annotated[int, typer.Option(min=0, help="Local epochs for label-flipping attackers (0 uses --local-epochs).", rich_help_panel="Topology & Attack")] = 0,
    topology: Annotated[TopologyChoice, typer.Option(help="Peer communication graph topology.", rich_help_panel="Topology & Attack")] = "random",
    er_p: Annotated[float, typer.Option(help="Edge probability p for Erdos-Renyi random graphs.", rich_help_panel="Topology & Attack")] = 0.3,
    ws_k: Annotated[int, typer.Option(help="Nearest-neighbor degree k for Watts-Strogatz small-world graphs.", rich_help_panel="Topology & Attack")] = 4,
    ws_p: Annotated[float, typer.Option(help="Rewiring probability p for Watts-Strogatz small-world graphs.", rich_help_panel="Topology & Attack")] = 0.3,
    byzantine_frac: Annotated[float, typer.Option(min=0.0, max=1.0, help="Fraction of peers marked Byzantine.", rich_help_panel="Topology & Attack")] = 0.3,
    attack: Annotated[AttackChoice, typer.Option(help="Byzantine attack applied on malicious peers.", rich_help_panel="Topology & Attack")] = "label_flip",
    poison_lambda: Annotated[float, typer.Option(help="Scaling multiplier lambda for model_poison.", rich_help_panel="Topology & Attack")] = 10.0,
    noise_sigma: Annotated[float, typer.Option(help="Gaussian std sigma for random_noise.", rich_help_panel="Topology & Attack")] = 0.5,
    aggregator: Annotated[AggregatorChoice, typer.Option(help="Decentralized aggregation rule.", rich_help_panel="TrustGraph Scoring")] = "trustgraph",
    alpha: Annotated[float, typer.Option(min=0.0, max=1.0, help="Self-weight alpha retained on local model during aggregation.", rich_help_panel="TrustGraph Scoring")] = 0.5,
    val_grad_sign: Annotated[ValGradSignChoice, typer.Option(help="Validation reference sign: 'descent_neg' (-grad) or 'paper_pos' (+grad).", rich_help_panel="TrustGraph Scoring")] = "descent_neg",
    no_relu: Annotated[bool, typer.Option("--no-relu", help="Keep negative cosine scores instead of zeroing them with ReLU.", rich_help_panel="TrustGraph Scoring")] = False,
    trustgraph_metric: Annotated[MetricChoice, typer.Option(help="Similarity metric used to score neighbor updates.", rich_help_panel="TrustGraph Scoring")] = "cosine",
    trustgraph_score_threshold: Annotated[float, typer.Option(help="Minimum cosine score tau required for positive weight.", rich_help_panel="TrustGraph Scoring")] = 0.0,
    trustgraph_clip_norm: Annotated[bool, typer.Option("--trustgraph-clip-norm", help="Cap accepted neighbor step size relative to local update norm.", rich_help_panel="TrustGraph Scoring")] = False,
    trustgraph_clip_mode: Annotated[ClipModeChoice, typer.Option(help="Clip 'update' delta or 'model' displacement relative to receiver.", rich_help_panel="TrustGraph Scoring")] = "update",
    trustgraph_clip_factor: Annotated[float, typer.Option(help="Norm clipping multiplier relative to receiver update norm.", rich_help_panel="TrustGraph Scoring")] = 2.0,
    trustgraph_delta_mode: Annotated[DeltaModeChoice, typer.Option(help="Neighbor delta anchor: 'sender' (2 vectors) or 'receiver' (rxdelta, 1 vector).", rich_help_panel="TrustGraph Scoring")] = "sender",
    seeds: Annotated[list[str], typer.Option("--seed", "--seeds", help="Random seed(s); repeat or comma-separate (e.g. --seeds 42,43,44).", rich_help_panel="Execution & Logging")] = ["42"],
    device: Annotated[DeviceChoice, typer.Option(help="Compute device ('auto', 'cpu', 'cuda', or 'mps').", rich_help_panel="Execution & Logging")] = "auto",
    eval_every: Annotated[int, typer.Option(help="Evaluate test accuracy every N rounds (0 evaluates final round only).", rich_help_panel="Execution & Logging")] = 5,
    skip_existing: Annotated[bool, typer.Option("--skip-existing", help="Skip seeds whose summary.json already exists.", rich_help_panel="Execution & Logging")] = False,
    output_root: Annotated[Path, typer.Option(help="Root directory where single/<tag>/seed_<id> folders are written.", rich_help_panel="Execution & Logging")] = Path("outputs/phase1"),
    log_level: Annotated[LogLevelChoice, typer.Option(help="Minimum loguru log level.", rich_help_panel="Execution & Logging")] = "INFO",
    log_file: Annotated[Path | None, typer.Option(help="Optional extra log file capturing all CLI logs.", rich_help_panel="Execution & Logging")] = None,
    quiet: Annotated[bool, typer.Option("--quiet", help="Suppress INFO logs on stderr (per-run run.log still records INFO).", rich_help_panel="Execution & Logging")] = False,
) -> None:
    """Run one custom TrustGraph-DFL experiment in a single process across one or more seeds.

    Args:
        dataset: Dataset to train on.
        data_dir: Local dataset directory.
        no_download: Disable dataset downloads.
        model: Neural network architecture key.
        n_nodes: Number of peers.
        samples_per_node: Samples allocated per peer.
        alpha_dir: Dirichlet non-IID skew parameter.
        val_frac: Validation split ratio.
        val_split: Validation split mode ('local' or 'iid').
        rounds: Communication rounds.
        local_epochs: Benign local SGD epochs per round.
        lr: Initial SGD learning rate.
        lr_schedule: Learning rate schedule.
        momentum: SGD momentum.
        weight_decay: L2 weight decay.
        batch_size: Mini-batch size.
        augment: Enable CIFAR-10 augmentation.
        grad_clip_norm: Local gradient norm cap.
        val_grad_weight_decay: Include L2 decay in validation gradient.
        byz_local_epochs: Label-flip attacker local epochs.
        topology: Communication graph topology.
        er_p: Erdos-Renyi edge probability.
        ws_k: Small-world nearest-neighbor degree.
        ws_p: Small-world rewiring probability.
        byzantine_frac: Fraction of Byzantine peers.
        attack: Attack type.
        poison_lambda: Model poisoning scaling multiplier.
        noise_sigma: Random noise standard deviation.
        aggregator: Aggregation method.
        alpha: TrustGraph self-weight.
        val_grad_sign: Validation gradient sign.
        no_relu: Disable ReLU zeroing.
        trustgraph_metric: Similarity metric.
        trustgraph_score_threshold: Score threshold tau.
        trustgraph_clip_norm: Enable norm clipping.
        trustgraph_clip_mode: Clipping mode.
        trustgraph_clip_factor: Clipping multiplier.
        trustgraph_delta_mode: Delta reference anchor.
        seeds: Seed strings (repeated or comma-separated).
        device: Compute device.
        eval_every: Evaluation stride in rounds.
        skip_existing: Skip runs with existing summary.json.
        output_root: Root output directory.
        log_level: Minimum loguru log level.
        log_file: Optional extra CLI log file path.
        quiet: Suppress stderr INFO logs.

    Returns:
        None.
    """
    configure_logging(level=log_level, log_file=log_file, quiet=quiet)
    parsed_seeds = _parse_seeds(seeds)
    base_cfg = _build_experiment_config(
        dataset=dataset,
        data_dir=data_dir,
        no_download=no_download,
        n_nodes=n_nodes,
        samples_per_node=samples_per_node,
        alpha_dir=alpha_dir,
        val_frac=val_frac,
        val_split=val_split,
        model=model,
        rounds=rounds,
        local_epochs=local_epochs,
        lr=lr,
        lr_schedule=lr_schedule,
        momentum=momentum,
        weight_decay=weight_decay,
        batch_size=batch_size,
        augment=augment,
        grad_clip_norm=grad_clip_norm,
        val_grad_weight_decay=val_grad_weight_decay,
        byz_local_epochs=byz_local_epochs,
        topology=topology,
        er_p=er_p,
        ws_k=ws_k,
        ws_p=ws_p,
        byzantine_frac=byzantine_frac,
        attack=attack,
        poison_lambda=poison_lambda,
        noise_sigma=noise_sigma,
        aggregator=aggregator,
        alpha=alpha,
        val_grad_sign=val_grad_sign,
        no_relu=no_relu,
        trustgraph_metric=trustgraph_metric,
        trustgraph_score_threshold=trustgraph_score_threshold,
        trustgraph_clip_norm=trustgraph_clip_norm,
        trustgraph_clip_mode=trustgraph_clip_mode,
        trustgraph_clip_factor=trustgraph_clip_factor,
        trustgraph_delta_mode=trustgraph_delta_mode,
        device=device,
        eval_every=eval_every,
        quiet=quiet,
    )
    tag = _run_tag(base_cfg)
    for seed in parsed_seeds:
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
    seeds: Annotated[list[str], typer.Option("--seed", "--seeds", help="Random seed(s); repeat or comma-separate (e.g. --seeds 42,43,44).", rich_help_panel="Suite Selection")] = ["42"],
    shard_index: Annotated[int, typer.Option(min=0, help="Zero-based worker shard index for parallel multi-GPU execution.", rich_help_panel="Sharding")] = 0,
    num_shards: Annotated[int, typer.Option(min=1, help="Total number of parallel worker shards.", rich_help_panel="Sharding")] = 1,
    skip_existing: Annotated[bool, typer.Option("--skip-existing", help="Skip runs that already have a summary.json.", rich_help_panel="Sharding")] = False,
    dataset: Annotated[DatasetChoice, typer.Option(help="Dataset used by the suite.", rich_help_panel="Base Overrides")] = "cifar10",
    data_dir: Annotated[str, typer.Option(help="Directory storing or receiving CIFAR-10 batches.", rich_help_panel="Base Overrides")] = "data",
    no_download: Annotated[bool, typer.Option("--no-download", help="Disable automatic CIFAR-10 downloads.", rich_help_panel="Base Overrides")] = False,
    rounds: Annotated[int, typer.Option(min=1, help="Base communication rounds (50k ResNet-18 suites override to 25).", rich_help_panel="Base Overrides")] = 200,
    eval_every: Annotated[int, typer.Option(help="Evaluate test accuracy every N rounds (0 evaluates final round only).", rich_help_panel="Base Overrides")] = 5,
    device: Annotated[DeviceChoice, typer.Option(help="Compute device ('auto', 'cpu', 'cuda', or 'mps').", rich_help_panel="Execution & Logging")] = "auto",
    output_root: Annotated[Path, typer.Option(help="Root directory where <suite>/<run_name>/seed_<id> outputs are saved.", rich_help_panel="Execution & Logging")] = Path("outputs/phase1"),
    log_level: Annotated[LogLevelChoice, typer.Option(help="Minimum loguru log level.", rich_help_panel="Execution & Logging")] = "INFO",
    log_file: Annotated[Path | None, typer.Option(help="Optional extra log file capturing all CLI logs.", rich_help_panel="Execution & Logging")] = None,
    quiet: Annotated[bool, typer.Option("--quiet", help="Suppress INFO logs on stderr.", rich_help_panel="Execution & Logging")] = False,
) -> None:
    """Run one worker shard of a named benchmark suite across one or more random seeds.

    Args:
        suite: Suite name registered in SUITES.
        seeds: Seed strings (repeated or comma-separated).
        shard_index: Zero-based worker shard index.
        num_shards: Total parallel shards.
        skip_existing: Skip runs with existing summary.json.
        dataset: Dataset name.
        data_dir: Local dataset directory.
        no_download: Disable automatic dataset download.
        rounds: Base rounds for suites that do not pin rounds.
        eval_every: Test evaluation stride in rounds.
        device: Compute device.
        output_root: Root output directory.
        log_level: Minimum loguru log level.
        log_file: Optional extra CLI log file path.
        quiet: Suppress stderr INFO logs.

    Returns:
        None.
    """
    configure_logging(level=log_level, log_file=log_file, quiet=quiet)
    if suite not in SUITES:
        raise typer.BadParameter(f"Unknown suite {suite!r}. Choose one of: {', '.join(sorted(SUITES))}")
    if not (0 <= shard_index < num_shards):
        raise typer.BadParameter(f"Invalid shard {shard_index}/{num_shards}: expected 0 <= shard_index < num_shards.")

    parsed_seeds = _parse_seeds(seeds)
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
    for seed in parsed_seeds:
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
        logger.info(
            "[{} shard {} ({}/{})] {} (seed={})",
            suite,
            shard_index,
            job_idx,
            len(shard_jobs),
            run_name,
            seed,
        )
        res = run_experiment(run_cfg, preloaded_data=data_cache[cache_key])
        s = dict(res["summary"])
        s["run_name"] = run_name
        summaries.append(s)

    summary_name = "suite_summaries.json" if num_shards == 1 else f"suite_summaries_shard{shard_index}.json"
    with open(suite_root / summary_name, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    logger.info("Suite '{}' shard {}/{} summary", suite, shard_index, num_shards)
    logger.info(
        "{:44s} {:>5s} {:>9s} {:>9s} {:>9s} {:>8s} {:>8s} {:>8s}",
        "Run",
        "Seed",
        "FinalAcc",
        "ConsAcc",
        "FPR(cum)",
        "ByzAcc",
        "CosBen",
        "CosByz",
    )
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
    dataset: Annotated[DatasetChoice, typer.Option(help="Dataset to train on.", rich_help_panel="Model & Data")] = "cifar10",
    data_dir: Annotated[str, typer.Option(help="Directory storing or receiving CIFAR-10 batches.", rich_help_panel="Model & Data")] = "data",
    no_download: Annotated[bool, typer.Option("--no-download", help="Disable automatic CIFAR-10 downloads.", rich_help_panel="Model & Data")] = False,
    model: Annotated[ModelChoice, typer.Option(help="Model architecture to train.", rich_help_panel="Model & Data")] = "resnet18_pretrained",
    n_nodes: Annotated[int, typer.Option(min=2, help="Logical peer count (overridden by torchrun/Slurm WORLD_SIZE at runtime).", rich_help_panel="Model & Data")] = 4,
    samples_per_node: Annotated[int, typer.Option(min=10, help="Total train + validation samples assigned to each peer.", rich_help_panel="Model & Data")] = 2500,
    alpha_dir: Annotated[float, typer.Option(help="Dirichlet non-IID concentration parameter.", rich_help_panel="Model & Data")] = 0.5,
    val_frac: Annotated[float, typer.Option(help="Fraction of each peer's samples reserved for validation.", rich_help_panel="Model & Data")] = 0.1,
    val_split: Annotated[ValSplitChoice, typer.Option(help="Validation split: 'local' or 'iid'.", rich_help_panel="Model & Data")] = "iid",
    rounds: Annotated[int, typer.Option(min=1, help="Number of decentralized communication rounds.", rich_help_panel="Training")] = 25,
    local_epochs: Annotated[int, typer.Option(min=1, help="Local SGD epochs per round on benign peers.", rich_help_panel="Training")] = 1,
    lr: Annotated[float, typer.Option(help="Initial local SGD learning rate.", rich_help_panel="Training")] = 0.005,
    lr_schedule: Annotated[LrScheduleChoice, typer.Option(help="Learning rate schedule across rounds.", rich_help_panel="Training")] = "cosine",
    momentum: Annotated[float, typer.Option(help="SGD momentum factor.", rich_help_panel="Training")] = 0.9,
    weight_decay: Annotated[float, typer.Option(help="L2 weight decay during local SGD.", rich_help_panel="Training")] = 1e-4,
    batch_size: Annotated[int, typer.Option(min=1, help="Mini-batch size for local SGD.", rich_help_panel="Training")] = 64,
    augment: Annotated[bool, typer.Option("--augment", help="Enable CIFAR-10 random crop and horizontal flip augmentation.", rich_help_panel="Training")] = False,
    grad_clip_norm: Annotated[float, typer.Option(help="Clip local gradient L2 norm to this cap (0 disables).", rich_help_panel="Training")] = 0.0,
    val_grad_weight_decay: Annotated[bool, typer.Option("--val-grad-weight-decay", help="Add L2 weight decay term to validation gradients.", rich_help_panel="Training")] = False,
    byz_local_epochs: Annotated[int, typer.Option(min=0, help="Local epochs for label-flipping attackers (0 uses --local-epochs).", rich_help_panel="Topology & Attack")] = 0,
    topology: Annotated[TopologyChoice, typer.Option(help="Peer communication graph topology.", rich_help_panel="Topology & Attack")] = "random",
    er_p: Annotated[float, typer.Option(help="Edge probability p for Erdos-Renyi random graphs (1.0 = complete graph).", rich_help_panel="Topology & Attack")] = 0.3,
    ws_k: Annotated[int, typer.Option(help="Nearest-neighbor degree k for Watts-Strogatz graphs.", rich_help_panel="Topology & Attack")] = 4,
    ws_p: Annotated[float, typer.Option(help="Rewiring probability p for Watts-Strogatz graphs.", rich_help_panel="Topology & Attack")] = 0.3,
    byzantine_frac: Annotated[float, typer.Option(min=0.0, max=1.0, help="Fraction of peers marked Byzantine.", rich_help_panel="Topology & Attack")] = 0.25,
    attack: Annotated[AttackChoice, typer.Option(help="Byzantine attack applied on malicious peers.", rich_help_panel="Topology & Attack")] = "model_poison",
    poison_lambda: Annotated[float, typer.Option(help="Scaling multiplier lambda for model_poison.", rich_help_panel="Topology & Attack")] = 10.0,
    noise_sigma: Annotated[float, typer.Option(help="Gaussian std sigma for random_noise.", rich_help_panel="Topology & Attack")] = 0.5,
    aggregator: Annotated[AggregatorChoice, typer.Option(help="Decentralized aggregation rule.", rich_help_panel="TrustGraph Scoring")] = "trustgraph",
    alpha: Annotated[float, typer.Option(min=0.0, max=1.0, help="Self-weight alpha retained on local model.", rich_help_panel="TrustGraph Scoring")] = 0.5,
    val_grad_sign: Annotated[ValGradSignChoice, typer.Option(help="Validation reference sign: 'descent_neg' or 'paper_pos'.", rich_help_panel="TrustGraph Scoring")] = "descent_neg",
    no_relu: Annotated[bool, typer.Option("--no-relu", help="Keep negative cosine scores instead of zeroing with ReLU.", rich_help_panel="TrustGraph Scoring")] = False,
    trustgraph_metric: Annotated[MetricChoice, typer.Option(help="Similarity metric used to score neighbor updates.", rich_help_panel="TrustGraph Scoring")] = "cosine",
    trustgraph_score_threshold: Annotated[float, typer.Option(help="Minimum cosine score tau required for positive weight.", rich_help_panel="TrustGraph Scoring")] = 0.003,
    trustgraph_clip_norm: Annotated[bool, typer.Option("--trustgraph-clip-norm", help="Cap accepted neighbor step size relative to local update norm.", rich_help_panel="TrustGraph Scoring")] = False,
    trustgraph_clip_mode: Annotated[ClipModeChoice, typer.Option(help="Clip 'update' delta or 'model' displacement relative to receiver.", rich_help_panel="TrustGraph Scoring")] = "model",
    trustgraph_clip_factor: Annotated[float, typer.Option(help="Norm clipping multiplier relative to receiver update norm.", rich_help_panel="TrustGraph Scoring")] = 2.0,
    trustgraph_delta_mode: Annotated[DeltaModeChoice, typer.Option(help="Wire/scoring anchor: 'sender' (89.45 MB/nbr) or 'receiver' (44.73 MB/nbr).", rich_help_panel="TrustGraph Scoring")] = "sender",
    dist_iface: Annotated[str | None, typer.Option(help="Linux network interface for inter-node TCP (e.g. 'enp68s0f0'; default loopback).", rich_help_panel="Distributed TCP")] = None,
    dist_port: Annotated[int, typer.Option(min=1024, max=65535, help="Base TCP listening port; rank r binds dist_port + r.", rich_help_panel="Distributed TCP")] = 29600,
    seed: Annotated[int, typer.Option("--seed", "--seeds", help="Random seed for this distributed run.", rich_help_panel="Execution & Logging")] = 42,
    device: Annotated[DeviceChoice, typer.Option(help="Compute device ('auto', 'cpu', 'cuda', or 'mps').", rich_help_panel="Execution & Logging")] = "auto",
    eval_every: Annotated[int, typer.Option(help="Evaluate test accuracy every N rounds.", rich_help_panel="Execution & Logging")] = 5,
    output_root: Annotated[Path, typer.Option(help="Root directory where rank 0 saves distributed/<tag>/seed_<id>.", rich_help_panel="Execution & Logging")] = Path("outputs/phase1"),
    log_level: Annotated[LogLevelChoice, typer.Option(help="Minimum loguru log level.", rich_help_panel="Execution & Logging")] = "INFO",
    log_file: Annotated[Path | None, typer.Option(help="Optional extra log file capturing all CLI logs.", rich_help_panel="Execution & Logging")] = None,
    quiet: Annotated[bool, typer.Option("--quiet", help="Suppress INFO logs on stderr.", rich_help_panel="Execution & Logging")] = False,
) -> None:
    """Launch this process's peer rank in a real multi-process TCP DFL run (torchrun or Slurm srun).

    Args:
        dataset: Dataset to train on.
        data_dir: Local dataset directory.
        no_download: Disable automatic dataset download.
        model: Neural network architecture key.
        n_nodes: Logical peer count (overridden by WORLD_SIZE).
        samples_per_node: Samples allocated per peer.
        alpha_dir: Dirichlet non-IID skew parameter.
        val_frac: Validation split ratio.
        val_split: Validation split mode ('local' or 'iid').
        rounds: Communication rounds.
        local_epochs: Benign local SGD epochs per round.
        lr: Initial SGD learning rate.
        lr_schedule: Learning rate schedule.
        momentum: SGD momentum.
        weight_decay: L2 weight decay.
        batch_size: Mini-batch size.
        augment: Enable CIFAR-10 augmentation.
        grad_clip_norm: Local gradient norm cap.
        val_grad_weight_decay: Include L2 decay in validation gradient.
        byz_local_epochs: Label-flip attacker local epochs.
        topology: Communication graph topology.
        er_p: Erdos-Renyi edge probability.
        ws_k: Small-world nearest-neighbor degree.
        ws_p: Small-world rewiring probability.
        byzantine_frac: Fraction of Byzantine peers.
        attack: Attack type.
        poison_lambda: Model poisoning scaling multiplier.
        noise_sigma: Random noise standard deviation.
        aggregator: Aggregation method.
        alpha: TrustGraph self-weight.
        val_grad_sign: Validation gradient sign.
        no_relu: Disable ReLU zeroing.
        trustgraph_metric: Similarity metric.
        trustgraph_score_threshold: Score threshold tau.
        trustgraph_clip_norm: Enable norm clipping.
        trustgraph_clip_mode: Clipping mode.
        trustgraph_clip_factor: Clipping multiplier.
        trustgraph_delta_mode: Delta reference anchor.
        dist_iface: Network interface name for TCP sockets.
        dist_port: Base TCP port.
        seed: Random seed.
        device: Compute device.
        eval_every: Evaluation stride in rounds.
        output_root: Root output directory.
        log_level: Minimum loguru log level.
        log_file: Optional extra CLI log file path.
        quiet: Suppress stderr INFO logs.

    Returns:
        None.
    """
    configure_logging(level=log_level, log_file=log_file, quiet=quiet)
    base_cfg = _build_experiment_config(
        dataset=dataset,
        data_dir=data_dir,
        no_download=no_download,
        n_nodes=n_nodes,
        samples_per_node=samples_per_node,
        alpha_dir=alpha_dir,
        val_frac=val_frac,
        val_split=val_split,
        model=model,
        rounds=rounds,
        local_epochs=local_epochs,
        lr=lr,
        lr_schedule=lr_schedule,
        momentum=momentum,
        weight_decay=weight_decay,
        batch_size=batch_size,
        augment=augment,
        grad_clip_norm=grad_clip_norm,
        val_grad_weight_decay=val_grad_weight_decay,
        byz_local_epochs=byz_local_epochs,
        topology=topology,
        er_p=er_p,
        ws_k=ws_k,
        ws_p=ws_p,
        byzantine_frac=byzantine_frac,
        attack=attack,
        poison_lambda=poison_lambda,
        noise_sigma=noise_sigma,
        aggregator=aggregator,
        alpha=alpha,
        val_grad_sign=val_grad_sign,
        no_relu=no_relu,
        trustgraph_metric=trustgraph_metric,
        trustgraph_score_threshold=trustgraph_score_threshold,
        trustgraph_clip_norm=trustgraph_clip_norm,
        trustgraph_clip_mode=trustgraph_clip_mode,
        trustgraph_clip_factor=trustgraph_clip_factor,
        trustgraph_delta_mode=trustgraph_delta_mode,
        device=device,
        eval_every=eval_every,
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
    payload_bytes: Annotated[int, typer.Option("--payload-bytes", "--bench-payload-bytes", min=1024, help="Single FP32 model payload size in bytes (default: ResNet-18 = 44,726,568).")] = 11181642 * 4,
    repeats: Annotated[int, typer.Option("--repeats", "--bench-repeats", min=1, help="Measured repeats for single-update transfers.")] = 25,
    burst_repeats: Annotated[int, typer.Option("--burst-repeats", "--bench-burst-repeats", min=1, help="Measured repeats for multi-neighbor bursts.")] = 10,
    out: Annotated[Path | None, typer.Option("--out", "--bench-out", help="Optional JSON path where the client saves benchmark timings.")] = None,
    log_level: Annotated[LogLevelChoice, typer.Option(help="Minimum loguru log level.")] = "INFO",
    log_file: Annotated[Path | None, typer.Option(help="Optional extra log file path.")] = None,
) -> None:
    """Measure point-to-point TCP transfer goodput for ResNet-18 FP32 payload sizes.

    Args:
        role: Process role ('auto', 'server', or 'client').
        host: Server host or IPv4 address.
        port: Server TCP listening port.
        payload_bytes: Single-model payload size in bytes.
        repeats: Repeat count for single-update transfers.
        burst_repeats: Repeat count for burst transfers.
        out: Optional JSON output path written on the client.
        log_level: Minimum loguru log level.
        log_file: Optional extra CLI log file path.

    Returns:
        None.
    """
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
    """Invoke the Typer CLI application for Phase 1 experiments.

    Returns:
        None.
    """
    app()


if __name__ == "__main__":
    main()
