from __future__ import annotations

from pathlib import Path

from src.logging_utils import _format_context, configure_logging, get_logger, run_log_sink
from src.simulator import ExperimentConfig, run_experiment


def test_format_context_and_noop_sink() -> None:
    assert _format_context() == ""
    assert _format_context(rank=3) == "r=3 | "
    with_host = _format_context(rank=1, include_host=True)
    assert with_host.endswith(":r=1 | ")

    with run_log_sink(None):
        get_logger().debug("no-op sink check")


def test_logging_utils_writes_run_log_and_respects_level(tmp_path: Path) -> None:
    extra_log = tmp_path / "cli.log"
    configure_logging(level="INFO", log_file=extra_log, quiet=True)
    log = get_logger(rank=2)
    log.debug("hidden debug message")
    log.info("visible info message")

    out_dir = tmp_path / "logged_run"
    run_experiment(
        ExperimentConfig(
            dataset="synthetic",
            n_nodes=4,
            samples_per_node=20,
            rounds=2,
            model_name="cnn2x2",
            topology="ring",
            byzantine_frac=0.0,
            attack="none",
            device="cpu",
            eval_every=2,
            seed=3,
            output_dir=str(out_dir),
            verbose=True,
        )
    )
    cli_text = extra_log.read_text(encoding="utf-8")
    assert "hidden debug message" not in cli_text
    assert "r=2 | visible info message" in cli_text
    run_log = out_dir / "run.log"
    assert run_log.exists()
    run_text = run_log.read_text(encoding="utf-8")
    assert "Start local run:" in run_text
    assert "Finished local run" in run_text
    configure_logging(level="INFO")
