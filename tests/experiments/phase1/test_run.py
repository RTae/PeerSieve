from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from experiments.phase1 import plot_figures, run as run_cli, summarize


def test_parse_seeds_coerce_value_and_config_layering(tmp_path: Path) -> None:
    assert run_cli._parse_seeds(["42,43", " 44 "]) == [42, 43, 44]
    with pytest.raises(typer.BadParameter):
        run_cli._parse_seeds(["abc"])
    with pytest.raises(typer.BadParameter):
        run_cli._parse_seeds([""])

    assert run_cli._coerce_value("augment", "true", bool) is True
    assert run_cli._coerce_value("augment", "0", bool) is False
    assert run_cli._coerce_value("rounds", "15", int) == 15
    assert run_cli._coerce_value("lr", "0.05", float) == 0.05
    with pytest.raises(typer.BadParameter):
        run_cli._coerce_value("augment", "maybe", bool)
    with pytest.raises(typer.BadParameter):
        run_cli._coerce_value("rounds", "ten", int)

    cfg_file = tmp_path / "base.json"
    cfg_file.write_text(json.dumps({"rounds": 12, "er_p": 0.6, "lr": 0.02}), encoding="utf-8")
    resolved = run_cli._resolve_config(
        preset="resnet18_50k",
        config_file=cfg_file,
        direct_overrides={"rounds": 8, "attack": None},
        set_items=["er_p=0.9", "augment=false"],
        quiet=True,
    )
    assert resolved.model_name == "resnet18_pretrained"
    assert resolved.lr == 0.02
    assert resolved.rounds == 8
    assert resolved.er_p == 0.9
    assert resolved.augment is False
    assert resolved.verbose is False

    with pytest.raises(typer.BadParameter, match="Expected --set key=value"):
        run_cli._resolve_config("cnn_10k", None, {}, ["bad_override"], quiet=False)
    with pytest.raises(typer.BadParameter, match="Unknown config field"):
        run_cli._resolve_config("cnn_10k", None, {}, ["not_a_field=1"], quiet=False)

    tag_rx = run_cli._run_tag(run_cli._resolve_config("cnn_10k", None, {"trustgraph_delta_mode": "receiver"}, [], quiet=False))
    assert tag_rx.endswith("_descent_rxdelta")


def test_typer_cli_subcommands_and_single_run(tmp_path: Path) -> None:
    runner = CliRunner()
    root_help = runner.invoke(run_cli.app, ["--help"])
    assert root_help.exit_code == 0
    assert "--install-completion" not in root_help.stdout
    for cmd in ("single", "suite", "distributed", "transfer-bench", "list-suites"):
        assert cmd in root_help.stdout
        sub_help = runner.invoke(run_cli.app, [cmd, "--help"])
        assert sub_help.exit_code == 0
        assert "Args:" not in sub_help.stdout

    suites_out = runner.invoke(run_cli.app, ["list-suites"])
    assert suites_out.exit_code == 0

    for sub_app in (summarize.app, plot_figures.app):
        help_res = runner.invoke(sub_app, ["--help"])
        assert help_res.exit_code == 0
        assert "Args:" not in help_res.stdout
        assert "--install-completion" not in help_res.stdout

    single_res = runner.invoke(
        run_cli.app,
        [
            "single",
            "--preset",
            "cnn_10k",
            "--dataset",
            "synthetic",
            "--model",
            "cnn2x2",
            "--n-nodes",
            "4",
            "--samples-per-node",
            "20",
            "--rounds",
            "2",
            "--val-grad-sign",
            "descent",
            "--set",
            "er_p=0.5",
            "--set",
            "grad_clip_norm=5.0",
            "--eval-every",
            "2",
            "--device",
            "cpu",
            "--seeds",
            "42,43",
            "--quiet",
            "--output-root",
            str(tmp_path),
        ],
    )
    assert single_res.exit_code == 0, single_res.output
    saved_cfg_path: Path | None = None
    for seed in (42, 43):
        matches = list((tmp_path / "single").glob(f"*/seed_{seed}/summary.json"))
        assert len(matches) == 1
        assert (matches[0].parent / "run.log").exists()
        saved_cfg_path = matches[0].parent / "config.json"
        saved_cfg = json.loads(saved_cfg_path.read_text(encoding="utf-8"))["config"]
        assert saved_cfg["er_p"] == 0.5
        assert saved_cfg["grad_clip_norm"] == 5.0
        assert saved_cfg["val_grad_sign"] == "descent"

    assert saved_cfg_path is not None
    replay_res = runner.invoke(
        run_cli.app,
        [
            "single",
            "--config",
            str(saved_cfg_path),
            "--rounds",
            "1",
            "--seed",
            "99",
            "--quiet",
            "--output-root",
            str(tmp_path / "from_cfg"),
        ],
    )
    assert replay_res.exit_code == 0, replay_res.output
    replay_matches = list((tmp_path / "from_cfg" / "single").glob("*/seed_99/config.json"))
    assert len(replay_matches) == 1
    replay_cfg = json.loads(replay_matches[0].read_text(encoding="utf-8"))["config"]
    assert replay_cfg["rounds"] == 1
    assert replay_cfg["er_p"] == 0.5
    assert replay_cfg["grad_clip_norm"] == 5.0
