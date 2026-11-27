from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from experiments.phase1 import plot_figures


def test_parse_variant_key_and_lookup_helpers() -> None:
    assert plot_figures._parse_variant_key("trustgraph(descent,mclip,t=0.003,val=iid)[gc=10,spn=2500]") == "tg_stab"
    assert plot_figures._parse_variant_key("trustgraph(descent,mclip,t=0.003,a=0.3,val=iid)[gc=10]") == "tg_a03"
    assert plot_figures._parse_variant_key("trustgraph(descent,mclip,t=0.003,a=0.7,val=iid)[gc=10]") == "tg_a07"
    assert plot_figures._parse_variant_key("trustgraph(descent,euclidean,mclip,val=iid)[gc=10]") == "tg_euclidean"
    assert plot_figures._parse_variant_key("trustgraph(descent,norelu,mclip,t=0.003,val=iid)[gc=10]") == "tg_no_relu"
    assert plot_figures._parse_variant_key("trustgraph(ascent)[spn=500]") is None

    rows = [
        {"topology": "random", "attack": "none", "byz_pct": 0, "method": "trustgraph(descent,mclip,t=0.003,val=iid)[gc=10]"},
        {"topology": "random", "attack": "none", "byz_pct": 0, "method": "unknown"},
    ]
    idx = plot_figures._index_summary_rows(rows)
    assert list(idx.keys()) == [("random", "none", 0, "tg_stab")]

    runs = [{"run_name": "a/b/target"}]
    assert plot_figures._select_run_by_suffix(runs, "b/target") is runs[0]
    with pytest.raises(KeyError):
        plot_figures._select_run_by_suffix(runs, "missing")

    rx_data = {"baseline_50k_3seed": {"rows": [{"attack": "model_poison", "byz_pct": 30, "fpr_mean": 43.9}]}}
    assert plot_figures._select_baseline_row(rx_data, "model_poison")["fpr_mean"] == 43.9
    with pytest.raises(KeyError):
        plot_figures._select_baseline_row(rx_data, "random_noise")


def test_plot_figures_cli_renders_all_six_figures(tmp_path: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(plot_figures.app, ["--out-dir", str(tmp_path)])
    assert res.exit_code == 0, res.output
    expected_stems = (
        "fig1-attacks-50k",
        "fig2-topologies-and-stress-50k",
        "fig3-ablations-and-fpr-dynamics-50k",
        "fig4-fpr-rxdelta-50k",
        "fig5-v100-bandwidth-waste",
        "fig6-v100-distributed-ethernet",
    )
    for stem in expected_stems:
        assert (tmp_path / f"{stem}.png").exists()
        assert (tmp_path / f"{stem}.pdf").exists()
