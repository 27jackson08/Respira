import json

import pandas as pd
import pytest

from respira import cli


def test_estimate_then_evaluate_end_to_end(bidmc_dir, tmp_path):
    out = tmp_path / "results"
    assert cli.main(["estimate", "--data", str(bidmc_dir), "--out", str(out), "--window", "32", "--step", "16"]) == 0
    estimates = pd.read_csv(out / "estimates.csv")
    assert set(estimates["subject"].astype(str).str.zfill(2)) == {"01", "02"}
    assert json.loads((out / "config.json").read_text())["step_s"] == 16.0

    assert cli.main(["evaluate", "--out", str(out)]) == 0
    for name in (
        "summary.md",
        "metrics_by_algorithm.csv",
        "metrics_by_subject.csv",
        "sqi_sweep.csv",
        "comparators.csv",
    ):
        assert (out / name).is_file(), name
    figures = out / "figures"
    assert {p.name for p in figures.glob("*.png")} >= {
        "mae_by_algorithm.png",
        "bland_altman_best.png",
        "quality_tradeoff_best.png",
    }
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "Best algorithm" in summary
    assert "32 s windows every 16 s" in summary


def test_run_combines_both_steps(bidmc_dir, tmp_path):
    out = tmp_path / "results"
    assert cli.main(["run", "--data", str(bidmc_dir), "--out", str(out), "--subjects", "01", "--step", "16"]) == 0
    assert (out / "summary.md").is_file()


def test_evaluate_without_estimates_fails(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["evaluate", "--out", str(tmp_path)])


def test_estimate_with_empty_data_dir_fails(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["estimate", "--data", str(tmp_path), "--out", str(tmp_path / "r")])


def test_inspect_and_demo_write_figures(bidmc_dir, tmp_path):
    fig = tmp_path / "inspect.png"
    assert cli.main(["inspect", "01", "--data", str(bidmc_dir), "--window-index", "2", "--out", str(fig)]) == 0
    assert fig.stat().st_size > 1000
    demo = tmp_path / "demo.png"
    assert cli.main(["demo", "--duration", "96", "--rr", "12", "--out", str(demo)]) == 0
    assert demo.is_file()


def test_download_command_uses_downloader(tmp_path, monkeypatch):
    calls = {}

    def fake_download(dest, subjects, overwrite, progress):
        calls.update(dest=dest, subjects=subjects, overwrite=overwrite)
        return ()

    monkeypatch.setattr(cli, "download", fake_download)
    assert cli.main(["download", "--dest", str(tmp_path), "--subjects", "01", "02"]) == 0
    assert calls["subjects"] == ["01", "02"] and calls["overwrite"] is False


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "respira" in capsys.readouterr().out


def test_band_flag_changes_config(bidmc_dir, tmp_path):
    out = tmp_path / "results"
    assert (
        cli.main(["estimate", "--data", str(bidmc_dir), "--out", str(out), "--subjects", "01", "--band", "8", "40"])
        == 0
    )
    assert json.loads((out / "config.json").read_text())["rr_band_bpm"] == [8.0, 40.0]
    assert cli.main(["evaluate", "--out", str(out)]) == 0
    assert "(8.0, 40.0)" in (out / "summary.md").read_text(encoding="utf-8")
