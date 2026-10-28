from pathlib import Path

import numpy as np
import pytest

from respira.data import bidmc


def test_load_recording_parses_all_four_files(bidmc_dir):
    rec = bidmc.load_recording(bidmc_dir, "01")
    assert rec.subject == "01"
    assert rec.fs == 125.0
    assert rec.ppg.size == rec.resp.size == rec.time.size
    assert rec.ecg is not None
    assert list(rec.numerics.columns) == ["Time [s]", "HR", "PULSE", "RESP", "SpO2"]
    assert rec.breaths_ann1.dtype.kind == "i"
    assert rec.meta["Age"] == "61"
    assert rec.meta["Location"] == "micu"
    assert rec.duration_s == pytest.approx(100.0, abs=0.1)


def test_ragged_breath_columns_are_handled(bidmc_dir):
    rec = bidmc.load_recording(bidmc_dir, "02")
    assert rec.breaths_ann2.size == rec.breaths_ann1.size - 1
    assert np.all(np.diff(rec.breaths_ann1) > 0)


def test_available_and_load_all(bidmc_dir):
    assert bidmc.available_subjects(bidmc_dir) == ("01", "02")
    subjects = [r.subject for r in bidmc.load_all(bidmc_dir)]
    assert subjects == ["01", "02"]
    assert [r.subject for r in bidmc.load_all(bidmc_dir, ["02"])] == ["02"]


def test_missing_subject_raises(bidmc_dir):
    with pytest.raises(FileNotFoundError, match="missing"):
        bidmc.load_recording(bidmc_dir, "03")


def test_signals_without_pleth_raise(tmp_path, bidmc_dir):
    for name in bidmc.expected_files("01"):
        (tmp_path / name).write_bytes((bidmc_dir / name).read_bytes())
    (tmp_path / "bidmc_01_Signals.csv").write_text("Time [s], RESP\n0,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="PLETH"):
        bidmc.load_recording(tmp_path, "01")


def test_parse_fix():
    meta = bidmc.parse_fix("bidmc_01\nAge: 88\nGender: M\nURL: http://x/y\n")
    assert meta == {"Age": "88", "Gender": "M", "URL": "http://x/y"}


def test_download_skips_existing_and_fetches_missing(tmp_path, monkeypatch):
    fetched: list[str] = []

    def fake_fetch(url: str, dest: Path) -> None:
        fetched.append(url)
        dest.write_text("x", encoding="utf-8")

    monkeypatch.setattr(bidmc, "_fetch", fake_fetch)
    (tmp_path / "bidmc_05_Fix.txt").write_text("existing", encoding="utf-8")
    written = bidmc.download(tmp_path, ["05"], progress=lambda _: None)
    assert len(written) == 3
    assert all(u.startswith(bidmc.BASE_URL) for u in fetched)
    assert (tmp_path / "bidmc_05_Fix.txt").read_text(encoding="utf-8") == "existing"
    written = bidmc.download(tmp_path, ["05"], overwrite=True)
    assert len(written) == 4


def test_download_rejects_unknown_subject(tmp_path):
    with pytest.raises(ValueError):
        bidmc.download(tmp_path, ["99"])


def test_fetch_retries_then_raises(tmp_path, monkeypatch):
    import urllib.error

    calls = {"n": 0}

    def failing_urlopen(*_args, **_kwargs):
        calls["n"] += 1
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(bidmc.urllib.request, "urlopen", failing_urlopen)
    with pytest.raises(RuntimeError):
        bidmc._fetch("https://example.invalid/x", tmp_path / "x")
    assert calls["n"] == bidmc.DOWNLOAD_RETRIES


def test_missing_markers_in_numerics_and_signal_gaps(tmp_path, bidmc_dir):
    for name in bidmc.expected_files("01"):
        (tmp_path / name).write_bytes((bidmc_dir / name).read_bytes())
    numerics = tmp_path / "bidmc_01_Numerics.csv"
    lines = numerics.read_text(encoding="utf-8").splitlines()
    lines[1] = "0,-,93,-,97"
    numerics.write_text("\n".join(lines) + "\n", encoding="utf-8")
    signals = tmp_path / "bidmc_01_Signals.csv"
    sig_lines = signals.read_text(encoding="utf-8").splitlines()
    parts = sig_lines[5].split(",")
    parts[2] = "nan"
    sig_lines[5] = ",".join(parts)
    signals.write_text("\n".join(sig_lines) + "\n", encoding="utf-8")

    rec = bidmc.load_recording(tmp_path, "01")
    assert np.isnan(rec.numerics["RESP"].iloc[0]) and rec.numerics["PULSE"].iloc[0] == 93
    assert rec.numerics["RESP"].dtype == float
    assert np.all(np.isfinite(rec.ppg))
    assert rec.ppg[4] == pytest.approx(0.5 * (rec.ppg[3] + rec.ppg[5]))


def test_all_nan_signal_column_becomes_zeros():
    import pandas as pd

    frame = pd.DataFrame({"PLETH": [np.nan, np.nan]})
    assert bidmc._signal_column(frame, "PLETH").tolist() == [0.0, 0.0]
