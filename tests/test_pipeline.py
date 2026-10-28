import numpy as np
import pandas as pd

from respira import evaluate as ev
from respira.config import ALL_RIV_NAMES
from respira.pipeline import FUSION_SOURCES, WINDOW_COLUMNS, process_recording, process_recordings


def test_process_recording_layout(recording, cfg):
    df = process_recording(recording, cfg)
    assert set(WINDOW_COLUMNS).issubset(df.columns)
    assert set(df["source"]) == set(ALL_RIV_NAMES) | set(FUSION_SOURCES)
    n_windows = df["window"].nunique()
    assert n_windows == 12  # (120 - 32) / 8 + 1
    per_window = df.groupby("window").size().unique()
    assert len(per_window) == 1
    assert df["algorithm"].str.contains(":").all()


def test_estimates_match_true_rate(recording, cfg):
    df = process_recording(recording, cfg)
    summary = ev.metrics_by_algorithm(df)
    assert summary["coverage"].max() == 1.0
    best = summary.iloc[0]
    assert best["mae"] < 1.0
    # every single-source periodogram estimate should land near 15 bpm on a clean signal
    for source in ALL_RIV_NAMES:
        row = summary[summary["algorithm"] == f"{source}:periodogram"].iloc[0]
        assert row["mae"] < 1.5, source


def test_reference_and_quality_columns(recording, cfg):
    df = process_recording(recording, cfg)
    win = ev.window_level(df)
    assert win["ref_valid"].all()
    assert np.allclose(win["rr_ref"], recording.numerics["RESP"].iloc[0], atol=0.6)
    assert (win["sqi"] > 0.9).all()
    assert win["plausible"].all()
    assert np.allclose(win["rr_monitor"], 15.0)
    assert np.all(np.abs(win["rr_impedance"] - 15.0) < 1.5)


def test_temporal_smoothing_rows_exist(recording, cfg):
    df = process_recording(recording, cfg)
    ts = df[df["source"] == "sf_ts"]
    sf = df[df["source"] == "sf"]
    assert len(ts) == len(sf)
    assert ts["rr_est"].notna().sum() >= sf["rr_est"].notna().sum()


def test_process_recordings_concatenates_and_reports_progress(recording, cfg):
    seen = []
    df = process_recordings([recording, recording], cfg, progress=seen.append)
    assert seen == ["01", "01"]
    assert len(df) == 2 * len(process_recording(recording, cfg))
    empty = process_recordings([], cfg)
    assert isinstance(empty, pd.DataFrame) and empty.empty
