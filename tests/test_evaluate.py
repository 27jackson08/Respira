import numpy as np
import pandas as pd
import pytest

from respira import evaluate as ev


def test_compute_metrics_known_values():
    ref = np.array([10.0, 20.0, 30.0, 40.0])
    est = np.array([11.0, 18.0, np.nan, 40.0])
    m = ev.compute_metrics(est, ref)
    assert m.n_ref == 4 and m.n_est == 3
    assert m.coverage == pytest.approx(0.75)
    assert m.mae == pytest.approx(1.0)
    assert m.bias == pytest.approx(-1 / 3)
    assert m.cp2 == pytest.approx(1.0)
    assert m.loa_low < m.bias < m.loa_high
    assert m.rmse == pytest.approx(np.sqrt(5 / 3))


def test_compute_metrics_no_estimates():
    m = ev.compute_metrics(np.array([np.nan, np.nan]), np.array([10.0, 12.0]))
    assert m.n_est == 0 and m.coverage == 0.0 and np.isnan(m.mae)
    with pytest.raises(ValueError):
        ev.compute_metrics(np.zeros(2), np.zeros(3))


@pytest.fixture
def frame() -> pd.DataFrame:
    rows = []
    for subject in ("01", "02"):
        for w in range(4):
            ctx = {
                "subject": subject,
                "window": w,
                "t0": 8.0 * w,
                "t1": 8.0 * w + 32,
                "sqi": 0.5 + 0.15 * w,
                "hr_bpm": 70.0,
                "n_beats": 40,
                "plausible": True,
                "rr_ref": 15.0,
                "rr_ann1": 15.0,
                "rr_ann2": 15.0,
                "ref_valid": w != 3,
                "rr_monitor": 16.0,
                "rr_impedance": 14.0,
            }
            rows.append({**ctx, "source": "bw", "method": "ar", "rr_est": 15.0 + w})
            rows.append({**ctx, "source": "sf", "method": "ar", "rr_est": 15.5 if w < 2 else np.nan})
    df = pd.DataFrame(rows)
    return df.assign(algorithm=df["source"] + ":" + df["method"])


def test_metrics_by_algorithm_sorted_and_gated(frame):
    summary = ev.metrics_by_algorithm(frame)
    assert list(summary["algorithm"]) == ["sf:ar", "bw:ar"]
    assert summary.iloc[0]["coverage"] == pytest.approx(2 / 3)
    gated = ev.metrics_by_algorithm(frame, sqi_threshold=0.7)
    assert gated[gated["algorithm"] == "bw:ar"].iloc[0]["n_est"] == 2


def test_best_algorithm_respects_coverage(frame):
    summary = ev.metrics_by_algorithm(frame)
    assert ev.best_algorithm(summary, min_coverage=0.9) == ("bw", "ar")
    assert ev.best_algorithm(summary, min_coverage=0.5) == ("sf", "ar")
    with pytest.raises(ValueError):
        ev.best_algorithm(summary.iloc[0:0])


def test_subject_sqi_and_comparator_tables(frame):
    by_subject = ev.metrics_by_subject(frame, "bw", "ar")
    assert list(by_subject["subject"]) == ["01", "02"]
    assert by_subject["mae"].tolist() == pytest.approx([1.0, 1.0])

    sweep = ev.sqi_sweep(frame, "bw", "ar", thresholds=[0.0, 0.6, 0.9])
    assert sweep["coverage"].tolist() == pytest.approx([1.0, 2 / 3, 0.0])
    assert np.isnan(sweep["mae"].iloc[-1])

    bins = ev.metrics_by_sqi_bin(frame, "bw", "ar", edges=(0.0, 0.6, 1.01))
    assert bins["n_est"].tolist() == [2, 4]

    comp = ev.comparator_metrics(frame)
    assert set(comp["comparator"]) == {"monitor_numeric", "impedance_spectral"}
    assert comp.set_index("comparator").loc["monitor_numeric", "mae"] == pytest.approx(1.0)

    win = ev.window_level(frame)
    assert len(win) == 8 and "rr_est" not in win.columns


def test_quality_gate_can_ignore_plausibility(frame):
    flagged = frame.assign(plausible=False)
    assert ev.quality_gate(flagged, 0.0).empty
    assert len(ev.quality_gate(flagged, 0.0, require_plausible=False)) == len(frame)


def test_sqi_sweep_ignores_plausibility_and_reports_full_coverage_at_zero(frame):
    flagged = frame.assign(plausible=False)
    sweep = ev.sqi_sweep(flagged, "bw", "ar", thresholds=[0.0, 0.9])
    assert sweep["coverage"].iloc[0] == pytest.approx(1.0)
    assert sweep["coverage"].iloc[1] < 1.0


def test_confidence_sweep_drops_least_confident(frame):
    conf = frame.assign(confidence=frame["window"].astype(float) / 10.0)
    sweep = ev.confidence_sweep(conf, "bw", "ar", drop_fractions=[0.0, 0.5, 0.9])
    assert sweep["coverage"].iloc[0] == pytest.approx(1.0)
    assert sweep["coverage"].is_monotonic_decreasing
    assert sweep["n_windows"].iloc[-1] >= 1
    empty = ev.confidence_sweep(conf.assign(confidence=np.nan), "bw", "ar", drop_fractions=[0.0])
    assert empty["n_windows"].iloc[0] == 0


def test_plausibility_gate_table_partitions_windows(frame):
    mixed = frame.assign(plausible=frame["window"] % 2 == 0)
    table = ev.plausibility_gate_table(mixed, "bw", "ar")
    assert list(table["gate"]) == ["all windows", "plausible beats only", "implausible beats only"]
    assert table["n_windows"].iloc[1] + table["n_windows"].iloc[2] == table["n_windows"].iloc[0]
