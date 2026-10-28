import numpy as np
import pandas as pd

from respira import reference as ref


def test_rr_from_breath_times_uses_mean_interval():
    breaths = np.arange(0.0, 60.0, 4.0)  # 15 bpm
    assert abs(ref.rr_from_breath_times(breaths, 0.0, 32.0) - 15.0) < 1e-9
    assert np.isnan(ref.rr_from_breath_times(breaths[:2], 0.0, 32.0))
    assert np.isnan(ref.rr_from_breath_times(np.array([1.0, 1.0, 1.0]), 0.0, 32.0))


def test_reference_rr_requires_agreement():
    a1 = np.arange(0.0, 60.0, 4.0)
    a2 = a1 + 0.2
    out = ref.reference_rr(a1, a2, 0.0, 32.0)
    assert out.is_valid and abs(out.rr_bpm - 15.0) < 0.5
    far = np.arange(0.0, 60.0, 2.5)  # 24 bpm
    out = ref.reference_rr(a1, far, 0.0, 32.0, max_disagreement_bpm=2.0)
    assert not out.is_valid and np.isfinite(out.rr_bpm)
    out = ref.reference_rr(a1, np.array([]), 0.0, 32.0)
    assert not out.is_valid and np.isnan(out.rr_bpm)


def test_monitor_rr_averages_window():
    numerics = pd.DataFrame({"Time [s]": np.arange(10.0), "RESP": np.r_[np.full(5, 10.0), np.full(5, 20.0)]})
    assert ref.monitor_rr(numerics, 0.0, 5.0) == 10.0
    assert ref.monitor_rr(numerics, 0.0, 10.0) == 15.0
    assert np.isnan(ref.monitor_rr(numerics, 50.0, 60.0))
    assert np.isnan(ref.monitor_rr(pd.DataFrame({"x": [1]}), 0.0, 1.0))


def test_impedance_spectral_rr_recovers_rate():
    fs = 125.0
    t = np.arange(int(64 * fs)) / fs
    resp = np.sin(2 * np.pi * 0.3 * t)
    rr = ref.impedance_spectral_rr(resp, fs, 0.0, 32.0, 4.0, (4 / 60, 1.0))
    assert abs(rr - 18.0) < 1.0
    assert np.isnan(ref.impedance_spectral_rr(resp, fs, 0.0, 0.1, 4.0, (4 / 60, 1.0)))
