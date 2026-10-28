import numpy as np
import pytest

from respira import preprocess as pp

FS = 100.0


def _tone(freq: float, duration: float = 60.0) -> np.ndarray:
    t = np.arange(int(duration * FS)) / FS
    return np.sin(2 * np.pi * freq * t)


def test_bandpass_keeps_in_band_and_removes_out_of_band():
    x = _tone(1.0) + _tone(10.0) + 5.0
    y = pp.bandpass(x, FS, 0.5, 3.0)
    # the 1 Hz component survives with near-unity gain, the 10 Hz one and the offset are gone
    assert np.isclose(np.std(y), np.std(_tone(1.0)), rtol=0.05)
    assert abs(np.mean(y)) < 0.05


def test_highpass_and_lowpass_complement():
    x = _tone(0.1) + _tone(5.0)
    hp = pp.highpass(x, FS, 1.0)
    lp = pp.lowpass(x, FS, 1.0)
    assert np.isclose(np.std(hp), np.std(_tone(5.0)), rtol=0.05)
    assert np.isclose(np.std(lp[200:-200]), np.std(_tone(0.1)[200:-200]), rtol=0.1)


def test_filters_do_not_mutate_input():
    x = _tone(1.0)
    before = x.copy()
    pp.bandpass(x, FS, 0.5, 3.0)
    assert np.array_equal(x, before)


@pytest.mark.parametrize("low,high", [(0.0, 1.0), (60.0, 70.0), (2.0, 1.0)])
def test_invalid_band_raises(low, high):
    with pytest.raises(ValueError):
        pp.bandpass(_tone(1.0), FS, low, high)


def test_short_signal_is_mean_removed_not_filtered():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = pp.bandpass(x, FS, 0.5, 3.0)
    assert np.allclose(y, x - 2.5)


def test_resample_uniform_interpolates_linearly():
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 10.0, 20.0])
    grid, out = pp.resample_uniform(t, y, fs=2.0, t_start=0.0, t_end=2.0)
    assert np.allclose(grid, [0.0, 0.5, 1.0, 1.5])
    assert np.allclose(out, [0.0, 5.0, 10.0, 15.0])


def test_resample_uniform_empty_series_gives_nan():
    grid, out = pp.resample_uniform(np.array([]), np.array([]), 4.0, 0.0, 1.0)
    assert grid.size == 4
    assert np.all(np.isnan(out))


def test_unit_conversions():
    assert pp.bpm_to_hz(60.0) == 1.0
    assert pp.hz_to_bpm(0.25) == 15.0
    assert pp.remove_mean(np.array([1.0, 3.0])).tolist() == [-1.0, 1.0]
