import numpy as np
import pytest

from respira import estimators as est

FS = 4.0
BAND = (4.0 / 60.0, 1.0)


def _resp(rr_bpm: float, duration: float = 32.0, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * FS)) / FS
    return np.sin(2 * np.pi * rr_bpm / 60.0 * t) + noise * rng.normal(size=t.size)


@pytest.fixture(scope="module")
def registry():
    return est.build_estimators(ar_order=8)


@pytest.mark.parametrize("name", ["periodogram", "welch", "ar", "autocorr", "count"])
@pytest.mark.parametrize("rr", [8.0, 15.0, 24.0, 36.0])
def test_estimator_recovers_sinusoid_rate(registry, name, rr):
    result = registry[name](_resp(rr), FS, BAND)
    tolerance = 2.5 if name in ("welch", "autocorr") else 1.5
    assert result.is_valid
    assert abs(result.rr_bpm - rr) < tolerance
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.parametrize("name", ["periodogram", "welch", "ar", "autocorr", "count"])
def test_estimator_handles_degenerate_input(registry, name):
    assert not registry[name](np.zeros(128), FS, BAND).is_valid
    assert not registry[name](np.ones(5), FS, BAND).is_valid
    bad = _resp(15.0)
    bad[3] = np.nan
    assert not registry[name](bad, FS, BAND).is_valid


def test_noisy_sinusoid_lowers_spectral_confidence():
    clean = est.estimate_periodogram(_resp(15.0), FS, BAND)
    noisy = est.estimate_periodogram(_resp(15.0, noise=1.5, seed=3), FS, BAND)
    assert clean.confidence > noisy.confidence


def test_burg_recovers_ar2_coefficients():
    rng = np.random.default_rng(0)
    a_true = np.array([1.0, -1.5, 0.7])
    n = 20000
    e = rng.normal(size=n)
    x = np.zeros(n)
    for i in range(2, n):
        x[i] = -a_true[1] * x[i - 1] - a_true[2] * x[i - 2] + e[i]
    a, var = est.burg(x, 2)
    assert np.allclose(a, a_true, atol=0.03)
    assert abs(var - 1.0) < 0.1


def test_burg_rejects_bad_order():
    with pytest.raises(ValueError):
        est.burg(np.arange(10.0), 0)
    with pytest.raises(ValueError):
        est.burg(np.arange(10.0), 10)


def test_count_rejects_out_of_band_rate():
    # 2 bpm is below the 4 bpm floor
    assert not est.estimate_count(_resp(2.0, duration=120.0), FS, BAND).is_valid


def test_estimate_empty_helper():
    e = est.Estimate.empty()
    assert not e.is_valid and np.isnan(e.confidence)
