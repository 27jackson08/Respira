"""Respiratory-rate estimators operating on a uniformly sampled respiratory signal.

Each estimator has the signature ``(y, fs, band_hz) -> Estimate`` and returns
``nan`` when no estimate can be made. ``confidence`` is an estimator-specific
score in ``[0, 1]`` (higher is better) used for quality analysis.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import signal

MIN_SAMPLES = 16
SPECTRUM_NFFT = 4096
PEAK_POWER_HALF_WIDTH_HZ = 0.05
COUNT_THRESHOLD_FRACTION = 0.2


@dataclass(frozen=True)
class Estimate:
    rr_bpm: float
    confidence: float

    @classmethod
    def empty(cls) -> Estimate:
        return cls(float("nan"), float("nan"))

    @property
    def is_valid(self) -> bool:
        return bool(np.isfinite(self.rr_bpm))


Estimator = Callable[[np.ndarray, float, tuple[float, float]], Estimate]


def _prepare(y: np.ndarray) -> np.ndarray | None:
    y = np.asarray(y, dtype=float)
    if y.size < MIN_SAMPLES or not np.all(np.isfinite(y)) or np.ptp(y) == 0:
        return None
    return y - np.mean(y)


def _peak_from_spectrum(freqs: np.ndarray, psd: np.ndarray, band_hz: tuple[float, float]) -> Estimate:
    mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not np.any(mask) or not np.any(psd[mask] > 0):
        return Estimate.empty()
    f_band, p_band = freqs[mask], psd[mask]
    i = int(np.argmax(p_band))
    f_peak = f_band[i]
    near = np.abs(f_band - f_peak) <= PEAK_POWER_HALF_WIDTH_HZ
    confidence = float(np.sum(p_band[near]) / np.sum(p_band))
    return Estimate(60.0 * float(f_peak), confidence)


def estimate_periodogram(y: np.ndarray, fs: float, band_hz: tuple[float, float]) -> Estimate:
    """Hann-windowed periodogram over the whole window, zero-padded for resolution."""
    y = _prepare(y)
    if y is None:
        return Estimate.empty()
    freqs, psd = signal.periodogram(y, fs=fs, window="hann", nfft=max(SPECTRUM_NFFT, y.size))
    return _peak_from_spectrum(freqs, psd, band_hz)


def estimate_welch(y: np.ndarray, fs: float, band_hz: tuple[float, float]) -> Estimate:
    """Welch PSD with two 50 %-overlapping segments (lower variance, coarser resolution)."""
    y = _prepare(y)
    if y is None:
        return Estimate.empty()
    nperseg = max(y.size // 2, MIN_SAMPLES // 2)
    freqs, psd = signal.welch(y, fs=fs, nperseg=nperseg, nfft=max(SPECTRUM_NFFT, nperseg))
    return _peak_from_spectrum(freqs, psd, band_hz)


def burg(x: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    """Burg's method for autoregressive coefficients.

    Returns ``(a, noise_variance)`` where ``a[0] == 1`` and the model is
    ``x[n] + a[1] x[n-1] + ... + a[p] x[n-p] = e[n]``.
    """
    x = np.asarray(x, dtype=float)
    if order < 1 or order >= x.size:
        raise ValueError("order must satisfy 1 <= order < len(x)")
    f = x.copy()
    b = x.copy()
    a = np.array([1.0])
    err = float(np.dot(x, x)) / x.size
    for m in range(1, order + 1):
        ff = f[m:]
        bb = b[m - 1 : -1]
        denom = float(np.dot(ff, ff) + np.dot(bb, bb))
        if denom <= 0:
            break
        k = -2.0 * float(np.dot(ff, bb)) / denom
        a_ext = np.append(a, 0.0)
        a = a_ext + k * a_ext[::-1]
        f_next = ff + k * bb
        b_next = bb + k * ff
        f = np.concatenate([f[:m], f_next])
        b = np.concatenate([b[:m], b_next])
        err *= 1.0 - k * k
    return a, err


def make_ar_estimator(order: int = 8) -> Estimator:
    """Autoregressive spectral estimator (Burg). Confidence is the magnitude of the pole
    nearest the spectral peak (Pimentel et al. 2017 use pole magnitude for ranking)."""

    def estimate_ar(y: np.ndarray, fs: float, band_hz: tuple[float, float]) -> Estimate:
        y = _prepare(y)
        if y is None or y.size <= order:
            return Estimate.empty()
        a, err = burg(y, order)
        freqs = np.linspace(band_hz[0], band_hz[1], 2048)
        _, h = signal.freqz([1.0], a, worN=freqs, fs=fs)
        psd = err * np.abs(h) ** 2
        est = _peak_from_spectrum(freqs, psd, band_hz)
        if not est.is_valid:
            return est
        poles = np.roots(a)
        pole_freqs = np.angle(poles) * fs / (2.0 * np.pi)
        nearest = int(np.argmin(np.abs(pole_freqs - est.rr_bpm / 60.0)))
        return Estimate(est.rr_bpm, float(min(np.abs(poles[nearest]), 1.0)))

    estimate_ar.__name__ = "estimate_ar"
    return estimate_ar


def estimate_autocorrelation(y: np.ndarray, fs: float, band_hz: tuple[float, float]) -> Estimate:
    """RR from the first autocorrelation maximum within the plausible lag range."""
    y = _prepare(y)
    if y is None:
        return Estimate.empty()
    acf = signal.correlate(y, y, mode="full")[y.size - 1 :]
    acf = acf / acf[0]
    lag_min = int(np.ceil(fs / band_hz[1]))
    lag_max = int(np.floor(fs / band_hz[0]))
    lag_max = min(lag_max, acf.size - 1)
    if lag_max <= lag_min:
        return Estimate.empty()
    segment = acf[lag_min : lag_max + 1]
    peaks, _ = signal.find_peaks(segment)
    if peaks.size == 0:
        return Estimate.empty()
    best = peaks[int(np.argmax(segment[peaks]))]
    lag = lag_min + best
    return Estimate(60.0 * fs / lag, float(np.clip(segment[best], 0.0, 1.0)))


def _valid_breath_maxima(y: np.ndarray, maxima: np.ndarray, threshold: float) -> list[int]:
    """Count-orig rule: keep maxima above threshold separated by a sub-zero minimum."""
    accepted: list[int] = []
    for idx in maxima:
        if y[idx] < threshold:
            continue
        if accepted and np.min(y[accepted[-1] : idx + 1]) >= 0:
            if y[idx] > y[accepted[-1]]:
                accepted = [*accepted[:-1], int(idx)]
            continue
        accepted = [*accepted, int(idx)]
    return accepted


def estimate_count(y: np.ndarray, fs: float, band_hz: tuple[float, float]) -> Estimate:
    """Time-domain breath counting ("Count-orig", Schäfer & Kratky 2008)."""
    y = _prepare(y)
    if y is None:
        return Estimate.empty()
    maxima, _ = signal.find_peaks(y)
    if maxima.size < 2:
        return Estimate.empty()
    threshold = COUNT_THRESHOLD_FRACTION * np.percentile(y[maxima], 75)
    accepted = _valid_breath_maxima(y, maxima, threshold)
    if len(accepted) < 2:
        return Estimate.empty()
    intervals = np.diff(np.asarray(accepted)) / fs
    rr = 60.0 / float(np.mean(intervals))
    if not band_hz[0] * 60.0 <= rr <= band_hz[1] * 60.0:
        return Estimate.empty()
    cv = float(np.std(intervals) / np.mean(intervals)) if intervals.size > 1 else 0.0
    return Estimate(rr, float(np.clip(1.0 - cv, 0.0, 1.0)))


def build_estimators(ar_order: int = 8) -> dict[str, Estimator]:
    """Registry of estimators keyed by short name."""
    return {
        "periodogram": estimate_periodogram,
        "welch": estimate_welch,
        "ar": make_ar_estimator(ar_order),
        "autocorr": estimate_autocorrelation,
        "count": estimate_count,
    }
