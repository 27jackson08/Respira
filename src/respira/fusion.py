"""Fusion of per-RIV respiratory rate estimates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy import signal

from respira.estimators import SPECTRUM_NFFT, Estimate, _peak_from_spectrum, _prepare


def _finite_values(estimates: Mapping[str, float]) -> np.ndarray:
    values = np.asarray(list(estimates.values()), dtype=float)
    return values[np.isfinite(values)]


def smart_fusion(estimates: Mapping[str, float], sd_threshold_bpm: float = 4.0) -> Estimate:
    """Smart Fusion (Karlen et al. 2013).

    Average the estimates only if all are available and their standard deviation
    is at most ``sd_threshold_bpm``; otherwise return no estimate. Confidence is
    ``1 - sd / threshold``.
    """
    values = _finite_values(estimates)
    if values.size != len(estimates) or values.size == 0:
        return Estimate.empty()
    sd = float(np.std(values))
    if sd > sd_threshold_bpm:
        return Estimate.empty()
    return Estimate(float(np.mean(values)), 1.0 - sd / sd_threshold_bpm if sd_threshold_bpm else 1.0)


def median_fusion(estimates: Mapping[str, float]) -> Estimate:
    """Median of the available estimates (robust to one outlier)."""
    values = _finite_values(estimates)
    if values.size == 0:
        return Estimate.empty()
    spread = float(np.max(values) - np.min(values)) if values.size > 1 else 0.0
    return Estimate(float(np.median(values)), float(1.0 / (1.0 + spread)))


def spectral_fusion(signals: Sequence[np.ndarray], fs: float, band_hz: tuple[float, float]) -> Estimate:
    """Average the normalised periodograms of several RIVs and pick the joint peak."""
    spectra = []
    freqs = None
    for y in signals:
        prepared = _prepare(y)
        if prepared is None:
            continue
        f, psd = signal.periodogram(prepared, fs=fs, window="hann", nfft=max(SPECTRUM_NFFT, prepared.size))
        total = np.sum(psd)
        if total <= 0:
            continue
        spectra.append(psd / total)
        freqs = f
    if not spectra or freqs is None:
        return Estimate.empty()
    return _peak_from_spectrum(freqs, np.mean(spectra, axis=0), band_hz)


def temporal_smooth(rr: Sequence[float], alpha: float = 0.2) -> np.ndarray:
    """Exponential smoothing across consecutive windows (Charlton et al. 2016, FT1).

    ``rr_i = alpha * est_i + (1 - alpha) * rr_(i-1)``; missing estimates carry the
    previous value forward and the first valid estimate seeds the recursion.
    """
    out = np.full(len(rr), np.nan)
    previous = np.nan
    for i, value in enumerate(rr):
        if np.isfinite(value):
            previous = value if not np.isfinite(previous) else alpha * value + (1 - alpha) * previous
        out[i] = previous
    return out
