"""Filtering and resampling primitives.

Every function returns a new array and never mutates its input.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

MIN_FILTER_SAMPLES = 16


def _validate_band(fs: float, low_hz: float | None, high_hz: float | None) -> None:
    nyq = fs / 2.0
    if low_hz is not None and not 0 < low_hz < nyq:
        raise ValueError(f"low cutoff {low_hz} Hz must lie in (0, {nyq})")
    if high_hz is not None and not 0 < high_hz < nyq:
        raise ValueError(f"high cutoff {high_hz} Hz must lie in (0, {nyq})")
    if low_hz is not None and high_hz is not None and low_hz >= high_hz:
        raise ValueError("low cutoff must be below high cutoff")


def _sosfiltfilt_safe(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size < MIN_FILTER_SAMPLES:
        return x - np.mean(x) if x.size else x.copy()
    padlen = min(3 * (2 * sos.shape[0] + 1), x.size - 1)
    return signal.sosfiltfilt(sos, x, padlen=padlen)


def bandpass(x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter."""
    _validate_band(fs, low_hz, high_hz)
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return _sosfiltfilt_safe(sos, x)


def highpass(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth high-pass filter."""
    _validate_band(fs, cutoff_hz, None)
    sos = signal.butter(order, cutoff_hz, btype="highpass", fs=fs, output="sos")
    return _sosfiltfilt_safe(sos, x)


def lowpass(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter."""
    _validate_band(fs, None, cutoff_hz)
    sos = signal.butter(order, cutoff_hz, btype="lowpass", fs=fs, output="sos")
    return _sosfiltfilt_safe(sos, x)


def uniform_grid(t_start: float, t_end: float, fs: float) -> np.ndarray:
    """Uniform time grid ``[t_start, t_end)`` at ``fs`` Hz."""
    n = int(np.floor((t_end - t_start) * fs))
    return t_start + np.arange(max(n, 0)) / fs


def resample_uniform(
    t: np.ndarray, y: np.ndarray, fs: float, t_start: float, t_end: float
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate an irregularly sampled series onto a uniform grid.

    Samples outside the support of ``t`` are held at the edge values.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    grid = uniform_grid(t_start, t_end, fs)
    if t.size == 0:
        return grid, np.full(grid.shape, np.nan)
    order = np.argsort(t)
    return grid, np.interp(grid, t[order], y[order])


def remove_mean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x - np.nanmean(x) if x.size else x.copy()


def bpm_to_hz(bpm: float) -> float:
    return bpm / 60.0


def hz_to_bpm(hz: float) -> float:
    return hz * 60.0
