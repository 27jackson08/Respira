"""PPG pulse detection.

Peaks are located on a band-passed copy of the PPG (which removes baseline
wander) and then refined on the lightly filtered signal used for amplitude
measurement. Each returned beat has a systolic peak and the preceding trough
(pulse onset).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import signal

from respira.preprocess import bandpass

PROMINENCE_FRACTION = 0.3
"""Peaks with prominence below this fraction of the median prominence are rejected."""

REFINE_HALF_WIDTH_S = 0.06
"""Half-width of the neighbourhood searched when refining peak/trough positions."""


@dataclass(frozen=True)
class Beats:
    """Indices of detected pulses. ``troughs[i]`` precedes ``peaks[i]``."""

    peaks: np.ndarray
    troughs: np.ndarray
    fs: float

    def __post_init__(self) -> None:
        if self.peaks.shape != self.troughs.shape:
            raise ValueError("peaks and troughs must have the same length")

    @property
    def n(self) -> int:
        return int(self.peaks.size)

    @property
    def peak_times(self) -> np.ndarray:
        return self.peaks / self.fs

    @property
    def intervals(self) -> np.ndarray:
        """Peak-to-peak intervals in seconds."""
        return np.diff(self.peak_times)

    def in_window(self, t0: float, t1: float) -> Beats:
        mask = (self.peak_times >= t0) & (self.peak_times < t1)
        return Beats(self.peaks[mask], self.troughs[mask], self.fs)


def _candidate_peaks(x: np.ndarray, fs: float, max_hr_bpm: float) -> np.ndarray:
    min_distance = max(int(fs * 60.0 / max_hr_bpm), 1)
    peaks, props = signal.find_peaks(x, distance=min_distance, prominence=0)
    if peaks.size == 0:
        return peaks
    prominences = props["prominences"]
    threshold = PROMINENCE_FRACTION * np.median(prominences)
    return peaks[prominences >= threshold]


def _troughs_between(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Minimum between each consecutive pair of peaks (length ``len(peaks) - 1``)."""
    return np.array(
        [a + int(np.argmin(x[a:b])) for a, b in itertools.pairwise(peaks)],
        dtype=int,
    )


def _refine(x: np.ndarray, idx: np.ndarray, half_width: int, find_max: bool) -> np.ndarray:
    refined = []
    for i in idx:
        lo, hi = max(i - half_width, 0), min(i + half_width + 1, x.size)
        seg = x[lo:hi]
        offset = int(np.argmax(seg)) if find_max else int(np.argmin(seg))
        refined.append(lo + offset)
    return np.array(refined, dtype=int)


def _plausible(peaks: np.ndarray, troughs: np.ndarray, fs: float, min_hr: float, max_hr: float) -> np.ndarray:
    """Mask of beats whose interval to the previous peak is physiologically plausible."""
    if peaks.size == 0:
        return np.zeros(0, dtype=bool)
    ibi = np.diff(peaks, prepend=peaks[0] - int(fs * 60.0 / 75.0)) / fs
    return (ibi >= 60.0 / max_hr) & (ibi <= 60.0 / min_hr) & (peaks > troughs)


def detect_beats(
    ppg: np.ndarray,
    fs: float,
    *,
    beat_band_hz: tuple[float, float] = (0.5, 8.0),
    min_hr_bpm: float = 40.0,
    max_hr_bpm: float = 180.0,
    measurement_signal: np.ndarray | None = None,
) -> Beats:
    """Detect systolic peaks and preceding troughs in a PPG signal.

    ``measurement_signal`` is the (lightly filtered) signal on which peak and
    trough positions are refined; it defaults to the raw input.
    """
    ppg = np.asarray(ppg, dtype=float)
    if ppg.ndim != 1:
        raise ValueError("ppg must be one-dimensional")
    empty = Beats(np.zeros(0, dtype=int), np.zeros(0, dtype=int), fs)
    if ppg.size < int(fs):
        return empty

    x = bandpass(ppg, fs, *beat_band_hz)
    peaks = _candidate_peaks(x, fs, max_hr_bpm)
    if peaks.size < 2:
        return empty
    troughs = _troughs_between(x, peaks)
    peaks = peaks[1:]

    ref = ppg if measurement_signal is None else np.asarray(measurement_signal, dtype=float)
    half = int(REFINE_HALF_WIDTH_S * fs)
    peaks = _refine(ref, peaks, half, find_max=True)
    troughs = _refine(ref, troughs, half, find_max=False)

    keep = _plausible(peaks, troughs, fs, min_hr_bpm, max_hr_bpm)
    return Beats(peaks[keep], troughs[keep], fs)
