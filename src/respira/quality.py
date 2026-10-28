"""PPG signal quality assessment.

Implements a template-matching signal quality index (SQI) in the spirit of
Orphanidou et al. (2015): each pulse is correlated with the mean pulse of the
window, and the window is additionally checked for physiologically plausible
heart rate and beat-interval regularity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from respira.beats import Beats

TEMPLATE_PRE_S = 0.25
TEMPLATE_POST_S = 0.40
MIN_BEATS_FOR_TEMPLATE = 3
MAX_GAP_S = 3.0
MAX_INTERVAL_RATIO = 2.2


@dataclass(frozen=True)
class QualityReport:
    sqi: float
    hr_bpm: float
    n_beats: int
    is_plausible: bool

    def is_acceptable(self, sqi_threshold: float) -> bool:
        return bool(self.is_plausible and np.isfinite(self.sqi) and self.sqi >= sqi_threshold)


def _pulse_matrix(ppg: np.ndarray, peaks: np.ndarray, fs: float) -> np.ndarray:
    pre, post = int(TEMPLATE_PRE_S * fs), int(TEMPLATE_POST_S * fs)
    valid = peaks[(peaks - pre >= 0) & (peaks + post <= ppg.size)]
    if valid.size == 0:
        return np.zeros((0, pre + post))
    offsets = np.arange(-pre, post)
    return ppg[valid[:, None] + offsets[None, :]]


def template_correlation(ppg: np.ndarray, peaks: np.ndarray, fs: float) -> float:
    """Mean Pearson correlation between each pulse and the window's mean pulse."""
    pulses = _pulse_matrix(np.asarray(ppg, dtype=float), np.asarray(peaks, dtype=int), fs)
    if pulses.shape[0] < MIN_BEATS_FOR_TEMPLATE:
        return float("nan")
    centred = pulses - pulses.mean(axis=1, keepdims=True)
    template = centred.mean(axis=0)
    norms = np.linalg.norm(centred, axis=1) * np.linalg.norm(template)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = (centred @ template) / norms
    corr = corr[np.isfinite(corr)]
    return float(np.mean(corr)) if corr.size else float("nan")


def plausibility(beats: Beats, min_hr_bpm: float, max_hr_bpm: float) -> tuple[float, bool]:
    """Heart rate and a flag for plausible rate, gaps and interval regularity."""
    if beats.n < 2:
        return float("nan"), False
    ibi = beats.intervals
    hr = 60.0 / float(np.mean(ibi))
    ok = (
        min_hr_bpm <= hr <= max_hr_bpm
        and float(np.max(ibi)) <= MAX_GAP_S
        and float(np.max(ibi) / np.min(ibi)) <= MAX_INTERVAL_RATIO
    )
    return hr, bool(ok)


def assess_window(ppg: np.ndarray, beats: Beats, fs: float, min_hr_bpm: float, max_hr_bpm: float) -> QualityReport:
    """Quality report for the beats falling in one analysis window."""
    sqi = template_correlation(ppg, beats.peaks, fs)
    hr, plausible = plausibility(beats, min_hr_bpm, max_hr_bpm)
    return QualityReport(sqi=sqi, hr_bpm=hr, n_beats=beats.n, is_plausible=plausible)
