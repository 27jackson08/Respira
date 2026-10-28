"""Reference respiratory rate from BIDMC breath annotations and monitor numerics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from respira.estimators import estimate_periodogram
from respira.preprocess import bandpass, resample_uniform


@dataclass(frozen=True)
class ReferenceRR:
    rr_bpm: float
    rr_ann1: float
    rr_ann2: float
    is_valid: bool


def rr_from_breath_times(breath_times_s: np.ndarray, t0: float, t1: float, min_breaths: int = 3) -> float:
    """RR (bpm) from annotated breath instants in ``[t0, t1)``.

    Uses the mean inter-breath interval between the first and last breath in
    the window, which avoids the quantisation error of simply counting breaths.
    """
    times = np.asarray(breath_times_s, dtype=float)
    inside = np.sort(times[(times >= t0) & (times < t1)])
    if inside.size < min_breaths:
        return float("nan")
    span = inside[-1] - inside[0]
    if span <= 0:
        return float("nan")
    return 60.0 * (inside.size - 1) / span


def reference_rr(
    ann1_times_s: np.ndarray,
    ann2_times_s: np.ndarray,
    t0: float,
    t1: float,
    *,
    min_breaths: int = 3,
    max_disagreement_bpm: float = 2.0,
) -> ReferenceRR:
    """Consensus reference RR from two annotators.

    The window is valid only when both annotators yield an RR and they agree to
    within ``max_disagreement_bpm`` (Pimentel et al. 2017 used the same rule).
    """
    r1 = rr_from_breath_times(ann1_times_s, t0, t1, min_breaths)
    r2 = rr_from_breath_times(ann2_times_s, t0, t1, min_breaths)
    both = np.isfinite(r1) and np.isfinite(r2)
    valid = bool(both and abs(r1 - r2) <= max_disagreement_bpm)
    rr = 0.5 * (r1 + r2) if both else float("nan")
    return ReferenceRR(rr, r1, r2, valid)


def monitor_rr(numerics: pd.DataFrame, t0: float, t1: float, column: str = "RESP") -> float:
    """Mean of the bedside monitor's 1 Hz respiratory-rate numeric in the window."""
    if column not in numerics.columns or "Time [s]" not in numerics.columns:
        return float("nan")
    t = numerics["Time [s]"].to_numpy(dtype=float)
    values = numerics[column].to_numpy(dtype=float)
    mask = (t >= t0) & (t < t1) & np.isfinite(values)
    return float(np.mean(values[mask])) if np.any(mask) else float("nan")


def impedance_spectral_rr(
    resp: np.ndarray, fs: float, t0: float, t1: float, riv_fs: float, band_hz: tuple[float, float]
) -> float:
    """Spectral-peak RR of the raw impedance waveform, for comparison with the PPG estimates."""
    resp = np.asarray(resp, dtype=float)
    i0, i1 = round(t0 * fs), round(t1 * fs)
    seg = resp[i0:i1]
    if seg.size < fs:
        return float("nan")
    filtered = bandpass(seg, fs, band_hz[0], band_hz[1])
    _, y = resample_uniform(np.arange(seg.size) / fs, filtered, riv_fs, 0.0, seg.size / fs)
    return estimate_periodogram(y, riv_fs, band_hz).rr_bpm
