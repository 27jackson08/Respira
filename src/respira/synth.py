"""Synthetic PPG generator with known respiratory modulation.

Used by the test-suite and by ``respira demo`` so the pipeline can be exercised
without downloading the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticPPG:
    t: np.ndarray
    ppg: np.ndarray
    resp: np.ndarray
    breath_samples: np.ndarray
    fs: float
    hr_bpm: float
    rr_bpm: float


def _pulse_shape(phase: np.ndarray) -> np.ndarray:
    """Asymmetric pulse: sharp systolic upstroke, slower diastolic decay with a notch."""
    p = np.mod(phase, 2 * np.pi)
    systolic = np.exp(-((p - 1.0) ** 2) / (2 * 0.35**2))
    dicrotic = 0.35 * np.exp(-((p - 2.8) ** 2) / (2 * 0.5**2))
    return systolic + dicrotic


def synthetic_ppg(
    duration_s: float = 120.0,
    fs: float = 125.0,
    hr_bpm: float = 75.0,
    rr_bpm: float = 15.0,
    *,
    am_depth: float = 0.25,
    bw_depth: float = 0.25,
    fm_depth: float = 0.08,
    noise_std: float = 0.01,
    seed: int = 0,
) -> SyntheticPPG:
    """Generate a PPG with amplitude, baseline and frequency modulation at ``rr_bpm``."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    resp_phase = 2 * np.pi * rr_bpm / 60.0 * t
    resp = np.sin(resp_phase)

    inst_hr = hr_bpm * (1.0 + fm_depth * resp)
    cardiac_phase = np.cumsum(2 * np.pi * inst_hr / 60.0 / fs)
    pulse = _pulse_shape(cardiac_phase)

    amplitude = 1.0 + am_depth * resp
    baseline = bw_depth * np.sin(resp_phase + np.pi / 4)
    ppg = amplitude * pulse + baseline + rng.normal(0.0, noise_std, n)

    wrapped = np.mod(resp_phase, 2 * np.pi)
    onsets = np.flatnonzero(np.diff(wrapped) < 0) + 1
    breath_samples = np.concatenate([[0], onsets]) if onsets.size else np.array([0])
    return SyntheticPPG(t, ppg, resp, breath_samples, fs, hr_bpm, rr_bpm)


def synthetic_recording(subject: str = "99", annotator_offset_samples: int = 3, **kwargs: float):
    """Wrap :func:`synthetic_ppg` in a :class:`respira.data.bidmc.Recording`.

    The second annotator is simulated by shifting the true breath onsets by a
    few samples so the two annotations agree closely but not exactly.
    """
    import pandas as pd

    from respira.data.bidmc import Recording

    s = synthetic_ppg(**kwargs)
    n_sec = int(np.floor(s.t[-1])) + 1
    numerics = pd.DataFrame(
        {
            "Time [s]": np.arange(n_sec, dtype=float),
            "HR": np.full(n_sec, s.hr_bpm),
            "PULSE": np.full(n_sec, s.hr_bpm),
            "RESP": np.full(n_sec, s.rr_bpm),
            "SpO2": np.full(n_sec, 98.0),
        }
    )
    ann2 = s.breath_samples + annotator_offset_samples
    return Recording(
        subject=subject,
        fs=s.fs,
        time=s.t,
        ppg=s.ppg,
        resp=s.resp,
        ecg=None,
        numerics=numerics,
        breaths_ann1=s.breath_samples,
        breaths_ann2=ann2,
        meta={"Age": "synthetic", "Gender": "n/a", "Location": "synthetic"},
    )
