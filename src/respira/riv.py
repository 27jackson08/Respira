"""Respiratory-induced variations (RIVs) derived from detected PPG beats.

* **BW** (baseline wander / RIIV): mean of trough and peak amplitude per beat.
* **AM** (amplitude modulation / RIAV): peak minus trough amplitude per beat.
* **FM** (frequency modulation / RIFV): peak-to-peak interval, placed at the
  midpoint between the two peaks.

Each beat series is irregularly sampled; :func:`respiratory_signal` turns one
into a uniformly sampled, band-limited signal ready for RR estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from respira.beats import Beats
from respira.config import PipelineConfig
from respira.preprocess import bandpass, remove_mean, resample_uniform

MAX_FM_INTERVAL_S = 60.0 / 40.0
"""Longest peak-to-peak interval (40 beats/min) accepted as a single cardiac cycle."""


@dataclass(frozen=True)
class BeatSeries:
    """An irregularly sampled series of per-beat features."""

    t: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.t.shape != self.values.shape:
            raise ValueError("t and values must have the same shape")

    @property
    def n(self) -> int:
        return int(self.t.size)


@dataclass(frozen=True)
class RIVSet:
    bw: BeatSeries
    am: BeatSeries
    fm: BeatSeries

    def get(self, name: str) -> BeatSeries:
        try:
            return getattr(self, name)
        except AttributeError as exc:
            raise KeyError(f"unknown RIV {name!r}") from exc


@dataclass(frozen=True)
class UniformSignal:
    """A uniformly sampled signal with its sampling rate."""

    t: np.ndarray
    y: np.ndarray
    fs: float

    def slice(self, t0: float, t1: float) -> UniformSignal:
        mask = (self.t >= t0) & (self.t < t1)
        return UniformSignal(self.t[mask], remove_mean(self.y[mask]), self.fs)


def extract_rivs(ppg: np.ndarray, beats: Beats, max_interval_s: float = MAX_FM_INTERVAL_S) -> RIVSet:
    """Measure BW, AM and FM on ``ppg`` at the detected beat positions.

    Peak-to-peak intervals longer than ``max_interval_s`` are treated as gaps
    (a rejected or missed beat) and contribute no FM sample, so that the FM
    series does not contain spurious spikes spanning two cardiac cycles.
    """
    ppg = np.asarray(ppg, dtype=float)
    if beats.n == 0:
        empty = BeatSeries(np.zeros(0), np.zeros(0))
        return RIVSet(empty, empty, empty)

    peak_amp = ppg[beats.peaks]
    trough_amp = ppg[beats.troughs]
    t_peaks = beats.peak_times

    bw = BeatSeries(t_peaks, 0.5 * (peak_amp + trough_amp))
    am = BeatSeries(t_peaks, peak_amp - trough_amp)
    intervals = np.diff(t_peaks)
    keep = intervals <= max_interval_s
    fm = BeatSeries((0.5 * (t_peaks[1:] + t_peaks[:-1]))[keep], intervals[keep])
    return RIVSet(bw, am, fm)


def respiratory_signal(series: BeatSeries, cfg: PipelineConfig, t_start: float, t_end: float) -> UniformSignal:
    """Resample a beat series to ``cfg.riv_fs`` and band-pass it to the RR band."""
    t, y = resample_uniform(series.t, series.values, cfg.riv_fs, t_start, t_end)
    if series.n < 2:
        return UniformSignal(t, np.zeros_like(t), cfg.riv_fs)
    low, high = cfg.rr_band_hz
    return UniformSignal(t, bandpass(remove_mean(y), cfg.riv_fs, low, high), cfg.riv_fs)


def filtered_ppg_signal(ppg: np.ndarray, fs: float, cfg: PipelineConfig) -> UniformSignal:
    """Direct approach: band-pass the raw PPG to the RR band and downsample.

    This bypasses beat detection entirely and treats baseline wander as the
    respiratory signal; it is the ``filt`` source in the evaluation.
    """
    ppg = np.asarray(ppg, dtype=float)
    low, high = cfg.rr_band_hz
    resp = bandpass(ppg, fs, low, high)
    t_raw = np.arange(ppg.size) / fs
    t, y = resample_uniform(t_raw, resp, cfg.riv_fs, 0.0, ppg.size / fs)
    return UniformSignal(t, y, cfg.riv_fs)
