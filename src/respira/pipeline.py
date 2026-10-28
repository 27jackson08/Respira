"""End-to-end RR estimation for a recording: beats -> RIVs -> windows -> estimates."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from respira.beats import Beats, detect_beats
from respira.config import RIV_NAMES, PipelineConfig
from respira.data.bidmc import Recording
from respira.estimators import Estimate, Estimator, build_estimators
from respira.fusion import median_fusion, smart_fusion, spectral_fusion, temporal_smooth
from respira.preprocess import bandpass
from respira.quality import assess_window
from respira.reference import impedance_spectral_rr, monitor_rr, reference_rr
from respira.riv import UniformSignal, extract_rivs, filtered_ppg_signal, respiratory_signal
from respira.windows import Window, sliding_windows

LOGGER = logging.getLogger(__name__)

FUSION_SOURCES: tuple[str, ...] = ("sf", "sf_ts", "median", "spectral")
WINDOW_COLUMNS: tuple[str, ...] = (
    "subject",
    "window",
    "t0",
    "t1",
    "sqi",
    "hr_bpm",
    "n_beats",
    "plausible",
    "rr_ref",
    "rr_ann1",
    "rr_ann2",
    "ref_valid",
    "rr_monitor",
    "rr_impedance",
)


def respiratory_sources(rec: Recording, cfg: PipelineConfig) -> tuple[Beats, np.ndarray, dict[str, UniformSignal]]:
    """Detect beats and build the four uniformly sampled respiratory signals."""
    ppg_light = bandpass(rec.ppg, rec.fs, *cfg.ppg_band_hz)
    beats = detect_beats(
        rec.ppg,
        rec.fs,
        beat_band_hz=cfg.beat_band_hz,
        min_hr_bpm=cfg.min_hr_bpm,
        max_hr_bpm=cfg.max_hr_bpm,
        measurement_signal=ppg_light,
    )
    rivs = extract_rivs(ppg_light, beats)
    sources = {name: respiratory_signal(rivs.get(name), cfg, 0.0, rec.duration_s) for name in RIV_NAMES}
    sources["filt"] = filtered_ppg_signal(rec.ppg, rec.fs, cfg)
    return beats, ppg_light, sources


def _window_context(
    rec: Recording, cfg: PipelineConfig, beats: Beats, ppg_light: np.ndarray, w: Window
) -> dict[str, object]:
    quality = assess_window(ppg_light, beats.in_window(w.t0, w.t1), rec.fs, cfg.min_hr_bpm, cfg.max_hr_bpm)
    ref = reference_rr(
        rec.breaths_ann1 / rec.fs,
        rec.breaths_ann2 / rec.fs,
        w.t0,
        w.t1,
        min_breaths=cfg.ref_min_breaths,
        max_disagreement_bpm=cfg.ref_max_disagreement_bpm,
    )
    return {
        "subject": rec.subject,
        "window": w.index,
        "t0": w.t0,
        "t1": w.t1,
        "sqi": quality.sqi,
        "hr_bpm": quality.hr_bpm,
        "n_beats": quality.n_beats,
        "plausible": quality.is_plausible,
        "rr_ref": ref.rr_bpm,
        "rr_ann1": ref.rr_ann1,
        "rr_ann2": ref.rr_ann2,
        "ref_valid": ref.is_valid,
        "rr_monitor": monitor_rr(rec.numerics, w.t0, w.t1),
        "rr_impedance": impedance_spectral_rr(rec.resp, rec.fs, w.t0, w.t1, cfg.riv_fs, cfg.rr_band_hz),
    }


def _row(ctx: Mapping[str, object], source: str, method: str, est: Estimate) -> dict[str, object]:
    return {**ctx, "source": source, "method": method, "rr_est": est.rr_bpm, "confidence": est.confidence}


def _estimate_window(
    ctx: Mapping[str, object],
    segments: Mapping[str, np.ndarray],
    estimators: Mapping[str, Estimator],
    cfg: PipelineConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    single: dict[tuple[str, str], Estimate] = {}
    for source, y in segments.items():
        for method, estimator in estimators.items():
            est = estimator(y, cfg.riv_fs, cfg.rr_band_hz)
            single[(source, method)] = est
            rows.append(_row(ctx, source, method, est))
    for method in estimators:
        triple = {name: single[(name, method)].rr_bpm for name in RIV_NAMES}
        rows.append(_row(ctx, "sf", method, smart_fusion(triple, cfg.sf_sd_threshold_bpm)))
        rows.append(_row(ctx, "median", method, median_fusion(triple)))
    spectral = spectral_fusion([segments[name] for name in RIV_NAMES], cfg.riv_fs, cfg.rr_band_hz)
    rows.append(_row(ctx, "spectral", "periodogram", spectral))
    return rows


def _add_temporal_smoothing(df: pd.DataFrame, source: str = "sf", new_source: str = "sf_ts") -> pd.DataFrame:
    """Append rows with exponentially smoothed estimates for one fusion source."""
    base = df[df["source"] == source]
    if base.empty:
        return df
    smoothed = []
    for _, group in base.groupby("method", sort=False):
        ordered = group.sort_values("window")
        smoothed.append(ordered.assign(source=new_source, rr_est=temporal_smooth(ordered["rr_est"].to_numpy())))
    return pd.concat([df, *smoothed], ignore_index=True)


def process_recording(
    rec: Recording, cfg: PipelineConfig, estimators: Mapping[str, Estimator] | None = None
) -> pd.DataFrame:
    """One row per (window, source, method) with the estimate and window-level context."""
    estimators = dict(estimators) if estimators is not None else build_estimators(cfg.ar_order)
    beats, ppg_light, sources = respiratory_sources(rec, cfg)
    LOGGER.debug("subject %s: %d beats", rec.subject, beats.n)

    rows: list[dict[str, object]] = []
    for w in sliding_windows(rec.duration_s, cfg.window_s, cfg.step_s):
        ctx = _window_context(rec, cfg, beats, ppg_light, w)
        segments = {name: sig.slice(w.t0, w.t1).y for name, sig in sources.items()}
        rows.extend(_estimate_window(ctx, segments, estimators, cfg))

    df = pd.DataFrame(rows, columns=[*WINDOW_COLUMNS, "source", "method", "rr_est", "confidence"])
    df = _add_temporal_smoothing(df)
    return df.assign(algorithm=df["source"] + ":" + df["method"])


def process_recordings(
    recordings: Iterable[Recording],
    cfg: PipelineConfig,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Run :func:`process_recording` over many recordings and concatenate."""
    frames = []
    for rec in recordings:
        if progress:
            progress(rec.subject)
        frames.append(process_recording(rec, cfg))
    if not frames:
        return pd.DataFrame(columns=[*WINDOW_COLUMNS, "source", "method", "rr_est", "confidence", "algorithm"])
    return pd.concat(frames, ignore_index=True)
