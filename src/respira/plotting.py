"""Figures for the evaluation report (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from respira.config import RIV_NAMES, PipelineConfig
from respira.data.bidmc import Recording
from respira.pipeline import respiratory_sources
from respira.reference import reference_rr
from respira.windows import sliding_windows

# Categorical palette in fixed slot order (validated for colour-vision deficiency).
SOURCE_COLORS: dict[str, str] = {
    "bw": "#2a78d6",
    "am": "#eb6834",
    "fm": "#1baf7a",
    "filt": "#eda100",
    "sf": "#e87ba4",
    "sf_ts": "#008300",
    "median": "#6250d6",
    "spectral": "#e34948",
}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
ACCENT = "#2a78d6"
REFERENCE = "#e34948"

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT_PRIMARY,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "font.size": 9,
    "legend.frameon": False,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
}


def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_algorithm_comparison(summary: pd.DataFrame, path: Path, top_n: int = 20) -> Path:
    """Horizontal bars of MAE per algorithm, coloured by respiratory source."""
    data = summary.sort_values("mae").head(top_n).iloc[::-1]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 0.32 * len(data) + 1.2))
        colors = [SOURCE_COLORS.get(s, TEXT_SECONDARY) for s in data["source"]]
        bars = ax.barh(data["algorithm"], data["mae"], color=colors, height=0.6)
        for bar, (_, row) in zip(bars, data.iterrows(), strict=True):
            ax.text(
                bar.get_width() + 0.05,
                bar.get_y() + bar.get_height() / 2,
                f"{row['mae']:.2f}  ({row['coverage']:.0%} cov.)",
                va="center",
                ha="left",
                fontsize=8,
                color=TEXT_SECONDARY,
            )
        ax.set_xlabel("Mean absolute error (breaths/min)")
        ax.set_title("Respiratory rate error by algorithm (source:method)")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, float(data["mae"].max()) * 1.35)
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=SOURCE_COLORS[s]) for s in SOURCE_COLORS if s in set(data["source"])
        ]
        labels = [s for s in SOURCE_COLORS if s in set(data["source"])]
        ax.legend(handles, labels, title="source", loc="lower right", fontsize=8, title_fontsize=8)
        return _save(fig, path)


def plot_bland_altman(est: np.ndarray, ref: np.ndarray, path: Path, title: str) -> Path:
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    ok = np.isfinite(est) & np.isfinite(ref)
    diff, mean = est[ok] - ref[ok], 0.5 * (est[ok] + ref[ok])
    bias, sd = float(np.mean(diff)), float(np.std(diff, ddof=1))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.scatter(mean, diff, s=10, alpha=0.35, color=ACCENT, edgecolors="none")
        ax.axhline(bias, color=TEXT_PRIMARY, linewidth=1.2)
        for level in (bias - 1.96 * sd, bias + 1.96 * sd):
            ax.axhline(level, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
        ax.text(ax.get_xlim()[1], bias, f" bias {bias:+.1f}", va="center", fontsize=8, color=TEXT_PRIMARY)
        ax.text(
            ax.get_xlim()[1],
            bias + 1.96 * sd,
            f" +1.96 SD {bias + 1.96 * sd:+.1f}",
            va="center",
            fontsize=8,
            color=TEXT_SECONDARY,
        )
        ax.text(
            ax.get_xlim()[1],
            bias - 1.96 * sd,
            f" -1.96 SD {bias - 1.96 * sd:+.1f}",
            va="center",
            fontsize=8,
            color=TEXT_SECONDARY,
        )
        ax.set_xlabel("Mean of estimate and reference (breaths/min)")
        ax.set_ylabel("Estimate - reference (breaths/min)")
        ax.set_title(title)
        return _save(fig, path)


def plot_estimate_vs_reference(est: np.ndarray, ref: np.ndarray, path: Path, title: str) -> Path:
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    ok = np.isfinite(est) & np.isfinite(ref)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5, 5))
        lo, hi = 0.0, float(max(np.max(ref[ok]), np.max(est[ok])) * 1.05) if ok.any() else 40.0
        ax.plot([lo, hi], [lo, hi], color=TEXT_SECONDARY, linewidth=1, linestyle="--")
        ax.fill_between([lo, hi], [lo - 2, hi - 2], [lo + 2, hi + 2], color=GRID, alpha=0.8, linewidth=0)
        ax.scatter(ref[ok], est[ok], s=10, alpha=0.35, color=ACCENT, edgecolors="none")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Reference RR (breaths/min)")
        ax.set_ylabel("Estimated RR (breaths/min)")
        ax.set_title(f"{title}\nshaded band: ±2 breaths/min")
        return _save(fig, path)


def plot_quality_tradeoff(sqi_sweep: pd.DataFrame, conf_sweep: pd.DataFrame, path: Path, title: str) -> Path:
    """MAE against coverage when withholding low-quality windows, for two gating signals."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        for frame, label, color in (
            (sqi_sweep, "template SQI (pulse morphology)", ACCENT),
            (conf_sweep, "estimator confidence (respiratory signal)", SOURCE_COLORS["am"]),
        ):
            ok = np.isfinite(frame["mae"]) & (frame["coverage"] > 0)
            ax.plot(
                100 * frame.loc[ok, "coverage"],
                frame.loc[ok, "mae"],
                color=color,
                linewidth=2,
                marker="o",
                markersize=4,
                label=label,
            )
        best_full = sqi_sweep.iloc[0]
        ax.annotate(
            f"no gating: MAE {best_full['mae']:.2f}",
            (100 * best_full["coverage"], best_full["mae"]),
            textcoords="offset points",
            xytext=(12, -26),
            ha="left",
            fontsize=8,
            color=TEXT_SECONDARY,
        )
        ax.set_xlabel("Coverage (% of reference windows reported)")
        ax.set_ylabel("MAE (breaths/min)")
        ax.set_title(title)
        ax.legend(loc="lower left", fontsize=8)
        ax.invert_xaxis()
        return _save(fig, path)


def plot_per_subject(by_subject: pd.DataFrame, path: Path, title: str) -> Path:
    data = by_subject.sort_values("mae", ascending=False)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.bar(data["subject"], data["mae"], color=ACCENT, width=0.7)
        ax.axhline(float(data["mae"].median()), color=TEXT_PRIMARY, linewidth=1, linestyle="--")
        ax.text(
            len(data) - 0.5,
            float(data["mae"].median()),
            f" median {data['mae'].median():.1f}",
            fontsize=8,
            va="bottom",
            ha="right",
            color=TEXT_PRIMARY,
        )
        ax.set_ylabel("MAE (breaths/min)")
        ax.set_xlabel("BIDMC subject")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=7, rotation=90)
        return _save(fig, path)


def _normalised_spectrum(y: np.ndarray, fs: float, band_hz: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    if y.size < 16 or np.ptp(y) == 0:
        return np.zeros(1), np.zeros(1)
    f, p = signal.periodogram(y - y.mean(), fs=fs, window="hann", nfft=4096)
    mask = (f >= band_hz[0]) & (f <= band_hz[1])
    p = p[mask]
    return 60 * f[mask], p / p.max() if p.max() > 0 else p


def plot_recording_overview(rec: Recording, cfg: PipelineConfig, window_index: int, path: Path) -> Path:
    """Four panels for one window: PPG with beats, impedance with annotated breaths,
    the four respiratory signals, and their spectra against the reference RR."""
    windows = sliding_windows(rec.duration_s, cfg.window_s, cfg.step_s)
    if not windows:
        raise ValueError("recording shorter than one window")
    w = windows[min(window_index, len(windows) - 1)]
    beats, ppg_light, sources = respiratory_sources(rec, cfg)
    wb = beats.in_window(w.t0, w.t1)
    ref = reference_rr(
        rec.breaths_ann1 / rec.fs,
        rec.breaths_ann2 / rec.fs,
        w.t0,
        w.t1,
        min_breaths=cfg.ref_min_breaths,
        max_disagreement_bpm=cfg.ref_max_disagreement_bpm,
    )
    i0, i1 = int(w.t0 * rec.fs), int(w.t1 * rec.fs)
    t = rec.time[i0:i1]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(4, 1, figsize=(10, 10.5), gridspec_kw={"height_ratios": [1.2, 1, 1.4, 1.2]})
        ax = axes[0]
        ax.plot(t, ppg_light[i0:i1], color=ACCENT, linewidth=1)
        ax.scatter(
            wb.peaks / rec.fs, ppg_light[wb.peaks], s=14, color=SOURCE_COLORS["am"], zorder=3, label="systolic peak"
        )
        ax.scatter(
            wb.troughs / rec.fs, ppg_light[wb.troughs], s=14, color=SOURCE_COLORS["fm"], zorder=3, label="trough"
        )
        ax.set_title(
            f"Subject {rec.subject}, window {w.index} ({w.t0:.0f}-{w.t1:.0f} s): filtered PPG with detected beats"
        )
        ax.set_ylabel("PPG (a.u.)")
        ax.legend(loc="upper right", fontsize=8, ncol=2)

        ax = axes[1]
        ax.plot(t, rec.resp[i0:i1], color=TEXT_SECONDARY, linewidth=1)
        ann = rec.breaths_ann1[(rec.breaths_ann1 >= i0) & (rec.breaths_ann1 < i1)]
        for s in ann:
            ax.axvline(s / rec.fs, color=REFERENCE, linewidth=0.8, alpha=0.7)
        ax.set_title(f"Impedance respiration with annotator-1 breaths (reference RR {ref.rr_bpm:.1f} bpm)")
        ax.set_ylabel("Impedance (a.u.)")

        ax = axes[2]
        offset = 0.0
        for name in (*RIV_NAMES, "filt"):
            seg = sources[name].slice(w.t0, w.t1)
            scale = np.ptp(seg.y) or 1.0
            ax.plot(seg.t, seg.y / scale + offset, color=SOURCE_COLORS[name], linewidth=1.4, label=name)
            ax.text(w.t1 + 0.3, offset, name, color=SOURCE_COLORS[name], va="center", fontsize=8)
            offset -= 1.3
        ax.set_title("Respiratory-induced variations (normalised, band-passed to 4-60 breaths/min)")
        ax.set_yticks([])
        ax.set_xlabel("Time (s)")
        ax.grid(axis="y", visible=False)

        ax = axes[3]
        for name in (*RIV_NAMES, "filt"):
            seg = sources[name].slice(w.t0, w.t1)
            f_bpm, p = _normalised_spectrum(seg.y, seg.fs, cfg.rr_band_hz)
            ax.plot(f_bpm, p, color=SOURCE_COLORS[name], linewidth=1.4, label=name)
        if np.isfinite(ref.rr_bpm):
            ax.axvline(ref.rr_bpm, color=REFERENCE, linewidth=1.2, linestyle="--", label="reference RR")
        ax.set_xlabel("Frequency (breaths/min)")
        ax.set_ylabel("Normalised power")
        ax.set_title("Spectra of the respiratory signals")
        ax.legend(loc="upper right", fontsize=8, ncol=5)
        fig.tight_layout()
        return _save(fig, path)
