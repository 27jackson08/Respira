"""Agreement metrics between estimated and reference respiratory rate."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

CP_THRESHOLD_BPM = 2.0
DEFAULT_SQI_THRESHOLDS: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99)
"""Template-SQI acceptance thresholds; dense near 1 because ICU PPG pulses are mostly regular."""

DEFAULT_CONFIDENCE_QUANTILES: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
"""Fractions of windows dropped when gating on the estimator's own confidence."""


@dataclass(frozen=True)
class Metrics:
    n_ref: int
    n_est: int
    coverage: float
    mae: float
    median_ae: float
    rmse: float
    bias: float
    sd: float
    loa_low: float
    loa_high: float
    cp2: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_metrics(est: np.ndarray, ref: np.ndarray) -> Metrics:
    """Metrics over windows where the reference is defined; ``est`` may contain NaN.

    * ``coverage``: fraction of reference windows with an estimate.
    * ``cp2``: fraction of estimated windows within 2 bpm of the reference.
    * ``loa_*``: 95 % limits of agreement, ``bias ± 1.96 sd``.
    """
    est = np.asarray(est, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if est.shape != ref.shape:
        raise ValueError("est and ref must have the same shape")
    has_ref = np.isfinite(ref)
    both = has_ref & np.isfinite(est)
    n_ref, n_est = int(has_ref.sum()), int(both.sum())
    if n_est == 0:
        nan = float("nan")
        return Metrics(n_ref, 0, 0.0, nan, nan, nan, nan, nan, nan, nan, nan)
    err = est[both] - ref[both]
    abs_err = np.abs(err)
    bias, sd = float(np.mean(err)), float(np.std(err, ddof=1)) if n_est > 1 else 0.0
    return Metrics(
        n_ref=n_ref,
        n_est=n_est,
        coverage=n_est / n_ref if n_ref else 0.0,
        mae=float(np.mean(abs_err)),
        median_ae=float(np.median(abs_err)),
        rmse=float(np.sqrt(np.mean(err**2))),
        bias=bias,
        sd=sd,
        loa_low=bias - 1.96 * sd,
        loa_high=bias + 1.96 * sd,
        cp2=float(np.mean(abs_err <= CP_THRESHOLD_BPM)),
    )


def reference_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose reference RR is valid."""
    return df[df["ref_valid"].astype(bool) & np.isfinite(df["rr_ref"])]


def effective_sqi(df: pd.DataFrame) -> pd.Series:
    """Template SQI with undefined (too few beats) or negative values mapped to 0.

    A threshold of 0 therefore means "every window", and such windows land in
    the lowest SQI bin instead of silently disappearing.
    """
    return df["sqi"].astype(float).fillna(0.0).clip(lower=0.0)


def quality_gate(df: pd.DataFrame, sqi_threshold: float, *, require_plausible: bool = True) -> pd.DataFrame:
    """Rows from windows with template SQI >= threshold and, optionally, plausible beat intervals."""
    ok = effective_sqi(df) >= sqi_threshold
    if require_plausible:
        ok = ok & df["plausible"].astype(bool)
    return df[ok]


def window_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (subject, window) with only window-level columns."""
    cols = [c for c in df.columns if c not in ("source", "method", "rr_est", "confidence", "algorithm")]
    return df.drop_duplicates(subset=["subject", "window"])[cols].reset_index(drop=True)


def metrics_by_algorithm(df: pd.DataFrame, sqi_threshold: float | None = None) -> pd.DataFrame:
    """One row per algorithm, sorted by MAE.

    With ``sqi_threshold`` the error metrics are computed on windows passing the
    quality gate, while ``coverage`` stays relative to *all* reference windows
    so that gated and ungated rows are comparable.
    """
    base = reference_windows(df)
    records = []
    for (source, method), group in base.groupby(["source", "method"], sort=False):
        scored = group if sqi_threshold is None else quality_gate(group, sqi_threshold)
        m = compute_metrics(scored["rr_est"].to_numpy(), scored["rr_ref"].to_numpy())
        m = replace(m, n_ref=len(group), coverage=m.n_est / len(group) if len(group) else 0.0)
        records.append({"source": source, "method": method, "algorithm": f"{source}:{method}", **m.as_dict()})
    out = pd.DataFrame(records)
    return (
        out.sort_values(["mae", "coverage"], ascending=[True, False]).reset_index(drop=True) if not out.empty else out
    )


def select_algorithm(df: pd.DataFrame, source: str, method: str) -> pd.DataFrame:
    return df[(df["source"] == source) & (df["method"] == method)]


def metrics_by_subject(df: pd.DataFrame, source: str, method: str) -> pd.DataFrame:
    base = reference_windows(select_algorithm(df, source, method))
    records = []
    for subject, group in base.groupby("subject", sort=True):
        m = compute_metrics(group["rr_est"].to_numpy(), group["rr_ref"].to_numpy())
        records.append({"subject": subject, "median_sqi": float(np.nanmedian(group["sqi"])), **m.as_dict()})
    return pd.DataFrame(records)


def _gate_record(gated: pd.DataFrame, n_total: int) -> dict[str, float | int]:
    m = compute_metrics(gated["rr_est"].to_numpy(), gated["rr_ref"].to_numpy())
    return {
        "n_windows": len(gated),
        "coverage": (m.n_est / n_total) if n_total else 0.0,
        "mae": m.mae,
        "median_ae": m.median_ae,
        "cp2": m.cp2,
    }


def sqi_sweep(
    df: pd.DataFrame, source: str, method: str, thresholds: Sequence[float] = DEFAULT_SQI_THRESHOLDS
) -> pd.DataFrame:
    """MAE and coverage when only windows with template SQI >= threshold are reported.

    Coverage is relative to *all* reference windows, so it captures both the SQI
    rejection and any windows where the estimator itself abstains. The beat
    plausibility flag is reported separately by :func:`plausibility_gate_table`.
    """
    base = reference_windows(select_algorithm(df, source, method))
    n_total = len(base)
    return pd.DataFrame(
        [
            {"sqi_threshold": float(thr), **_gate_record(quality_gate(base, thr, require_plausible=False), n_total)}
            for thr in thresholds
        ]
    )


def confidence_sweep(
    df: pd.DataFrame, source: str, method: str, drop_fractions: Sequence[float] = DEFAULT_CONFIDENCE_QUANTILES
) -> pd.DataFrame:
    """MAE and coverage when the least-confident fraction of estimates is withheld.

    Uses the estimator's own confidence score (spectral peak purity, AR pole
    magnitude, breath-interval regularity), which reflects how respiratory the
    signal is rather than how clean the cardiac pulses are.
    """
    base = reference_windows(select_algorithm(df, source, method))
    n_total = len(base)
    conf = base["confidence"].to_numpy(dtype=float)
    records = []
    for frac in drop_fractions:
        cutoff = float(np.nanquantile(conf, frac)) if np.isfinite(conf).any() else float("nan")
        gated = base[conf >= cutoff] if np.isfinite(cutoff) else base.iloc[0:0]
        records.append({"drop_fraction": float(frac), "confidence_cutoff": cutoff, **_gate_record(gated, n_total)})
    return pd.DataFrame(records)


def plausibility_gate_table(df: pd.DataFrame, source: str, method: str) -> pd.DataFrame:
    """Accuracy with and without the beat-interval plausibility gate."""
    base = reference_windows(select_algorithm(df, source, method))
    n_total = len(base)
    plausible = base["plausible"].astype(bool)
    return pd.DataFrame(
        [
            {"gate": "all windows", **_gate_record(base, n_total)},
            {"gate": "plausible beats only", **_gate_record(base[plausible], n_total)},
            {"gate": "implausible beats only", **_gate_record(base[~plausible], n_total)},
        ]
    )


def metrics_by_sqi_bin(
    df: pd.DataFrame, source: str, method: str, edges: Sequence[float] = (0.0, 0.5, 0.7, 0.86, 0.95, 1.0001)
) -> pd.DataFrame:
    base = reference_windows(select_algorithm(df, source, method))
    labels = [f"[{lo:.2f}, {hi:.2f})" for lo, hi in itertools.pairwise(edges)]
    binned = pd.cut(effective_sqi(base), bins=list(edges), labels=labels, right=False, include_lowest=True)
    records = []
    for label in labels:
        group = base[binned == label]
        m = compute_metrics(group["rr_est"].to_numpy(), group["rr_ref"].to_numpy())
        records.append({"sqi_bin": label, **m.as_dict()})
    return pd.DataFrame(records)


def comparator_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Non-PPG comparators against the same reference: the monitor's RR numeric and
    a spectral estimate from the raw impedance waveform."""
    win = reference_windows(window_level(df))
    records = []
    for name, column in (("monitor_numeric", "rr_monitor"), ("impedance_spectral", "rr_impedance")):
        m = compute_metrics(win[column].to_numpy(), win["rr_ref"].to_numpy())
        records.append({"comparator": name, **m.as_dict()})
    return pd.DataFrame(records)


def best_algorithm(summary: pd.DataFrame, min_coverage: float = 0.5) -> tuple[str, str]:
    """Lowest-MAE algorithm with at least ``min_coverage`` coverage (falls back to lowest MAE)."""
    if summary.empty:
        raise ValueError("no algorithms to choose from")
    eligible = summary[summary["coverage"] >= min_coverage]
    row = (eligible if not eligible.empty else summary).sort_values("mae").iloc[0]
    return str(row["source"]), str(row["method"])
