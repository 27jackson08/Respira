"""Markdown summary of an evaluation run."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from respira.config import PipelineConfig

SOURCE_LABELS = {
    "bw": "Baseline wander (BW)",
    "am": "Amplitude modulation (AM)",
    "fm": "Frequency modulation (FM)",
    "filt": "Direct PPG band-pass",
    "sf": "Smart Fusion of BW/AM/FM",
    "sf_ts": "Smart Fusion + temporal smoothing",
    "median": "Median of BW/AM/FM",
    "spectral": "Averaged spectra of BW/AM/FM",
}

METRIC_COLUMNS = ["algorithm", "mae", "median_ae", "rmse", "bias", "loa_low", "loa_high", "cp2", "coverage", "n_est"]


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    cols = columns or list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(" --- " for _ in cols) + "|"
    rows = ["| " + " | ".join(_fmt(v) for v in record) + " |" for record in df[cols].itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def _config_table(cfg: PipelineConfig) -> str:
    items = pd.DataFrame([{"parameter": k, "value": str(v)} for k, v in asdict(cfg).items()])
    return markdown_table(items)


def build_summary(
    *,
    cfg: PipelineConfig,
    n_subjects: int,
    n_windows: int,
    n_ref_windows: int,
    by_algorithm: pd.DataFrame,
    by_algorithm_gated: pd.DataFrame,
    best: tuple[str, str],
    comparators: pd.DataFrame,
    sweep: pd.DataFrame,
    conf_sweep: pd.DataFrame,
    plausibility: pd.DataFrame,
    by_sqi_bin: pd.DataFrame,
    by_subject: pd.DataFrame,
) -> str:
    """Compose the full markdown report."""
    best_label = f"{best[0]}:{best[1]}"
    best_row = by_algorithm[by_algorithm["algorithm"] == best_label].iloc[0]
    worst_subjects = by_subject.sort_values("mae", ascending=False).head(5)
    best_subjects = by_subject.sort_values("mae").head(5)
    parts = [
        "# Respira evaluation summary",
        "",
        f"Dataset: BIDMC PPG and Respiration Dataset, {n_subjects} recordings, "
        f"{cfg.window_s:.0f} s windows every {cfg.step_s:.0f} s.",
        "",
        f"- Windows analysed: {n_windows}",
        f"- Windows with a valid reference (both annotators agree within "
        f"{cfg.ref_max_disagreement_bpm:g} bpm): {n_ref_windows}",
        f"- Best algorithm (lowest MAE with >= 50 % coverage): `{best_label}` "
        f"({SOURCE_LABELS.get(best[0], best[0])} + {best[1]})",
        "",
        "## Best algorithm",
        "",
        markdown_table(pd.DataFrame([best_row]), METRIC_COLUMNS),
        "",
        "## All algorithms (no quality gating)",
        "",
        "`source:method`. Sources are the respiratory signal; methods are the RR estimator. "
        "MAE/bias/LoA in breaths per minute; coverage is the fraction of reference windows "
        "for which the algorithm produced an estimate.",
        "",
        markdown_table(by_algorithm, METRIC_COLUMNS),
        "",
        f"## All algorithms with quality gating (template SQI >= {cfg.sqi_template_threshold})",
        "",
        markdown_table(by_algorithm_gated, METRIC_COLUMNS),
        "",
        "## Non-PPG comparators",
        "",
        "The bedside monitor's own impedance-derived RR numeric and a spectral estimate "
        "from the raw impedance waveform, scored against the same manual-annotation reference.",
        "",
        markdown_table(
            comparators, ["comparator", "mae", "median_ae", "bias", "loa_low", "loa_high", "cp2", "coverage", "n_est"]
        ),
        "",
        f"## Signal quality vs accuracy for `{best_label}`",
        "",
        "### MAE by SQI bin",
        "",
        markdown_table(by_sqi_bin, ["sqi_bin", "n_est", "mae", "median_ae", "cp2"]),
        "",
        "### Template-SQI threshold sweep (report only windows with SQI >= threshold)",
        "",
        markdown_table(sweep, ["sqi_threshold", "n_windows", "coverage", "mae", "median_ae", "cp2"]),
        "",
        "### Estimator-confidence sweep (withhold the least confident fraction of windows)",
        "",
        markdown_table(
            conf_sweep, ["drop_fraction", "confidence_cutoff", "n_windows", "coverage", "mae", "median_ae", "cp2"]
        ),
        "",
        "### Beat-interval plausibility gate (HR 40-180, no gap > 3 s, max/min interval <= 2.2)",
        "",
        markdown_table(plausibility, ["gate", "n_windows", "coverage", "mae", "median_ae", "cp2"]),
        "",
        f"## Per-subject spread for `{best_label}`",
        "",
        "Lowest MAE:",
        "",
        markdown_table(best_subjects, ["subject", "mae", "coverage", "median_sqi", "n_est"]),
        "",
        "Highest MAE:",
        "",
        markdown_table(worst_subjects, ["subject", "mae", "coverage", "median_sqi", "n_est"]),
        "",
        "## Configuration",
        "",
        _config_table(cfg),
        "",
    ]
    return "\n".join(parts)


def write_summary(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
