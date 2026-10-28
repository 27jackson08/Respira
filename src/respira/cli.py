"""Command-line interface: ``respira download | estimate | evaluate | run | inspect | demo``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from respira import __version__
from respira.config import PipelineConfig
from respira.data.bidmc import SUBJECTS, available_subjects, download, load_all, load_recording

LOGGER = logging.getLogger("respira")

DEFAULT_DATA_DIR = Path("data/raw/bidmc_csv")
DEFAULT_RESULTS_DIR = Path("results")
ESTIMATES_FILE = "estimates.csv"
CONFIG_FILE = "config.json"


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig()
    updates = {
        key: getattr(args, key)
        for key in ("window_s", "step_s", "ar_order", "sf_sd_threshold_bpm", "sqi_template_threshold")
        if getattr(args, key, None) is not None
    }
    band = getattr(args, "rr_band_bpm", None)
    if band is not None:
        updates["rr_band_bpm"] = (float(band[0]), float(band[1]))
    return cfg.with_updates(**updates)


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--window", dest="window_s", type=float, help="window length in seconds (default 32)")
    parser.add_argument("--step", dest="step_s", type=float, help="window hop in seconds (default 8)")
    parser.add_argument("--ar-order", dest="ar_order", type=int, help="Burg AR model order (default 8)")
    parser.add_argument(
        "--sf-threshold", dest="sf_sd_threshold_bpm", type=float, help="Smart Fusion SD threshold in bpm (default 4)"
    )
    parser.add_argument(
        "--band",
        dest="rr_band_bpm",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="respiratory band in breaths/min (default 4 60)",
    )
    parser.add_argument(
        "--sqi-threshold",
        dest="sqi_template_threshold",
        type=float,
        help="template SQI threshold for quality gating (default 0.86)",
    )


def _subjects_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subjects", nargs="*", metavar="NN", help="subject ids such as 01 07 53 (default: all)")


def cmd_download(args: argparse.Namespace) -> int:
    written = download(
        args.dest, args.subjects, overwrite=args.overwrite, progress=lambda n: LOGGER.info("downloading %s", n)
    )
    LOGGER.info(
        "%d files written to %s (%d subjects available)", len(written), args.dest, len(available_subjects(args.dest))
    )
    return 0


def _save_config(cfg: PipelineConfig, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / CONFIG_FILE).write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")


def _load_config(results_dir: Path, fallback: PipelineConfig) -> PipelineConfig:
    path = results_dir / CONFIG_FILE
    if not path.is_file():
        return fallback
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
    return PipelineConfig(**raw)


def run_estimate(data_dir: Path, results_dir: Path, cfg: PipelineConfig, subjects: Sequence[str] | None) -> Path:
    from respira.pipeline import process_recordings

    chosen = tuple(subjects) if subjects else available_subjects(data_dir)
    if not chosen:
        raise SystemExit(f"no complete BIDMC subjects found in {data_dir}; run `respira download` first")
    LOGGER.info("estimating RR for %d subjects from %s", len(chosen), data_dir)
    df = process_recordings(load_all(data_dir, chosen), cfg, progress=lambda s: LOGGER.info("subject %s", s))
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / ESTIMATES_FILE
    df.to_csv(out, index=False)
    _save_config(cfg, results_dir)
    LOGGER.info("wrote %d rows to %s", len(df), out)
    return out


def cmd_estimate(args: argparse.Namespace) -> int:
    run_estimate(args.data, args.out, _config_from_args(args), args.subjects)
    return 0


def run_evaluate(results_dir: Path, cfg: PipelineConfig) -> Path:
    from respira import evaluate as ev
    from respira import plotting, report

    estimates_path = results_dir / ESTIMATES_FILE
    if not estimates_path.is_file():
        raise SystemExit(f"{estimates_path} not found; run `respira estimate` first")
    df = pd.read_csv(estimates_path)
    figures = results_dir / "figures"

    by_alg = ev.metrics_by_algorithm(df)
    by_alg_gated = ev.metrics_by_algorithm(df, sqi_threshold=cfg.sqi_template_threshold)
    best = ev.best_algorithm(by_alg)
    best_df = ev.reference_windows(ev.select_algorithm(df, *best))
    by_subject = ev.metrics_by_subject(df, *best)
    sweep = ev.sqi_sweep(df, *best)
    conf_sweep = ev.confidence_sweep(df, *best)
    plaus = ev.plausibility_gate_table(df, *best)
    by_bin = ev.metrics_by_sqi_bin(df, *best)
    comparators = ev.comparator_metrics(df)
    windows = ev.window_level(df)

    by_alg.to_csv(results_dir / "metrics_by_algorithm.csv", index=False)
    by_alg_gated.to_csv(results_dir / "metrics_by_algorithm_sqi_gated.csv", index=False)
    by_subject.to_csv(results_dir / "metrics_by_subject.csv", index=False)
    sweep.to_csv(results_dir / "sqi_sweep.csv", index=False)
    conf_sweep.to_csv(results_dir / "confidence_sweep.csv", index=False)
    plaus.to_csv(results_dir / "plausibility_gate.csv", index=False)
    by_bin.to_csv(results_dir / "metrics_by_sqi_bin.csv", index=False)
    comparators.to_csv(results_dir / "comparators.csv", index=False)

    label = f"{best[0]}:{best[1]}"
    plotting.plot_algorithm_comparison(by_alg, figures / "mae_by_algorithm.png")
    plotting.plot_bland_altman(
        best_df["rr_est"].to_numpy(),
        best_df["rr_ref"].to_numpy(),
        figures / "bland_altman_best.png",
        f"Bland-Altman: {label}",
    )
    plotting.plot_estimate_vs_reference(
        best_df["rr_est"].to_numpy(),
        best_df["rr_ref"].to_numpy(),
        figures / "estimate_vs_reference_best.png",
        f"Estimate vs reference: {label}",
    )
    plotting.plot_quality_tradeoff(
        sweep, conf_sweep, figures / "quality_tradeoff_best.png", f"Quality gating vs accuracy: {label}"
    )
    plotting.plot_per_subject(by_subject, figures / "mae_by_subject_best.png", f"Per-subject MAE: {label}")

    text = report.build_summary(
        cfg=cfg,
        n_subjects=int(windows["subject"].nunique()),
        n_windows=len(windows),
        n_ref_windows=len(ev.reference_windows(windows)),
        by_algorithm=by_alg,
        by_algorithm_gated=by_alg_gated,
        best=best,
        comparators=comparators,
        sweep=sweep,
        conf_sweep=conf_sweep,
        plausibility=plaus,
        by_sqi_bin=by_bin,
        by_subject=by_subject,
    )
    summary = report.write_summary(results_dir / "summary.md", text)
    LOGGER.info("best algorithm %s: MAE %.2f bpm; summary at %s", label, float(by_alg.iloc[0]["mae"]), summary)
    return summary


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = _load_config(args.out, _config_from_args(args))
    run_evaluate(args.out, cfg)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    run_estimate(args.data, args.out, cfg, args.subjects)
    run_evaluate(args.out, cfg)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from respira.plotting import plot_recording_overview

    rec = load_recording(args.data, args.subject)
    out = args.out or Path("results/figures") / f"subject_{args.subject}_window_{args.window_index}.png"
    path = plot_recording_overview(rec, _config_from_args(args), args.window_index, out)
    LOGGER.info("wrote %s", path)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from respira.evaluate import metrics_by_algorithm
    from respira.pipeline import process_recording
    from respira.plotting import plot_recording_overview
    from respira.synth import synthetic_recording

    cfg = _config_from_args(args)
    rec = synthetic_recording(duration_s=args.duration, hr_bpm=args.hr, rr_bpm=args.rr)
    df = process_recording(rec, cfg)
    summary = metrics_by_algorithm(df)
    out = args.out or Path("results/figures/demo_synthetic.png")
    plot_recording_overview(rec, cfg, 0, out)
    print(summary[["algorithm", "mae", "coverage"]].head(10).to_string(index=False))
    LOGGER.info("true RR %.1f bpm; figure at %s", args.rr, out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="respira", description="Respiratory rate from PPG, evaluated on BIDMC.")
    parser.add_argument("--version", action="version", version=f"respira {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download the BIDMC CSV files from PhysioNet")
    p.add_argument("--dest", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--overwrite", action="store_true")
    _subjects_arg(p)
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("estimate", help="run the pipeline and write per-window estimates")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_RESULTS_DIR, help="results directory")
    _subjects_arg(p)
    _add_config_args(p)
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("evaluate", help="compute metrics, figures and summary.md from estimates")
    p.add_argument("--out", type=Path, default=DEFAULT_RESULTS_DIR, help="results directory")
    _add_config_args(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("run", help="estimate then evaluate")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_RESULTS_DIR)
    _subjects_arg(p)
    _add_config_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("inspect", help="plot one window of one subject")
    p.add_argument("subject", choices=SUBJECTS)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--window-index", type=int, default=10)
    p.add_argument("--out", type=Path)
    _add_config_args(p)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("demo", help="run the pipeline on a synthetic PPG with known RR")
    p.add_argument("--rr", type=float, default=15.0, help="true respiratory rate (bpm)")
    p.add_argument("--hr", type=float, default=75.0, help="heart rate (bpm)")
    p.add_argument("--duration", type=float, default=180.0, help="seconds")
    p.add_argument("--out", type=Path)
    _add_config_args(p)
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
