import pandas as pd

from respira import evaluate as ev
from respira import report
from respira.config import PipelineConfig
from respira.pipeline import process_recording


def test_markdown_table_formats_floats():
    df = pd.DataFrame({"a": [1.23456, 2.0], "b": ["x", "y"]})
    text = report.markdown_table(df)
    assert text.splitlines()[0] == "| a | b |"
    assert "| 1.23 | x |" in text


def test_build_summary_contains_sections(recording, tmp_path):
    cfg = PipelineConfig()
    df = process_recording(recording, cfg)
    by_alg = ev.metrics_by_algorithm(df)
    best = ev.best_algorithm(by_alg)
    text = report.build_summary(
        cfg=cfg,
        n_subjects=1,
        n_windows=12,
        n_ref_windows=12,
        by_algorithm=by_alg,
        by_algorithm_gated=ev.metrics_by_algorithm(df, sqi_threshold=0.86),
        best=best,
        comparators=ev.comparator_metrics(df),
        sweep=ev.sqi_sweep(df, *best),
        conf_sweep=ev.confidence_sweep(df, *best),
        plausibility=ev.plausibility_gate_table(df, *best),
        by_sqi_bin=ev.metrics_by_sqi_bin(df, *best),
        by_subject=ev.metrics_by_subject(df, *best),
    )
    for heading in ("# Respira evaluation summary", "## Best algorithm", "## Non-PPG comparators", "## Configuration"):
        assert heading in text
    out = report.write_summary(tmp_path / "nested" / "summary.md", text)
    assert out.read_text(encoding="utf-8") == text
