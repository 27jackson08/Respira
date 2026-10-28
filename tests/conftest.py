from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from respira.config import PipelineConfig
from respira.data.bidmc import Recording
from respira.synth import SyntheticPPG, synthetic_ppg, synthetic_recording

FS = 125.0


@pytest.fixture(scope="session")
def synth() -> SyntheticPPG:
    return synthetic_ppg(duration_s=120.0, fs=FS, hr_bpm=72.0, rr_bpm=15.0, seed=1)


@pytest.fixture(scope="session")
def recording() -> Recording:
    return synthetic_recording(subject="01", duration_s=120.0, fs=FS, hr_bpm=72.0, rr_bpm=15.0, seed=1)


@pytest.fixture(scope="session")
def cfg() -> PipelineConfig:
    return PipelineConfig()


def write_bidmc_subject(data_dir: Path, subject: str, rec: Recording) -> None:
    """Write a Recording to disk in the exact BIDMC CSV layout (with ragged breath columns)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    signals = pd.DataFrame(
        {"Time [s]": rec.time, " RESP": rec.resp, " PLETH": rec.ppg, " V": 0.0, " AVR": 0.0, " II": 0.0}
    )
    signals.to_csv(data_dir / f"bidmc_{subject}_Signals.csv", index=False)
    rec.numerics.rename(columns=lambda c: c if c == "Time [s]" else f" {c}").to_csv(
        data_dir / f"bidmc_{subject}_Numerics.csv", index=False
    )
    n = max(rec.breaths_ann1.size, rec.breaths_ann2.size)
    ann1 = np.full(n, np.nan)
    ann2 = np.full(n, np.nan)
    ann1[: rec.breaths_ann1.size] = rec.breaths_ann1
    ann2[: rec.breaths_ann2.size] = rec.breaths_ann2
    lines = ["breaths ann1 [signal sample no], breaths ann2 [signal sample no]"]
    for a, b in zip(ann1, ann2, strict=True):
        left = "" if np.isnan(a) else str(int(a))
        right = "" if np.isnan(b) else str(int(b))
        lines.append(f"{left},{right}")
    (data_dir / f"bidmc_{subject}_Breaths.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (data_dir / f"bidmc_{subject}_Fix.txt").write_text(
        f"bidmc_{subject}\nSignals: RESP; PLETH; V; AVR; II\nSignals sampling frequency: 125 Hz\n"
        "Numerics: HR; PULSE; RESP; SpO2\nNumerics sampling frequency: 1 Hz\nAge: 61\nGender: F\nLocation: micu\n",
        encoding="utf-8",
    )


@pytest.fixture(scope="session")
def bidmc_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two short synthetic subjects in BIDMC format; subject 02 drops the last ann2 breath."""
    root = tmp_path_factory.mktemp("bidmc")
    rec1 = synthetic_recording(subject="01", duration_s=100.0, fs=FS, hr_bpm=72.0, rr_bpm=15.0, seed=1)
    rec2 = synthetic_recording(subject="02", duration_s=100.0, fs=FS, hr_bpm=90.0, rr_bpm=22.0, seed=2)
    rec2 = Recording(**{**rec2.__dict__, "breaths_ann2": rec2.breaths_ann2[:-1]})
    write_bidmc_subject(root, "01", rec1)
    write_bidmc_subject(root, "02", rec2)
    return root
