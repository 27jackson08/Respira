"""Loader and downloader for the BIDMC PPG and Respiration Dataset (PhysioNet).

Pimentel, M.A.F. et al. (2017) *Toward a Robust Estimation of Respiratory Rate
From Pulse Oximeters*, IEEE TBME 64(8). https://physionet.org/content/bidmc/1.0.0/

Each subject ``NN`` (01-53) has four CSV/TXT files:

* ``bidmc_NN_Signals.csv``  -- 125 Hz waveforms: ``Time [s], RESP, PLETH, <ECG leads>``
* ``bidmc_NN_Numerics.csv`` -- 1 Hz monitor numerics: ``Time [s], HR, PULSE, RESP, SpO2``
* ``bidmc_NN_Breaths.csv``  -- breath annotations as signal sample numbers, two annotators
* ``bidmc_NN_Fix.txt``      -- fixed metadata (age, gender, location, ...)
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from respira.config import BIDMC_FS_HZ

LOGGER = logging.getLogger(__name__)

BASE_URL = "https://physionet.org/files/bidmc/1.0.0/bidmc_csv/"
SUBJECTS: tuple[str, ...] = tuple(f"{i:02d}" for i in range(1, 54))
FILE_KINDS: tuple[tuple[str, str], ...] = (
    ("Signals", "csv"),
    ("Numerics", "csv"),
    ("Breaths", "csv"),
    ("Fix", "txt"),
)
ECG_LEAD_PREFERENCE: tuple[str, ...] = ("II", "I", "V", "AVR", "MCL", "III")
DOWNLOAD_RETRIES = 3


@dataclass(frozen=True)
class Recording:
    subject: str
    fs: float
    time: np.ndarray
    ppg: np.ndarray
    resp: np.ndarray
    ecg: np.ndarray | None
    numerics: pd.DataFrame
    breaths_ann1: np.ndarray
    breaths_ann2: np.ndarray
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.ppg.size / self.fs


def file_name(subject: str, kind: str, ext: str) -> str:
    return f"bidmc_{subject}_{kind}.{ext}"


def expected_files(subject: str) -> tuple[str, ...]:
    return tuple(file_name(subject, kind, ext) for kind, ext in FILE_KINDS)


def is_subject_complete(data_dir: Path, subject: str) -> bool:
    return all((data_dir / name).is_file() for name in expected_files(subject))


def available_subjects(data_dir: Path) -> tuple[str, ...]:
    return tuple(s for s in SUBJECTS if is_subject_complete(data_dir, s))


def _fetch(url: str, dest: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(response.read())
                tmp.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            LOGGER.warning("attempt %d/%d failed for %s: %s", attempt, DOWNLOAD_RETRIES, url, exc)
    raise RuntimeError(f"could not download {url}") from last_error


def download(
    dest: Path,
    subjects: Iterable[str] | None = None,
    *,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, ...]:
    """Download the CSV release of BIDMC into ``dest``; returns the written paths."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for subject in subjects or SUBJECTS:
        if subject not in SUBJECTS:
            raise ValueError(f"unknown BIDMC subject {subject!r}; expected 01..53")
        for name in expected_files(subject):
            target = dest / name
            if target.is_file() and not overwrite:
                continue
            if progress:
                progress(name)
            _fetch(BASE_URL + name, target)
            written.append(target)
    return tuple(written)


def parse_fix(text: str) -> dict[str, str]:
    """Parse the ``Key: value`` lines of a ``Fix.txt`` file."""
    meta: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta


MISSING_MARKERS: tuple[str, ...] = ("-", "--", "")
"""Placeholders PhysioNet uses for missing monitor numerics."""


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True, na_values=list(MISSING_MARKERS))
    return df.rename(columns=lambda c: str(c).strip())


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce every column to float; unparseable cells become NaN."""
    return df.apply(pd.to_numeric, errors="coerce").astype(float)


def _signal_column(signals: pd.DataFrame, name: str) -> np.ndarray:
    """A waveform column as float with gaps linearly interpolated (edges held)."""
    values = pd.to_numeric(signals[name], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    if finite.all():
        return values
    if not finite.any():
        return np.zeros_like(values)
    idx = np.arange(values.size)
    return np.interp(idx, idx[finite], values[finite])


def _pick_ecg(signals: pd.DataFrame) -> np.ndarray | None:
    for lead in ECG_LEAD_PREFERENCE:
        if lead in signals.columns:
            return _signal_column(signals, lead)
    return None


def _breath_column(breaths: pd.DataFrame, index: int) -> np.ndarray:
    if breaths.shape[1] <= index:
        return np.zeros(0, dtype=int)
    col = pd.to_numeric(breaths.iloc[:, index], errors="coerce").dropna()
    return np.sort(col.to_numpy(dtype=float).astype(int))


def load_recording(data_dir: Path, subject: str, fs: float = BIDMC_FS_HZ) -> Recording:
    """Load one subject. Raises ``FileNotFoundError`` if any of the four files is missing."""
    data_dir = Path(data_dir)
    if not is_subject_complete(data_dir, subject):
        missing = [n for n in expected_files(subject) if not (data_dir / n).is_file()]
        raise FileNotFoundError(f"subject {subject}: missing {missing} in {data_dir}")

    signals = _read_csv(data_dir / file_name(subject, "Signals", "csv"))
    for required in ("Time [s]", "PLETH", "RESP"):
        if required not in signals.columns:
            raise ValueError(f"subject {subject}: Signals file lacks column {required!r}")
    numerics = _numeric_frame(_read_csv(data_dir / file_name(subject, "Numerics", "csv")))
    breaths = _read_csv(data_dir / file_name(subject, "Breaths", "csv"))
    meta = parse_fix((data_dir / file_name(subject, "Fix", "txt")).read_text(encoding="utf-8"))

    return Recording(
        subject=subject,
        fs=fs,
        time=_signal_column(signals, "Time [s]"),
        ppg=_signal_column(signals, "PLETH"),
        resp=_signal_column(signals, "RESP"),
        ecg=_pick_ecg(signals),
        numerics=numerics,
        breaths_ann1=_breath_column(breaths, 0),
        breaths_ann2=_breath_column(breaths, 1),
        meta=meta,
    )


def load_all(data_dir: Path, subjects: Iterable[str] | None = None) -> Iterator[Recording]:
    """Yield recordings for the requested (default: all available) subjects."""
    data_dir = Path(data_dir)
    chosen = tuple(subjects) if subjects is not None else available_subjects(data_dir)
    for subject in chosen:
        yield load_recording(data_dir, subject)
