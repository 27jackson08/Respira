"""Pipeline configuration.

All tunable constants live here so that experiments are reproducible and no
magic numbers are scattered through the signal-processing code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

BIDMC_FS_HZ = 125.0
"""Sampling frequency of the BIDMC waveform signals."""

RIV_NAMES: tuple[str, ...] = ("bw", "am", "fm")
"""Beat-derived respiratory-induced variations used for fusion."""

ALL_RIV_NAMES: tuple[str, ...] = (*RIV_NAMES, "filt")
"""All single-source respiratory signals, including direct PPG band-pass filtering."""


@dataclass(frozen=True)
class PipelineConfig:
    """Parameters of the RR-estimation pipeline.

    Frequencies are expressed in Hz, rates in breaths/beats per minute (bpm).
    """

    window_s: float = 32.0
    """Analysis window length in seconds (Charlton et al. 2016; Pimentel et al. 2017)."""

    step_s: float = 8.0
    """Hop between consecutive analysis windows in seconds."""

    riv_fs: float = 4.0
    """Uniform sampling rate the beat-derived RIVs are resampled to."""

    rr_band_bpm: tuple[float, float] = (4.0, 60.0)
    """Plausible respiratory rate band; the RIVs are band-passed to it."""

    ppg_band_hz: tuple[float, float] = (0.0667, 8.0)
    """Light band-pass applied before measuring pulse amplitudes (keeps baseline wander)."""

    beat_band_hz: tuple[float, float] = (0.5, 8.0)
    """Band-pass applied before pulse peak detection (removes baseline wander)."""

    min_hr_bpm: float = 40.0
    max_hr_bpm: float = 180.0

    ar_order: int = 8
    """Order of the Burg autoregressive model used by the AR estimator."""

    sf_sd_threshold_bpm: float = 4.0
    """Smart Fusion rejects windows where the BW/AM/FM estimates have SD above this."""

    ref_max_disagreement_bpm: float = 2.0
    """Reference windows where the two annotators disagree by more than this are excluded."""

    ref_min_breaths: int = 3
    """Minimum annotated breaths in a window for a reference RR to be defined."""

    sqi_template_threshold: float = 0.86
    """Minimum mean beat-template correlation for a window to be labelled good quality."""

    def with_updates(self, **changes: Any) -> PipelineConfig:
        """Return a copy with the given fields replaced (the dataclass is frozen)."""
        return replace(self, **changes)

    @property
    def rr_band_hz(self) -> tuple[float, float]:
        low, high = self.rr_band_bpm
        return (low / 60.0, high / 60.0)
