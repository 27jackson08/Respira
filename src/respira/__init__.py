"""Respira: respiratory rate estimation from photoplethysmography (PPG).

The package implements a classical digital-signal-processing pipeline:

1. Filter the PPG and detect pulse peaks and troughs.
2. Extract the three respiratory-induced variations (RIVs): baseline wander
   (BW), amplitude modulation (AM) and frequency modulation (FM).
3. Resample each RIV to a uniform grid and band-limit it to plausible
   breathing frequencies.
4. Estimate respiratory rate (RR) with frequency-domain and time-domain
   estimators, and fuse the per-RIV estimates.
5. Evaluate against manually annotated breaths from the BIDMC dataset.
"""

from respira.config import PipelineConfig

__all__ = ["PipelineConfig", "__version__"]
__version__ = "0.1.0"
