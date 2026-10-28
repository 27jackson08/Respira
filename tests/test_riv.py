import numpy as np
import pytest
from scipy import signal

from respira.beats import detect_beats
from respira.config import PipelineConfig
from respira.preprocess import bandpass
from respira.riv import BeatSeries, RIVSet, extract_rivs, filtered_ppg_signal, respiratory_signal


def _dominant_bpm(y: np.ndarray, fs: float) -> float:
    f, p = signal.periodogram(y - y.mean(), fs=fs, nfft=8192)
    band = (f >= 0.0667) & (f <= 1.0)
    return 60.0 * f[band][np.argmax(p[band])]


@pytest.fixture(scope="module")
def rivs(synth):
    light = bandpass(synth.ppg, synth.fs, 0.0667, 8.0)
    beats = detect_beats(synth.ppg, synth.fs, measurement_signal=light)
    return extract_rivs(light, beats), beats


def test_riv_shapes(rivs):
    rset, beats = rivs
    assert rset.bw.n == beats.n
    assert rset.am.n == beats.n
    assert rset.fm.n == beats.n - 1
    assert np.all(rset.am.values > 0)


@pytest.mark.parametrize("name", ["bw", "am", "fm"])
def test_each_riv_is_modulated_at_respiratory_rate(rivs, synth, name):
    rset, _ = rivs
    cfg = PipelineConfig()
    sig = respiratory_signal(rset.get(name), cfg, 0.0, synth.t[-1])
    assert sig.fs == cfg.riv_fs
    assert abs(_dominant_bpm(sig.y, sig.fs) - synth.rr_bpm) < 1.0


def test_direct_filtering_recovers_rate(synth):
    cfg = PipelineConfig()
    sig = filtered_ppg_signal(synth.ppg, synth.fs, cfg)
    assert abs(_dominant_bpm(sig.y, sig.fs) - synth.rr_bpm) < 1.0


def test_slice_is_mean_removed(rivs, synth):
    rset, _ = rivs
    sig = respiratory_signal(rset.bw, PipelineConfig(), 0.0, synth.t[-1])
    seg = sig.slice(10.0, 42.0)
    assert seg.t.size == 128
    assert abs(seg.y.mean()) < 1e-9


def test_unknown_riv_name(rivs):
    rset, _ = rivs
    with pytest.raises(KeyError):
        rset.get("nope")


def test_empty_beats_produce_empty_rivs():
    from respira.beats import Beats

    empty = Beats(np.zeros(0, dtype=int), np.zeros(0, dtype=int), 125.0)
    rset = extract_rivs(np.zeros(100), empty)
    assert isinstance(rset, RIVSet)
    assert rset.bw.n == 0
    sig = respiratory_signal(rset.bw, PipelineConfig(), 0.0, 10.0)
    assert np.all(sig.y == 0)


def test_beat_series_shape_check():
    with pytest.raises(ValueError):
        BeatSeries(np.zeros(3), np.zeros(2))
