import numpy as np

from respira.beats import Beats, detect_beats
from respira.preprocess import bandpass


def test_detects_roughly_one_beat_per_cardiac_cycle(synth):
    beats = detect_beats(synth.ppg, synth.fs)
    expected = synth.hr_bpm * (synth.t[-1] / 60.0)
    assert abs(beats.n - expected) / expected < 0.03
    assert np.all(beats.troughs < beats.peaks)
    assert np.all(np.diff(beats.peaks) > 0)


def test_measurement_signal_refines_positions(synth):
    light = bandpass(synth.ppg, synth.fs, 0.0667, 8.0)
    beats = detect_beats(synth.ppg, synth.fs, measurement_signal=light)
    # peaks should be local maxima of the measurement signal
    for p in beats.peaks[5:15]:
        assert light[p] >= light[p - 1] and light[p] >= light[p + 1]


def test_intervals_track_heart_rate(synth):
    beats = detect_beats(synth.ppg, synth.fs)
    hr = 60.0 / np.mean(beats.intervals)
    assert abs(hr - synth.hr_bpm) < 1.0


def test_in_window_selects_by_time(synth):
    beats = detect_beats(synth.ppg, synth.fs)
    sub = beats.in_window(10.0, 20.0)
    assert sub.n > 0
    assert np.all((sub.peak_times >= 10.0) & (sub.peak_times < 20.0))


def test_flat_or_short_signal_returns_no_beats():
    assert detect_beats(np.zeros(5000), 125.0).n == 0
    assert detect_beats(np.ones(10), 125.0).n == 0


def test_beats_requires_matching_lengths():
    import pytest

    with pytest.raises(ValueError):
        Beats(np.array([1, 2]), np.array([0]), 125.0)


def test_rejects_two_dimensional_input():
    import pytest

    with pytest.raises(ValueError):
        detect_beats(np.zeros((10, 10)), 125.0)
