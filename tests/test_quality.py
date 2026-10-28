import numpy as np

from respira.beats import Beats, detect_beats
from respira.preprocess import bandpass
from respira.quality import assess_window, plausibility, template_correlation


def test_clean_synthetic_ppg_has_high_sqi(synth):
    light = bandpass(synth.ppg, synth.fs, 0.0667, 8.0)
    beats = detect_beats(synth.ppg, synth.fs, measurement_signal=light).in_window(0.0, 32.0)
    report = assess_window(light, beats, synth.fs, 40.0, 180.0)
    assert report.sqi > 0.95
    assert report.is_plausible
    assert abs(report.hr_bpm - synth.hr_bpm) < 2.0
    assert report.is_acceptable(0.86)


def test_noise_has_low_sqi(synth):
    rng = np.random.default_rng(0)
    noise = rng.normal(size=synth.ppg.size)
    beats = detect_beats(noise, synth.fs).in_window(0.0, 32.0)
    sqi = template_correlation(noise, beats.peaks, synth.fs)
    assert np.isnan(sqi) or sqi < 0.6


def test_too_few_beats_gives_nan():
    assert np.isnan(template_correlation(np.zeros(1000), np.array([500]), 125.0))


def test_plausibility_flags_gaps_and_rates():
    fs = 125.0
    regular = Beats(np.arange(10) * 100, np.arange(10) * 100 - 20, fs)
    hr, ok = plausibility(regular, 40.0, 180.0)
    assert ok and abs(hr - 75.0) < 1e-6
    gappy = Beats(np.array([0, 100, 200, 800]), np.array([-20, 80, 180, 780]), fs)
    assert not plausibility(gappy, 40.0, 180.0)[1]
    single = Beats(np.array([0]), np.array([-20]), fs)
    assert not plausibility(single, 40.0, 180.0)[1]
