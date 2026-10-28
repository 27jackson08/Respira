import numpy as np

from respira import fusion

FS = 4.0
BAND = (4.0 / 60.0, 1.0)


def test_smart_fusion_averages_when_estimates_agree():
    out = fusion.smart_fusion({"bw": 15.0, "am": 16.0, "fm": 14.0}, sd_threshold_bpm=4.0)
    assert out.rr_bpm == 15.0
    assert 0.0 < out.confidence <= 1.0


def test_smart_fusion_abstains_on_disagreement_or_missing():
    assert not fusion.smart_fusion({"bw": 10.0, "am": 25.0, "fm": 15.0}).is_valid
    assert not fusion.smart_fusion({"bw": 15.0, "am": float("nan"), "fm": 15.0}).is_valid
    assert not fusion.smart_fusion({}).is_valid


def test_median_fusion_ignores_single_outlier():
    out = fusion.median_fusion({"bw": 15.0, "am": 40.0, "fm": 14.0})
    assert out.rr_bpm == 15.0
    assert not fusion.median_fusion({"bw": float("nan")}).is_valid


def test_spectral_fusion_finds_shared_peak():
    t = np.arange(128) / FS
    shared = np.sin(2 * np.pi * 0.3 * t)
    signals = [shared, shared + 0.5 * np.sin(2 * np.pi * 0.6 * t), np.sin(2 * np.pi * 0.3 * t + 1.0)]
    out = fusion.spectral_fusion(signals, FS, BAND)
    assert abs(out.rr_bpm - 18.0) < 1.0
    assert not fusion.spectral_fusion([np.zeros(128)], FS, BAND).is_valid


def test_temporal_smooth_carries_forward_and_blends():
    out = fusion.temporal_smooth([np.nan, 10.0, np.nan, 20.0], alpha=0.5)
    assert np.isnan(out[0])
    assert out[1] == 10.0
    assert out[2] == 10.0
    assert out[3] == 15.0
