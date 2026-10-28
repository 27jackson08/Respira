import pytest

from respira.windows import sliding_windows


def test_windows_cover_signal_without_overrun():
    windows = sliding_windows(duration_s=480.0, window_s=32.0, step_s=8.0)
    assert len(windows) == 57
    assert windows[0].t0 == 0.0 and windows[0].t1 == 32.0
    assert windows[-1].t1 <= 480.0
    assert windows[-1].index == 56
    assert windows[3].centre == pytest.approx(40.0)
    assert windows[3].duration == pytest.approx(32.0)


def test_short_signal_has_no_windows():
    assert sliding_windows(10.0, 32.0, 8.0) == ()


@pytest.mark.parametrize("window,step", [(0, 8), (32, 0), (-1, 8)])
def test_invalid_sizes_raise(window, step):
    with pytest.raises(ValueError):
        sliding_windows(100.0, window, step)
