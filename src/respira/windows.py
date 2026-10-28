"""Helpers for sliding analysis windows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    index: int
    t0: float
    t1: float

    @property
    def duration(self) -> float:
        return self.t1 - self.t0

    @property
    def centre(self) -> float:
        return 0.5 * (self.t0 + self.t1)


def sliding_windows(duration_s: float, window_s: float, step_s: float) -> tuple[Window, ...]:
    """Return consecutive windows fully contained in ``[0, duration_s]``.

    Raises ``ValueError`` for non-positive sizes.
    """
    if window_s <= 0 or step_s <= 0:
        raise ValueError("window_s and step_s must be positive")
    if duration_s < window_s:
        return ()
    n = int((duration_s - window_s) // step_s) + 1
    return tuple(Window(i, i * step_s, i * step_s + window_s) for i in range(n))
