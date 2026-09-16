from __future__ import annotations

import time

import pytest

from connections.prover import prover as prover_module
from connections.prover.prover import WallClockExceeded, _wall_clock_alarm


def test_wall_clock_alarm_repeats_when_the_first_alarm_is_swallowed(monkeypatch):
    monkeypatch.setattr(prover_module, "_WALL_CLOCK_REPEAT_SECONDS", 0.05, raising=False)
    swallowed = 0
    started = time.monotonic()
    with pytest.raises(WallClockExceeded):
        with _wall_clock_alarm(0.05):
            while time.monotonic() - started < 5:
                try:
                    while time.monotonic() - started < 5:
                        sum(range(1000))
                except WallClockExceeded:
                    if swallowed:
                        raise
                    swallowed += 1
    assert swallowed == 1
    assert time.monotonic() - started < 1


def test_wall_clock_alarm_is_disarmed_after_the_block(monkeypatch):
    monkeypatch.setattr(prover_module, "_WALL_CLOCK_REPEAT_SECONDS", 0.01)
    with pytest.raises(WallClockExceeded):
        with _wall_clock_alarm(0.01):
            time.sleep(1)
    time.sleep(0.1)
