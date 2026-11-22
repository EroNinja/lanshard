"""Accurate short sleeps (Windows defaults to a 15.6 ms timer tick)."""
import os
import time


def _hires_timer():
    if os.name == "nt":
        try:
            import ctypes

            ctypes.WinDLL("winmm").timeBeginPeriod(1)
        except OSError:
            pass


_hires_timer()


def precise_sleep(d: float):
    """Sleep ``d`` seconds with sub-millisecond accuracy (coarse sleep + short spin)."""
    if d <= 0:
        return
    end = time.perf_counter() + d
    if d > 0.002:
        time.sleep(d - 0.0015)
    while time.perf_counter() < end:
        pass
