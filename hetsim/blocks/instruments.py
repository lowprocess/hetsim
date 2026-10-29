"""Measurement blocks: a power detector and an X/Y plot.

They take part in the circuit like other blocks, but their outputs are *control* ports:
connections that carry no RF, only tell the viewer what to plot. A source's `sweep` port
stands for its swept tone's frequency; wire it to an X/Y plot's `x`, and a detector's `out`
to its `y`, and the viewer plots the detector's reading at each band-sweep step.
"""
from __future__ import annotations

import math

from ..spectrum import Spectrum
from ..units import parse_freq
from .base import Block, register

DETECT = ("peak", "total")


@register("detector")
class Detector(Block):
    """Terminates what arrives at `in` and reads its power, in dBm.

        detect: peak   the strongest single product in the band (a peak detector)
                total  the power sum of the band (an RMS / thermal power meter)
        fmin, fmax     the band read (default: everything)

    `out` is a control port carrying the reading, for an X/Y plot's `y`.
    """
    ports = ("in", "out")
    controls = ("out",)
    default_in = "in"

    def __init__(self, fmin=None, fmax=None, detect="peak"):
        super().__init__(fmin=fmin, fmax=fmax, detect=detect)
        if detect not in DETECT:
            raise ValueError(f"detect must be one of {DETECT}, not {detect!r}")
        self.fmin, self.fmax, self.detect = parse_freq(fmin), parse_freq(fmax), detect
        if self.fmin is not None and self.fmax is not None and self.fmax <= self.fmin:
            raise ValueError("detector band: fmax must be above fmin")

    def read(self, spectrum: Spectrum) -> float:
        """The reading (dBm) for the wave arriving at `in`; -inf when nothing is in the band."""
        lo = self.fmin if self.fmin is not None else -math.inf
        hi = self.fmax if self.fmax is not None else math.inf
        p = [t.power_dbm for t in spectrum if lo <= t.freq <= hi]
        if not p:
            return -math.inf
        if self.detect == "peak":
            return max(p)
        return 10 * math.log10(sum(10 ** (x / 10) for x in p))

    def respond(self, incident, ctx):
        return {"in": Spectrum()}   # a matched termination

    def respond_td(self, incident, ctx):
        return {}


@register("xyplot")
class XYPlot(Block):
    """An X/Y display: `x` from a source's `sweep` port (the swept tone's frequency), `y` from a
    detector's `out` (its reading at each sweep step). Both are control ports."""
    ports = ("x", "y")
    controls = ("x", "y")

    def __init__(self):
        super().__init__()

    def respond(self, incident, ctx):
        return {}

    def respond_td(self, incident, ctx):
        return {}
