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
SCALES = ("log", "linear")
IMPEDANCE = 50.0   # ohms, for the linear (voltage) scale


@register("detector")
class Detector(Block):
    """Terminates what arrives at `in` and reads its power: in dBm (log) or volts (linear).

        detect: peak   the strongest single product in the band (a peak detector)
                total  the power sum of the band (an RMS / thermal power meter)
        scale:  log    reads in dBm (a log detector)
                linear reads in volts: the peak amplitude of a sine carrying that power into
                       50 ohm, sqrt(2 R P) (a linear / envelope detector); 0 V when nothing
                       is in the band
        fmin, fmax     the band read (default: everything)

    `out` is a control port carrying the reading, for an X/Y plot's `y`.
    """
    ports = ("in", "out")
    controls = ("out",)
    default_in = "in"

    def __init__(self, fmin=None, fmax=None, detect="peak", scale="log", vswr=None):
        super().__init__(fmin=fmin, fmax=fmax, detect=detect, scale=None if scale == "log" else scale, vswr=vswr)
        if detect not in DETECT:
            raise ValueError(f"detect must be one of {DETECT}, not {detect!r}")
        if scale not in SCALES:
            raise ValueError(f"scale must be one of {SCALES}, not {scale!r}")
        self.scale = scale
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

    @property
    def unit(self) -> str:
        return "V" if self.scale == "linear" else "dBm"

    def output(self, spectrum: Spectrum) -> float:
        """The reading on the detector's scale: dBm (-inf for nothing), or volts (0 for nothing)."""
        p = self.read(spectrum)
        if self.scale == "log":
            return p
        return 0.0 if p == -math.inf else math.sqrt(2 * IMPEDANCE * 10 ** ((p - 30) / 10))

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
