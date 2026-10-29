"""Linear passive blocks: attenuator, filters (analytic, ideal, Touchstone), splitter."""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from scipy import signal

from ..spectrum import Spectrum
from ..units import parse_db, parse_freq
from .base import Block, TwoPort, register


@register("attenuator")
class Attenuator(TwoPort):
    """Flat loss (pad, cable, a passive block you only know the loss of)."""

    def __init__(self, loss_db=0.0):
        super().__init__(loss_db=loss_db)
        self.loss_db = parse_db(loss_db)

    def s21_db(self, f):
        return np.full(np.shape(f), -self.loss_db)


_FAMILIES = {"butter": "butter", "butterworth": "butter", "cheby1": "cheby1", "chebyshev": "cheby1",
             "cheby2": "cheby2", "ellip": "ellip", "elliptic": "ellip", "bessel": "bessel"}
_RESPONSES = {"lowpass": "lowpass", "lp": "lowpass", "highpass": "highpass", "hp": "highpass",
              "bandpass": "bandpass", "bp": "bandpass", "bandstop": "bandstop", "bs": "bandstop",
              "notch": "bandstop"}


@register("filter")
class Filter(TwoPort):
    """Analytic lumped-prototype filter evaluated as |H(j2πf)|.

    response : lowpass | highpass | bandpass | bandstop
    family   : butter | cheby1 | cheby2 | ellip | bessel
    order    : prototype order (a bandpass/bandstop has 2*order poles)
    fc       : cutoff for lowpass/highpass
    f0, bw   : centre (geometric) and bandwidth for bandpass/bandstop, or f1, f2 edges
    ripple_db: passband ripple (cheby1, ellip); stop_db: stopband level (cheby2, ellip)
    insertion_loss_db : flat loss added in the passband (finite Q, connectors...)
    max_rejection_db  : ultimate rejection; real filters leak around themselves,
                        crystal filters typically 70-90 dB, a good cavity >100 dB.

    The edge definition follows scipy: -3 dB for butter, ripple edge for cheby1/ellip,
    stopband edge for cheby2.
    """

    def __init__(self, response="bandpass", family="butter", order=3, fc=None, f0=None, bw=None,
                 f1=None, f2=None, ripple_db=0.5, stop_db=60.0, insertion_loss_db=0.0,
                 max_rejection_db=120.0):
        super().__init__(response=response, family=family, order=order, fc=fc, f0=f0, bw=bw, f1=f1, f2=f2,
                         ripple_db=ripple_db, stop_db=stop_db, insertion_loss_db=insertion_loss_db,
                         max_rejection_db=max_rejection_db)
        self.response = _RESPONSES[response.lower()]
        self.family = _FAMILIES[family.lower()]
        self.order = int(order)
        if self.order < 1:
            raise ValueError("filter order must be at least 1")
        self.il = parse_db(insertion_loss_db)
        self.max_rej = parse_db(max_rejection_db)
        if self.response in ("lowpass", "highpass"):
            if fc is None:
                raise ValueError(f"{self.response} filter needs fc")
            wn = 2 * math.pi * parse_freq(fc)
        else:
            if f1 is not None and f2 is not None:
                lo, hi = parse_freq(f1), parse_freq(f2)
            elif f0 is not None and bw is not None:
                c, b = parse_freq(f0), parse_freq(bw)
                lo = (-b + math.sqrt(b * b + 4 * c * c)) / 2   # f1*f2 = f0^2, f2-f1 = bw
                hi = lo + b
            else:
                raise ValueError(f"{self.response} filter needs f0+bw or f1+f2")
            wn = [2 * math.pi * lo, 2 * math.pi * hi]
        kw = {}
        if self.family in ("cheby1", "ellip"):
            kw["rp"] = parse_db(ripple_db)
        if self.family in ("cheby2", "ellip"):
            kw["rs"] = parse_db(stop_db)
        z, p, k = signal.iirfilter(self.order, wn, btype=self.response, analog=True,
                                   ftype=self.family, output="zpk", **kw)
        self._z, self._p, self._logk = np.asarray(z), np.asarray(p), math.log10(abs(k))
        # normalise so the passband peak sits at 0 dB (bandpass gain constant isn't 1 at f0 for all families)
        self._norm = 0.0
        probe = self._probe_freqs(wn)
        self._norm = -float(np.max(self._raw_db(probe)))

    @staticmethod
    def _probe_freqs(wn):
        if np.ndim(wn) == 0:
            return np.geomspace(wn / 1000, wn * 1000, 4001) / (2 * math.pi)
        lo, hi = wn
        return np.geomspace(lo / 10, hi * 10, 8001) / (2 * math.pi)

    def _raw_db(self, f):
        s = 2j * math.pi * np.atleast_1d(np.asarray(f, dtype=float))
        # log-domain evaluation avoids overflow for high orders at GHz frequencies
        num = np.sum(np.log10(np.abs(s[:, None] - self._z[None, :]) + 1e-300), axis=1) if self._z.size else 0.0
        den = np.sum(np.log10(np.abs(s[:, None] - self._p[None, :])), axis=1)
        return 20 * (self._logk + num - den)

    def s21_db(self, f):
        g = self._raw_db(f) + self._norm
        return np.maximum(g, -self.max_rej) - self.il

    def s21_c(self, f):
        """Complex response with the prototype's phase; same magnitude rules as s21_db."""
        s = 2j * math.pi * np.atleast_1d(np.asarray(f, dtype=float))
        # multiply pole/zero ratios one at a time: no overflow, and far cheaper than logs
        h = np.full(s.shape, 10 ** self._logk * 10 ** (self._norm / 20), dtype=complex)
        z, p = list(self._z), list(self._p)
        while z or p:
            if z and p:
                h *= (s - z.pop()) / (s - p.pop())
            elif p:
                h /= s - p.pop()
            else:
                h *= s - z.pop()
        mag = np.abs(h)
        floor = 10 ** (-self.max_rej / 20)
        with np.errstate(invalid="ignore", divide="ignore"):
            h = np.where(mag < floor, floor * np.exp(1j * np.angle(h)), h)
        return h * 10 ** (-self.il / 20)

    s12_c = s21_c


@register("ideal_filter")
class IdealFilter(TwoPort):
    """Brick-wall filter: passes [fmin, fmax] with insertion loss, rejects the rest by rejection_db."""

    def __init__(self, fmin=None, fmax=None, insertion_loss_db=0.0, rejection_db=200.0):
        super().__init__(fmin=fmin, fmax=fmax, insertion_loss_db=insertion_loss_db, rejection_db=rejection_db)
        self.fmin = parse_freq(fmin) if fmin is not None else 0.0
        self.fmax = parse_freq(fmax) if fmax is not None else math.inf
        self.il = parse_db(insertion_loss_db)
        self.rej = parse_db(rejection_db)

    def s21_db(self, f):
        f = np.asarray(f, dtype=float)
        return np.where((f >= self.fmin) & (f <= self.fmax), -self.il, -self.rej)


def read_touchstone_s2p(path):
    """Minimal Touchstone v1 .s2p reader. Returns (freq_hz, S) with S shape (N, 2, 2)."""
    unit_scale = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}
    scale, fmt = 1e9, "MA"
    nums: list[float] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            toks = line[1:].upper().split()
            for t in toks:
                if t in unit_scale:
                    scale = unit_scale[t]
                elif t in ("MA", "DB", "RI"):
                    fmt = t
            continue
        if line.startswith("["):
            continue  # v2 keywords: not supported beyond plain data
        nums.extend(float(x) for x in re.split(r"[\s,]+", line) if x)
    data = np.array(nums).reshape(-1, 9)
    f = data[:, 0] * scale
    a, b = data[:, 1::2], data[:, 2::2]
    if fmt == "RI":
        s = a + 1j * b
    elif fmt == "MA":
        s = a * np.exp(1j * np.deg2rad(b))
    else:
        s = 10 ** (a / 20) * np.exp(1j * np.deg2rad(b))
    # v1 order: S11 S21 S12 S22
    S = np.empty((len(f), 2, 2), complex)
    S[:, 0, 0], S[:, 1, 0], S[:, 0, 1], S[:, 1, 1] = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    return f, S


@register("sparam")
class SParamFilter(TwoPort):
    """Measured or simulated 2-port from a Touchstone .s2p file (e.g. out of openEMS / mom25d / qucs).

    Only |S21| and |S12| are used for now. Outside the file's frequency range the
    edge value is held and a warning is issued.
    """

    def __init__(self, file, extra_loss_db=0.0):
        super().__init__(file=str(file), extra_loss_db=extra_loss_db)
        self.f, S = read_touchstone_s2p(file)
        self._S21, self._S12 = S[:, 1, 0], S[:, 0, 1]
        self._s21 = 20 * np.log10(np.abs(S[:, 1, 0]) + 1e-30)
        self._s12 = 20 * np.log10(np.abs(S[:, 0, 1]) + 1e-30)
        self.extra = parse_db(extra_loss_db)
        self._warned = False

    def _interp(self, tab, f):
        return np.interp(np.asarray(f, dtype=float), self.f, tab) - self.extra

    def _check(self, f, ctx):
        f = np.asarray(f)
        if not self._warned and f.size and (f.min() < self.f[0] or f.max() > self.f[-1]):
            ctx.warn(f"tones outside {self.params['file']} range; edge |S| values held")
            self._warned = True

    def s21_db(self, f):
        return self._interp(self._s21, f)

    def s12_db(self, f):
        return self._interp(self._s12, f)

    def _interp_c(self, S, f):
        mag = self._interp(20 * np.log10(np.abs(S) + 1e-30), f)
        ph = np.interp(np.asarray(f, dtype=float), self.f, np.unwrap(np.angle(S)))
        return 10 ** (mag / 20) * np.exp(1j * ph)

    def s21_c(self, f):
        return self._interp_c(self._S21, f)

    def s12_c(self, f):
        return self._interp_c(self._S12, f)

    def respond(self, incident, ctx):
        for s in incident.values():
            self._check([t.freq for t in s], ctx)
        return super().respond(incident, ctx)


@register("splitter")
class Splitter(Block):
    """Ideal 2-way power splitter / combiner (reciprocal).

    Ports: 'sum', 'a', 'b'. sum<->a and sum<->b lose 3.01 dB + excess_loss_db;
    a<->b leak by isolation_db.
    """
    ports = ("sum", "a", "b")
    default_in = "sum"
    default_out = "a"

    def __init__(self, excess_loss_db=0.0, isolation_db=25.0):
        super().__init__(excess_loss_db=excess_loss_db, isolation_db=isolation_db)
        self.loss = 10 * math.log10(2) + parse_db(excess_loss_db)
        self.iso = parse_db(isolation_db)

    def respond(self, incident, ctx):
        s, a, b = incident["sum"], incident["a"], incident["b"]
        out_sum = a.scaled(-self.loss)
        out_sum.add_spectrum(b, -self.loss)
        out_a = s.scaled(-self.loss)
        out_a.add_spectrum(b, -self.iso)
        out_b = s.scaled(-self.loss)
        out_b.add_spectrum(a, -self.iso)
        return {"sum": out_sum, "a": out_a, "b": out_b}
