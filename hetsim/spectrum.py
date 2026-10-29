"""Tones and spectra.

A tone's identity is its *mixing vector*: the integer combination of
fundamental source frequencies that produced it, e.g. {LO1: 1, RF: -1}. Its
frequency follows from that vector, so two different products that happen to
land on the same frequency stay separate (and labelled), while the same
product reached by two paths merges by power sum.

Signals are real, so frequencies are kept positive: a product computed as
negative is flipped (vector negated). Products at DC are dropped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import numpy as np

from .units import dbm_to_mw, fmt_freq, mw_to_dbm

Vec = tuple  # tuple of (symbol, int coefficient) pairs, sorted by symbol, no zeros


def vec(**coeffs: int) -> Vec:
    return tuple(sorted((k, int(v)) for k, v in coeffs.items() if v))


def vec_of(symbol: str, k: int = 1) -> Vec:
    return ((symbol, int(k)),) if k else ()


def vec_comb(a: Vec, ka: int, b: Vec, kb: int) -> Vec:
    """ka*a + kb*b."""
    d: dict[str, int] = {}
    for s, c in a:
        d[s] = d.get(s, 0) + ka * c
    for s, c in b:
        d[s] = d.get(s, 0) + kb * c
    return tuple(sorted((s, c) for s, c in d.items() if c))


def vec_neg(a: Vec) -> Vec:
    return tuple((s, -c) for s, c in a)


def vec_order(a: Vec) -> int:
    return sum(abs(c) for _, c in a)


def vec_coeff(a: Vec, symbol: str) -> int:
    for s, c in a:
        if s == symbol:
            return c
    return 0


def vec_label(a: Vec) -> str:
    if not a:
        return "DC"
    terms = sorted(a, key=lambda t: (t[1] < 0, -abs(t[1]), t[0]))
    out = []
    for i, (s, c) in enumerate(terms):
        mag = abs(c)
        term = s if mag == 1 else f"{mag}*{s}"
        if i == 0:
            out.append(term if c > 0 else f"-{term}")
        else:
            out.append(f"{'+' if c > 0 else '-'} {term}")
    return " ".join(out)


def parse_label(label: str) -> Vec:
    """Inverse of vec_label for simple labels like 'LO1 - RF' or '2*LO - 3*RF'."""
    d: dict[str, int] = {}
    toks = label.replace("-", " - ").replace("+", " + ").split()
    sign = 1
    for t in toks:
        if t in "+-":
            sign = -1 if t == "-" else 1
            continue
        if "*" in t:
            k, s = t.split("*", 1)
            k = int(k)
        else:
            k, s = 1, t
        d[s] = d.get(s, 0) + sign * k
        sign = 1
    return tuple(sorted((s, c) for s, c in d.items() if c))


@dataclass(frozen=True)
class Tone:
    vec: Vec
    freq: float
    power_dbm: float

    @property
    def label(self) -> str:
        return vec_label(self.vec)

    @property
    def order(self) -> int:
        return vec_order(self.vec)

    def __repr__(self) -> str:
        return f"Tone({self.label}, {fmt_freq(self.freq)}, {self.power_dbm:.2f} dBm)"


def canon(v: Vec, f: float, scale: float | None = None):
    """Make a product real-positive. Returns (vec, freq) or None for DC."""
    s = abs(scale) if scale else abs(f)
    if abs(f) <= 1e-9 * max(s, 1.0):
        return None
    if f < 0:
        return vec_neg(v), -f
    return v, f


class Spectrum:
    """A set of tones keyed by mixing vector. Power stored in mW internally."""

    __slots__ = ("_t",)

    def __init__(self, tones: Iterable[Tone] = ()):
        self._t: dict[Vec, list] = {}
        for t in tones:
            self.add(t.vec, t.freq, t.power_dbm)

    # -- building ----------------------------------------------------------
    def add(self, v: Vec, f: float, p_dbm: float, scale: float | None = None) -> None:
        """Add a product; merges by power sum with an existing tone of the same vector."""
        if p_dbm == -math.inf:
            return
        c = canon(v, f, scale)
        if c is None:
            return
        v, f = c
        mw = dbm_to_mw(p_dbm)
        e = self._t.get(v)
        if e is None:
            self._t[v] = [f, mw]
        else:
            e[1] += mw

    def add_spectrum(self, other: "Spectrum", gain_db: float = 0.0) -> None:
        g = 10 ** (gain_db / 10)
        for v, (f, mw) in other._t.items():
            e = self._t.get(v)
            if e is None:
                self._t[v] = [f, mw * g]
            else:
                e[1] += mw * g

    def scaled(self, gain_db: float) -> "Spectrum":
        s = Spectrum()
        s.add_spectrum(self, gain_db)
        return s

    def shaped(self, gain_fn: Callable[[np.ndarray], np.ndarray]) -> "Spectrum":
        """Apply a frequency-dependent gain (dB) given as a vectorised function of Hz."""
        s = Spectrum()
        if not self._t:
            return s
        vs = list(self._t.keys())
        fs = np.array([self._t[v][0] for v in vs])
        mws = np.array([self._t[v][1] for v in vs])
        g = np.asarray(gain_fn(fs), dtype=float)
        out = mws * 10 ** (g / 10)
        for v, f, mw in zip(vs, fs, out):
            if mw > 0:
                s._t[v] = [float(f), float(mw)]
        return s

    def prune(self, floor_dbm: float, max_tones: int | None = None) -> "Spectrum":
        lim = dbm_to_mw(floor_dbm)
        items = [(v, e) for v, e in self._t.items() if e[1] >= lim]
        if max_tones and len(items) > max_tones:
            items.sort(key=lambda ve: -ve[1][1])
            items = items[:max_tones]
        self._t = dict(items)
        return self

    def copy(self) -> "Spectrum":
        s = Spectrum()
        s._t = {v: list(e) for v, e in self._t.items()}
        return s

    # -- queries -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._t)

    def __iter__(self) -> Iterator[Tone]:
        for v, (f, mw) in sorted(self._t.items(), key=lambda kv: kv[1][0]):
            yield Tone(v, f, mw_to_dbm(mw))

    def __contains__(self, label_or_vec) -> bool:
        return self._key(label_or_vec) in self._t

    def _key(self, label_or_vec) -> Vec:
        return parse_label(label_or_vec) if isinstance(label_or_vec, str) else tuple(label_or_vec)

    def get(self, label_or_vec) -> Tone | None:
        v = self._key(label_or_vec)
        e = self._t.get(v)
        if e is None:
            # the same product may be stored with the opposite sign
            e = self._t.get(vec_neg(v))
            v = vec_neg(v)
        return Tone(v, e[0], mw_to_dbm(e[1])) if e else None

    def power(self, label_or_vec) -> float:
        t = self.get(label_or_vec)
        return t.power_dbm if t else -math.inf

    def tones(self) -> list[Tone]:
        return list(self)

    def band(self, fmin: float | None = None, fmax: float | None = None) -> "Spectrum":
        s = Spectrum()
        s._t = {v: list(e) for v, e in self._t.items()
                if (fmin is None or e[0] >= fmin) and (fmax is None or e[0] <= fmax)}
        return s

    def total_dbm(self, fmin: float | None = None, fmax: float | None = None) -> float:
        return mw_to_dbm(sum(e[1] for e in self._t.values()
                             if (fmin is None or e[0] >= fmin) and (fmax is None or e[0] <= fmax)))

    def strongest(self, n: int = 1) -> list[Tone]:
        return sorted(self, key=lambda t: -t.power_dbm)[:n]

    def involving(self, symbol: str) -> "Spectrum":
        s = Spectrum()
        s._t = {v: list(e) for v, e in self._t.items() if vec_coeff(v, symbol)}
        return s

    def table(self, fmin=None, fmax=None, min_dbm: float = -math.inf, ref=None,
              limit: int | None = None, sort: str = "freq") -> str:
        """Human-readable table. `ref` (label/vec) adds a dBc column relative to that tone."""
        tones = [t for t in self.band(fmin, fmax) if t.power_dbm >= min_dbm]
        if sort == "power":
            tones.sort(key=lambda t: -t.power_dbm)
        if limit:
            tones = tones[:limit]
        ref_p = self.power(ref) if ref is not None else None
        lines = [f"{'frequency':>16}  {'dBm':>9}" + (f"  {'dBc':>8}" if ref_p is not None else "")
                 + f"  {'ord':>3}  product"]
        for t in tones:
            row = f"{fmt_freq(t.freq, 9):>16}  {t.power_dbm:9.2f}"
            if ref_p is not None:
                row += f"  {t.power_dbm - ref_p:8.2f}"
            lines.append(row + f"  {t.order:3d}  {t.label}")
        return "\n".join(lines)

    def to_records(self) -> list[dict]:
        return [{"freq": t.freq, "power_dbm": t.power_dbm, "label": t.label, "order": t.order}
                for t in self]

    def approx_equal(self, other: "Spectrum", tol_db: float, floor_dbm: float) -> bool:
        lim = dbm_to_mw(floor_dbm)
        keys = {v for v, e in self._t.items() if e[1] >= lim} | \
               {v for v, e in other._t.items() if e[1] >= lim}
        for v in keys:
            a = self._t.get(v)
            b = other._t.get(v)
            if a is None or b is None:
                return False
            if abs(10 * math.log10(a[1] / b[1])) > tol_db:
                return False
        return True

    def __repr__(self) -> str:
        return f"Spectrum({len(self)} tones)"
