"""Time-domain (sampled) solver: periodic steady state on a coherent FFT grid.

Every source tone is placed exactly on an FFT bin, so the whole simulation is
periodic with period 1/resolution. That makes the steady state exact rather
than approximate:

* linear blocks (filters, S-parameters) multiply the spectrum by their complex
  response, i.e. circular convolution, the periodic steady state with true phase;
* nonlinear blocks (amplifiers, mixers) act sample by sample on the waveform;
* the same bidirectional wave iteration as the tone solver couples the blocks.

The differences from the tone solver are the point of having both:

* waveforms add in amplitude and phase, so products landing in the same bin
  interfere (image vs wanted, two IM paths...), rather than adding in power;
* nonlinearities act on the whole waveform: every tone interacts with every
  other, including triple beats and products of products, and compression
  follows the instantaneous peak rather than the total power;
* anything above fs/2 aliases. The sampling plan uses the tone solver to find
  how high the significant products reach and warns about the ones it cannot
  cover.

Plain RF sampling is used, so the record length scales as fmax / resolution;
a few million samples is fine.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, fields
from functools import reduce

import numpy as np
from scipy import fft as sfft
from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import splu

from ..circuit import Circuit, Settings
from ..spectrum import Spectrum
from ..units import fmt_freq, parse_freq


@dataclass
class TDSettings:
    resolution: float | str | None = None  # FFT bin spacing; default: GCD of all source frequencies
    fmax: float | str | None = None        # highest frequency represented; default: planned (see alias_db)
    fs: float | str | None = None          # sample rate; overrides fmax. Rounded so tones stay on bins
    oversample: float = 1.25               # fs = 2 * oversample * fmax
    alias_db: float = 120.0                # cover products within this many dB of the strongest
    max_samples: int = 1 << 24
    impedance: float = 50.0
    seed: int = 1                          # for source phases left unspecified
    tol: float = 1e-12                     # relative waveform change for convergence
    max_iter: int = 200

    @classmethod
    def from_any(cls, x) -> "TDSettings":
        if x is None:
            return cls()
        if isinstance(x, TDSettings):
            return copy.copy(x)
        names = {f.name for f in fields(cls)}
        bad = set(x) - names
        if bad:
            raise ValueError(f"unknown td settings {sorted(bad)}; known: {sorted(names)}")
        return cls(**x)


@dataclass
class Plan:
    fs: float
    n: int
    resolution: float
    fmax_needed: float
    notes: list = field(default_factory=list)

    @property
    def nyquist(self) -> float:
        return self.fs / 2

    def __str__(self):
        return (f"fs = {fmt_freq(self.fs)}, N = {self.n}, resolution = {fmt_freq(self.resolution)}, "
                f"Nyquist = {fmt_freq(self.nyquist)} (products reach {fmt_freq(self.fmax_needed)})")


def _gcd_hz(freqs) -> float:
    ints = [int(round(f)) for f in freqs if f > 0]
    return float(reduce(math.gcd, ints)) if ints else 1.0


def make_plan(circuit: Circuit, tds: TDSettings, tone_result) -> Plan:
    notes = []
    freqs = []
    for b in circuit.blocks.values():
        for t in getattr(b, "tones", []):
            freqs.append(t.freq)
        if getattr(b, "sample_rate", None):   # an ADC's samples must fall on the grid too
            freqs.append(b.sample_rate)
    if tds.resolution is not None:
        df = parse_freq(tds.resolution)
    else:
        df = _gcd_hz(freqs)
        if any(abs(f - round(f)) > 1e-6 for f in freqs):
            notes.append("non-integer source frequencies rounded to 1 Hz for the grid")
    off = [f for f in freqs if abs(f / df - round(f / df)) > 1e-6]
    if off:
        notes.append(f"{len(off)} source tone(s) not on the {fmt_freq(df)} grid; moved to the nearest bin "
                     f"(e.g. {fmt_freq(off[0])})")

    # how high do the significant products go?
    tones = [t for (_, _), s in tone_result._out.items() for t in s]
    top = max((t.power_dbm for t in tones), default=0.0)
    sig = [t for t in tones if t.power_dbm >= top - tds.alias_db]
    fneed = max((t.freq for t in sig), default=max(freqs, default=df))

    if tds.fs is not None:
        fs = parse_freq(tds.fs)
        n = max(2, int(round(fs / df)))
    else:
        fmax = parse_freq(tds.fmax) if tds.fmax is not None else fneed
        n = sfft.next_fast_len(int(math.ceil(2 * tds.oversample * fmax / df)) + 1)
    if n % 2:
        n = sfft.next_fast_len(n + 1)
        n += n % 2
    fs = n * df
    if n > tds.max_samples:
        raise ValueError(
            f"time-domain record would need {n} samples (fs {fmt_freq(fs)} at {fmt_freq(df)} resolution). "
            f"Coarsen 'resolution' (move tones onto a coarser grid), lower 'fmax', or raise 'max_samples'.")
    above = [t for t in sig if t.freq >= fs / 2]
    if above:
        worst = max(above, key=lambda t: t.power_dbm)
        notes.append(f"{len(above)} significant product(s) above Nyquist {fmt_freq(fs / 2)} will alias, "
                     f"strongest {worst.label} at {fmt_freq(worst.freq)}, {worst.power_dbm:.1f} dBm")
    return Plan(fs, n, df, fneed, notes)


@dataclass
class TDContext:
    fs: float
    n: int
    df: float
    f: np.ndarray            # rfft bin frequencies
    impedance: float
    seed: int
    cache: dict
    warn: callable
    pn_span: float = 0.0     # widest phase-noise offset of any source (Hz); 0: no phase noise

    def amp(self, p_dbm: float) -> float:
        """Peak amplitude (V) of a sinusoid of power p_dbm into the reference impedance."""
        return math.sqrt(2 * self.impedance * 10 ** ((p_dbm - 30) / 10))

    def filt(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        if not x.any():
            return x
        return sfft.irfft(sfft.rfft(x) * h, self.n)

    def zeros(self) -> np.ndarray:
        return np.zeros(self.n)


PRECISION_FLOOR = 1e-3     # with tol 1e-12, a change 1e-15 of the largest wave is rounding
RUNAWAY_VOLTS = 1e3        # RMS: no real wave gets here (over 70 dBm)
RUNAWAY = "waves kept growing, so the solve stopped and these levels mean nothing: products gain on each pass. A mixer " \
          "with reflective filters on both ports does this: put a pad of about 6 dB on one side, or give one filter a plain vswr"


def _factorise(circuit, ops, f, index, chunk=20000):
    """Sparse LUs of I - A(f), one per chunk of bins, and the largest loop gain seen. A bin's block is mostly
    empty, about two entries per port along a chain, which beats a dense solve per bin. Bins don't
    couple, so chunks keep the matrices within what the factorisation can handle."""
    from ..coherent import BATCH_TONES, _loop_gain, _network
    size = len(index)
    lus, loop = [], 0.0
    for first in range(0, len(f), chunk):
        part = f[first:first + chunk]
        rows, cols, vals = [], [], []
        for start in range(0, len(part), BATCH_TONES):
            a = _network(circuit, ops, part[start:start + BATCH_TONES], index)
            loop = max(loop, _loop_gain(a))
            k, i, j = np.nonzero(a)
            at = (k + start) * size
            rows.append(at + i)
            cols.append(at + j)
            vals.append(a[k, i, j])
        n = len(part) * size
        m = identity(n, format="csc", dtype=complex) - csc_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n), dtype=complex)
        lus.append((first, first + len(part), splu(m)))
    return lus, loop


def _solve_reflecting(circuit, plan, tds, models, ctxs, all_ports, zero):
    """The wave iteration for a circuit with reflections, where plain rounds would crawl or stall.

    As in the coherent tone solver, the linear network is solved directly, here per FFT bin, for the
    bins holding anything. Only what the blocks add beyond their linear paths goes round: everything
    a source or mixer makes, and the difference between an amplifier and its small-signal gain.
    That settles in a few rounds however strong the reflections.
    """
    rf = [(n, p) for n, b in circuit.blocks.items() for p in b.ports if p not in b.controls]
    index = {k: i for i, k in enumerate(rf)}
    ops = {n: getattr(b, "gain", None) for n, b in circuit.blocks.items()}
    f = ctxs[next(iter(ctxs))].f
    n_bins = len(f)
    incident = {p: zero for p in all_ports}
    outgoing = {p: zero for p in all_ports}
    active = [n for n, b in circuit.blocks.items() if not b.td_linear]
    converged, loop = False, 0.0
    lu, lu_bins = None, None
    it = 0
    for it in range(1, tds.max_iter + 1):
        s = {}
        for n in active:
            b = circuit.blocks[n]
            inc = {p: incident[(n, p)] for p in b.ports}
            res = models[n](b, inc, ctxs[n])
            specs = {q: sfft.rfft(inc[q]) for q in b.ports if q not in b.controls and inc[q].any()}
            for p in b.ports:
                if p in b.controls:
                    continue
                y = res.get(p)
                y = sfft.rfft(y) if y is not None and y.any() else np.zeros(n_bins, complex)
                for q, g in b.linear_c(p, f, ops[n]):
                    if q in specs:
                        y = y - g * specs[q]
                if np.any(y):
                    s[(n, p)] = y
        power = np.zeros(n_bins)
        for y in s.values():
            power += np.abs(y) ** 2
        top = power.max() if len(power) else 0.0
        bins = np.nonzero(power > 1e-34 * top)[0] if top > 0 else np.zeros(0, int)
        x = np.zeros((n_bins, len(rf)), complex)
        if len(bins):
            # the network doesn't change between rounds, so factorise once, for as many bins as have held anything
            if lu is None or len(np.setdiff1d(bins, lu_bins, assume_unique=True)):
                lu_bins = bins if lu is None else np.union1d(lu_bins, bins)
                lu, loop = _factorise(circuit, ops, f[lu_bins], index)
            src = np.zeros((len(lu_bins), len(rf)), complex)
            for key, y in s.items():
                src[:, index[key]] = y[lu_bins]
            for first, last, part in lu:
                x[lu_bins[first:last]] = part.solve(src[first:last].ravel()).reshape(last - first, len(rf))
        outgoing = {p: zero for p in all_ports}
        for key, i in index.items():
            if np.any(x[:, i]):
                outgoing[key] = sfft.irfft(x[:, i], plan.n)
        new_inc = {p: (outgoing[circuit.wires[p]] if p in circuit.wires else zero) for p in all_ports}
        # relative to each wave, but a wave below the largest's rounding error never settles: it is only that error
        norms = {p: np.linalg.norm(new_inc[p]) for p in all_ports}
        floor = PRECISION_FLOOR * max(norms.values())
        done = all(np.linalg.norm(new_inc[p] - incident[p]) <= tds.tol * max(norms[p], floor, 1e-300) for p in all_ports)
        incident = new_inc
        if done:
            converged = True
            break
        if max(norms.values()) > RUNAWAY_VOLTS * math.sqrt(plan.n):
            break
    return incident, outgoing, it, converged and loop < 1.0, loop


def simulate_td(circuit: Circuit, settings: Settings | None = None, td=None, **td_kw) -> "TDResult":
    """Run the time-domain solver. `td` is a TDSettings or dict; keyword arguments override it."""
    from .models import get_model

    st = settings or Settings()
    tds = TDSettings.from_any(td if td is not None else st.td)
    for k, v in td_kw.items():
        if not hasattr(tds, k):
            raise TypeError(f"unknown td setting {k!r}")
        setattr(tds, k, v)

    tone = circuit.simulate(st)
    plan = make_plan(circuit, tds, tone)

    warnings: dict[str, list[str]] = {}
    if plan.notes:
        warnings["<plan>"] = list(plan.notes)

    def make_ctx(name):
        def warn(msg, _n=name):
            lst = warnings.setdefault(_n, [])
            if msg not in lst:
                lst.append(msg)
        cache = circuit.blocks[name].__dict__.setdefault("_td_cache", {})
        return TDContext(plan.fs, plan.n, plan.resolution, sfft.rfftfreq(plan.n, 1 / plan.fs),
                         tds.impedance, tds.seed, cache, warn, pn_span)

    pn_span = max((t.phase_noise[-1][0] for b in circuit.blocks.values() for t in getattr(b, "tones", [])
                   if getattr(t, "phase_noise", None)), default=0.0)

    ctxs = {n: make_ctx(n) for n in circuit.blocks}
    models = {n: get_model(b) for n, b in circuit.blocks.items()}
    zero = np.zeros(plan.n)
    zero.flags.writeable = False
    all_ports = [(n, p) for n, b in circuit.blocks.items() for p in b.ports]
    incident = {p: zero for p in all_ports}
    outgoing: dict = {}
    last_inputs: dict = {}
    converged = False
    it = 0
    reflecting = any(b.gamma2 for b in circuit.blocks.values())
    for it in range(1, 0 if reflecting else tds.max_iter + 1):
        prev_out, outgoing = outgoing, {}
        for n, b in circuit.blocks.items():
            inc = [incident[(n, p)] for p in b.ports]
            # blocks are memoryless: if every incident wave is the same array as last time,
            # the outputs are too, so reuse them (and keep their identity, so the skip
            # propagates downstream once the wavefront has passed)
            if n in last_inputs and all(a is b_ for a, b_ in zip(inc, last_inputs[n])):
                for p in b.ports:
                    outgoing[(n, p)] = prev_out[(n, p)]
                continue
            last_inputs[n] = inc
            res = models[n](b, dict(zip(b.ports, inc)), ctxs[n])
            for p in b.ports:
                x = res.get(p)
                x = zero if x is None else x
                if p in b.gamma2 and incident[(n, p)].any():
                    h = ctxs[n].cache.get(("gamma", p, plan.n, plan.fs))
                    if h is None:
                        h = ctxs[n].cache[("gamma", p, plan.n, plan.fs)] = b.gamma_c(p, ctxs[n].f)
                    x = x + ctxs[n].filt(incident[(n, p)], h)
                outgoing[(n, p)] = x
        new_inc = {p: (outgoing[circuit.wires[p]] if p in circuit.wires else zero) for p in all_ports}
        done = True
        for p in all_ports:
            a, b_ = new_inc[p], incident[p]
            if a is b_:
                continue
            d = np.linalg.norm(a - b_)
            if d > tds.tol * max(np.linalg.norm(a), 1e-300):
                done = False
                break
        incident = new_inc
        if done:
            converged = True
            break
    if reflecting:
        incident, outgoing, it, converged, loop = _solve_reflecting(circuit, plan, tds, models, ctxs, all_ports, zero)
        if not converged and max(np.linalg.norm(w) for w in outgoing.values()) > RUNAWAY_VOLTS * math.sqrt(plan.n):
            warnings.setdefault("<solver>", []).append(RUNAWAY)
        if loop >= 1.0:
            warnings.setdefault("<solver>", []).append(f"a feedback loop has gain {loop:.2f} at some frequency: "
                                                       "no steady state, these levels mean nothing")
    if not converged:
        warnings.setdefault("<solver>", []).append(f"did not converge in {tds.max_iter} iterations")
    return TDResult(circuit, plan, tds, incident, outgoing, it, converged, warnings, tone)


class TDSpectrum:
    """FFT of a periodic waveform: one value per bin, in dBm into the reference impedance.

    With the coherent grid every tone sits exactly in a bin, so no window is
    used and bin power is tone power. When the tone-solver result is attached,
    bins are labelled with the products the tone solver put there.
    """

    def __init__(self, x: np.ndarray, fs: float, impedance: float = 50.0, tone: Spectrum | None = None):
        n = len(x)
        X = sfft.rfft(x)
        a = 2 * np.abs(X) / n
        a[0] /= 2
        p_w = a ** 2 / (2 * impedance)
        p_w[0] *= 2
        if n % 2 == 0:
            p_w[-1] /= 2  # Nyquist bin is real, like DC
        with np.errstate(divide="ignore"):
            self.p_dbm = np.maximum(10 * np.log10(p_w) + 30, -400.0)
        self.freqs = sfft.rfftfreq(n, 1 / fs)
        self.fs = fs
        self.df = fs / n
        self.tone = tone

    def _sel(self, fmin, fmax):
        m = np.ones(len(self.freqs), bool)
        m[0] = False
        if fmin is not None:
            m &= self.freqs >= fmin - self.df / 2
        if fmax is not None:
            m &= self.freqs <= fmax + self.df / 2
        return m

    def power_at(self, f: float) -> float:
        return float(self.p_dbm[int(round(f / self.df))])

    def total_dbm(self, fmin=None, fmax=None) -> float:
        m = self._sel(fmin, fmax)
        s = np.sum(10 ** ((self.p_dbm[m] - 30) / 10))
        return 10 * math.log10(s) + 30 if s > 0 else -math.inf

    def peaks(self, min_dbm: float = -200.0, fmin=None, fmax=None) -> list[tuple[float, float]]:
        m = self._sel(fmin, fmax) & (self.p_dbm >= min_dbm)
        idx = np.nonzero(m)[0]
        return [(float(self.freqs[i]), float(self.p_dbm[i])) for i in idx]

    def products_at(self, f: float) -> list:
        """Tone-solver products in the same bin, strongest first."""
        if self.tone is None:
            return []
        return sorted((t for t in self.tone.band(f - self.df / 2, f + self.df / 2)), key=lambda t: -t.power_dbm)

    def table(self, min_dbm: float = -200.0, fmin=None, fmax=None, sort: str = "freq",
              limit: int | None = None) -> str:
        rows = self.peaks(min_dbm, fmin, fmax)
        if sort == "power":
            rows.sort(key=lambda r: -r[1])
        if limit:
            rows = rows[:limit]
        head = f"{'frequency':>16}  {'TD dBm':>9}"
        if self.tone is not None:
            head += f"  {'tone dBm':>9}  {'diff':>6}  products (tone solver)"
        lines = [head]
        for f, p in rows:
            line = f"{fmt_freq(f, 9):>16}  {p:9.2f}"
            if self.tone is not None:
                ts = self.products_at(f)
                if ts:
                    pt = 10 * math.log10(sum(10 ** (t.power_dbm / 10) for t in ts))
                    names = " | ".join(t.label for t in ts[:4]) + (" | ..." if len(ts) > 4 else "")
                    line += f"  {pt:9.2f}  {p - pt:6.2f}  {names}"
                else:
                    line += f"  {'-':>9}  {'':>6}  (not predicted)"
            lines.append(line)
        return "\n".join(lines)


class TDResult:
    """Time-domain result. result.waveform('iff.out') is the wave leaving that port (V into the
    reference impedance); result.spectrum('iff.out') its FFT, labelled from the tone solver."""

    def __init__(self, circuit, plan, settings, incident, outgoing, iterations, converged, warnings, tone):
        self.circuit = circuit
        self.plan = plan
        self.settings = settings
        self._inc = incident
        self._out = outgoing
        self.iterations = iterations
        self.converged = converged
        self.warnings = warnings
        self.tone = tone   # the tone-solver result used for planning and labelling

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.plan.n) / self.plan.fs

    def waveform(self, ref) -> np.ndarray:
        return self._out[self.circuit.port(ref, "out")]

    def incident_waveform(self, ref) -> np.ndarray:
        return self._inc[self.circuit.port(ref, "in")]

    def spectrum(self, ref) -> TDSpectrum:
        return TDSpectrum(self.waveform(ref), self.plan.fs, self.settings.impedance, self.tone[ref])

    def incident_spectrum(self, ref) -> TDSpectrum:
        return TDSpectrum(self.incident_waveform(ref), self.plan.fs, self.settings.impedance,
                          self.tone.incident(ref))

    __getitem__ = spectrum

    def warning_text(self) -> str:
        return "\n".join(f"[{b}] {m}" for b, ms in self.warnings.items() for m in ms)

    def summary(self, port, **table_kw) -> str:
        head = (f"{self.circuit.name or 'circuit'} (time domain): wave leaving {port}  "
                f"({'converged' if self.converged else 'NOT converged'} in {self.iterations} it.)\n{self.plan}")
        txt = head + "\n" + self.spectrum(port).table(**table_kw)
        w = self.warning_text()
        return txt + ("\nwarnings:\n" + w if w else "")

    def __repr__(self):
        return f"TDResult({self.circuit.name!r}, {self.plan}, converged={self.converged})"
