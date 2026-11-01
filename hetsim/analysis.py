"""Sweeps and spurious-response analysis."""
from __future__ import annotations

import copy
import json
import math
import os
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .circuit import Circuit, Settings
from .spectrum import vec_coeff, vec_label, vec_order
from .units import fmt_freq, parse_freq


def _settings(settings, **over):
    st = copy.copy(settings) if settings else Settings()
    for k, v in over.items():
        setattr(st, k, v)
    return st


@dataclass
class Sweep:
    freqs: np.ndarray
    out_dbm: np.ndarray
    gain_db: np.ndarray
    source: str
    port: str
    band: tuple


def response_sweep(circuit: Circuit, source: str, port: str, band, freqs, tone: int = 0,
                   settings: Settings | None = None, mute=()) -> Sweep:
    """Sweep one source tone across `freqs`; record total power landing in `band` at `port`.

    Narrow responses (as narrow as the IF bandwidth divided by the product order)
    fall between points of a coarse sweep; use find_responses() to locate them exactly.
    """
    band = (parse_freq(band[0]), parse_freq(band[1]))
    src = circuit[source]
    f0 = src.tones[tone].freq
    p_in = src.tones[tone].power
    freqs = np.asarray([parse_freq(f) for f in freqs], dtype=float)
    out = np.empty(len(freqs))
    with _muted(circuit, mute, keep=src.tones[tone]):
        try:
            for i, f in enumerate(freqs):
                src.set(tone, freq=f)
                r = circuit.simulate(_settings(settings))
                out[i] = r[port].total_dbm(*band)
        finally:
            src.set(tone, freq=f0)
    return Sweep(freqs, out, out - p_in, source, port, band)


@dataclass
class Response:
    freq: float          # stimulus frequency
    gain_db: float       # power in the output band minus stimulus power
    rejection_db: float  # relative to the strongest (normally the wanted) response
    label: str           # strongest product in the band
    order: int
    out_freq: float


@dataclass
class ResponseTable:
    responses: list = field(default_factory=list)
    source: str = ""
    port: str = ""
    band: tuple = ()

    def table(self, limit: int | None = None, min_rejection_db: float | None = None) -> str:
        rows = sorted(self.responses, key=lambda r: -r.gain_db)
        if min_rejection_db is not None:
            rows = [r for r in rows if r.rejection_db >= -min_rejection_db]
        if limit:
            rows = rows[:limit]
        lines = [f"{'stimulus':>16}  {'gain dB':>8}  {'rej dB':>7}  {'ord':>3}  product at output"]
        for r in rows:
            lines.append(f"{fmt_freq(r.freq, 9):>16}  {r.gain_db:8.2f}  {max(0.0, -r.rejection_db):7.2f}  "
                         f"{r.order:3d}  {r.label}")
        return "\n".join(lines)


class _muted:
    """Temporarily silence other sources (e.g. blockers) during a test."""

    def __init__(self, circuit, names, keep=None):
        self.c, self.names, self.keep, self.saved = circuit, list(names), keep, []

    def __enter__(self):
        for n in self.names:
            for t in self.c[n].tones:
                if t is self.keep:
                    continue  # never silence the stimulus itself
                self.saved.append((t, t.power))
                t.power = -1000.0

    def __exit__(self, *exc):
        for t, p in self.saved:
            t.power = p


def find_responses(circuit: Circuit, source: str, port: str, band, fmin, fmax, tone: int = 0,
                   settings: Settings | None = None, mute=(), max_order: int = 12,
                   probe_floor_dbm: float = -400.0, min_gain_db: float = -250.0) -> ResponseTable:
    """Find every stimulus frequency in [fmin, fmax] that produces output in `band` at `port`.

    Works from the mixing vectors: one simulation at the source's present
    frequency, with a very low floor, shows every product the stimulus takes
    part in. A product c*f + K lands in the band centre when f = (+/-fc - K) / c,
    so each product predicts its own response frequencies, which are then
    simulated exactly. This finds image, IF-feedthrough, half-IF and m x n
    spur responses however narrow they are.
    """
    band = (parse_freq(band[0]), parse_freq(band[1]))
    fmin, fmax = parse_freq(fmin), parse_freq(fmax)
    fc = 0.5 * (band[0] + band[1])
    src = circuit[source]
    sym = src.tones[tone].name
    f_nom = src.tones[tone].freq
    p_in = src.tones[tone].power

    cands: list[float] = []
    with _muted(circuit, mute, keep=src.tones[tone]):
        try:
            probe = circuit.simulate(_settings(settings, floor_dbm=probe_floor_dbm))
            for t in probe[port]:
                c = vec_coeff(t.vec, sym)
                if c == 0 or t.order > max_order:
                    continue
                k = t.freq - c * f_nom
                for s in (1, -1):
                    f = (s * fc - k) / c
                    if fmin <= f <= fmax:
                        cands.append(f)
            # an ADC's alias responses: inputs to it that fold into the band, which no probe shows
            for name, b in circuit.blocks.items():
                if not hasattr(b, "sample_rate"):
                    continue
                for t in probe.incident((name, "in")):
                    c = vec_coeff(t.vec, sym)
                    if c == 0 or t.order > max_order:
                        continue
                    k = t.freq - c * f_nom
                    for z in range(1, int(b.bw / b.fs) + 2):
                        for fa in (z * b.fs - fc, z * b.fs + fc):
                            for s in (1, -1):
                                f = (s * fa - k) / c
                                if fmin <= f <= fmax:
                                    cands.append(f)
            cands = sorted(set(round(f, 3) for f in cands))

            found: dict[str, Response] = {}
            for f in cands:
                src.set(tone, freq=f)
                r = circuit.simulate(_settings(settings))
                spec = r[port].band(*band)
                p = spec.total_dbm()
                if p - p_in < min_gain_db or not len(spec):
                    continue
                mine = [t for t in spec if vec_coeff(t.vec, sym)]
                best = max(mine or list(spec), key=lambda t: t.power_dbm)
                resp = Response(f, p - p_in, 0.0, vec_label(best.vec), best.order, best.freq)
                key = f"{best.label}@{round(f, -1)}"
                if key not in found or found[key].gain_db < resp.gain_db:
                    found[key] = resp
        finally:
            src.set(tone, freq=f_nom)

    rs = sorted(found.values(), key=lambda r: r.freq)
    if rs:
        top = max(r.gain_db for r in rs)
        for r in rs:
            r.rejection_db = r.gain_db - top
    return ResponseTable(rs, source, port, band)


# ---------------------------------------------------------------- band sweep
@dataclass
class SweptProduct:
    """One product followed across a band sweep. NaN where it was absent (below the floor)."""
    vec: tuple
    freqs: np.ndarray    # Hz at each step
    powers: np.ndarray   # dBm at each step

    @property
    def label(self) -> str:
        return vec_label(self.vec)

    @property
    def i_max(self) -> int:
        return int(np.nanargmax(self.powers))

    @property
    def p_max(self) -> float:
        return float(self.powers[self.i_max])

    @property
    def f_at_max(self) -> float:
        return float(self.freqs[self.i_max])

    @property
    def fmin(self) -> float:
        return float(np.nanmin(self.freqs))

    @property
    def fmax(self) -> float:
        return float(np.nanmax(self.freqs))


class BandSweep:
    """Results of Circuit.simulate_sweep(): one tone-solver Result per step.

    steps[i] maps each swept tone's symbol to its frequency at step i.
    """

    def __init__(self, circuit: Circuit, steps: list[dict], results: list):
        self.circuit, self.steps, self.results = circuit, steps, results

    def products(self, ref, incident: bool = False, min_dbm: float = -200.0,
                 fmin=None, fmax=None) -> list[SweptProduct]:
        """Every product at a port over the sweep, strongest first. With fmin/fmax, only the
        steps where the product lies in that band count (e.g. spurs that cross the IF)."""
        lo = parse_freq(fmin) if fmin is not None else -math.inf
        hi = parse_freq(fmax) if fmax is not None else math.inf
        n = len(self.results)
        seen: dict = {}
        for i, r in enumerate(self.results):
            for t in (r.incident(ref) if incident else r[ref]):
                if t.power_dbm < min_dbm or not lo <= t.freq <= hi:
                    continue
                if t.vec not in seen:
                    seen[t.vec] = (np.full(n, np.nan), np.full(n, np.nan))
                seen[t.vec][0][i] = t.freq
                seen[t.vec][1][i] = t.power_dbm
        out = [SweptProduct(v, f, p) for v, (f, p) in seen.items()]
        return sorted(out, key=lambda s: -s.p_max)

    def step_text(self, i: int) -> str:
        return ", ".join(f"{k} {fmt_freq(f, 7)}" for k, f in self.steps[i].items())

    def table(self, ref, incident: bool = False, min_dbm: float = -200.0, fmin=None, fmax=None,
              limit: int | None = None) -> str:
        rows = self.products(ref, incident, min_dbm, fmin, fmax)[:limit]
        lines = [f"{'from':>14} {'to':>14}  {'max dBm':>8}  {'ord':>3}  {'product':<24} worst at"]
        for s in rows:
            lines.append(f"{fmt_freq(s.fmin, 7):>14} {fmt_freq(s.fmax, 7):>14}  {s.p_max:8.2f}  "
                         f"{vec_order(s.vec):3d}  {s.label:<24} {self.step_text(s.i_max)}")
        return "\n".join(lines)


# ---------------------------------------------------------------- parallel sweep
# The tone solver is pure Python, so threads can't run steps at once (the GIL); worker
# processes can. Each worker keeps the last circuit it built and reuses it while the
# netlist is unchanged, so a task costs a few kB of netlist plus the spectra sent back.
_POOL = None
_POOL_LOCK = threading.Lock()
_WORKER = None            # in a worker: (netlist key, circuit)
_POOL_FAILED = False      # the pool couldn't start (e.g. no importable main script): stay serial
PARALLEL_MIN_S = 0.03     # below this much remaining work a sweep stays in-process
# one worker per physical core: on SMT machines that beat using every logical CPU
POOL_SIZE = max(1, (os.cpu_count() or 2) // 2)


def _pool() -> ProcessPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ProcessPoolExecutor(max_workers=POOL_SIZE)
        return _POOL


def warm_pool() -> bool:
    """Start the worker processes now, so the first parallel sweep doesn't wait for them.
    Returns False (and sweeps stay serial) if they can't be started."""
    try:
        for f in [_pool().submit(_noop) for _ in range(POOL_SIZE)]:
            f.result()
        return True
    except Exception as e:  # noqa: BLE001
        _pool_failed(e)
        return False


def _pool_failed(e: Exception) -> None:
    global _POOL, _POOL_FAILED
    _POOL_FAILED = True
    with _POOL_LOCK:
        if _POOL is not None:
            _POOL.shutdown(wait=False, cancel_futures=True)
            _POOL = None
    warnings.warn(f"parallel band sweep unavailable, running steps in-process ({type(e).__name__}: {e}). "
                  "Scripts that use workers need an `if __name__ == \"__main__\":` guard.", RuntimeWarning,
                  stacklevel=3)


def shutdown_pool() -> None:
    """Stop the worker processes (the viewer calls this on exit)."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            _POOL.shutdown(wait=False, cancel_futures=True)
            _POOL = None


def _noop():
    return None


def _sweep_chunk(key: str, netlist: dict, settings, swept: list, freqs: list, ports):
    """Worker: simulate one chunk of sweep steps. Returns compact per-step results."""
    global _WORKER
    from .netlist import from_dict
    if _WORKER is None or _WORKER[0] != key:
        _WORKER = (key, from_dict(netlist)[0])
    c = _WORKER[1]
    tones = [c.blocks[b].specs[i] for b, i in swept]
    out = []
    for fs in freqs:
        for t, f in zip(tones, fs):
            t.freq = f
        r = c.simulate(settings)
        inc, outg = r._inc, r._out
        if ports is not None:
            inc = {k: v for k, v in inc.items() if k in ports}
            outg = {k: v for k, v in outg.items() if k in ports}
        out.append((inc, outg, r.iterations, r.converged, r.warnings))
    return out


def _port_keys(circuit: Circuit, refs) -> set:
    keys = set()
    for ref in refs:
        for role in ("out", "in"):
            try:
                keys.add(circuit.port(ref, role))
            except (KeyError, ValueError):
                pass
    return keys


def band_sweep(circuit: Circuit, settings: Settings | None = None, points: int | None = None,
               workers: int | None = 1, ports=None, fractions=None) -> BandSweep:
    """Step every source tone with mode 'sweep' across its band, together, and simulate each step.

    workers: processes to spread the steps over; None uses one per physical core. Above 1, the calling
    script needs the usual `if __name__ == "__main__":` guard (worker processes import it).
    Small sweeps stay in-process whatever `workers` says. ports: keep only these ports'
    spectra in the results (less to send back from the workers).

    The number of steps is `points`, else the largest `points` of the swept tones (default 41).
    At every step all swept tones sit at the same fraction of their bands, so an LO whose band
    is the RF band plus the IF tracks exactly. The steps are spaced evenly, or geometrically
    along the band of the first tone with spacing 'log'. `fractions` (0..1, e.g. from
    output_grid_fractions) places the steps explicitly instead.
    """
    swept = [t for b in circuit.blocks.values() for t in getattr(b, "sweep_specs", lambda: [])()]
    if not swept:
        raise ValueError("no source tone has mode: sweep (give a tone a band and mode: sweep)")
    if fractions is not None:
        frac = np.asarray(fractions, float)
        if len(frac) < 1:
            raise ValueError("no sweep steps to run")
        grids = [t.band[0] + (t.band[1] - t.band[0]) * frac for t in swept]
        steps = [{t.name: float(g[i]) for t, g in zip(swept, grids)} for i in range(len(frac))]
        return BandSweep(circuit, steps, run_steps(circuit, settings, swept, grids, workers, ports))
    n = points or max(len(t.grid()) for t in swept)
    if n < 2:
        raise ValueError("a sweep needs at least 2 points")
    lead = next((t for t in swept if t.spacing == "log"), None)
    if lead:
        frac = (np.geomspace(*lead.band, n) - lead.band[0]) / (lead.band[1] - lead.band[0])
    else:
        frac = np.linspace(0.0, 1.0, n)
    grids = [t.band[0] + (t.band[1] - t.band[0]) * frac for t in swept]
    if lead:
        grids[swept.index(lead)] = np.geomspace(*lead.band, n)   # exact, free of rounding
    steps = [{t.name: float(g[i]) for t, g in zip(swept, grids)} for i in range(n)]
    return BandSweep(circuit, steps, run_steps(circuit, settings, swept, grids, workers, ports))


def output_grid_fractions(circuit: Circuit, anchor, f_out) -> np.ndarray:
    """Sweep positions (0..1 along the swept bands) that put product `anchor` (a vector, or a
    label like 'LO2 - LO1 + RF') at each output frequency in `f_out`.

    The anchor's frequency is |A + B*s| with s the shared position of the swept tones, so each
    target gives s directly; targets outside the swept bands are dropped. The other tones'
    current frequencies go into A, so the points stay on `f_out` whatever they are set to.
    """
    from .spectrum import parse_label
    vec = parse_label(anchor) if isinstance(anchor, str) else anchor
    swept = {t.name: t for b in circuit.blocks.values() for t in getattr(b, "sweep_specs", lambda: [])()}
    now = {t.name: t.freq for b in circuit.blocks.values() for t in getattr(b, "tones", [])}
    a = b = 0.0
    for sym, c in vec:
        if sym in swept:
            a += c * swept[sym].band[0]
            b += c * (swept[sym].band[1] - swept[sym].band[0])
        elif sym in now:
            a += c * now[sym]
        else:
            raise ValueError(f"anchor {vec_label(vec)!r}: no tone {sym!r}")
    if b == 0:
        raise ValueError(f"anchor {vec_label(vec)!r} doesn't move with the swept tones")
    s = [(sign * f - a) / b for f in f_out for sign in (1, -1)]
    return np.unique(np.clip([x for x in s if -1e-9 <= x <= 1 + 1e-9], 0.0, 1.0))


def run_steps(circuit: Circuit, settings, tones: list, grids: list, workers: int | None = 1, ports=None) -> list:
    """One tone-solver Result per step i, with tones[k].freq = grids[k][i]; the tones are restored.
    Spread over worker processes when that pays (see band_sweep for workers and ports)."""
    n = len(grids[0])
    saved = [t.freq for t in tones]
    results = []
    try:
        def run(i):
            for t, g in zip(tones, grids):
                t.freq = float(g[i])
            return circuit.simulate(settings)

        t0 = time.perf_counter()
        results.append(run(0))   # also measures a step: is the rest worth farming out?
        per_step = time.perf_counter() - t0
        workers = min(POOL_SIZE if workers is None else workers, POOL_SIZE, n - 1)
        parallel = None
        if workers > 1 and per_step * (n - 1) > PARALLEL_MIN_S and not _POOL_FAILED:
            try:
                parallel = _parallel(circuit, settings, tones, grids, range(1, n), workers, ports)
            except Exception as e:  # noqa: BLE001  - a broken pool must not break the sweep
                _pool_failed(e)
        results += parallel if parallel is not None else [run(i) for i in range(1, n)]
    finally:
        for t, f in zip(tones, saved):
            t.freq = f
    return results


# ---------------------------------------------------------------- phase noise (tone solver)
@dataclass
class Skirt:
    """Phase-noise sideband around one product at a port, due to one noisy tone.

    The density at product offset o is level(o) + L(|o|/m) + 20 log m: the product's level
    with the noisy tone detuned by o/m (sampled, with the filters' shape in it), plus the
    tone's phase-noise profile (smooth, evaluated exactly).
    """
    vec: tuple
    source: str           # symbol of the noisy tone
    m: int                # |coefficient| of that tone in the product
    f0: float             # the product's frequency
    offsets: np.ndarray   # product offsets sampled, Hz, ascending (both sides)
    level: np.ndarray     # the product's level (dBm) with the tone detuned to each offset
    profile: tuple        # the tone's ((offset, dBc/Hz), ...)

    @property
    def label(self) -> str:
        return vec_label(self.vec)

    def density(self, offsets=None) -> np.ndarray:
        """dBm/Hz at product offsets (default: the sampled ones); NaN outside the samples."""
        from .blocks.sources import ToneSpec
        o = self.offsets if offsets is None else np.asarray(offsets, float)
        spec = ToneSpec(0.0, 0.0, phase_noise=self.profile)
        out = np.full(o.shape, np.nan)
        for side in (-1, 1):
            sel = self.offsets * side > 0
            if sel.sum() == 0:
                continue
            xs, ys = self.offsets[sel], self.level[sel]
            want = (o * side > 0) & (o >= xs.min()) & (o <= xs.max())
            for i in np.nonzero(want)[0]:
                pn = spec.pn_dbc(o[i] / self.m)
                if pn is not None:
                    out[i] = np.interp(o[i], xs, ys) + pn + 20 * math.log10(self.m)
        return out


def phase_noise_skirts(circuit: Circuit, ref, settings: Settings | None = None, incident: bool = False,
                       per_side: int = 12, workers: int | None = 1, nominal=None, min_dbm_hz: float = -250.0,
                       refine_db: float = 2.0, rounds: int = 4, max_new: int = 48) -> list[Skirt]:
    """Noise density that the source tones' phase noise puts around every product at a port.

    Each noisy tone is detuned by offsets d (log-spaced over its profile, both sides) and the
    circuit re-solved; a product carrying m times that tone moves by m*d, at whatever level the
    filters on its path give it there. Where a product's level jumps by more than `refine_db`
    between neighbouring offsets (a filter edge), midpoints are added, for up to `rounds`
    rounds of at most `max_new` solves. A blocker's converted noise is thus followed through
    the IF filter: reciprocal mixing. Valid while the phase noise is a small perturbation.
    """
    noisy = [t for b in circuit.blocks.values() for t in getattr(b, "specs", [])
             if t.phase_noise and not t.mute and t.mode != "wideband"]
    if not noisy:
        return []
    nominal = nominal or circuit.simulate(settings)
    nom = {t.vec: t.freq for t in (nominal.incident(ref) if incident else nominal[ref])}
    out = []
    for t in noisy:
        lo, hi = t.phase_noise[0][0], t.phase_noise[-1][0]
        offs = np.geomspace(lo, hi, per_side) if hi > lo else np.array([lo])
        new = sorted(np.concatenate([-offs, offs]).tolist())
        seen: dict = {}   # vec -> {detune: (level, |m|, product offset)}
        for rnd in range(rounds + 1):
            for d, r in zip(new, run_steps(circuit, settings, [t], [t.freq + np.array(new)], workers, [ref])):
                for p in (r.incident(ref) if incident else r[ref]):
                    c = vec_coeff(p.vec, t.name)
                    if c and p.vec in nom:
                        seen.setdefault(p.vec, {})[d] = (p.power_dbm, abs(c), p.freq - nom[p.vec])
            if rnd == rounds:
                break
            want = set()
            for v, smp in seen.items():
                ds = sorted(smp)
                for a, b in zip(ds, ds[1:]):
                    (pa, m, _), (pb, _, _) = smp[a], smp[b]
                    if a * b <= 0 or abs(pa - pb) <= refine_db or abs(b - a) < max(10.0, 2e-3 * abs(a)):
                        continue
                    la, lb = t.pn_dbc(a), t.pn_dbc(b)
                    if max(pa, pb) + max(x for x in (la, lb, -400) if x is not None) + 20 * math.log10(m) >= min_dbm_hz:
                        want.add((a + b) / 2)
            new = sorted(want)[:max_new]
            if not new:
                break
        for v, smp in seen.items():
            rows = sorted(smp.values(), key=lambda x: x[2])
            out.append(Skirt(v, t.name, rows[0][1], nom[v], np.array([x[2] for x in rows]),
                             np.array([x[0] for x in rows]), t.phase_noise))
    return out


def _parallel(circuit, settings, swept, grids, idx, workers, ports) -> list:
    from .circuit import Result
    from .netlist import to_dict
    netlist = to_dict(circuit)
    key = json.dumps(netlist, sort_keys=True, default=str)
    where = []
    for t in swept:   # address each swept tone as (block, index) in the worker's own copy
        where += [(n, b.specs.index(t)) for n, b in circuit.blocks.items() if t in getattr(b, "specs", [])]
    st = settings or Settings()
    keep = _port_keys(circuit, ports) if ports is not None else None
    idx = list(idx)
    size = -(-len(idx) // workers)
    chunks = [idx[k:k + size] for k in range(0, len(idx), size)]
    futs = [_pool().submit(_sweep_chunk, key, netlist, st, where,
                           [[float(g[i]) for g in grids] for i in chunk], keep) for chunk in chunks]
    out = []
    for f in futs:
        for inc, outg, it, conv, warn in f.result():
            out.append(Result(circuit, inc, outg, it, conv, warn, st))
    return out
