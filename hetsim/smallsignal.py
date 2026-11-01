"""Noise figure and port match: small-signal analyses around a tone-solver operating point.

Both trace a small wave backwards from one port at one frequency. Each block says what feeds
each of its ports (Block.transfers: leakage, reverse paths, every m x 1 mixer response) and a
port with a `vswr` feeds itself. The (port, frequency) pairs reached form a linear system, so
loops are solved exactly rather than iterated, and the solution gives the power gain from every
point in the circuit to the port traced. The tone solver supplies the operating point: LO
frequencies and compressed amplifier gain, with any blockers in the netlist present.

Noise: every port emits what it would at T0 (Block.noise_emission), so a passive circuit
delivers exactly kT0 whatever it contains. The noise figure is single-sideband, the IEEE
definition: noise from the input termination at the image and spur responses counts against the
receiver, which is why an unfiltered image costs 3 dB.

Match: the same trace ends where it started. A port's reflection is the gain back to itself
with its wire cut. Adding bounces in power gives the typical value, adding them in amplitude the
worst case, and adding them with the phases of lines and filters the actual complex reflection.

Not included: LO wideband noise converted by the mixer (phase noise has analysis.phase_noise_skirts),
and shot or flicker noise beyond what nf_db says.
"""
from __future__ import annotations

import copy
import heapq
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import spsolve

from .circuit import Circuit, Settings
from .spectrum import parse_label, vec_coeff, vec_label, vec_neg
from .units import KT0_DBM_HZ, fmt_freq, mw_to_dbm, parse_freq

T0 = 290.0
STIMULUS_DBM = -150.0     # level given to a muted stimulus tone: never disturbs the operating point
SAME_F_HZ = 1.0           # a response this close to the frequency sought is that frequency
REFINE_DB = 1.0           # band integration: add points where gain or noise moves more than this between two
MIN_BAND_STEP_HZ = 2.0    # ...down to this spacing
MAX_BAND_POINTS = 600     # ...and up to this many in a band


# ---------------------------------------------------------------- the trace
def path_gain(circuit: Circuit, nominal, ref, f_out: float, source_key, f_in: float) -> float:
    """Power gain from a source port at f_in to the wave leaving `ref` at f_out; 0 if no path."""
    if ref == source_key:
        return 1.0 if abs(f_in - f_out) <= SAME_F_HZ else 0.0
    tr = _Trace(circuit, nominal, ref, f_out)
    return float(sum(tr.x[k] for k, (p, f) in enumerate(tr.nodes) if p == source_key and abs(f - f_in) <= SAME_F_HZ))


class _Trace:
    """Gain from every (port, frequency) that reaches `root`, by best-first search and a sparse solve.

    x[i] is the power gain to the root from the wave leaving node i. terms holds the same for
    waves entering at an unwired or cut port: (port, freq, gain).
    """

    def __init__(self, circuit: Circuit, nominal, root, f, amplitude=False, cut=None,
                 prune=1e-14, max_nodes=20000):
        self.c, self.amplitude, self.cut = circuit, amplitude, cut
        ops = {}
        for n, b in circuit.blocks.items():
            ops[n] = b.operating_point({p: nominal.incident((n, p)) for p in b.ports})
        self.ops = ops
        self.nodes: list = [(root, float(f))]
        index = {(root, round(float(f), 3)): 0}
        best = [1.0]
        expanded = [False]
        edges, term_edges = [], []
        heap = [(-1.0, 0)]
        top = 1.0
        self.truncated = False
        while heap:
            w, v = heapq.heappop(heap)
            w = -w
            if expanded[v] or w < prune * top:
                continue
            if len(self.nodes) >= max_nodes:
                self.truncated = True
                break
            expanded[v] = True
            (name, port), fv = self.nodes[v]
            b = circuit.blocks[name]
            paths = list(b.transfers(port, fv, ops[name]))
            g2 = float(b.reflection(port, fv))
            if g2 > 0:
                paths.append((port, fv, g2))
            for q, fi, g in paths:
                g = math.sqrt(g) if amplitude else g
                if g <= 0:
                    continue
                inp = (name, q)
                peer = circuit.wires.get(inp)
                if peer is None or inp == cut:
                    term_edges.append((inp, fi, v, g))
                    continue
                wu = w * g
                top = max(top, wu)
                if wu < prune * top:
                    continue
                key = (peer, round(fi, 3))
                u = index.get(key)
                if u is None:
                    u = index[key] = len(self.nodes)
                    self.nodes.append((peer, fi))
                    best.append(0.0)
                    expanded.append(False)
                edges.append((u, v, g))
                if wu > best[u] and not expanded[u]:
                    best[u] = wu
                    heapq.heappush(heap, (-wu, u))
        n = len(self.nodes)
        if edges:
            us, vs, gs = zip(*edges)
            m = csc_matrix((gs, (us, vs)), shape=(n, n))   # duplicates sum, as parallel paths should
            e = np.zeros(n)
            e[0] = 1.0
            self.x = np.atleast_1d(spsolve((identity(n, format="csc") - m).tocsc(), e))
        else:
            self.x = np.array([1.0])
        # a loop with gain of one or more has no steady state; the solution betrays it by going negative
        self.diverged = not np.all(np.isfinite(self.x)) or bool(np.any(self.x < -1e-9))
        self.expanded = expanded
        self.terms = [(inp, fi, self.x[v] * g) for inp, fi, v, g in term_edges]


# ---------------------------------------------------------------- noise figure
@dataclass
class NoiseTerm:
    name: str
    share: float         # fraction of the output noise
    dbm_hz: float        # its density at the output, band average


@dataclass
class NoiseFigure:
    """Result of noise_figure(). With a band, nf_db, gain and noise are over the band (F is the ratio of
    the integrated noises); without one they are the spot values at the wanted product."""
    source: str
    tone: str
    port: str
    product: str
    f_in: float
    f_out: float
    nf_db: float
    gain_db: float                 # peak gain of the wanted response
    noise_out_dbm: float           # total noise leaving the port: in the band, or per Hz for a spot figure
    noise_bw_hz: float | None      # noise bandwidth of the wanted response within the band
    floor_dbm: float               # the output noise referred to the input (MDS at 0 dB SNR)
    freqs_out: np.ndarray
    nf_at: np.ndarray              # spot NF across the band
    gain_at: np.ndarray            # wanted-response gain across the band
    budget: list = field(default_factory=list)      # NoiseTerm, largest first
    responses: list = field(default_factory=list)   # (input freq, gain dB) reaching the band centre, strongest first
    warnings: list = field(default_factory=list)

    def table(self) -> str:
        lines = [f"noise figure {self.nf_db:6.2f} dB   {self.source} {self.tone} -> {self.port}, "
                 f"{self.product} ({fmt_freq(self.f_in, 9)} -> {fmt_freq(self.f_out, 9)})",
                 f"gain         {self.gain_db:6.2f} dB"]
        if self.noise_bw_hz:
            lines += [f"noise out    {self.noise_out_dbm:6.1f} dBm in {fmt_freq(self.noise_bw_hz, 4)} noise bandwidth",
                      f"input floor  {self.floor_dbm:6.1f} dBm"]
        else:
            lines += [f"noise out    {self.noise_out_dbm:6.1f} dBm/Hz", f"input floor  {self.floor_dbm:6.1f} dBm/Hz"]
        lines += ["", f"{'share':>7}  {'dBm/Hz':>8}  noise at the output, from"]
        lines += [f"{100 * t.share:6.1f}%  {t.dbm_hz:8.1f}  {t.name}" for t in self.budget if t.share >= 5e-4]
        lines += ["", f"{'input freq':>16}  {'gain dB':>8}  input responses reaching the output"]
        lines += [f"{fmt_freq(f, 9):>16}  {g:8.2f}" for f, g in self.responses[:8]]
        if self.warnings:
            lines += ["", "warnings:"] + self.warnings
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("source", "tone", "port", "product", "f_in", "f_out", "nf_db", "gain_db",
                                           "noise_out_dbm", "noise_bw_hz", "floor_dbm", "warnings")}
        d["budget"] = [{"name": t.name, "share": t.share, "dbm_hz": t.dbm_hz} for t in self.budget]
        d["responses"] = [{"freq": f, "gain_db": g} for f, g in self.responses]
        return d


class _stimulus:
    """The tone to test with; a muted one is present, far too weak to matter, while inside."""

    def __init__(self, src, tone):
        self.src = src
        if isinstance(tone, str):
            match = [t for t in src.specs if t.name == tone]
            if not match:
                raise ValueError(f"source {src.name!r} has no tone {tone!r}; it has {[t.name for t in src.specs]}")
            self.spec = match[0]
        else:
            self.spec = src.specs[tone]
        if self.spec.mode == "wideband":
            raise ValueError(f"tone {self.spec.name!r} is a wideband comb; a noise figure needs a single tone")
        self.saved = None

    def __enter__(self):
        if self.spec not in self.src.tones:
            self.saved = self.spec.power
            self.spec.power = STIMULUS_DBM
            self.src.tones.append(self.spec)
        return self.spec

    def __exit__(self, *exc):
        if self.saved is not None:
            self.src.tones.remove(self.spec)
            self.spec.power = self.saved


def _settings(settings):
    st = copy.copy(settings) if settings else Settings()
    st.floor_dbm = min(st.floor_dbm, -300.0)   # a weak stimulus must stay visible
    return st


def _moved(a: dict, b: dict, limit_db: float) -> bool:
    """Has the gain or the total noise changed by more than limit_db between two solved points?"""
    for key in ("gain", "total"):
        x, y = a[key], b[key]
        if (x > 0) != (y > 0) or (x > 0 and abs(10 * math.log10(x / y)) > limit_db):
            return True
    return False


def _trapz_weights(f: np.ndarray) -> np.ndarray:
    if len(f) == 1:
        return np.ones(1)
    d = np.diff(f)
    return np.concatenate([[d[0]], d[:-1] + d[1:], [d[-1]]]) / 2


def noise_figure(circuit: Circuit, source: str, port, band=None, tone=0, settings: Settings | None = None,
                 points: int = 21, product=None, nominal=None) -> NoiseFigure:
    """Noise figure from source block `source` to the wave leaving `port`.

    `tone` (index or name) picks the signal; the product of it reaching the port, within `band` if
    given, is the wanted response. A muted tone works: the rest of the netlist then sets the operating
    point. With `band` (fmin, fmax) the noise is integrated over `points` frequencies and more wherever
    a filter edge calls for them, otherwise it is the spot figure at the product. `product` (a label like
    'LO - RF', or a vector) names the wanted response, where the strongest isn't it. `nominal`: a tone
    solve with the stimulus present, to save repeating it. Raises ValueError when the tone doesn't reach the port.
    """
    band = (parse_freq(band[0]), parse_freq(band[1])) if band else None
    src = circuit[source]
    ref = circuit.port(port, "out")
    src_key = (source, src.default_out)
    with _stimulus(src, tone) as spec:
        nominal = nominal or circuit.simulate(_settings(settings))
        wanted = [t for t in nominal[ref].band(*(band or (None, None))) if abs(vec_coeff(t.vec, spec.name)) == 1]
        if product is not None:
            v = parse_label(product) if isinstance(product, str) else tuple(product)
            wanted = [t for t in wanted if t.vec in (v, vec_neg(v))]
        if not wanted:
            where = f" in {fmt_freq(band[0])} to {fmt_freq(band[1])}" if band else ""
            raise ValueError(f"no product of {spec.name} at {fmt_freq(spec.freq)} reaches {ref[0]}.{ref[1]}{where}; "
                             "retune the tone or its LOs")
        best = max(wanted, key=lambda t: t.power_dbm)
        c = vec_coeff(best.vec, spec.name)
        warn = [f"[{b}] {m}" for b, ms in nominal.warnings.items() for m in ms]
        f_centre = best.freq

        def solve(fo: float) -> dict:
            """Everything at one output frequency: signal, per-contributor noise, input responses."""
            tr = _Trace(circuit, nominal, ref, fo)
            if tr.diverged:
                raise ValueError(f"a feedback loop has gain of one or more at {fmt_freq(fo)}: the circuit "
                                 "has no steady state there, so no noise figure")
            if tr.truncated and not any("truncated" in w for w in warn):
                warn.append(f"trace truncated at {len(tr.nodes)} points: noise may be understated")
            f_sig = spec.freq + c * (fo - f_centre)
            pt = {"sig": 0.0, "gain": 0.0, "own": {}, "resp": {}}

            def book(name, x):
                pt["own"][name] = pt["own"].get(name, 0.0) + x

            for k, ((name, q), f) in enumerate(tr.nodes):
                if not tr.expanded[k] or tr.x[k] <= 0:
                    continue
                e = circuit.blocks[name].noise_emission(q, f, tr.ops[name])
                if (name, q) == src_key:
                    if abs(f - f_sig) <= SAME_F_HZ:
                        pt["sig"] += tr.x[k] * e
                        pt["gain"] += tr.x[k]
                    else:
                        book(f"{name}: input termination at other responses", tr.x[k] * e)
                    pt["resp"][round(f, 1)] = tr.x[k]
                else:
                    book(name, tr.x[k] * e)
            for (name, q), f, x in tr.terms:
                book(f"{name}.{q} termination", x)
            pt["total"] = sum(pt["own"].values()) + pt["sig"]
            return pt

        c = vec_coeff(best.vec, spec.name)
        pts = {float(f): solve(float(f)) for f in (np.linspace(band[0], band[1], points) if band else [best.freq])}
        if band:
            if band[0] <= best.freq <= band[1]:
                pts.setdefault(best.freq, solve(best.freq))
            # a coarse grid misses or misjudges a filter narrower than the band: refine where anything moves
            for _ in range(40):
                fs = sorted(pts)
                new = [(x + y) / 2 for x, y in zip(fs, fs[1:])
                       if y - x > MIN_BAND_STEP_HZ and _moved(pts[x], pts[y], REFINE_DB)]
                if not new or len(pts) + len(new) > MAX_BAND_POINTS:
                    break
                for f in new:
                    pts[f] = solve(f)
        grid = np.array(sorted(pts))
        total = np.array([pts[f]["total"] for f in grid])
        sig = np.array([pts[f]["sig"] for f in grid])
        gain = np.array([pts[f]["gain"] for f in grid])
        names = {n for f in grid for n in pts[f]["own"]}
        own = {n: np.array([pts[f]["own"].get(n, 0.0) for f in grid]) for n in names}
        responses = pts[float(grid[np.argmin(np.abs(grid - best.freq))])]["resp"]
    w = _trapz_weights(grid)
    if not sig.dot(w) > 0:
        raise ValueError(f"the path from {spec.name} to {ref[0]}.{ref[1]} is too weak to trace")
    f_lin = total.dot(w) / sig.dot(w)
    if any(getattr(b, "nf", 0) is None and getattr(b, "gain", 0) > 0 for b in circuit.blocks.values()):
        names = [n for n, b in circuit.blocks.items() if getattr(b, "nf", 0) is None and getattr(b, "gain", 0) > 0]
        warn.append(f"{', '.join(names)}: no nf_db, so taken as noiseless")
    g_peak = float(gain.max())
    nf_db = 10 * math.log10(f_lin)
    sum_w = float(w.sum()) if band else 1.0
    noise = total.dot(w) if band else total[0]
    bw = float(gain.dot(w) / g_peak) if band else None
    budget = [NoiseTerm(f"{source}: input termination at the wanted response", float(sig.dot(w) / total.dot(w)),
                        KT0_DBM_HZ + mw_to_dbm(float(sig.dot(w)) / sum_w))]
    budget += [NoiseTerm(n, float(v.dot(w) / total.dot(w)), KT0_DBM_HZ + mw_to_dbm(float(v.dot(w)) / sum_w))
               for n, v in own.items()]
    budget.sort(key=lambda t: -t.share)
    with np.errstate(divide="ignore"):
        nf_at = 10 * np.log10(total / sig)
        gain_at = 10 * np.log10(gain)
    resp = sorted(((f, 10 * math.log10(g)) for f, g in responses.items() if g > 0), key=lambda r: -r[1])
    out_dbm = KT0_DBM_HZ + mw_to_dbm(float(noise))
    return NoiseFigure(source, spec.name, f"{ref[0]}.{ref[1]}", vec_label(best.vec), spec.freq, best.freq, nf_db,
                       10 * math.log10(g_peak), out_dbm, bw, out_dbm - 10 * math.log10(g_peak), grid, nf_at, gain_at,
                       budget, resp, warn)


@dataclass
class NoiseSweep:
    """noise_sweep(): one NoiseFigure per step, None where the tone didn't reach the band."""
    steps: list
    results: list
    f_in: np.ndarray
    nf_db: np.ndarray
    gain_db: np.ndarray

    def table(self) -> str:
        lines = [f"{'stimulus':>16}  {'NF dB':>7}  {'gain dB':>8}"]
        for f, nf, g in zip(self.f_in, self.nf_db, self.gain_db):
            lines.append(f"{fmt_freq(f, 9):>16}  {nf:7.2f}  {g:8.2f}")
        return "\n".join(lines)


def noise_sweep(circuit: Circuit, source: str, port, band=None, tone=0, settings: Settings | None = None,
                points: int | None = None, npoints: int = 9) -> NoiseSweep:
    """Noise figure across the band: every tone with mode 'sweep' steps as in band_sweep, and the stimulus
    with them if it has a band of its own, so give an LO and the stimulus tracking bands.
    `points`: sweep steps; `npoints`: frequencies within the output band at each step."""
    src = circuit[source]
    with _stimulus(src, tone) as spec:
        swept = [t for b in circuit.blocks.values() for t in getattr(b, "sweep_specs", lambda: [])()]
        if spec not in swept and spec.band:
            swept.append(spec)
        if not swept:
            raise ValueError("no tone has mode: sweep (give the stimulus and its LO a band and mode: sweep)")
        n = points or max(len(t.grid()) for t in swept if t.mode == "sweep" or t is spec)
        saved = [t.freq for t in swept]
        results = []
        try:
            for s in np.linspace(0.0, 1.0, n):
                for t in swept:
                    t.freq = t.band[0] + (t.band[1] - t.band[0]) * float(s)
                try:
                    results.append(noise_figure(circuit, source, port, band, spec.name, settings, npoints))
                except ValueError:
                    results.append(None)
            steps = [{t.name: t.band[0] + (t.band[1] - t.band[0]) * float(s) for t in swept}
                     for s in np.linspace(0.0, 1.0, n)]
            f_in = np.array([st[spec.name] if spec.name in st else spec.freq for st in steps])
        finally:
            for t, f in zip(swept, saved):
                t.freq = f
    nan = float("nan")
    return NoiseSweep(steps, results, f_in, np.array([r.nf_db if r else nan for r in results]),
                      np.array([r.gain_db if r else nan for r in results]))


# ---------------------------------------------------------------- port match
def _phased_reflection(circuit: Circuit, nominal, ref, f: float, prune=1e-9, max_nodes=20000) -> complex:
    """Complex reflection looking into `ref` at f, with its wire cut: every path back, with its phase.

    As in _Trace, but gains are complex, and a lower-sideband conversion reverses phase. That isn't
    linear in a complex wave, so a node remembers whether the gain it carries applies to the wave or
    to its conjugate; two reversals in a round trip cancel.
    """
    arr = np.array([f])
    ops = {n: b.operating_point({p: nominal.incident((n, p)) for p in b.ports}) for n, b in circuit.blocks.items()}
    root = (ref, round(f, 3), False)
    index = {root: 0}
    nodes, heap, done = [(ref, f, False)], [(-1.0, 0)], set()
    rows, cols, vals, terms = [], [], [], []
    while heap and len(nodes) < max_nodes:
        w, v = heapq.heappop(heap)
        w = -w
        if v in done or w < prune:
            continue
        done.add(v)
        (name, port), fv, conj_v = nodes[v]
        b = circuit.blocks[name]
        at = np.array([fv])
        paths = [(q, fv, complex(g[0]), False) for q, g in b.linear_c(port, at, ops[name])]
        if port in b.gamma2:
            paths.append((port, fv, complex(np.ravel(b.gamma_c(port, at))[0]), False))
        paths += b.conversions_c(port, fv, ops[name])
        for q, fi, g, conjugates in paths:
            if g == 0:
                continue
            # X_v carries a gain on w_v, or on conj(w_v). Writing w_v = g*w_u, or g*conj(w_u), for the wave u:
            coeff = g.conjugate() if conj_v else g
            conj_u = conj_v != conjugates
            inp = (name, q)
            peer = circuit.wires.get(inp)
            if peer is None or inp == ref:
                terms.append((inp, fi, conj_u, v, coeff))
                continue
            key = (peer, round(fi, 3), conj_u)
            u = index.get(key)
            if u is None:
                u = index[key] = len(nodes)
                nodes.append((peer, fi, conj_u))
            rows.append(u)
            cols.append(v)
            vals.append(coeff)
            if u not in done:
                heapq.heappush(heap, (-w * abs(g), u))
    n = len(nodes)
    e = np.zeros(n, complex)
    e[0] = 1.0
    if rows:
        m = csc_matrix((vals, (rows, cols)), shape=(n, n), dtype=complex)
        x = np.atleast_1d(spsolve((identity(n, format="csc", dtype=complex) - m).tocsc(), e))
    else:
        x = e
    return complex(sum(x[v] * coeff for inp, fi, conj_u, v, coeff in terms
                       if inp == ref and abs(fi - f) <= SAME_F_HZ and not conj_u))


def _vswr(gamma: float) -> float:
    return (1 + gamma) / (1 - gamma) if gamma < 1 else math.inf


@dataclass
class PortMatch:
    """Looking into a port with its wire cut. gamma is typical (bounces add in power), gamma_worst the
    limit when they all add in phase, and gamma_c the actual complex reflection: the delays of lines and
    the phases of filters and of measured S-parameters are in it. Without those it can't know better
    than zero phase, and it then equals the worst case."""
    port: str
    freq: float
    gamma: float
    gamma_worst: float
    gamma_c: complex = 0j

    @property
    def vswr_phased(self) -> float:
        return _vswr(abs(self.gamma_c))

    @property
    def vswr(self) -> float:
        return _vswr(self.gamma)

    @property
    def vswr_worst(self) -> float:
        return _vswr(self.gamma_worst)

    @property
    def return_loss_db(self) -> float:
        return -20 * math.log10(self.gamma) if self.gamma > 0 else math.inf

    @property
    def mismatch_loss_db(self) -> float:
        """Power not accepted by the port, against a matched source."""
        return -10 * math.log10(1 - self.gamma ** 2) if self.gamma < 1 else math.inf


def port_match(circuit: Circuit, port, freq, settings: Settings | None = None, nominal=None) -> PortMatch:
    """What a wave at `freq` sees looking into `port`, including everything behind it. Reflections of
    later ports count: through a pad at twice its loss, through an amplifier at its gain less its
    reverse isolation. A mixer contributes through its conversions, so what is on its far side matters too."""
    ref = circuit.port(port, "in")
    f = parse_freq(freq)
    nominal = nominal or circuit.simulate(_settings(settings))
    back = []
    for amplitude in (False, True):
        tr = _Trace(circuit, nominal, ref, f, amplitude=amplitude, cut=ref)
        x = sum(x for p, fi, x in tr.terms if p == ref and abs(fi - f) <= SAME_F_HZ)
        back.append(math.inf if tr.diverged else float(x))
    phased = _phased_reflection(circuit, nominal, ref, f) if not math.isinf(back[1]) else complex(math.inf)
    return PortMatch(f"{ref[0]}.{ref[1]}", f, math.sqrt(back[0]), back[1], phased)


@dataclass
class MatchSweep:
    port: str
    freqs: np.ndarray
    vswr: np.ndarray
    vswr_worst: np.ndarray
    return_loss_db: np.ndarray
    vswr_phased: np.ndarray = None

    def table(self) -> str:
        lines = [f"looking into {self.port}",
                 f"{'frequency':>16}  {'VSWR':>7}  {'RL dB':>7}  {'worst VSWR':>10}  {'phased VSWR':>11}"]
        for f, v, rl, w, c in zip(self.freqs, self.vswr, self.return_loss_db, self.vswr_worst, self.vswr_phased):
            lines.append(f"{fmt_freq(f, 9):>16}  {v:7.3f}  {rl:7.2f}  {w:10.3f}  {c:11.3f}")
        return "\n".join(lines)


def match_sweep(circuit: Circuit, port, freqs, settings: Settings | None = None) -> MatchSweep:
    """port_match at each of `freqs`."""
    freqs = np.array([parse_freq(f) for f in freqs], dtype=float)
    nominal = circuit.simulate(_settings(settings))
    ms = [port_match(circuit, port, f, nominal=nominal) for f in freqs]
    return MatchSweep(ms[0].port, freqs, np.array([m.vswr for m in ms]), np.array([m.vswr_worst for m in ms]),
                      np.array([m.return_loss_db for m in ms]), np.array([m.vswr_phased for m in ms]))


@dataclass
class Interface:
    """One wire: what each end sees looking into its own port, and the ripple they cause between them."""
    a: PortMatch
    b: PortMatch
    freq: float
    product: str

    @property
    def ripple_db(self) -> tuple[float, float]:
        """Gain change from the two reflections, (low, high): they beat as the frequency moves."""
        k = self.a.gamma * self.b.gamma
        return (20 * math.log10(1 - k) if k < 1 else -math.inf, 20 * math.log10(1 + k))


@dataclass
class MismatchReport:
    interfaces: list

    def table(self) -> str:
        lines = [f"{'wire':<26} {'frequency':>14} {'VSWR a':>7} {'VSWR b':>7} {'ripple dB':>15}  product"]
        for i in self.interfaces:
            lo, hi = i.ripple_db
            wire = f"{i.a.port} - {i.b.port}"
            lines.append(f"{wire:<26} {fmt_freq(i.freq, 7):>14} {i.a.vswr:7.3f} {i.b.vswr:7.3f} "
                         f"{lo:+7.3f}/{hi:+6.3f}  {i.product}")
        return "\n".join(lines)


def _lo_wires(circuit: Circuit) -> set:
    """Ports on wires that drive a mixer's LO port: from each one, back to the sources."""
    seen, stack = set(), [(n, "lo") for n, b in circuit.blocks.items() if "lo" in b.ports]
    while stack:
        peer = circuit.wires.get(stack.pop())
        if peer is None or peer in seen:
            continue
        seen.add(peer)
        seen.add(circuit.wires[peer])
        b = circuit[peer[0]]
        if "lo" not in b.ports:
            stack += [(peer[0], p) for p in b.ports if p != peer[1] and p not in b.controls]
    return seen


def mismatch_report(circuit: Circuit, settings: Settings | None = None, tone: str | None = None) -> MismatchReport:
    """Every RF wire at the frequency it carries: its strongest product, or the strongest one
    involving `tone` where the wire has one (so a signal is followed down a chain past LO leakage).
    Wires driving a mixer's LO port stay at the LO."""
    nominal = circuit.simulate(_settings(settings))
    lo = _lo_wires(circuit)
    rows = []
    for a, b in circuit.wire_list():
        pa, pb = tuple(a.split(".", 1)), tuple(b.split(".", 1))
        if pa[1] in circuit[pa[0]].controls:
            continue
        waves = list(nominal[pa]) + list(nominal[pb])
        if tone and pa not in lo:
            waves = [t for t in waves if vec_coeff(t.vec, tone)] or waves
        if not waves:
            continue
        t = max(waves, key=lambda x: x.power_dbm)
        rows.append(Interface(port_match(circuit, pa, t.freq, nominal=nominal),
                              port_match(circuit, pb, t.freq, nominal=nominal), t.freq, t.label))
    return MismatchReport(rows)
