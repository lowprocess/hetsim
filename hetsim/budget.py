"""Receiver budget: the signal path stage by stage, and the dynamic range it adds up to.

The gains and noise come from the small-signal trace, so filter shapes, image noise, mismatches
and compression are all in them. Intercepts are cascaded by hand on those gains, because the tone
solver adds IM from different stages in power, while the usual cascade formula is for in-phase
addition (the worst case, and what a time-domain two-tone run shows):

    1 / IIP3 = sum over stages of 1 / IIP3_i, each referred to the input (in mW)

With a blocker `offset`, the two tones sit at f + offset and f + 2*offset and each stage is referred
with the gain they actually have up to it, so a roofing filter ahead of a stage takes it out of the
sum. That is the comparison between architectures that an in-band figure can't make.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

from .circuit import Circuit, Settings
from .smallsignal import _lo_wires, _settings, _stimulus, noise_figure, path_gain
from .spectrum import vec_coeff, vec_label
from .units import dbm_to_mw, fmt_freq, mw_to_dbm, parse_freq


@dataclass
class Stage:
    block: str
    port: str              # the port the signal leaves by
    freq: float            # the signal's frequency there
    gain_db: float         # this stage's own gain
    cum_gain_db: float
    cum_nf_db: float
    cum_iip3_dbm: float    # at the input, to this point; inf: nothing nonlinear yet
    cum_ip1db_dbm: float
    iip3_dbm: float | None  # the stage's own, at its input
    op1db_dbm: float | None  # the stage's own, at its output


@dataclass
class Budget:
    name: str
    source: str
    tone: str
    port: str
    product: str
    f_in: float
    f_out: float
    bw_hz: float | None         # channel asked for
    noise_bw_hz: float | None   # what the filters leave of it
    offset_hz: float | None
    snr_db: float
    stages: list
    gain_db: float
    nf_db: float
    floor_dbm: float            # in the noise bandwidth, or per Hz without a channel
    sensitivity_dbm: float
    iip3_dbm: float
    ip1db_dbm: float            # in-band
    blocking_ip1db_dbm: float   # against a blocker at the offset
    sfdr_db: float
    bdr_compression_db: float
    bdr_phase_noise_db: float | None   # reciprocal mixing; None: no profile covers the offset
    rejection_db: float | None         # to the strongest other m x 1 response
    rejection_at: float | None
    limits: dict = field(default_factory=dict)     # figure -> what sets it
    warnings: list = field(default_factory=list)
    iip2_dbm: float = math.inf  # the lowest mixer's, referred to the input: what lands at baseband in zero-IF

    @property
    def bdr_db(self) -> float:
        return min(self.bdr_compression_db, self.bdr_phase_noise_db if self.bdr_phase_noise_db is not None else math.inf)

    @property
    def unit(self) -> str:
        return "dBm" if self.bw_hz else "dBm/Hz"

    def table(self) -> str:
        bw = f", channel {fmt_freq(self.bw_hz, 4)}" if self.bw_hz else ", per Hz (give a channel bandwidth)"
        tones = f"blocker offset {fmt_freq(self.offset_hz, 4)}" if self.offset_hz else "in-band tones"
        lines = [f"{self.name or 'receiver'}: {self.source} {self.tone} -> {self.port}, {self.product} "
                 f"({fmt_freq(self.f_in, 9)} -> {fmt_freq(self.f_out, 9)}){bw}; {tones}", "",
                 f"{'stage':<14} {'frequency':>12} {'gain':>7} {'cum gain':>8} {'cum NF':>7} {'cum IIP3':>8} {'cum IP1dB':>9}"]
        for s in self.stages:
            lines.append(f"{s.block:<14} {fmt_freq(s.freq, 5):>12} {s.gain_db:7.2f} {s.cum_gain_db:8.2f} {s.cum_nf_db:7.2f} "
                         f"{_lvl(s.cum_iip3_dbm):>8} {_lvl(s.cum_ip1db_dbm):>9}")
        lines += ["", *self._summary()]
        if self.warnings:
            lines += ["", "warnings:", *self.warnings]
        return "\n".join(lines)

    def rows(self) -> list[tuple[str, str]]:
        """The headline figures, as (label, text)."""
        rej = f"{self.rejection_db:.1f} dB at {fmt_freq(self.rejection_at, 6)}" if self.rejection_db is not None else "-"
        return [("peak gain", f"{self.gain_db:.2f} dB"),
                ("noise figure", f"{self.nf_db:.2f} dB"),
                ("noise bandwidth", fmt_freq(self.noise_bw_hz, 4) if self.noise_bw_hz else "-"),
                ("input floor (MDS)", f"{self.floor_dbm:.1f} {self.unit}"),
                (f"sensitivity at {self.snr_db:g} dB SNR", f"{self.sensitivity_dbm:.1f} {self.unit}"),
                ("IIP3", f"{_lvl(self.iip3_dbm)} dBm"),
                ("IIP2 at the mixers", f"{_lvl(self.iip2_dbm)} dBm"),
                ("input P1dB", f"{_lvl(self.ip1db_dbm)} dBm"),
                ("spur-free dynamic range", _db(self.sfdr_db)),
                ("blocking DR, compression", _db(self.bdr_compression_db)),
                ("blocking DR, phase noise", _db(self.bdr_phase_noise_db) if self.bdr_phase_noise_db is not None else "-"),
                ("worst other response", rej)]

    def _summary(self) -> list[str]:
        out = []
        for label, text in self.rows():
            why = self.limits.get(label)
            out.append(f"{label:<28} {text:<24}" + (f" set by {why}" if why else ""))
        return [ln.rstrip() for ln in out]

    def to_dict(self) -> dict:
        d = {k: _finite(getattr(self, k)) for k in (
            "name", "source", "tone", "port", "product", "f_in", "f_out", "bw_hz", "noise_bw_hz", "offset_hz", "snr_db",
            "gain_db", "nf_db", "floor_dbm", "sensitivity_dbm", "iip3_dbm", "iip2_dbm", "ip1db_dbm", "blocking_ip1db_dbm", "sfdr_db",
            "bdr_compression_db", "bdr_phase_noise_db", "bdr_db", "rejection_db", "rejection_at", "limits", "warnings")}
        d["stages"] = [{k: _finite(v) for k, v in vars(s).items()} for s in self.stages]
        return d


def _finite(v):
    return None if isinstance(v, float) and math.isinf(v) else v


def _lvl(p: float) -> str:
    return "-" if math.isinf(p) else f"{p:.1f}"


def _db(x: float) -> str:
    return "-" if math.isinf(x) else f"{x:.1f} dB"


def compare_budgets(budgets: list) -> str:
    """The headline figures of several budgets side by side."""
    cols = [dict(b.rows()) for b in budgets]
    names = [b.name or f"receiver {i + 1}" for i, b in enumerate(budgets)]
    w = max(24, *(len(n) + 1 for n in names))
    lines = [" " * 28 + "".join(f"{n:>{w}}" for n in names)]
    for label in cols[0]:
        lines.append(f"{label:<28}" + "".join(f"{c[label]:>{w}}" for c in cols))
    for n, b in zip(names, budgets):
        lines += [f"{n}: {w_}" for w_ in b.warnings]
    return "\n".join(lines)


# ---------------------------------------------------------------- the signal path
def _exits(b, port: str) -> list[str]:
    """Where a signal arriving at `port` goes on to."""
    if "lo" in b.ports:
        return {"rf": ["if"], "if": ["rf"]}.get(port, [])
    if "in" in b.ports and "out" in b.ports:
        return ["out"] if port == "in" and "out" not in b.controls else []
    return [p for p in b.ports if p != port and p not in b.controls]


def signal_path(circuit: Circuit, source: str, end=None) -> list[tuple[str, str, str]]:
    """[(block, in port, out port)] from `source` to the port `end`, or to the farthest output reachable."""
    start = (source, circuit[source].default_out)
    prev: dict = {start: None}
    frontier, order = [start], []
    while frontier:
        out = frontier.pop(0)
        peer = circuit.wires.get(out)
        if peer is None:
            continue
        for x in _exits(circuit[peer[0]], peer[1]):
            k = (peer[0], x)
            if k not in prev:
                prev[k] = (out, peer[1])
                order.append(k)
                frontier.append(k)
    if end is None:
        if not order:
            raise ValueError(f"nothing is wired after {source!r}")
        end = order[-1]
    elif end not in prev:
        raise ValueError(f"no signal path from {source!r} to {end[0]}.{end[1]}")
    path = []
    k = end
    while prev[k] is not None:
        last, q = prev[k]
        path.append((k[0], q, k[1]))
        k = last
    return path[::-1]


def default_source(circuit: Circuit) -> str:
    lo = _lo_wires(circuit)
    for n, b in circuit.blocks.items():
        if b.kind == "source" and (n, b.default_out) not in lo:
            return n
    raise ValueError("no source block that isn't an LO")


def _fits(v, final) -> bool:
    """Is product v on the way to `final`? Its terms all appear there, with one overall sign."""
    want = dict(final)
    return any(all(want.get(s) == sign * c for s, c in v) for sign in (1, -1))


def _lo_phase_noise_dbc(circuit: Circuit, nominal, path, offset: float) -> float | None:
    """Total SSB phase noise (dBc/Hz) at `offset` of the LOs of the mixers on the path; None if none covers it."""
    specs = {t.name: t for b in circuit.blocks.values() for t in getattr(b, "specs", [])}
    mw = 0.0
    found = False
    for name, _, _ in path:
        if "lo" not in circuit[name].ports:
            continue
        los = nominal.incident((name, "lo")).strongest(1)
        for sym, m in (los[0].vec if los else ()):
            spec = specs.get(sym)
            pn = spec.pn_dbc(offset / abs(m)) if spec is not None else None
            if pn is not None:
                mw += dbm_to_mw(pn + 20 * math.log10(abs(m)))
                found = True
    return mw_to_dbm(mw) if found else None


# ---------------------------------------------------------------- the budget
def receiver_budget(circuit: Circuit, source: str | None = None, tone=None, port=None, bw=None, offset=None,
                    snr_db: float = 10.0, settings: Settings | None = None, product=None) -> Budget:
    """Stage-by-stage budget of the signal path, and its sensitivity and dynamic range.

    Defaults find the input source, its first tone and the end of the path, so that netlists with
    different block names can be compared. `bw` sets the channel, centred on the product; `offset`
    places the blocker tones for intercepts and blocking. Muted tones work, and tones in the netlist
    set the operating point, so strong ones show as desensitisation.
    """
    source = source or default_source(circuit)
    src = circuit[source]
    src_key = (source, src.default_out)
    bw = parse_freq(bw) if bw is not None else None
    offset = parse_freq(offset) if offset is not None else None
    end = circuit.port(port, "out") if port else None
    path = signal_path(circuit, source, end)
    end = (path[-1][0], path[-1][2])
    if tone is None:
        tone = next((t.name for t in src.specs if t.mode != "wideband"), 0)

    with _stimulus(src, tone) as spec:
        st = _settings(settings)
        nominal = circuit.simulate(copy.copy(st))
        spot = noise_figure(circuit, source, end, None, spec.name, settings, product=product, nominal=nominal)
        final = next(t.vec for t in nominal[end] if vec_label(t.vec) == spot.product)
        band = (spot.f_out - bw / 2, spot.f_out + bw / 2) if bw else None
        whole = noise_figure(circuit, source, end, band, spec.name, settings, product=final, nominal=nominal) if bw else spot

        def gain_to(ref, sig_vec, f_sig, f_in):
            c = vec_coeff(sig_vec, spec.name) if sig_vec else 1
            return path_gain(circuit, nominal, ref, f_sig + c * (f_in - spec.freq), src_key, f_in)

        stages, warn = [], list(whole.warnings)
        inv_iip3 = 0.0          # 1/mW at the input
        terms, p1_terms, block_p1, iip2_terms = {}, {}, {}, {}
        here, vec_here, f_here = src_key, ((spec.name, 1),), spec.freq
        cum_prev = 0.0
        cum_p1 = math.inf
        for name, q, p in path:
            out_ref = (name, p)
            b = circuit[name]
            fits = [t for t in nominal[out_ref] if vec_coeff(t.vec, spec.name) and _fits(t.vec, final)]
            if not fits:
                failed = nominal.warnings.get("<solver>")
                raise ValueError(failed[0] if failed else f"the signal doesn't reach {name}.{p}")
            t = max(fits, key=lambda x: x.power_dbm)
            r = noise_figure(circuit, source, out_ref, None, spec.name, settings, product=t.vec, nominal=nominal)
            iip3, op1 = b.linearity()
            if getattr(b, "iip2", None) is not None and "lo" in b.ports:
                g2_in = gain_to(here, vec_here, f_here, spec.freq)
                if g2_in > 0:
                    iip2_terms[name] = b.iip2 - _db10(g2_in)
            if iip3 is not None:
                g_s = gain_to(here, vec_here, f_here, spec.freq)
                if offset:
                    g1 = gain_to(here, vec_here, f_here, spec.freq + offset)
                    g2 = gain_to(here, vec_here, f_here, spec.freq + 2 * offset)
                else:
                    g1 = g2 = g_s
                if g1 > 0 and g2 > 0 and g_s > 0:
                    # IM at this stage's input: 2(P + G1) + (P + G2) - 2*IIP3; the signal has G_s to it
                    eq = iip3 - (2 * _db10(g1) + _db10(g2) - _db10(g_s)) / 2
                    terms[name] = 1 / dbm_to_mw(eq)
                    inv_iip3 += terms[name]
            if op1 is not None:
                g_out = gain_to(out_ref, t.vec, t.freq, spec.freq)
                g_blk = gain_to(out_ref, t.vec, t.freq, spec.freq + offset) if offset else g_out
                if g_out > 0:
                    p1_terms[name] = op1 + 1 - _db10(g_out)       # OP1dB is the compressed output, so +1
                    cum_p1 = min(cum_p1, p1_terms[name])
                if g_blk > 0:
                    block_p1[name] = op1 + 1 - _db10(g_blk)
            stages.append(Stage(name, p, t.freq, r.gain_db - cum_prev, r.gain_db, r.nf_db,
                                -mw_to_dbm(inv_iip3) if inv_iip3 else math.inf, cum_p1, iip3, op1))
            cum_prev = r.gain_db
            here, vec_here, f_here = out_ref, t.vec, t.freq

        pn = _lo_phase_noise_dbc(circuit, nominal, path, offset) if offset else None

    iip3_total = -mw_to_dbm(inv_iip3) if inv_iip3 else math.inf
    ip1 = min(p1_terms.values(), default=math.inf)
    ip1_blk = min(block_p1.values(), default=math.inf)
    floor = whole.floor_dbm
    n_bw = whole.noise_bw_hz
    limits = {}
    noisy = [t for t in whole.budget if "wanted response" not in t.name]
    if noisy:
        limits["noise figure"] = noisy[0].name
    if terms:
        limits["IIP3"] = max(terms, key=terms.get)
    if iip2_terms:
        limits["IIP2 at the mixers"] = min(iip2_terms, key=iip2_terms.get)
    if p1_terms:
        limits["input P1dB"] = min(p1_terms, key=p1_terms.get)
    if block_p1:
        limits["blocking DR, compression"] = min(block_p1, key=block_p1.get)
    rej = rej_at = None
    others = [(f, g) for f, g in spot.responses if abs(f - spot.f_in) > 1.0]
    if others:
        rej_at, g = max(others, key=lambda x: x[1])
        rej = spot.gain_db - g
    bdr_pn = -(pn + 10 * math.log10(n_bw or 1.0)) if pn is not None else None
    if any(math.isinf(x) for x in (iip3_total,)):
        warn.append("no stage on the path has an IP3: spur-free dynamic range not available")
    return Budget(circuit.name, source, spec.name, f"{end[0]}.{end[1]}", vec_label(final), spec.freq, spot.f_out, bw, n_bw,
                  offset, snr_db, stages, whole.gain_db, whole.nf_db, floor, floor + snr_db, iip3_total, ip1, ip1_blk,
                  (2 / 3) * (iip3_total - floor), ip1_blk - floor, bdr_pn, rej, rej_at, limits, warn,
                  min(iip2_terms.values(), default=math.inf))


# figure -> (label, unit), for parameter sweeps and their charts
METRICS = {"nf_db": ("Noise figure", "dB"), "gain_db": ("Peak gain", "dB"), "floor_dbm": ("Input floor", "dBm"),
           "iip3_dbm": ("IIP3", "dBm"), "ip1db_dbm": ("Input P1dB", "dBm"), "sfdr_db": ("Spur-free dynamic range", "dB"),
           "bdr_db": ("Blocking dynamic range", "dB")}
SHORT = {"nf_db": "NF", "gain_db": "gain", "floor_dbm": "floor", "iip3_dbm": "IIP3", "ip1db_dbm": "IP1dB",
         "sfdr_db": "SFDR", "bdr_db": "BDR"}


@dataclass
class BudgetSweep:
    """budget_sweep(): one Budget per value, None where the circuit gave no answer there."""
    param: str
    values: list
    budgets: list

    def series(self, metric: str) -> list:
        return [getattr(b, metric) if b is not None else math.nan for b in self.budgets]

    def table(self) -> str:
        lines = [f"{self.param:>24}" + "".join(f"{SHORT[k]:>9}" for k in METRICS)]
        for v, b in zip(self.values, self.budgets):
            row = "".join(f"{getattr(b, k):9.2f}" if b is not None and not math.isinf(getattr(b, k)) else f"{'-':>9}"
                          for k in METRICS)
            lines.append(f"{v:>24.6g}" + row)
        return "\n".join(lines)


def _set_path(spec: dict, keys: list, value) -> None:
    node = spec
    for k in keys[:-1]:
        node = node[int(k)] if isinstance(node, list) else node[k]
    last = keys[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def budget_sweep(circuit: Circuit, param: str, values, settings: Settings | None = None, **kw) -> BudgetSweep:
    """The budget with `param` ('lna.gain_db', 'ant.tones.0.power') set to each of `values`. A step whose
    tone doesn't reach the output gives None there; a parameter the block doesn't have is an error."""
    from .netlist import from_dict, to_dict
    block, _, path = param.partition(".")
    base = to_dict(circuit, settings)
    if block not in base["blocks"] or not path:
        raise ValueError(f"no parameter {param!r}: write it block.parameter, with a block in this circuit")
    budgets = []
    for v in values:
        d = copy.deepcopy(base)
        try:
            _set_path(d["blocks"][block], path.split("."), v)
            c, st = from_dict(d)
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"no parameter {param!r}: {e}") from None
        try:
            budgets.append(receiver_budget(c, settings=st, **kw))
        except ValueError as e:
            if "no tone" in str(e) or "no signal path" in str(e):
                raise
            budgets.append(None)
    return BudgetSweep(param, list(values), budgets)


def _db10(x: float) -> float:
    return 10 * math.log10(x)
