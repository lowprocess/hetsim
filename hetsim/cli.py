"""Command line: python -m hetsim {run,responses,noise,match,blocks,view,app} ..."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import __datecode__
from .analysis import find_responses, response_sweep
from .blocks import REGISTRY
from .netlist import load
from .smallsignal import match_sweep, mismatch_report, noise_figure, noise_sweep, port_match
from .units import parse_db, parse_freq


def _band(s):
    a, b = s.split(":")
    return parse_freq(a), parse_freq(b)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hetsim", description=f"tone-domain RF chain simulator ({__datecode__})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="simulate a netlist and print the spectrum at one or more ports")
    r.add_argument("netlist")
    r.add_argument("--port", "-p", action="append", required=True, help="block.port (repeatable)")
    r.add_argument("--incident", action="store_true", help="show the wave arriving at the port instead")
    r.add_argument("--band", type=_band, help="fmin:fmax, e.g. 10.5MHz:10.9MHz")
    r.add_argument("--min-dbm", type=parse_db, default=-200)
    r.add_argument("--ref", help="product label to express levels in dBc against, e.g. 'LO - RF'")
    r.add_argument("--sort", choices=["freq", "power"], default="freq")
    r.add_argument("--limit", type=int)
    r.add_argument("--json", action="store_true")
    r.add_argument("--plot", help="save a spectrum plot (png/svg/pdf) of the first port")
    r.add_argument("--sweep", action="store_true",
                   help="also step the tones with mode: sweep across their bands and list each product's range "
                        "and worst level")
    r.add_argument("--sweep-points", type=int, help="steps for --sweep (default: the tones' points, else 41)")
    r.add_argument("--workers", type=int, help="processes for --sweep (default: one per physical core; 1: none)")
    r.add_argument("--solver", choices=["tone", "td"], default="tone",
                   help="tone (default) or td: time-domain sampling + FFT")
    r.add_argument("--resolution", help="td: FFT bin spacing (default: GCD of source frequencies)")
    r.add_argument("--fmax", help="td: highest frequency to represent (default: planned)")
    r.add_argument("--fs", help="td: sample rate (overrides --fmax)")
    r.add_argument("--waveform", help="td: save a time-waveform plot of the first port")
    r.add_argument("--tmax", help="td: time span for --waveform, e.g. 2e-6")

    s = sub.add_parser("responses", help="find every stimulus frequency that reaches the output band")
    s.add_argument("netlist")
    s.add_argument("--source", required=True, help="source block to use as the stimulus")
    s.add_argument("--tone", type=int, default=0)
    s.add_argument("--port", required=True)
    s.add_argument("--band", type=_band, required=True)
    s.add_argument("--range", type=_band, required=True, help="stimulus fmin:fmax")
    s.add_argument("--mute", action="append", default=[], help="source blocks to silence (repeatable)")
    s.add_argument("--limit", type=int, default=30)
    s.add_argument("--sweep-points", type=int, default=0, help="also run a swept curve with N points")
    s.add_argument("--plot")

    n = sub.add_parser("noise", help="noise figure from a source tone to a port, with a budget of who adds what")
    n.add_argument("netlist")
    n.add_argument("--source", required=True, help="source block whose termination is the reference")
    n.add_argument("--tone", default="0", help="the signal tone: name or index (a muted tone works)")
    n.add_argument("--port", required=True)
    n.add_argument("--band", type=_band, help="output band fmin:fmax to integrate over (default: spot, at the product)")
    n.add_argument("--points", type=int, default=21, help="frequencies across --band")
    n.add_argument("--sweep", action="store_true", help="step the tones with mode: sweep and list NF at each step")
    n.add_argument("--sweep-points", type=int, help="steps for --sweep (default: the tones' points)")
    n.add_argument("--json", action="store_true")

    m = sub.add_parser("match", help="VSWR looking into a port, or at both ends of every wire")
    m.add_argument("netlist")
    m.add_argument("--port", help="block.port to look into (default: report every wire)")
    m.add_argument("--freq", help="frequency, with --port")
    m.add_argument("--range", type=_band, help="sweep fmin:fmax, with --port")
    m.add_argument("--points", type=int, default=21)
    m.add_argument("--tone", help="without --port: follow this tone down the chain instead of each wire's strongest")
    m.add_argument("--json", action="store_true")

    sub.add_parser("blocks", help="list block types and their parameters")

    v = sub.add_parser("view", help="open the live FFT viewer in a browser")
    v.add_argument("netlist")
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("--no-browser", action="store_true", help="just print the URL")
    v.add_argument("--workers", type=int, help="processes for band sweeps (default: one per physical core; 1: none)")

    d = sub.add_parser("app", help="open the native desktop viewer (needs PySide6)")
    d.add_argument("netlist", nargs="?", help="netlist to open (default: the start page)")
    d.add_argument("--workers", type=int, help="processes for band sweeps (default: one per physical core; 1: none)")

    a = ap.parse_args(argv)

    if a.cmd == "blocks":
        import inspect
        for k, cls in sorted(REGISTRY.items()):
            sig = inspect.signature(cls.__init__)
            params = ", ".join(str(p) for n, p in sig.parameters.items() if n != "self")
            print(f"{k:13s} ports={cls.ports}\n              {params}")
        return 0

    if a.cmd == "app":
        try:
            from .desktop import run
            return run(a.netlist, workers=a.workers)
        except ImportError as e:
            print(f"the desktop viewer needs PySide6 (pip install 'hetsim[desktop]'): {e}", file=sys.stderr)
            return 1

    if a.cmd == "view":
        from .viewer import serve
        serve(a.netlist, port=a.port, open_browser=not a.no_browser, workers=a.workers)
        return 0

    circuit, st = load(a.netlist)

    if a.cmd == "run" and a.solver == "td":
        return _run_td(a, circuit, st)

    if a.cmd == "run":
        res = circuit.simulate(st)
        fmin, fmax = a.band or (None, None)
        sw = circuit.simulate_sweep(st, a.sweep_points, workers=a.workers) if a.sweep else None
        if a.json:
            out = {p: (res.incident(p) if a.incident else res[p]).band(fmin, fmax).to_records() for p in a.port}
            doc = {"converged": res.converged, "warnings": res.warnings, "ports": out}
            if sw:
                doc["sweep"] = {p: [{"product": s.label, "fmin": s.fmin, "fmax": s.fmax, "max_dbm": s.p_max,
                                     "worst_at": sw.steps[s.i_max]}
                                    for s in sw.products(p, a.incident, a.min_dbm, fmin, fmax)[:a.limit]]
                                for p in a.port}
            json.dump(doc, sys.stdout, indent=1)
            print()
        else:
            for p in a.port:
                spec = res.incident(p) if a.incident else res[p]
                print(f"== {'arriving at' if a.incident else 'leaving'} {p}")
                print(spec.table(fmin, fmax, a.min_dbm, a.ref, a.limit, a.sort))
                print()
                if sw:
                    print(f"== over the band sweep ({len(sw.steps)} steps): {sw.step_text(0)} to "
                          f"{sw.step_text(-1)}")
                    print(sw.table(p, a.incident, a.min_dbm, fmin, fmax, a.limit))
                    print()
            w = res.warning_text()
            if w:
                print("warnings:\n" + w)
        if a.plot:
            import matplotlib.pyplot as plt
            from .plot import plot_spectrum
            p = a.port[0]
            spec = res.incident(p) if a.incident else res[p]
            ax = plot_spectrum(spec, fmin=fmin, fmax=fmax, min_dbm=a.min_dbm if a.min_dbm > -200 else None,
                               title=f"{circuit.name}: {p}")
            ax.figure.tight_layout()
            ax.figure.savefig(a.plot, dpi=150)
            plt.close(ax.figure)
        return 0

    if a.cmd == "noise":
        return _noise(a, circuit, st)

    if a.cmd == "match":
        return _match(a, circuit, st)

    if a.cmd == "responses":
        tab = find_responses(circuit, a.source, a.port, a.band, *a.range, tone=a.tone, settings=st, mute=a.mute)
        print(f"responses of {a.source} -> {a.port} in band {a.band[0]:g}:{a.band[1]:g} Hz")
        print(tab.table(limit=a.limit))
        if a.plot:
            import matplotlib.pyplot as plt
            from .plot import plot_response
            sw = None
            if a.sweep_points:
                sw = response_sweep(circuit, a.source, a.port, a.band,
                                    np.linspace(*a.range, a.sweep_points), tone=a.tone, settings=st, mute=a.mute)
            ax = plot_response(sw, tab, title=f"{circuit.name}: response at {a.port}")
            ax.figure.tight_layout()
            ax.figure.savefig(a.plot, dpi=150)
            plt.close(ax.figure)
        return 0


def _noise(a, circuit, st):
    tone = int(a.tone) if a.tone.lstrip("-").isdigit() else a.tone
    try:
        if a.sweep:
            sw = noise_sweep(circuit, a.source, a.port, a.band, tone, st, a.sweep_points)
            if a.json:
                nan_none = lambda x: None if x != x else float(x)  # noqa: E731
                json.dump([{"f_in": float(f), "nf_db": nan_none(n), "gain_db": nan_none(g)}
                           for f, n, g in zip(sw.f_in, sw.nf_db, sw.gain_db)], sys.stdout, indent=1)
                print()
            else:
                print(sw.table())
            return 0
        r = noise_figure(circuit, a.source, a.port, a.band, tone, st, a.points)
    except ValueError as e:
        print(f"hetsim noise: {e}", file=sys.stderr)
        return 1
    if a.json:
        json.dump(r.to_dict(), sys.stdout, indent=1)
        print()
    else:
        print(r.table())
    return 0


def _match(a, circuit, st):
    if a.port is None:
        rep = mismatch_report(circuit, st, a.tone)
        if a.json:
            json.dump([{"a": i.a.port, "b": i.b.port, "freq": i.freq, "product": i.product, "vswr_a": i.a.vswr,
                        "vswr_b": i.b.vswr, "ripple_db": list(i.ripple_db)} for i in rep.interfaces],
                      sys.stdout, indent=1)
            print()
        else:
            print(rep.table())
        return 0
    if a.range:
        sw = match_sweep(circuit, a.port, np.linspace(*a.range, a.points), st)
        if a.json:
            json.dump([{"freq": float(f), "vswr": float(v), "vswr_worst": float(w), "return_loss_db": float(r)}
                       for f, v, w, r in zip(sw.freqs, sw.vswr, sw.vswr_worst, sw.return_loss_db)], sys.stdout, indent=1)
            print()
        else:
            print(sw.table())
        return 0
    if a.freq is None:
        print("hetsim match: --port needs --freq or --range", file=sys.stderr)
        return 1
    m = port_match(circuit, a.port, a.freq, st)
    if a.json:
        json.dump({"port": m.port, "freq": m.freq, "vswr": m.vswr, "vswr_worst": m.vswr_worst,
                   "return_loss_db": m.return_loss_db, "mismatch_loss_db": m.mismatch_loss_db}, sys.stdout, indent=1)
        print()
    else:
        print(f"looking into {m.port} at {parse_freq(a.freq):g} Hz: VSWR {m.vswr:.3f}:1, return loss "
              f"{m.return_loss_db:.2f} dB, mismatch loss {m.mismatch_loss_db:.3f} dB, worst case VSWR {m.vswr_worst:.3f}:1")
    return 0


def _run_td(a, circuit, st):
    kw = {k: getattr(a, k) for k in ("resolution", "fmax", "fs") if getattr(a, k) is not None}
    res = circuit.simulate_td(st, **kw)
    fmin, fmax = a.band or (None, None)
    min_dbm = a.min_dbm if a.min_dbm > -200 else -200
    if a.json:
        out = {}
        for p in a.port:
            sp = res.incident_spectrum(p) if a.incident else res.spectrum(p)
            out[p] = [{"freq": f, "power_dbm": pw, "products": [t.label for t in sp.products_at(f)]}
                      for f, pw in sp.peaks(min_dbm, fmin, fmax)]
        json.dump({"converged": res.converged, "plan": str(res.plan), "warnings": res.warnings, "ports": out},
                  sys.stdout, indent=1)
        print()
    else:
        print(f"time domain: {res.plan}")
        for p in a.port:
            sp = res.incident_spectrum(p) if a.incident else res.spectrum(p)
            print(f"== {'arriving at' if a.incident else 'leaving'} {p}")
            print(sp.table(min_dbm, fmin, fmax, a.sort, a.limit))
            print()
        w = res.warning_text()
        if w:
            print("warnings:\n" + w)
    if a.plot or a.waveform:
        import matplotlib.pyplot as plt
        from .plot import plot_td_spectrum, plot_waveform
        p = a.port[0]
        if a.plot:
            sp = res.incident_spectrum(p) if a.incident else res.spectrum(p)
            ax = plot_td_spectrum(sp, fmin=fmin, fmax=fmax, min_dbm=a.min_dbm if a.min_dbm > -200 else None,
                                  title=f"{circuit.name}: {p} (time domain)", overlay_tone=True)
            ax.figure.tight_layout()
            ax.figure.savefig(a.plot, dpi=150)
            plt.close(ax.figure)
        if a.waveform:
            ax = plot_waveform(res, p, tmax=float(a.tmax) if a.tmax else None, incident=a.incident,
                               title=f"{circuit.name}: {p}")
            ax.figure.tight_layout()
            ax.figure.savefig(a.waveform, dpi=150)
            plt.close(ax.figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
