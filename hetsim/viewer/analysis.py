"""Noise, match and budget for both viewers: text tables and chart data as plain JSON.

A chart is {id, title, x, x_label, x_unit, y_unit, series: [{name, y}]}, with one y unit per chart:
measures on different scales become separate charts, which the viewers show one at a time.
"""
from __future__ import annotations

import math

import numpy as np

from ..budget import METRICS, budget_sweep, receiver_budget
from ..netlist import from_dict
from ..smallsignal import match_sweep, mismatch_report, noise_figure
from ..tolerance import monte_carlo, tolerances_of
from ..units import parse_freq
from .model import describe


def _clean(ys) -> list:
    """Floats for JSON; a value with no number becomes a gap in the line."""
    return [float(y) if y is not None and math.isfinite(y) else None for y in ys]


def _band(text):
    """'low:high', or nothing, to a pair of frequencies."""
    text = (text or "").strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"a band is written low:high, e.g. 10.6MHz:10.8MHz, not {text!r}")
    lo, hi = parse_freq(parts[0]), parse_freq(parts[1])
    if not 0 < lo < hi:
        raise ValueError(f"band {text!r}: need 0 < low < high")
    return lo, hi


def options(model) -> dict:
    """What the analysis controls offer for the current circuit."""
    c, _ = from_dict(model.base, model.base_dir)
    sources = {n: [t.name for t in b.specs if t.mode != "wideband"] for n, b in c.blocks.items() if b.kind == "source"}
    ports = [f"{n}.{p}" for n, b in c.blocks.items() for p in b.ports if p not in b.controls]
    outputs = [f"{n}.{p}" for n, b in c.blocks.items() if b.kind != "source" for p in b.ports if p not in b.controls]
    vary = []
    for g in describe(model.base):
        for p in g["params"]:
            if p["kind"] in ("number", "level") and (not p["optional"] or p["enabled"]) and not p["path"].endswith("phase"):
                vary.append({"path": f"{g['block']}.{p['path']}", "value": p["value"], "min": p["min"], "max": p["max"]})
    return {"sources": sources, "ports": ports, "outputs": outputs, "vary": vary, "tolerances": tolerances_of(c),
            "metrics": [{"id": k, "label": v[0], "unit": v[1]} for k, v in METRICS.items()]}


def analyse(model, req: dict) -> dict:
    """req: kind (noise | wires | match | budget | vary), params (slider edits), and what the kind needs.
    Returns {text, charts}; {error} for anything the circuit or the request can't answer."""
    try:
        return _run(model, req)
    except (ValueError, KeyError, TypeError) as e:
        return {"error": str(e.args[0]) if e.args else str(e)}


def _run(model, req: dict) -> dict:
    c, st = from_dict(model.netlist(req.get("params") or {}), model.base_dir)
    kind = req.get("kind")
    source, tone, port = req.get("source") or None, req.get("tone") or None, req.get("port") or None
    if kind == "noise":
        r = noise_figure(c, source, port, _band(req.get("band")), tone if tone is not None else 0, st)
        charts = []
        if len(r.freqs_out) > 1:
            x = r.freqs_out.tolist()
            charts = [_chart("nf", "Noise figure across the band", x, "Output frequency", "Hz", "dB",
                             [("Noise figure", r.nf_at)]),
                      _chart("gain", "Gain across the band", x, "Output frequency", "Hz", "dB",
                             [("Gain", r.gain_at)])]
        return {"text": r.table(), "charts": charts}
    if kind == "wires":
        return {"text": mismatch_report(c, st, tone).table(), "charts": []}
    if kind == "match":
        rng = _band(req.get("range"))
        if rng is None:
            raise ValueError("give a frequency range to sweep, e.g. 88MHz:108MHz")
        sw = match_sweep(c, port, np.linspace(*rng, int(req.get("points") or 41)), st)
        x = sw.freqs.tolist()
        return {"text": sw.table(), "charts": [
            _chart("vswr", f"VSWR looking into {sw.port}", x, "Frequency", "Hz", ":1",
                   [("Typical", sw.vswr), ("Worst case", sw.vswr_worst)], "VSWR"),
            _chart("rl", f"Return loss looking into {sw.port}", x, "Frequency", "Hz", "dB",
                   [("Return loss", sw.return_loss_db)])]}
    kw = dict(source=source, tone=tone, port=port, bw=req.get("bw") or None, offset=req.get("offset") or None,
              snr_db=float(req.get("snr") if req.get("snr") not in (None, "") else 10.0))
    if kind == "budget":
        return {"text": receiver_budget(c, settings=st, **kw).table(), "charts": []}
    if kind == "vary":
        param = req.get("param")
        if not param:
            raise ValueError("pick a parameter to vary")
        lo, hi = float(req["from"]), float(req["to"])
        n = int(req.get("points") or 11)
        if n < 2:
            raise ValueError("varying a parameter needs at least 2 points")
        sw = budget_sweep(c, param, np.linspace(lo, hi, n).tolist(), st, **kw)
        x = [float(v) for v in sw.values]
        charts = [_chart(k, label, x, param, "", unit, [(label, sw.series(k))]) for k, (label, unit) in METRICS.items()]
        return {"text": sw.table(), "charts": charts}
    if kind == "tolerance":
        mc = monte_carlo(c, int(req.get("runs") or 200), None, req.get("specs") or (), settings=st, workers=model.sweep_workers,
                         **kw)
        charts = []
        for k, (label, unit) in METRICS.items():
            xs, pct = mc.cdf(k)
            if len(xs):
                charts.append(_chart(k, f"{label}: share of runs at or below", xs, f"{label} ({unit})", "", "%",
                                     [("Runs", pct)], "Runs (%)"))
        return {"text": mc.table(), "charts": charts}
    raise ValueError(f"unknown analysis {kind!r}")


def _chart(id_, title, x, x_label, x_unit, y_unit, series, y_label=None) -> dict:
    return {"id": id_, "title": title, "x": x, "x_label": x_label, "x_unit": x_unit, "y_unit": y_unit,
            "y_label": y_label or y_unit,
            "series": [{"name": n, "y": _clean(y)} for n, y in series]}


__all__ = ["analyse", "options"]
