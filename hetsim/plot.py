"""Matplotlib plots (imported lazily so the engine works without matplotlib)."""
from __future__ import annotations

import math

import numpy as np

# reference palette (light): categorical slots 1-3, text and surface
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _ax(ax, figsize=(10, 4.5)):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return ax


def _labels(ax, pts, scale, xspan):
    """Annotate (freq, y, text) points, stacking labels that would collide horizontally."""
    placed = []
    for f, y, text in sorted(pts, key=lambda p: -p[1]):
        x = f / scale
        dy = 4
        for px, py, pdy in placed:
            if abs(px - x) < 0.06 * xspan and abs((py + pdy / 10) - (y + dy / 10)) < 2.5:
                dy = pdy + 12
        placed.append((x, y, dy))
        ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points", fontsize=8, color=INK)


def _hz_axis(ax, fmax):
    for scale, unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz"), (1, "Hz")):
        if fmax >= scale:
            return scale, unit
    return 1, "Hz"


def plot_spectrum(spec, ax=None, fmin=None, fmax=None, min_dbm=None, labels: int = 8, title=None,
                  band=None):
    """Stem plot of a Spectrum, coloured by product order, top `labels` tones annotated."""
    ax = _ax(ax)
    tones = [t for t in spec.band(fmin, fmax)]
    if not tones:
        return ax
    top = max(t.power_dbm for t in tones)
    lo = min_dbm if min_dbm is not None else max(min(t.power_dbm for t in tones), top - 150)
    tones = [t for t in tones if t.power_dbm >= lo]
    scale, unit = _hz_axis(ax, max(t.freq for t in tones))
    groups = [("order 1-2", lambda o: o <= 2, C1), ("order 3-4", lambda o: 3 <= o <= 4, C2),
              ("order 5+", lambda o: o >= 5, C3)]
    for name, pred, col in groups:
        sel = [t for t in tones if pred(t.order)]
        if not sel:
            continue
        f = np.array([t.freq for t in sel]) / scale
        p = np.array([t.power_dbm for t in sel])
        ax.vlines(f, lo - 5, p, color=col, linewidth=2)
        ax.scatter(f, p, s=22, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=name)
    x0 = (fmin if fmin is not None else min(t.freq for t in tones)) / scale
    x1 = (fmax if fmax is not None else max(t.freq for t in tones)) / scale
    if fmin is not None and fmax is not None:
        ax.set_xlim(x0, x1)
    _labels(ax, [(t.freq, t.power_dbm, t.label) for t in sorted(tones, key=lambda t: -t.power_dbm)[:labels]],
            scale, max(x1 - x0, 1e-12))
    if band is not None:
        ax.axvspan(band[0] / scale, band[1] / scale, color=C1, alpha=0.08, lw=0)
    ax.set_ylim(lo - 5, top + 10)
    ax.set_xlabel(f"frequency ({unit})", color=INK2)
    ax.set_ylabel("power (dBm)", color=INK2)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    if title:
        ax.set_title(title, color=INK, loc="left", fontsize=11)
    return ax


def plot_response(sweep=None, responses=None, ax=None, title=None, labels: int = 8, floor_db=None,
                  name=None, color=C1):
    """Receiver response vs stimulus frequency: swept curve plus exact found responses."""
    ax = _ax(ax)
    fmax = 1.0
    if sweep is not None:
        fmax = max(fmax, float(np.max(sweep.freqs)))
    if responses is not None and responses.responses:
        fmax = max(fmax, max(r.freq for r in responses.responses))
    scale, unit = _hz_axis(ax, fmax)
    lo = floor_db
    if sweep is not None:
        g = np.where(np.isfinite(sweep.gain_db), sweep.gain_db, np.nan)
        if lo is None:
            lo = np.nanmax(g) - 140
        ax.plot(sweep.freqs / scale, np.maximum(g, lo), color=color, linewidth=2, label=name or "swept")
    if responses is not None and responses.responses:
        rs = responses.responses
        if lo is None:
            lo = max(r.gain_db for r in rs) - 140
        rs = [r for r in rs if r.gain_db >= lo]
        f = np.array([r.freq for r in rs]) / scale
        g = np.array([r.gain_db for r in rs])
        ax.vlines(f, lo, g, color=color, linewidth=1, alpha=0.5)
        ax.scatter(f, g, s=30, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label=None if sweep is not None else (name or "responses"))
        xspan = (max(r.freq for r in rs) - min(0, min(r.freq for r in rs))) / scale
        _labels(ax, [(r.freq, r.gain_db, r.label) for r in sorted(rs, key=lambda r: -r.gain_db)[:labels]],
                scale, max(xspan, 1e-12))
    if lo is not None:
        ax.set_ylim(bottom=lo - 5, top=ax.get_ylim()[1] + 10)
    ax.set_xlabel(f"stimulus frequency ({unit})", color=INK2)
    ax.set_ylabel("gain to output band (dB)", color=INK2)
    if title:
        ax.set_title(title, color=INK, loc="left", fontsize=11)
    return ax


def plot_td_spectrum(tds, ax=None, fmin=None, fmax=None, min_dbm=None, labels: int = 8, title=None,
                     band=None, color=C1, name=None, overlay_tone=False):
    """FFT of a time-domain result (TDSpectrum): one stem per occupied bin.

    Labels come from the tone solver when it is attached. overlay_tone=True also
    draws the tone solver's per-bin power as open markers, for comparison.
    """
    ax = _ax(ax)
    pk = tds.peaks(min_dbm if min_dbm is not None else -400, fmin, fmax)
    if not pk:
        return ax
    top = max(p for _, p in pk)
    lo = min_dbm if min_dbm is not None else top - 150
    pk = [(f, p) for f, p in pk if p >= lo]
    scale, unit = _hz_axis(ax, fmax if fmax is not None else max(f for f, _ in pk))
    f = np.array([f for f, _ in pk]) / scale
    p = np.array([p for _, p in pk])
    ax.vlines(f, lo - 5, p, color=color, linewidth=2)
    ax.scatter(f, p, s=22, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=name or "time domain")
    if overlay_tone and tds.tone is not None:
        pts = []
        for fr, _ in pk:
            ts = tds.products_at(fr)
            if ts:
                pts.append((fr / scale, 10 * math.log10(sum(10 ** (t.power_dbm / 10) for t in ts))))
        if pts:
            ax.scatter([a for a, _ in pts], [b for _, b in pts], s=60, facecolor="none", edgecolor=C2,
                       linewidth=1.5, zorder=4, label="tone solver (power sum per bin)")
    x0 = (fmin if fmin is not None else min(fr for fr, _ in pk)) / scale
    x1 = (fmax if fmax is not None else max(fr for fr, _ in pk)) / scale
    if fmin is not None and fmax is not None:
        ax.set_xlim(x0, x1)
    lab = []
    for fr, pw in sorted(pk, key=lambda r: -r[1])[:labels]:
        ts = tds.products_at(fr)
        lab.append((fr, pw, ts[0].label + (" +" if len(ts) > 1 else "") if ts else f"{fr / scale:g}"))
    _labels(ax, lab, scale, max(x1 - x0, 1e-12))
    if band is not None:
        ax.axvspan(band[0] / scale, band[1] / scale, color=C1, alpha=0.08, lw=0)
    ax.set_ylim(lo - 5, top + 10)
    ax.set_xlabel(f"frequency ({unit})", color=INK2)
    ax.set_ylabel("power per bin (dBm)", color=INK2)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    if title:
        ax.set_title(title, color=INK, loc="left", fontsize=11)
    return ax


def plot_waveform(result, port, ax=None, tmax=None, incident=False, title=None, color=C1):
    """Waveform at a port from a TDResult (volts across the reference impedance)."""
    ax = _ax(ax)
    x = result.incident_waveform(port) if incident else result.waveform(port)
    t = result.t
    if tmax is not None:
        m = t <= tmax
        t, x = t[m], x[m]
    for scale, unit in ((1, "s"), (1e-3, "ms"), (1e-6, "µs"), (1e-9, "ns")):
        if t[-1] >= scale or scale == 1e-9:
            break
    ax.plot(t / scale, x, color=color, linewidth=1.2)
    ax.set_xlabel(f"time ({unit})", color=INK2)
    ax.set_ylabel("V", color=INK2)
    if title:
        ax.set_title(title, color=INK, loc="left", fontsize=11)
    return ax
