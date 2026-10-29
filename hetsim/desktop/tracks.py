"""Numeric helpers for the desktop viewer's plots, free of Qt so they can be tested alone.

The same calculations as the browser viewer's page: band-sweep track interpolation, the
swept-analyser RBW trace, phase-noise profile lookup, and the schematic's automatic layout.
"""
from __future__ import annotations

import math

import numpy as np


def interp_track(f: list, p: list, k: int, smooth: bool) -> tuple[list, list, list]:
    """k points per step interval of one product's sweep track (None where absent).

    A product's frequency is linear in the swept frequencies, so it interpolates exactly; its
    level is estimated, linearly or by a monotone cubic (no overshoot). Runs are not joined
    across steps where the product is absent. Returns (freqs, levels, step positions).
    """
    fo, po, at = [], [], []
    n = len(f)
    a = 0
    while a < n:
        if f[a] is None:
            a += 1
            continue
        b = a
        while b + 1 < n and f[b + 1] is not None:
            b += 1
        d = [p[i + 1] - p[i] for i in range(a, b)]   # secants, then Fritsch-Butland tangents
        m = []
        for i in range(a, b + 1):
            j = i - a
            dl = d[j - 1] if j > 0 else None
            dr = d[j] if j < len(d) else None
            if dl is None:
                m.append(dr if dr is not None else 0.0)
            elif dr is None:
                m.append(dl)
            else:
                m.append(0.0 if dl * dr <= 0 else 2 * dl * dr / (dl + dr))
        for i in range(a, b + 1):
            for s in range(k if i < b else 1):
                t, j = s / k, i - a
                fo.append(f[i] + (f[i + 1] - f[i]) * t if s else f[i])
                if not s:
                    po.append(p[i])
                elif not smooth:
                    po.append(p[i] + (p[i + 1] - p[i]) * t)
                else:
                    t2, t3 = t * t, t * t * t
                    po.append((2 * t3 - 3 * t2 + 1) * p[i] + (t3 - 2 * t2 + t) * m[j]
                              + (-2 * t3 + 3 * t2) * p[i + 1] + (t3 - t2) * m[j + 1])
                at.append(i + t)
        if b + 1 < n:   # keep the gap
            fo.append(None)
            po.append(None)
            at.append(None)
        a = b + 1
    return fo, po, at


def step_at(steps: list[dict], x: float) -> dict:
    """Swept tones' frequencies at a (possibly fractional) step; they move linearly between steps."""
    i = min(int(math.floor(x)), len(steps) - 1)
    t = x - i
    a, b = steps[i], steps[min(i + 1, len(steps) - 1)]
    return {k: a[k] + (b[k] - a[k]) * t for k in a}


def pn_at(profile: list, off: float) -> float | None:
    """SSB phase noise (dBc/Hz) at an offset, log-interpolated; None outside the profile."""
    if not profile or off < profile[0][0] or off > profile[-1][0]:
        return None
    if len(profile) == 1:
        return profile[0][1]
    for (f0, l0), (f1, l1) in zip(profile, profile[1:]):
        if off <= f1:
            return l0 + (l1 - l0) * math.log10(off / f0) / math.log10(f1 / f0)
    return None


def rbw_trace(x0: float, x1: float, n: int, lines: list[tuple[float, float]], rbw: float,
              skirts: list | None = None, profiles: dict | None = None) -> np.ndarray:
    """Swept-analyser view over [x0, x1] at n display points, in dBm.

    Each line (f, dBm) goes through a Gaussian RBW (-3 dB at +-rbw/2) with a peak detector per
    point (a line narrower than a point is not missed); phase-noise skirts (the server's
    'skirts' records) add density x noise bandwidth. Summed in mW.
    """
    mw = np.zeros(n)
    df = (x1 - x0) / n
    k = 4 * math.log(2) / (rbw * rbw)
    reach = 3.5 * rbw
    for f, p in lines:
        a = max(0, int(math.floor((f - reach - x0) / df)))
        b = min(n - 1, int(math.floor((f + reach - x0) / df)))
        if b < a:
            continue
        fa = x0 + np.arange(a, b + 1) * df
        d = np.where(f < fa, fa - f, np.where(f > fa + df, f - (fa + df), 0.0))
        mw[a:b + 1] += 10 ** (p / 10) * np.exp(-k * d * d)
    bn = 1.0645 * rbw   # Gaussian noise bandwidth
    for s in skirts or []:
        prof, g = profiles[s["src"]], 20 * math.log10(s["m"])
        off, lvl = s["off"], s["lvl"]
        a = max(0, int(math.floor((s["f0"] + off[0] - x0) / df)))
        b = min(n - 1, int(math.ceil((s["f0"] + off[-1] - x0) / df)))
        j = 0
        for i in range(a, b + 1):
            o = x0 + (i + 0.5) * df - s["f0"]
            while j < len(off) - 2 and off[j + 1] < o:
                j += 1
            oa, ob = off[j], off[j + 1] if j + 1 < len(off) else off[j]
            if ob == oa or o < oa or o > ob or (oa * ob <= 0 and abs(o) < min(abs(oa), abs(ob))):
                continue
            pn = pn_at(prof, abs(o) / s["m"])
            if pn is None:
                continue
            level = lvl[j] + (lvl[j + 1] - lvl[j]) * (o - oa) / (ob - oa)
            mw[i] += 10 ** ((level + pn + g) / 10) * bn
    with np.errstate(divide="ignore"):
        return np.where(mw > 0, 10 * np.log10(np.maximum(mw, 1e-300)), -400.0)


# ---------------------------------------------------------------- schematic layout
BW, BH = 64, 48   # block body size in schematic units
COL_PITCH, ROW_PITCH = 150.0, 130.0   # auto-layout grid: room for a block's name and three detail lines


def port_layout(ports: list[str]) -> dict[str, tuple[float, float, int, int]]:
    """Port positions relative to a block: inputs left, outputs right, LO on top, a source's
    sweep (control) port at the bottom. Returns {port: (x, y, dx, dy)} with (dx, dy) the
    direction a wire leaves in."""
    left, right, top, bottom = [], [], [], []
    for p in ports:
        if p == "lo":
            top.append(p)
        elif p == "sweep":
            bottom.append(p)
        elif p in ("out", "if", "a", "b"):
            right.append(p)
        elif p in ("in", "rf", "sum", "x", "y"):
            left.append(p)
        else:
            (left if len(left) <= len(right) else right).append(p)
    out = {}
    for i, p in enumerate(left):
        out[p] = (0.0, BH * (i + 1) / (len(left) + 1), -1, 0)
    for i, p in enumerate(right):
        out[p] = (float(BW), BH * (i + 1) / (len(right) + 1), 1, 0)
    for i, p in enumerate(top):
        out[p] = (BW * (i + 1) / (len(top) + 1), 0.0, 0, -1)
    for i, p in enumerate(bottom):
        out[p] = (BW * (i + 1) / (len(bottom) + 1), float(BH), 0, 1)
    return out


def auto_layout(blocks: list[dict], wires: list) -> dict[str, tuple[float, float]]:
    """Columns follow the signal (longest path from the sources); a branch that drives a mixer's
    LO port sits one row above that mixer, ending one column before it. Returns positions on a
    COL_PITCH x ROW_PITCH grid."""
    names = [b["name"] for b in blocks]
    side, succ, pred, edges, feeds = {}, {n: [] for n in names}, {n: [] for n in names}, [], {}
    for b in blocks:
        for p, pos in port_layout(b["ports"]).items():
            side[f"{b['name']}.{p}"] = pos
    for a, b in wires:
        pa, pb = side.get(a), side.get(b)
        if not pa or not pb:
            continue
        if pa[2] > 0 >= pb[2]:
            u, v = a, b
        elif pb[2] > 0 >= pa[2]:
            u, v = b, a
        else:
            continue
        bu, bv = u.split(".")[0], v.split(".")[0]
        if bu == bv:
            continue
        succ[bu].append(bv)
        pred[bv].append(bu)
        if v.endswith(".lo"):
            feeds[bu] = bv
        else:
            edges.append((bu, bv))
    col = {n: 0 for n in names}
    for _ in names:
        for u, v in edges:
            col[v] = max(col[v], col[u] + 1)
    lo_branch: set[str] = set()
    for u, mix in feeds.items():
        shift = col[mix] - 1 - col[u]
        stack, seen = [u], set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            lo_branch.add(n)
            stack += [q for q in pred[n] if q not in feeds or q == u]
        if shift > 0:
            for n in seen:
                col[n] += shift
    row, used = {}, set()

    def place(n, pref):
        d = 0
        while True:
            for r in ([pref] if d == 0 else [pref + d, pref - d]):
                if (col[n], r) not in used:
                    used.add((col[n], r))
                    row[n] = r
                    return
            d += 1

    for n in sorted((n for n in names if n not in lo_branch), key=lambda n: col[n]):
        p = next((q for q in pred[n] if q in row), None)
        place(n, row[p] if p is not None else 0)
    for n in sorted(lo_branch, key=lambda n: -col[n]):
        if n in feeds and feeds[n] in row:
            place(n, row[feeds[n]] - 1)
        else:
            s = next((q for q in succ[n] if q in row), None)
            place(n, row[s] if s is not None else -1)
    return {n: (col[n] * COL_PITCH, row[n] * ROW_PITCH) for n in names}


def route_bus(a: tuple, b: tuple, bus_y: float, a_gap: float, b_gap: float) -> list[tuple[float, float]]:
    """A control wire kept clear of the blocks: out of `a`, across to the gap between columns
    at x = a_gap, down to a bus line below the circuit, along it to x = b_gap, and up into `b`."""
    s = 12
    a1 = (a[0] + a[2] * s, a[1] + a[3] * s)
    b1 = (b[0] + b[2] * s, b[1] + b[3] * s)
    return [(a[0], a[1]), a1, (a_gap, a1[1]), (a_gap, bus_y), (b_gap, bus_y), (b_gap, b1[1]), b1, (b[0], b[1])]


def route(a: tuple, b: tuple) -> list[tuple[float, float]]:
    """Orthogonal wire between two ports (x, y, dx, dy), leaving each along its direction."""
    s = 12
    a1 = (a[0] + a[2] * s, a[1] + a[3] * s)
    b1 = (b[0] + b[2] * s, b[1] + b[3] * s)
    pts = [(a[0], a[1]), a1]
    ha, hb = a[2] != 0, b[2] != 0
    if ha and hb:
        mx = (a1[0] + b1[0]) / 2
        pts += [(mx, a1[1]), (mx, b1[1])]
    elif ha:
        pts.append((b1[0], a1[1]))
    elif hb:
        pts.append((a1[0], b1[1]))
    else:
        my = (a1[1] + b1[1]) / 2
        pts += [(a1[0], my), (b1[0], my)]
    return pts + [b1, (b[0], b[1])]
