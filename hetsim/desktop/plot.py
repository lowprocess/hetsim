"""Spectrum and waveform plots for the desktop viewer, drawn with QPainter.

Same content as the browser viewer: FFT bins or tone-solver lines (fundamental mixer
products in their own colour), tone-solver rings, band-sweep tracks with interpolation and
single-track selection, the RBW trace with phase noise, a legend band, hover details, and
drag / wheel zoom.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QFont, QFontMetricsF, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QToolTip, QWidget

from . import theme
from .tracks import interp_track, rbw_trace, step_at


@dataclass
class ViewState:
    """What the plots draw and how: shared by the window, its controls and the plots."""
    data: dict | None = None
    solver: str = "td"            # td | both | tone
    floor: float = -160.0
    resolution: float = 1e5
    incident: bool = False
    rbw: float | None = None
    show_sweep: bool = True
    interp: str = "0"             # 0 | lin4 | lin8 | pchip4 | pchip8 | pchip16
    sw_sel: str | None = None     # one band-sweep track shown
    sw_out: tuple | None = None   # output-grid range for the sweep steps
    view: tuple | None = None     # zoomed frequency range
    extra: dict = field(default_factory=dict)


def fmt_f(f: float, digits: int = 6) -> str:
    a = abs(f)
    for s, u in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if a >= s:
            return f"{float(f'{f / s:.{digits}g}'):g} {u}"
    return f"{float(f'{f:.{digits}g}'):g} Hz"


def nice_step(span: float, target: float) -> float:
    raw = span / max(target, 1)
    p = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * p:
            return m * p
    return 10 * p


def unit_for(f: float):
    for s, u in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if abs(f) >= s:
            return s, u
    return 1, "Hz"


PAD_L, PAD_R, PAD_B = 58, 16, 36


class SpectrumPlot(QWidget):
    trackClicked = Signal(str)   # label of a band-sweep track clicked
    viewChanged = Signal()

    def __init__(self, state: ViewState, parent=None):
        super().__init__(parent)
        self.s = state
        self.setMouseTracking(True)
        self.setMinimumHeight(220)
        self.points: list[dict] = []
        self.drag = None
        self.geom = None
        self.font_ = QFont()
        self.font_.setPixelSize(11)

    # ------------------------------------------------------------ data
    def series(self):
        d = self.s.data
        if not d:
            return [], []
        tone = [{"f": t["f"], "p": t["p"], "labels": [t["label"]], "fund": t.get("fund")} for t in d.get("tone", [])]
        if self.s.solver == "tone" or not d.get("td"):
            return tone, []
        td = d["td"]
        fund = td.get("fund") or [False] * len(td["f"])
        stems = [{"f": f, "p": p, "labels": lab, "fund": fu} for f, p, lab, fu in zip(td["f"], td["p"], td["labels"], fund)]
        if self.s.solver == "td":
            return stems, []
        res = d.get("resolution") or self.s.resolution
        bins: dict = {}
        for t in d.get("tone", []):   # tone overlay: power sum per TD bin
            k = round(t["f"] / res)
            e = bins.setdefault(k, {"f": k * res, "mw": 0.0, "labels": []})
            e["mw"] += 10 ** (t["p"] / 10)
            e["labels"].append(t["label"])
        marks = [{"f": e["f"], "p": 10 * math.log10(e["mw"]), "labels": e["labels"]} for e in bins.values() if e["mw"] > 0]
        return stems, marks

    def sweep_tracks(self):
        sw = self.s.data and self.s.data.get("sweep")
        if not sw or not self.s.show_sweep:
            return []
        out = []
        m = self.s.interp
        k = int("".join(ch for ch in m if ch.isdigit()) or 0)
        for pr in sw["products"]:
            if k:
                f, p, at = interp_track(pr["f"], pr["p"], k, m.startswith("pchip"))
            else:
                f, p, at = pr["f"], pr["p"], list(range(len(pr["f"])))
            out.append({"label": pr["label"], "fund": pr.get("fund"), "f": f, "p": p, "at": at, "src": pr})
        return out

    def x_range(self, stems, tracks):
        if self.s.view:
            return self.s.view
        hi = max((st["f"] for st in stems if st["p"] >= self.s.floor), default=0.0)
        for pr in tracks:
            for f, p in zip(pr["f"], pr["p"]):
                if f is not None and p >= self.s.floor:
                    hi = max(hi, f)
        return 0.0, (hi or 1e6) * 1.05

    # ------------------------------------------------------------ drawing
    def paintEvent(self, _):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        qp.setFont(self.font_)
        fm = QFontMetricsF(self.font_)
        w, h = self.width(), self.height()
        qp.fillRect(self.rect(), theme.c("surface-1"))
        stems, marks = self.series()
        tracks = self.sweep_tracks()
        x0, x1 = self.x_range(stems, tracks)
        in_view = [st for st in stems if x0 <= st["f"] <= x1]
        top = max([st["p"] for st in in_view] + [m["p"] for m in marks if x0 <= m["f"] <= x1], default=-math.inf)
        for pr in tracks:
            for f, p in zip(pr["f"], pr["p"]):
                if f is not None and x0 <= f <= x1:
                    top = max(top, p)
        if not math.isfinite(top):
            top = 0.0
        any_fund = any(st.get("fund") for st in in_view) or any(pr.get("fund") for pr in tracks)

        # legend, in a band above the plot (wrapped to the width)
        items = [("series-1", "tone solver" if self.s.solver == "tone" else "time domain (FFT bin)", "dot")]
        if self.s.rbw:
            pn = ", phase noise" if self.s.extra.get("pn_on") and self.s.data and self.s.data.get("has_pn") else ""
            items.insert(0, ("text-primary", f"RBW {fmt_f(self.s.rbw, 3)} trace (Gaussian, peak detector{pn})", "line"))
        if any_fund:
            items.append(("series-2", "fundamental mixer products", "dot"))
        if marks:
            items.append(("text-muted", "tone solver, power sum per bin", "ring"))
        if tracks:
            sw = self.s.data["sweep"]
            extra = "" if self.s.interp == "0" else f", {self.s.extra.get('interp_text', '')} between"
            anc = (f", on a {fmt_f(self.s.sw_out[0], 4)} – {fmt_f(self.s.sw_out[1], 4)} grid for {sw['anchor']}"
                   if self.s.sw_out and sw.get("anchor") else "")
            only = f" · showing {self.s.sw_sel} (Esc: all)" if self.s.sw_sel else ""
            items.append(("series-3", f"band sweep, {len(sw['steps'])} steps{anc}{extra} (tone solver; □ worst computed){only}", "line"))
        legend, lx, line = [], PAD_L + 10, 0
        for it in items:
            wd = fm.horizontalAdvance(it[1]) + 30
            if lx > PAD_L + 10 and lx + wd - 20 > w - PAD_R:
                lx, line = PAD_L + 10, line + 1
            legend.append((lx, line, it))
            lx += wd
        pad_t = 14 + 15 * (line + 1)

        y1 = math.ceil((top + 8) / 10) * 10
        y0 = self.s.floor
        pw, ph = w - PAD_L - PAD_R, h - pad_t - PAD_B
        if pw < 20 or ph < 20 or x1 <= x0 or y1 <= y0:
            return
        X = lambda f: PAD_L + (f - x0) / (x1 - x0) * pw  # noqa: E731
        Y = lambda p: pad_t + (y1 - max(p, y0)) / (y1 - y0) * ph  # noqa: E731

        # grid and axes
        qp.setPen(QPen(theme.c("grid"), 1))
        ystep = nice_step(y1 - y0, ph / 40)
        v = math.ceil(y0 / ystep) * ystep
        while v <= y1:
            qp.setPen(QPen(theme.c("grid"), 1))
            qp.drawLine(QPointF(PAD_L, Y(v)), QPointF(w - PAD_R, Y(v)))
            qp.setPen(theme.c("text-muted"))
            qp.drawText(QRectF(0, Y(v) - 8, PAD_L - 6, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, str(round(v)))
            v += ystep
        us, uu = unit_for(x1)
        xstep = nice_step(x1 - x0, pw / 90)
        v = math.ceil(x0 / xstep) * xstep
        while v <= x1 + 1e-9:
            qp.setPen(QPen(theme.c("grid"), 1))
            qp.drawLine(QPointF(X(v), pad_t), QPointF(X(v), h - PAD_B))
            qp.setPen(theme.c("text-muted"))
            qp.drawText(QRectF(X(v) - 40, h - PAD_B + 4, 80, 14), Qt.AlignmentFlag.AlignHCenter, f"{float(f'{v / us:.8g}'):g}")
            v += xstep
        qp.setPen(theme.c("text-secondary"))
        qp.drawText(QRectF(PAD_L, h - 18, pw, 14), Qt.AlignmentFlag.AlignHCenter, f"frequency ({uu})")
        qp.save()
        qp.translate(14, pad_t + ph / 2)
        qp.rotate(-90)
        ylab = (f"power in {fmt_f(self.s.rbw, 3)} RBW (dBm)" if self.s.rbw
                else "power (dBm)" if self.s.solver == "tone" else "power per bin (dBm)")
        qp.drawText(QRectF(-ph / 2, -8, ph, 16), Qt.AlignmentFlag.AlignHCenter, ylab)
        qp.restore()

        qp.save()
        qp.setClipRect(QRectF(PAD_L, pad_t, pw, ph))
        self.points = []
        # band sweep tracks: each product's path while the swept tones step across their bands
        if tracks:
            sel = self.s.sw_sel
            for pr in sorted(tracks, key=lambda t: bool(t.get("fund"))):   # fundamentals on top
                mine = not sel or pr["label"] == sel
                col = theme.c("series-2" if pr.get("fund") else "series-3")
                qp.setOpacity((1.0 if sel else 0.75) if mine else 0.12)
                qp.setPen(QPen(col, 2.5 if sel and mine else 1.5))
                path, pen = QPainterPath(), False
                for f, p, at in zip(pr["f"], pr["p"], pr["at"]):
                    if f is None:
                        pen = False
                        continue
                    pt = QPointF(X(f), Y(p))
                    path.lineTo(pt) if pen else path.moveTo(pt)
                    pen = True
                    if x0 <= f <= x1 and p >= y0:
                        self.points.append({"x": pt.x(), "y": pt.y(), "f": f, "p": p, "labels": [pr["label"]],
                                            "kind": "sweep", "at": at, "pr": pr["src"]})
                qp.drawPath(path)
                fr = self.s.extra.get("frame")
                if mine and fr is not None and fr < len(pr["src"]["f"]) and pr["src"]["f"][fr] is not None:
                    src = pr["src"]   # the animation's step
                    qp.setBrush(col)
                    qp.drawEllipse(QPointF(X(src["f"][fr]), Y(src["p"][fr])), 3.5, 3.5)
                    qp.setBrush(Qt.BrushStyle.NoBrush)
                if mine:   # worst case over the band: computed steps only
                    src = pr["src"]
                    vals = [(p, i) for i, p in enumerate(src["p"]) if p is not None]
                    if vals:
                        pb, i = max(vals)
                        if pb >= y0:
                            qp.setBrush(Qt.BrushStyle.NoBrush)
                            qp.drawRect(QRectF(X(src["f"][i]) - 3, Y(pb) - 3, 6, 6))
            qp.setOpacity(1.0)
        # stems: other products, then fundamental mixer products on top
        if self.s.rbw:
            qp.setOpacity(0.25)   # the RBW trace is the reading; stems stay for hover
        for fund_pass in (False, True):
            col = theme.c("series-2" if fund_pass else "series-1")
            sel_stems = [st for st in in_view if bool(st.get("fund")) == fund_pass and st["p"] >= y0]
            qp.setPen(QPen(col, 2))
            for st in sel_stems:
                x = round(X(st["f"])) + 0.5
                qp.drawLine(QPointF(x, Y(y0)), QPointF(x, Y(st["p"])))
            qp.setPen(QPen(theme.c("surface-1"), 1.5))
            qp.setBrush(col)
            r = 4.5 if fund_pass else 3.5
            for st in sel_stems:
                x, y = X(st["f"]), Y(st["p"])
                qp.drawEllipse(QPointF(x, y), r, r)
                self.points.append({"x": x, "y": y, "f": st["f"], "p": st["p"], "labels": st["labels"],
                                    "kind": "tone" if self.s.solver == "tone" else "td", "fund": st.get("fund")})
        qp.setOpacity(1.0)
        if self.s.rbw and self.s.data:
            n = max(200, round(pw))
            skirts = self.s.data.get("skirts") if self.s.solver == "tone" else None
            tr = rbw_trace(x0, x1, n, [(st["f"], st["p"]) for st in stems], self.s.rbw, skirts, self.s.data.get("pn"))
            path = QPainterPath()
            for i, v in enumerate(tr):
                pt = QPointF(PAD_L + (i + 0.5) / n * pw, Y(v))
                path.lineTo(pt) if i else path.moveTo(pt)
            qp.setBrush(Qt.BrushStyle.NoBrush)
            qp.setPen(QPen(theme.c("text-primary"), 1.5))
            qp.drawPath(path)
        if marks:   # tone overlay rings (shape, not hue, sets them apart)
            qp.setBrush(Qt.BrushStyle.NoBrush)
            qp.setPen(QPen(theme.c("text-muted"), 1.5))
            for m in marks:
                if x0 <= m["f"] <= x1 and m["p"] >= y0:
                    x, y = X(m["f"]), Y(m["p"])
                    qp.drawEllipse(QPointF(x, y), 6, 6)
                    self.points.append({"x": x, "y": y, "f": m["f"], "p": m["p"], "labels": m["labels"], "kind": "tone"})
        # labels on the strongest stems
        qp.setPen(theme.c("text-primary"))
        placed = []
        for st in sorted(in_view, key=lambda t: -t["p"])[:8]:
            if st["p"] < y0 or not st["labels"]:
                continue
            text = st["labels"][0] + (" +" if len(st["labels"]) > 1 else "")
            x, y = X(st["f"]) + 5, Y(st["p"]) - 4
            for q in placed:
                if abs(q[0] - x) < 120 and abs(q[1] - y) < 13:
                    y = q[1] - 13
            placed.append((x, y))
            qp.drawText(QPointF(x, y), text)
        if self.drag and self.drag.get("x1") is not None:
            a, b = sorted((self.drag["x0"], self.drag["x1"]))
            qp.fillRect(QRectF(a, pad_t, b - a, ph), theme.c("select"))
        qp.restore()

        # legend
        for lx, ln, (col, text, shape) in legend:
            y = 12 + 15 * ln
            colr = theme.c(col)
            if shape == "dot":
                qp.setPen(Qt.PenStyle.NoPen)
                qp.setBrush(colr)
                qp.drawEllipse(QPointF(lx, y), 3.5, 3.5)
            elif shape == "line":
                qp.setPen(QPen(colr, 2))
                qp.drawLine(QPointF(lx - 6, y), QPointF(lx + 6, y))
            else:
                qp.setPen(QPen(colr, 1.5))
                qp.setBrush(Qt.BrushStyle.NoBrush)
                qp.drawEllipse(QPointF(lx, y), 5, 5)
            qp.setPen(theme.c("text-secondary"))
            qp.drawText(QPointF(lx + 9, y + 4), text)
        qp.setPen(theme.c("text-muted"))
        qp.drawText(QRectF(0, h - 16, w - 10, 14), Qt.AlignmentFlag.AlignRight,
                    "drag to zoom · wheel to zoom · double-click to reset")
        self.geom = {"x0": x0, "x1": x1, "pw": pw}

    # ------------------------------------------------------------ interaction
    def nearest(self, mx, my):
        best, bd = None, 12 * 12
        for p in self.points:
            d = (p["x"] - mx) ** 2 + (p["y"] - my) ** 2
            if d < bd:
                bd, best = d, p
        return best

    def tooltip(self, p) -> str:
        if p["kind"] == "td":
            who = "time-domain bin"
        elif p["kind"] == "sweep":
            steps = self.s.data["sweep"]["steps"]
            at = p["at"]
            where = ", ".join(f"{k} {fmt_f(v, 7)}" for k, v in step_at(steps, at).items())
            who = (f"band sweep, step {int(at) + 1}/{len(steps)}: {where}" if float(at).is_integer()
                   else f"band sweep, interpolated (level estimated) near step {at + 1:.2f}: {where}")
        else:
            who = "tone solver"
        rows = [f"<b>{fmt_f(p['f'], 9)} · {p['p']:.2f} dBm</b>", f"<span style='color:gray'>{who}</span>"]
        rows += [str(x) for x in p["labels"][:6]]
        if p.get("fund") or (p.get("pr") or {}).get("fund"):
            rows.append("<span style='color:gray'>fundamental mixer product</span>")
        if p["kind"] == "td" and not p["labels"]:
            rows.append("<span style='color:gray'>(not predicted by the tone solver)</span>")
        if p["kind"] == "sweep":
            ps = [v for v in p["pr"]["p"] if v is not None]
            fs = [v for v in p["pr"]["f"] if v is not None]
            rows.append(f"<span style='color:gray'>worst {max(ps):.2f} dBm · spans {fmt_f(min(fs), 6)} – {fmt_f(max(fs), 6)}</span>")
        return "<br>".join(rows)

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self.drag is not None:
            self.drag["x1"] = pos.x()
            QToolTip.hideText()
            self.update()
            return
        p = self.nearest(pos.x(), pos.y())
        if p:
            QToolTip.showText(e.globalPosition().toPoint(), self.tooltip(p), self)
        else:
            QToolTip.hideText()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.drag = {"x0": e.position().x(), "y0": e.position().y(), "x1": None}

    def mouseReleaseEvent(self, e):
        d, self.drag = self.drag, None
        if d is None or self.geom is None:
            return
        g = self.geom
        if d["x1"] is not None and abs(d["x1"] - d["x0"]) > 5:
            f = lambda x: g["x0"] + (x - PAD_L) / g["pw"] * (g["x1"] - g["x0"])  # noqa: E731
            a, b = max(0.0, f(min(d["x0"], d["x1"]))), f(max(d["x0"], d["x1"]))
            if b > a:
                self.s.view = (a, b)
                self.viewChanged.emit()
        else:   # a click: on a sweep track, show only that one (the window toggles it)
            p = self.nearest(d["x0"], d["y0"])
            if p and p["kind"] == "sweep":
                self.trackClicked.emit(p["pr"]["label"])
        self.update()

    def mouseDoubleClickEvent(self, e):
        self.s.view = None
        self.viewChanged.emit()
        self.update()

    def wheelEvent(self, e):
        if not self.geom:
            return
        g = self.geom
        fc = g["x0"] + (e.position().x() - PAD_L) / g["pw"] * (g["x1"] - g["x0"])
        k = 1.25 if e.angleDelta().y() < 0 else 0.8
        self.s.view = (max(0.0, fc - (fc - g["x0"]) * k), fc + (g["x1"] - fc) * k)
        self.viewChanged.emit()
        self.update()

    def leaveEvent(self, _):
        QToolTip.hideText()


class WaveformPlot(QWidget):
    """One record period of the time waveform at the probed port (min/max per pixel bucket)."""

    def __init__(self, state: ViewState, parent=None):
        super().__init__(parent)
        self.s = state
        self.setMinimumHeight(110)
        self.font_ = QFont()
        self.font_.setPixelSize(11)

    def paintEvent(self, _):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        qp.setFont(self.font_)
        w, h = self.width(), self.height()
        qp.fillRect(self.rect(), theme.c("surface-1"))
        d = self.s.data
        if not d or not d.get("waveform"):
            qp.setPen(theme.c("text-muted"))
            qp.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                        "waveform needs the time-domain solver" if self.s.solver == "tone" else "")
            return
        wf = d["waveform"]
        lo_, hi_ = wf["min"], wf["max"]
        n = len(lo_)
        lo, hi = min(lo_), max(hi_)
        if not hi > lo:
            lo, hi = -1e-9, 1e-9
        m = max(abs(lo), abs(hi)) * 1.1
        pl, pr, pt, pb = PAD_L, PAD_R, 10, 24
        pw, ph = w - pl - pr, h - pt - pb
        X = lambda i: pl + i / max(n - 1, 1) * pw  # noqa: E731
        Y = lambda v: pt + (m - v) / (2 * m) * ph  # noqa: E731
        qp.setPen(QPen(theme.c("grid"), 1))
        qp.drawLine(QPointF(pl, Y(0)), QPointF(w - pr, Y(0)))
        vs, vu = (1, "V") if m >= 1 else (1e-3, "mV") if m >= 1e-3 else (1e-6, "µV")
        qp.setPen(theme.c("text-muted"))
        qp.drawText(QRectF(0, Y(m / 1.1) - 8, pl - 6, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    f"{m / 1.1 / vs:.3g} {vu}")
        qp.drawText(QRectF(0, Y(-m / 1.1) - 8, pl - 6, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    f"{-m / 1.1 / vs:.3g}")
        qp.drawText(QRectF(pl, h - pb + 5, 40, 14), Qt.AlignmentFlag.AlignLeft, "0")
        span = wf["span"]
        unit = next(((s, u) for s, u in ((1, "s"), (1e-3, "ms"), (1e-6, "µs"), (1e-9, "ns")) if span >= s), (1e-12, "ps"))
        qp.drawText(QRectF(pl, h - pb + 5, pw, 14), Qt.AlignmentFlag.AlignRight,
                    f"{span / unit[0]:.4g} {unit[1]} (one period of the record)")
        qp.setPen(QPen(theme.c("series-1"), 1.2))
        path = QPainterPath()
        if lo_ == hi_:
            for i in range(n):
                pt_ = QPointF(X(i), Y(hi_[i]))
                path.lineTo(pt_) if i else path.moveTo(pt_)
        else:
            for i in range(n):
                x = round(X(i)) + 0.5
                path.moveTo(QPointF(x, Y(hi_[i])))
                path.lineTo(QPointF(x, Y(lo_[i]) + 0.5))
        qp.drawPath(path)




class XYView(QWidget):
    """An X/Y plot block's trace: the detector reading (y) against the swept tone's frequency (x),
    one point per band-sweep step, with the animation's current step marked."""

    def __init__(self, state: ViewState, name: str, parent=None):
        super().__init__(parent)
        self.s, self.name = state, name
        self.setMouseTracking(True)
        self.setMinimumHeight(140)
        self.font_ = QFont()
        self.font_.setPixelSize(11)
        self.pts: list[tuple] = []

    def record(self):
        for r in (self.s.data or {}).get("xy", []) or []:
            if r["name"] == self.name:
                return r
        return None

    def paintEvent(self, _):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        qp.setFont(self.font_)
        w, h = self.width(), self.height()
        qp.fillRect(self.rect(), theme.c("surface-1"))
        r = self.record()
        self.pts = []
        if r is None or not r["x"]:
            qp.setPen(theme.c("text-muted"))
            qp.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                        (r or {}).get("note") or "wire a swept source's sweep port to x and a detector's out to y")
            return
        xs, ys = r["x"], r["y"]
        valid = [y for y in ys if y is not None]
        linear = r.get("y_unit") == "V"
        if linear:   # volts: from 0 up, in a unit that suits the largest reading
            vmax = max(valid, default=0.0) or 1e-6
            vs, vu = next(((s_, u) for s_, u in ((1, "V"), (1e-3, "mV"), (1e-6, "µV")) if vmax >= s_), (1e-9, "nV"))
            y0, y1 = 0.0, nice_step(vmax * 1.1, 1) * math.ceil(vmax * 1.1 / nice_step(vmax * 1.1, 1))
        else:
            y0 = self.s.floor
            top = max(valid) if valid else y0 + 10
            y1 = math.ceil((max(top, y0 + 10) + 5) / 10) * 10
        x0, x1 = min(xs), max(xs)
        if x1 <= x0:
            x1 = x0 + 1
        pl, pr, pt, pb = PAD_L, PAD_R, 12, 36
        pw, ph = w - pl - pr, h - pt - pb
        X = lambda f: pl + (f - x0) / (x1 - x0) * pw  # noqa: E731
        Y = lambda p: pt + (y1 - max(p, y0)) / (y1 - y0) * ph  # noqa: E731
        ystep = nice_step(y1 - y0, ph / 30)
        v = math.ceil(y0 / ystep) * ystep
        while v <= y1 + 1e-12:
            qp.setPen(QPen(theme.c("grid"), 1))
            qp.drawLine(QPointF(pl, Y(v)), QPointF(w - pr, Y(v)))
            qp.setPen(theme.c("text-muted"))
            label = f"{float(f'{v / vs:.6g}'):g}" if linear else str(round(v))
            qp.drawText(QRectF(0, Y(v) - 8, pl - 6, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, label)
            v += ystep
        us, uu = unit_for(x1)
        xstep = nice_step(x1 - x0, pw / 90)
        v = math.ceil(x0 / xstep) * xstep
        while v <= x1 + 1e-9:
            qp.setPen(QPen(theme.c("grid"), 1))
            qp.drawLine(QPointF(X(v), pt), QPointF(X(v), h - pb))
            qp.setPen(theme.c("text-muted"))
            qp.drawText(QRectF(X(v) - 40, h - pb + 4, 80, 14), Qt.AlignmentFlag.AlignHCenter, f"{float(f'{v / us:.8g}'):g}")
            v += xstep
        qp.setPen(theme.c("text-secondary"))
        qp.drawText(QRectF(pl, h - 18, pw, 14), Qt.AlignmentFlag.AlignHCenter, f"{r['x_label']} ({uu})")
        qp.save()
        qp.translate(14, pt + ph / 2)
        qp.rotate(-90)
        qp.drawText(QRectF(-ph / 2, -8, ph, 16), Qt.AlignmentFlag.AlignHCenter,
                    r["y_label"].replace("(V)", f"({vu})") if linear else r["y_label"])
        qp.restore()
        qp.save()
        qp.setClipRect(QRectF(pl, pt, pw, ph))
        path = QPainterPath()
        for i, (x, y) in enumerate(zip(xs, ys)):   # nothing in the detector's band: drawn at the floor
            p = QPointF(X(x), Y(y0 if y is None else y))
            path.lineTo(p) if i else path.moveTo(p)
            self.pts.append((p.x(), p.y(), x, y))
        qp.setPen(QPen(theme.c("series-1"), 1.8))
        qp.setBrush(Qt.BrushStyle.NoBrush)
        qp.drawPath(path)
        if len(xs) <= 80:   # few steps: show each one
            qp.setPen(Qt.PenStyle.NoPen)
            qp.setBrush(theme.c("series-1"))
            for px, py, _, _ in self.pts:
                qp.drawEllipse(QPointF(px, py), 2.5, 2.5)
        fx = (self.s.extra.get("frame_freqs") or {}).get(r.get("x_tone"))
        if fx is not None and x0 <= fx <= x1:   # the animation's step, placed by the swept tone's frequency
            qp.setPen(QPen(theme.c("series-2"), 1.5, Qt.PenStyle.DashLine))
            qp.drawLine(QPointF(X(fx), pt), QPointF(X(fx), h - pb))
            i = min(range(len(xs)), key=lambda k: abs(xs[k] - fx))
            if ys[i] is not None or linear:
                qp.setPen(QPen(theme.c("surface-1"), 1.5))
                qp.setBrush(theme.c("series-2"))
                qp.drawEllipse(QPointF(X(xs[i]), Y(y0 if ys[i] is None else ys[i])), 5, 5)
        qp.restore()

    def mouseMoveEvent(self, e):
        best, bd = None, 15 * 15
        for px, py, x, y in self.pts:
            d = (px - e.position().x()) ** 2 + (py - e.position().y()) ** 2
            if d < bd:
                bd, best = d, (x, y)
        if best:
            r = self.record() or {}
            from .schematic import fmt_volts
            level = ("nothing in the detector's band" if best[1] is None
                     else fmt_volts(best[1]) if r.get("y_unit") == "V" else f"{best[1]:.2f} dBm")
            QToolTip.showText(e.globalPosition().toPoint(), f"<b>{fmt_f(best[0], 9)}</b><br>{level}", self)
        else:
            QToolTip.hideText()


__all__ = ["SpectrumPlot", "ViewState", "WaveformPlot", "XYView", "fmt_f"]
