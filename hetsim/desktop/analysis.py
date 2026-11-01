"""Noise figure, VSWR and the receiver budget for the circuit as the sliders have it: tables and charts."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QFont, QFontDatabase, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QComboBox, QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QPushButton, QSpinBox, QSplitter, QToolTip, QVBoxLayout, QWidget)

from ..viewer.analysis import analyse, options
from . import theme
from .plot import fmt_f, nice_step, unit_for

SERIES = ("series-1", "series-2")   # identity follows the series' position: typical, then worst case


def _num(v: float) -> str:
    return f"{float(f'{v:.5g}'):g}"


class LinePlot(QWidget):
    """One chart from the analysis: one y unit, a legend above for two series, a crosshair on hover."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.chart: dict | None = None
        self.setMinimumHeight(200)
        self.setMouseTracking(True)
        self.font_ = QFont()
        self.font_.setPixelSize(11)
        self.hover: int | None = None
        self.geom = None

    def set_chart(self, chart: dict | None):
        self.chart, self.hover = chart, None
        self.update()

    def _range(self):
        ys = [y for s in self.chart["series"] for y in s["y"] if y is not None]
        if not ys:
            return None
        lo, hi = min(ys), max(ys)
        pad = max((hi - lo) * 0.08, 0.5 if hi - lo < 1 else 0)
        return lo - pad, hi + pad

    def paintEvent(self, _):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        qp.setFont(self.font_)
        w, h = self.width(), self.height()
        qp.fillRect(self.rect(), theme.c("surface-1"))
        self.geom = None
        c = self.chart
        rng = self._range() if c else None
        if rng is None:
            qp.setPen(theme.c("text-muted"))
            qp.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                        "No values to chart" if c else "Noise figure with a band, a VSWR sweep or a varied parameter draws a chart")
            return
        xs = c["x"]
        y0, y1 = rng
        x0, x1 = min(xs), max(xs)
        if x1 <= x0:
            x1 = x0 + 1
        pl, pr, pt, pb = 58, 16, 34, 36
        pw, ph = max(w - pl - pr, 1), max(h - pt - pb, 1)
        X = lambda v: pl + (v - x0) / (x1 - x0) * pw  # noqa: E731
        Y = lambda v: pt + (y1 - v) / (y1 - y0) * ph  # noqa: E731
        self.geom = (X, pl, pt, ph, pw)
        qp.setPen(theme.c("text-primary"))
        qp.drawText(QRectF(pl, 2, pw, 14), Qt.AlignmentFlag.AlignLeft, c["title"])
        if len(c["series"]) > 1:   # identity is never colour alone: names beside their marks
            lx = pl
            for i, s in enumerate(c["series"]):
                qp.setPen(QPen(theme.c(SERIES[i]), 2))
                qp.drawLine(QPointF(lx, 24), QPointF(lx + 16, 24))
                qp.setPen(theme.c("text-secondary"))
                qp.drawText(QRectF(lx + 20, 17, 120, 14), Qt.AlignmentFlag.AlignLeft, s["name"])
                lx += 28 + qp.fontMetrics().horizontalAdvance(s["name"])
        step = nice_step(y1 - y0, ph / 30)
        v = math.ceil(y0 / step) * step
        while v <= y1 + 1e-12:
            qp.setPen(QPen(theme.c("grid"), 1))
            qp.drawLine(QPointF(pl, Y(v)), QPointF(w - pr, Y(v)))
            qp.setPen(theme.c("text-muted"))
            qp.drawText(QRectF(0, Y(v) - 8, pl - 6, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, _num(v))
            v += step
        hz = c["x_unit"] == "Hz"
        us, uu = unit_for(x1) if hz else (1, "")
        step = nice_step(x1 - x0, pw / 90)
        v = math.ceil(x0 / step) * step
        while v <= x1 + 1e-9 * max(abs(x1), 1):
            qp.setPen(theme.c("text-muted"))
            qp.drawText(QRectF(X(v) - 40, h - pb + 4, 80, 14), Qt.AlignmentFlag.AlignHCenter, _num(v / us))
            v += step
        qp.setPen(theme.c("text-secondary"))
        qp.drawText(QRectF(pl, h - 18, pw, 14), Qt.AlignmentFlag.AlignHCenter, f"{c['x_label']} ({uu})" if uu else c["x_label"])
        qp.save()
        qp.translate(14, pt + ph / 2)
        qp.rotate(-90)
        qp.drawText(QRectF(-ph / 2, -8, ph, 16), Qt.AlignmentFlag.AlignHCenter, c["y_label"])
        qp.restore()
        qp.save()
        qp.setClipRect(QRectF(pl - 6, pt - 6, pw + 12, ph + 12))
        for i, s in enumerate(c["series"]):
            path = QPainterPath()
            pen_up = True
            for x, y in zip(xs, s["y"]):
                if y is None:
                    pen_up = True
                    continue
                p = QPointF(X(x), Y(y))
                path.moveTo(p) if pen_up else path.lineTo(p)
                pen_up = False
            qp.setPen(QPen(theme.c(SERIES[i]), 2))
            qp.setBrush(Qt.BrushStyle.NoBrush)
            qp.drawPath(path)
            if len(xs) <= 41:
                qp.setPen(QPen(theme.c("surface-1"), 2))
                qp.setBrush(theme.c(SERIES[i]))
                for x, y in zip(xs, s["y"]):
                    if y is not None:
                        qp.drawEllipse(QPointF(X(x), Y(y)), 4, 4)
        if self.hover is not None:
            x = xs[self.hover]
            qp.setPen(QPen(theme.c("text-muted"), 1))
            qp.drawLine(QPointF(X(x), pt), QPointF(X(x), pt + ph))
        qp.restore()

    def mouseMoveEvent(self, e):
        if not self.geom or not self.chart:
            return
        X, _, pt, ph, _ = self.geom
        xs = self.chart["x"]
        mx = e.position().x()
        self.hover = min(range(len(xs)), key=lambda i: abs(X(xs[i]) - mx))
        x = xs[self.hover]
        head = fmt_f(x, 9) if self.chart["x_unit"] == "Hz" else f"{self.chart['x_label']} {_num(x)}"
        unit = self.chart["y_unit"]
        unit = unit if unit.startswith(":") else f" {unit}"
        rows = [f"{s['name']}: {'-' if s['y'][self.hover] is None else format(s['y'][self.hover], '.2f')}{unit}"
                for s in self.chart["series"]]
        QToolTip.showText(e.globalPosition().toPoint(), f"<b>{head}</b><br>" + "<br>".join(rows), self)
        self.update()

    def leaveEvent(self, _):
        self.hover = None
        QToolTip.hideText()
        self.update()


class AnalysisDialog(QDialog):
    def __init__(self, model, edits: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Noise figure, VSWR and receiver budget")
        self.resize(860, 760)
        self.model, self.edits = model, edits
        self.opts = options(model)
        self.source = QComboBox()
        self.source.addItems(list(self.opts["sources"]))
        self.tone = QComboBox()
        self.source.currentTextChanged.connect(self._tones)
        self.port = QComboBox()
        self.port.addItems(self.opts["outputs"])
        self.port.setCurrentIndex(self.port.count() - 1)
        self.band = QLineEdit()
        self.band.setPlaceholderText("output band, e.g. 10.6MHz:10.8MHz (noise figure; empty: spot figure)")
        self.bw = QLineEdit()
        self.bw.setPlaceholderText("channel, e.g. 200kHz (empty: per Hz)")
        self.offset = QLineEdit()
        self.offset.setPlaceholderText("blocker offset, e.g. 1MHz (empty: in-band)")
        self.range = QLineEdit()
        self.range.setPlaceholderText("VSWR sweep, e.g. 88MHz:108MHz")
        self.look = QComboBox()
        self.look.addItems(self.opts["ports"])
        self.vary = QComboBox()
        self.vary.addItems([v["path"] for v in self.opts["vary"]])
        self.vary.currentIndexChanged.connect(self._vary_range)
        self.v_from, self.v_to = QLineEdit(), QLineEdit()
        self.v_points = QSpinBox(minimum=2, maximum=101, value=11)

        form = QFormLayout()
        form.addRow("Source, signal tone", _row(self.source, self.tone))
        form.addRow("Output port", self.port)
        form.addRow("Band", self.band)
        form.addRow("Budget", _row(self.bw, self.offset))
        form.addRow("Look into", _row(self.look, self.range))
        form.addRow("Vary", _row(self.vary, QLabel("from"), self.v_from, QLabel("to"), self.v_to, QLabel("points"), self.v_points))
        self.runs = QSpinBox(minimum=2, maximum=100000, value=200)
        self.specs = QLineEdit()
        self.specs.setPlaceholderText("limits for the yield, e.g. nf_db<=4, sfdr_db>=80")
        n = len(self.opts["tolerances"])
        form.addRow("Tolerances", _row(QLabel(f"{n} in the netlist; runs"), self.runs, self.specs))
        buttons = QHBoxLayout()
        for text, fn in (("Noise figure", self.run_noise), ("VSWR at every wire", self.run_wires),
                         ("VSWR sweep", self.run_match), ("Receiver budget", self.run_budget),
                         ("Vary parameter", self.run_vary), ("Monte Carlo", self.run_tolerance)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            buttons.addWidget(b)
        buttons.addStretch(1)
        self.pick = QComboBox()
        self.pick.currentIndexChanged.connect(self._show_chart)
        self.plot = LinePlot()
        self.out = QPlainTextEdit(readOnly=True)
        self.out.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.out.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.pick)
        tl.addWidget(self.plot, 1)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(self.out)
        split.setSizes([320, 260])
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addLayout(buttons)
        lay.addWidget(split, 1)
        self.charts: list = []
        self._tones(self.source.currentText())
        self._vary_range()
        self._set_charts([])

    def _tones(self, name: str):
        self.tone.clear()
        self.tone.addItems(self.opts["sources"].get(name, []))

    def _vary_range(self):
        i = self.vary.currentIndex()
        if i >= 0:
            v = self.opts["vary"][i]
            self.v_from.setText(_num(v["min"]))
            self.v_to.setText(_num(v["max"]))

    def _set_charts(self, charts: list):
        self.charts = charts
        keep = self.pick.currentText()
        self.pick.blockSignals(True)
        self.pick.clear()
        self.pick.addItems([c["title"] for c in charts])
        if keep in [c["title"] for c in charts]:
            self.pick.setCurrentText(keep)
        self.pick.blockSignals(False)
        self.pick.setVisible(len(charts) > 1)
        self._show_chart()

    def _show_chart(self, *_):
        i = self.pick.currentIndex()
        self.plot.set_chart(self.charts[i] if 0 <= i < len(self.charts) else None)

    def _run(self, kind: str, **kw):
        r = analyse(self.model, {"kind": kind, "params": self.edits, "source": self.source.currentText(),
                                 "tone": self.tone.currentText(), **kw})
        if "error" in r:
            self.out.setPlainText(r["error"])
            self._set_charts([])
            return
        self.out.setPlainText(r["text"])
        self._set_charts(r["charts"])

    def run_noise(self):
        self._run("noise", port=self.port.currentText(), band=self.band.text())

    def run_wires(self):
        self._run("wires")

    def run_match(self):
        self._run("match", port=self.look.currentText(), range=self.range.text())

    def _budget_args(self) -> dict:
        return {"port": self.port.currentText(), "bw": self.bw.text().strip(), "offset": self.offset.text().strip()}

    def run_budget(self):
        self._run("budget", **self._budget_args())

    def run_vary(self):
        self._run("vary", param=self.vary.currentText(), points=self.v_points.value(), **self._budget_args(),
                  **{"from": self.v_from.text(), "to": self.v_to.text()})

    def run_tolerance(self):
        specs = [s.strip() for s in self.specs.text().split(",") if s.strip()]
        self._run("tolerance", runs=self.runs.value(), specs=specs, **self._budget_args())


def _row(*widgets) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        lay.addWidget(x)
    return w
