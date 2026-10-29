"""The parameter panel: one collapsible section per block, a control per adjustable parameter.

Built from ViewerModel.describe() groups, the same records the browser viewer uses: freq,
number and int sliders, choices, booleans, phase-noise tables; optional parameters with an
on/off box; rows shown only `when` another parameter allows; band-bound tone sliders; muted
tones dimmed; click a label to set the slider's limits.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFrame, QGraphicsOpacityEffect, QGridLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QScrollArea, QSlider, QToolButton, QVBoxLayout,
                               QWidget)

from ..units import parse_freq
from .plot import fmt_f

STEPS = 1000   # slider resolution for continuous parameters


def parse_f(text: str) -> float:
    try:
        return parse_freq(text.replace(" ", ""))
    except (ValueError, TypeError):
        return math.nan


def show_val(p: dict) -> str:
    if p["kind"] == "freq":
        return fmt_f(p["value"], 7)
    if p["kind"] == "int":
        return str(int(p["value"]))
    nd = 2 if p.get("step", 1) < 0.1 else 1
    return f"{float(p['value']):.{nd}f}" + (f" {p['unit']}" if p.get("unit") else "")


def visible(group: dict, p: dict) -> bool:
    for key, allowed in (p.get("when") or {}).items():
        other = next((q for q in group["params"] if q["path"] == key), None)
        if other is not None and other["value"] not in allowed:
            return False
    return True


def muted(group: dict, p: dict) -> bool:
    """A muted tone's other controls are dimmed (they stay editable)."""
    return any(q.get("mutes") and q["value"] and q is not p and p["path"].startswith(q["mutes"]) for q in group["params"])


class ClickLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, e):
        self.clicked.emit()


class Row(QWidget):
    """Controls for one parameter `p` (a describe() record, edited in place)."""

    def __init__(self, panel: "ParamPanel", group: dict, p: dict):
        super().__init__()
        self.panel, self.g, self.p = panel, group, p
        self.slider = self.edit = self.box = self.limits = None
        lay = QGridLayout(self)
        lay.setContentsMargins(12, 2, 12, 2)
        lay.setHorizontalSpacing(6)
        lay.setVerticalSpacing(1)
        self.en = QCheckBox()
        self.en.setChecked(bool(p["enabled"]))
        self.en.setToolTip("include this parameter" if p["optional"] else "")
        sp = self.en.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self.en.setSizePolicy(sp)
        self.en.setVisible(bool(p["optional"]))
        self.en.toggled.connect(self._enabled)
        self.lab = ClickLabel(p["label"])
        self.lab.setToolTip(p["label"])
        lay.addWidget(self.en, 0, 0)
        lay.addWidget(self.lab, 0, 1)
        lay.setColumnStretch(1, 1)
        kind = p["kind"]
        if kind in ("freq", "number", "int", "level"):
            self.lab.setToolTip(p["label"] + " · click to set the slider's limits")
            self.lab.setCursor(Qt.CursorShape.PointingHandCursor)
            if p.get("limits"):
                self.lab.setText(f"{p['label']}  <span style='color:#2a78d6'>[limits]</span>")
            self.lab.clicked.connect(self._toggle_limits)
            self.edit = QLineEdit(show_val(p))
            self.edit.setFixedWidth(100)
            self.edit.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.edit.editingFinished.connect(self._typed)
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.valueChanged.connect(self._slid)
            lay.addWidget(self.edit, 0, 2)
            lay.addWidget(self.slider, 1, 1, 1, 2)
            self.set_range()
        elif kind == "choice":
            self.edit = QComboBox()
            self.edit.addItems([str(c) for c in p["choices"]])
            self.edit.setCurrentText(str(p["value"]))
            self.edit.setFixedWidth(100)
            self.edit.currentTextChanged.connect(self._chose)
            lay.addWidget(self.edit, 0, 2)
        elif kind == "bool":
            self.edit = QCheckBox()
            self.edit.setChecked(bool(p["value"]))
            self.edit.toggled.connect(self._ticked)
            lay.addWidget(self.edit, 0, 2, Qt.AlignmentFlag.AlignRight)
        elif kind == "pn":
            self.box = QWidget()
            QVBoxLayout(self.box).setContentsMargins(0, 0, 0, 0)
            self.pn_table = None
            lay.addWidget(self.box, 1, 1, 1, 2)
            self.render_pn()
        self.fx = QGraphicsOpacityEffect(self)
        self.fx.setOpacity(1.0)
        self.setGraphicsEffect(self.fx)

    # ------------------------------------------------------------ slider mapping
    def _log(self):
        return self.p["kind"] == "freq" and self.p.get("scale") == "log" and self.p["min"] > 0

    def set_range(self):
        p, s = self.p, self.slider
        if s is None:
            return
        s.blockSignals(True)
        if p["kind"] == "int":
            s.setRange(int(p["min"]), int(p["max"]))
            s.setValue(int(p["value"]))
        else:
            s.setRange(0, STEPS)
            lo, hi, v = p["min"], p["max"], p["value"]
            if hi <= lo:
                pos = 0
            elif self._log():
                pos = round(STEPS * math.log(max(v, lo) / lo) / math.log(hi / lo))
            else:
                pos = round(STEPS * (v - lo) / (hi - lo))
            s.setValue(max(0, min(STEPS, pos)))
        s.blockSignals(False)

    def _from_slider(self, i):
        p = self.p
        if p["kind"] == "int":
            return int(i)
        lo, hi = p["min"], p["max"]
        if self._log():
            return lo * (hi / lo) ** (i / STEPS)
        v = lo + (hi - lo) * i / STEPS
        step = p.get("step")
        if p["kind"] == "number" and step:
            v = round(v / step) * step
        return v

    # ------------------------------------------------------------ edits
    def _slid(self, i):
        self.p["value"] = self._from_slider(i)
        self.edit.setText(show_val(self.p))
        self.panel.edited(self.g)

    def _typed(self):
        p = self.p
        v = parse_f(self.edit.text()) if p["kind"] == "freq" else _float(self.edit.text())
        if not math.isfinite(v):
            self.edit.setText(show_val(p))
            return
        v = max(1, round(v)) if p["kind"] == "int" else v
        if p.get("limits") or p.get("_in_band"):
            v = min(max(v, p["min"]), p["max"])   # keep to the limits
        if v == p["value"]:
            self.edit.setText(show_val(p))
            return
        p["value"] = v
        if v < p["min"]:
            p["min"] = v / 2 if self._log() else v
        if v > p["max"]:
            p["max"] = v * 2 if self._log() else v
        self.set_range()
        self.edit.setText(show_val(p))
        self.panel.edited(self.g)

    def _chose(self, text):
        self.p["value"] = text
        self.panel.edited(self.g)

    def _ticked(self, on):
        self.p["value"] = on
        self.panel.edited(self.g)

    def _enabled(self, on):
        self.p["enabled"] = on
        self.panel.edited(self.g)

    # ------------------------------------------------------------ state from outside
    def set_value(self, v):
        """A value from the server (e.g. a frequency snapped to the FFT grid)."""
        self.p["value"] = v
        if self.edit is not None and not self.edit.hasFocus() and isinstance(self.edit, QLineEdit):
            self.edit.setText(show_val(self.p))
        if self.slider is not None and not self.slider.isSliderDown():
            self.set_range()

    def refresh(self):
        p, g = self.p, self.g
        if p.get("band"):
            self._band_range()
        off = p["optional"] and not p["enabled"]
        self.setVisible(visible(g, p))
        self.fx.setOpacity(0.45 if off or muted(g, p) else 1.0)
        for wdg in (self.edit, self.slider, self.box):
            if wdg is not None and p["kind"] not in ("choice", "bool"):
                wdg.setEnabled(not off)
        if self.box is not None:
            self.box.setVisible(not off)

    def _band_range(self):
        """A tuned or swept tone's slider spans its band; a cw tone gets its original range back."""
        p, g = self.p, self.g
        lo, hi, mode = (next((q for q in g["params"] if q["path"] == path), None) for path in p["band"])
        p.setdefault("_min0", p["min"])
        p.setdefault("_max0", p["max"])
        p["_in_band"] = bool(lo and hi and mode and mode["value"] in ("tune", "sweep"))
        a, b = (min(lo["value"], hi["value"]), max(lo["value"], hi["value"])) if p["_in_band"] else (p["_min0"], p["_max0"])
        if (a, b) == (p["min"], p["max"]):
            return
        p["min"], p["max"] = a, b
        if p["_in_band"]:
            p["value"] = min(max(p["value"], a), b)
        self.set_range()
        if not self.edit.hasFocus():
            self.edit.setText(show_val(p))

    # ------------------------------------------------------------ limits editor
    def _toggle_limits(self):
        if self.limits is not None:
            self.limits.deleteLater()
            self.limits = None
            return
        p = self.p
        fmt = (lambda v: fmt_f(v, 7)) if p["kind"] == "freq" else (lambda v: f"{float(v):.6g}")
        parse = parse_f if p["kind"] == "freq" else _float
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 4)
        lo, hi = QLineEdit(fmt(p["min"])), QLineEdit(fmt(p["max"]))
        for e in (lo, hi):
            e.setFixedWidth(90)
        save = QPushButton("Save limits")
        save.setToolTip("Store these limits in the netlist; the slider keeps to them")
        h.addWidget(QLabel("min"))
        h.addWidget(lo)
        h.addWidget(QLabel("max"))
        h.addWidget(hi)
        h.addWidget(save)

        def do_save():
            a, b = parse(lo.text()), parse(hi.text())
            if not (math.isfinite(a) and math.isfinite(b) and a < b):
                self.panel.message.emit(f"limits for {self.g['block']} {p['label']}: need min < max")
                return
            self.panel.limitsRequested.emit(self.g["block"], p["path"], [a, b])
        save.clicked.connect(do_save)
        for e in (lo, hi):
            e.returnPressed.connect(do_save)
        if p.get("limits"):
            clear = QPushButton("Clear")
            clear.clicked.connect(lambda: self.panel.limitsRequested.emit(self.g["block"], p["path"], None))
            h.addWidget(clear)
        h.addStretch(1)
        self.layout().addWidget(box, 2, 1, 1, 2)
        self.limits = box
        lo.setFocus()

    # ------------------------------------------------------------ phase-noise table
    def render_pn(self):
        p = self.p
        if self.pn_table is not None:   # rebuilt whole: a new table replaces the old
            self.box.layout().removeWidget(self.pn_table)
            self.pn_table.deleteLater()
        self.pn_table = QWidget()
        self.box.layout().addWidget(self.pn_table)
        grid = QGridLayout(self.pn_table)
        grid.setContentsMargins(0, 0, 0, 4)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)
        for col, t in enumerate(("offset", "dBc/Hz")):
            h = QLabel(t)
            h.setStyleSheet("color: gray; font-size: 11px")
            grid.addWidget(h, 0, col)

        def commit():
            p["value"].sort(key=lambda v: v[0])
            p["value"] = [v for i, v in enumerate(p["value"]) if i == 0 or v[0] > p["value"][i - 1][0]]
            QTimer.singleShot(0, self, self.render_pn)   # dropped if the row is gone by then
            self.panel.edited(self.g)

        for i, pt in enumerate(p["value"]):
            off, lvl = QLineEdit(fmt_f(pt[0], 4)), QLineEdit(f"{float(pt[1]):g}")
            for e in (off, lvl):
                e.setAlignment(Qt.AlignmentFlag.AlignRight)
            dl = QToolButton()
            dl.setText("×")
            dl.setToolTip("remove this point")
            dl.setEnabled(len(p["value"]) > 1)

            def set_off(pt=pt, e=off):
                v = parse_f(e.text())
                if math.isfinite(v) and v > 0 and v != pt[0]:
                    pt[0] = v
                    commit()

            def set_lvl(pt=pt, e=lvl):
                v = _float(e.text())
                if math.isfinite(v) and v != pt[1]:
                    pt[1] = v
                    commit()

            def remove(i=i):
                del p["value"][i]
                commit()
            off.editingFinished.connect(set_off)
            lvl.editingFinished.connect(set_lvl)
            dl.clicked.connect(remove)
            grid.addWidget(off, i + 1, 0)
            grid.addWidget(lvl, i + 1, 1)
            grid.addWidget(dl, i + 1, 2)
        btns = QHBoxLayout()

        def add():
            last = p["value"][-1] if p["value"] else [1e3, -70.0]
            p["value"].append([last[0] * 10, last[1] - 10])
            commit()

        def shift(db):
            for v in p["value"]:
                v[1] += db
            commit()
        for text, tip, fn in (("+ point", "add a point a decade further out, 10 dB lower", add),
                              ("−5 dB", "lower the whole profile by 5 dB", lambda: shift(-5)),
                              ("+5 dB", "raise the whole profile by 5 dB", lambda: shift(5))):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            btns.addWidget(b)
        btns.addStretch(1)
        grid.addLayout(btns, len(p["value"]) + 1, 0, 1, 3)


def _float(text: str) -> float:
    try:
        return float(str(text).split()[0])
    except (ValueError, IndexError):
        return math.nan


class Section(QWidget):
    """A block's parameters under a header that folds them away."""

    def __init__(self, group: dict):
        super().__init__()
        self.g = group
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 4)
        v.setSpacing(0)
        self.head = QToolButton()
        self.head.setText(f"  {group['block']}   {group['type']}")
        self.head.setCheckable(True)
        self.head.setChecked(True)
        self.head.setArrowType(Qt.ArrowType.DownArrow)
        self.head.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.head.setAutoRaise(True)
        self.head.setStyleSheet("QToolButton { font-weight: 600; padding: 4px; text-align: left; }")
        self.head.toggled.connect(self.fold)
        self.body = QWidget()
        self.body_lay = QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(0, 0, 0, 0)
        self.body_lay.setSpacing(0)
        v.addWidget(self.head)
        v.addWidget(self.body)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Plain)
        line.setStyleSheet("color: rgba(128,128,128,60)")
        v.addWidget(line)

    def fold(self, open_):
        self.body.setVisible(open_)
        self.head.setArrowType(Qt.ArrowType.DownArrow if open_ else Qt.ArrowType.RightArrow)


class ParamPanel(QScrollArea):
    changed = Signal()                          # a value changed: simulate
    touched = Signal()                          # the user edited a value
    limitsRequested = Signal(str, str, object)  # block, path, [min, max] or None
    message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(320)
        self.groups: list[dict] = []
        self.rows: dict[tuple, Row] = {}
        self.sections: dict[str, Section] = {}

    def build(self, groups: list[dict]):
        self.groups = groups
        self.rows, self.sections = {}, {}
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 4, 0, 16)
        lay.setSpacing(0)
        for g in groups:
            sec = Section(g)
            self.sections[g["block"]] = sec
            for p in g["params"]:
                row = Row(self, g, p)
                self.rows[(g["block"], p["path"])] = row
                sec.body_lay.addWidget(row)
            lay.addWidget(sec)
        lay.addStretch(1)
        self.setWidget(inner)
        for g in groups:
            self.refresh_group(g)

    def refresh_group(self, g):
        for p in g["params"]:
            self.rows[(g["block"], p["path"])].refresh()

    def edited(self, g):
        self.refresh_group(g)   # a band edge moves its tone's slider range, a choice shows rows...
        self.touched.emit()
        self.changed.emit()

    def collect(self) -> dict:
        out = {}
        for g in self.groups:
            out[g["block"]] = {p["path"]: (p["value"] if (not p["optional"] or p["enabled"]) and visible(g, p) else None)
                               for p in g["params"]}
        return out

    def apply_snapped(self, snapped: dict):
        for block, kv in (snapped or {}).items():
            for path, f in kv.items():
                row = self.rows.get((block, path))
                if row is not None and abs(row.p["value"] - f) > 1e-6:
                    row.set_value(f)

    def focus_block(self, name: str):
        sec = self.sections.get(name)
        if sec is None:
            return
        sec.head.setChecked(True)
        QTimer.singleShot(0, sec, lambda: self.ensureWidgetVisible(sec, 0, 0))   # after the layout settles
        sec.head.setStyleSheet("QToolButton { font-weight: 600; padding: 4px; text-align: left; "
                               "background: rgba(42,120,214,40); }")
        QTimer.singleShot(900, sec.head, lambda: sec.head.setStyleSheet(
            "QToolButton { font-weight: 600; padding: 4px; text-align: left; }"))


def carry_values(prev: list[dict], groups: list[dict], renames: dict) -> None:
    """After a schematic edit: keep slider values the user had set, matched by block and path."""
    old = {(renames.get(g["block"], g["block"]), p["path"]): p for g in prev for p in g["params"]}
    for g in groups:
        for p in g["params"]:
            o = old.get((g["block"], p["path"]))
            if o is None or o["kind"] != p["kind"]:
                continue
            p["value"], p["enabled"] = o["value"], o["enabled"]
            if p.get("limits"):
                p["value"] = min(max(p["value"], p["min"]), p["max"])
            elif "min" in o and not o.get("limits") and not p.get("band"):
                p["min"], p["max"] = o["min"], o["max"]
