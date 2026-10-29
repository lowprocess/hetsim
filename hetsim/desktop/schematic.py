"""Schematic editor: the circuit drawn with RF symbols on a QGraphicsScene.

Same operations as the browser viewer: drag a block to move it, drag from port to port to
wire, click a port to probe it, click a block to open its parameters, Del deletes the
selection, double-click renames, wheel zooms, F fits. Every edit is a ViewerModel.edit()
operation, emitted as `op` for the window to apply.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QBrush, QFont, QKeySequence, QPainter, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem, QGraphicsScene,
                               QGraphicsView, QInputDialog)

from . import theme
from .params import visible
from .plot import fmt_f
from .tracks import BH, BW, COL_PITCH, auto_layout, port_layout, route, route_bus

GRID = 20
LABEL_W = 146   # width of the name and detail lines under a block (the layout's column pitch is 150)
FIT_MIN = 0.85  # Fit doesn't zoom out further: the details stay readable


def _snap(v: float) -> float:
    return round(v / GRID) * GRID


def _tilde(path: QPainterPath, y: float):
    """Three half-waves from x=20 to 44 at height y."""
    path.moveTo(20, y)
    for x in (20, 28, 36):
        path.quadTo(x + 4, y - 6 if x != 28 else y + 6, x + 8, y)


def symbol(kind: str, response: str | None):
    """(body path, glyph path, glyph text, insets) for a block type; body drawn filled."""
    body, glyph, text = QPainterPath(), QPainterPath(), None
    boxed = {"l": 10, "r": 10, "t": 4, "b": 44}
    if kind == "amplifier":
        body.moveTo(14, 6)
        body.lineTo(52, 24)
        body.lineTo(14, 42)
        body.closeSubpath()
        return body, glyph, None, {"l": 14, "r": 12, "t": 0, "b": 42}
    if kind == "mixer":
        body.addEllipse(QPointF(32, 24), 17, 17)
        glyph.moveTo(20, 12)
        glyph.lineTo(44, 36)
        glyph.moveTo(44, 12)
        glyph.lineTo(20, 36)
        return body, glyph, None, {"l": 15, "r": 15, "t": 7, "b": 41}
    if kind == "source":
        body.addEllipse(QPointF(26, 24), 16, 16)
        glyph.moveTo(16, 24)
        glyph.quadTo(21, 14, 26, 24)
        glyph.quadTo(31, 34, 36, 24)
        return body, glyph, None, {"l": 10, "r": 22, "t": 8, "b": 39}
    if kind == "load":
        body.addRoundedRect(QRectF(12, 8, 40, 32), 3, 3)
        return body, glyph, "50 Ω", {"l": 12, "r": 12, "t": 8, "b": 40}
    body.addRoundedRect(QRectF(10, 4, 44, 40), 3, 3)
    if kind in ("filter", "ideal_filter"):
        r = "bandpass" if kind == "ideal_filter" else (response or "bandpass")
        for y in (14, 24, 34):
            _tilde(glyph, y)
        for y in {"lowpass": [14], "highpass": [34], "bandpass": [14, 34], "bandstop": [24]}.get(r, []):
            glyph.moveTo(28, y + 5)
            glyph.lineTo(36, y - 5)
    elif kind == "attenuator":
        glyph.moveTo(14, 24)
        for x, y in ((18, 24), (21, 17), (26, 31), (31, 17), (36, 31), (41, 17), (44, 24), (48, 24)):
            glyph.lineTo(x, y)
    elif kind == "splitter":
        glyph.moveTo(10, 24)
        glyph.lineTo(26, 24)
        glyph.lineTo(54, 16)
        glyph.moveTo(26, 24)
        glyph.lineTo(54, 32)
    elif kind == "detector":   # a diode: what a power detector is
        glyph.moveTo(20, 15)
        glyph.lineTo(20, 33)
        glyph.lineTo(35, 24)
        glyph.closeSubpath()
        glyph.moveTo(35, 15)
        glyph.lineTo(35, 33)
        glyph.moveTo(10, 24)
        glyph.lineTo(20, 24)
        glyph.moveTo(35, 24)
        glyph.lineTo(44, 24)
    elif kind == "xyplot":     # axes and a trace
        glyph.moveTo(16, 10)
        glyph.lineTo(16, 38)
        glyph.lineTo(48, 38)
        glyph.moveTo(18, 34)
        glyph.cubicTo(24, 34, 26, 14, 30, 14)
        glyph.cubicTo(34, 14, 36, 30, 46, 22)
    elif kind == "sparam":
        text = "S21"
    else:
        text = kind
    return body, glyph, text, boxed


def _param_of(groups, name, path):
    g = next((x for x in groups if x["block"] == name), None)
    p = g and next((q for q in g["params"] if q["path"] == path), None)
    return p["value"] if p and (not p["optional"] or p["enabled"]) and visible(g, p) else None


def details(b: dict, groups: list, readings: dict | None = None) -> list[str]:
    """A block's critical values, a few short lines for under its name; they follow the sliders.
    `readings` holds the detectors' latest values (dBm, None when nothing is in their band)."""
    v = lambda path: _param_of(groups, b["name"], path)  # noqa: E731
    n = lambda x: f"{float(x):.1f}".rstrip("0").rstrip(".")  # noqa: E731
    f = lambda x: fmt_f(x, 4)  # noqa: E731

    def rng(lo, hi):
        for s, u in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
            if abs(hi) >= s:
                return f"{lo / s:.4g}–{hi / s:.4g} {u}" if abs(lo) >= s else f"{f(lo)} – {f(hi)}"
        return f"{f(lo)} – {f(hi)}"

    def opt(path, label, unit="dBm"):
        x = v(path)
        return None if x is None else f"{label} {n(x)} {unit}".strip()

    def join(*parts):
        return " · ".join(p for p in parts if p)
    t = b["type"]
    if t == "amplifier":
        return [x for x in (opt("gain_db", "", "dB"),
                            join(opt("oip3_dbm", "OIP3") or opt("iip3_dbm", "IIP3"), opt("op1db_dbm", "P1")),
                            join(opt("oip2_dbm", "OIP2") or opt("iip2_dbm", "IIP2"))) if x]
    if t == "attenuator":
        return [opt("loss_db", "", "dB")] if v("loss_db") is not None else []
    if t == "mixer":
        return [x for x in (join(opt("conversion_loss_db", "CL", "dB"), opt("lo_drive_dbm", "LO")),
                            join(opt("iip3_dbm", "IIP3"), opt("iip2_dbm", "IIP2")),
                            opt("max_order", "order ≤", "")) if x]
    if t == "ideal_filter":
        return [x for x in (rng(v("fmin"), v("fmax")) if v("fmin") is not None and v("fmax") is not None else None,
                            opt("insertion_loss_db", "IL", "dB")) if x]
    if t == "filter":
        r = v("response")
        if r in ("lowpass", "highpass"):
            band = ("LP " if r == "lowpass" else "HP ") + (f(v("fc")) if v("fc") else "")
        else:
            tag = "BS " if r == "bandstop" else "BP "
            band = tag + (rng(v("f1"), v("f2")) if v("f1") is not None and v("f2") is not None
                          else f"{f(v('f0'))} / {f(v('bw'))}" if v("f0") else "")
        order = join(f"{v('family')} {v('order')}" if v("family") else None, opt("insertion_loss_db", "IL", "dB"))
        return [x for x in (band, order) if x]
    if t == "detector":
        lo, hi = v("fmin"), v("fmax")
        band = rng(lo, hi) if lo is not None and hi is not None else "all" if lo is None and hi is None else \
            f"≥ {f(lo)}" if lo is not None else f"≤ {f(hi)}"
        r = (readings or {}).get(b["name"], "")
        return [f"{v('detect')} · {band}", "" if r == "" else "— dBm" if r is None else f"{r:.1f} dBm"]
    if t == "splitter":
        return [join(opt("excess_loss_db", "+", "dB"), opt("isolation_db", "iso", "dB"))]
    if t == "source":
        g = next((x for x in groups if x["block"] == b["name"]), None)
        tones = [p for p in (g["params"] if g else []) if p["path"].endswith(".freq") and p["path"].count(".") == 2]
        lines = []
        for p in tones:
            i = p["path"].split(".")[1]
            tag = p["label"].removesuffix(" freq")
            mode, pw = v(f"tones.{i}.mode"), v(f"tones.{i}.power")
            where = (f"{rng(v(f'tones.{i}.band.0'), v(f'tones.{i}.band.1'))} {mode}"
                     + (" log" if v(f"tones.{i}.spacing") == "log" else "")
                     if mode and mode != "cw" else f(p["value"]))
            lines.append(f"{tag} muted" if v(f"tones.{i}.mute") else join(f"{tag} {where}", f"{n(pw)} dBm" if pw is not None else None))
        if len(lines) > 3:
            lines = lines[:2] + [f"+{len(lines) - 2} more tones"]
        return lines
    return []


class PortItem(QGraphicsEllipseItem):
    """A port: a circle for RF, a diamond for a control port (X/Y plot wiring)."""

    def __init__(self, view: "SchematicView", ref: str, pos, parent, control: bool = False):
        super().__init__(-4, -4, 8, 8, parent)
        self.view, self.ref, self.dir, self.control = view, ref, pos[2:], control
        self.setPos(pos[0], pos[1])
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip(ref)
        self.setZValue(2)
        self.hot = False
        self.style()

    def style(self):
        probe = self.ref == self.view.probe
        wired = self.ref in self.view.wired
        col = theme.c("series-2") if probe else theme.c("accent") if self.hot else theme.c("text-secondary")
        self.setPen(QPen(col, 1.5))
        self.setBrush(QBrush(col if (probe or wired or self.hot) else theme.c("surface-2")))

    def paint(self, qp, opt, widget=None):
        if not self.control:
            return super().paint(qp, opt, widget)
        qp.setPen(self.pen())
        qp.setBrush(self.brush())
        d = QPainterPath()
        d.moveTo(0, -5)
        d.lineTo(5, 0)
        d.lineTo(0, 5)
        d.lineTo(-5, 0)
        d.closeSubpath()
        qp.drawPath(d)

    def hoverEnterEvent(self, e):
        self.hot = True
        self.style()

    def hoverLeaveEvent(self, e):
        self.hot = False
        self.style()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.view.begin_wire(self, e.scenePos())
            e.accept()

    def mouseMoveEvent(self, e):
        self.view.move_wire(e.scenePos())

    def mouseReleaseEvent(self, e):
        self.view.end_wire(e.scenePos())


class BlockItem(QGraphicsItem):
    def __init__(self, view: "SchematicView", b: dict, xy):
        super().__init__()
        self.view, self.b, self.name = view, b, b["name"]
        self.setPos(*xy)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
                      | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setZValue(1)
        self.ports = {}
        self.layout = port_layout(b["ports"])
        controls = set(b.get("controls", []))
        for p, pos in self.layout.items():
            self.ports[p] = PortItem(view, f"{self.name}.{p}", pos, self, p in controls)
        self.start = None
        self.font_b, self.font_s = QFont(), QFont()
        self.font_b.setPixelSize(12)
        self.font_b.setBold(True)
        self.font_s.setPixelSize(11)

    def boundingRect(self):
        return QRectF(-LABEL_W / 2 + BW / 2, -14, LABEL_W, BH + 18 + 13 * 3 + 20)

    def shape(self):
        p = QPainterPath()
        p.addRect(QRectF(-2, -2, BW + 4, BH + 4))
        return p

    def paint(self, qp: QPainter, opt, widget=None):
        resp = _param_of(self.view.groups, self.name, "response")
        body, glyph, text, inset = symbol(self.b["type"], resp)
        sel = self.isSelected()
        line = theme.c("accent") if sel else theme.c("text-secondary")
        lead = QPainterPath()
        for x, y, dx, dy in self.layout.values():
            lead.moveTo(x, y)
            if dx < 0:
                lead.lineTo(inset["l"], y)
            elif dx > 0:
                lead.lineTo(BW - inset["r"], y)
            elif dy < 0:
                lead.lineTo(x, inset["t"])
            else:
                lead.lineTo(x, inset["b"])
        qp.setPen(QPen(theme.c("text-secondary"), 1.5))
        qp.drawPath(lead)
        qp.setPen(QPen(line, 2.5 if sel else 1.5))
        qp.setBrush(theme.c("surface-2"))
        qp.drawPath(body)
        qp.setBrush(Qt.BrushStyle.NoBrush)
        qp.setPen(QPen(theme.c("text-primary"), 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                       Qt.PenJoinStyle.RoundJoin))
        qp.drawPath(glyph)
        if text:
            qp.setFont(self.font_s)
            qp.drawText(QRectF(0, 0, BW, BH), Qt.AlignmentFlag.AlignCenter, text)
        qp.setFont(self.font_b)
        qp.setPen(theme.c("text-primary"))
        x0 = BW / 2 - LABEL_W / 2
        qp.drawText(QRectF(x0, BH + 2, LABEL_W, 16), Qt.AlignmentFlag.AlignHCenter, self.name)
        qp.setFont(self.font_s)
        qp.setPen(theme.c("text-muted"))
        fm = qp.fontMetrics()
        for k, line in enumerate(details(self.b, self.view.groups, self.view.readings)[:3]):
            text = fm.elidedText(line, Qt.TextElideMode.ElideRight, LABEL_W - 6)
            qp.drawText(QRectF(x0, BH + 16 + 13 * k, LABEL_W, 15), Qt.AlignmentFlag.AlignHCenter, text)
        if len(self.b["ports"]) > 2:   # name the ports of mixers and splitters
            qp.setFont(self.font_s)
            for p, (x, y, dx, dy) in self.layout.items():
                if dx < 0:
                    qp.drawText(QPointF(3, y - 5), p)
                elif dx > 0:
                    qp.drawText(QRectF(BW - 40, y - 17, 37, 14), Qt.AlignmentFlag.AlignRight, p)
                else:
                    qp.drawText(QPointF(x + 5, y - 2), p)
        for p, (x, y, dx, dy) in self.layout.items():
            if f"{self.name}.{p}" == self.view.probe:
                qp.setPen(theme.c("series-2"))
                tag = "probe ←" if self.view.incident else "probe →"
                if dy < 0:
                    qp.drawText(QRectF(x - 60, y - 16, 54, 14), Qt.AlignmentFlag.AlignRight, tag)
                elif dx < 0:
                    qp.drawText(QRectF(x - 60, y + 4, 54, 14), Qt.AlignmentFlag.AlignRight, tag)
                else:
                    qp.drawText(QPointF(x + 7, y + 14), tag)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            return QPointF(_snap(value.x()), _snap(value.y()))
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.view.pos[self.name] = (self.pos().x(), self.pos().y())
            self.view.reroute()
        return super().itemChange(change, value)

    def mousePressEvent(self, e):
        self.start = (self.pos().x(), self.pos().y())
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        if self.start and self.start != (self.pos().x(), self.pos().y()):
            self.view.moved()
        else:
            self.view.blockClicked.emit(self.name)
        self.start = None

    def mouseDoubleClickEvent(self, e):
        to, ok = QInputDialog.getText(self.view, "Rename block", f'Rename block "{self.name}" to:', text=self.name)
        if ok and to.strip() and to.strip() != self.name:
            self.view.op.emit({"op": "rename", "block": self.name, "to": to.strip()})


class WireItem(QGraphicsPathItem):
    """A wire; dashed when it joins control ports (X/Y plot wiring, no RF)."""

    def __init__(self, a: str, b: str, control: bool = False):
        super().__init__()
        self.a, self.b, self.control = a, b, control
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setZValue(0)

    def shape(self):
        s = QPainterPathStroker()
        s.setWidth(10)
        return s.createStroke(self.path())

    def paint(self, qp, opt, widget=None):
        self.setPen(QPen(theme.c("accent") if self.isSelected() else theme.c("text-secondary"),
                         2.5 if self.isSelected() else 1.5,
                         Qt.PenStyle.DashLine if self.control else Qt.PenStyle.SolidLine))
        qp.setPen(self.pen())
        qp.setBrush(Qt.BrushStyle.NoBrush)
        qp.drawPath(self.path())


class SchematicView(QGraphicsView):
    op = Signal(dict)          # a ViewerModel.edit() operation to apply
    probeClicked = Signal(str)
    blockClicked = Signal(str)
    undoRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumHeight(140)
        self.pos: dict[str, tuple] = {}
        self.blocks: dict[str, BlockItem] = {}
        self.wires: list[WireItem] = []
        self.wired: set[str] = set()
        self.groups: list = []
        self.readings: dict = {}   # detectors' latest values, shown under them
        self.sc: dict = {}
        self.probe = None
        self.incident = False
        self.temp = None
        self.wire_from = None
        self.scene_.selectionChanged.connect(self._selection)

    def _selection(self):
        """A block selected (click, or rubber band around one) is the one whose parameters show."""
        try:
            sel = [it for it in self.scene_.selectedItems() if isinstance(it, BlockItem)]
        except RuntimeError:   # the scene is being torn down
            return
        if len(sel) == 1:
            self.blockClicked.emit(sel[0].name)

    def select_block(self, name: str | None):
        """Make `name` the only selected item."""
        it = self.blocks.get(name)
        if it is None or self.scene_.selectedItems() == [it]:
            return
        self.scene_.clearSelection()
        it.setSelected(True)
        self.ensureVisible(it, 40, 40)

    # ------------------------------------------------------------ building
    def load(self, sc: dict, groups: list, probe: str, incident: bool, fit: bool = False):
        self.sc, self.groups, self.probe, self.incident = sc, groups, probe, incident
        lay = sc.get("layout") or {}
        missing = [b for b in sc["blocks"] if b["name"] not in lay]
        auto = auto_layout(sc["blocks"], sc["wires"]) if missing else {}
        self.pos = {b["name"]: (float(lay[b["name"]]["x"]), float(lay[b["name"]]["y"])) if b["name"] in lay
                    else auto[b["name"]] for b in sc["blocks"]}
        self.rebuild()
        if fit:
            self.fit()

    def rebuild(self):
        sel = {it.name for it in self.blocks.values() if it.isSelected()}
        blocker = QSignalBlocker(self.scene_)   # rebuilding is not the user changing the selection
        self.scene_.clear()
        self.blocks, self.wires = {}, []
        self.wired = {r for w in self.sc.get("wires", []) for r in w}
        self.scene_.setBackgroundBrush(theme.c("surface-1"))
        for b in self.sc.get("blocks", []):
            it = BlockItem(self, b, self.pos[b["name"]])
            self.scene_.addItem(it)
            self.blocks[b["name"]] = it
            it.setSelected(b["name"] in sel)
        ctl = {f"{b['name']}.{p}" for b in self.sc.get("blocks", []) for p in b.get("controls", [])}
        for a, b in self.sc.get("wires", []):
            w = WireItem(a, b, a in ctl)
            self.scene_.addItem(w)
            self.wires.append(w)
        blocker.unblock()
        self.reroute()

    def refresh(self, groups=None, probe=None, incident=None, readings=None):
        """Summaries follow the sliders; the probe marker follows the port."""
        if groups is not None:
            self.groups = groups
        if readings is not None:
            self.readings = readings
        if probe is not None:
            self.probe = probe
        if incident is not None:
            self.incident = incident
        for it in self.blocks.values():
            for p in it.ports.values():
                p.style()
            it.update()

    def port_at(self, ref):
        name, port = ref.split(".", 1)
        it = self.blocks.get(name)
        if it is None or port not in it.layout:
            return None
        x, y, dx, dy = it.layout[port]
        return (it.pos().x() + x, it.pos().y() + y, dx, dy)

    def _clear_between(self, a, b) -> bool:
        """Side by side on one row with no block in between: a control wire can go straight."""
        if a[2] <= 0 or b[2] >= 0 or abs(a[1] - b[1]) > BH or b[0] <= a[0]:
            return False
        lo, hi = a[0], b[0]
        return not any(lo < x + BW / 2 < hi and abs(y + BH / 2 - a[1]) < BH for x, y in self.pos.values())

    def reroute(self):
        bus_y = max((y for _, y in self.pos.values()), default=0) + BH + 64   # below the details lines
        for w in self.wires:
            pa, pb = self.port_at(w.a), self.port_at(w.b)
            if pa and pb:
                if w.control and not self._clear_between(pa, pb):
                    # run in the gaps between columns: right of the start, left of the end
                    ax = self.pos[w.a.split(".")[0]][0] + BW / 2 + COL_PITCH / 2 if pa[2] >= 0 else pa[0] - 12
                    bx = pb[0] - 12 if pb[2] <= 0 else pb[0] + 12
                    pts = route_bus(pa, pb, bus_y, ax, bx)
                else:
                    pts = route(pa, pb)
                path = QPainterPath(QPointF(*pts[0]))
                for pt in pts[1:]:
                    path.lineTo(QPointF(*pt))
                w.setPath(path)

    def fit(self):
        """Fit the circuit to the panel, but not so small that the block details can't be read
        (pan or wheel-zoom out for the rest)."""
        r = self.scene_.itemsBoundingRect()
        if r.isEmpty():
            return
        self.fitInView(r.adjusted(-20, -20, 20, 20), Qt.AspectRatioMode.KeepAspectRatio)
        k = min(max(self.transform().m11(), FIT_MIN), 1.4)
        if k != self.transform().m11():
            self.resetTransform()
            self.scale(k, k)
            self.centerOn(r.center())

    def layout_now(self) -> dict:
        return {n: {"x": round(x), "y": round(y)} for n, (x, y) in self.pos.items()}

    # ------------------------------------------------------------ edits
    def moved(self):
        self.op.emit({"op": "move"})

    def begin_wire(self, port: PortItem, at: QPointF):
        self.wire_from = port
        self.temp = QGraphicsPathItem()
        pen = QPen(theme.c("accent"), 1.5, Qt.PenStyle.DashLine)
        self.temp.setPen(pen)
        self.temp.setZValue(3)
        self.scene_.addItem(self.temp)
        self.moved_far = False
        self.press_at = at

    def move_wire(self, at: QPointF):
        if self.wire_from is None:
            return
        if (at - self.press_at).manhattanLength() > 4:
            self.moved_far = True
        a = self.wire_from.scenePos()
        path = QPainterPath(a)
        path.lineTo(at)
        self.temp.setPath(path)

    def end_wire(self, at: QPointF):
        src, self.wire_from = self.wire_from, None
        if self.temp is not None:
            self.scene_.removeItem(self.temp)
            self.temp = None
        if src is None:
            return
        if not self.moved_far:
            self.probeClicked.emit(src.ref)
            return
        tgt = next((it for it in self.scene_.items(at) if isinstance(it, PortItem) and it is not src), None)
        if tgt is not None:
            self.op.emit({"op": "connect", "a": src.ref, "b": tgt.ref})

    def delete_selected(self):
        for it in self.scene_.selectedItems():
            if isinstance(it, BlockItem):
                self.op.emit({"op": "delete", "block": it.name})
                return
            if isinstance(it, WireItem):
                self.op.emit({"op": "disconnect", "a": it.a})
                return

    def add_block_at_centre(self, kind: str):
        c = self.mapToScene(self.viewport().rect().center())
        x, y = _snap(c.x() - BW / 2), _snap(c.y() - BH / 2)
        while any(abs(px - x) < 60 and abs(py - y) < 60 for px, py in self.pos.values()):
            x, y = x + 2 * GRID, y + 2 * GRID
        self.op.emit({"op": "add", "type": kind, "x": x, "y": y})

    # ------------------------------------------------------------ view
    def wheelEvent(self, e):
        k = 1.1 if e.angleDelta().y() > 0 else 1 / 1.1
        s = self.transform().m11() * k
        if 0.25 <= s <= 3:
            self.scale(k, k)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
        elif e.matches(QKeySequence.StandardKey.Undo):
            self.undoRequested.emit()
        elif e.key() == Qt.Key.Key_F:
            self.fit()
        elif e.key() == Qt.Key.Key_Escape:
            self.scene_.clearSelection()
        else:
            super().keyPressEvent(e)

    def mouseDoubleClickEvent(self, e):
        if self.itemAt(e.position().toPoint()) is None:
            self.fit()
        else:
            super().mouseDoubleClickEvent(e)

    def drawBackground(self, qp: QPainter, rect: QRectF):
        qp.fillRect(rect, theme.c("surface-1"))
        qp.setPen(QPen(theme.c("border"), 1.5 / max(self.transform().m11(), 0.1)))
        x0, y0 = math.floor(rect.left() / GRID) * GRID, math.floor(rect.top() / GRID) * GRID
        y = y0
        while y < rect.bottom():
            x = x0
            while x < rect.right():
                qp.drawPoint(QPointF(x, y))
                x += GRID
            y += GRID
