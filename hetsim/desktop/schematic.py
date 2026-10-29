"""Schematic editor: the circuit drawn with RF symbols on a QGraphicsScene.

Same operations as the browser viewer: drag a block to move it, drag from port to port to
wire, click a port to probe it, click a block to open its parameters, Del deletes the
selection, double-click renames, wheel zooms, F fits. Every edit is a ViewerModel.edit()
operation, emitted as `op` for the window to apply.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QFont, QKeySequence, QPainter, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem, QGraphicsScene,
                               QGraphicsView, QInputDialog)

from . import theme
from .params import visible
from .plot import fmt_f
from .tracks import BH, BW, auto_layout, port_layout, route

GRID = 20


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
    boxed = {"l": 10, "r": 10, "t": 4}
    if kind == "amplifier":
        body.moveTo(14, 6)
        body.lineTo(52, 24)
        body.lineTo(14, 42)
        body.closeSubpath()
        return body, glyph, None, {"l": 14, "r": 12, "t": 0}
    if kind == "mixer":
        body.addEllipse(QPointF(32, 24), 17, 17)
        glyph.moveTo(20, 12)
        glyph.lineTo(44, 36)
        glyph.moveTo(44, 12)
        glyph.lineTo(20, 36)
        return body, glyph, None, {"l": 15, "r": 15, "t": 7}
    if kind == "source":
        body.addEllipse(QPointF(26, 24), 16, 16)
        glyph.moveTo(16, 24)
        glyph.quadTo(21, 14, 26, 24)
        glyph.quadTo(31, 34, 36, 24)
        return body, glyph, None, {"l": 10, "r": 22, "t": 8}
    if kind == "load":
        body.addRoundedRect(QRectF(12, 8, 40, 32), 3, 3)
        return body, glyph, "50 Ω", {"l": 12, "r": 12, "t": 8}
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
    elif kind == "sparam":
        text = "S21"
    else:
        text = kind
    return body, glyph, text, boxed


def _param_of(groups, name, path):
    g = next((x for x in groups if x["block"] == name), None)
    p = g and next((q for q in g["params"] if q["path"] == path), None)
    return p["value"] if p and (not p["optional"] or p["enabled"]) and visible(g, p) else None


def summary(b: dict, groups: list) -> str:
    """Key values under a block's name; follows the sliders."""
    v = lambda path: _param_of(groups, b["name"], path)  # noqa: E731
    n = lambda x: f"{float(x):.1f}".rstrip("0").rstrip(".")  # noqa: E731
    f = lambda x: fmt_f(x, 4)  # noqa: E731

    def rng(lo, hi):
        for s, u in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
            if abs(hi) >= s:
                return f"{lo / s:.4g}–{hi / s:.4g} {u}" if abs(lo) >= s else f"{f(lo)} – {f(hi)}"
        return f"{f(lo)} – {f(hi)}"
    t = b["type"]
    if t == "amplifier":
        return "" if v("gain_db") is None else f"{n(v('gain_db'))} dB"
    if t == "attenuator":
        return "" if v("loss_db") is None else f"{n(v('loss_db'))} dB"
    if t == "mixer":
        return "" if v("conversion_loss_db") is None else f"CL {n(v('conversion_loss_db'))} dB"
    if t == "ideal_filter":
        return rng(v("fmin"), v("fmax")) if v("fmin") is not None and v("fmax") is not None else ""
    if t == "filter":
        r = v("response")
        if r in ("lowpass", "highpass"):
            return ("LP " if r == "lowpass" else "HP ") + (f(v("fc")) if v("fc") else "")
        tag = "BS " if r == "bandstop" else "BP "
        if v("f1") is not None and v("f2") is not None:
            return tag + rng(v("f1"), v("f2"))
        return tag + (f"{f(v('f0'))} / {f(v('bw'))}" if v("f0") else "")
    if t == "source":
        g = next((x for x in groups if x["block"] == b["name"]), None)
        tones = [p for p in (g["params"] if g else []) if p["path"].endswith(".freq") and p["path"].count(".") == 2]
        if not tones:
            return ""
        tag = tones[0]["label"].removesuffix(" freq")
        if len(tones) == 1 and v("tones.0.mute"):
            return f"{tag} muted"
        mode = v("tones.0.mode")
        first = (f"{tag} {rng(v('tones.0.band.0'), v('tones.0.band.1'))} {mode}"
                 + (" log" if v("tones.0.spacing") == "log" else "")
                 if mode and mode != "cw" else f"{tag} {f(tones[0]['value'])}")
        return f"{first} +{len(tones) - 1}" if len(tones) > 1 else first
    return ""


class PortItem(QGraphicsEllipseItem):
    def __init__(self, view: "SchematicView", ref: str, pos, parent):
        super().__init__(-4, -4, 8, 8, parent)
        self.view, self.ref, self.dir = view, ref, pos[2:]
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
        for p, pos in self.layout.items():
            self.ports[p] = PortItem(view, f"{self.name}.{p}", pos, self)
        self.start = None
        self.font_b, self.font_s = QFont(), QFont()
        self.font_b.setPixelSize(12)
        self.font_b.setBold(True)
        self.font_s.setPixelSize(11)

    def boundingRect(self):
        return QRectF(-60, -14, BW + 120, BH + 50)

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
            else:
                lead.lineTo(x, inset["t"])
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
        qp.drawText(QRectF(-60, BH + 2, BW + 120, 16), Qt.AlignmentFlag.AlignHCenter, self.name)
        qp.setFont(self.font_s)
        qp.setPen(theme.c("text-muted"))
        qp.drawText(QRectF(-60, BH + 16, BW + 120, 16), Qt.AlignmentFlag.AlignHCenter, summary(self.b, self.view.groups))
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
    def __init__(self, a: str, b: str):
        super().__init__()
        self.a, self.b = a, b
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setZValue(0)

    def shape(self):
        s = QPainterPathStroker()
        s.setWidth(10)
        return s.createStroke(self.path())

    def paint(self, qp, opt, widget=None):
        self.setPen(QPen(theme.c("accent") if self.isSelected() else theme.c("text-secondary"),
                         2.5 if self.isSelected() else 1.5))
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
        self.sc: dict = {}
        self.probe = None
        self.incident = False
        self.temp = None
        self.wire_from = None

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
        self.scene_.clear()
        self.blocks, self.wires = {}, []
        self.wired = {r for w in self.sc.get("wires", []) for r in w}
        self.scene_.setBackgroundBrush(theme.c("surface-1"))
        for b in self.sc.get("blocks", []):
            it = BlockItem(self, b, self.pos[b["name"]])
            self.scene_.addItem(it)
            self.blocks[b["name"]] = it
            it.setSelected(b["name"] in sel)
        for a, b in self.sc.get("wires", []):
            w = WireItem(a, b)
            self.scene_.addItem(w)
            self.wires.append(w)
        self.reroute()

    def refresh(self, groups=None, probe=None, incident=None):
        """Summaries follow the sliders; the probe marker follows the port."""
        if groups is not None:
            self.groups = groups
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

    def reroute(self):
        for w in self.wires:
            pa, pb = self.port_at(w.a), self.port_at(w.b)
            if pa and pb:
                pts = route(pa, pb)
                path = QPainterPath(QPointF(*pts[0]))
                for pt in pts[1:]:
                    path.lineTo(QPointF(*pt))
                w.setPath(path)

    def fit(self):
        r = self.scene_.itemsBoundingRect()
        if r.isEmpty():
            return
        self.fitInView(r.adjusted(-20, -20, 20, 20), Qt.AspectRatioMode.KeepAspectRatio)
        if self.transform().m11() > 1.4:
            self.resetTransform()
            self.scale(1.4, 1.4)
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
