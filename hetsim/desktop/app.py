"""hetsim desktop viewer: python -m hetsim app netlist.yaml

A native window around ViewerModel (the same model the browser viewer serves): parameters
on the left, schematic, spectrum and waveform on the right, simulations on a worker thread.
No server and no browser.
"""
from __future__ import annotations

import math
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
                               QLineEdit, QMainWindow, QMessageBox, QPushButton, QSplitter, QStackedWidget,
                               QToolBar, QVBoxLayout, QWidget)

from ..viewer.model import FileChangedError, NetlistTextError, ViewerModel, dump_text
from . import theme
from .editor import NetlistEditor
from .params import ParamPanel, carry_values, parse_f
from .plot import SpectrumPlot, ViewState, WaveformPlot, fmt_f
from .schematic import SchematicView
from .sim import SimRunner

RBWS = ["off", "1 kHz", "3 kHz", "10 kHz", "30 kHz", "100 kHz", "300 kHz", "1 MHz", "3 MHz"]
INTERP = [("0", "no interpolation"), ("lin4", "linear ×4"), ("lin8", "linear ×8"),
          ("pchip4", "smooth ×4"), ("pchip8", "smooth ×8"), ("pchip16", "smooth ×16")]


class MainWindow(QMainWindow):
    def __init__(self, model: ViewerModel, sync: bool = False):
        super().__init__()
        self.model = model
        self.sync = sync   # tests: simulate on the calling thread
        self.vs = ViewState()
        self.info = None
        self.probe = None
        self.sweep_pts = None
        self.pn_on = True
        self.undo: list[str] = []
        self.sliders_dirty = False
        self.runner = SimRunner(model, self)
        self.runner.done.connect(self.on_result)
        self.runner.busy.connect(lambda b: self.busy.setText("● simulating" if b else ""))
        self.resize(1400, 900)
        self._build()
        self.load_model(model.info(), first=True)

    # ------------------------------------------------------------ layout
    def _build(self):
        tb = QToolBar("view")
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addWidget(QLabel(" Port "))
        self.port = QComboBox()
        self.port.setMinimumWidth(110)
        self.port.currentTextChanged.connect(self._port_changed)
        tb.addWidget(self.port)
        tb.addSeparator()
        self.dir_group = self._segmented(tb, [("out", "leaving"), ("in", "arriving")], self._dir_changed)
        tb.addSeparator()
        self.solver_group = self._segmented(tb, [("td", "Time domain"), ("both", "TD + tone"), ("tone", "Tone")],
                                            self._solver_changed)
        tb.addSeparator()
        tb.addWidget(QLabel("Resolution "))
        self.res = QLineEdit()
        self.res.setFixedWidth(90)
        self.res.editingFinished.connect(self._res_changed)
        tb.addWidget(self.res)
        tb.addWidget(QLabel("  RBW "))
        self.rbw = QComboBox()
        self.rbw.addItems(RBWS)
        self.rbw.setToolTip("Draw the spectrum as a swept analyser would: each line through a Gaussian RBW filter, "
                            "peak detector, phase noise included")
        self.rbw.currentTextChanged.connect(self._rbw_changed)
        tb.addWidget(self.rbw)
        self.pn = QCheckBox("Phase noise")
        self.pn.setChecked(True)
        self.pn.setToolTip("Include the sources' phase noise (seen through the RBW trace)")
        self.pn.toggled.connect(self._pn_changed)
        self.pn_act = tb.addWidget(self.pn)
        tb.addWidget(QLabel("  Floor "))
        self.floor = QDoubleSpinBox()
        self.floor.setRange(-400, 100)
        self.floor.setSingleStep(10)
        self.floor.setValue(-160)
        self.floor.setSuffix(" dBm")
        self.floor.valueChanged.connect(self._floor_changed)
        tb.addWidget(self.floor)
        tb.addSeparator()
        self.sch_btn = QPushButton("Schematic")
        self.sch_btn.setCheckable(True)
        self.sch_btn.setChecked(True)
        self.sch_btn.toggled.connect(lambda on: self.sch_panel.setVisible(on))
        tb.addWidget(self.sch_btn)
        self.ed_btn = QPushButton("Netlist")
        self.ed_btn.setCheckable(True)
        self.ed_btn.setToolTip("Edit the netlist source")
        self.ed_btn.toggled.connect(lambda on: self.left.setCurrentIndex(1 if on else 0))
        tb.addWidget(self.ed_btn)
        exp = QPushButton("Export netlist")
        exp.setToolTip("Save the netlist with the current settings")
        exp.clicked.connect(self.export)
        tb.addWidget(exp)

        self.addToolBarBreak()
        self.swbar = QToolBar("band sweep")
        self.swbar.setMovable(False)
        self.addToolBar(self.swbar)
        self.sw_show = QCheckBox("Band sweep")
        self.sw_show.setChecked(True)
        self.sw_show.toggled.connect(self._sw_show)
        self.swbar.addWidget(self.sw_show)
        self.swbar.addWidget(QLabel("  Steps "))
        self.sw_pts = QLineEdit()
        self.sw_pts.setPlaceholderText("auto")
        self.sw_pts.setFixedWidth(60)
        self.sw_pts.setToolTip("Sweep steps to compute. Empty: the tones' points, reduced to keep updates quick")
        self.sw_pts.editingFinished.connect(self._sw_pts)
        self.swbar.addWidget(self.sw_pts)
        self.sw_interp = QComboBox()
        for v, t in INTERP:
            self.sw_interp.addItem(t, v)
        self.sw_interp.setToolTip("Draw the tracks between computed steps. Frequencies interpolate exactly; levels are estimated")
        self.sw_interp.currentIndexChanged.connect(self._sw_interp)
        self.swbar.addWidget(self.sw_interp)
        self.sw_place = QComboBox()
        self.sw_place.addItem("steps across band", "band")
        self.sw_place.addItem("steps on output grid", "out")
        self.sw_place.setToolTip("Where the sweep steps go: evenly across the swept band, or so the anchor product "
                                 "(the selected track, else the signal's conversion) lands on fixed output frequencies")
        self.sw_place.currentIndexChanged.connect(self._sw_place)
        self.swbar.addWidget(self.sw_place)
        self.sw_from, self.sw_to = QLineEdit(), QLineEdit()
        for e in (self.sw_from, self.sw_to):
            e.setFixedWidth(90)
            e.editingFinished.connect(self._sw_range)
        self.sw_view = QPushButton("view")
        self.sw_view.setToolTip("Use the plot's current frequency range")
        self.sw_view.clicked.connect(lambda: self.set_sw_out(self._view_range()))
        self.sw_out_acts = [self.swbar.addWidget(self.sw_from), self.swbar.addWidget(QLabel("–")),
                            self.swbar.addWidget(self.sw_to), self.swbar.addWidget(self.sw_view)]
        self.sw_pick = QComboBox()
        self.sw_pick.setMinimumWidth(200)
        self.sw_pick.setToolTip("Show one product's track (or click a track on the plot; Esc shows all)")
        self.sw_pick.activated.connect(lambda i: self.set_sw_sel(self.sw_pick.itemData(i)))
        self.swbar.addWidget(self.sw_pick)
        self.swbar.setVisible(False)
        for a in self.sw_out_acts:
            a.setVisible(False)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.left = QStackedWidget()
        self.panel = ParamPanel()
        self.panel.changed.connect(self.simulate)
        self.panel.touched.connect(self._touched)
        self.panel.limitsRequested.connect(lambda b, p, r: self.sch_op({"op": "limits", "block": b, "path": p, "range": r},
                                                                         layout=False))
        self.panel.message.connect(lambda m: self.show_msgs([("err", m)]))
        self.editor = NetlistEditor()
        self.editor.applyRequested.connect(self.apply_text)
        self.pull = QPushButton("Pull slider values into text")
        self.pull.setToolTip("Rewrite the text from the current settings. Comments are dropped (Ctrl+Z undoes it).")
        self.pull.clicked.connect(self.pull_sliders)
        self.pull.hide()
        self.editor.layout().insertWidget(2, self.pull)
        self.left.addWidget(self.panel)
        self.left.addWidget(self.editor)
        split.addWidget(self.left)

        right = QSplitter(Qt.Orientation.Vertical)
        self.sch_panel = QWidget()
        sv = QVBoxLayout(self.sch_panel)
        sv.setContentsMargins(0, 0, 0, 0)
        sbar = QHBoxLayout()
        sbar.setContentsMargins(6, 4, 6, 0)
        self.add = QComboBox()
        self.add.addItem("+ Add block")
        self.add.activated.connect(self._add_block)
        self.dele = QPushButton("Delete")
        self.dele.setToolTip("Delete the selected block or wire (Del)")
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setToolTip("Undo the last netlist change (Ctrl+Z)")
        fit = QPushButton("Fit")
        fit.setToolTip("Fit the schematic to the panel (F)")
        hint = QLabel("drag port to port to wire · click a port to probe it · double-click a block to rename")
        hint.setStyleSheet("color: gray; font-size: 11px")
        for w in (self.add, self.dele, self.undo_btn, fit):
            sbar.addWidget(w)
        sbar.addStretch(1)
        sbar.addWidget(hint)
        sv.addLayout(sbar)
        self.sch = SchematicView()
        sv.addWidget(self.sch, 1)
        self.sch.op.connect(self.sch_op)
        self.sch.probeClicked.connect(self.set_probe)
        self.sch.blockClicked.connect(lambda n: None if self.ed_btn.isChecked() else self.panel.focus_block(n))
        self.sch.undoRequested.connect(self.do_undo)
        self.dele.clicked.connect(self.sch.delete_selected)
        self.undo_btn.clicked.connect(self.do_undo)
        fit.clicked.connect(self.sch.fit)
        right.addWidget(self.sch_panel)
        self.plot = SpectrumPlot(self.vs)
        self.plot.trackClicked.connect(lambda lab: self.set_sw_sel(None if lab == self.vs.sw_sel else lab))
        right.addWidget(self.plot)
        self.wave = WaveformPlot(self.vs)
        right.addWidget(self.wave)
        self.msgs = QLabel()
        self.msgs.setWordWrap(True)
        self.msgs.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.msgs.setContentsMargins(8, 2, 8, 2)
        right.addWidget(self.msgs)
        right.setSizes([230, 420, 150, 40])
        right.setStretchFactor(1, 1)
        split.addWidget(right)
        split.setSizes([360, 1040])
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)

        self.status = QLabel()
        self.busy = QLabel()
        self.busy.setStyleSheet(f"color: {theme.c('accent').name()}")
        self.statusBar().addWidget(self.busy)
        self.statusBar().addPermanentWidget(self.status)
        esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        esc.activated.connect(lambda: self.set_sw_sel(None) if self.vs.sw_sel else None)
        QShortcut(QKeySequence.StandardKey.Undo, self.sch, activated=self.do_undo)

    def _segmented(self, tb, items, fn):
        group = QActionGroup(self)
        group.setExclusive(True)
        for i, (val, text) in enumerate(items):
            act = QAction(text, self)
            act.setCheckable(True)
            act.setData(val)
            act.setChecked(i == 0)
            group.addAction(act)
            tb.addAction(act)
        group.triggered.connect(lambda a: fn(a.data()))
        return group

    # ------------------------------------------------------------ model
    def load_model(self, info: dict, first: bool = False, keep: bool = False, renames: dict | None = None):
        """First load, a netlist applied from the editor, or a schematic edit (keep: the sliders keep
        their values). Keep the port and a hand-set resolution if they still fit."""
        renames = renames or {}
        prev = self.info["groups"] if self.info else []
        if first or self.vs.resolution == self.info["resolution"]:
            self.vs.resolution = info["resolution"]
        if self.probe:
            b, p = self.probe.split(".", 1)
            if b in renames:
                self.probe = f"{renames[b]}.{p}"
        if first or self.probe not in info["ports"]:
            self.probe = info["probe"]
            self.vs.view = None
        self.info = info
        if keep:
            carry_values(prev, info["groups"], renames)
        else:
            self.sliders_dirty = False
        self.setWindowTitle((info["name"] + " · " if info["name"] else "") + "hetsim")
        with QSignalBlocker(self.port):
            self.port.clear()
            self.port.addItems(info["ports"])
            self.port.setCurrentText(self.probe)
        self.res.setText(fmt_f(self.vs.resolution))
        self.panel.build(info["groups"])
        self.editor.load(info["text"], info["file"], info["can_save"])
        self._refresh_pull()
        with QSignalBlocker(self.add):
            self.add.clear()
            self.add.addItem("+ Add block")
            for t in info["schematic"]["addable"]:
                self.add.addItem(t.replace("_", " "), t)
        self.sch.load(info["schematic"], info["groups"], self.probe, self.vs.incident, fit=first)
        if first:
            QTimer.singleShot(0, self.sch.fit)
        self.undo_btn.setEnabled(bool(self.undo))
        self.simulate()

    def request(self) -> dict:
        rbw = self.vs.rbw
        return {
            "params": self.panel.collect(), "probe": self.probe, "incident": self.vs.incident,
            "solver": self.vs.solver, "resolution": self.vs.resolution, "sweep_points": self.sweep_pts,
            "rbw": rbw, "phase_noise": self.pn_on,
            # with an RBW, lines below the floor can still add up inside it: ask for them too
            "min_dbm": self.vs.floor - (max(0.0, 10 * math.log10(rbw / self.vs.resolution)) + 10 if rbw else 0),
            "sweep_out": self.vs.sw_out and {"anchor": self.vs.sw_sel, "fmin": self.vs.sw_out[0],
                                             "fmax": self.vs.sw_out[1], "points": self.sweep_pts},
        }

    def simulate(self):
        if self.sync:
            self.on_result(self.runner.run_now(self.request()), None)
        else:
            self.runner.submit(self.request())

    def on_result(self, d: dict, req):
        if d.get("error"):
            self.show_msgs([("err", d["error"])])
            return
        self.vs.data = d
        self.vs.extra["pn_on"] = self.pn_on
        self.panel.apply_snapped(d.get("snapped"))
        self.swbar.setVisible(bool(d.get("sweep")))
        self.pn_act.setVisible(bool(d.get("has_pn")))
        msgs = [("warn", w) for w in (d.get("warnings") or "").split("\n") if w]
        if d.get("sweep") and d["sweep"].get("note"):
            msgs.append(("", d["sweep"]["note"]))
        res = d.get("resolution") or self.vs.resolution
        if self.vs.rbw and self.vs.solver != "tone" and self.vs.rbw < res:
            msgs.append(("", f"RBW {fmt_f(self.vs.rbw, 3)} is narrower than the FFT resolution ({fmt_f(res, 3)}): "
                             "noise exists only as separate bins, each holding a whole bin's power, so it shows as "
                             "peaks. Set Resolution to the RBW or finer, or use the tone solver, for a continuous "
                             "noise trace."))
        self.show_msgs(msgs)
        plan = d.get("plan")
        self.status.setText((plan.split(" (products")[0] + " · " if plan else "tone solver · ") + f"{d['elapsed_ms']} ms")
        self._fill_pick()
        self.plot.update()
        self.wave.update()

    def show_msgs(self, items):
        colors = {"err": theme.c("error").name(), "warn": theme.c("warn").name(), "": theme.c("text-secondary").name()}
        self.msgs.setText("<br>".join(f"<span style='color:{colors[k]}'>{_html(t)}</span>" for k, t in items))

    # ------------------------------------------------------------ header controls
    def _port_changed(self, ref):
        if ref and ref != self.probe:
            self.set_probe(ref)

    def set_probe(self, ref: str):
        if ref not in self.info["ports"]:
            return
        self.probe = ref
        with QSignalBlocker(self.port):
            self.port.setCurrentText(ref)
        self.vs.view = None
        self.sch.refresh(probe=ref)
        self.simulate()

    def _dir_changed(self, v):
        self.vs.incident = v == "in"
        self.sch.refresh(incident=self.vs.incident)
        self.simulate()

    def _solver_changed(self, v):
        self.vs.solver = v
        self.simulate()

    def _res_changed(self):
        v = parse_f(self.res.text())
        if math.isfinite(v) and v > 0 and v != self.vs.resolution:
            self.vs.resolution = v
            self.simulate()
        self.res.setText(fmt_f(self.vs.resolution))

    def _rbw_changed(self, text):
        self.vs.rbw = None if text == "off" else parse_f(text)
        self.simulate()

    def _pn_changed(self, on):
        self.pn_on = on
        self.simulate()

    def _floor_changed(self, v):
        self.vs.floor = v
        self.simulate()

    def _touched(self):
        self.sch.refresh(groups=self.info["groups"])   # block summaries follow the sliders
        if not self.sliders_dirty:
            self.sliders_dirty = True
            self._refresh_pull()

    # ------------------------------------------------------------ band sweep controls
    def _sw_show(self, on):
        self.vs.show_sweep = on
        self.vs.view = None
        self.plot.update()

    def _sw_pts(self):
        t = self.sw_pts.text().strip()
        try:
            v = round(float(t))
        except ValueError:
            v = None
        self.sweep_pts = min(v, 2001) if v and v >= 2 else None
        self.sw_pts.setText(str(self.sweep_pts or ""))
        self.simulate()

    def _sw_interp(self, i):
        self.vs.interp = self.sw_interp.itemData(i)
        self.vs.extra["interp_text"] = self.sw_interp.currentText()
        self.plot.update()

    def _view_range(self):
        g = self.plot.geom
        return (max(g["x0"], (g["x1"] - g["x0"]) / 20), g["x1"]) if g else None

    def _sw_place(self, i):
        self.set_sw_out(self._view_range() if self.sw_place.itemData(i) == "out" else None)

    def set_sw_out(self, rng):
        self.vs.sw_out = rng
        with QSignalBlocker(self.sw_place):
            self.sw_place.setCurrentIndex(1 if rng else 0)
        for a in self.sw_out_acts:
            a.setVisible(bool(rng))
        if rng:
            self.sw_from.setText(fmt_f(rng[0], 6))
            self.sw_to.setText(fmt_f(rng[1], 6))
        self.simulate()

    def _sw_range(self):
        if not self.vs.sw_out:
            return
        a, b = parse_f(self.sw_from.text()), parse_f(self.sw_to.text())
        if math.isfinite(a) and math.isfinite(b) and 0 < a < b:
            if (a, b) != tuple(self.vs.sw_out):
                self.set_sw_out((a, b))
        else:
            self.sw_from.setText(fmt_f(self.vs.sw_out[0], 6))
            self.sw_to.setText(fmt_f(self.vs.sw_out[1], 6))

    def set_sw_sel(self, label):
        self.vs.sw_sel = label or None
        self._fill_pick()
        if self.vs.sw_out:
            self.simulate()   # the selected track anchors the output grid
        self.plot.update()

    def _fill_pick(self):
        sw = self.vs.data and self.vs.data.get("sweep")
        with QSignalBlocker(self.sw_pick):
            self.sw_pick.clear()
            self.sw_pick.addItem("all tracks", None)
            if sw:
                rows = sorted(((pr["label"], max(v for v in pr["p"] if v is not None)) for pr in sw["products"]),
                              key=lambda r: -r[1])
                if self.vs.sw_sel and not any(r[0] == self.vs.sw_sel for r in rows):
                    rows.append((self.vs.sw_sel, None))
                for lab, worst in rows:
                    self.sw_pick.addItem(f"{lab}  ({'not present' if worst is None else f'{worst:.1f} dBm'})", lab)
            i = self.sw_pick.findData(self.vs.sw_sel) if self.vs.sw_sel else 0
            self.sw_pick.setCurrentIndex(max(i, 0))

    # ------------------------------------------------------------ edits: schematic, editor, undo
    def _discard_editor_edits(self) -> bool:
        return not self.editor.modified() or QMessageBox.question(
            self, "Netlist", "The netlist editor has changes that aren't applied. Discard them?") == QMessageBox.StandardButton.Yes

    def sch_op(self, op: dict, layout: bool = True):
        if not self._discard_editor_edits():
            self.sch.rebuild()
            return
        if layout:   # pin every block where it is drawn, so auto-placed ones stay put
            op["layout"] = self.sch.layout_now()
        before = self.model.text
        try:
            info = self.model.edit(op)
        except NetlistTextError as e:
            self.show_msgs([("err", str(e))])
            self.sch.load(self.info["schematic"], self.info["groups"], self.probe, self.vs.incident)
            return
        self._push_undo(before, info["text"])
        if op["op"] == "move":   # only the layout changed: no need to re-simulate
            self.info["schematic"], self.info["text"] = info["schematic"], info["text"]
            self.editor.load(info["text"], info["file"], info["can_save"])
            self.undo_btn.setEnabled(True)
            return
        self.load_model(info, keep=True, renames={op["block"]: op["to"]} if op["op"] == "rename" else {})
        if info.get("added"):
            item = self.sch.blocks.get(info["added"])
            if item:
                item.setSelected(True)
            self.panel.focus_block(info["added"])

    def _push_undo(self, before, after):
        if before != after:
            self.undo.append(before)
            del self.undo[:-100]
            self.undo_btn.setEnabled(True)

    def do_undo(self):
        if not self.undo or not self._discard_editor_edits():
            return
        text = self.undo.pop()
        try:
            info = self.model.set_text(text)
        except NetlistTextError as e:
            self.show_msgs([("err", f"undo failed: {e}")])
            return
        self.load_model(info, keep=True)

    def apply_text(self, text: str, save: bool, force: bool = False):
        try:
            info = self.model.set_text(text)
        except NetlistTextError as e:
            self.editor.show_note((f"line {e.line}: " if e.line else "") + str(e), "err")
            self.editor.code.mark_error(e.line)
            return
        before = self.info["text"]
        self._push_undo(before, info["text"])
        self.load_model(info)
        if not save:
            self.editor.show_note("Applied.")
            return
        try:
            self.model.save(force=force)
        except FileChangedError as e:
            if QMessageBox.question(self, "Save", f"{e}.\nOverwrite it with the editor text?") == QMessageBox.StandardButton.Yes:
                return self.apply_text(text, True, True)
            self.editor.show_note(f"Applied, not saved: {e}", "warn")
            return
        except (OSError, ValueError) as e:
            self.editor.show_note(f"Applied, not saved: {e}", "err")
            return
        self.editor.show_note(f"Applied and saved to {info['file']}.")

    def _refresh_pull(self):
        self.pull.setVisible(self.sliders_dirty)
        self.pull.setText("The sliders have changes that aren't in this text (applying the text resets them). "
                          "Pull slider values into text" if self.sliders_dirty else "")

    def pull_sliders(self):
        d = self.model.netlist(self.panel.collect(), self.vs.resolution if self.vs.solver != "tone" else None)
        cur = self.editor.code.textCursor()
        cur.select(QTextCursor.SelectionType.Document)
        cur.insertText(dump_text(d, self.model.format))   # one undo step in the editor
        self.sliders_dirty = False
        self._refresh_pull()
        self.editor.show_note("Text rewritten from the slider values. Apply or Save to keep it.")

    def export(self):
        name = (self.info["name"] or "circuit").replace(" ", "_") + "-edited.yaml"
        path, _ = QFileDialog.getSaveFileName(self, "Export netlist", name, "YAML (*.yaml *.yml);;JSON (*.json)")
        if not path:
            return
        d = self.model.netlist(self.panel.collect(), self.vs.resolution if self.vs.solver != "tone" else None)
        Path(path).write_text(dump_text(d, "json" if path.lower().endswith(".json") else "yaml"))
        self.statusBar().showMessage(f"exported to {path}", 5000)

    def _add_block(self, i):
        kind = self.add.itemData(i)
        with QSignalBlocker(self.add):
            self.add.setCurrentIndex(0)
        if kind:
            self.sch.add_block_at_centre(kind)

    def closeEvent(self, e):
        if self.editor.modified() and QMessageBox.question(
                self, "Quit", "The netlist editor has changes that aren't applied. Quit anyway?") != QMessageBox.StandardButton.Yes:
            e.ignore()
            return
        self.runner.stop()
        from ..analysis import shutdown_pool
        shutdown_pool()
        e.accept()


def _html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def run(netlist=None, circuit=None, settings=None, workers: int | None = None) -> int:
    """Open the desktop viewer for a netlist file or an in-memory Circuit; returns when it closes."""
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("hetsim")
    model = ViewerModel.from_file(netlist) if netlist else ViewerModel.from_circuit(circuit, settings)
    model.sweep_workers = workers
    if workers is None or workers > 1:   # start the band-sweep workers while the window opens
        from ..analysis import warm_pool
        threading.Thread(target=warm_pool, daemon=True).start()
    win = MainWindow(model)
    win.show()
    # Ctrl-C / kill quit like closing the window; the timer lets Python see the signal while Qt waits
    import signal
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: app.quit())
    tick = QTimer()
    tick.timeout.connect(lambda: None)
    tick.start(200)
    try:
        return app.exec()
    finally:
        win.runner.stop()
        from ..analysis import shutdown_pool
        shutdown_pool()


def main() -> int:
    """hetsim-app [netlist.yaml]: the desktop viewer, asking for a netlist when none is given."""
    import argparse
    ap = argparse.ArgumentParser(prog="hetsim-app", description="hetsim desktop viewer")
    ap.add_argument("netlist", nargs="?")
    ap.add_argument("--workers", type=int)
    a = ap.parse_args()
    netlist = a.netlist
    if not netlist:
        app = QApplication.instance() or QApplication(sys.argv[:1])
        netlist, _ = QFileDialog.getOpenFileName(None, "Open netlist", "", "Netlists (*.yaml *.yml *.json)")
        if not netlist:
            return 0
    return run(netlist, workers=a.workers)


if __name__ == "__main__":
    sys.exit(main())
