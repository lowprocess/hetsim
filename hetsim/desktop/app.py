"""hetsim desktop viewer: python -m hetsim app [netlist.yaml]

A native window around ViewerModel (the same model the browser viewer serves): parameters
on the left, schematic, spectrum and waveform on the right, simulations on a worker thread.
No server and no browser. Without a netlist it opens on a start page (new, open, recent).
"""
from __future__ import annotations

import math
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QSettings, QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
                               QLineEdit, QMainWindow, QMessageBox, QPushButton, QSlider, QSpinBox, QSplitter,
                               QStackedWidget, QTabWidget, QToolBar, QVBoxLayout, QWidget)

from ..viewer.model import FileChangedError, NetlistTextError, ViewerModel, dump_text
from . import theme
from .editor import NetlistEditor
from .params import ParamPanel, carry_values, parse_f
from .plot import SpectrumPlot, ViewState, WaveformPlot, XYView, fmt_f
from .schematic import SchematicView
from .sim import SimRunner
from .start import StartPage

NETLISTS = "Netlists (*.yaml *.yml *.json);;All files (*)"
RECENT_MAX = 10
RBWS = ["off", "1 kHz", "3 kHz", "10 kHz", "30 kHz", "100 kHz", "300 kHz", "1 MHz", "3 MHz"]
INTERP = [("0", "no interpolation"), ("lin4", "linear ×4"), ("lin8", "linear ×8"),
          ("pchip4", "smooth ×4"), ("pchip8", "smooth ×8"), ("pchip16", "smooth ×16")]


class MainWindow(QMainWindow):
    def __init__(self, model: ViewerModel | None = None, sync: bool = False, settings: QSettings | None = None,
                 workers: int | None = None):
        super().__init__()
        self.model = model
        self.sync = sync   # tests: simulate on the calling thread
        self.settings = settings or QSettings("hetsim", "hetsim")
        self.workers = workers if model is None else model.sweep_workers
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
        self.setAcceptDrops(True)
        self._build()
        if model is not None:
            self.set_model(model)
        else:
            self.show_start()

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
        self.view_bar = tb
        self._menus()

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
        # animated sweep: step the swept tones through the sweep, a full simulation per frame
        self.swbar.addSeparator()
        self.play = QPushButton("▶")
        self.play.setCheckable(True)
        self.play.setFixedWidth(34)
        self.play.setToolTip("Animate the sweep: the swept tones step through it and the spectrum follows")
        self.play.toggled.connect(self._play)
        self.stop_btn = QPushButton("■")
        self.stop_btn.setFixedWidth(34)
        self.stop_btn.setToolTip("Stop and go back to the tones' own frequencies")
        self.stop_btn.clicked.connect(self.stop_animation)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimumWidth(160)
        self.frame_slider.setToolTip("Sweep step shown (drag to scrub)")
        self.frame_slider.valueChanged.connect(self._scrub)
        self.fps = QSpinBox()
        self.fps.setRange(1, 30)
        self.fps.setValue(8)
        self.fps.setSuffix(" fps")
        self.fps.setToolTip("Animation speed (frames per second, at most; each frame is a simulation)")
        self.fps.valueChanged.connect(lambda v: self.anim.setInterval(int(1000 / v)))
        self.loop = QCheckBox("loop")
        self.loop.setChecked(True)
        self.frame_label = QLabel()
        for w in (self.play, self.stop_btn, self.frame_slider, self.fps, self.loop, self.frame_label):
            self.swbar.addWidget(w)
        self.anim = QTimer(self)
        self.anim.setInterval(int(1000 / self.fps.value()))
        self.anim.timeout.connect(self._tick)
        self.frame = None          # sweep step shown, or None: the tones' own frequencies
        self.frame_busy = False
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
        # the sliders differ from the text: say so (wrapped: the editor's width must not grow
        # the panel beside it, which shares its space) and offer to write them in
        self.pull_box = QWidget()
        pl = QVBoxLayout(self.pull_box)
        pl.setContentsMargins(8, 0, 8, 6)
        note = QLabel("The sliders have changes that aren't in this text; applying the text resets them.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {theme.c('warn').name()}")
        self.pull = QPushButton("Pull slider values into text")
        self.pull.setToolTip("Rewrite the text from the current settings. Comments are dropped (Ctrl+Z undoes it).")
        self.pull.clicked.connect(self.pull_sliders)
        pl.addWidget(note)
        pl.addWidget(self.pull, 0, Qt.AlignmentFlag.AlignLeft)
        self.pull_box.hide()
        self.editor.layout().insertWidget(2, self.pull_box)
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
        self.sch.blockClicked.connect(self.panel.show_block)
        self.panel.blockChosen.connect(self.sch.select_block)
        self.sch.undoRequested.connect(self.do_undo)
        self.dele.clicked.connect(self.sch.delete_selected)
        self.undo_btn.clicked.connect(self.do_undo)
        fit.clicked.connect(self.sch.fit)
        right.addWidget(self.sch_panel)
        self.plot = SpectrumPlot(self.vs)
        self.plot.trackClicked.connect(lambda lab: self.set_sw_sel(None if lab == self.vs.sw_sel else lab))
        right.addWidget(self.plot)
        self.wave = WaveformPlot(self.vs)
        self.xy_tabs = QTabWidget()
        self.xy_tabs.setDocumentMode(True)
        self.xy_tabs.setVisible(False)
        self.xy_title = QLabel()
        self.xy_title.setStyleSheet("color: gray; padding-right: 8px")
        self.xy_tabs.setCornerWidget(self.xy_title, Qt.Corner.TopRightCorner)
        self.xy_tabs.currentChanged.connect(lambda _: self._update_xy())
        right.addWidget(self.xy_tabs)
        right.addWidget(self.wave)
        self.msgs = QLabel()
        self.msgs.setWordWrap(True)
        self.msgs.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.msgs.setContentsMargins(8, 2, 8, 2)
        right.addWidget(self.msgs)
        right.setSizes([290, 400, 220, 130, 40])
        right.setStretchFactor(1, 1)
        split.addWidget(right)
        split.setSizes([360, 1040])
        split.setStretchFactor(1, 1)
        self.start = StartPage()
        self.start.newRequested.connect(self.new_circuit)
        self.start.openRequested.connect(self.open_dialog)
        self.start.openPath.connect(self.open_path)
        self.pages = QStackedWidget()
        self.pages.addWidget(self.start)
        self.pages.addWidget(split)
        self.setCentralWidget(self.pages)

        self.status = QLabel()
        self.busy = QLabel()
        self.busy.setStyleSheet(f"color: {theme.c('accent').name()}")
        self.statusBar().addWidget(self.busy)
        self.statusBar().addPermanentWidget(self.status)
        esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        esc.activated.connect(lambda: self.set_sw_sel(None) if self.vs.sw_sel else None)
        QShortcut(QKeySequence.StandardKey.Undo, self.sch, activated=self.do_undo)

    def _menus(self):
        m = self.menuBar().addMenu("&File")

        def act(text, key, fn, tip=""):
            a = QAction(text, self)
            if key:
                a.setShortcut(key)
            a.setStatusTip(tip)
            a.triggered.connect(fn)
            m.addAction(a)
            return a
        act("&New circuit", QKeySequence.StandardKey.New, self.new_circuit, "Start from one source wired to a load")
        act("&Open…", QKeySequence.StandardKey.Open, self.open_dialog, "Open a YAML or JSON netlist")
        self.recent_menu = m.addMenu("Open &Recent")
        m.addSeparator()
        self.save_act = act("&Save", QKeySequence.StandardKey.Save, self.save,
                            "Write the netlist to its file (applying the editor's text first)")
        self.save_as_act = act("Save &As…", QKeySequence.StandardKey.SaveAs, self.save_as, "Write the netlist to a new file")
        self.export_act = act("&Export netlist…", None, self.export, "Save the netlist with the current slider settings")
        m.addSeparator()
        self.close_act = act("&Close", QKeySequence.StandardKey.Close, self.close_model, "Back to the start page")
        act("&Quit", QKeySequence.StandardKey.Quit, self.close)
        self._refresh_recent()
        tools = self.menuBar().addMenu("&Analysis")
        a = QAction("&Noise figure, VSWR and budget…", self)
        a.setStatusTip("Noise figure, port match, receiver budget, parameter sweeps and Monte Carlo")
        a.triggered.connect(self.open_analysis)
        tools.addAction(a)

    def open_analysis(self):
        if self.model is None:
            return
        from .analysis import AnalysisDialog
        self.analysis = AnalysisDialog(self.model, self.panel.collect(), self)
        self.analysis.show()

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
        self._title()
        with QSignalBlocker(self.port):
            self.port.clear()
            self.port.addItems(info["ports"])
            self.port.setCurrentText(self.probe)
        self.res.setText(fmt_f(self.vs.resolution))
        shown = renames.get(self.panel.current, self.panel.current)
        self.panel.build(info["groups"], shown)
        self.editor.load(info["text"], info["file"], info["can_save"])
        self._refresh_pull()
        with QSignalBlocker(self.add):
            self.add.clear()
            self.add.addItem("+ Add block")
            for t in info["schematic"]["addable"]:
                self.add.addItem(t.replace("_", " "), t)
        self.sch.load(info["schematic"], info["groups"], self.probe, self.vs.incident, fit=first)
        self.sch.select_block(self.panel.current)
        if first:
            QTimer.singleShot(0, self.sch, self.sch.fit)   # dropped if the view is gone by then
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
            # paused or playing an animation: the swept tones sit at that step
            "tone_freqs": self._frame_freqs(),
        }

    def simulate(self, frame_only: bool = False):
        req = self.request()
        if frame_only:   # an animation frame: keep the sweep and X/Y plots, only move the tones
            req["sweep_frame"] = True
            self.frame_busy = True
        if self.sync:
            self.on_result(self.runner.run_now(req), req)
        else:
            self.runner.submit(req)

    def on_result(self, d: dict, req):
        if req and req.get("sweep_frame"):
            self.frame_busy = False
            if self.vs.data:   # a frame doesn't recompute these
                for k in ("sweep", "xy"):
                    if k in self.vs.data:
                        d.setdefault(k, self.vs.data[k])
        if d.get("error"):
            self.show_msgs([("err", d["error"])])
            return
        self.vs.data = d
        self.vs.extra["pn_on"] = self.pn_on
        if not (req and req.get("tone_freqs")):   # a frame's tone positions must not stick to the sliders
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
        self.sch.refresh(readings=d.get("detectors", {}))
        self._update_frames()
        self._update_xy()
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

    # ------------------------------------------------------------ animated sweep, X/Y plots
    def _steps(self) -> list:
        sw = self.vs.data and self.vs.data.get("sweep")
        return sw["steps"] if sw else []

    def _frame_freqs(self):
        steps = self._steps()
        return steps[self.frame] if self.frame is not None and self.frame < len(steps) else None

    def _update_frames(self):
        steps = self._steps()
        if not steps:
            if self.frame is not None:
                self.stop_animation(resimulate=False)
            return
        with QSignalBlocker(self.frame_slider):
            self.frame_slider.setRange(0, len(steps) - 1)
            if self.frame is not None:
                self.frame = min(self.frame, len(steps) - 1)
                self.frame_slider.setValue(self.frame)
        self.vs.extra["frame"] = self.frame
        self.vs.extra["frame_freqs"] = steps[self.frame] if self.frame is not None else None
        if self.frame is None:
            self.frame_label.setText("")
        else:
            at = ", ".join(f"{k} {fmt_f(v, 7)}" for k, v in steps[self.frame].items())
            self.frame_label.setText(f" step {self.frame + 1}/{len(steps)}: {at}")

    def _play(self, on: bool):
        self.play.setText("⏸" if on else "▶")
        if on and self._steps():
            if self.frame is None:
                self.frame = -1
            self.anim.start()
        else:
            self.anim.stop()

    def _tick(self):
        steps = self._steps()
        if not steps:
            self.play.setChecked(False)
            return
        if self.frame_busy:   # the last frame is still simulating: drop this tick
            return
        nxt = (self.frame if self.frame is not None else -1) + 1
        if nxt >= len(steps):
            if not self.loop.isChecked():
                self.play.setChecked(False)
                return
            nxt = 0
        self.frame = nxt
        self._update_frames()
        self.simulate(frame_only=True)

    def _scrub(self, v: int):
        self.play.setChecked(False)
        self.frame = v
        self._update_frames()
        self.simulate(frame_only=True)

    def stop_animation(self, resimulate: bool = True):
        self.anim.stop()
        with QSignalBlocker(self.play):
            self.play.setChecked(False)
            self.play.setText("▶")
        was, self.frame = self.frame, None
        self.frame_busy = False
        self.vs.extra["frame"] = self.vs.extra["frame_freqs"] = None
        self.frame_label.setText("")
        if resimulate and was is not None and self.model is not None:
            self.simulate()

    def _update_xy(self):
        recs = (self.vs.data or {}).get("xy") or []
        names = [r["name"] for r in recs]
        have = [self.xy_tabs.tabText(i) for i in range(self.xy_tabs.count())]
        if names != have:
            with QSignalBlocker(self.xy_tabs):   # adding tabs would call back in here
                while self.xy_tabs.count():
                    w = self.xy_tabs.widget(0)
                    self.xy_tabs.removeTab(0)
                    w.deleteLater()
                for n in names:
                    self.xy_tabs.addTab(XYView(self.vs, n), n)
        self.xy_tabs.setVisible(bool(names))
        for i, r in enumerate(recs):
            self.xy_tabs.setTabToolTip(i, r.get("title") or r.get("note") or "")
            self.xy_tabs.setTabText(i, r["name"])
            self.xy_tabs.widget(i).update()
        if recs:
            t = recs[min(self.xy_tabs.currentIndex(), len(recs) - 1)]
            self.xy_title.setText(t.get("title") or t.get("note") or "")

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
        if info.get("added"):   # a new block: select it and show its parameters
            self.panel.show_block(info["added"])
            self.sch.select_block(info["added"])

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
        self.pull_box.setVisible(self.sliders_dirty)

    def pull_sliders(self):
        d = self.model.netlist(self.panel.collect(), self.vs.resolution if self.vs.solver != "tone" else None)
        cur = self.editor.code.textCursor()
        cur.select(QTextCursor.SelectionType.Document)
        cur.insertText(dump_text(d, self.model.format))   # one undo step in the editor
        self.sliders_dirty = False
        self._refresh_pull()
        self.editor.show_note("Text rewritten from the slider values. Apply or Save to keep it.")

    # ------------------------------------------------------------ files
    def set_model(self, model: ViewerModel) -> bool:
        """Show `model` in the window (a file opened, a new circuit); False if the user kept the old one."""
        if self.model is not None and self.info is not None and not self._discard_editor_edits():
            return False
        if model.sweep_workers is None:
            model.sweep_workers = self.workers
        self.model = self.runner.model = model
        self.stop_animation(resimulate=False)
        self.info, self.undo, self.sliders_dirty = None, [], False
        self.vs.data, self.vs.view, self.vs.sw_sel = None, None, None
        self.vs.sw_out = None
        with QSignalBlocker(self.sw_place):
            self.sw_place.setCurrentIndex(0)
        for a in self.sw_out_acts:
            a.setVisible(False)
        self.ed_btn.setChecked(False)
        self.editor.show_note(None)
        self._workspace(True)
        self.load_model(model.info(), first=True)
        if model.source_path:
            self._remember(model.source_path)
        return True

    def _workspace(self, on: bool):
        self.pages.setCurrentIndex(1 if on else 0)
        self.view_bar.setVisible(on)
        if not on:
            self.swbar.setVisible(False)
        for a in (self.save_act, self.save_as_act, self.export_act, self.close_act):
            a.setEnabled(on)

    def show_start(self):
        self.model = self.info = None
        self.setWindowTitle("hetsim")
        self.start.set_recent(self._recent())
        self.status.setText("")
        self._workspace(False)

    def close_model(self):
        if self.model is None or self._discard_editor_edits():
            self.show_start()

    def new_circuit(self):
        self.set_model(ViewerModel.new())

    def open_dialog(self):
        start = str(self.model.source_path.parent) if self.model and self.model.source_path else ""
        path, _ = QFileDialog.getOpenFileName(self, "Open netlist", start, NETLISTS)
        if path:
            self.open_path(path)

    def open_path(self, path: str):
        try:
            model = ViewerModel.from_file(path)
        except Exception as e:  # noqa: BLE001  - a bad file must not close the window
            self._forget(path) if not Path(path).exists() else None
            QMessageBox.warning(self, "Open netlist", f"Couldn't open {path}:\n{type(e).__name__}: {e}")
            if self.model is None:
                self.start.set_recent(self._recent())
            return
        self.set_model(model)

    def save(self):
        """Write the netlist to its file; text waiting in the editor is applied first."""
        if self.model is None:
            return
        if self.model.source_path is None:
            return self.save_as()
        if self.editor.modified():
            return self.apply_text(self.editor.code.toPlainText(), True)
        try:
            self.model.save()
        except FileChangedError as e:
            if QMessageBox.question(self, "Save", f"{e}.\nOverwrite it?") != QMessageBox.StandardButton.Yes:
                return
            self.model.save(force=True)
        except OSError as e:
            QMessageBox.warning(self, "Save", f"Couldn't save: {e}")
            return
        self.statusBar().showMessage(f"saved {self.model.source_path}", 5000)

    def save_as(self):
        if self.model is None:
            return
        if self.editor.modified():   # save what the editor shows
            self.apply_text(self.editor.code.toPlainText(), False)
            if self.editor.modified():
                return   # it didn't apply: the error is shown in the editor
        here = self.model.source_path
        name = str(here) if here else (self.info["name"] or "circuit").replace(" ", "_") + ".yaml"
        path, _ = QFileDialog.getSaveFileName(self, "Save netlist as", name, "YAML (*.yaml *.yml);;JSON (*.json)")
        if not path:
            return
        try:
            info = self.model.save_as(path)
        except (OSError, NetlistTextError) as e:
            QMessageBox.warning(self, "Save as", f"Couldn't save: {e}")
            return
        self.editor.load(info["text"], info["file"], info["can_save"])
        self.info.update(file=info["file"], can_save=True, text=info["text"])
        self._title()
        self._remember(path)
        self.statusBar().showMessage(f"saved {path}", 5000)

    def _title(self):
        name = self.info["name"] if self.info else ""
        f = Path(self.info["file"]).name if self.info and self.info.get("file") else "not saved"
        self.setWindowTitle(f"{name or 'circuit'} — {f} · hetsim")

    def _recent(self) -> list[str]:
        v = self.settings.value("recent", [])
        v = [v] if isinstance(v, str) else list(v or [])
        return [p for p in v if Path(p).exists()]

    def _remember(self, path):
        p = str(Path(path).resolve())
        self.settings.setValue("recent", ([p] + [q for q in self._recent() if q != p])[:RECENT_MAX])
        self._refresh_recent()

    def _forget(self, path):
        self.settings.setValue("recent", [q for q in self._recent() if q != str(Path(path).resolve())])
        self._refresh_recent()

    def _refresh_recent(self):
        self.recent_menu.clear()
        paths = self._recent()
        for p in paths:
            a = self.recent_menu.addAction(Path(p).name)
            a.setStatusTip(p)
            a.triggered.connect(lambda _=False, p=p: self.open_path(p))
        self.recent_menu.setEnabled(bool(paths))

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(u.isLocalFile() for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        files = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if files:
            self.open_path(files[0])

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
        if self.model is not None and self.editor.modified() and QMessageBox.question(
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
    """Open the desktop viewer on a netlist file, an in-memory Circuit, or (neither) the start
    page; returns when it closes."""
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("hetsim")
    model = (ViewerModel.from_file(netlist) if netlist
             else ViewerModel.from_circuit(circuit, settings) if circuit is not None else None)
    if model is not None:
        model.sweep_workers = workers
    if workers is None or workers > 1:   # start the band-sweep workers while the window opens
        from ..analysis import warm_pool
        threading.Thread(target=warm_pool, daemon=True).start()
    win = MainWindow(model, workers=workers)
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
    """hetsim-app [netlist.yaml]: the desktop viewer; without a netlist it opens on the start page."""
    import argparse
    ap = argparse.ArgumentParser(prog="hetsim-app", description="hetsim desktop viewer")
    ap.add_argument("netlist", nargs="?")
    ap.add_argument("--workers", type=int)
    a = ap.parse_args()
    return run(a.netlist, workers=a.workers)


if __name__ == "__main__":
    sys.exit(main())
