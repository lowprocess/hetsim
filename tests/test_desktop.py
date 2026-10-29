"""Desktop viewer: the Qt window driven in-process (offscreen), and its pure plotting helpers."""
import gc
import math
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, QSettings, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from hetsim.desktop.tracks import auto_layout, interp_track, pn_at, rbw_trace, route, step_at  # noqa: E402
from hetsim.viewer import ViewerModel  # noqa: E402

VHF = Path(__file__).parent.parent / "examples" / "vhf_single_conversion.yaml"


# ---------------------------------------------------------------- pure helpers
def test_interp_track_matches_the_page():
    f, p, at = interp_track([10, 12, 14, None, 20, 22], [-80, -60, -60, None, -50, -40], 4, smooth=False)
    assert len(f) == 4 * 2 + 1 + 1 + 4 + 1 and f[1] == 10.5 and f[4] == 12
    assert f[9] is None and p[2] == -70 and at[2] == 0.5
    _, ps, _ = interp_track([10, 12, 14], [-80, -60, -60], 4, smooth=True)
    assert ps[4:9] == [-60] * 5                                  # flat stays flat: no overshoot
    assert all(b >= a for a, b in zip(ps[:5], ps[1:5]))         # monotone where the data is


def test_rbw_trace_and_profile():
    tr = rbw_trace(0, 2e6, 2000, [(1e6, -20.0)], 10e3)
    i = int(1e6 / 1e3)
    assert max(tr) == pytest.approx(-20, abs=0.01) and abs(int(tr.argmax()) - i) <= 1
    assert tr[i + 5] == pytest.approx(-20 - 10 * math.log10(2) * (2 * 5e3 / 10e3) ** 2, abs=0.5)   # -3 dB at +-RBW/2
    prof = [[1e3, -80], [1e5, -120]]
    assert pn_at(prof, 1e4) == pytest.approx(-100) and pn_at(prof, 1e6) is None
    assert step_at([{"LO": 100.0}, {"LO": 110.0}], 0.25) == {"LO": 102.5}


def test_auto_layout_and_route():
    blocks = [{"name": "ant", "ports": ["out"]}, {"name": "mix", "ports": ["rf", "lo", "if"]},
              {"name": "lo", "ports": ["out"]}, {"name": "iff", "ports": ["in", "out"]}]
    pos = auto_layout(blocks, [["ant.out", "mix.rf"], ["lo.out", "mix.lo"], ["mix.if", "iff.in"]])
    assert pos["ant"] == (0, 0) and pos["mix"] == (120, 0) and pos["iff"] == (240, 0)
    assert pos["lo"] == (0, -100)                                 # LO above, one column before its mixer
    pts = route((64, 24, 1, 0), (120, 24, -1, 0))
    assert pts[0] == (64, 24) and pts[-1] == (120, 24)


# ---------------------------------------------------------------- the window
@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(app, tmp_path):
    from hetsim.desktop.app import MainWindow
    f = tmp_path / "rx.yaml"
    f.write_text(VHF.read_text())
    m = ViewerModel.from_file(f)
    m.sweep_workers = 1
    w = MainWindow(m, sync=True, settings=_settings(tmp_path))
    w.resize(1400, 900)
    w.show()
    app.processEvents()
    yield w
    _dispose(w, app)


def _settings(tmp_path):
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)   # not the user's own


def _dispose(w, app):
    w.runner.stop()
    w.editor.load(w.model.text, None, False)   # no unsaved-edits prompt on close
    # delete the window here, not whenever the garbage collector runs (which may be in the
    # middle of building the next test's widgets)
    w.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    w.close()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    del w
    gc.collect()


def _paint(w):
    """Render the window as if on screen (runs every paintEvent)."""
    return w.grab()


def test_window_simulates_and_draws(win):
    _paint(win)
    assert win.vs.data["tone"] and win.plot.points and len(win.sch.blocks) == 7 and len(win.sch.wires) == 6
    assert win.probe == "ifamp.out" and "N = 21504" in win.status.text()


def test_slider_edit_resimulates(win):
    row = win.panel.rows[("lna", "gain_db")]
    before = {t["label"]: t["p"] for t in win.vs.data["tone"]}["LO - RF"]
    row.edit.setText("25")
    row._typed()
    after = {t["label"]: t["p"] for t in win.vs.data["tone"]}["LO - RF"]
    assert after == pytest.approx(before + 10, abs=0.2) and win.sliders_dirty
    assert win.panel.collect()["lna"]["gain_db"] == 25


def test_schematic_edits_and_undo(win):
    win.sch_op({"op": "add", "type": "amplifier", "x": 600, "y": 200})
    assert "amp1" in win.sch.blocks and "amp1:" in win.model.text
    win.sch_op({"op": "connect", "a": "ifamp.out", "b": "amp1.in"})
    assert ["ifamp.out", "amp1.in"] in win.info["schematic"]["wires"]
    win.sch_op({"op": "rename", "block": "amp1", "to": "post"})
    assert "post" in win.sch.blocks and win.panel.rows.get(("post", "gain_db"))
    for _ in range(3):
        win.do_undo()
    assert win.model.text == VHF.read_text() and "amp1" not in win.sch.blocks


def test_dragging_a_block_and_wiring_with_the_mouse(win, app):
    view, vp = win.sch, win.sch.viewport()

    def send(kind, scene_pt, buttons):
        p = QPointF(view.mapFromScene(scene_pt))
        btn = Qt.MouseButton.LeftButton
        e = QMouseEvent(kind, p, view.viewport().mapToGlobal(p), btn if kind != QEvent.Type.MouseMove else Qt.MouseButton.NoButton,
                        buttons, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(vp, e)
        app.processEvents()

    lna = view.blocks["lna"]
    start = lna.scenePos() + QPointF(32, 24)
    send(QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
    for k in range(1, 6):
        send(QEvent.Type.MouseMove, start + QPointF(0, 16 * k), Qt.MouseButton.LeftButton)
    send(QEvent.Type.MouseButtonRelease, start + QPointF(0, 80), Qt.MouseButton.NoButton)
    assert view.pos["lna"] == (240.0, 80.0)                     # auto-placed at (240, 0), dragged 80 down, on the grid
    assert win.info["schematic"]["layout"]["lna"] == {"x": 240, "y": 80} and "lna: {x: 240, y: 80}" in win.model.text

    win.sch_op({"op": "disconnect", "a": "lo.out"})
    a = view.blocks["lo"].ports["out"].scenePos()
    b = view.blocks["mix"].ports["lo"].scenePos()
    send(QEvent.Type.MouseButtonPress, a, Qt.MouseButton.LeftButton)
    for k in range(1, 6):
        send(QEvent.Type.MouseMove, a + (b - a) * (k / 5), Qt.MouseButton.LeftButton)
    send(QEvent.Type.MouseButtonRelease, b, Qt.MouseButton.NoButton)
    assert ["lo.out", "mix.lo"] in win.info["schematic"]["wires"]

    p = view.blocks["ant"].ports["out"].scenePos()   # a click on a port probes it
    send(QEvent.Type.MouseButtonPress, p, Qt.MouseButton.LeftButton)
    send(QEvent.Type.MouseButtonRelease, p, Qt.MouseButton.NoButton)
    assert win.probe == "ant.out" and win.port.currentText() == "ant.out"


def test_editor_reports_errors_and_applies(win):
    text = win.model.text
    win.apply_text(text.replace("type: amplifier, gain_db: 15", "type: amplifier, gane_db: 15"), False)
    assert "gane_db" in win.editor.note.text() and win.editor.code.err_line == 20
    win.apply_text(text.replace("ifamp:", "ifa:").replace("iff, ifamp]", "iff, ifa]"), True)
    assert "ifa" in win.sch.blocks and win.probe == "ifa.out" and "ifa:" in Path(win.info["file"]).read_text()


def test_sweep_controls_rbw_and_phase_noise(win):
    text = win.model.text.replace(
        "lo:    {type: source, name: LO, freq: 110.7MHz, power: 7, harmonics_dbc: [-30, -40]}",
        "lo:    {type: source, tones: [{name: LO, freq: 110.7MHz, power: 7, band: [98.7MHz, 118.7MHz], mode: sweep, "
        "phase_noise: [[10kHz, -90], [1MHz, -135], [10MHz, -145]]}]}")
    win.apply_text(text, False)
    win.set_probe("mix.if")
    assert win.vs.data["sweep"]["products"] and not win.swbar.isHidden()
    win.set_sw_out((5e6, 15e6))
    win.sweep_pts = 11
    win.simulate()
    lo_rf = {p["label"]: p for p in win.vs.data["sweep"]["products"]}["LO - RF"]
    assert [round(f) for f in lo_rf["f"]] == [round(5e6 + 1e6 * k) for k in range(11)]
    win.sw_interp.setCurrentIndex(5)                 # smooth x16
    win.set_sw_sel("LO - RF")
    _paint(win)
    assert any(p["kind"] == "sweep" and not float(p["at"]).is_integer() for p in win.plot.points)
    win.solver_group.actions()[2].trigger()          # tone solver
    win.rbw.setCurrentText("10 kHz")
    assert win.vs.data["skirts"] and win.vs.data["has_pn"] and win.pn_act.isVisible()
    _paint(win)
    win.pn.setChecked(False)
    assert win.vs.data["skirts"] == []


def test_phase_noise_table_and_limits(win):
    row = win.panel.rows[("lo", "tones.0.phase_noise")]
    row.en.setChecked(True)
    assert win.panel.collect()["lo"]["tones.0.phase_noise"][0] == [1e3, -80.0]
    win.panel.limitsRequested.emit("pre", "f1", [80e6, 95e6])
    assert "f1: [80MHz, 95MHz]" in win.model.text
    assert win.panel.rows[("pre", "f1")].p["limits"] == [80e6, 95e6]


# ---------------------------------------------------------------- without a netlist
@pytest.fixture
def bare(app, tmp_path):
    from hetsim.desktop.app import MainWindow
    w = MainWindow(None, sync=True, settings=_settings(tmp_path), workers=1)
    w.show()
    app.processEvents()
    yield w
    if w.model is not None:
        _dispose(w, app)
    else:
        w.runner.stop()
        w.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        w.close()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        del w
        gc.collect()


def test_opens_on_the_start_page(bare):
    assert bare.pages.currentWidget() is bare.start and bare.model is None
    assert bare.view_bar.isHidden() and not bare.save_act.isEnabled() and not bare.recent_menu.isEnabled()
    assert bare.start.examples is not None and bare.start.examples.count() >= 3      # source checkout
    assert bare.start.recent.isHidden()
    _paint(bare)


def test_new_circuit_then_save_as_and_reopen(bare, tmp_path, monkeypatch):
    bare.new_circuit()
    assert bare.pages.currentIndex() == 1 and set(bare.sch.blocks) == {"src1", "load1"}
    assert bare.vs.data["tone"] and bare.probe == "src1.out" and "not saved" in bare.windowTitle()
    bare.sch_op({"op": "add", "type": "amplifier", "x": 120, "y": 100})
    from PySide6.QtWidgets import QFileDialog
    target = tmp_path / "mine.yaml"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    bare.save()                                           # no file yet: Save goes to Save As
    assert "amp1:" in target.read_text() and "mine.yaml" in bare.windowTitle() and bare.editor.can_save
    assert bare._recent() == [str(target.resolve())] and bare.recent_menu.isEnabled()
    bare.close_model()
    assert bare.pages.currentWidget() is bare.start and bare.start.recent.count() == 1
    bare.start.openPath.emit(str(target))                 # reopen from the recent list
    assert "amp1" in bare.sch.blocks and bare.model.source_path == target


def test_open_errors_keep_the_window(bare, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a[2]))
    bad = tmp_path / "bad.yaml"
    bad.write_text("blocks: {x: {type: nosuch}}\n")
    bare.open_path(str(bad))
    assert shown and "nosuch" in shown[0] and bare.pages.currentWidget() is bare.start
    bare.open_path(str(VHF))
    assert bare.pages.currentIndex() == 1 and len(bare.sch.blocks) == 7
