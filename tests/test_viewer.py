"""Viewer back end: parameter discovery, edits, and the HTTP API."""
import json
import math
import threading
import urllib.request
from pathlib import Path

import pytest

from hetsim.viewer import NetlistTextError, ViewerModel, apply_edits, describe, make_server
from hetsim.viewer.model import FileChangedError

VHF = Path(__file__).parent.parent / "examples" / "vhf_single_conversion.yaml"


@pytest.fixture(scope="module")
def model():
    return ViewerModel.from_file(VHF)


def test_describe_covers_every_block(model):
    groups = {g["block"]: g for g in describe(model.base)}
    assert set(groups) == set(model.base["blocks"])
    paths = {p["path"] for p in groups["ant"]["params"]}
    assert {"tones.0.freq", "tones.0.power", "tones.3.freq"} <= paths
    mix = {p["path"]: p for p in groups["mix"]["params"]}
    assert mix["iip3_dbm"]["enabled"] and mix["iip3_dbm"]["value"] == 15
    assert mix["iip2_dbm"]["optional"] and not mix["iip2_dbm"]["enabled"]
    pre = {p["path"]: p for p in groups["pre"]["params"]}
    assert pre["fc"]["when"] == {"response": ["lowpass", "highpass"]}
    assert "f0" not in pre                       # netlist uses f1/f2, so f0/bw are not offered


def test_edits_snap_to_resolution(model):
    d, snapped = apply_edits(model.base, {"ant": {"tones.0.freq": 100.03e6}}, 100e3)
    assert d["blocks"]["ant"]["tones"][0]["freq"] == pytest.approx(100.0e6)
    assert snapped["ant"]["tones.0.freq"] == pytest.approx(100.0e6)
    assert model.base["blocks"]["ant"]["tones"][0]["freq"] == 100e6   # base untouched


def test_simulate_td_and_tone(model):
    r = model.simulate({"probe": "ifamp.out", "solver": "both", "min_dbm": -150,
                        "params": {"lo": {"tones.0.freq": 110.8e6}}})
    assert r["converged"] and r["resolution"] == 100e3
    f = dict(zip(r["td"]["f"], r["td"]["labels"]))
    assert any("LO - RF" in labs for fq, labs in f.items() if abs(fq - 10.8e6) < 1)
    assert any(t["label"] == "LO - RF" and abs(t["f"] - 10.8e6) < 1 for t in r["tone"])
    assert len(r["waveform"]["min"]) == len(r["waveform"]["max"]) > 0


def test_disabling_a_parameter(model):
    on = model.simulate({"probe": "mix.if", "solver": "tone", "min_dbm": -200, "params": {}})
    off = model.simulate({"probe": "mix.if", "solver": "tone", "min_dbm": -200,
                          "params": {"mix": {"lo_if_isolation_db": None}}})
    has_lo = lambda r: any(t["label"] == "LO" for t in r["tone"])  # noqa: E731
    assert has_lo(on) and not has_lo(off)


def test_http_api():
    srv = make_server(ViewerModel.from_file(VHF), port=0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/"
    try:
        page = urllib.request.urlopen(base).read().decode()
        assert "<canvas id=\"fft\">" in page
        info = json.load(urllib.request.urlopen(base + "api/model"))
        assert info["probe"] == "ifamp.out"

        def post(path, body):
            req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST")
            return urllib.request.urlopen(req).read().decode()

        r = json.loads(post("api/sim", {"probe": "ant.out", "incident": True, "solver": "td"}))
        i = min(range(len(r["td"]["f"])), key=lambda k: abs(r["td"]["f"][k] - 110.7e6))
        assert abs(r["td"]["f"][i] - 110.7e6) < 1 and r["td"]["labels"][i] == ["LO"]
        bad = json.loads(post("api/sim", {"probe": "ifamp.out", "params": {"pre": {"order": 0}}}))
        assert "order" in bad["error"]
        text = post("api/export", {"params": {"lna": {"gain_db": 20}}})
        assert "gain_db: 20" in text
    finally:
        srv.shutdown()
        srv.server_close()


def test_set_text_rebuilds_and_keeps_source():
    m = ViewerModel.from_file(VHF)
    text = VHF.read_text()
    assert m.info()["text"] == text and m.info()["can_save"]
    edited = text.replace("lna:   {type: amplifier, gain_db: 15,", "lna:   {type: amplifier, gain_db: 22,")
    info = m.set_text(edited)
    assert m.base["blocks"]["lna"]["gain_db"] == 22 and info["text"] == edited   # comments kept
    lna = {p["path"]: p for g in info["groups"] if g["block"] == "lna" for p in g["params"]}
    assert lna["gain_db"]["value"] == 22


def test_set_text_errors_point_at_the_line():
    m = ViewerModel.from_file(VHF)
    text = VHF.read_text()
    with pytest.raises(NetlistTextError) as e:
        m.set_text(text.replace("  pre:   {type: filter,", "  pre:   {type: filter,,"))
    assert e.value.line is not None
    with pytest.raises(NetlistTextError) as e:
        m.set_text(text.replace("type: amplifier, gain_db: 15", "type: amplifier, gane_db: 15"))
    assert "lna" in str(e.value) and text.splitlines()[e.value.line - 1].lstrip().startswith("lna:")
    with pytest.raises(NetlistTextError):
        m.set_text(text.replace("[lo.out, mix.lo]", "[lo.out, nowhere.lo]"))
    assert m.text == text and m.base["blocks"]["lna"]["gain_db"] == 15            # old netlist kept


def test_save_refuses_a_file_changed_on_disk(tmp_path):
    f = tmp_path / "rx.yaml"
    f.write_text(VHF.read_text())
    m = ViewerModel.from_file(f)
    m.set_text(m.text.replace("gain_db: 20", "gain_db: 21"))
    m.save()
    assert "gain_db: 21" in f.read_text()
    f.write_text(f.read_text() + "# edited elsewhere\n")
    with pytest.raises(FileChangedError):
        m.save()
    m.save(force=True)
    assert "edited elsewhere" not in f.read_text()


def test_http_netlist_endpoint(tmp_path):
    f = tmp_path / "rx.yaml"
    f.write_text(VHF.read_text())
    srv = make_server(ViewerModel.from_file(f), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/"

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST")
        return json.load(urllib.request.urlopen(req))

    try:
        text = f.read_text()
        bad_text = text.replace("blocks:\n", "blocks:\n  oops: {type: nosuch}\n")
        bad = post("api/netlist", {"text": bad_text})
        assert "nosuch" in bad["error"] and bad_text.splitlines()[bad["line"] - 1] == "  oops: {type: nosuch}"
        r = post("api/netlist", {"text": text.replace("ifamp:", "ifa:").replace("iff, ifamp]", "iff, ifa]"),
                                 "save": True})
        assert r["saved"] and "ifa.out" in r["ports"] and "ifa:" in f.read_text()
        assert post("api/sim", {"probe": "ifa.out", "solver": "tone"})["tone"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_schematic_edits_keep_comments():
    m = ViewerModel.from_file(VHF)
    text = VHF.read_text()
    info = m.edit({"op": "move", "layout": {"ant": {"x": 0, "y": 0}, "lna": {"x": 240.4, "y": 20}}})
    assert m.text.startswith(text.rstrip("\n")) and "  lna: {x: 240, y: 20}" in m.text   # appended, rest intact
    assert info["schematic"]["layout"]["lna"] == {"x": 240, "y": 20}

    m.edit({"op": "rename", "block": "lna", "to": "preamp"})
    assert "chain: [ant, pre, preamp, mix, iff, ifamp]" in m.text and "# wanted" in m.text
    assert "  preamp: {x: 240, y: 20}" in m.text and "lna" not in m.text.split("blocks:")[1]

    added = m.edit({"op": "add", "type": "source", "x": 40, "y": -200})["added"]
    assert added == "src1" and m.base["blocks"]["src1"]["tones"][0]["name"] == "SRC1"

    m.edit({"op": "connect", "a": "src1.out", "b": "mix.lo"})          # replaces lo -> mix.lo
    wires = {frozenset(w) for w in m.info()["schematic"]["wires"]}
    assert frozenset(["src1.out", "mix.lo"]) in wires and not any("lo.out" in w for w in wires)
    assert "chain:" in m.text            # the chain's own links are untouched, so it stays

    m.edit({"op": "disconnect", "a": "mix.if"})                          # breaks the chain
    assert "chain:" not in m.text and "[iff.out, ifamp.in]" in m.text
    m.edit({"op": "delete", "block": "src1"})
    assert "src1" not in m.text and "# sits on the image" in m.text


def test_schematic_edit_errors_keep_the_netlist():
    m = ViewerModel.from_file(VHF)
    text = m.text
    for op in ({"op": "rename", "block": "lna", "to": "mix"}, {"op": "rename", "block": "lna", "to": "a.b"},
               {"op": "add", "type": "sparam"}, {"op": "connect", "a": "lna.in", "b": "lna.out"},
               {"op": "connect", "a": "lna.nope", "b": "mix.lo"}, {"op": "explode"}):
        with pytest.raises(NetlistTextError):
            m.edit(op)
    assert m.text == text


def test_schematic_json_netlist(tmp_path):
    from hetsim.viewer.model import parse_text
    f = tmp_path / "rx.json"
    f.write_text(json.dumps(parse_text(VHF.read_text())))
    m = ViewerModel.from_file(f)
    m.edit({"op": "add", "type": "load", "x": 0, "y": 0})
    m.edit({"op": "connect", "a": "ifamp.out", "b": "load1.in"})
    d = json.loads(m.text)
    assert d["blocks"]["load1"] == {"type": "load"} and ["ifamp.out", "load1.in"] in d["wires"]


def _band_model(tmp_path, mode="sweep", **extra):
    text = VHF.read_text().replace("{name: RF,  freq: 100MHz,   power: -80}",
                                   f"{{name: RF,  freq: 100MHz,   power: -80, band: [88MHz, 108MHz], mode: {mode}}}")
    f = tmp_path / "rx.yaml"
    f.write_text(text)
    return ViewerModel.from_file(f)


def test_describe_band_controls_and_limits(tmp_path):
    m = _band_model(tmp_path)
    ant = {p["path"]: p for g in m.info()["groups"] if g["block"] == "ant" for p in g["params"]}
    assert ant["tones.0.mode"]["value"] == "sweep" and ant["tones.1.mode"]["value"] == "cw"
    assert ant["tones.0.band.0"]["value"] == 88e6 and ant["tones.0.band.0"]["when"]["tones.0.mode"][0] == "tune"
    assert ant["tones.0.freq"]["band"] == ["tones.0.band.0", "tones.0.band.1", "tones.0.mode"]
    assert ant["tones.0.points"]["value"] == 41
    m.edit({"op": "limits", "block": "pre", "path": "f1", "range": [80e6, 95e6]})
    assert "limits: {f1: [80MHz, 95MHz]}" in m.text or "f1: [80MHz, 95MHz]" in m.text
    pre = {p["path"]: p for g in m.info()["groups"] if g["block"] == "pre" for p in g["params"]}
    assert pre["f1"]["limits"] == [80e6, 95e6] and pre["f1"]["min"] == 80e6
    d, _ = apply_edits(m.base, {"pre": {"f1": 50e6}}, None)
    assert d["blocks"]["pre"]["f1"] == 80e6                                   # clamped to the limit
    m.edit({"op": "limits", "block": "pre", "path": "f1", "range": None})
    assert "limits" not in m.base["blocks"]["pre"]


def test_apply_edits_band_modes(tmp_path):
    m = _band_model(tmp_path, "tune")
    d, _ = apply_edits(m.base, {"ant": {"tones.0.freq": 120e6}}, None)
    assert d["blocks"]["ant"]["tones"][0]["freq"] == 108e6                    # kept inside the band
    d, _ = apply_edits(m.base, {"ant": {"tones.0.mode": "cw", "tones.0.band.0": None}}, None)
    assert "band" not in d["blocks"]["ant"]["tones"][0] and "mode" not in d["blocks"]["ant"]["tones"][0]
    d, _ = apply_edits(m.base, {"ant": {"tones.1.mode": "wideband", "tones.1.band.0": 99.03e6,
                                        "tones.1.band.1": 101e6, "tones.1.points": 9}}, 100e3)
    t = d["blocks"]["ant"]["tones"][1]
    assert t["band"] == [99e6, 101e6] and t["step"] == pytest.approx(200e3)    # comb on the FFT grid


def test_simulate_returns_sweep_tracks(tmp_path):
    m = _band_model(tmp_path)
    r = m.simulate({"probe": "mix.if", "solver": "tone", "min_dbm": -150, "params": {}})
    tracks = {p["label"]: p for p in r["sweep"]["products"]}
    fs = [f for f in tracks["LO - RF"]["f"] if f is not None]
    assert len(r["sweep"]["steps"]) == 41 and min(fs) == pytest.approx(2.7e6) and max(fs) == pytest.approx(22.7e6)
    assert "sweep" not in ViewerModel.from_file(VHF).simulate({"probe": "mix.if", "solver": "tone", "params": {}})
    td = _band_model(tmp_path, "wideband").simulate({"probe": "ant.out", "solver": "td", "params": {}})
    assert td["converged"] and any("RF~1" in labs for labs in td["td"]["labels"])


def test_mute_a_tone(tmp_path):
    m = ViewerModel.from_file(VHF)
    ant = {p["path"]: p for g in m.info()["groups"] if g["block"] == "ant" for p in g["params"]}
    assert ant["tones.1.mute"]["kind"] == "bool" and ant["tones.1.mute"]["value"] is False
    r = m.simulate({"probe": "ant.out", "solver": "td", "min_dbm": -150,
                    "params": {"ant": {"tones.1.mute": True, "tones.2.mute": True}}})
    labels = {t["label"] for t in r["tone"]}
    assert "RF" in labels and not labels & {"B1", "B2"}
    d, _ = apply_edits(m.base, {"ant": {"tones.1.mute": False}}, None)
    assert "mute" not in d["blocks"]["ant"]["tones"][1]                       # off is the default: not written
    from hetsim.netlist import from_dict, to_dict
    d, _ = apply_edits(m.base, {"ant": {"tones.1.mute": True}}, None)
    c, st = from_dict(d)
    assert [t.name for t in c["ant"].tones] == ["RF", "B2", "IMG"]
    assert to_dict(c, st)["blocks"]["ant"]["tones"][1]["mute"] is True        # kept in the netlist


def test_sweep_steps_set_by_hand(tmp_path, monkeypatch):
    import hetsim.viewer.model as vm
    m = _band_model(tmp_path)
    req = {"probe": "mix.if", "solver": "tone", "min_dbm": -150, "params": {}}
    assert len(m.simulate({**req, "sweep_points": 7})["sweep"]["steps"]) == 7
    r = m.simulate({**req, "sweep_points": 1})
    assert len(r["sweep"]["steps"]) == 2                                     # at least 2
    monkeypatch.setattr(vm, "SWEEP_MAX_S", 1e-6)                             # everything over the cap
    r = m.simulate({**req, "sweep_points": 300})
    assert len(r["sweep"]["steps"]) == 2 and "capped" in r["sweep"]["note"]


def test_export_keeps_wideband_points_not_the_derived_step(tmp_path):
    m = _band_model(tmp_path, "wideband")
    d = m.netlist({"ant": {"tones.0.points": 9}}, 100e3)
    t = d["blocks"]["ant"]["tones"][0]
    assert t["points"] == 9 and "step" not in t and t["band"] == [88e6, 108e6]


def test_spacing_control(tmp_path):
    m = _band_model(tmp_path, "wideband")
    ant = {p["path"]: p for g in m.info()["groups"] if g["block"] == "ant" for p in g["params"]}
    assert ant["tones.0.spacing"]["choices"] == ["lin", "log"] and ant["tones.0.spacing"]["when"]
    d, snapped = apply_edits(m.base, {"ant": {"tones.0.spacing": "log", "tones.1.spacing": "log"}}, 100e3)
    t0, t1 = d["blocks"]["ant"]["tones"][:2]
    assert t0["spacing"] == "log" and "step" not in t0           # no linear grid step for a log comb
    assert "spacing" not in t1                                   # meaningless for a cw tone: dropped
    r = m.simulate({"probe": "ant.out", "solver": "td", "min_dbm": -150,
                    "params": {"ant": {"tones.0.spacing": "log", "tones.0.points": 5}}})
    assert r["converged"] and sum("RF~" in t["label"] for t in r["tone"]) == 5


def test_fundamental_products_are_flagged():
    from hetsim.netlist import from_dict
    from hetsim.spectrum import parse_label
    from hetsim.viewer.model import fundamental_test, lo_sources
    c, _ = from_dict(ViewerModel.from_file(VHF).base)
    assert lo_sources(c) == {"lo"}
    fund = fundamental_test(c)
    for label, want in [("LO - RF", True), ("IMG - LO", True), ("LO - B1", True), ("RF", False), ("LO", False),
                        ("2*LO - RF", False), ("B2 - B1", False), ("B2 + LO - 2*B1", False), ("LO + RF - B1", False)]:
        assert fund(parse_label(label)) is want, label
    r = ViewerModel.from_file(VHF).simulate({"probe": "ifamp.out", "solver": "both", "min_dbm": -150, "params": {}})
    tones = {t["label"]: t["fund"] for t in r["tone"]}
    assert tones["LO - RF"] and tones["IMG - LO"] and not tones["B2 + LO - 2*B1"]
    i = min(range(len(r["td"]["f"])), key=lambda k: abs(r["td"]["f"][k] - 10.3e6))
    assert r["td"]["labels"][i][0] == "LO - B1" and r["td"]["fund"][i]


def test_mixer_max_order_limits_spurs():
    m = ViewerModel.from_file(VHF)
    full = m.simulate({"probe": "mix.if", "solver": "tone", "min_dbm": -200, "params": {}})
    cut = m.simulate({"probe": "mix.if", "solver": "td", "min_dbm": -200, "params": {"mix": {"max_order": 2}}})
    spur = lambda t: "*LO" in t["label"] and (" + " in t["label"] or " - " in t["label"])  # noqa: E731  (not a leaking 3*LO)
    assert any(spur(t) for t in full["tone"])                                   # m x n spurs from the table
    assert not any(spur(t) for t in cut["tone"])                                # max_order 2: the 1 x 1 only
    assert len(cut["tone"]) < len(full["tone"]) and cut["converged"]
    mix = {p["path"]: p for g in m.info()["groups"] if g["block"] == "mix" for p in g["params"]}
    assert mix["max_order"]["kind"] == "int" and mix["max_order"]["optional"] and not mix["max_order"]["enabled"]
    from hetsim import Mixer
    assert set(Mixer(spur_table="dbm_generic", max_order=3).spur_entries()) <= {(1, 1), (1, 2), (2, 1), (0, 2), (0, 3), (3, 0)}
    with pytest.raises(ValueError, match="max_order"):
        Mixer(max_order=1)


def test_rbw_request_returns_skirts_and_switch(tmp_path):
    text = VHF.read_text().replace("lo:    {type: source, name: LO, freq: 110.7MHz, power: 7, harmonics_dbc: [-30, -40]}",
                                   "lo:    {type: source, tones: [{name: LO, freq: 110.7MHz, power: 7, "
                                   "phase_noise: [[10kHz, -90], [1MHz, -135], [10MHz, -145]]}]}")
    f = tmp_path / "rx.yaml"
    f.write_text(text)
    m = ViewerModel.from_file(f)
    base = {"probe": "ifamp.out", "solver": "tone", "min_dbm": -180, "params": {}}
    r = m.simulate({**base, "rbw": 10e3})
    assert r["has_pn"] and r["pn"]["LO"][0] == [10e3, -90]
    sk = {s["label"]: s for s in r["skirts"]}
    assert "LO - B1" in sk and sk["LO - B1"]["m"] == 1 and len(sk["LO - B1"]["off"]) == len(sk["LO - B1"]["lvl"])
    assert "skirts" not in m.simulate(base)                                         # no RBW: not computed
    off = m.simulate({**base, "rbw": 10e3, "phase_noise": False})
    assert off["skirts"] == [] and off["has_pn"]                                    # switched off for this update


def _lo_sweep_model(tmp_path):
    text = VHF.read_text().replace(
        "lo:    {type: source, name: LO, freq: 110.7MHz, power: 7, harmonics_dbc: [-30, -40]}",
        "lo:    {type: source, tones: [{name: LO, freq: 110.7MHz, power: 7, band: [98.7MHz, 118.7MHz], mode: sweep}]}")
    f = tmp_path / "rx.yaml"
    f.write_text(text)
    return ViewerModel.from_file(f)


def test_output_grid_steps_stay_put_when_the_rf_moves(tmp_path):
    m = _lo_sweep_model(tmp_path)
    grid = {"fmin": 5e6, "fmax": 15e6, "points": 11}
    for rf in (100e6, 101.3e6):
        r = m.simulate({"probe": "mix.if", "solver": "tone", "min_dbm": -160,
                        "params": {"ant": {"tones.0.freq": rf}}, "sweep_out": grid})
        assert r["sweep"]["anchor"] == "LO - RF"                           # default: strongest fundamental
        lo_rf = {p["label"]: p for p in r["sweep"]["products"]}["LO - RF"]
        assert lo_rf["f"] == pytest.approx([5e6 + 1e6 * k for k in range(11)], abs=0.5)
        assert [s["LO"] for s in r["sweep"]["steps"]] == pytest.approx([rf + 5e6 + 1e6 * k for k in range(11)])
    r = m.simulate({"probe": "mix.if", "solver": "tone", "min_dbm": -160, "params": {},
                    "sweep_out": {**grid, "anchor": "LO - B1"}})                # a chosen anchor
    lo_b1 = {p["label"]: p for p in r["sweep"]["products"]}["LO - B1"]
    assert lo_b1["f"] == pytest.approx([5e6 + 1e6 * k for k in range(11)], abs=0.5)
    r = m.simulate({"probe": "mix.if", "solver": "tone", "min_dbm": -160, "params": {},
                    "sweep_out": {"fmin": 30e6, "fmax": 40e6, "points": 5}})   # out of reach of the band
    assert r["sweep"]["steps"] == [] and "doesn't reach" in r["sweep"]["note"]


def test_phase_noise_parameter_round_trip(tmp_path):
    m = _lo_sweep_model(tmp_path)
    lo = {p["path"]: p for g in m.info()["groups"] if g["block"] == "lo" for p in g["params"]}
    pn = lo["tones.0.phase_noise"]
    assert pn["kind"] == "pn" and pn["optional"] and not pn["enabled"] and pn["value"][0] == [1e3, -80.0]
    prof = [[1e4, -90], [1e6, -130]]
    d, _ = apply_edits(m.base, {"lo": {"tones.0.phase_noise": prof}}, None)
    assert d["blocks"]["lo"]["tones"][0]["phase_noise"] == prof
    r = m.simulate({"probe": "mix.if", "solver": "tone", "min_dbm": -180, "rbw": 10e3,
                    "params": {"lo": {"tones.0.phase_noise": prof}}})
    assert r["has_pn"] and r["pn"]["LO"] == prof and r["skirts"]


def _xy_model(tmp_path):
    text = VHF.read_text()
    text = text.replace("lo:    {type: source, name: LO, freq: 110.7MHz, power: 7, harmonics_dbc: [-30, -40]}",
                        "lo:    {type: source, tones: [{name: LO, freq: 110.7MHz, power: 7, band: [108.7MHz, 112.7MHz], "
                        "mode: sweep, points: 81}]}")
    text = text.replace("  ifamp: {type: amplifier, gain_db: 20, oip3_dbm: 25}",
                        "  ifamp: {type: amplifier, gain_db: 20, oip3_dbm: 25}\n"
                        "  det1:  {type: detector, fmin: 10.6MHz, fmax: 10.8MHz}\n  xy1:   {type: xyplot}")
    text = text.replace("chain: [ant, pre, lna, mix, iff, ifamp]", "chain: [ant, pre, lna, mix, iff, ifamp, det1]")
    text = text.replace("  - [lo.out, mix.lo]", "  - [lo.out, mix.lo]\n  - [lo.sweep, xy1.x]\n  - [det1.out, xy1.y]")
    f = tmp_path / "xy.yaml"
    f.write_text(text)
    m = ViewerModel.from_file(f)
    m.sweep_workers = 1
    return m


def test_xy_plot_of_a_detector_over_the_sweep(tmp_path):
    m = _xy_model(tmp_path)
    info = m.info()
    assert not any(p.endswith((".sweep", ".x", ".y")) or p == "det1.out" for p in info["ports"])   # not probes
    assert {b["name"]: b["controls"] for b in info["schematic"]["blocks"]}["xy1"] == ["x", "y"]
    req = {"probe": "ifamp.out", "solver": "tone", "min_dbm": -160, "params": {}}
    r = m.simulate(req)
    assert r["detectors"]["det1"] == {"value": pytest.approx(-43.06, abs=0.05), "unit": "dBm",
                                      "dbm": pytest.approx(-43.06, abs=0.05)}  # the image: IMG - LO at 10.7 MHz
    xy = r["xy"][0]
    assert xy["note"] is None and len(xy["x"]) == len(xy["y"]) == 81 and xy["x"][0] == 108.7e6
    peak_x = xy["x"][max(range(81), key=lambda i: xy["y"][i] if xy["y"][i] is not None else -999)]
    assert 111.0e6 <= peak_x <= 111.6e6                                  # the -30 dBm blockers B1/B2 tuned in
    f = m.simulate({**req, "sweep_frame": True, "tone_freqs": {"LO": 111.1e6}})
    assert "sweep" not in f and "xy" not in f and f["detectors"]["det1"]["dbm"] == pytest.approx(-3.0, abs=0.1)
    lin = m.simulate({**req, "params": {"det1": {"scale": "linear"}}})
    assert lin["detectors"]["det1"]["unit"] == "V"
    assert lin["detectors"]["det1"]["value"] == pytest.approx(math.sqrt(100 * 10 ** ((-43.06 - 30) / 10)), rel=0.01)
    assert lin["xy"][0]["y_unit"] == "V" and min(lin["xy"][0]["y"]) == 0.0   # nothing in band: 0 V, not missing
    assert lin["xy"][0]["y_label"] == "det1 (V)" and "peak, linear" in lin["xy"][0]["title"]
    m.edit({"op": "disconnect", "a": "xy1.y"})
    assert "wire y" in m.simulate(req)["xy"][0]["note"]
    with pytest.raises(NetlistTextError, match="control port"):
        m.edit({"op": "connect", "a": "lo.sweep", "b": "mix.rf"})
    added = m.edit({"op": "add", "type": "detector", "x": 0, "y": 300})["added"]
    assert added == "det2" and m.base["blocks"]["det2"]["detect"] == "peak"


def test_xy_plot_spans_the_band_on_an_output_grid(tmp_path):
    m = _xy_model(tmp_path)
    r = m.simulate({"probe": "ifamp.out", "solver": "tone", "min_dbm": -160, "params": {}, "sweep_points": 41,
                    "sweep_out": {"fmin": 10.4e6, "fmax": 11.0e6, "points": 41}})
    lo = [s["LO"] for s in r["sweep"]["steps"]]
    assert min(lo) >= 110.0e6 and max(lo) <= 111.4e6                     # tracks: only the grid's slice
    xy = r["xy"][0]
    assert xy["x_tone"] == "LO" and len(xy["x"]) == 41
    assert xy["x"][0] == pytest.approx(108.7e6) and xy["x"][-1] == pytest.approx(112.7e6)   # X/Y: the whole band
    assert xy["x"][1] - xy["x"][0] == pytest.approx(0.1e6)                # evenly
