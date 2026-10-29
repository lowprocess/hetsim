"""Checks against hand-calculated values."""
import math
from pathlib import Path

import numpy as np
import pytest

from hetsim import (Amplifier, Attenuator, Circuit, Filter, IdealFilter, Load, Mixer, Source, SParamFilter,
                    Splitter, SpurTable, from_dict, to_dict)
from hetsim.analysis import find_responses, phase_noise_skirts
from hetsim.spectrum import Spectrum, parse_label, vec, vec_label
from hetsim.units import db_sum, parse_db, parse_freq

A = dict(abs=1e-6)


def mixer_rig(rf_dbm=-20.0, lo_dbm=7.0, **mixer_kw):
    c = Circuit("rig")
    c.add("rf", Source(freq=100e6, power=rf_dbm, name="RF"))
    c.add("lo", Source(freq=110e6, power=lo_dbm, name="LO"))
    c.add("mix", Mixer(**mixer_kw))
    c.connect("rf", "mix.rf")
    c.connect("lo", "mix.lo")
    return c


# ---------------------------------------------------------------- basics
def test_units():
    assert parse_freq("10.7MHz") == 10.7e6
    assert parse_freq("455k") == 455e3
    assert parse_freq("2.4 GHz") == 2.4e9
    assert parse_db("-30dBm") == -30
    assert parse_db(6) == 6


def test_labels_and_canonical_sign():
    s = Spectrum()
    s.add(vec(RF=1, LO=-1), 100e6 - 110e6, -10)      # negative -> flipped
    t = s.tones()[0]
    assert t.freq == pytest.approx(10e6)
    assert t.label == "LO - RF"
    assert parse_label("2*LO - 3*RF") == vec(LO=2, RF=-3)
    assert vec_label(vec(LO1=1, LO2=1, RF=-1)) == "LO1 + LO2 - RF"
    s.add(vec(LO=1, RF=-1), 10e6, -10)                # same product again: power sum
    assert s.power("LO - RF") == pytest.approx(-10 + 10 * math.log10(2), **A)


# ---------------------------------------------------------------- mixer
def test_ideal_mixer_is_a_multiplier():
    r = mixer_rig().simulate()
    out = r["mix.if"]
    assert {t.label for t in out} == {"LO - RF", "LO + RF"}
    assert out.get("LO - RF").freq == pytest.approx(10e6)
    assert out.get("LO + RF").freq == pytest.approx(210e6)
    assert out.power("LO - RF") == pytest.approx(-20, **A)
    assert out.power("LO + RF") == pytest.approx(-20, **A)
    assert len(r["mix.rf"]) == 0 and len(r["mix.lo"]) == 0


def test_conversion_loss_and_isolation():
    r = mixer_rig(conversion_loss_db=6, lo_if_isolation_db=30, lo_rf_isolation_db=35,
                  rf_if_isolation_db=25).simulate()
    out = r["mix.if"]
    assert out.power("LO - RF") == pytest.approx(-26, **A)
    assert out.power("LO") == pytest.approx(7 - 30, **A)
    assert out.power("RF") == pytest.approx(-20 - 25, **A)
    assert r["mix.rf"].power("LO") == pytest.approx(7 - 35, **A)
    assert r.incident("rf.out").power("LO") == pytest.approx(7 - 35, **A)   # reaches the source
    assert r["mix.lo"].power("RF") == pytest.approx(-20 - 35, **A)           # rf_lo defaults to lo_rf


def test_spur_table_levels_and_scaling():
    t = SpurTable({(3, 1): 15, (2, 2): 65, (0, 2): 60}, ref_rf_dbm=-10)
    for p_rf in (-10.0, -20.0):
        out = mixer_rig(rf_dbm=p_rf, conversion_loss_db=6, spur_table=t).simulate()["mix.if"]
        assert out.power("3*LO - RF") == pytest.approx(p_rf - 6 - 15, **A)
        # n=2 products move 2 dB per dB of RF drive
        assert out.power("2*LO - 2*RF") == pytest.approx(p_rf - 6 - 65 + (p_rf + 10), **A)
        assert out.power("2*RF") == pytest.approx(p_rf - 6 - 60 + (p_rf + 10), **A)


def test_spur_csv_roundtrip():
    from hetsim.blocks.spurtable import PRESETS
    t = PRESETS["dbm_generic"]()
    t2 = SpurTable.from_csv(t.to_csv())
    assert t2.entries == t.entries and t2.ref_rf_dbm == t.ref_rf_dbm


def test_mixer_iip3():
    c = Circuit()
    c.add("rf", Source(tones=[{"name": "A", "freq": 100e6, "power": -20},
                              {"name": "B", "freq": 100.1e6, "power": -20}]))
    c.add("lo", Source(freq=110e6, power=7, name="LO"))
    c.add("mix", Mixer(conversion_loss_db=6, iip3_dbm=10))
    c.connect("rf", "mix.rf")
    c.connect("lo", "mix.lo")
    out = c.simulate()["mix.if"]
    # input IM3 = 3*(-20) - 2*10 = -80 dBm, then -6 dB conversion
    assert out.power("B + LO - 2*A") == pytest.approx(-86, **A)
    assert out.get("B + LO - 2*A").freq == pytest.approx(110e6 - (2 * 100e6 - 100.1e6))


def test_mixer_reciprocal_upconversion():
    c = Circuit()
    c.add("ifs", Source(freq=10e6, power=-10, name="IF"))
    c.add("lo", Source(freq=110e6, power=7, name="LO"))
    c.add("mix", Mixer(conversion_loss_db=6))
    c.connect("ifs", "mix.if")
    c.connect("lo", "mix.lo")
    rf = c.simulate()["mix.rf"]
    assert rf.power("LO - IF") == pytest.approx(-16, **A)
    assert rf.power("IF + LO") == pytest.approx(-16, **A)


def test_lo_drive_warning():
    r = mixer_rig(lo_dbm=0, lo_drive_dbm=7).simulate()
    assert "LO drive" in r.warning_text()


# ---------------------------------------------------------------- amplifier
def two_tone_amp(p_in=-20.0, **kw):
    c = Circuit()
    c.add("s", Source(tones=[{"name": "A", "freq": 100e6, "power": p_in},
                             {"name": "B", "freq": 101e6, "power": p_in}]))
    c.add("amp", Amplifier(**kw))
    c.chain("s", "amp")
    return c.simulate()["amp.out"]


def test_amplifier_im3_and_harmonics():
    out = two_tone_amp(gain_db=10, oip3_dbm=20)
    assert out.power("A") == pytest.approx(-10, **A)
    assert out.power("2*A - B") == pytest.approx(3 * -10 - 2 * 20, **A)     # -70
    assert out.power("2*B - A") == pytest.approx(-70, **A)
    assert out.power("2*A + B") == pytest.approx(-70, **A)
    assert out.power("3*A") == pytest.approx(-70 - 20 * math.log10(3), **A)
    # iip3 is the same thing referred to the input
    out2 = two_tone_amp(gain_db=10, iip3_dbm=10)
    assert out2.power("2*A - B") == pytest.approx(-70, **A)


def test_amplifier_im3_slope_is_3():
    a = two_tone_amp(p_in=-30, gain_db=10, oip3_dbm=20).power("2*A - B")
    b = two_tone_amp(p_in=-20, gain_db=10, oip3_dbm=20).power("2*A - B")
    assert b - a == pytest.approx(30, **A)


def test_amplifier_ip2():
    out = two_tone_amp(gain_db=10, oip2_dbm=40)
    assert out.power("A + B") == pytest.approx(-10 - 10 - 40, **A)
    assert out.power("B - A") == pytest.approx(-60, **A)
    assert out.power("2*A") == pytest.approx(-60 - 20 * math.log10(2), **A)


def test_amplifier_p1db():
    c = Circuit()
    c.add("s", Source(freq=1e6, power=1.0, name="S"))   # linear output would be 11 dBm
    c.add("amp", Amplifier(gain_db=10, op1db_dbm=10))
    c.chain("s", "amp")
    assert c.simulate()["amp.out"].power("S") == pytest.approx(10, abs=1e-6)


def test_amplifier_reverse_isolation():
    c = Circuit()
    c.add("s", Load())
    c.add("amp", Amplifier(gain_db=20, reverse_isolation_db=30))
    c.add("back", Source(freq=1e6, power=0, name="X"))
    c.connect("amp.in", "s.in")
    c.connect("back", "amp.out")
    assert c.simulate().incident("s.in").power("X") == pytest.approx(-30, **A)


# ---------------------------------------------------------------- passives
def through(block, f):
    c = Circuit()
    c.add("s", Source(freq=f, power=0, name="S"))
    c.add("dut", block)
    c.chain("s", "dut")
    return c.simulate()["dut.out"].power("S")


def test_butterworth_lowpass():
    assert through(Filter("lowpass", "butter", 3, fc=10e6), 10e6) == pytest.approx(-3.0103, abs=1e-3)
    assert through(Filter("lowpass", "butter", 3, fc=10e6), 20e6) == pytest.approx(-10 * math.log10(65), abs=1e-3)


def test_butterworth_bandpass_edges_and_loss():
    f = Filter("bandpass", "butter", 3, f1=88e6, f2=108e6, insertion_loss_db=1.5)
    assert through(f, 88e6) == pytest.approx(-3.0103 - 1.5, abs=1e-3)
    assert through(f, 108e6) == pytest.approx(-3.0103 - 1.5, abs=1e-3)
    assert through(f, math.sqrt(88e6 * 108e6)) == pytest.approx(-1.5, abs=1e-6)


def test_chebyshev_ripple_and_ultimate_rejection():
    f = Filter("lowpass", "cheby1", 5, fc=1e6, ripple_db=0.5, max_rejection_db=60)
    assert through(f, 1e6) == pytest.approx(-0.5, abs=1e-3)
    assert through(f, 100e6) == pytest.approx(-60, abs=1e-6)


def test_high_order_ghz_filter_is_finite():
    f = Filter("bandpass", "cheby1", 9, f0=5.8e9, bw=100e6)
    assert math.isfinite(through(f, 5.8e9)) and through(f, 1e9) < -100


def test_ideal_filter_and_attenuator_and_splitter():
    assert through(IdealFilter(fmin=1e6, fmax=2e6, insertion_loss_db=1), 1.5e6) == pytest.approx(-1)
    assert through(IdealFilter(fmin=1e6, fmax=2e6, rejection_db=80), 3e6) == pytest.approx(-80)
    assert through(Attenuator(6), 1e6) == pytest.approx(-6)
    c = Circuit()
    c.add("s", Source(freq=1e6, power=0, name="S"))
    c.add("sp", Splitter(isolation_db=20))
    c.chain("s", "sp")
    r = c.simulate()
    assert r["sp.a"].power("S") == pytest.approx(-3.0103, abs=1e-4)
    assert r["sp.b"].power("S") == pytest.approx(-3.0103, abs=1e-4)


def test_touchstone(tmp_path):
    p = tmp_path / "f.s2p"
    p.write_text("! test\n# MHz S DB R 50\n"
                 "100 -20 0 -1 0 -1 0 -20 0\n"
                 "200 -20 0 -11 0 -11 0 -20 0\n")
    f = SParamFilter(p)
    assert through(f, 150e6) == pytest.approx(-6)


# ---------------------------------------------------------------- system level
def superhet():
    c = Circuit("sh")
    c.add("ant", Source(freq=100e6, power=-60, name="RF"))
    c.add("pre", Filter("bandpass", "butter", 3, f1=88e6, f2=108e6))
    c.add("lna", Amplifier(gain_db=15, oip3_dbm=30, reverse_isolation_db=30))
    c.add("lo", Source(freq=110.7e6, power=7, name="LO"))
    c.add("mix", Mixer(conversion_loss_db=6, lo_rf_isolation_db=35, lo_if_isolation_db=30,
                       rf_if_isolation_db=25, spur_table="dbm_generic"))
    c.add("iff", Filter("bandpass", "cheby1", 4, f0=10.7e6, bw=200e3, max_rejection_db=90))
    c.chain("ant", "pre", "lna", "mix", "iff")
    c.connect("lo", "mix.lo")
    return c


def test_lo_leaks_back_out_of_antenna():
    r = superhet().simulate()
    pre_loss = -through(Filter("bandpass", "butter", 3, f1=88e6, f2=108e6), 110.7e6)
    assert r.converged
    assert r.incident("ant.out").power("LO") == pytest.approx(7 - 35 - 30 - pre_loss, abs=1e-6)


def test_find_responses_image_and_spurs():
    tab = find_responses(superhet(), "ant", "iff.out", (10.6e6, 10.8e6), 1e6, 500e6)
    by_label = {r.label: r for r in tab.responses}
    wanted = by_label["LO - RF"]
    image = by_label["RF - LO"]
    assert wanted.freq == pytest.approx(100e6) and image.freq == pytest.approx(121.4e6)
    assert wanted.rejection_db == 0
    # image rejection = preselector attenuation at 121.4 MHz (both through the same chain otherwise)
    pre = Filter("bandpass", "butter", 3, f1=88e6, f2=108e6)
    assert -image.rejection_db == pytest.approx(-through(pre, 121.4e6) + through(pre, 100e6), abs=1e-3)
    assert "3*LO - RF" in by_label and by_label["3*LO - RF"].freq == pytest.approx(3 * 110.7e6 - 10.7e6)
    assert "RF" in by_label  # IF feedthrough at 10.7 MHz


def test_netlist_roundtrip():
    c = superhet()
    d = to_dict(c)
    c2, _ = from_dict(d)
    a, b = c.simulate()["iff.out"], c2.simulate()["iff.out"]
    assert [(t.label, round(t.power_dbm, 9)) for t in a] == [(t.label, round(t.power_dbm, 9)) for t in b]


def test_netlist_units_and_chain():
    d = {"blocks": {"s": {"type": "source", "freq": "1MHz", "power": "-10dBm", "name": "S"},
                    "a": {"type": "attenuator", "loss_db": "3 dB"}},
         "chain": ["s", "a"]}
    c, _ = from_dict(d)
    assert c.simulate()["a.out"].power("S") == pytest.approx(-13)


def test_wiring_errors():
    c = superhet()
    with pytest.raises(ValueError):
        c.connect("ant", "lna")          # ant.out already wired
    with pytest.raises(KeyError):
        c.connect("mix.xx", "lna.in")


def test_replace_keeps_wiring():
    c = superhet()
    c.replace("pre", Attenuator(0))
    r = c.simulate()
    assert r["iff.out"].power("LO - RF") == pytest.approx(-60 + 15 - 6 + through(
        Filter("bandpass", "cheby1", 4, f0=10.7e6, bw=200e3), 10.7e6), abs=1e-6)


# ---------------------------------------------------------------- source bands
def _vhf_dict():
    from hetsim.viewer.model import parse_text
    return parse_text((Path(__file__).parent.parent / "examples" / "vhf_single_conversion.yaml").read_text())


def test_tone_band_modes_parse_and_round_trip():
    from hetsim.blocks.sources import ToneSpec
    t = ToneSpec.from_dict({"name": "RF", "band": "88MHz:108MHz", "power": -60})
    assert t.mode == "tune" and t.band == (88e6, 108e6) and t.freq == 98e6      # centre by default
    assert ToneSpec.from_dict({"band": [1e6, 2e6], "mode": "wideband", "step": "250kHz"}).grid() == \
        [1e6, 1.25e6, 1.5e6, 1.75e6, 2e6]
    for bad in ({"freq": 1e6, "band": [1e6, 2e6], "mode": "cw"}, {"freq": 3e6, "band": [1e6, 2e6]},
                {"freq": 1e6, "mode": "sweep"}, {"band": [2e6, 1e6]}, {"band": [1e6, 2e6], "mode": "chirp"},
                {"band": [1e6, 2e6], "mode": "sweep", "points": 1}):
        with pytest.raises(ValueError):
            ToneSpec.from_dict(bad)
    d = _vhf_dict()
    d["blocks"]["ant"]["tones"][0].update(band=["88MHz", "108MHz"], mode="sweep", points=11)
    d["blocks"]["pre"]["limits"] = {"f1": ["80MHz", "95MHz"]}
    c, st = from_dict(d)
    back = to_dict(c, st)
    assert back["blocks"]["ant"]["tones"][0] == {"name": "RF", "freq": 100e6, "power": -80.0,
                                                 "band": [88e6, 108e6], "mode": "sweep", "points": 11}
    assert back["blocks"]["pre"]["limits"] == {"f1": ["80MHz", "95MHz"]}
    with pytest.raises(ValueError, match="limits"):
        d["blocks"]["pre"]["limits"] = {"f1": "80MHz"}
        from_dict(d)


def test_wideband_comb_shares_power_in_both_solvers():
    d = _vhf_dict()
    d["blocks"]["ant"]["tones"] = [{"name": "FM", "band": ["99MHz", "101MHz"], "mode": "wideband",
                                    "points": 5, "power": -50}]
    c, st = from_dict(d)
    tones = c["ant"].tones
    assert [t.name for t in tones] == [f"FM~{k}" for k in range(1, 6)]
    assert [t.freq for t in tones] == [99e6, 99.5e6, 100e6, 100.5e6, 101e6]
    assert db_sum(*[t.power for t in tones]) == pytest.approx(-50)
    r = c.simulate(st)
    assert {t.label for t in r["mix.if"]} >= {"LO - FM~3", "LO - FM~1"}
    td = c.simulate_td(st)
    assert td.spectrum("ant.out").power_at(99.5e6) == pytest.approx(-50 - 10 * np.log10(5), abs=0.01)


def test_band_sweep_tracks_lo():
    d = _vhf_dict()
    d["blocks"]["ant"]["tones"][0].update(band=["88MHz", "108MHz"], mode="sweep", points=21)
    c, st = from_dict(d)
    only_rf = {s.label: s for s in c.simulate_sweep(st).products("mix.if")}
    assert only_rf["LO - RF"].fmin == pytest.approx(2.7e6) and only_rf["LO - RF"].fmax == pytest.approx(22.7e6)
    d["blocks"]["lo"] = {"type": "source", "harmonics_dbc": [-30, -40],
                         "tones": [{"name": "LO", "band": ["98.7MHz", "118.7MHz"], "mode": "sweep", "power": 7}]}
    c, st = from_dict(d)
    sw = c.simulate_sweep(st)
    lo_rf = {s.label: s for s in sw.products("ifamp.out")}["LO - RF"]
    assert len(sw.steps) == 41 and lo_rf.fmin == pytest.approx(10.7e6) and lo_rf.fmax == pytest.approx(10.7e6)
    assert sw.steps[0] == {"RF": 88e6, "LO": 98.7e6}
    assert c["ant"].specs[0].freq == 100e6                      # nominal restored after the sweep
    assert "worst at" in sw.table("ifamp.out", limit=3)
    with pytest.raises(ValueError, match="sweep"):
        from_dict(_vhf_dict())[0].simulate_sweep()


def test_log_spacing():
    from hetsim.blocks.sources import ToneSpec
    g = ToneSpec.from_dict({"band": [10e6, 1e9], "mode": "sweep", "spacing": "log", "points": 3}).grid()
    assert g == pytest.approx([10e6, 100e6, 1e9])
    with pytest.raises(ValueError, match="points, not step"):
        ToneSpec.from_dict({"band": [1e6, 2e6], "mode": "wideband", "spacing": "log", "step": 1e5})
    with pytest.raises(ValueError, match="spacing"):
        ToneSpec.from_dict({"band": [1e6, 2e6], "mode": "sweep", "spacing": "octave"})
    comb = Source(tones=[{"name": "W", "band": [1e6, 16e6], "mode": "wideband", "spacing": "log", "points": 5}])
    assert [t.freq for t in comb.tones] == pytest.approx([1e6, 2e6, 4e6, 8e6, 16e6])
    assert comb.to_params()["tones"][0]["spacing"] == "log"

    # log RF sweep with a linear LO band = RF band + IF: the LO still tracks, steps follow the RF's log grid
    d = _vhf_dict()
    d["blocks"]["ant"]["tones"] = [{"name": "RF", "band": ["50MHz", "200MHz"], "mode": "sweep", "spacing": "log",
                                    "points": 9, "power": -60}]
    d["blocks"]["pre"].update(f1="40MHz", f2="250MHz")
    d["blocks"]["lo"] = {"type": "source",
                         "tones": [{"name": "LO", "band": ["60.7MHz", "210.7MHz"], "mode": "sweep", "points": 9, "power": 7}]}
    c, st = from_dict(d)
    sw = c.simulate_sweep(st)
    rf = [s["RF"] for s in sw.steps]
    assert rf == pytest.approx(list(np.geomspace(50e6, 200e6, 9)))
    assert [s["LO"] - s["RF"] for s in sw.steps] == pytest.approx([10.7e6] * 9)
    lo_rf = {p.label: p for p in sw.products("mix.if")}["LO - RF"]
    assert lo_rf.fmin == pytest.approx(10.7e6) and lo_rf.fmax == pytest.approx(10.7e6)


def test_parallel_band_sweep_matches_serial(monkeypatch):
    from hetsim import analysis
    d = _vhf_dict()
    d["blocks"]["ant"]["tones"][0].update(band=["88MHz", "108MHz"], mode="sweep", points=9)
    c, st = from_dict(d)
    serial = c.simulate_sweep(st, workers=1)
    monkeypatch.setattr(analysis, "PARALLEL_MIN_S", 0.0)       # farm out even this small sweep
    monkeypatch.setattr(analysis, "POOL_SIZE", 3)
    monkeypatch.setattr(analysis, "_POOL_FAILED", False)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # a fallback to in-process must fail this test, not hide
        par = c.simulate_sweep(st, workers=3, ports=["mix.if"])
    assert par.steps == serial.steps
    a = {p.label: p for p in serial.products("mix.if")}
    b = {p.label: p for p in par.products("mix.if")}
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_allclose(b[k].freqs, a[k].freqs, rtol=1e-12)    # NaN where absent, in both
        np.testing.assert_allclose(b[k].powers, a[k].powers, rtol=1e-9)
    assert c["ant"].specs[0].freq == 100e6


def test_band_sweep_falls_back_when_the_pool_fails(monkeypatch):
    from hetsim import analysis
    d = _vhf_dict()
    d["blocks"]["ant"]["tones"][0].update(band=["88MHz", "108MHz"], mode="sweep", points=5)
    c, st = from_dict(d)
    monkeypatch.setattr(analysis, "PARALLEL_MIN_S", 0.0)
    monkeypatch.setattr(analysis, "_POOL_FAILED", False)

    def broken():
        raise OSError("no processes here")
    monkeypatch.setattr(analysis, "_pool", broken)
    with pytest.warns(RuntimeWarning, match="in-process"):
        sw = c.simulate_sweep(st, workers=4)
    assert len(sw.results) == 5 and analysis._POOL_FAILED



# ---------------------------------------------------------------- phase noise
PN = [["10kHz", -90], ["100kHz", -115], ["1MHz", -135], ["10MHz", -145]]


def _pn_circuit(only=("B1",)):
    d = _vhf_dict()
    for t in d["blocks"]["ant"]["tones"]:
        if t["name"] not in only:
            t["mute"] = True
    d["blocks"]["lo"] = {"type": "source",
                         "tones": [{"name": "LO", "freq": "110.7MHz", "power": 7, "phase_noise": PN}]}
    return from_dict(d)


def test_phase_noise_profile():
    from hetsim.blocks.sources import ToneSpec
    t = ToneSpec.from_dict({"freq": 1e9, "phase_noise": PN})
    assert t.pn_dbc(100e3) == -115 and t.pn_dbc(-100e3) == -115
    assert t.pn_dbc(math.sqrt(10e3 * 100e3)) == pytest.approx(-102.5)   # log-interpolated
    assert t.pn_dbc(5e3) is None and t.pn_dbc(20e6) is None
    for bad in ([["1MHz", -100], ["100kHz", -90]], [[0, -90]], ["junk"]):
        with pytest.raises(ValueError):
            ToneSpec.from_dict({"freq": 1e9, "phase_noise": bad})
    with pytest.raises(ValueError, match="comb"):
        ToneSpec.from_dict({"band": [1e6, 2e6], "mode": "wideband", "phase_noise": PN})
    c, st = _pn_circuit()
    assert to_dict(c, st)["blocks"]["lo"]["tones"][0]["phase_noise"][1] == [100e3, -115]


def test_td_phase_noise_sidebands_follow_the_profile():
    c, st = _pn_circuit()
    sp = c.simulate_td(st, resolution=100e3).spectrum("lo.out")
    for off, l in ((100e3, -115), (1e6, -135), (10e6, -145)):
        for f in (110.7e6 + off, 110.7e6 - off):
            assert sp.power_at(f) == pytest.approx(7 + l + 50, abs=0.2)     # 10 log10(100 kHz) = 50
    assert sp.power_at(110.7e6 + 20e6) < -150                               # past the profile: none
    assert sp.power_at(110.7e6) == pytest.approx(7, abs=0.01)


def test_reciprocal_mixing_tone_skirts_match_td():
    """B1's converted LO noise lands on the IF through the IF filter: both solvers agree."""
    c, st = _pn_circuit()
    s = {k.label: k for k in phase_noise_skirts(c, "ifamp.out", st, min_dbm_hz=-200)}["LO - B1"]
    td = c.simulate_td(st, resolution=100e3).spectrum("ifamp.out")
    for f in (10.6e6, 10.7e6, 10.8e6):
        assert float(s.density([f - s.f0])[0]) + 50 == pytest.approx(td.power_at(f), abs=1.0)
    assert np.isnan(s.density([1e3])[0])                                    # inside the first offset: none


def test_output_grid_fractions():
    from hetsim.analysis import output_grid_fractions
    d = _vhf_dict()
    d["blocks"]["lo"] = {"type": "source",
                         "tones": [{"name": "LO", "freq": "110.7MHz", "band": ["100MHz", "120MHz"], "mode": "sweep",
                                    "power": 7}]}
    c, st = from_dict(d)
    frac = output_grid_fractions(c, "LO - RF", [1e6, 5e6, 10e6, 30e6])     # RF at 100 MHz: LO 101, 105, 110
    assert frac == pytest.approx([0.05, 0.25, 0.5])                           # 30 MHz is out of the band
    sw = c.simulate_sweep(st, fractions=frac)
    assert [s["LO"] for s in sw.steps] == pytest.approx([101e6, 105e6, 110e6])
    with pytest.raises(ValueError, match="doesn't move"):
        output_grid_fractions(c, "B2 - B1", [1e6])
    with pytest.raises(ValueError, match="no tone"):
        output_grid_fractions(c, "LO - XYZ", [1e6])


# ---------------------------------------------------------------- detector, X/Y plot, control ports
def test_detector_reads_peak_and_total():
    from hetsim.blocks import Detector
    s = Spectrum()
    s.add(vec(A=1), 10e6, -20)
    s.add(vec(B=1), 10.1e6, -20)
    s.add(vec(C=1), 50e6, 0)
    assert Detector(fmin="9MHz", fmax="11MHz").read(s) == pytest.approx(-20)
    assert Detector(fmin="9MHz", fmax="11MHz", detect="total").read(s) == pytest.approx(-20 + 10 * math.log10(2))
    assert Detector().read(s) == pytest.approx(0)                          # no band: everything
    assert Detector(fmin="1GHz").read(s) == -math.inf
    for bad in ({"detect": "rms"}, {"fmin": "2MHz", "fmax": "1MHz"}):
        with pytest.raises(ValueError):
            Detector(**bad)


def test_control_ports_carry_no_rf():
    from hetsim.blocks import Detector, XYPlot
    c = Circuit("t")
    c.add("src", Source(freq="10MHz", power=-10, name="S"))
    c.add("det", Detector())
    c.add("xy", XYPlot())
    c.connect("src", "det")
    c.connect("src.sweep", "xy.x")
    c.connect("det.out", "xy.y")
    c.add("xy2", XYPlot())
    c.add("ld", Load())
    with pytest.raises(ValueError, match="control port"):
        c.connect("xy2.y", "ld.in")
    r = c.simulate()
    assert c["det"].read(r.incident("det.in")) == pytest.approx(-10)
    assert len(r["src.sweep"]) == 0 and len(r["det.out"]) == 0
    td = c.simulate_td(resolution=1e6)
    assert td.spectrum("det.in").peaks(-100) == [] and td.incident_spectrum("det.in").power_at(10e6) == pytest.approx(-10, abs=0.01)
