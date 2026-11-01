"""VSWR in the solvers, and the noise-figure and port-match analyses, against hand calculations."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from hetsim import (Amplifier, Attenuator, Circuit, Filter, IdealFilter, Load, Mixer, Source, Splitter, from_dict,
                    match_sweep, mismatch_report, noise_figure, noise_sweep, port_match, to_dict)
from hetsim.cli import main
from hetsim.smallsignal import KT0_DBM_HZ

EXAMPLE = Path(__file__).parent.parent / "examples" / "vhf_noise_match.yaml"


def gamma(s):
    return (s - 1) / (s + 1)


def chain(*blocks):
    c = Circuit()
    c.add("ant", Source(freq="100MHz", power=-60, name="RF"))
    names = ["ant"]
    for i, b in enumerate(blocks):
        names.append(f"b{i}")
        c.add(names[-1], b)
    c.chain(*names)
    return c


def receiver(*front, nf=None, cl=6.0, **mix):
    c = Circuit()
    c.add("ant", Source(freq="100MHz", power=-60, name="RF"))
    c.add("lo", Source(freq="110.7MHz", power=7, name="LO"))
    c.add("mix", Mixer(conversion_loss_db=cl, nf_db=nf, **mix))
    c.add("iff", Filter("bandpass", "butter", 4, f0="10.7MHz", bw="200kHz"))
    names = ["ant"]
    for i, b in enumerate(front):
        names.append(f"f{i}")
        c.add(names[-1], b)
    c.chain(*names, "mix", "iff")
    c.connect("lo", "mix.lo")
    return c


# ---------------------------------------------------------------- tone solver
def test_load_reflection_comes_back_at_the_source():
    c = chain(Load(vswr=3))
    assert c.simulate().incident("ant.out").power("RF") == pytest.approx(-60 + 20 * math.log10(gamma(3)), abs=0.01)


def test_no_vswr_reflects_nothing():
    assert len(chain(Load()).simulate().incident("ant.out")) == 0
    assert len(chain(Load(vswr=1)).simulate().incident("ant.out")) == 0


def test_bounces_between_two_ports_add_in_power():
    c = Circuit()
    c.add("ant", Source(freq="100MHz", power=-60, name="RF", vswr=3))
    c.add("l", Load(vswr=3))
    c.chain("ant", "l")
    k = gamma(3) ** 4
    assert c.simulate().incident("l.in").power("RF") == pytest.approx(-60 - 10 * math.log10(1 - k), abs=0.01)


def test_reflection_passes_back_through_blocks():
    c = chain(Attenuator(6), Load(vswr=2))
    want = -60 - 12 + 20 * math.log10(gamma(2))
    assert c.simulate().incident("ant.out").power("RF") == pytest.approx(want, abs=0.01)


def test_table_interpolates_and_holds_its_edges_and_ports_are_separate():
    b = Attenuator(1, vswr={"in": [["1GHz", 1.0], ["2GHz", 3.0]]})
    assert b.reflection("in", 1.5e9) == pytest.approx(0.5 * gamma(3) ** 2)
    assert b.reflection("in", 5e9) == pytest.approx(gamma(3) ** 2)
    assert b.reflection("out", 1.5e9) == 0


@pytest.mark.parametrize("bad", [0.5, [[1e9, 0.9]], {"nope": 2}, [1, 2]])
def test_bad_vswr_is_refused(bad):
    with pytest.raises((ValueError, TypeError)):
        Attenuator(1, vswr=bad)


def test_control_ports_take_no_vswr():
    assert set(Source(freq=1e6, power=0, vswr=2).gamma2) == {"out"}
    with pytest.raises(ValueError):
        Source(freq=1e6, power=0, vswr={"sweep": 2})


def test_netlist_round_trip():
    d = {"blocks": {"s": {"type": "source", "freq": "1MHz", "power": 0, "vswr": 1.5},
                    "a": {"type": "amplifier", "gain_db": 10, "nf_db": 2, "vswr": {"in": 2, "out": [[1e6, 1.2]]}},
                    "l": {"type": "load", "vswr": 3}},
         "chain": ["s", "a", "l"]}
    c, st = from_dict(d)
    back = to_dict(c, st)
    assert back["blocks"]["s"]["vswr"] == 1.5
    assert back["blocks"]["a"]["vswr"] == d["blocks"]["a"]["vswr"] and back["blocks"]["a"]["nf_db"] == 2
    assert back["blocks"]["l"]["vswr"] == 3
    c2, _ = from_dict(back)
    assert c2["l"].reflection("in", 1e6) == c["l"].reflection("in", 1e6)


# ---------------------------------------------------------------- time domain
def test_td_reflections_add_in_phase():
    c = Circuit()
    c.add("ant", Source(freq="1MHz", power=-30, name="RF", vswr=2))
    c.add("l", Load(vswr=2))
    c.chain("ant", "l")
    r = c.simulate_td(resolution="1MHz", fmax="5MHz")
    want = -30 - 20 * math.log10(1 - gamma(2) ** 2)    # amplitude series, not power
    assert r.incident_spectrum("l.in").peaks(-100)[0][1] == pytest.approx(want, abs=0.01)


def test_td_without_vswr_unchanged():
    r = chain(Attenuator(3), Load()).simulate_td(resolution="100MHz", fmax="500MHz")
    assert r.incident_spectrum("b1.in").peaks(-100)[0][1] == pytest.approx(-63, abs=0.01)


# ---------------------------------------------------------------- noise: a passive circuit is a load at T0
@pytest.mark.parametrize("blocks", [
    [Attenuator(10)],
    [Attenuator(3, vswr=2), Attenuator(6, vswr={"in": 1.5, "out": 3})],
    [Filter("lowpass", "butter", 3, fc="50MHz", insertion_loss_db=1, vswr=1.5)],
    [IdealFilter(fmin="1GHz", rejection_db=60), Attenuator(1)],
])
def test_passive_chain_delivers_exactly_kt0(blocks):
    c = chain(*blocks)
    last = f"b{len(blocks) - 1}.out"
    assert noise_figure(c, "ant", last).noise_out_dbm == pytest.approx(KT0_DBM_HZ, abs=1e-6)


def test_passive_with_mismatched_source_and_splitter():
    c = Circuit()
    c.add("ant", Source(freq="100MHz", power=-60, name="RF", vswr=2))
    c.add("s", Splitter(excess_loss_db=0.5, vswr=1.4))
    c.add("p", Attenuator(2, vswr=2))
    c.connect("ant.out", "s.sum")
    c.connect("s.a", "p.in")
    assert noise_figure(c, "ant", "p.out").noise_out_dbm == pytest.approx(KT0_DBM_HZ, abs=1e-6)


def test_pad_nf_is_its_loss():
    assert noise_figure(chain(Attenuator(6)), "ant", "b0.out").nf_db == pytest.approx(6, abs=1e-9)


def test_source_mismatch_loss_counts():
    c = Circuit()
    c.add("ant", Source(freq="100MHz", power=-60, name="RF", vswr=3))
    c.add("a", Attenuator(6))
    c.chain("ant", "a")
    # total stays kT0; the signal share of the source's noise falls by 1 - gamma^2
    want = 6 - 10 * math.log10(1 - gamma(3) ** 2)
    assert noise_figure(c, "ant", "a.out").nf_db == pytest.approx(want, abs=1e-6)


# ---------------------------------------------------------------- cascades
def test_friis():
    c = chain(Amplifier(gain_db=15, nf_db=2), Attenuator(6), Amplifier(gain_db=20, nf_db=8))
    f = 10 ** 0.2 + (10 ** 0.6 - 1) / 10 ** 1.5 + (10 ** 0.8 - 1) / 10 ** 0.9
    r = noise_figure(c, "ant", "b2.out")
    assert r.nf_db == pytest.approx(10 * math.log10(f), abs=1e-6)
    assert r.gain_db == pytest.approx(29, abs=1e-9)
    assert r.noise_out_dbm == pytest.approx(KT0_DBM_HZ + 29 + 10 * math.log10(f), abs=1e-6)


def test_amplifier_without_nf_is_noiseless_and_says_so():
    r = noise_figure(chain(Amplifier(gain_db=20)), "ant", "b0.out")
    assert r.nf_db == pytest.approx(0, abs=1e-9)
    assert any("b0" in w and "noiseless" in w for w in r.warnings)


def test_compression_reduces_noise_gain():
    c = chain(Amplifier(gain_db=20, nf_db=3, op1db_dbm=-20))
    c["ant"].tones[0].power = -30   # 10 dBm linear output against -20 dBm P1dB
    r = noise_figure(c, "ant", "b0.out")
    assert r.gain_db < 15
    assert r.nf_db == pytest.approx(3, abs=1e-6)
    assert r.gain_db == pytest.approx(c.simulate()["b0.out"].power("RF") + 30, abs=0.01)


def test_blocker_compression_degrades_a_later_stage():
    def run(blocker):
        c = Circuit()
        c.add("ant", Source(tones=[{"name": "RF", "freq": "100MHz", "power": -80},
                                   {"name": "B", "freq": "101MHz", "power": blocker}]))
        c.add("a", Amplifier(gain_db=15, nf_db=1, op1db_dbm=-5))
        c.add("b", Amplifier(gain_db=10, nf_db=10))
        c.chain("ant", "a", "b")
        return noise_figure(c, "ant", "b.out", tone="RF").nf_db
    assert run(-5) > run(-60) + 1


# ---------------------------------------------------------------- mixers
def test_passive_mixer_nf_is_its_conversion_loss():
    r = noise_figure(receiver(), "ant", "iff.out")
    assert r.product == "LO - RF" and r.f_out == pytest.approx(10.7e6)
    assert r.nf_db == pytest.approx(6, abs=1e-6)
    assert r.noise_out_dbm == pytest.approx(KT0_DBM_HZ, abs=1e-6)


def test_mixer_nf_above_its_loss():
    assert noise_figure(receiver(nf=8.5), "ant", "iff.out").nf_db == pytest.approx(8.5, abs=1e-6)


def test_image_noise_costs_3_db_behind_an_lna():
    lna = dict(gain_db=30, nf_db=1)
    without = noise_figure(receiver(Amplifier(**lna)), "ant", "iff.out").nf_db
    filt = Filter("bandpass", "butter", 5, f1="88MHz", f2="108MHz")
    with_it = noise_figure(receiver(Amplifier(**lna), filt), "ant", "iff.out").nf_db
    assert without - with_it == pytest.approx(10 * math.log10(2), abs=0.02)
    assert with_it == pytest.approx(1, abs=0.02)


def test_image_shows_as_a_response():
    r = noise_figure(receiver(Amplifier(gain_db=30, nf_db=1)), "ant", "iff.out")
    freqs = sorted(f for f, _ in r.responses)
    assert freqs == pytest.approx([100e6, 121.4e6])
    assert abs(r.responses[0][1] - r.responses[1][1]) < 0.01
    assert any("other responses" in t.name and t.share > 0.3 for t in r.budget)


def test_spur_table_responses_add_noise():
    plain = noise_figure(receiver(Amplifier(gain_db=30, nf_db=1)), "ant", "iff.out")
    spurs = noise_figure(receiver(Amplifier(gain_db=30, nf_db=1), spur_table={"entries": {"3,1": 10}}),
                         "ant", "iff.out")
    # the 3 x 1 responses are two more sidebands at -10 dBc
    assert 10 ** (spurs.nf_db / 10) - 10 ** (plain.nf_db / 10) == pytest.approx(2 * 0.1 * 10 ** 0.1, abs=0.01)


def test_no_lo_leaves_the_tone_unconverted():
    c = receiver()
    c.remove("lo")
    with pytest.raises(ValueError, match="no product"):
        noise_figure(c, "ant", "iff.out", band=("10.6MHz", "10.8MHz"))


# ---------------------------------------------------------------- band, stimulus, sweep
def test_flat_band_matches_spot_and_noise_bandwidth():
    c = chain(Amplifier(gain_db=10, nf_db=3), IdealFilter(fmin="99MHz", fmax="101MHz"))
    r = noise_figure(c, "ant", "b1.out", band=("99.5MHz", "100.5MHz"))
    assert r.nf_db == pytest.approx(noise_figure(c, "ant", "b1.out").nf_db, abs=1e-9)
    assert r.noise_bw_hz == pytest.approx(1e6)
    assert r.floor_dbm == pytest.approx(KT0_DBM_HZ + r.nf_db + 60, abs=1e-6)


def test_noise_bandwidth_of_a_filter():
    r = noise_figure(receiver(), "ant", "iff.out", band=("10MHz", "11.4MHz"), points=801)
    assert r.noise_bw_hz == pytest.approx(200e3 * (math.pi / 8) / math.sin(math.pi / 8), rel=0.01)


def test_narrow_filter_in_a_wide_band_is_not_missed():
    c = chain(Amplifier(gain_db=10, nf_db=3), IdealFilter(fmin="100MHz", fmax="100.001MHz", rejection_db=100))
    r = noise_figure(c, "ant", "b1.out", band=("50MHz", "150MHz"))
    assert r.noise_bw_hz == pytest.approx(1e3, rel=0.25)
    assert r.gain_db == pytest.approx(10, abs=1e-6)


def test_muted_stimulus_is_found_and_put_back():
    c = Circuit()
    c.add("ant", Source(tones=[{"name": "RF", "freq": "100MHz", "power": -40, "mute": True},
                               {"name": "B", "freq": "150MHz", "power": -10}]))
    c.add("a", Amplifier(gain_db=20, nf_db=3, op1db_dbm=0))
    c.chain("ant", "a")
    r = noise_figure(c, "ant", "a.out", tone="RF")
    assert r.f_in == 100e6 and r.gain_db < 20         # compressed by the blocker alone
    assert [t.name for t in c["ant"].tones] == ["B"]
    assert c["ant"].specs[0].power == -40


def test_unknown_tone_and_wideband_refused():
    c = chain(Attenuator(1))
    with pytest.raises(ValueError, match="no tone"):
        noise_figure(c, "ant", "b0.out", tone="NOPE")
    w = Circuit()
    w.add("ant", Source(tones=[{"name": "W", "band": ["1MHz", "2MHz"], "mode": "wideband", "power": 0}]))
    w.add("a", Attenuator(1))
    w.chain("ant", "a")
    with pytest.raises(ValueError, match="wideband"):
        noise_figure(w, "ant", "a.out")


def test_sweep_tracks_and_restores():
    c = Circuit()
    c.add("ant", Source(tones=[{"name": "RF", "band": ["88MHz", "108MHz"], "mode": "sweep", "power": -60, "points": 5}]))
    c.add("lo", Source(tones=[{"name": "LO", "band": ["98.7MHz", "118.7MHz"], "mode": "sweep", "power": 7}]))
    c.add("pre", Filter("lowpass", "butter", 3, fc="90MHz"))
    c.add("mix", Mixer(conversion_loss_db=6))
    c.add("iff", Filter("bandpass", "butter", 4, f0="10.7MHz", bw="200kHz"))
    c.chain("ant", "pre", "mix", "iff")
    c.connect("lo", "mix.lo")
    before = c["ant"].specs[0].freq, c["lo"].specs[0].freq
    sw = noise_sweep(c, "ant", "iff.out", ("10.6MHz", "10.8MHz"), points=5)
    assert len(sw.nf_db) == 5 and sw.f_in[0] == 88e6 and sw.f_in[-1] == 108e6
    assert sw.nf_db[-1] > sw.nf_db[0] + 3                 # the low-pass eats the signal at the top
    assert (c["ant"].specs[0].freq, c["lo"].specs[0].freq) == before


def test_sweep_without_swept_tones_is_refused():
    with pytest.raises(ValueError, match="mode: sweep"):
        noise_sweep(chain(Attenuator(1)), "ant", "b0.out")


# ---------------------------------------------------------------- match
def test_matched_port():
    m = port_match(chain(Attenuator(3)), "b0.in", "100MHz")
    assert m.vswr == 1 and m.return_loss_db == math.inf and m.mismatch_loss_db == 0


def test_own_vswr():
    m = port_match(chain(Load(vswr=2)), "b0.in", "100MHz")
    assert m.vswr == pytest.approx(2) and m.vswr_worst == pytest.approx(2)
    assert m.return_loss_db == pytest.approx(9.542, abs=0.001)
    assert m.mismatch_loss_db == pytest.approx(0.512, abs=0.001)


def test_pad_improves_return_loss_by_twice_its_loss():
    m = port_match(chain(Attenuator(6), Load(vswr=2)), "b0.in", "100MHz")
    assert m.return_loss_db == pytest.approx(9.542 + 12, abs=0.001)


def test_typical_and_worst_case_add_differently():
    m = port_match(chain(Attenuator(6, vswr=1.2), Load(vswr=2)), "b0.in", "100MHz")
    near, far = gamma(1.2), gamma(2) * 10 ** (-12 / 20)
    assert math.hypot(near, far) <= m.gamma < near + far <= m.gamma_worst


def test_amplifier_isolates_its_load():
    m = port_match(chain(Amplifier(gain_db=20, reverse_isolation_db=40), Load(vswr=3)), "b0.in", "100MHz")
    assert m.return_loss_db == pytest.approx(-20 * math.log10(gamma(3)) + 20, abs=0.001)


def test_looking_back_towards_the_source():
    m = port_match(chain(Attenuator(3)), "b0.out", "100MHz")
    assert m.vswr == 1
    c = Circuit()
    c.add("ant", Source(freq="100MHz", power=0, name="RF", vswr=3))
    c.add("a", Attenuator(3))
    c.chain("ant", "a")
    assert port_match(c, "a.out", "100MHz").return_loss_db == pytest.approx(-20 * math.log10(gamma(3)) + 6, abs=0.001)


def test_mixer_passes_its_far_side_match_through_conversion():
    c = receiver(cl=6)
    c["mix"].rf_if = None
    c.remove("iff")
    c.add("l", Load(vswr=3))
    c.connect("mix.if", "l.in")
    m = port_match(c, "mix.rf", "100MHz")
    # 6 dB each way, and both sidebands come back: 10.7 MHz and 210.7 MHz both meet the load
    assert m.return_loss_db == pytest.approx(-20 * math.log10(gamma(3)) + 12 - 10 * math.log10(2), abs=0.001)


def test_match_sweep_follows_a_table():
    c = chain(Load(vswr=[["1GHz", 1.0], ["3GHz", 3.0]]))
    sw = match_sweep(c, "b0.in", ["1GHz", "3GHz"])
    assert sw.vswr == pytest.approx([1, 3])
    assert "VSWR" in sw.table()


def test_mismatch_report_follows_the_signal():
    c = receiver(Attenuator(1, vswr=1.5))
    rep = mismatch_report(c, tone="RF")
    by = {i.a.port + " " + i.b.port: i for i in rep.interfaces}
    i = next(v for k, v in by.items() if "iff" in k)
    assert i.freq == pytest.approx(10.7e6) and "RF" in i.product
    lo, hi = next(v for k, v in by.items() if "mix.rf" in k).ripple_db
    assert lo == 0 and hi == 0                 # nothing facing the mismatch on the far side


def test_mismatch_report_keeps_lo_chains_at_the_lo():
    c = receiver(Attenuator(1), lo_rf_isolation_db=30, rf_lo_isolation_db=30)
    c.disconnect("lo.out")
    c.add("drv", Amplifier(gain_db=3))
    c.connect("lo.out", "drv.in")
    c.connect("drv.out", "mix.lo")
    by = {i.a.port: i for i in mismatch_report(c, tone="RF").interfaces}
    assert by["lo.out"].freq == pytest.approx(110.7e6) and by["drv.out"].freq == pytest.approx(110.7e6)
    assert by["f0.out"].freq == pytest.approx(100e6)


def test_ripple_between_two_mismatches():
    c = Circuit()
    c.add("ant", Source(freq="100MHz", power=0, name="RF", vswr=2))
    c.add("l", Load(vswr=2))
    c.chain("ant", "l")
    i = mismatch_report(c).interfaces[0]
    assert i.ripple_db == pytest.approx((20 * math.log10(1 - 1 / 9), 20 * math.log10(1 + 1 / 9)))


# ---------------------------------------------------------------- sparam
S2P = """# MHz S MA R 50
50  0.5 0  0.5 0  0.5 0  0.25 90
150 0.5 0  0.5 0  0.5 0  0.25 90
"""


def test_sparam_uses_s11_only_when_asked(tmp_path):
    from hetsim import SParamFilter
    f = tmp_path / "d.s2p"
    f.write_text(S2P)
    assert port_match(chain(SParamFilter(f), Load()), "b0.in", "100MHz").vswr == 1
    c = chain(SParamFilter(f, vswr="file"), Load())
    assert port_match(c, "b0.in", "100MHz").gamma == pytest.approx(0.5)
    assert port_match(c, "b0.out", "100MHz").return_loss_db == pytest.approx(12.04, abs=0.01)
    # both terminations at T0 make it a load at T0, measured reflections included
    assert noise_figure(c, "ant", "b0.out").noise_out_dbm == pytest.approx(KT0_DBM_HZ, abs=1e-6)


# ---------------------------------------------------------------- loops
def test_loop_with_gain_is_reported():
    c = Circuit()
    c.add("ant", Source(freq="100MHz", power=-60, name="RF", vswr=20))
    c.add("a", Amplifier(gain_db=20, nf_db=3, reverse_isolation_db=6))
    c.add("l", Load(vswr=6))
    c.chain("ant", "a", "l")
    with pytest.raises(ValueError, match="feedback loop"):
        noise_figure(c, "ant", "a.out")


# ---------------------------------------------------------------- the example and the CLI
def test_example_runs(capsys):
    assert main(["noise", str(EXAMPLE), "--source", "ant", "--tone", "RF", "--port", "ifamp.out",
                 "--band", "10.6MHz:10.8MHz"]) == 0
    out = capsys.readouterr().out
    assert "noise figure" in out and "lna" in out
    assert main(["match", str(EXAMPLE), "--tone", "RF"]) == 0
    assert "lna.in" in capsys.readouterr().out
    assert main(["match", str(EXAMPLE), "--port", "lna.in", "--range", "88MHz:108MHz", "--points", "3"]) == 0


def test_cli_json(capsys):
    assert main(["noise", str(EXAMPLE), "--source", "ant", "--tone", "RF", "--port", "ifamp.out", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert 0 < d["nf_db"] < 10 and d["product"] == "LO - RF" and d["budget"]
    assert main(["match", str(EXAMPLE), "--port", "lna.in", "--freq", "100MHz", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["vswr"] >= 1.8


def test_cli_explains_a_tone_that_does_not_arrive(capsys):
    assert main(["noise", str(EXAMPLE), "--source", "ant", "--tone", "RF", "--port", "ifamp.out",
                 "--band", "1GHz:2GHz"]) == 1
    assert "no product" in capsys.readouterr().err
    assert main(["match", str(EXAMPLE), "--port", "lna.in"]) == 1


def test_example_nf_against_a_hand_cascade():
    from hetsim import load
    c, st = load(EXAMPLE)
    r = noise_figure(c, "ant", "ifamp.out", tone="RF", settings=st)
    # filter 0.5 dB, LNA 1.2 dB, image filter 1.5 dB, mixer 6.5 dB, IF filter ~0, IF amp 4 dB; mismatches move it a little
    db = [0.5, 1.2, 1.5, 6.5, 4.0]
    gain = [-0.5, 18, -1.5, -6, 20]
    f, g = 1.0, 1.0
    for nf, g_db in zip(db, gain):
        f += (10 ** (nf / 10) - 1) / g
        g *= 10 ** (g_db / 10)
    assert abs(r.nf_db - 10 * math.log10(f)) < 1.0


# ---------------------------------------------------------------- GUI parameters
def test_viewer_offers_vswr_and_nf():
    from hetsim.viewer.model import describe
    d = {"blocks": {"s": {"type": "source", "freq": "1MHz", "power": 0, "tones": [{"name": "A", "freq": 1e6, "power": 0}]},
                    "a": {"type": "amplifier", "gain_db": 10, "nf_db": 2, "vswr": 1.5},
                    "t": {"type": "attenuator", "loss_db": 1, "vswr": {"in": 2}},
                    "l": {"type": "load"}}}
    g = {x["block"]: {p["path"]: p for p in x["params"]} for x in describe(d)}
    assert g["a"]["vswr"]["value"] == 1.5 and g["a"]["vswr"]["unit"] == ":1" and g["a"]["nf_db"]["value"] == 2
    assert "vswr" not in g["t"]                      # per-port: edited in the netlist
    assert g["l"]["vswr"]["enabled"] is False and g["s"]["vswr"]["enabled"] is False
