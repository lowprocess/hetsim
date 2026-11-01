"""Receiver budget: cascades against hand calculations and against simulation."""
import json
import math
from pathlib import Path

import pytest

from hetsim import (Amplifier, Attenuator, Circuit, Filter, IdealFilter, Mixer, Source, Load, compare_budgets, load,
                    noise_figure, phase_noise_skirts, receiver_budget)
from hetsim.budget import signal_path
from hetsim.cli import main
from hetsim.smallsignal import KT0_DBM_HZ

EX = Path(__file__).parent.parent / "examples"


def amps(*blocks, power=-90):
    c = Circuit("chain")
    c.add("ant", Source(freq="100MHz", power=power, name="RF"))
    names = ["ant"]
    for i, b in enumerate(blocks):
        names.append(f"b{i}")
        c.add(names[-1], b)
    c.chain(*names)
    return c


def test_hand_cascade():
    c = amps(Amplifier(gain_db=15, nf_db=2, iip3_dbm=5), Attenuator(6), Amplifier(gain_db=20, nf_db=8, iip3_dbm=10))
    b = receiver_budget(c)
    f = 10 ** 0.2 + (10 ** 0.6 - 1) / 10 ** 1.5 + (10 ** 0.8 - 1) / 10 ** 0.9
    inv = 1 / 10 ** 0.5 + 10 ** 0.9 / 10
    assert b.gain_db == pytest.approx(29, abs=1e-6)
    assert b.nf_db == pytest.approx(10 * math.log10(f), abs=1e-6)
    assert b.iip3_dbm == pytest.approx(-10 * math.log10(inv), abs=1e-6)
    assert [s.block for s in b.stages] == ["b0", "b1", "b2"]
    assert [s.gain_db for s in b.stages] == pytest.approx([15, -6, 20])
    assert b.stages[0].cum_nf_db == pytest.approx(2, abs=1e-6)
    assert b.stages[0].cum_iip3_dbm == pytest.approx(5, abs=1e-6)
    assert b.limits["IIP3"] == "b2"


def test_iip3_cascade_matches_a_time_domain_two_tone():
    def chain(power):
        c = Circuit()
        c.add("ant", Source(tones=[{"name": "A", "freq": "100MHz", "power": power},
                                   {"name": "B", "freq": "101MHz", "power": power}]))
        c.add("a", Amplifier(gain_db=10, oip3_dbm=20))
        c.add("b", Amplifier(gain_db=10, oip3_dbm=25))
        c.chain("ant", "a", "b")
        return c
    want = receiver_budget(chain(-40), tone="A").iip3_dbm
    p = -40
    sp = chain(p).simulate_td(resolution="1MHz", fmax="500MHz").spectrum("b.out")
    level = dict(sp.peaks(-200))
    measured = p + (level[100e6] - level[99e6]) / 2
    assert measured == pytest.approx(want, abs=0.05)


def test_ip1db_matches_simulated_compression():
    c = amps(Amplifier(gain_db=20, nf_db=3, op1db_dbm=10), Attenuator(3))
    b = receiver_budget(c)
    assert b.ip1db_dbm == pytest.approx(10 + 1 - 20, abs=1e-6)
    c["ant"].tones[0].power = b.ip1db_dbm
    assert c.simulate()["b0.out"].power("RF") - b.ip1db_dbm == pytest.approx(19, abs=0.01)   # 1 dB down


def test_limiting_p1db_is_the_earliest_to_reach_it():
    c = amps(Amplifier(gain_db=30, op1db_dbm=20, nf_db=1), Amplifier(gain_db=20, op1db_dbm=15, nf_db=1))
    b = receiver_budget(c)
    assert b.ip1db_dbm == pytest.approx(15 + 1 - 50, abs=1e-6) and b.limits["input P1dB"] == "b1"


def test_floor_sensitivity_and_sfdr():
    c = amps(Amplifier(gain_db=20, nf_db=3, iip3_dbm=0), IdealFilter(fmin="90MHz", fmax="110MHz"))
    b = receiver_budget(c, bw="1MHz", snr_db=12)
    assert b.noise_bw_hz == pytest.approx(1e6)
    assert b.floor_dbm == pytest.approx(KT0_DBM_HZ + 3 + 60, abs=0.01)
    assert b.sensitivity_dbm == pytest.approx(b.floor_dbm + 12)
    assert b.sfdr_db == pytest.approx((2 / 3) * (0 - b.floor_dbm), abs=0.01)


def test_channel_wider_than_the_filter_sees_the_filter():
    c = amps(Amplifier(gain_db=20, nf_db=3), IdealFilter(fmin="99.9MHz", fmax="100.1MHz", rejection_db=100))
    b = receiver_budget(c, bw="2MHz")
    assert b.noise_bw_hz == pytest.approx(0.2e6, rel=0.01)
    # the 1.8 MHz outside the filter still delivers kT0, referred to the input through the gain of 100
    noise = 0.2e6 * 100 * 10 ** 0.3 + 1.8e6
    assert b.floor_dbm == pytest.approx(KT0_DBM_HZ + 10 * math.log10(noise / 100), abs=0.05)


def receiver(roof=None, lo_pn=None):
    c = Circuit("rx" if roof is None else "roofed")
    c.add("ant", Source(freq="100MHz", power=-100, name="RF"))
    lo = {"name": "LO", "freq": "110.7MHz", "power": 7}
    if lo_pn:
        lo["phase_noise"] = lo_pn
    c.add("lo", Source(tones=[lo]))
    c.add("lna", Amplifier(gain_db=20, nf_db=1, oip3_dbm=30))
    c.add("mix", Mixer(conversion_loss_db=6, iip3_dbm=10))
    c.add("if1", Filter("bandpass", "butter", 4, f0="10.7MHz", bw="200kHz"))
    c.add("ifa", Amplifier(gain_db=20, nf_db=4, iip3_dbm=-10, op1db_dbm=5))
    if roof:
        c.add("roof", Filter("bandpass", "butter", 5, f0="10.7MHz", bw=roof))
        c.chain("ant", "lna", "mix", "roof", "ifa", "if1")
    else:
        c.chain("ant", "lna", "mix", "ifa", "if1")
    c.connect("lo", "mix.lo")
    return c


def test_mixer_product_followed_through_the_chain():
    b = receiver_budget(receiver())
    assert b.product == "LO - RF" and b.f_out == pytest.approx(10.7e6)
    assert [s.freq for s in b.stages] == pytest.approx([100e6, 10.7e6, 10.7e6, 10.7e6])
    assert b.port == "if1.out"


def test_roofing_filter_removes_a_stage_from_the_out_of_band_iip3():
    plain = receiver_budget(receiver(), offset="1MHz")
    roofed = receiver_budget(receiver("200kHz"), offset="1MHz")
    assert plain.iip3_dbm == pytest.approx(-10 - 14, abs=0.3) and plain.limits["IIP3"] == "ifa"
    # with the IF amplifier protected, the mixer behind the LNA is what's left: 10 - 20
    assert roofed.iip3_dbm == pytest.approx(-10, abs=0.2) and roofed.limits["IIP3"] == "mix"
    # in-band tones get through the roofing filter, so they see no such benefit
    assert receiver_budget(receiver("200kHz")).iip3_dbm == pytest.approx(receiver_budget(receiver()).iip3_dbm, abs=0.1)


def test_blocking_compression_sees_the_filters():
    plain = receiver_budget(receiver(), bw="100kHz", offset="1MHz")
    roofed = receiver_budget(receiver("200kHz"), bw="100kHz", offset="1MHz")
    assert roofed.blocking_ip1db_dbm > plain.blocking_ip1db_dbm + 15
    assert roofed.blocking_ip1db_dbm > roofed.ip1db_dbm


PN = [[1e3, -80], [1e4, -100], [1e5, -120], [1e6, -140]]


def test_reciprocal_mixing_limit():
    b = receiver_budget(receiver(lo_pn=PN), bw="100kHz", offset="100kHz")
    assert b.bdr_phase_noise_db == pytest.approx(120 - 10 * math.log10(b.noise_bw_hz), abs=0.01)
    assert b.bdr_db <= b.bdr_phase_noise_db
    assert receiver_budget(receiver(), bw="100kHz", offset="100kHz").bdr_phase_noise_db is None


def test_reciprocal_limit_agrees_with_the_phase_noise_skirts():
    c = Circuit()
    c.add("ant", Source(tones=[{"name": "RF", "freq": "100MHz", "power": -100, "mute": True},
                               {"name": "B", "freq": "101MHz", "power": -30}]))
    c.add("lo", Source(tones=[{"name": "LO", "freq": "110.7MHz", "power": 7, "phase_noise": PN}]))
    c.add("mix", Mixer(conversion_loss_db=6))
    c.add("ift", IdealFilter(fmin="10MHz", fmax="11.5MHz", rejection_db=100))
    c.chain("ant", "mix", "ift")
    c.connect("lo", "mix.lo")
    b = receiver_budget(c, tone="RF", bw="100kHz", offset="1MHz")
    # the blocker's converted skirt on the channel: its level + L(offset), in the noise bandwidth
    skirts = phase_noise_skirts(c, "ift.out")
    sk = next(s for s in skirts if s.label == "LO - B" or s.label == "B - LO")
    density = float(sk.density([1e6])[0])
    in_channel = density + 10 * math.log10(b.noise_bw_hz)
    assert (-30 - 6) + (-140) + 10 * math.log10(b.noise_bw_hz) == pytest.approx(in_channel, abs=0.5)
    assert b.bdr_phase_noise_db == pytest.approx(-30 - 6 - in_channel, abs=0.5)


def test_rejection_is_the_image():
    b = receiver_budget(receiver())
    assert b.rejection_db == pytest.approx(0, abs=0.01) and b.rejection_at == pytest.approx(121.4e6)
    c = receiver()
    c.remove("lna")
    c.add("pre", Filter("bandpass", "butter", 5, f1="88MHz", f2="108MHz"))
    c.chain("ant", "pre", "mix")
    assert receiver_budget(c).rejection_db > 20


def test_paths_and_defaults():
    c = receiver()
    assert signal_path(c, "ant")[-1] == ("if1", "in", "out")
    with pytest.raises(ValueError, match="no signal path"):
        signal_path(c, "ant", ("lo", "out"))
    # stops at the last output, before a load
    a = amps(Amplifier(gain_db=10, nf_db=1), Load())
    assert receiver_budget(a).port == "b0.out"


def test_lo_chain_is_not_the_input():
    c = Circuit()
    c.add("lo", Source(freq="110.7MHz", power=7, name="LO"))
    c.add("ant", Source(freq="100MHz", power=-90, name="RF"))
    c.add("mix", Mixer(conversion_loss_db=6))
    c.chain("ant", "mix")
    c.connect("lo", "mix.lo")
    assert receiver_budget(c, product="LO - RF").source == "ant"


def test_muted_stimulus_and_restored():
    c = Circuit()
    c.add("ant", Source(tones=[{"name": "RF", "freq": "100MHz", "power": -40, "mute": True},
                               {"name": "B", "freq": "150MHz", "power": -10}]))
    c.add("a", Amplifier(gain_db=20, nf_db=3, op1db_dbm=0))
    c.chain("ant", "a")
    b = receiver_budget(c, tone="RF")
    assert b.gain_db < 20 and [t.name for t in c["ant"].tones] == ["B"]


def test_compare_and_text():
    a, b = receiver_budget(receiver(), bw="100kHz", offset="1MHz"), receiver_budget(receiver("200kHz"), bw="100kHz", offset="1MHz")
    text = compare_budgets([a, b])
    assert "rx" in text and "roofed" in text and "IIP3" in text
    t = a.table()
    assert "cum NF" in t and "set by" in t and "spur-free" in t


def test_no_intercept_says_so():
    b = receiver_budget(amps(Amplifier(gain_db=10, nf_db=1)))
    assert b.iip3_dbm == math.inf and any("IP3" in w for w in b.warnings) and "IIP3" in b.table()


def test_cli(capsys):
    a = str(EX / "vhf_noise_match.yaml")
    assert main(["budget", a, "--bw", "200kHz", "--offset", "1MHz", "--tone", "RF"]) == 0
    assert "cum IIP3" in capsys.readouterr().out
    assert main(["budget", a, str(EX / "vhf_single_conversion.yaml"), "--bw", "200kHz", "--tone", "RF"]) == 0
    out = capsys.readouterr().out
    assert "VHF noise and match" in out and "VHF single conversion" in out
    assert main(["budget", a, "--tone", "RF", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d[0]["stages"][0]["block"] == "pre" and d[0]["nf_db"] > 0
    assert main(["budget", a, "--tone", "NOPE"]) == 1
    assert "no tone" in capsys.readouterr().err


# ---------------------------------------------------------------- varying a parameter
def test_budget_sweep_against_friis():
    from hetsim import budget_sweep
    c = amps(Amplifier(gain_db=10, nf_db=2), Amplifier(gain_db=20, nf_db=10))
    sw = budget_sweep(c, "b0.gain_db", [0.0, 10.0, 20.0])
    for g, nf in zip([0, 10, 20], sw.series("nf_db")):
        assert nf == pytest.approx(10 * math.log10(10 ** 0.2 + (10 - 1) / 10 ** (g / 10)), abs=1e-6)
    assert c["b0"].gain == 10                         # the circuit given is left alone
    assert "b0.gain_db" in sw.table()


def test_budget_sweep_nested_path_and_bad_names():
    from hetsim import budget_sweep
    c = amps(Amplifier(gain_db=10, nf_db=2, op1db_dbm=-10))
    sw = budget_sweep(c, "ant.tones.0.power", [-90.0, 0.0])
    assert sw.series("gain_db")[1] < sw.series("gain_db")[0] - 1     # the drive compresses it
    with pytest.raises(ValueError, match="no parameter"):
        budget_sweep(c, "nope.gain_db", [1.0])
    with pytest.raises(ValueError, match="no parameter"):
        budget_sweep(c, "b0", [1.0])
    with pytest.raises(ValueError, match="no parameter"):
        budget_sweep(c, "ant.tones.9.power", [1.0])
    with pytest.raises(ValueError, match="no parameter"):
        budget_sweep(c, "b0.nonsense", [1.0])


def test_budget_sweep_cli(capsys):
    a = str(EX / "vhf_noise_match.yaml")
    assert main(["budget", a, "--tone", "RF", "--bw", "200kHz", "--vary", "lna.gain_db=5:25:3"]) == 0
    out = capsys.readouterr().out
    assert "lna.gain_db" in out and len(out.strip().splitlines()) == 4
    assert main(["budget", a, "--tone", "RF", "--vary", "lna.gain_db=nonsense"]) == 1
    assert "from:to:points" in capsys.readouterr().err
