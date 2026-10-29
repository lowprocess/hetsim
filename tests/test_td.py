"""Time-domain solver: agreement with the tone solver, and the things only it shows."""
import math

import pytest

from hetsim import (Amplifier, Circuit, Filter, Mixer, Source, SpurTable, from_dict)
from test_hetsim import mixer_rig, superhet

T = dict(abs=0.05)   # dB agreement between solvers for small-signal cases


def both(c, port, label, **td):
    tone = c.simulate()[port]
    t = tone.get(label)
    tdr = c.simulate_td(**td)
    return t.power_dbm, tdr.spectrum(port).power_at(t.freq), tdr


def test_ideal_mixer_td():
    c = mixer_rig()
    r = c.simulate_td()
    sp = r.spectrum("mix.if")
    assert sp.power_at(10e6) == pytest.approx(-20, abs=1e-6)
    assert sp.power_at(210e6) == pytest.approx(-20, abs=1e-6)
    others = [(f, p) for f, p in sp.peaks(-250) if f not in (10e6, 210e6)]
    assert not [o for o in others if abs(o[0] - 10e6) > 1 and abs(o[0] - 210e6) > 1]


@pytest.mark.parametrize("label", ["LO - RF", "LO", "RF"])
def test_mixer_isolation_and_loss_match(label):
    c = mixer_rig(conversion_loss_db=6, lo_if_isolation_db=30, lo_rf_isolation_db=35, rf_if_isolation_db=25)
    a, b, _ = both(c, "mix.if", label)
    assert b == pytest.approx(a, abs=1e-6)


@pytest.mark.parametrize("label", ["3*LO - RF", "2*LO - 2*RF", "2*RF", "LO + 2*RF"])
def test_spur_table_matches(label):
    t = SpurTable({(3, 1): 15, (2, 2): 65, (0, 2): 60, (1, 2): 60}, ref_rf_dbm=-10)
    a, b, _ = both(mixer_rig(rf_dbm=-10, conversion_loss_db=6, spur_table=t), "mix.if", label)
    assert b == pytest.approx(a, **T)


def test_square_lo_odd_harmonics():
    a, b, _ = both(mixer_rig(conversion_loss_db=4, lo_waveform="square"), "mix.if", "3*LO - RF")
    assert a == pytest.approx(-20 - 4 - 20 * math.log10(3), abs=1e-6)
    assert b == pytest.approx(a, abs=1e-6)


def test_mixer_iip3_matches():
    c = Circuit()
    c.add("rf", Source(tones=[{"name": "A", "freq": 100e6, "power": -20},
                              {"name": "B", "freq": 100.1e6, "power": -20}]))
    c.add("lo", Source(freq=110e6, power=7, name="LO"))
    c.add("mix", Mixer(conversion_loss_db=6, iip3_dbm=10))
    c.connect("rf", "mix.rf")
    c.connect("lo", "mix.lo")
    a, b, _ = both(c, "mix.if", "B + LO - 2*A")
    assert b == pytest.approx(a, **T)


def two_tone(p_in=-20, **amp):
    c = Circuit()
    c.add("s", Source(tones=[{"name": "A", "freq": 100e6, "power": p_in},
                             {"name": "B", "freq": 101e6, "power": p_in}]))
    c.add("amp", Amplifier(**amp))
    c.chain("s", "amp")
    return c


@pytest.mark.parametrize("label", ["A", "2*A - B", "2*B - A", "2*A + B", "3*A"])
def test_amplifier_ip3_matches(label):
    a, b, _ = both(two_tone(gain_db=10, oip3_dbm=20), "amp.out", label)
    assert b == pytest.approx(a, abs=0.1)


@pytest.mark.parametrize("label", ["A + B", "B - A", "2*A"])
def test_amplifier_ip2_matches(label):
    a, b, _ = both(two_tone(gain_db=10, oip2_dbm=40), "amp.out", label)
    assert b == pytest.approx(a, abs=0.01)


def test_amplifier_p1db_single_tone():
    c = Circuit()
    c.add("s", Source(freq=1e6, power=1.0, name="S"))
    c.add("amp", Amplifier(gain_db=10, op1db_dbm=10))
    c.chain("s", "amp")
    r = c.simulate_td(fmax=20e6)
    assert r.spectrum("amp.out").power_at(1e6) == pytest.approx(10, abs=0.02)
    # the instantaneous limiter also makes odd harmonics, which the tone model doesn't
    assert r.spectrum("amp.out").power_at(3e6) > -60


def test_filter_magnitude_and_phase():
    c = Circuit()
    c.add("s", Source(freq=10e6, power=0, name="S", phase=0))
    c.add("f", Filter("lowpass", "butter", 3, fc=10e6))
    c.chain("s", "f")
    r = c.simulate_td(fmax=40e6)
    assert r.spectrum("f.out").power_at(10e6) == pytest.approx(-3.0103, abs=1e-3)
    # 3rd-order Butterworth: -135 degrees at the cutoff
    import numpy as np
    X = np.fft.rfft(r.waveform("f.out"))
    k = int(round(10e6 / r.plan.resolution))
    assert math.degrees(np.angle(X[k])) == pytest.approx(-135, abs=0.01)


def test_lo_back_out_of_antenna_matches():
    c = superhet()
    tone = c.simulate().incident("ant.out").power("LO")
    td = c.simulate_td().incident_spectrum("ant.out").power_at(110.7e6)
    assert td == pytest.approx(tone, abs=1e-6)


def test_superhet_main_products_match():
    c = superhet()
    r = c.simulate_td()
    sp = r.spectrum("iff.out")
    tone = r.tone["iff.out"]
    for label in ("LO - RF", "LO"):
        t = tone.get(label)
        assert sp.power_at(t.freq) == pytest.approx(t.power_dbm, abs=0.05)
    assert r.converged


@pytest.mark.parametrize("phase_b, expected", [(0, 0 + 20 * math.log10(2)), (180, -400)])
def test_same_bin_products_add_coherently(phase_b, expected):
    """Tone solver adds same-frequency tones in power; TD in amplitude and phase."""
    c = Circuit()
    c.add("a", Source(freq=1e6, power=0, name="A", phase=0))
    c.add("b", Source(freq=1e6, power=0, name="B", phase=phase_b))
    from hetsim import Splitter
    c.add("sp", Splitter(isolation_db=200))
    c.connect("a", "sp.a")
    c.connect("b", "sp.b")
    tone_p = 10 * math.log10(sum(10 ** (t.power_dbm / 10) for t in c.simulate()["sp.sum"]))
    td_p = c.simulate_td(fmax=4e6).spectrum("sp.sum").power_at(1e6)
    assert tone_p == pytest.approx(0, abs=1e-6)        # two -3 dB tones, power sum
    assert td_p == pytest.approx(expected - 3.0103, abs=1e-3) or (expected < -300 and td_p < -250)


def test_plan_resolution_and_alias_warning():
    c = mixer_rig()
    r = c.simulate_td()
    assert r.plan.resolution == pytest.approx(10e6)           # gcd(100 MHz, 110 MHz)
    assert r.plan.nyquist > 210e6
    r2 = c.simulate_td(fmax=150e6)                             # LO + RF at 210 MHz cannot fit
    assert "alias" in r2.warning_text()


def test_too_many_samples_is_an_error():
    c = Circuit()
    c.add("s", Source(tones=[{"name": "A", "freq": 1e9, "power": 0},
                             {"name": "B", "freq": 1e9 + 1, "power": 0}]))   # 1 Hz grid at 1 GHz
    with pytest.raises(ValueError, match="samples"):
        c.simulate_td()


def test_netlist_td_settings():
    d = {"settings": {"td": {"resolution": "1MHz", "fmax": "50MHz"}},
         "blocks": {"s": {"type": "source", "freq": "10MHz", "power": 0, "name": "S"},
                    "a": {"type": "attenuator", "loss_db": 3}},
         "chain": ["s", "a"]}
    c, st = from_dict(d)
    r = c.simulate_td(st)
    assert r.plan.resolution == 1e6
    assert r.spectrum("a.out").power_at(10e6) == pytest.approx(-3)
