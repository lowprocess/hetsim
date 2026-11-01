"""The coherent tone solver and phased port match, against closed forms and the time-domain solver."""
import cmath
import math
import time
from pathlib import Path

import numpy as np
import pytest

from hetsim import (ADC, Amplifier, Attenuator, Circuit, Filter, IQMixer, Line, Load, Mixer, Settings, Source, Splitter, from_dict,
                    load, match_sweep, noise_figure, port_match, receiver_budget, to_dict)
from hetsim.blocks.nonlinear import _X1DB, rapp_compression_db
from hetsim.blocks.passive import REACTIVE_MIN_RL_DB
from hetsim.tolerance import monte_carlo


def gamma(s):
    return (s - 1) / (s + 1)


def on(c, st=None):
    return c.simulate(st or Settings(coherent=True))


def td_peak(c, port, f, res="1MHz", fmax="600MHz", incident=False):
    r = c.simulate_td(resolution=res, fmax=fmax)
    sp = r.incident_spectrum(port) if incident else r.spectrum(port)
    return dict(sp.peaks(-300)).get(f, -math.inf)


# ---------------------------------------------------------------- the switch
def test_power_solver_stays_the_default_and_ignores_phase():
    c = Circuit()
    c.add("s", Source(freq="10MHz", power=0, name="A", vswr=3))
    c.add("l", Load(vswr=3))
    c.chain("s", "l")
    assert c.simulate()["s.out"].power("A") == pytest.approx(-10 * math.log10(1 - gamma(3) ** 4), abs=0.01)


def test_a_line_switches_it_on_and_false_overrides():
    c = Circuit()
    c.add("s", Source(freq="10MHz", power=0, name="A", vswr=3))
    c.add("t", Line(delay_ns=25))        # a quarter wave each way: the bounce comes back in antiphase
    c.add("l", Load(vswr=3))
    c.chain("s", "t", "l")
    want = -20 * math.log10(1 + gamma(3) ** 2)
    assert c.simulate().incident("l.in").power("A") == pytest.approx(want, abs=1e-6)
    off = c.simulate(Settings(coherent=False)).incident("l.in").power("A")
    assert off != pytest.approx(want, abs=0.01)


def test_settings_round_trip():
    c, st = from_dict({"blocks": {"s": {"type": "source", "freq": "1MHz", "power": 0}, "t": {"type": "line", "length_m": 2, "loss_db": 1}},
                       "chain": ["s", "t"], "settings": {"coherent": True}})
    assert st.coherent is True
    assert to_dict(c, st)["settings"]["coherent"] is True and from_dict(to_dict(c, st))[0]["t"].delay == c["t"].delay


@pytest.mark.parametrize("kw", [dict(), dict(delay_ns=1, length_m=1), dict(delay_ns=-1), dict(length_m=1, velocity_factor=2)])
def test_bad_line(kw):
    with pytest.raises(ValueError):
        Line(**kw)


def test_line_length_to_delay():
    assert Line(length_m=1, velocity_factor=0.5).delay == pytest.approx(2 / 299792458)


# ---------------------------------------------------------------- closed forms
@pytest.mark.parametrize("delay", [0.0, 7.0, 12.5, 25.0, 31.0])
def test_bounce_series_is_exact(delay):
    c = Circuit()
    c.add("s", Source(freq="10MHz", power=-10, name="A", vswr=2, phase=0))
    c.add("t", Line(delay_ns=delay, loss_db=1))
    c.add("l", Load(vswr=3))
    c.chain("s", "t", "l")
    tau = delay * 1e-9
    h = 10 ** (-1 / 20) * cmath.exp(-2j * math.pi * 10e6 * tau)
    want = h / (1 - gamma(2) * gamma(3) * h * h)
    t = on(c).incident("l.in").get("A")
    assert t.power_dbm == pytest.approx(-10 + 20 * math.log10(abs(want)), abs=1e-6)
    assert t.phase == pytest.approx(cmath.phase(want), abs=1e-6)


def test_near_total_reflection_costs_nothing():
    """The case where the iterating solvers give up: the time domain fails at VSWR 30, the power solver at 1000."""
    c = Circuit()
    c.add("s", Source(freq="10MHz", power=0, name="A", vswr=1000, phase=0))
    c.add("t", Line(delay_ns=3))
    c.add("l", Load(vswr=1000))
    c.chain("s", "t", "l")
    r = on(c)
    assert r.converged and r.iterations <= 3
    h = cmath.exp(-2j * math.pi * 10e6 * 3e-9)
    g = gamma(1000)
    assert r.incident("l.in").power("A") == pytest.approx(20 * math.log10(abs(h / (1 - g * g * h * h))), abs=1e-6)


def test_two_paths_interfere():
    def built(delay):
        c = Circuit()
        c.add("s", Source(freq="100MHz", power=0, name="A", phase=0))
        c.add("d", Splitter(isolation_db=200))
        c.add("t", Line(delay_ns=delay))
        c.add("u", Attenuator(0))
        c.add("q", Splitter(isolation_db=200))
        c.connect("s.out", "d.sum")
        c.connect("d.a", "t.in")
        c.connect("d.b", "u.in")
        c.connect("t.out", "q.a")
        c.connect("u.out", "q.b")
        return c
    # a splitter pair is lossless in phase, and in antiphase everything goes to the isolation ports
    assert on(built(0.0))["q.sum"].power("A") == pytest.approx(0, abs=1e-6)
    assert on(built(5.0))["q.sum"].power("A") < -100
    assert on(built(2.5))["q.sum"].power("A") == pytest.approx(-3.01, abs=0.01)


def test_filter_phase_is_used():
    c = Circuit()
    c.add("s", Source(freq="10MHz", power=0, name="A", phase=0))
    c.add("f", Filter("lowpass", "butter", 3, fc="10MHz"))
    c.chain("s", "f")
    t = on(c)["f.out"].get("A")
    h = c["f"].s21_c(np.array([10e6]))[0]
    assert t.phase == pytest.approx(cmath.phase(h), abs=1e-9) and t.power_dbm == pytest.approx(20 * math.log10(abs(h)), abs=1e-6)


# ---------------------------------------------------------------- against the time domain
def rx(delay=0.0, reflect=2.0, **mix):
    c = Circuit()
    c.add("ant", Source(freq="100MHz", power=-30, name="RF", phase=20))
    c.add("lo", Source(freq="110MHz", power=7, name="LO", phase=50))
    c.add("mix", Mixer(conversion_loss_db=6, **mix))
    c.add("t", Line(delay_ns=delay))
    c.add("l", Load(vswr=reflect))
    c.chain("ant", "mix", "t", "l")
    c.connect("lo", "mix.lo")
    return c


@pytest.mark.parametrize("delay", [0.0, 3.0, 11.0, 17.0, 40.0])
def test_mixer_reflection_matches_td(delay):
    c = rx(delay, 3.0)
    r = c.simulate()
    # the reflected 10 MHz converts back to 100 MHz, arriving at the antenna
    assert td_peak(c, "ant.out", 100e6, incident=True) == pytest.approx(r.incident("ant.out").power("RF"), abs=0.02)
    assert td_peak(c, "t.out", 10e6) == pytest.approx(r["t.out"].power("LO - RF"), abs=0.02)


@pytest.mark.parametrize("delay", [0.0, 1.0, 2.5, 4.0])
def test_im3_of_two_stages_adds_like_td(delay):
    def chain():
        c = Circuit()
        c.add("s", Source(tones=[{"name": "A", "freq": "100MHz", "power": -40, "phase": 10},
                                 {"name": "B", "freq": "101MHz", "power": -40, "phase": 40}]))
        c.add("a", Amplifier(gain_db=10, oip3_dbm=20))
        c.add("t", Line(delay_ns=delay))
        c.add("b", Amplifier(gain_db=10, oip3_dbm=25))
        c.chain("s", "a", "t", "b")
        return c
    c = chain()
    assert c.simulate()["b.out"].power("2*A - B") == pytest.approx(td_peak(c, "b.out", 99e6), abs=0.05)


def test_in_phase_im3_matches_the_budget_cascade():
    c = Circuit()
    c.add("s", Source(tones=[{"name": "A", "freq": "100MHz", "power": -40}, {"name": "B", "freq": "101MHz", "power": -40}]))
    c.add("a", Amplifier(gain_db=10, oip3_dbm=20))
    c.add("b", Amplifier(gain_db=10, oip3_dbm=25))
    c.chain("s", "a", "b")
    tone = c.simulate(Settings(coherent=True))["b.out"]
    measured = -40 + (tone.power("A") - tone.power("2*A - B")) / 2
    assert measured == pytest.approx(receiver_budget(c, tone="A").iip3_dbm, abs=0.01)
    # the power solver adds the stages' IM in power, and reads optimistic
    p = c.simulate(Settings(coherent=False))["b.out"]
    assert -40 + (p.power("A") - p.power("2*A - B")) / 2 > measured + 0.5


def test_iq_mixer_output_phase_matches_td():
    c = Circuit()
    c.add("ant", Source(freq="120MHz", power=-30, name="RF", phase=20))
    c.add("lo", Source(freq="110MHz", power=7, name="LO", phase=50))
    c.add("m", IQMixer(conversion_loss_db=6, gain_imbalance_db=0.5, phase_imbalance_deg=3))
    c.add("t", Line(delay_ns=7))
    c.add("l", Load(vswr=3))
    c.chain("ant", "m", "t", "l")
    c.connect("lo", "m.lo")
    r = c.simulate()
    assert td_peak(c, "t.out", 10e6) == pytest.approx(r["t.out"].power("RF - LO"), abs=0.02)
    assert td_peak(c, "ant.out", 120e6, incident=True) == pytest.approx(r.incident("ant.out").power("RF"), abs=0.05)


def test_adc_alias_keeps_its_phase():
    c = Circuit()
    c.add("s", Source(freq="130MHz", power=-3, name="RF", phase=33))
    c.add("a", ADC(sample_rate="100MHz", bits=24))
    c.chain("s", "a")
    r = c.simulate(Settings(coherent=True))["a.out"].tones()[0]
    td = c.simulate_td(resolution="1MHz", fmax="400MHz").waveform("a.out")
    n = len(td)
    z = np.exp(-2j * np.pi * 30 * np.arange(n) / n) @ td * 2 / n
    assert r.phase == pytest.approx(np.angle(z), abs=0.01)
    assert r.power_dbm == pytest.approx(10 * math.log10(abs(z) ** 2 / 100 * 1e3), abs=0.01)


def test_example_chain_agrees_with_power_solver_when_nothing_reflects():
    c, st = from_dict({"blocks": {"s": {"type": "source", "tones": [{"name": "RF", "freq": "100MHz", "power": -30}]},
                                  "a": {"type": "amplifier", "gain_db": 10, "oip3_dbm": 30, "op1db_dbm": 20},
                                  "lo": {"type": "source", "name": "LO", "freq": "110.7MHz", "power": 7},
                                  "m": {"type": "mixer", "conversion_loss_db": 6, "spur_table": "dbm_generic",
                                        "lo_rf_isolation_db": 35, "rf_if_isolation_db": 25},
                                  "f": {"type": "filter", "response": "bandpass", "f0": "10.7MHz", "bw": "1MHz"}},
                       "chain": ["s", "a", "m", "f"], "wires": [["lo.out", "m.lo"]]})
    p, q = c.simulate(st)["f.out"], c.simulate(Settings(coherent=True))["f.out"]
    for t in p.strongest(12):
        assert q.power(t.vec) == pytest.approx(t.power_dbm, abs=0.02), t.label


def test_compression_and_warnings_are_kept():
    c = Circuit()
    c.add("s", Source(freq="100MHz", power=-5, name="RF"))
    c.add("a", Amplifier(gain_db=20, op1db_dbm=10, oip3_dbm=30))
    c.chain("s", "a")
    assert c.simulate()["a.out"].power("RF") == pytest.approx(c.simulate(Settings(coherent=True))["a.out"].power("RF"), abs=0.01)
    assert "a" in c.simulate(Settings(coherent=True)).warnings


# ---------------------------------------------------------------- feedback
def test_loop_with_gain_is_flagged():
    c = Circuit()
    c.add("s", Source(freq="100MHz", power=-60, name="RF", vswr=20))
    c.add("a", Amplifier(gain_db=20, reverse_isolation_db=6))
    c.add("l", Load(vswr=6))
    c.chain("s", "a", "l")
    r = on(c)
    assert not r.converged and "feedback" in r.warning_text()


# ---------------------------------------------------------------- the rest of the program on top of it
def rxl():
    c = Circuit("rxl")
    c.add("ant", Source(freq="100MHz", power=-90, name="RF", vswr=2))
    c.add("t", Line(length_m=3))
    c.add("lna", Amplifier(gain_db=20, nf_db=2, oip3_dbm=20, vswr={"in": 2}))
    c.add("lo", Source(freq="110.7MHz", power=7, name="LO"))
    c.add("m", Mixer(conversion_loss_db=6))
    c.add("f", Filter("bandpass", "butter", 3, f0="10.7MHz", bw="1MHz"))
    c.chain("ant", "t", "lna", "m", "f")
    c.connect("lo", "m.lo")
    return c


def test_analyses_run_on_a_coherent_circuit():
    c = rxl()
    assert noise_figure(c, "ant", "f.out", tone="RF").nf_db > 2
    assert receiver_budget(c, bw="1MHz").nf_db > 2


def test_noise_of_a_passive_line_is_kt0():
    from hetsim.smallsignal import KT0_DBM_HZ
    c = Circuit()
    c.add("s", Source(freq="100MHz", power=-60, name="RF"))
    c.add("t", Line(delay_ns=3, loss_db=4))
    c.chain("s", "t")
    r = noise_figure(c, "s", "t.out")
    assert r.nf_db == pytest.approx(4, abs=1e-6) and r.noise_out_dbm == pytest.approx(KT0_DBM_HZ, abs=1e-6)


def test_band_sweep_with_workers_keeps_phases():
    c = Circuit()
    c.add("s", Source(tones=[{"name": "A", "band": ["90MHz", "110MHz"], "mode": "sweep", "points": 9, "power": 0, "phase": 0}], vswr=3))
    c.add("t", Line(delay_ns=10))
    c.add("l", Load(vswr=3))
    c.chain("s", "t", "l")
    one = c.simulate_sweep(workers=1).products("t.out")[0].powers
    two = c.simulate_sweep(workers=2).products("t.out")[0].powers
    assert np.allclose(one, two)
    assert np.ptp(one) > 1


def test_viewer_simulates_a_line():
    from hetsim.viewer import ViewerModel
    from hetsim.viewer.model import describe
    m = ViewerModel.from_circuit(rxl())
    g = {x["block"]: {p["path"]: p for p in x["params"]} for x in describe(m.base)}
    assert g["t"]["length_m"]["unit"] == "m"
    r = m.simulate({"probe": "f.out", "solver": "tone", "min_dbm": -200, "params": {}})
    assert r["tone"]


# ---------------------------------------------------------------- phased match
@pytest.mark.parametrize("f", [40e6, 50e6, 63e6, 100e6])
def test_phased_match_through_a_line(f):
    c = Circuit()
    c.add("s", Source(freq="10MHz", power=0, name="A"))
    c.add("p", Attenuator(1, vswr=1.5))
    c.add("t", Line(delay_ns=5, loss_db=1))
    c.add("l", Load(vswr=2))
    c.chain("s", "p", "t", "l")
    m = port_match(c, "p.in", f)
    h = 10 ** (-2 / 20) * cmath.exp(-2j * math.pi * f * 5e-9)     # pad and line, one way
    near, far = gamma(1.5), gamma(2)
    # p's own output port also reflects, so solve the small loop by hand
    loop = 10 ** (-1 / 20) * cmath.exp(-2j * math.pi * f * 5e-9)
    b = far * loop * loop / (1 - near * far * loop * loop)        # looking into p.out
    want = near + 10 ** (-2 / 20) * b
    assert m.gamma_c == pytest.approx(want, abs=1e-9)
    assert abs(m.gamma_c) <= m.gamma_worst + 1e-12


def test_standing_wave_ripple_period():
    c = Circuit()
    c.add("s", Source(freq="10MHz", power=0, name="A"))
    c.add("p", Attenuator(0, vswr=2))
    c.add("t", Line(delay_ns=10))
    c.add("l", Load(vswr=2))
    c.chain("s", "p", "t", "l")
    f = np.linspace(100e6, 200e6, 401)
    sw = match_sweep(c, "p.in", f)
    peaks = f[1:-1][(sw.vswr_phased[1:-1] > sw.vswr_phased[:-2]) & (sw.vswr_phased[1:-1] > sw.vswr_phased[2:])]
    assert np.diff(peaks) == pytest.approx(1 / (2 * 10e-9), rel=0.02)       # 50 MHz
    assert sw.vswr_phased.max() > sw.vswr.max() > sw.vswr_phased.min()
    assert sw.vswr_phased.max() == pytest.approx(sw.vswr_worst.max(), abs=0.02)


def test_lower_sideband_round_trip_through_a_mixer():
    c = rx(delay=4.0, reflect=3.0)
    c.remove("ant")                # looking in at the RF port, at 100 MHz: IF is 10 MHz (lower) and 210 MHz (sum)
    m = port_match(c, "mix.rf", 100e6)
    h = 10 ** (-6 / 20)
    g = gamma(3)
    want = h * h * g * (cmath.exp(2j * math.pi * 10e6 * 2 * 4e-9) + cmath.exp(-2j * math.pi * 210e6 * 2 * 4e-9))
    assert m.gamma_c == pytest.approx(want, abs=1e-9)


def test_zero_phase_means_worst_case():
    c = Circuit()
    c.add("s", Source(freq="10MHz", power=0, name="A"))
    c.add("p", Attenuator(3, vswr=1.5))
    c.add("l", Load(vswr=2))
    c.chain("s", "p", "l")
    m = port_match(c, "p.in", 100e6)
    assert abs(m.gamma_c) == pytest.approx(m.gamma_worst, abs=1e-9)


def test_measured_s11_phase_is_used(tmp_path):
    from hetsim import SParamFilter
    f = tmp_path / "d.s2p"
    f.write_text("# MHz S MA R 50\n50 0.5 40 0.5 0 0.5 0 0.25 90\n150 0.5 40 0.5 0 0.5 0 0.25 90\n")
    c = Circuit()
    c.add("s", Source(freq="10MHz", power=0, name="A"))
    c.add("d", SParamFilter(f, vswr="file"))
    c.chain("s", "d")
    assert port_match(c, "d.in", 100e6).gamma_c == pytest.approx(cmath.rect(0.5, math.radians(40)), abs=1e-9)


# ---------------------------------------------------------------- the time domain between reflecting ports
@pytest.mark.parametrize("s", [3, 30, 100, 1000])
def test_td_converges_where_it_used_to_give_up(s):
    c = Circuit()
    c.add("a", Source(freq="10MHz", power=-10, name="RF", vswr=s, phase=0))
    c.add("t", Line(delay_ns=3))
    c.add("l", Load(vswr=s))
    c.chain("a", "t", "l")
    r = c.simulate_td(resolution="1MHz", fmax="50MHz")
    assert r.converged and r.iterations <= 3
    want = c.simulate().incident("l.in").power("RF")
    assert dict(r.incident_spectrum("l.in").peaks(-200))[10e6] == pytest.approx(want, abs=1e-6)


def test_td_network_solve_handles_a_nonlinear_block_in_the_loop():
    c = Circuit()
    c.add("a", Source(tones=[{"name": "A", "freq": "10MHz", "power": -10, "phase": 0},
                             {"name": "B", "freq": "11MHz", "power": -10, "phase": 30}], vswr=30))
    c.add("t", Line(delay_ns=3))
    c.add("g", Amplifier(gain_db=0, oip3_dbm=20, reverse_isolation_db=0))
    c.add("l", Load(vswr=30))
    c.chain("a", "t", "g", "l")
    r = c.simulate_td(resolution="1MHz", fmax="200MHz")
    assert r.converged
    tone = c.simulate()
    assert tone.converged
    peaks = dict(r.spectrum("g.out").peaks(-200))
    assert peaks[10e6] == pytest.approx(tone["g.out"].power("A"), abs=0.3)


def test_no_reflection_stays_on_plain_iteration():
    c = Circuit()
    c.add("a", Source(freq="10MHz", power=-10, name="RF"))
    c.add("t", Line(delay_ns=3))
    c.add("l", Load())
    c.chain("a", "t", "l")
    assert c.simulate_td(resolution="1MHz", fmax="50MHz").iterations <= 4


def test_cable_example_ripples_at_its_electrical_length():
    from pathlib import Path
    from hetsim import load
    c, st = load(Path(__file__).parent.parent / "examples" / "vhf_cable.yaml")
    f = np.linspace(88e6, 108e6, 401)
    sw = match_sweep(c, "coax.in", f, st)
    assert np.ptp(sw.vswr_phased) > 0.3 and np.ptp(sw.vswr) < 0.01          # only the phased figure sees it
    peaks = f[1:-1][(sw.vswr_phased[1:-1] > sw.vswr_phased[:-2]) & (sw.vswr_phased[1:-1] > sw.vswr_phased[2:])]
    assert np.diff(peaks).mean() == pytest.approx(0.66 * 299792458 / (2 * 10), rel=0.02)
    # the signal itself ripples across the band too, between the antenna and the LNA
    power = c.simulate_sweep(st, points=81).products("iff.out", fmin="10.6MHz", fmax="10.8MHz")[0].powers
    assert np.nanmax(power) - np.nanmin(power) > 0.3


# -- reactive filters: reflection from the response -------------------------------------------------

def _ladder(g, x):
    """S11, S21, S22 at jx of a series-first LC ladder between 1 ohm ports, from its ABCD matrix."""
    out = []
    for w in np.atleast_1d(x):
        m = np.eye(2, dtype=complex)
        for i, v in enumerate(g):
            m = m @ (np.array([[1, 1j * w * v], [0, 1]]) if i % 2 == 0 else np.array([[1, 0], [1j * w * v, 1]]))
        (a, b), (c, d) = m
        den = a + b + c + d
        out.append(((a + b - c - d) / den, 2 / den, (-a + b - c + d) / den))
    return np.array(out).T


def _g_butter(n):
    return [2 * math.sin((2 * k - 1) * math.pi / (2 * n)) for k in range(1, n + 1)]


def _g_cheby(n, rp):
    beta = math.log(1 / math.tanh(rp * math.log(10) / 40))
    y = math.sinh(beta / (2 * n))
    a = [math.sin((2 * k - 1) * math.pi / (2 * n)) for k in range(1, n + 1)]
    b = [y * y + math.sin(k * math.pi / n) ** 2 for k in range(1, n + 1)]
    g = [2 * a[0] / y]
    for k in range(1, n):
        g.append(4 * a[k - 1] * a[k] / (b[k - 1] * g[k - 1]))
    return g


@pytest.mark.parametrize("family,order,g,kw", [
    ("butter", 1, _g_butter(1), {}), ("butter", 3, _g_butter(3), {}), ("butter", 4, _g_butter(4), {}),
    ("butter", 7, _g_butter(7), {}), ("cheby1", 3, _g_cheby(3, 0.5), {"ripple_db": 0.5}),
    ("cheby1", 5, _g_cheby(5, 0.5), {"ripple_db": 0.5})])
@pytest.mark.parametrize("response,args,to_x", [
    ("lowpass", {"fc": 1e9}, lambda f: f / 1e9),
    ("highpass", {"fc": 1e9}, lambda f: -1e9 / f),
    ("bandpass", {"f1": 90e6, "f2": 110e6}, lambda f: (f * f - 90e6 * 110e6) / (f * 20e6)),
    ("bandstop", {"f1": 90e6, "f2": 110e6}, lambda f: f * 20e6 / (90e6 * 110e6 - f * f))])
def test_reactive_matches_an_lc_ladder(family, order, g, kw, response, args, to_x):
    """The reflection, both ports and phase included, against the network the filter's formulas describe."""
    fl = Filter(response=response, family=family, order=order, max_rejection_db=400, vswr="reactive", **args, **kw)
    centre = args.get("fc") or 100e6
    f = np.linspace(0.3, 2.5, 401) * centre + 1234.5
    s11, s21, s22 = _ladder(g, to_x(f))
    h = fl._prototype_s11(fl._prototype_w(f), "in"), fl._prototype_s11(fl._prototype_w(f), "out")
    assert np.abs(h[0] - s11).max() < 1e-6
    assert np.abs(h[1] - s22).max() < 1e-6
    assert np.abs(fl.s21_c(f) - s21).max() < 1e-2     # the filter normalises its peak, to a grid: close, not exact


@pytest.mark.parametrize("family,kw", [("butter", {}), ("cheby1", {"ripple_db": 0.5}), ("cheby2", {"stop_db": 60}),
                                       ("ellip", {"ripple_db": 0.5, "stop_db": 60}), ("bessel", {})])
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 8, 11])
def test_reactive_is_lossless_and_unitary(family, kw, order):
    fl = Filter(response="lowpass", family=family, order=order, fc=1.0, max_rejection_db=1000, vswr="reactive", **kw)
    x = np.concatenate([[0.0], np.linspace(1e-3, 2, 801), np.geomspace(1e-3, 1e3, 401)])
    s = 1j * x
    z, p, k = fl._proto
    s21 = np.full(s.shape, k, dtype=complex)
    for a in z:
        s21 *= s - a
    for a in p:
        s21 /= s - a
    s11, s22 = fl._prototype_s11(x, "in"), fl._prototype_s11(x, "out")
    assert np.abs(np.abs(s11) ** 2 + np.abs(s21) ** 2 - 1).max() < 1e-7
    assert np.abs(np.abs(s22) ** 2 + np.abs(s21) ** 2 - 1).max() < 1e-7
    assert np.abs(s11 * s21.conj() + s21 * s22.conj()).max() < 1e-7


def test_reactive_refuses_an_order_it_cannot_work_out():
    with pytest.raises(ValueError, match="reactive"):
        Filter(response="lowpass", family="ellip", order=20, fc=1e6, vswr="reactive")
    Filter(response="lowpass", family="ellip", order=20, fc=1e6)    # without it, as before


def test_reactive_magnitude_stays_passive_with_loss_leakage_and_a_stopband_cap():
    fl = Filter(f0=100e6, bw=10e6, order=5, insertion_loss_db=3, max_rejection_db=30, vswr="reactive")
    f = np.geomspace(1e6, 10e9, 4001)
    r, t = fl.reflection("in", f), np.abs(fl.s21_c(f)) ** 2
    assert (r + t).max() <= 1 + 1e-12
    assert r.max() <= 10 ** (-REACTIVE_MIN_RL_DB / 10) + 1e-12
    assert fl.reflection("in", 1e9) == pytest.approx(10 ** (-REACTIVE_MIN_RL_DB / 10))
    assert fl.reflection("in", 100e6) < 1e-3        # in band it matches, insertion loss or not
    assert isinstance(fl.reflection("out", 100e6), float) and isinstance(fl.gamma_c("out", 100e6), complex)


def test_reactive_at_dc_and_at_a_notch_centre_gives_numbers():
    for resp in ("bandpass", "highpass", "bandstop"):
        args = {"fc": 1e6} if resp == "highpass" else {"f1": 1e6, "f2": 2e6}
        fl = Filter(response=resp, order=3, vswr="reactive", **args)
        g = fl.gamma_c("in", np.array([0.0, math.sqrt(2) * 1e6, -1e6, 1e6]))
        assert np.all(np.isfinite(g)) and np.abs(g).max() <= 1
        assert g[2] == pytest.approx(np.conj(g[3]))     # a real filter: conjugate at negative frequency


def _reactive_chain(**kw):
    """A source into a reactive filter, a mismatched cable, then a load that reflects."""
    c = Circuit("reactive")
    c.add("src", Source(name="A", freq=kw.pop("freq", 100e6), power=0.0, vswr=kw.pop("src_vswr", None)))
    c.add("flt", Filter(f1=90e6, f2=110e6, order=3, vswr="reactive", insertion_loss_db=kw.pop("il", 0.0)))
    c.add("end", Load(vswr=kw.pop("load_vswr", 3.0)))
    c.chain("src", "flt", "end")
    return c


def test_reactive_uses_the_coherent_solver_and_returns_a_stopband_to_the_source():
    c = _reactive_chain(freq=300e6, load_vswr=1.0)
    r = c.simulate()
    assert r.converged and r.iterations <= 3
    fl = c.blocks["flt"]
    level = r.incident("src.out").power("A")
    assert level == pytest.approx(10 * math.log10(fl.reflection("in", 300e6)), abs=1e-6)
    assert level > -0.11


def test_reactive_passband_mismatch_settles_to_the_closed_form():
    f = 103e6
    c = _reactive_chain(freq=f)
    r = c.simulate()
    fl, end = c.blocks["flt"], c.blocks["end"]
    s21, s11, s12 = fl.s21_c(f), fl.gamma_c("in", f), fl.s12_c(f)
    s22, gl = fl.gamma_c("out", f), end.gamma_c("in", f)
    want = np.ravel(s11 + s21 * gl * s12 / (1 - s22 * gl))[0]
    got = r.incident("src.out").power("A")
    assert got == pytest.approx(20 * math.log10(abs(want)), abs=1e-6)


@pytest.mark.parametrize("f", [100e6, 300e6, 20e6])
def test_facing_reactive_stopbands_stay_solvable(f):
    c = Circuit("two filters")
    c.add("src", Source(name="A", freq=f, power=0.0, vswr=2.0))
    c.add("a", Filter(f1=90e6, f2=110e6, order=5, vswr="reactive"))
    c.add("b", Filter(f1=90e6, f2=110e6, order=3, vswr="reactive"))
    c.add("end", Load())
    c.chain("src", "a", "b", "end")
    r = c.simulate()
    assert r.converged and not r.warnings


@pytest.mark.parametrize("f", [100e6, 112e6, 130e6])
def test_reactive_tone_and_time_domain_solvers_agree(f):
    c = _reactive_chain(freq=f, src_vswr=1.5)
    assert td_peak(c, "flt.out", f) == pytest.approx(c.simulate()["flt.out"].power("A"), abs=0.05)
    assert td_peak(c, "src.out", f, incident=True) == pytest.approx(c.simulate().incident("src.out").power("A"), abs=0.05)


def test_reactive_match_and_noise_figures():
    f = 100e6 * 1.2
    c = _reactive_chain(freq=100e6, load_vswr=1.0)
    m = port_match(c, "flt.in", f)
    fl = c.blocks["flt"]
    assert m.gamma_c == pytest.approx(fl.gamma_c("in", f), abs=1e-9)
    # lossless: what is not reflected passes, so a matched lossless filter adds no noise
    nf = noise_figure(c, "src", "flt.out", None, 0)
    assert nf.nf_db == pytest.approx(0.0, abs=0.01)
    lossy = _reactive_chain(freq=100e6, load_vswr=1.0, il=2.0)
    assert noise_figure(lossy, "src", "flt.out", None, 0).nf_db == pytest.approx(2.0, abs=0.05)


def test_reactive_round_trips_through_a_netlist():
    c = _reactive_chain()
    d = to_dict(c)
    assert d["blocks"]["flt"]["vswr"] == "reactive"
    c2, _ = from_dict(d)
    assert c2.blocks["flt"].reactive and c2.blocks["flt"].needs_coherent


def test_reactive_vswr_on_other_blocks_is_an_error():
    with pytest.raises(ValueError):
        Attenuator(loss_db=1, vswr="reactive")


def _receiver(pad=None, spurs="dbm_generic"):
    """Reactive filters on both ports of a mixer, and if asked, a pad between the mixer and one of them."""
    c = Circuit("both sides")
    c.add("ant", Source(name="RF", freq=100e6, power=-60.0))
    c.add("img", Filter(f1=88e6, f2=108e6, order=5, vswr="reactive"))
    c.add("mix", Mixer(conversion_loss_db=6, lo_rf_isolation_db=35, lo_if_isolation_db=30, rf_if_isolation_db=25,
                       spur_table=spurs))
    c.add("lo", Source(name="LO", freq=110.7e6, power=7.0))
    c.add("iff", Filter(f0=10.7e6, bw=200e3, order=4, family="cheby1", vswr="reactive"))
    c.add("end", Load())
    if pad:
        c.add("pad", Attenuator(loss_db=pad))
        c.chain("ant", "img", "pad", "mix", "iff", "end")
    else:
        c.chain("ant", "img", "mix", "iff", "end")
    c.connect("lo", "mix.lo")
    return c


def test_reactive_on_both_mixer_ports_stops_and_says_why():
    c = _receiver()
    r = c.simulate()
    assert not r.converged and any("pad" in w for w in r.warnings["<solver>"])
    td = c.simulate_td(resolution="1MHz", fmax="600MHz")
    assert not td.converged and td.iterations < 200 and any("pad" in w for w in td.warnings["<solver>"])


def test_an_ideal_mixer_between_reactive_filters_never_settles_and_says_why():
    r = _receiver(spurs=None).simulate(Settings(max_iter=30))
    assert not r.converged and any("pad" in w for w in r.warnings["<solver>"])


def test_a_pad_settles_a_mixer_between_reactive_filters():
    r = _receiver(pad=6, spurs=None).simulate()
    assert r.converged and not r.warnings.get("<solver>")
    assert r["iff.out"].power("LO - RF") == pytest.approx(-60 - 6 - 6, abs=1.5)


def test_a_budget_and_a_monte_carlo_on_a_runaway_circuit_say_so():
    c = _receiver()
    assert "pad" in receiver_budget(c, source="ant", tone="RF", port="iff.out").table()
    mc = monte_carlo(c, runs=3, tolerances={"iff.insertion_loss_db": 0.1}, source="ant", tone="RF", port="iff.out")
    assert mc.failed == 3


def test_rapp_compression_survives_any_drive():
    assert rapp_compression_db(5000.0, 10.0) < -4000
    assert rapp_compression_db(-200.0, 10.0) == pytest.approx(0.0, abs=1e-9)
    # the two forms agree where they meet
    psat = 10.0 + 1.0 - 20 * math.log10(_X1DB)
    assert rapp_compression_db(psat + 199.99, 10.0) == pytest.approx(rapp_compression_db(psat + 200.01, 10.0), abs=0.05)


def test_td_finishes_when_a_port_holds_only_rounding_error():
    c = Circuit("a stopband leaves next to nothing beyond the filter")
    c.add("ant", Source(name="RF", freq=100e6, power=-60.0))
    c.add("mix", Mixer(conversion_loss_db=6, lo_if_isolation_db=30, rf_if_isolation_db=25))
    c.add("lo", Source(name="LO", freq=110.7e6, power=7.0))
    c.add("iff", Filter(f0=10.7e6, bw=200e3, order=4, family="cheby1", vswr="reactive"))
    c.add("end", Load())
    c.chain("ant", "mix", "iff", "end")
    c.connect("lo", "mix.lo")
    td = c.simulate_td(resolution="10kHz", fmax="400MHz")
    assert td.converged and td.iterations < 60
    assert dict(td.spectrum("iff.out").peaks(-300)).get(10.7e6) == pytest.approx(c.simulate()["iff.out"].power("LO - RF"), abs=0.05)


@pytest.mark.parametrize("path", sorted(Path(__file__).parent.parent.joinpath("examples").glob("*.yaml")), ids=lambda p: p.name)
def test_every_example_netlist_settles(path):
    c, st = load(path)
    r = c.simulate(st)
    assert r.converged and not r.warnings.get("<solver>")
