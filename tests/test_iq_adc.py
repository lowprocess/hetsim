"""I/Q mixer and ADC: both solvers against the textbook formulas and each other."""
import math
from pathlib import Path

import numpy as np
import pytest

from hetsim import (ADC, Amplifier, Circuit, Filter, IdealFilter, IQMixer, Mixer, Source, compare_budgets, find_responses,
                    from_dict, load, noise_figure, receiver_budget, to_dict)
from hetsim.smallsignal import KT0_DBM_HZ

EX = Path(__file__).parent.parent / "examples"


def irr(g_db, phi_deg):
    g, p = 10 ** (g_db / 20), math.radians(phi_deg)
    return 10 * math.log10((1 + 2 * g * math.cos(p) + g * g) / (1 - 2 * g * math.cos(p) + g * g))


def iq(rf="120MHz", lo="110MHz", power=-30.0, **kw):
    c = Circuit()
    c.add("ant", Source(freq=rf, power=power, name="RF", phase=30))
    c.add("lo", Source(freq=lo, power=7, name="LO", phase=70))
    c.add("m", IQMixer(conversion_loss_db=6, **kw))
    c.chain("ant", "m")
    c.connect("lo", "m.lo")
    return c


def td_at(c, f, port="m.if", fmax="800MHz", res="1MHz"):
    peaks = dict(c.simulate_td(resolution=res, fmax=fmax).spectrum(port).peaks(-250))
    return peaks.get(f, -math.inf)


# ---------------------------------------------------------------- I/Q: tone solver
def test_upper_keeps_rf_above_the_lo_and_cancels_the_rest():
    s = iq().simulate()["m.if"]
    assert s.power("RF - LO") == pytest.approx(-36, abs=1e-6)
    assert s.power("LO + RF") == -math.inf


def test_upper_cancels_rf_below_the_lo():
    s = iq(rf="100MHz").simulate()["m.if"]
    assert s.power("LO - RF") == -math.inf and s.power("LO + RF") == -math.inf


def test_lower_is_the_mirror_and_keeps_the_sum():
    s = iq(rf="100MHz", sideband="lower").simulate()["m.if"]
    assert s.power("LO - RF") == pytest.approx(-36, abs=1e-6) and s.power("LO + RF") == pytest.approx(-36, abs=1e-6)
    assert iq(sideband="lower").simulate()["m.if"].power("RF - LO") == -math.inf


@pytest.mark.parametrize("g,phi", [(0.5, 0.0), (0.0, 3.0), (0.3, 2.0), (-1.0, -5.0)])
def test_image_rejection_formula(g, phi):
    kw = dict(gain_imbalance_db=g, phase_imbalance_deg=phi)
    want = iq(**kw).simulate()["m.if"].power("RF - LO")
    image = iq(rf="100MHz", **kw).simulate()["m.if"].power("LO - RF")
    assert want - image == pytest.approx(irr(g, phi), abs=1e-6)
    assert IQMixer(**kw).image_rejection_db() == pytest.approx(irr(g, phi), abs=1e-6)


def test_third_harmonic_prefers_the_opposite_sideband():
    s = iq(rf="340MHz", lo="110MHz", spur_table={"entries": {"3,1": 10}}).simulate()["m.if"]
    assert s.power("RF - 3*LO") == -math.inf           # upper for the LO is lower for its third harmonic
    assert s.power("3*LO + RF") == pytest.approx(-46, abs=1e-6)
    s = iq(rf="320MHz", lo="110MHz", spur_table={"entries": {"3,1": 10}}).simulate()["m.if"]
    assert s.power("3*LO - RF") == pytest.approx(-46, abs=1e-6)


def test_isolation_is_as_given_and_no_lo_warns():
    c = iq(lo_if_isolation_db=30, rf_if_isolation_db=25)
    s = c.simulate()["m.if"]
    assert s.power("LO") == pytest.approx(7 - 30) and s.power("RF") == pytest.approx(-30 - 25)
    c.remove("lo")
    r = c.simulate()
    assert any("no LO" in w for w in r.warnings["m"])


def test_round_trip():
    c, st = from_dict({"blocks": {"s": {"type": "source", "freq": "1MHz", "power": 0},
                                  "m": {"type": "iq_mixer", "sideband": "lower", "gain_imbalance_db": 0.5},
                                  "a": {"type": "adc", "sample_rate": "10MHz", "bits": 12, "hd3_dbc": -80}},
                       "wires": [["s.out", "m.rf"], ["m.if", "a.in"]]})
    d = to_dict(c, st)
    assert d["blocks"]["m"]["sideband"] == "lower" and d["blocks"]["a"]["bits"] == 12
    assert from_dict(d)[0]["m"].gain_imb == 0.5


def test_bad_sideband():
    with pytest.raises(ValueError, match="sideband"):
        IQMixer(sideband="both")


# ---------------------------------------------------------------- I/Q: the time domain agrees
@pytest.mark.parametrize("kw", [dict(), dict(gain_imbalance_db=0.5, phase_imbalance_deg=3.0)])
def test_td_wanted_and_image_match_the_tone_solver(kw):
    want = iq(**kw)
    image = iq(rf="100MHz", **kw)
    assert td_at(want, 10e6) == pytest.approx(want.simulate()["m.if"].power("RF - LO"), abs=0.02)
    t_img = image.simulate()["m.if"].power("LO - RF")
    td_img = td_at(image, 10e6)
    if t_img == -math.inf:
        assert td_img < -36 - 150
    else:
        assert td_img == pytest.approx(t_img, abs=0.05)


def test_td_lower_sideband_and_sum():
    c = iq(rf="100MHz", sideband="lower", gain_imbalance_db=0.5, phase_imbalance_deg=3.0)
    s = c.simulate()["m.if"]
    assert td_at(c, 10e6) == pytest.approx(s.power("LO - RF"), abs=0.02)
    assert td_at(c, 210e6) == pytest.approx(s.power("LO + RF"), abs=0.02)


def test_td_third_harmonic():
    c = iq(rf="320MHz", lo="110MHz", spur_table={"entries": {"3,1": 10}}, phase_imbalance_deg=2.0)
    s = c.simulate()["m.if"]
    assert td_at(c, 10e6, fmax="2GHz") == pytest.approx(s.power("3*LO - RF"), abs=0.05)


def test_td_iip2_in_quadrature_matches():
    c = Circuit()
    c.add("ant", Source(tones=[{"name": "A", "freq": "120MHz", "power": -20, "phase": 10},
                               {"name": "B", "freq": "125MHz", "power": -20, "phase": 50}]))
    c.add("lo", Source(freq="110MHz", power=7, name="LO"))
    c.add("m", IQMixer(conversion_loss_db=6, iip2_dbm=30))
    c.chain("ant", "m")
    c.connect("lo", "m.lo")
    tone = c.simulate()["m.if"].power("B - A")
    assert tone == pytest.approx(-20 - 20 - 30 - 6 - 10 * math.log10(2), abs=1e-6)
    assert td_at(c, 5e6) == pytest.approx(tone, abs=0.05)


# ---------------------------------------------------------------- I/Q: noise
def lna_then(mixer):
    c = Circuit()
    c.add("ant", Source(freq="120MHz", power=-90, name="RF"))
    c.add("lna", Amplifier(gain_db=30, nf_db=1))
    c.add("lo", Source(freq="110MHz", power=7, name="LO"))
    c.add("m", mixer)
    c.chain("ant", "lna", "m")
    c.connect("lo", "m.lo")
    return c


def test_no_image_noise_penalty_unlike_a_plain_mixer():
    plain = noise_figure(lna_then(Mixer(conversion_loss_db=6)), "ant", "m.if", product="RF - LO").nf_db
    quad = noise_figure(lna_then(IQMixer(conversion_loss_db=6)), "ant", "m.if").nf_db
    assert plain - quad == pytest.approx(10 * math.log10(2), abs=0.02)
    assert quad == pytest.approx(1, abs=0.02)


def test_passive_iq_mixer_is_a_load_at_t0():
    r = noise_figure(iq(gain_imbalance_db=1.0, phase_imbalance_deg=5.0), "ant", "m.if")
    assert r.noise_out_dbm == pytest.approx(KT0_DBM_HZ, abs=1e-6)


def test_datasheet_nf_is_kept():
    assert noise_figure(iq(nf_db=9.0), "ant", "m.if").nf_db == pytest.approx(9, abs=1e-6)


def test_image_noise_follows_the_imbalance():
    k = dict(gain_imbalance_db=1.0, phase_imbalance_deg=10.0)
    r = noise_figure(lna_then(IQMixer(conversion_loss_db=6, **k)), "ant", "m.if", product="RF - LO")
    rej = {round(f): g for f, g in r.responses}
    assert rej[120_000_000] - rej[100_000_000] == pytest.approx(irr(1.0, 10.0), abs=0.01)
    # 1 dB of LNA noise over both responses, the image scaled down by the rejection
    f = 10 ** 0.1 * (1 + 10 ** (-irr(1.0, 10.0) / 10))
    assert r.nf_db == pytest.approx(10 * math.log10(f), abs=0.02)


# ---------------------------------------------------------------- ADC: tone solver
def adc(f="130MHz", fs="100MHz", power=-20.0, **kw):
    c = Circuit()
    c.add("ant", Source(freq=f, power=power, name="RF", phase=20))
    c.add("a", ADC(sample_rate=fs, **kw))
    c.chain("ant", "a")
    return c


def test_first_zone_is_unchanged():
    s = adc("30MHz").simulate()["a.out"]
    assert s.power("RF") == pytest.approx(-20) and len(s) == 1


def test_aliases_are_labelled():
    t = adc("130MHz").simulate()["a.out"].tones()
    assert len(t) == 1 and t[0].label == "RF - a_FS" and t[0].freq == pytest.approx(30e6) and t[0].power_dbm == pytest.approx(-20)
    t = adc("170MHz").simulate()["a.out"].tones()
    assert t[0].label == "2*a_FS - RF" and t[0].freq == pytest.approx(30e6)


def test_harmonics_fold_and_scale():
    s = adc("30MHz", hd2_dbc=-70, hd3_dbc=-80, full_scale_dbm=10).simulate()["a.out"]
    h2 = next(t for t in s if "2*RF" in t.label)
    assert h2.freq == pytest.approx(40e6) and h2.power_dbm == pytest.approx(-20 - 70 - 30, abs=1e-6)
    h3 = next(t for t in s if "3*RF" in t.label)
    assert h3.freq == pytest.approx(10e6) and h3.power_dbm == pytest.approx(-20 - 80 - 60, abs=1e-6)


def test_overload_warns():
    assert any("full scale" in w for w in adc(power=15.0).simulate().warnings["a"])
    assert not adc(power=5.0).simulate().warnings


def test_find_responses_sees_aliases():
    t = find_responses(adc("30MHz"), "ant", "a.out", ("29MHz", "31MHz"), "1MHz", "400MHz")
    assert sorted(round(r.freq / 1e6) for r in t.responses) == [30, 70, 130, 170, 230, 270, 330, 370]


@pytest.mark.parametrize("kw", [dict(bits=0), dict(sample_rate=0), dict(hd3_dbc=70), dict(bits=2.5)])
def test_bad_adc(kw):
    with pytest.raises(ValueError):
        ADC(**{"sample_rate": 1e6, **kw})


# ---------------------------------------------------------------- ADC: time domain
def test_td_alias_level_matches():
    c = adc("130MHz", power=-3.0, bits=16)
    peaks = dict(c.simulate_td(resolution="1MHz", fmax="400MHz").spectrum("a.out").peaks(-250))
    assert peaks[30e6] == pytest.approx(-3, abs=0.02)


def test_td_quantisation_snr_is_6n_plus_1_76():
    c = adc("31.7MHz", fs="100MHz", power=9.9, bits=10, full_scale_dbm=10)
    sp = c.simulate_td(resolution="10kHz", fmax="100MHz").spectrum("a.out")
    bins = dict(sp.peaks(-300))
    sig = bins.pop(31.7e6)
    noise = 10 * math.log10(sum(10 ** (v / 10) for v in bins.values()))
    assert (sig + 0.1) - noise == pytest.approx(6.02 * 10 + 1.76, abs=1.5)


def test_td_snr_db_adds_noise():
    c = adc("31.7MHz", fs="100MHz", power=10.0, bits=14, full_scale_dbm=10, snr_db=60)
    bins = dict(c.simulate_td(resolution="10kHz", fmax="100MHz").spectrum("a.out").peaks(-300))
    sig = bins.pop(31.7e6)
    assert sig - 10 * math.log10(sum(10 ** (v / 10) for v in bins.values())) == pytest.approx(60, abs=1.0)


def test_td_clipping_makes_harmonics_and_warns():
    r = adc("30MHz", power=16.0, bits=14, full_scale_dbm=10).simulate_td(resolution="1MHz", fmax="400MHz")
    peaks = dict(r.spectrum("a.out").peaks(-250))
    assert peaks[30e6] < 15.5 and peaks[10e6] > -5         # third harmonic folded to 10 MHz
    assert any("clip" in w for w in r.warnings["a"])


def test_td_harmonic_levels_match_the_tone_solver():
    c = adc("30MHz", power=10.0, bits=24, hd2_dbc=-60, hd3_dbc=-70)
    tone = c.simulate()["a.out"]
    peaks = dict(c.simulate_td(resolution="1MHz", fmax="400MHz").spectrum("a.out").peaks(-250))
    assert peaks[40e6] == pytest.approx(next(t for t in tone if "2*RF" in t.label).power_dbm, abs=0.2)
    assert peaks[10e6] == pytest.approx(next(t for t in tone if "3*RF" in t.label).power_dbm, abs=0.2)


def test_sample_rate_joins_the_default_grid():
    r = adc("130MHz", fs="70MHz").simulate_td()
    assert r.plan.resolution == pytest.approx(10e6)
    assert not r.plan.notes


# ---------------------------------------------------------------- ADC: noise
def test_adc_noise_density():
    a = ADC(sample_rate="100MHz", bits=14, full_scale_dbm=10)
    assert a.noise_density_dbm_hz() == pytest.approx(10 - (6.02 * 14 + 1.76) - 10 * math.log10(50e6))
    assert ADC(sample_rate="100MHz", snr_db=70, full_scale_dbm=4).noise_density_dbm_hz() == pytest.approx(4 - 70 - 76.99, abs=0.01)


def front_end(aa=True, gain=30.0, fs="100MHz", **kw):
    c = Circuit()
    c.add("ant", Source(freq="30MHz", power=-90, name="RF"))
    c.add("amp", Amplifier(gain_db=gain, nf_db=3))
    names = ["ant", "amp"]
    if aa:
        c.add("aa", IdealFilter(fmax="50MHz", rejection_db=120))
        names.append("aa")
    c.add("a", ADC(sample_rate=fs, snr_db=70, full_scale_dbm=10, **kw))
    c.chain(*names, "a")
    return c


def test_nf_with_an_adc_is_the_cascade():
    c = front_end()
    r = noise_figure(c, "ant", "a.out")
    adc_kt0 = 10 ** ((c["a"].noise_density_dbm_hz() - KT0_DBM_HZ) / 10)
    # the filter stands in for kT0 at each of the 19 other responses up to the default analog_bw
    assert r.nf_db == pytest.approx(10 * math.log10(10 ** 0.3 + (adc_kt0 + 19) / 1000), abs=0.002)
    assert any(t.name == "a" for t in r.budget)


def test_without_anti_alias_filter_every_zone_folds_in():
    filtered = noise_figure(front_end(), "ant", "a.out")
    open_ = noise_figure(front_end(aa=False, analog_bw="500MHz"), "ant", "a.out")
    # single-sideband: source and amplifier noise, F kT0 x 1000, arrives at all ten responses, against one for the signal
    adc_kt0 = 10 ** ((front_end()["a"].noise_density_dbm_hz() - KT0_DBM_HZ) / 10)
    assert open_.nf_db == pytest.approx(10 * math.log10(10 * 10 ** 0.3 + adc_kt0 / 1000), abs=0.01)
    assert open_.nf_db > filtered.nf_db + 5
    assert len(open_.responses) == 10                       # up to 500 MHz, both sides of each multiple of fs


def test_nothing_above_nyquist():
    a = ADC(sample_rate=100e6)
    assert a.transfers("out", 60e6, None) == [] and a.noise_emission("out", 60e6, None) == 0


def test_budget_sees_full_scale_and_distortion():
    c = front_end(hd3_dbc=-80)
    b = receiver_budget(c, bw="1MHz")
    assert b.ip1db_dbm == pytest.approx(10 - 30, abs=1e-6) and b.limits["input P1dB"] == "a"
    want_iip3 = 10 + (80 - 20 * math.log10(3)) / 2 - 30
    assert b.iip3_dbm == pytest.approx(want_iip3, abs=0.01) and b.limits["IIP3"] == "a"


# ---------------------------------------------------------------- comparing architectures
def test_superhet_against_direct_conversion():
    sh = receiver_budget(*_load("rx_superhet"), bw="200kHz", offset="1MHz")
    dc = receiver_budget(*_load("rx_direct_conversion"), bw="200kHz", offset="1MHz")
    # direct conversion: image held only by the balance, and the mixer's IIP2 behind the LNA gain
    assert dc.rejection_db == pytest.approx(irr(0.2, 1.5), abs=0.1)
    assert dc.iip2_dbm == pytest.approx(50 - 17.5, abs=0.1)
    assert dc.limits["IIP2 at the mixers"] == "iq"
    assert dc.product == "RF - LO" and sh.product == "LO - RF"
    assert [s.block for s in dc.stages][-1] == "adc"
    assert "direct conversion" in compare_budgets([sh, dc])


def _load(name):
    c, st = load(EX / f"{name}.yaml")
    return c, None, "RF", None


def test_examples_run_in_both_solvers():
    for name in ("rx_superhet", "rx_direct_conversion"):
        c, st = load(EX / f"{name}.yaml")
        assert c.simulate(st).converged
    r = load(EX / "rx_direct_conversion.yaml")[0].simulate_td(resolution="100kHz", fmax="500MHz")
    assert r.converged and not any("clip" in w for ws in r.warnings.values() for w in ws)


def test_viewer_offers_the_new_blocks():
    from hetsim.viewer.model import ADD_DEFAULTS, describe
    from hetsim.viewer import ViewerModel
    m = ViewerModel.from_file(EX / "rx_direct_conversion.yaml")
    g = {x["block"]: {p["path"]: p for p in x["params"]} for x in describe(m.base)}
    assert g["iq"]["sideband"]["choices"] == ["upper", "lower"] and g["iq"]["phase_imbalance_deg"]["unit"] == "deg"
    assert g["adc"]["bits"]["kind"] == "int" and g["adc"]["sample_rate"]["kind"] == "freq"
    assert m.default_resolution == pytest.approx(100e3)
    assert {"iq_mixer", "adc"} <= set(ADD_DEFAULTS)
    r = m.simulate({"probe": "adc.out", "solver": "tone", "min_dbm": -200, "params": {}})
    assert r["tone"] and "error" not in r
    assert m.simulate({"probe": "adc.out", "solver": "td", "min_dbm": -200, "params": {}})["td"]
