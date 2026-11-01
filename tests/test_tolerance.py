"""Monte Carlo of the receiver budget."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from hetsim import Amplifier, Attenuator, Circuit, Filter, Source, from_dict, load, monte_carlo, to_dict
from hetsim.cli import main
from hetsim.tolerance import tolerances_of

EX = Path(__file__).parent.parent / "examples" / "vhf_noise_match.yaml"


def amp(gain=20.0, nf=3.0, **kw):
    c = Circuit("a")
    c.add("ant", Source(freq="100MHz", power=-90, name="RF"))
    c.add("a", Amplifier(gain_db=gain, nf_db=nf, **kw))
    c.chain("ant", "a")
    return c


def test_tolerances_round_trip_and_leave_the_solver_alone():
    d = {"blocks": {"s": {"type": "source", "freq": "1MHz", "power": 0},
                    "a": {"type": "amplifier", "gain_db": 10, "tolerance": {"gain_db": 1.5}}}, "chain": ["s", "a"]}
    c, st = from_dict(d)
    assert tolerances_of(c) == {"a.gain_db": 1.5}
    assert to_dict(c, st)["blocks"]["a"]["tolerance"] == {"gain_db": 1.5}
    assert c.simulate()["a.out"].power("s") == pytest.approx(10)
    with pytest.raises(ValueError, match="tolerance must map"):
        from_dict({"blocks": {"a": {"type": "load", "tolerance": 3}}})


def test_normal_spread_matches_3_sigma_and_stays_inside():
    mc = monte_carlo(amp(), 4000, {"a.gain_db": 3.0})
    g = mc.results["gain_db"]
    assert g.mean() == pytest.approx(20, abs=0.05) and g.std() == pytest.approx(1.0, abs=0.05)
    assert g.min() >= 17 - 1e-9 and g.max() <= 23 + 1e-9
    assert mc.nominal["gain_db"] == pytest.approx(20)


def test_uniform_spread():
    g = monte_carlo(amp(), 4000, {"a.gain_db": 3.0}, dist="uniform").results["gain_db"]
    assert g.std() == pytest.approx(3 / math.sqrt(3), abs=0.05) and g.max() - g.min() > 5.9


def test_yield_of_a_known_distribution():
    # uniform over +/-1 dB about 3 dB: half the runs are at or below 3 dB
    mc = monte_carlo(amp(), 4000, {"a.nf_db": 1.0}, ["nf_db<=3"], dist="uniform")
    assert mc.yield_fraction == pytest.approx(0.5, abs=0.03)
    lo, hi = mc.yield_interval()
    assert lo < 0.5 < hi and hi - lo < 0.04
    assert monte_carlo(amp(), 50, {"a.nf_db": 1.0}, ["nf_db<=9"]).yield_fraction == 1
    assert monte_carlo(amp(), 50, {"a.nf_db": 1.0}, ["nf_db>=9"]).yield_fraction == 0


def test_several_limits_must_all_pass():
    mc = monte_carlo(amp(), 2000, {"a.nf_db": 1.0, "a.gain_db": 1.0}, ["nf_db<=3", "gain_db>=20"], dist="uniform")
    assert mc.yield_fraction == pytest.approx(0.25, abs=0.04)
    assert "nf_db <= 3" in mc.table()


def test_drivers_name_the_parameter_responsible():
    c = amp()
    c.add("b", Attenuator(1))
    c.chain("a", "b")
    mc = monte_carlo(c, 800, {"a.gain_db": 3.0, "b.loss_db": 0.1})
    top, share = mc.drivers("gain_db")[0]
    assert top == "a.gain_db" and share > 0.95


def test_seed_repeats_and_changes():
    a = monte_carlo(amp(), 30, {"a.gain_db": 1.0}, seed=5).results["gain_db"]
    assert np.array_equal(a, monte_carlo(amp(), 30, {"a.gain_db": 1.0}, seed=5).results["gain_db"])
    assert not np.array_equal(a, monte_carlo(amp(), 30, {"a.gain_db": 1.0}, seed=6).results["gain_db"])


def test_workers_give_the_same_answer():
    a = monte_carlo(amp(), 40, {"a.gain_db": 1.0}, workers=1).results["gain_db"]
    b = monte_carlo(amp(), 40, {"a.gain_db": 1.0}, workers=2).results["gain_db"]
    assert np.array_equal(a, b)


def test_floors_keep_a_draw_legal():
    mc = monte_carlo(amp(nf=0.1, vswr=1.05), 300, {"a.nf_db": 1.0, "a.vswr": 1.0}, dist="uniform")
    assert mc.failed == 0 and mc.draws.min() >= 0


def test_frequency_tolerance_with_units_and_a_lost_signal():
    c = Circuit("f")
    c.add("ant", Source(freq="100MHz", power=-60, name="RF"))
    c.add("fl", Filter("bandpass", "butter", 3, f0="100MHz", bw="2MHz"))
    c.chain("ant", "fl")
    mc = monte_carlo(c, 500, {"fl.f0": "1MHz"}, ["gain_db>=-3.1"], dist="uniform")
    assert mc.draws.min() >= 99e6 and mc.draws.max() <= 101e6
    assert mc.results["gain_db"].min() < -2.5 and 0.9 < mc.yield_fraction <= 1


def test_tone_power_path():
    mc = monte_carlo(amp(op1db_dbm=-20), 200, {"ant.tones.0.power": 5.0, "a.gain_db": 0.0})
    assert mc.params[0][1] == -90


@pytest.mark.parametrize("args,match", [
    (dict(tolerances=None), "no tolerances"),
    (dict(tolerances={"nope.x": 1}), "write it block.parameter"),
    (dict(tolerances={"a.oip3_dbm": 1}), "doesn't set"),
    (dict(tolerances={"a.gain_db": -1}), "negative"),
    (dict(tolerances={"a.gain_db": 1}, specs=["junk"]), "figure<=value"),
    (dict(tolerances={"a.gain_db": 1}, specs=["nope<=1"]), "figure<=value"),
    (dict(tolerances={"a.gain_db": 1}, dist="cauchy"), "normal or uniform"),
    (dict(tolerances={"a.gain_db": 1}, runs=1), "at least 2"),
])
def test_bad_requests(args, match):
    with pytest.raises(ValueError, match=match):
        monte_carlo(amp(), **{"runs": 5, **args})


def test_example_cli(capsys):
    assert main(["tolerance", str(EX), "--tone", "RF", "--bw", "200kHz", "--runs", "10", "--workers", "1",
                 "--spec", "nf_db<=3.2"]) == 0
    out = capsys.readouterr().out
    assert "10 runs" in out and "yield" in out and "lna.nf_db" in out
    assert main(["tolerance", str(EX), "--tone", "RF", "--runs", "10", "--workers", "1", "--json",
                 "--tol", "lna.gain_db=0.5"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["figures"]["nf_db"]["mean"] > 0 and any(p["tolerance"] == 0.5 for p in d["params"])
    assert main(["tolerance", str(EX), "--spec", "junk"]) == 1
    assert "figure<=value" in capsys.readouterr().err


def test_analysis_charts_are_cumulative():
    from hetsim.viewer import ViewerModel
    from hetsim.viewer.analysis import analyse, options
    m = ViewerModel.from_file(EX)
    m.sweep_workers = 1
    assert "lna.gain_db" in options(m)["tolerances"]
    r = analyse(m, {"kind": "tolerance", "runs": 6, "tone": "RF", "bw": "200kHz", "specs": ["nf_db<=9"]})
    c = next(x for x in r["charts"] if x["id"] == "nf_db")
    assert c["x"] == sorted(c["x"]) and c["series"][0]["y"][-1] == 100 and "yield 100" in r["text"]
    assert "figure<=value" in analyse(m, {"kind": "tolerance", "runs": 5, "tone": "RF", "specs": ["x"]})["error"]
