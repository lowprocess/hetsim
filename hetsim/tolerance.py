"""Monte Carlo of the receiver budget: how much of a build meets its limits, and which tolerance decides.

Tolerances live on the blocks in a netlist, like limits, and the solvers ignore them:

    lna: {type: amplifier, gain_db: 18, nf_db: 1.2, tolerance: {gain_db: 1.5, nf_db: 0.3}}
    iff: {type: filter, ..., f0: 10.7MHz, tolerance: {f0: 20kHz}}

Each amount is +/- about the nominal. With dist 'normal' it is 3 sigma, cut off there, so no run
goes outside the tolerance; with 'uniform' every value within it is equally likely. Every run gets
its own draw, and the seed makes a set of runs repeatable whatever the number of workers.
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field

import numpy as np

from .budget import METRICS, SHORT, _set_path, receiver_budget
from .circuit import Circuit, Settings
from .units import parse_db, parse_freq

_FREQ = {"fc", "f0", "bw", "f1", "f2", "fmin", "fmax", "freq", "sample_rate", "analog_bw"}
_FLOORS = {"vswr": 1.0, "nf_db": 0.0}   # a draw can't take these past what the block accepts
_SPEC = re.compile(r"^\s*(\w+)\s*(<=|>=)\s*(.+?)\s*$")


def _value(key: str, x) -> float:
    return parse_freq(x) if key.split(".")[-1] in _FREQ else parse_db(x)


def tolerances_of(circuit: Circuit) -> dict:
    """{'block.param': amount} from the netlist."""
    return {f"{n}.{k}": v for n, b in circuit.blocks.items() for k, v in (getattr(b, "tolerance", None) or {}).items()}


def _get_path(spec: dict, keys: list):
    node = spec
    for k in keys:
        node = node[int(k)] if isinstance(node, list) else node[k]
    return node


def _parse_specs(specs) -> list:
    out = []
    for s in specs or ():
        m = _SPEC.match(s) if isinstance(s, str) else None
        if not m or m.group(1) not in METRICS:
            raise ValueError(f"limit {s!r}: write figure<=value or figure>=value, with a figure from "
                             f"{', '.join(METRICS)}")
        out.append((m.group(1), m.group(2), parse_db(m.group(3))))
    return out


def _worker(netlist: dict, kw: dict, draws: list) -> list:
    """Run budgets for a chunk of draws. Module-level, so it can go to a worker process."""
    from .netlist import from_dict
    rows = []
    for draw in draws:
        d = copy.deepcopy(netlist)
        for param, value in draw:
            block, _, path = param.partition(".")
            _set_path(d["blocks"][block], path.split("."), value)
        try:
            c, st = from_dict(d)
            b = receiver_budget(c, settings=st, **kw)
            unsettled = any(w.startswith("[<solver>]") for w in b.warnings)    # its figures mean nothing
            rows.append(None if unsettled else {k: float(getattr(b, k)) for k in METRICS})
        except ValueError:
            rows.append(None)   # this draw leaves no path, or a loop with gain: it counts as a failure
    return rows


@dataclass
class MonteCarlo:
    runs: int
    dist: str
    seed: int
    params: list            # (name, nominal, amount)
    draws: np.ndarray       # runs x params
    results: dict           # metric -> array of runs; NaN where a run gave no answer
    nominal: dict           # metric -> value with every parameter at nominal
    specs: list = field(default_factory=list)
    failed: int = 0

    def _ok(self, metric: str) -> np.ndarray:
        x = self.results[metric]
        return x[np.isfinite(x)]

    def stats(self, metric: str) -> dict | None:
        x = self._ok(metric)
        if not len(x):
            return None
        return {"nominal": self.nominal.get(metric), "mean": float(x.mean()), "std": float(x.std()),
                "min": float(x.min()), "p5": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)),
                "max": float(x.max())}

    def passes(self) -> np.ndarray:
        """Which runs meet every limit. A run with no answer, or with no number for a limit, fails."""
        ok = np.ones(self.runs, bool)
        with np.errstate(invalid="ignore"):
            for m, op, lim in self.specs:
                ok &= self.results[m] <= lim if op == "<=" else self.results[m] >= lim
        return ok

    @property
    def yield_fraction(self) -> float | None:
        return float(self.passes().mean()) if self.specs else None

    def yield_interval(self) -> tuple[float, float] | None:
        """95% Wilson interval: what the yield could be, given only this many runs."""
        if not self.specs:
            return None
        p, n, z = self.yield_fraction, self.runs, 1.96
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return max(0.0, c - h), min(1.0, c + h)

    def drivers(self, metric: str, top: int = 3) -> list:
        """The parameters behind a figure's spread: (name, share of its variance), largest first, from the
        squared correlation. Rough where a response is strongly nonlinear, such as a filter edge."""
        y = self.results[metric]
        ok = np.isfinite(y)
        if ok.sum() < 3 or np.ptp(y[ok]) < 1e-12:
            return []
        out = []
        for j, (name, _, _) in enumerate(self.params):
            x = self.draws[ok, j]
            if np.ptp(x) > 0:
                out.append((name, float(np.corrcoef(x, y[ok])[0, 1] ** 2)))
        return sorted(out, key=lambda r: -r[1])[:top]

    def cdf(self, metric: str):
        """(sorted values, % of runs with an answer at or below each)."""
        x = np.sort(self._ok(metric))
        return x.tolist(), (100 * np.arange(1, len(x) + 1) / max(len(x), 1)).tolist()

    def table(self) -> str:
        lines = [f"{self.runs} runs, {len(self.params)} tolerances, {self.dist}, seed {self.seed}", "",
                 f"{'figure':<8} {'nominal':>8} {'mean':>8} {'std':>6} {'min':>8} {'5%':>8} {'95%':>8} {'max':>8}  driven by"]
        for k in METRICS:
            s = self.stats(k)
            if s is None:
                continue
            why = ", ".join(f"{n} {100 * r:.0f}%" for n, r in self.drivers(k) if r >= 0.05)
            nom = "-" if s["nominal"] is None or math.isinf(s["nominal"]) else f"{s['nominal']:.2f}"
            lines.append(f"{SHORT[k]:<8} {nom:>8} {s['mean']:8.2f} {s['std']:6.2f} {s['min']:8.2f} {s['p5']:8.2f} "
                         f"{s['p95']:8.2f} {s['max']:8.2f}  {why}".rstrip())
        if self.specs:
            lo, hi = self.yield_interval()
            lines += ["", f"yield {100 * self.yield_fraction:.1f}% (95% interval {100 * lo:.1f} to {100 * hi:.1f}%)"]
            for m, op, lim in self.specs:
                with np.errstate(invalid="ignore"):
                    hit = self.results[m] <= lim if op == "<=" else self.results[m] >= lim
                lines.append(f"  {m} {op} {lim:g}: {100 * hit.mean():.1f}% of runs")
        if self.failed:
            lines += ["", f"{self.failed} of {self.runs} runs gave no answer (the signal lost, or a loop with gain); "
                          "they count as failures"]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        lo_hi = self.yield_interval()
        return {"runs": self.runs, "dist": self.dist, "seed": self.seed, "failed": self.failed,
                "params": [{"name": n, "nominal": v, "tolerance": a} for n, v, a in self.params],
                "figures": {k: {**(self.stats(k) or {}), "drivers": self.drivers(k)} for k in METRICS},
                "yield": self.yield_fraction, "yield_interval": list(lo_hi) if lo_hi else None}


def monte_carlo(circuit: Circuit, runs: int = 500, tolerances: dict | None = None, specs=(), dist: str = "normal",
                seed: int = 1, settings: Settings | None = None, workers: int | None = 1, **budget_kw) -> MonteCarlo:
    """Repeat receiver_budget() with each toleranced parameter drawn at random.

    tolerances: {'block.param': amount} added to, and overriding, those in the netlist.
    specs: limits for the yield, like 'nf_db<=4' or 'sfdr_db>=80'. Other keywords go to receiver_budget.
    workers: processes; None uses one per physical core. Above 1, a calling script needs the usual
    `if __name__ == "__main__":` guard, and if the workers can't start the runs stay in-process.
    """
    from .analysis import POOL_SIZE, _pool, _pool_failed
    from .netlist import to_dict
    if dist not in ("normal", "uniform"):
        raise ValueError(f"dist must be normal or uniform, not {dist!r}")
    if runs < 2:
        raise ValueError("a Monte Carlo needs at least 2 runs")
    limits = _parse_specs(specs)
    tols = {**tolerances_of(circuit), **(tolerances or {})}
    if not tols:
        raise ValueError("no tolerances: give a block tolerance: {param: amount} in the netlist, or pass some")
    base = to_dict(circuit, settings)
    params = []
    for name, amount in tols.items():
        block, _, path = name.partition(".")
        if block not in base["blocks"] or not path:
            raise ValueError(f"tolerance on {name!r}: write it block.parameter, with a block in this circuit")
        try:
            raw = _get_path(base["blocks"][block], path.split("."))
        except (KeyError, IndexError, ValueError):
            raise ValueError(f"tolerance on {name!r}: the netlist doesn't set that parameter") from None
        if isinstance(raw, (dict, list)):
            raise ValueError(f"tolerance on {name!r}: only a single number can vary")
        try:
            a = _value(path, amount)
            params.append((name, _value(path, raw), a))
        except ValueError as e:
            raise ValueError(f"tolerance on {name!r}: {e}") from None
        if a < 0:
            raise ValueError(f"tolerance on {name!r} can't be negative")

    rng = np.random.default_rng(seed)
    if dist == "normal":
        z = np.clip(rng.standard_normal((runs, len(params))) / 3, -1.0, 1.0)
    else:
        z = rng.uniform(-1.0, 1.0, (runs, len(params)))
    draws = np.empty((runs, len(params)))
    for j, (name, nom, a) in enumerate(params):
        floor = _FLOORS.get(name.split(".")[-1], -math.inf)
        draws[:, j] = np.maximum(nom + a * z[:, j], floor)

    names = [p[0] for p in params]
    all_draws = [[(names[j], float(draws[i, j])) for j in range(len(names))] for i in range(runs)]
    workers = min(POOL_SIZE if workers is None else workers, POOL_SIZE, runs)
    rows = None
    if workers > 1:
        size = -(-runs // (workers * 4))
        chunks = [all_draws[k:k + size] for k in range(0, runs, size)]
        try:
            futs = [_pool().submit(_worker, base, budget_kw, ch) for ch in chunks]
            rows = [r for f in futs for r in f.result()]
        except ValueError:
            raise
        except Exception as e:  # noqa: BLE001  - a broken pool must not lose the analysis
            _pool_failed(e)
    if rows is None:
        rows = _worker(base, budget_kw, all_draws)

    nominal = _worker(base, budget_kw, [[]])[0]
    if nominal is None:     # the circuit as drawn doesn't work: say why, rather than report only failures
        from .netlist import from_dict
        c, st = from_dict(base)
        receiver_budget(c, settings=st, **budget_kw)
    results = {k: np.array([r[k] if r is not None and math.isfinite(r[k]) else math.nan for r in rows]) for k in METRICS}
    return MonteCarlo(runs, dist, seed, params, draws, results, nominal or {}, limits, sum(r is None for r in rows))
