"""Block base class.

Every block is a set of named ports. The solver hands a block the wave
*incident* on each of its ports and the block returns the wave *leaving* each
port. That symmetric, wave-based interface is what lets leakage travel
backwards (LO radiating out of the antenna). A port mismatch (`vswr`) is an
incident->outgoing term on the same port, which the solvers add for every block.

Blocks also describe themselves to the small-signal analyses (hetsim.smallsignal):
transfers() lists what feeds a port at a frequency, and noise_target() how much noise
the port delivers when everything around it sits at T0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..spectrum import Spectrum
from ..units import parse_db, parse_freq

REGISTRY: dict[str, type] = {}


def register(kind: str):
    def deco(cls):
        cls.kind = kind
        REGISTRY[kind] = cls
        return cls
    return deco


@dataclass
class Context:
    """Per-simulation settings a block may use, plus a warning sink."""
    floor_dbm: float = -200.0          # products below this are dropped
    im_max_tones: int = 24             # strongest N tones considered for 2-tone IM products
    im_window_db: float = 100.0        # ...and only those within this many dB of the strongest
    warn: Callable[[str], None] = field(default=lambda msg: None)


def _gamma2(v) -> float:
    s = parse_db(v)
    if not s >= 1:
        raise ValueError(f"VSWR must be at least 1, not {v!r}")
    return ((s - 1) / (s + 1)) ** 2


def parse_vswr(x, ports) -> dict:
    """`vswr` -> {port: (freqs or None, |gamma|^2)}. A number or a [[freq, vswr], ...] table
    (linear between points, edge values held), for every port or as {port: value}."""
    def one(v):
        if isinstance(v, (list, tuple)):
            try:
                pts = sorted((parse_freq(f), _gamma2(s)) for f, s in v)
            except TypeError:
                raise ValueError(f"a VSWR table is a list of [freq, vswr] pairs, not {v!r}") from None
            if not pts:
                raise ValueError("a VSWR table needs at least one point")
            return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        return None, _gamma2(v)

    if isinstance(x, dict):
        bad = sorted(set(x) - set(ports))
        if bad:
            raise ValueError(f"vswr: no RF port {bad[0]!r}; ports: {list(ports)}")
        return {p: one(v) for p, v in x.items()}
    return {p: one(x) for p in ports}


class Block:
    kind = "block"
    ports: tuple[str, ...] = ()
    controls: tuple[str, ...] = ()   # ports that carry no RF (X/Y plot wiring); wired only to each other
    default_in: str | None = None
    default_out: str | None = None

    def __init__(self, **params):
        # raw parameters, kept verbatim so a netlist round-trips (strings like '10.7MHz' survive)
        self.params = {k: v for k, v in params.items() if v is not None}
        self.name: str | None = None
        vswr = self.params.get("vswr")
        rf = [p for p in self.ports if p not in self.controls]
        self.gamma2 = {} if vswr is None or vswr == "file" else parse_vswr(vswr, rf)

    def reflection(self, port: str, f):
        """|gamma|^2 of a port at frequency f (scalar or array): the power it sends back."""
        g = self.gamma2.get(port)
        if g is None:
            return np.zeros(np.shape(f)) if np.ndim(f) else 0.0
        return np.interp(f, g[0], g[1]) if g[0] is not None else (np.full(np.shape(f), g[1]) if np.ndim(f) else g[1])

    def gamma_c(self, port: str, f):
        """Voltage reflection coefficient for the time-domain solver. No wire has a length, so
        without measured data the phase is zero: reflections add in phase, the worst case."""
        return np.sqrt(self.reflection(port, f))

    def reflected(self, port: str, incident: Spectrum) -> Spectrum:
        """The part of the wave arriving at `port` that comes straight back."""
        def gain(f):
            with np.errstate(divide="ignore"):
                return 10 * np.log10(self.reflection(port, f))
        return incident.shaped(gain)

    # -- small signal (hetsim.smallsignal) ---------------------------------
    def operating_point(self, incident: dict[str, Spectrum]):
        """What the small-signal paths depend on, from the large-signal incident waves."""
        return None

    def transfers(self, port: str, f: float, op) -> list[tuple[str, float, float]]:
        """What feeds `port` at frequency f, reflection excluded: [(input port, input freq, power gain)]."""
        return []

    def noise_target(self, port: str, f: float, op) -> float:
        """Noise leaving `port` (in kT0) with everything around it at T0, reflection excluded. For
        any passive block that is 1 - |gamma|^2: a passive network at T0 is just a load at T0."""
        return 1.0 - float(self.reflection(port, f))

    def noise_emission(self, port: str, f: float, op) -> float:
        """The block's own noise leaving `port` (in kT0): the target less what it passes through."""
        passed = sum(g for _, _, g in self.transfers(port, f, op))
        return max(0.0, self.noise_target(port, f, op) - passed)

    def respond(self, incident: dict[str, Spectrum], ctx: Context) -> dict[str, Spectrum]:
        raise NotImplementedError

    def to_params(self) -> dict:
        """Parameters for writing this block back to a netlist."""
        return dict(self.params)

    def symbols(self) -> list[str]:
        """Fundamental tone symbols this block introduces (sources only)."""
        return []

    def bind(self, name: str) -> None:
        """Called when the block is added to a circuit."""
        self.name = name

    def __repr__(self) -> str:
        p = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({p})"


class TwoPort(Block):
    """Linear 2-port described by forward and reverse gain (dB) vs frequency."""
    ports = ("in", "out")
    default_in = "in"
    default_out = "out"

    def s21_db(self, f):
        raise NotImplementedError

    def s12_db(self, f):
        return self.s21_db(f)  # passive, reciprocal by default

    # complex responses, used by the time-domain solver. Default: zero phase.
    def s21_c(self, f):
        return 10 ** (np.asarray(self.s21_db(f), dtype=float) / 20)

    def s12_c(self, f):
        return 10 ** (np.asarray(self.s12_db(f), dtype=float) / 20)

    def transfers(self, port, f, op):
        db = self.s21_db(f) if port == "out" else self.s12_db(f)
        return [("in" if port == "out" else "out", f, 10 ** (float(np.ravel(db)[0]) / 10))]

    def respond(self, incident, ctx):
        return {
            "out": incident["in"].shaped(self.s21_db),
            "in": incident["out"].shaped(self.s12_db),
        }
