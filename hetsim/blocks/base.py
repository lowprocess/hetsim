"""Block base class.

Every block is a set of named ports. The solver hands a block the wave
*incident* on each of its ports and the block returns the wave *leaving* each
port. That symmetric, wave-based interface is what lets leakage travel
backwards (LO radiating out of the antenna) and is the hook for adding port
reflections (VSWR) later: a reflection is just an incident->outgoing term on
the same port.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..spectrum import Spectrum

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

    def respond(self, incident, ctx):
        return {
            "out": incident["in"].shaped(self.s21_db),
            "in": incident["out"].shaped(self.s12_db),
        }
