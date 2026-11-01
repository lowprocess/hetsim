"""Signal sources and terminations."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..spectrum import Spectrum, vec_of
from ..units import parse_db, parse_freq
from .base import Block, register


MODES = ("cw", "tune", "sweep", "wideband")
SPACINGS = ("lin", "log")
SWEEP_POINTS = 41      # default steps of a swept tone
WIDEBAND_POINTS = 21   # default tones in a wideband comb


@dataclass
class ToneSpec:
    freq: float
    power: float
    name: str | None = None
    phase: float | None = None   # degrees; used by the time-domain solver (None = fixed pseudo-random)
    band: tuple[float, float] | None = None   # (low, high) edges, Hz
    mode: str = "cw"             # what the band means; see Source
    points: int | None = None    # sweep steps / comb tones
    step: float | None = None    # comb spacing (wideband), instead of points
    mute: bool = False           # kept in the netlist but not emitted
    spacing: str = "lin"         # sweep steps / comb tones: even ('lin') or geometric ('log')
    phase_noise: tuple | None = None   # ((offset Hz, dBc/Hz), ...), offsets increasing
    member: "ToneSpec | None" = field(default=None, repr=False, compare=False)   # comb tone: its wideband spec
    index: int = 0               # comb tone: position in the comb

    @classmethod
    def from_dict(cls, t: dict) -> "ToneSpec":
        band = parse_band(t.get("band"))
        mode = t.get("mode") or ("tune" if band else "cw")
        if mode not in MODES:
            raise ValueError(f"tone mode {mode!r}: use one of {', '.join(MODES)}")
        if mode == "cw" and band:
            raise ValueError("a 'cw' tone takes no band; use mode tune, sweep or wideband")
        if mode != "cw" and not band:
            raise ValueError(f"tone mode {mode!r} needs a band, e.g. band: [88MHz, 108MHz]")
        freq = t.get("freq")
        if freq is None:
            if not band:
                raise ValueError("a tone needs a freq (or a band)")
            freq = 0.5 * (band[0] + band[1])
        freq = parse_freq(freq)
        if mode in ("tune", "sweep") and not band[0] <= freq <= band[1]:
            raise ValueError(f"freq {freq:g} Hz is outside the band {band[0]:g}-{band[1]:g} Hz")
        points = t.get("points")
        if points is not None and (int(points) != points or points < 2):
            raise ValueError(f"points must be a whole number >= 2, not {points!r}")
        step = parse_freq(t.get("step"))
        if step is not None and step <= 0:
            raise ValueError("step must be positive")
        spacing = t.get("spacing") or "lin"
        if spacing not in SPACINGS:
            raise ValueError(f"spacing {spacing!r}: use lin or log")
        if spacing == "log" and step is not None:
            raise ValueError("a log-spaced band takes points, not step")
        pn = parse_phase_noise(t.get("phase_noise"))
        if pn and mode == "wideband":
            raise ValueError("phase_noise applies to a single tone, not a wideband comb")
        return cls(freq, parse_db(t.get("power", 0)), t.get("name"), t.get("phase"), band, mode,
                   int(points) if points is not None else None, step, bool(t.get("mute", False)), spacing, pn)

    def pn_dbc(self, offset: float) -> float | None:
        """Single-sideband phase noise at `offset` Hz (dBc/Hz), log-interpolated; None outside the profile."""
        if not self.phase_noise:
            return None
        f = [p[0] for p in self.phase_noise]
        offset = abs(offset)
        if not f[0] <= offset <= f[-1]:
            return None
        if len(f) == 1:
            return self.phase_noise[0][1]
        x = math.log10(offset)
        for (f0, l0), (f1, l1) in zip(self.phase_noise, self.phase_noise[1:]):
            if offset <= f1:
                return l0 + (l1 - l0) * (x - math.log10(f0)) / (math.log10(f1) - math.log10(f0))
        return self.phase_noise[-1][1]

    def grid(self) -> list[float]:
        """Frequencies of a sweep or comb across the band."""
        default = SWEEP_POINTS if self.mode == "sweep" else WIDEBAND_POINTS
        return band_grid(self.band, self.points or default, self.step, self.spacing)

    def to_dict(self) -> dict:
        d = {"name": self.name, "freq": self.freq, "power": self.power}
        if self.phase is not None:
            d["phase"] = self.phase
        if self.band:
            d.update(band=list(self.band), mode=self.mode)
            if self.points is not None:
                d["points"] = self.points
            if self.step is not None:
                d["step"] = self.step
            if self.spacing != "lin":
                d["spacing"] = self.spacing
        if self.mute:
            d["mute"] = True
        if self.phase_noise:
            d["phase_noise"] = [list(p) for p in self.phase_noise]
        return d


def parse_phase_noise(x) -> tuple | None:
    """[[offset, dBc/Hz], ...] -> ((Hz, dBc/Hz), ...), checked: offsets positive and increasing."""
    if not x:
        return None
    try:
        pts = tuple((parse_freq(o), parse_db(l)) for o, l in x)
    except (TypeError, ValueError):
        raise ValueError(f"phase_noise must be a list of [offset, dBc/Hz] pairs, not {x!r}") from None
    if any(o <= 0 for o, _ in pts) or any(b[0] <= a[0] for a, b in zip(pts, pts[1:])):
        raise ValueError("phase_noise offsets must be positive and increasing")
    return pts


def parse_band(x) -> tuple[float, float] | None:
    """[low, high] or 'low:high' -> (low, high) in Hz."""
    if x is None:
        return None
    parts = x.split(":") if isinstance(x, str) else list(x)
    if len(parts) != 2:
        raise ValueError(f"band must be [low, high] or 'low:high', not {x!r}")
    lo, hi = parse_freq(parts[0]), parse_freq(parts[1])
    if not 0 < lo < hi:
        raise ValueError(f"band {x!r}: need 0 < low < high")
    return lo, hi


def band_grid(band, points: int, step: float | None = None, spacing: str = "lin") -> list[float]:
    """`points` frequencies from band edge to edge (evenly, or geometrically for 'log'),
    or every `step` from the low edge."""
    lo, hi = band
    if step:
        n = int(math.floor((hi - lo) / step + 1e-9)) + 1
        return [lo + k * step for k in range(n)]
    if spacing == "log":
        return [lo * (hi / lo) ** (k / (points - 1)) for k in range(points)]
    return [lo + (hi - lo) * k / (points - 1) for k in range(points)]


@register("source")
class Source(Block):
    """One or more CW tones (an antenna input, a test generator, an LO).

    Each tone is a fundamental with its own symbol, used in product labels
    ('LO - RF'). Symbol defaults to the block name (single tone) or
    '<block>_<i>'.

        Source(freq="100MHz", power=-60)                 # one tone
        Source(tones=[{"name": "RF", "freq": 100e6, "power": -60},
                      {"name": "BLK", "freq": 100.4e6, "power": -30}])
        Source(freq="110.7MHz", power=7, harmonics_dbc=[-25, -35])   # LO with 2H, 3H

    A tone can carry a band, and its `mode` says what the band means:

        tune      (default with a band) one CW tone at `freq` (default: band centre);
                  the band is its tuning range, which the viewer's slider keeps to
        sweep     like tune for a single simulation, but Circuit.simulate_sweep()
                  steps it across the band in `points` steps (default 41). All
                  sweep tones in a circuit step together, so an LO given its own
                  band tracks the RF
        wideband  `points` tones (default 21), or one every `step`, spread across
                  the band and sharing `power` equally: a modulated or noise-like
                  signal. Comb tones are named '<name>~1', '<name>~2'...

    `spacing: log` places sweep steps or comb tones geometrically across the band
    (equal ratios rather than equal Hz), for bands spanning decades.

    `mute: true` keeps a tone in the netlist but stops the source emitting it.

    `phase_noise: [[1kHz, -80], [100kHz, -120], [1MHz, -140]]` gives a tone single-sideband
    phase noise (dBc/Hz at each offset, log-interpolated, none outside the listed offsets). The
    time-domain solver modulates the tone's phase with it; analysis.phase_noise_skirts() gives
    the tone solver's view (products' noise density, reciprocal mixing included).

        Source(tones=[{"name": "RF", "band": ["88MHz", "108MHz"], "mode": "sweep", "power": -60}])

    Waves arriving back at the source port (e.g. LO leaking out of the
    antenna) are absorbed, apart from what `vswr` reflects, and can be inspected
    with Result.incident().
    """
    ports = ("out", "sweep")
    controls = ("sweep",)   # the swept tone's frequency, for an X/Y plot's x
    default_out = "out"

    def __init__(self, freq=None, power=None, name=None, tones=None, harmonics_dbc=None, phase=None, vswr=None):
        super().__init__(freq=freq, power=power, name=name, tones=tones, harmonics_dbc=harmonics_dbc, phase=phase,
                         vswr=vswr)
        specs = []
        if tones:
            for i, t in enumerate(tones):
                try:
                    specs.append(t if isinstance(t, ToneSpec) else ToneSpec.from_dict(t))
                except ValueError as e:
                    raise ValueError(f"tone {t.get('name') or i}: {e}") from None
        if freq is not None:
            specs.insert(0, ToneSpec(parse_freq(freq), parse_db(power if power is not None else 0), name, phase))
        if not specs:
            raise ValueError("Source needs freq/power or tones")
        self.specs: list[ToneSpec] = specs      # as written in the netlist
        self.tones: list[ToneSpec] = []         # what the source emits: a wideband spec becomes its comb
        for sp in specs:
            if sp.mute:
                continue
            if sp.mode == "wideband":
                grid = sp.grid()
                p_each = sp.power - 10 * math.log10(len(grid))
                self.tones += [ToneSpec(f, p_each, member=sp, index=k) for k, f in enumerate(grid)]
            else:
                self.tones.append(sp)
        self.harmonics_dbc = [parse_db(h) for h in (harmonics_dbc or [])]

    def bind(self, name):
        super().bind(name)
        single = len(self.specs) == 1
        for i, t in enumerate(self.specs):
            if t.name is None:
                t.name = name if single else f"{name}_{i}"
        for t in self.tones:
            if t.member is not None:
                t.name = f"{t.member.name}~{t.index + 1}"

    def sweep_specs(self) -> list[ToneSpec]:
        return [t for t in self.specs if t.mode == "sweep" and not t.mute]

    def symbols(self):
        return [t.name for t in self.tones]

    def set(self, index: int = 0, freq=None, power=None):
        """Change a tone in place (used by sweeps)."""
        if freq is not None:
            self.tones[index].freq = parse_freq(freq)
        if power is not None:
            self.tones[index].power = parse_db(power)

    def to_params(self) -> dict:
        p = {"tones": [t.to_dict() for t in self.specs]}
        if self.harmonics_dbc:
            p["harmonics_dbc"] = list(self.harmonics_dbc)
        if "vswr" in self.params:
            p["vswr"] = self.params["vswr"]
        return p

    def respond(self, incident, ctx):
        s = Spectrum()
        for t in self.tones:
            s.add(vec_of(t.name), t.freq, t.power)
            for k, dbc in enumerate(self.harmonics_dbc, start=2):
                s.add(vec_of(t.name, k), k * t.freq, t.power + dbc)
        return {"out": s}


@register("load")
class Load(Block):
    """Termination / probe point. Absorbs whatever arrives, apart from what `vswr` reflects."""
    ports = ("in",)
    default_in = "in"

    def __init__(self, vswr=None):
        super().__init__(vswr=vswr)

    def respond(self, incident, ctx):
        return {"in": Spectrum()}
