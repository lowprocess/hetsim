"""Three-port mixer in the tone domain.

The ideal mixer (all defaults) is a multiplier: an RF tone at f produces
LO + f and |LO - f| at the IF port, each at P - conversion_loss_db, and
nothing else. Each extra parameter adds one non-ideal effect:

    conversion_loss_db   loss of each 1x1 sideband
    lo_rf_isolation_db   LO leaking out of the RF port (towards the antenna)
    lo_if_isolation_db   LO leaking out of the IF port (and IF<->LO)
    rf_if_isolation_db   RF feedthrough to IF (and IF->RF)
    rf_lo_isolation_db   RF leaking out of the LO port (default: = lo_rf)
    iip3_dbm             two-tone IM3 of RF-port tones, converted to IF
    iip2_dbm             two-tone IM2 of RF-port tones appearing directly at IF
    spur_table           m x n spur chart (preset name, CSV path, dict or SpurTable)
    lo_waveform          'sine' (default) or 'square': a hard-switched mixer, which adds the
                         odd m x 1 products at 20*log10(m) dBc (3x1 at 9.5 dBc...)
    lo_drive_dbm         nominal LO level; a warning is raised if the actual LO
                         is more than lo_drive_tolerance_db away
    nf_db                single-sideband noise figure as on a datasheet (both sidebands open to
                         a load at T0). Default: the conversion loss, as for a passive mixer
    vswr                 port mismatch, as for every block; ports rf, lo, if

The mixer is treated as reciprocal: a tone arriving at the IF port is
converted to the RF port the same way (so it works as an up-converter too, and
reverse leakage from later stages comes back out of the RF port).

The strongest tone arriving at the LO port is taken as the LO; any other tones
there only leak through the isolation paths. Tones at the same frequency from
different products are kept separate and summed by power, not phase.
"""
from __future__ import annotations

import math

import numpy as np

from ..spectrum import Spectrum, vec_comb
from ..units import parse_db
from .base import Block, register
from .nonlinear import nonlinear_products
from .spurtable import PRESETS as SPUR_PRESETS, SpurTable, get_spur_table


@register("mixer")
class Mixer(Block):
    ports = ("rf", "lo", "if")
    default_in = "rf"
    default_out = "if"

    def __init__(self, conversion_loss_db=0.0, lo_rf_isolation_db=None, lo_if_isolation_db=None,
                 rf_if_isolation_db=None, rf_lo_isolation_db=None, iip3_dbm=None, iip2_dbm=None,
                 spur_table=None, lo_drive_dbm=None, lo_drive_tolerance_db=3.0, lo_waveform="sine",
                 max_order=None, nf_db=None, vswr=None):
        raw_table = spur_table.name if isinstance(spur_table, SpurTable) else spur_table
        super().__init__(conversion_loss_db=conversion_loss_db, lo_rf_isolation_db=lo_rf_isolation_db,
                         lo_if_isolation_db=lo_if_isolation_db, rf_if_isolation_db=rf_if_isolation_db,
                         rf_lo_isolation_db=rf_lo_isolation_db, iip3_dbm=iip3_dbm, iip2_dbm=iip2_dbm,
                         spur_table=raw_table, lo_drive_dbm=lo_drive_dbm,
                         lo_drive_tolerance_db=lo_drive_tolerance_db,
                         lo_waveform=None if lo_waveform == "sine" else lo_waveform, max_order=max_order,
                         nf_db=nf_db, vswr=vswr)
        self.nf = parse_db(nf_db) if nf_db is not None else None
        if self.nf is not None and self.nf < 0:
            raise ValueError("nf_db can't be negative")
        d = lambda x: parse_db(x) if x is not None else None  # noqa: E731
        self.cl = parse_db(conversion_loss_db)
        self.lo_rf = d(lo_rf_isolation_db)
        self.lo_if = d(lo_if_isolation_db)
        self.rf_if = d(rf_if_isolation_db)
        self.rf_lo = d(rf_lo_isolation_db) if rf_lo_isolation_db is not None else self.lo_rf
        self.iip3 = d(iip3_dbm)
        self.iip2 = d(iip2_dbm)
        self.table = get_spur_table(spur_table)
        self.lo_drive = d(lo_drive_dbm)
        self.lo_tol = parse_db(lo_drive_tolerance_db)
        self.lo_waveform = lo_waveform
        if lo_waveform == "square":
            # hard switching: odd LO harmonics at 1/m, i.e. m x 1 products 20*log10(m) below 1 x 1
            for m in (3, 5, 7, 9):
                self.table.entries.setdefault((m, 1), 20 * math.log10(m))
        elif lo_waveform != "sine":
            raise ValueError("lo_waveform must be 'sine' or 'square'")
        # depth: keep m x n spur products with m + n <= max_order (2 = the 1 x 1 product only)
        if max_order is not None and (int(max_order) != max_order or max_order < 2):
            raise ValueError(f"max_order must be a whole number >= 2, not {max_order!r}")
        self.max_order = int(max_order) if max_order is not None else None

    def _side_db(self, c: int) -> float:
        """Gain of a product whose positive-frequency form carries c times the LO. All of it here;
        an I/Q mixer's two branches add or cancel according to c."""
        return 0.0

    def _side_phase(self, c: int) -> float:
        """Phase the same combination adds, in radians."""
        return 0.0

    def spur_entries(self) -> dict:
        """The spur table entries {(m, n): dBc} this mixer generates, after max_order."""
        return {mn: v for mn, v in self.table.entries.items()
                if self.max_order is None or mn[0] + mn[1] <= self.max_order}

    def to_params(self) -> dict:
        p = dict(self.params)
        raw = p.get("spur_table")
        if raw is not None and not (isinstance(raw, str) and (raw in SPUR_PRESETS or raw.endswith(".csv"))):
            p["spur_table"] = {"name": self.table.name, "ref_rf_dbm": self.table.ref_rf_dbm,
                               "entries": {f"{m},{n}": v for (m, n), v in sorted(self.table.entries.items())}}
        return p

    # ------------------------------------------------------------------
    def _convert(self, inp: Spectrum, lo, out: Spectrum, ctx) -> None:
        """Mix the tones of `inp` with the LO into `out` (spur table + IM)."""
        vlo, flo, plo = lo.vec, lo.freq, lo.phase
        floor = ctx.floor_dbm
        entries = list(self.spur_entries())

        def put(v, f, lvl, c, phase, scale=None):
            # the side gain belongs to the positive-frequency form; add() flips a negative one
            s = self._side_phase(c)
            out.add(v, f, lvl + self._side_db(c), scale=scale, phase=phase + s if f >= 0 else phase - s)

        for t in inp:
            p = t.power_dbm
            for m, n in entries:
                lvl = self.table.level(m, n, p, self.cl)
                if lvl is None or lvl < floor:
                    continue
                if m == 0:  # RF harmonic straight through to the output
                    put(vec_comb(t.vec, n, (), 0), n * t.freq, lvl, 0, n * t.phase)
                    continue
                put(vec_comb(vlo, m, t.vec, n), m * flo + n * t.freq, lvl, m, m * plo + n * t.phase)
                diff = m * flo - n * t.freq
                put(vec_comb(vlo, m, t.vec, -n), diff, lvl, m if diff > 0 else -m, m * plo - n * t.phase, scale=m * flo)
        if self.iip3 is not None:
            for t in nonlinear_products(inp, ctx, ip3=self.iip3, harmonics=False):
                lvl = t.power_dbm - self.cl
                if lvl >= floor:
                    put(vec_comb(vlo, 1, t.vec, 1), flo + t.freq, lvl, 1, plo + t.phase)
                    diff = flo - t.freq
                    put(vec_comb(vlo, 1, t.vec, -1), diff, lvl, 1 if diff > 0 else -1, plo - t.phase, scale=flo)
        if self.iip2 is not None:
            out.add_spectrum(nonlinear_products(inp, ctx, ip2=self.iip2, harmonics=False), -self.cl + self._side_db(0),
                             phase=self._side_phase(0))

    def respond(self, incident, ctx):
        rf, lo_in, if_in = incident["rf"], incident["lo"], incident["if"]
        out_rf, out_lo, out_if = Spectrum(), Spectrum(), Spectrum()

        # isolation / leakage paths
        if self.lo_if is not None:
            out_if.add_spectrum(lo_in, -self.lo_if)
            out_lo.add_spectrum(if_in, -self.lo_if)
        if self.lo_rf is not None:
            out_rf.add_spectrum(lo_in, -self.lo_rf)
        if self.rf_lo is not None:
            out_lo.add_spectrum(rf, -self.rf_lo)
        if self.rf_if is not None:
            out_if.add_spectrum(rf, -self.rf_if)
            out_rf.add_spectrum(if_in, -self.rf_if)

        los = lo_in.strongest(1)
        if not los:
            if len(rf) or len(if_in):
                ctx.warn("no LO drive at the LO port; only leakage is produced")
            return {"rf": out_rf, "lo": out_lo, "if": out_if}
        lo = los[0]
        if self.lo_drive is not None and abs(lo.power_dbm - self.lo_drive) > self.lo_tol:
            ctx.warn(f"LO drive {lo.power_dbm:.1f} dBm vs nominal {self.lo_drive:.1f} dBm; "
                     f"conversion loss and spurs will differ from the datasheet")

        self._convert(rf, lo, out_if, ctx)      # down-conversion (RF -> IF)
        self._convert(if_in, lo, out_rf, ctx)   # reciprocal path (IF -> RF)
        return {"rf": out_rf, "lo": out_lo, "if": out_if}

    # -- coherent solver ----------------------------------------------------
    def linear_c(self, port, f, op):
        leak = {"if": (("lo", self.lo_if), ("rf", self.rf_if)),
                "rf": (("lo", self.lo_rf), ("if", self.rf_if)),
                "lo": (("if", self.lo_if), ("rf", self.rf_lo))}[port]
        return [(q, np.full(np.shape(f), 10 ** (-iso / 20) + 0j)) for q, iso in leak if iso is not None]

    def generate(self, incident, op, ctx):
        los = incident["lo"].strongest(1)
        out_rf, out_if = Spectrum(), Spectrum()
        if not los:
            if len(incident["rf"]) or len(incident["if"]):
                ctx.warn("no LO drive at the LO port; only leakage is produced")
            return {}
        lo = los[0]
        if self.lo_drive is not None and abs(lo.power_dbm - self.lo_drive) > self.lo_tol:
            ctx.warn(f"LO drive {lo.power_dbm:.1f} dBm vs nominal {self.lo_drive:.1f} dBm; "
                     f"conversion loss and spurs will differ from the datasheet")
        self._convert(incident["rf"], lo, out_if, ctx)
        self._convert(incident["if"], lo, out_rf, ctx)
        return {"rf": out_rf, "if": out_if}

    def conversions_c(self, port, f, flo):
        if port == "lo" or flo is None:
            return []
        q = "rf" if port == "if" else "if"
        out = []
        for (m, n), dbc in self.spur_entries().items():
            if n != 1 or m == 0:
                continue
            g = 10 ** (-(self.cl + dbc) / 20)
            mf = m * flo
            # f = fi - m*fLO keeps the input's phase; f = m*fLO + fi too; f = m*fLO - fi reverses it
            for fi, c, conj in ((mf + f, -m, False), (f - mf, m, False), (mf - f, m, True)):
                side = g * 10 ** (self._side_db(c) / 20) * np.exp(1j * self._side_phase(c))
                if fi > 0 and abs(side) > 0:
                    out.append((q, fi, complex(side), conj))
        return out

    # -- small signal -------------------------------------------------------
    def linearity(self):
        return self.iip3, None

    def operating_point(self, incident):
        """The LO frequency, or None without one."""
        los = incident["lo"].strongest(1)
        return los[0].freq if los else None

    def transfers(self, port, f, flo):
        leak = {"if": (("lo", self.lo_if), ("rf", self.rf_if)),
                "rf": (("lo", self.lo_rf), ("if", self.rf_if)),
                "lo": (("if", self.lo_if), ("rf", self.rf_lo))}[port]
        out = [(q, f, 10 ** (-iso / 10)) for q, iso in leak if iso is not None]
        if port == "lo" or flo is None:
            return out
        q = "rf" if port == "if" else "if"
        # noise is far below any reference level, so only the n = 1 products (linear in the input) carry it
        for (m, n), dbc in self.spur_entries().items():
            if n != 1 or m == 0:
                continue
            g = 10 ** (-(self.cl + dbc) / 10)
            # the output at f comes from an input at m*fLO + f (then f = fi - m*fLO, carrying -m LOs)
            # and from |m*fLO - f| (then f = m*fLO +/- fi, carrying +m)
            for fi, c in ((m * flo + f, -m), (abs(m * flo - f), m)):
                side = g * 10 ** (self._side_db(c) / 10)
                if fi > 0 and side > 0:
                    out.append((q, fi, side))
        return out

    def noise_target(self, port, f, flo):
        if port != "lo" and self.nf is not None and flo is not None:
            return 10 ** ((self.nf - self.cl) / 10)
        return super().noise_target(port, f, flo)


@register("iq_mixer")
class IQMixer(Mixer):
    """A quadrature mixer pair and the 90 degree combination of its branches, as one block.

    An image-reject (Hartley) mixer, or the analogue half of a low-IF or complex zero-IF receiver, where
    the combination happens digitally. It takes everything a mixer does, and the same conversion loss:

        sideband             'upper' keeps RF above the LO (RF - LO), 'lower' keeps RF below it (LO - RF)
        gain_imbalance_db    Q branch against I
        phase_imbalance_deg  error of the 90 degrees

    The other sideband comes through down by the classic image rejection,
        IRR = (1 + 2 g cos(phi) + g^2) / (1 - 2 g cos(phi) + g^2),
    which is infinite until an imbalance is given. Each product's Q branch sits c*90 degrees from its I
    branch, for c LOs in it, and the combination adds another 90: so 3 x 1 spurs favour the opposite
    sideband, and what doesn't pass through the LO, such as IM2, adds in quadrature, 3 dB down.
    Isolation paths are taken as given. A product landing exactly on DC is dropped, as everywhere.
    """

    def __init__(self, conversion_loss_db=0.0, sideband="upper", gain_imbalance_db=0.0, phase_imbalance_deg=0.0,
                 lo_rf_isolation_db=None, lo_if_isolation_db=None, rf_if_isolation_db=None, rf_lo_isolation_db=None,
                 iip3_dbm=None, iip2_dbm=None, spur_table=None, lo_drive_dbm=None, lo_drive_tolerance_db=3.0,
                 lo_waveform="sine", max_order=None, nf_db=None, vswr=None):
        super().__init__(conversion_loss_db, lo_rf_isolation_db, lo_if_isolation_db, rf_if_isolation_db,
                         rf_lo_isolation_db, iip3_dbm, iip2_dbm, spur_table, lo_drive_dbm, lo_drive_tolerance_db,
                         lo_waveform, max_order, nf_db, vswr)
        if sideband not in ("upper", "lower"):
            raise ValueError("sideband must be 'upper' or 'lower'")
        self.sideband = sideband
        self.gain_imb = parse_db(gain_imbalance_db)
        self.phase_imb = parse_db(phase_imbalance_deg)
        self.params.update(sideband=sideband, gain_imbalance_db=gain_imbalance_db,
                           phase_imbalance_deg=phase_imbalance_deg)

    def _side_db(self, c: int) -> float:
        hilbert = -90.0 if self.sideband == "upper" else 90.0
        theta = math.radians(-c * 90.0 + hilbert + self.phase_imb)
        g = 10 ** (self.gain_imb / 20)
        p = abs(1 + g * complex(math.cos(theta), math.sin(theta))) ** 2 / 4
        return 10 * math.log10(p) if p > 1e-30 else -math.inf

    def _side_phase(self, c: int) -> float:
        hilbert = -90.0 if self.sideband == "upper" else 90.0
        theta = math.radians(-c * 90.0 + hilbert + self.phase_imb)
        g = 10 ** (self.gain_imb / 20)
        z = 1 + g * complex(math.cos(theta), math.sin(theta))
        return math.atan2(z.imag, z.real)

    def image_rejection_db(self) -> float:
        """Wanted sideband over the other, from the imbalances alone."""
        c = -1 if self.sideband == "upper" else 1
        return self._side_db(c) - self._side_db(-c)
