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

The mixer is treated as reciprocal: a tone arriving at the IF port is
converted to the RF port the same way (so it works as an up-converter too, and
reverse leakage from later stages comes back out of the RF port).

The strongest tone arriving at the LO port is taken as the LO; any other tones
there only leak through the isolation paths. Tones at the same frequency from
different products are kept separate and summed by power, not phase.
Port mismatch (VSWR) is not modelled yet.
"""
from __future__ import annotations

import math

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
                 max_order=None):
        raw_table = spur_table.name if isinstance(spur_table, SpurTable) else spur_table
        super().__init__(conversion_loss_db=conversion_loss_db, lo_rf_isolation_db=lo_rf_isolation_db,
                         lo_if_isolation_db=lo_if_isolation_db, rf_if_isolation_db=rf_if_isolation_db,
                         rf_lo_isolation_db=rf_lo_isolation_db, iip3_dbm=iip3_dbm, iip2_dbm=iip2_dbm,
                         spur_table=raw_table, lo_drive_dbm=lo_drive_dbm,
                         lo_drive_tolerance_db=lo_drive_tolerance_db,
                         lo_waveform=None if lo_waveform == "sine" else lo_waveform, max_order=max_order)
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
        vlo, flo = lo.vec, lo.freq
        floor = ctx.floor_dbm
        entries = list(self.spur_entries())
        for t in inp:
            p = t.power_dbm
            for m, n in entries:
                lvl = self.table.level(m, n, p, self.cl)
                if lvl is None or lvl < floor:
                    continue
                if m == 0:  # RF harmonic straight through to the output
                    out.add(vec_comb(t.vec, n, (), 0), n * t.freq, lvl)
                    continue
                out.add(vec_comb(vlo, m, t.vec, n), m * flo + n * t.freq, lvl)
                out.add(vec_comb(vlo, m, t.vec, -n), m * flo - n * t.freq, lvl, scale=m * flo)
        if self.iip3 is not None:
            for t in nonlinear_products(inp, ctx, ip3=self.iip3, harmonics=False):
                lvl = t.power_dbm - self.cl
                if lvl >= floor:
                    out.add(vec_comb(vlo, 1, t.vec, 1), flo + t.freq, lvl)
                    out.add(vec_comb(vlo, 1, t.vec, -1), flo - t.freq, lvl, scale=flo)
        if self.iip2 is not None:
            out.add_spectrum(nonlinear_products(inp, ctx, ip2=self.iip2, harmonics=False), -self.cl)

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
