"""Polynomial (memoryless, weakly nonlinear) products on a tone set, and the amplifier block.

For a third-order nonlinearity with intercept IP3 (same reference plane as the
tone powers), two tones P1, P2 give
    2f1-f2, 2f1+f2 : 2*P1 + P2 - 2*IP3
    3f1            : 3*P1 - 2*IP3 - 9.54 dB
and second order with intercept IP2
    f1+f2, f1-f2   : P1 + P2 - IP2
    2f1            : 2*P1 - IP2 - 6.02 dB
Triple beats (f1+f2-f3) are not generated.
"""
from __future__ import annotations

import math

from ..spectrum import Spectrum, vec_comb
from ..units import parse_db
from .base import Block, Context, register

HD3_REL = 20 * math.log10(3)   # 9.54 dB: HD3 below IM3 at equal tone power
HD2_REL = 20 * math.log10(2)   # 6.02 dB: HD2 below IM2


def strongest_for_im(s: Spectrum, ctx: Context):
    tones = s.strongest(ctx.im_max_tones)
    if not tones:
        return []
    top = tones[0].power_dbm
    return [t for t in tones if t.power_dbm >= top - ctx.im_window_db and t.power_dbm >= ctx.floor_dbm]


def nonlinear_products(s: Spectrum, ctx: Context, ip3=None, ip2=None, harmonics=True) -> Spectrum:
    """Return the distortion products of tone set `s` (not including `s` itself)."""
    out = Spectrum()
    tones = strongest_for_im(s, ctx)
    floor = ctx.floor_dbm
    for i, a in enumerate(tones):
        if ip3 is not None:
            if harmonics:
                p = 3 * a.power_dbm - 2 * ip3 - HD3_REL
                if p >= floor:
                    out.add(vec_comb(a.vec, 3, (), 0), 3 * a.freq, p)
            for j, b in enumerate(tones):
                if i == j:
                    continue
                p = 2 * a.power_dbm + b.power_dbm - 2 * ip3
                if p < floor:
                    continue
                out.add(vec_comb(a.vec, 2, b.vec, -1), 2 * a.freq - b.freq, p, scale=2 * a.freq)
                out.add(vec_comb(a.vec, 2, b.vec, 1), 2 * a.freq + b.freq, p)
        if ip2 is not None:
            if harmonics:
                p = 2 * a.power_dbm - ip2 - HD2_REL
                if p >= floor:
                    out.add(vec_comb(a.vec, 2, (), 0), 2 * a.freq, p)
            for b in tones[i + 1:]:
                p = a.power_dbm + b.power_dbm - ip2
                if p < floor:
                    continue
                out.add(vec_comb(a.vec, 1, b.vec, 1), a.freq + b.freq, p)
                out.add(vec_comb(a.vec, 1, b.vec, -1), a.freq - b.freq, p, scale=a.freq)
    return out


# Rapp model with smoothness p=2 for gain compression; derive Psat from the output P1dB.
_RAPP_P = 2.0
# input drive (relative to saturation) at which gain is down 1 dB: (10^(2p/20) - 1)^(1/2p)
_X1DB = (10 ** (2 * _RAPP_P / 20) - 1) ** (1 / (2 * _RAPP_P))


def rapp_compression_db(p_lin_out_dbm: float, op1db_dbm: float) -> float:
    """Gain change (<= 0 dB) for a total linear output power, Rapp p=2, given output P1dB."""
    psat = op1db_dbm + 1.0 - 20 * math.log10(_X1DB)
    x = 10 ** ((p_lin_out_dbm - psat) / 20)
    g = (1 + x ** (2 * _RAPP_P)) ** (-1 / (2 * _RAPP_P))
    return 20 * math.log10(g)


@register("amplifier")
class Amplifier(Block):
    """Unilateral-ish amplifier with gain, intercepts and compression.

    gain_db              : small-signal gain
    oip3_dbm / iip3_dbm  : third-order intercept (give one; iip3 = oip3 - gain)
    oip2_dbm / iip2_dbm  : second-order intercept (optional)
    op1db_dbm            : output 1 dB compression (optional) -> Rapp p=2 compression of all tones
                           from the *total* output power
    reverse_isolation_db : |S12| loss for waves travelling out->in
    harmonics            : generate HD2/HD3 as well as two-tone products
    """
    ports = ("in", "out")
    default_in = "in"
    default_out = "out"

    def __init__(self, gain_db=0.0, oip3_dbm=None, iip3_dbm=None, oip2_dbm=None, iip2_dbm=None,
                 op1db_dbm=None, reverse_isolation_db=30.0, harmonics=True):
        super().__init__(gain_db=gain_db, oip3_dbm=oip3_dbm, iip3_dbm=iip3_dbm, oip2_dbm=oip2_dbm,
                         iip2_dbm=iip2_dbm, op1db_dbm=op1db_dbm,
                         reverse_isolation_db=reverse_isolation_db, harmonics=harmonics)
        self.gain = parse_db(gain_db)
        self.oip3 = parse_db(oip3_dbm) if oip3_dbm is not None else (
            parse_db(iip3_dbm) + self.gain if iip3_dbm is not None else None)
        self.oip2 = parse_db(oip2_dbm) if oip2_dbm is not None else (
            parse_db(iip2_dbm) + self.gain if iip2_dbm is not None else None)
        self.op1db = parse_db(op1db_dbm) if op1db_dbm is not None else None
        self.rev_iso = parse_db(reverse_isolation_db)
        self.harmonics = bool(harmonics)

    def respond(self, incident, ctx):
        lin = incident["in"].scaled(self.gain)
        if self.op1db is not None and len(lin):
            c = rapp_compression_db(lin.total_dbm(), self.op1db)
            if c < -0.5:
                ctx.warn(f"compressed by {-c:.1f} dB (total output {lin.total_dbm() + c:.1f} dBm)")
            lin = lin.scaled(c)
        if self.oip3 is not None or self.oip2 is not None:
            lin.add_spectrum(nonlinear_products(lin, ctx, self.oip3, self.oip2, self.harmonics))
        return {"out": lin, "in": incident["out"].scaled(-self.rev_iso)}
