"""An analogue-to-digital converter: aliasing, clipping, quantisation and its noise floor.

The tone solver folds every tone into the first Nyquist zone, labelling it with the sampling
frequency's own symbol (for an ADC block `adc`, `3*adc_FS - RF`), so alias responses can be traced
and searched for like any mixing product. The time-domain solver samples, clips and quantises for
real (hetsim.td.models).

The wave on `out` stands for the digital samples, at the level they represent: a tone at
full_scale_dbm on `in` reads full_scale_dbm on `out`.
"""
from __future__ import annotations

import math

from ..spectrum import Spectrum, vec_comb, vec_of
from ..units import KT0_DBM_HZ, parse_db, parse_freq
from .base import Block, register


@register("adc")
class ADC(Block):
    """sample_rate     samples per second
    bits            resolution; sets the quantisation step and, without snr_db, the ideal SNR 6.02*bits + 1.76
    full_scale_dbm  power of a sine that just reaches full scale, at `in`
    snr_db          measured SNR for a full-scale sine over the Nyquist band; sets the noise floor
    hd2_dbc, hd3_dbc  harmonics of a full-scale sine (negative). They fall 1 and 2 dB faster than a tone's level
    analog_bw       input bandwidth: noise folds in from zones up to here (default: 10 x sample_rate)
    vswr            port mismatch at `in`
    """
    ports = ("in", "out")
    default_in = "in"
    default_out = "out"

    def __init__(self, sample_rate, bits=14, full_scale_dbm=10.0, snr_db=None, hd2_dbc=None, hd3_dbc=None,
                 analog_bw=None, vswr=None):
        super().__init__(sample_rate=sample_rate, bits=bits, full_scale_dbm=full_scale_dbm, snr_db=snr_db,
                         hd2_dbc=hd2_dbc, hd3_dbc=hd3_dbc, analog_bw=analog_bw, vswr=vswr)
        self.fs = parse_freq(sample_rate)
        if not self.fs > 0:
            raise ValueError("sample_rate must be positive")
        if int(bits) != bits or not 1 <= bits <= 32:
            raise ValueError(f"bits must be a whole number from 1 to 32, not {bits!r}")
        self.bits = int(bits)
        self.fsdbm = parse_db(full_scale_dbm)
        self.snr = parse_db(snr_db) if snr_db is not None else 6.02 * self.bits + 1.76
        self.hd2 = parse_db(hd2_dbc) if hd2_dbc is not None else None
        self.hd3 = parse_db(hd3_dbc) if hd3_dbc is not None else None
        for h in (self.hd2, self.hd3):
            if h is not None and h > 0:
                raise ValueError("hd2_dbc and hd3_dbc are levels below the tone: use negative values")
        self.bw = parse_freq(analog_bw) if analog_bw is not None else 10 * self.fs
        self.sym = "ADC_FS"

    def bind(self, name):
        super().bind(name)
        self.sym = name.replace("-", "_") + "_FS"

    @property
    def sample_rate(self) -> float:
        return self.fs

    def _fold(self, out: Spectrum, v, f: float, p: float, phase: float = 0.0) -> None:
        k = round(f / self.fs)
        if k == 0:
            out.add(v, f, p, phase=phase)
        else:
            out.add(vec_comb(vec_of(self.sym, k), 1, v, -1), k * self.fs - f, p, scale=k * self.fs, phase=-phase)

    def respond(self, incident, ctx):
        s, out = incident["in"], Spectrum()
        if len(s) and s.total_dbm() > self.fsdbm:
            ctx.warn(f"input {s.total_dbm():.1f} dBm is over full scale ({self.fsdbm:.1f} dBm): it clips, which only "
                     "the time-domain solver shows")
        for t in s:
            self._fold(out, t.vec, t.freq, t.power_dbm, t.phase)
            for n, dbc in ((2, self.hd2), (3, self.hd3)):
                if dbc is not None:
                    p = t.power_dbm + dbc + (n - 1) * (t.power_dbm - self.fsdbm)
                    if p >= ctx.floor_dbm:
                        self._fold(out, vec_comb(t.vec, n, (), 0), n * t.freq, p, n * t.phase)
        return {"out": out}

    def generate(self, incident, op, ctx):
        return self.respond(incident, ctx)

    # -- small signal -------------------------------------------------------
    def transfers(self, port, f, op):
        if port != "out" or f > self.fs / 2:
            return []
        inputs = {f}
        for k in range(1, int(self.bw / self.fs) + 2):
            inputs |= {k * self.fs - f, k * self.fs + f}
        return [("in", fi, 1.0) for fi in sorted(inputs) if 0 < fi <= self.bw]

    def noise_density_dbm_hz(self) -> float:
        """Quantisation and thermal noise, spread evenly over the Nyquist band."""
        return self.fsdbm - self.snr - 10 * math.log10(self.fs / 2)

    def noise_emission(self, port, f, op):
        if port != "out":
            return super().noise_emission(port, f, op)
        return 10 ** ((self.noise_density_dbm_hz() - KT0_DBM_HZ) / 10) if f <= self.fs / 2 else 0.0

    def linearity(self):
        # from HD3 = -2(OIP3 - P) - 9.54 dB at full scale. It clips at full scale, while the budget's
        # output P1dB is the level one dB below where the tone would be, hence - 1
        iip3 = self.fsdbm + (-self.hd3 - 20 * math.log10(3)) / 2 if self.hd3 is not None else None
        return iip3, self.fsdbm - 1
