"""Time-domain behaviour of each block type.

Each model maps incident waveforms (V across the reference impedance) to
outgoing waveforms. They use the same parameters as the tone models and are
built so the two solvers agree for small signals:

amplifier   y = a1*x + a2*x^2 + a3*x^3 with a3 = -(4/3)*a1/A_IIP3^2 and
            a2 = a1/A_IIP2 (A_ = peak amplitude at the intercept). This gives the
            same IM3/HD3/IM2/HD2 levels as the tone formulas. The cubic is clamped
            where it would fold over (|x| = A_IIP3/2). P1dB adds a sample-wise Rapp
            limiter (p = 2) calibrated to compress a single tone by 1 dB at op1db.
mixer       the LO is the strongest bin at the LO port, rebuilt as a clean
            cos(phi). Spur entry (m, n) becomes c * cos(m*phi) * x^n with c chosen so
            a single tone at ref_rf_dbm gives the tabulated level. Because x^n acts
            on the whole waveform, multi-tone inputs also produce the intermods
            those terms imply. IIP3 is a cubic on the RF input ahead of the 1x1
            term; IIP2 is c * x^2 straight to the output.
filters     complex response (with phase) applied per FFT bin.

Outputs of nonlinear blocks are AC-coupled (mean removed), as real chains are.

To add a model for your own block: @td_model(MyBlock) def f(block, incident, ctx) -> dict,
or give the block a respond_td(incident, ctx) method.
"""
from __future__ import annotations

import dataclasses
import math
import zlib

import numpy as np

from ..blocks import ADC, Amplifier, IQMixer, Load, Mixer, Source, Splitter, TwoPort
from ..blocks.sources import tone_phase

TD_MODELS: dict[type, callable] = {}


def td_model(cls):
    def deco(fn):
        TD_MODELS[cls] = fn
        return fn
    return deco


def get_model(block):
    for klass in type(block).__mro__:
        if klass in TD_MODELS:
            return TD_MODELS[klass]
    if hasattr(block, "respond_td"):
        return lambda b, inc, ctx: b.respond_td(inc, ctx)
    raise NotImplementedError(f"block type {type(block).__name__!r} has no time-domain model")


def _db(x):
    return 10 ** (-x / 20)


# ---------------------------------------------------------------- sources
def _phase_noise(t, ctx) -> np.ndarray | None:
    """Periodic phase modulation phi(n) with the tone's SSB profile: one sinusoid per FFT bin
    offset k*df, amplitude 2*sqrt(L*df) (so each sideband holds L*df), random phase."""
    from scipy import fft as sfft
    ks = np.arange(1, ctx.n // 2)
    lv = [t.pn_dbc(k * ctx.df) for k in ks[: int(t.phase_noise[-1][0] / ctx.df) + 1]]
    if not any(v is not None for v in lv):
        ctx.warn(f"{t.name}: phase-noise profile ends at {t.phase_noise[-1][0]:g} Hz, inside one FFT bin "
                 f"({ctx.df:g} Hz); refine the resolution to see it")
        return None
    rng = np.random.default_rng([ctx.seed, zlib.crc32(t.name.encode()), 7])
    P = np.zeros(ctx.n // 2 + 1, complex)
    for k, v in zip(ks, lv):
        if v is not None:
            P[k] = 2 * math.sqrt(10 ** (v / 10) * ctx.df) * ctx.n / 2 * np.exp(1j * rng.uniform(0, 2 * math.pi))
    return sfft.irfft(P, ctx.n)


@td_model(Source)
def _source(b: Source, inc, ctx):
    key = ("src", ctx.n, ctx.fs, ctx.seed, tuple((t.name, t.freq, t.power, t.phase, t.phase_noise) for t in b.tones),
           tuple(b.harmonics_dbc))
    x = ctx.cache.get(key)
    if x is None:
        ctx.cache.clear()
        X = np.zeros(ctx.n // 2 + 1, complex)
        xt = None   # tones with phase noise are built in the time domain
        for t in b.tones:
            ph = tone_phase(ctx.seed, t.name, t.phase)
            phi = _phase_noise(t, ctx) if t.phase_noise else None
            for k, dbc in [(1, 0.0)] + list(enumerate(b.harmonics_dbc, start=2)):
                kb = int(round(k * t.freq / ctx.df))
                if kb >= len(X) - 1:
                    ctx.warn(f"{k}x {t.name} at {k * t.freq:g} Hz is above Nyquist; not generated")
                    continue
                if phi is None:
                    X[kb] += ctx.amp(t.power + dbc) * ctx.n / 2 * np.exp(1j * k * ph)
                else:   # harmonic k carries k times the phase noise
                    if xt is None:
                        xt = np.zeros(ctx.n)
                    n = np.arange(ctx.n)
                    xt += ctx.amp(t.power + dbc) * np.cos(2 * np.pi * kb * n / ctx.n + k * (ph + phi))
        from scipy import fft as sfft
        x = sfft.irfft(X, ctx.n)
        if xt is not None:
            x = x + xt
        x.flags.writeable = False
        ctx.cache[key] = x
    return {"out": x}


@td_model(Load)
def _load(b, inc, ctx):
    return {}


# ---------------------------------------------------------------- linear
@td_model(TwoPort)
def _twoport(b: TwoPort, inc, ctx):
    key = ("h", ctx.n, ctx.fs)
    h = ctx.cache.get(key)
    if h is None:
        h = (b.s21_c(ctx.f), b.s12_c(ctx.f))
        if hasattr(b, "_check"):
            b._check(ctx.f[[1, -1]], ctx)
        ctx.cache[key] = h
    return {"out": ctx.filt(inc["in"], h[0]), "in": ctx.filt(inc["out"], h[1])}


@td_model(Splitter)
def _splitter(b: Splitter, inc, ctx):
    g, iso = _db(b.loss), _db(b.iso)
    s, a, c = inc["sum"], inc["a"], inc["b"]
    return {"sum": g * (a + c), "a": g * s + iso * c, "b": g * s + iso * a}


# ---------------------------------------------------------------- nonlinear
_RAPP_R1 = None


def _rapp_r1():
    """Drive A/Vsat at which a single tone through the p=2 sample-wise Rapp limiter is 1 dB down."""
    global _RAPP_R1
    if _RAPP_R1 is None:
        th = 2 * np.pi * np.arange(4096) / 4096
        c = np.cos(th)

        def comp(r):
            y = r * c / (1 + np.abs(r * c) ** 4) ** 0.25
            return (2 * np.mean(y * c)) / r

        lo, hi, target = 1e-3, 100.0, 10 ** (-1 / 20)
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            if comp(mid) > target:
                lo = mid
            else:
                hi = mid
        _RAPP_R1 = math.sqrt(lo * hi)
    return _RAPP_R1


def cubic(x, a1, a2, a3, ctx=None, what=""):
    """a1*x + a2*x^2 + a3*x^3 (a3 <= 0), input clamped where the cubic would fold over."""
    if a3:
        lim = math.sqrt(a1 / (3 * -a3))
        if ctx is not None and np.max(np.abs(x)) > lim:
            ctx.warn(f"{what}input peaks exceed the cubic model's range (IIP3 - 6 dB); clamped")
        x = np.clip(x, -lim, lim)
    y = a1 * x
    if a2:
        y = y + a2 * x * x
    if a3:
        y = y + a3 * x * x * x
    return y


@td_model(Amplifier)
def _amplifier(b: Amplifier, inc, ctx):
    x = inc["in"]
    out = {"in": inc["out"] * _db(b.rev_iso) if inc["out"].any() else inc["out"]}
    if not x.any():
        out["out"] = x
        return out
    a1 = 10 ** (b.gain / 20)
    a3 = -(4 / 3) * a1 / ctx.amp(b.oip3 - b.gain) ** 2 if b.oip3 is not None else 0.0
    a2 = a1 / ctx.amp(b.oip2 - b.gain) if b.oip2 is not None else 0.0
    if not b.harmonics and (a2 or a3):
        ctx.warn("harmonics=False has no effect in the time domain")
    y = cubic(x, a1, a2, a3, ctx)
    if a2:
        y = y - y.mean()
    if b.op1db is not None:
        vs = ctx.amp(b.op1db + 1) / _rapp_r1()
        y = y / (1 + (np.abs(y) / vs) ** 4) ** 0.25
        if np.max(np.abs(y)) > 0.5 * vs:
            ctx.warn(f"output peaks reach {20 * math.log10(np.max(np.abs(y)) / vs):.1f} dB re Vsat (compressing)")
    out["out"] = y
    return out


def _lo_polynomials(b: Mixer, lo, ctx):
    """From the LO waveform: per RF power n, L_n(t) = sum over m of c_mn*cos(m*phi(t))."""
    from scipy import fft as sfft
    L = sfft.rfft(lo)
    L[0] = 0
    k = int(np.argmax(np.abs(L)))
    a_lo = 2 * abs(L[k]) / ctx.n
    p_lo = 10 * math.log10(a_lo ** 2 / (2 * ctx.impedance)) + 30
    if ctx.pn_span:
        # the LO's own phase, noise included: analytic signal of the bins around its fundamental
        w = int(ctx.pn_span / ctx.df)
        Z = np.zeros(ctx.n, complex)
        a, z = max(1, k - w), min(len(L) - 1, k + w + 1)
        Z[a:z] = 2 * L[a:z]
        phi = np.angle(sfft.ifft(Z))
    else:
        phi = 2 * np.pi * k * np.arange(ctx.n) / ctx.n + np.angle(L[k])
    entries = b.spur_entries()
    ms = sorted({m for m, _ in entries if m})
    alias = any(m * k >= ctx.n // 2 for m in ms)
    cos_m = {m: np.cos(m * phi) for m in ms}
    a_ref = ctx.amp(b.table.ref_rf_dbm)
    a3 = -(4 / 3) / ctx.amp(b.iip3) ** 2 if b.iip3 is not None else 0.0
    g2 = 10 ** (-b.cl / 20) / ctx.amp(b.iip2) if b.iip2 is not None else 0.0
    polys: dict[int, np.ndarray] = {}
    c11cos = None
    for (m, n), dbc in entries.items():
        g = 10 ** ((-b.cl - dbc) / 20)
        if (m, n) == (1, 1):
            term = 2 * g * cos_m[1]
            if a3:
                c11cos = term
                continue
        elif m == 0:
            term = np.full(ctx.n, 2 ** (n - 1) * g / a_ref ** (n - 1))
        else:
            term = (2 ** n * g / a_ref ** (n - 1)) * cos_m[m]
        polys[n] = polys[n] + term if n in polys else term
    return polys, c11cos, a3, g2, p_lo, alias


@td_model(Mixer)
def _mixer(b: Mixer, inc, ctx):
    rf, lo, iff = inc["rf"], inc["lo"], inc["if"]
    out_rf, out_lo, out_if = ctx.zeros(), ctx.zeros(), ctx.zeros()
    if b.lo_if is not None:
        out_if += lo * _db(b.lo_if)
        out_lo += iff * _db(b.lo_if)
    if b.lo_rf is not None:
        out_rf += lo * _db(b.lo_rf)
    if b.rf_lo is not None:
        out_lo += rf * _db(b.rf_lo)
    if b.rf_if is not None:
        out_if += rf * _db(b.rf_if)
        out_rf += iff * _db(b.rf_if)

    if not lo.any():
        if rf.any() or iff.any():
            ctx.warn("no LO drive at the LO port; only leakage is produced")
        return {"rf": out_rf, "lo": out_lo, "if": out_if}

    pack = ctx.cache.get("lo_pack")
    if pack is None or pack[0] is not lo:
        pack = (lo, _lo_polynomials(b, lo, ctx))
        ctx.cache["lo_pack"] = pack
        ctx.cache.pop("conv", None)
    polys, c11cos, a3, g2, p_lo, alias = pack[1]
    if b.lo_drive is not None and abs(p_lo - b.lo_drive) > b.lo_tol:
        ctx.warn(f"LO drive {p_lo:.1f} dBm vs nominal {b.lo_drive:.1f} dBm; "
                 f"conversion loss and spurs will differ from the datasheet")
    if alias:
        ctx.warn("some LO harmonics in the spur table are above Nyquist and will alias")

    conv = ctx.cache.setdefault("conv", {})
    for key, x, out in (("down", rf, out_if), ("up", iff, out_rf)):
        if not x.any():
            continue
        hit = conv.get(key)
        if hit is not None and hit[0] is x:
            out += hit[1]
            continue
        # y = sum_n L_n(t) * x^n, with L_n = sum_m c_mn cos(m phi) precomputed from the LO
        y = ctx.zeros()
        xn = None
        for n in range(1, max(polys, default=0) + 1):
            xn = x if n == 1 else xn * x          # repeated multiply: much faster than x ** n
            if n in polys:
                y += polys[n] * xn
        if c11cos is not None:   # 1x1 term through the IIP3 cubic
            y += c11cos * cubic(x, 1.0, 0.0, a3, ctx, "RF ")
        if g2:
            y += g2 * x * x
        y -= y.mean()
        conv[key] = (x, y)
        out += y
    return {"rf": out_rf, "lo": out_lo, "if": out_if}


# ---------------------------------------------------------------- I/Q mixer: two real branches
@td_model(IQMixer)
def _iq_mixer(b: IQMixer, inc, ctx):
    """I and Q are ordinary mixers on LOs 90 degrees apart; the output is (I + g*H[Q]) / 2, with H the
    +/-90 degree shift and its error, which keeps one sideband at the mixer's conversion loss."""
    rf, lo, iff = inc["rf"], inc["lo"], inc["if"]
    out_rf, out_lo, out_if = ctx.zeros(), ctx.zeros(), ctx.zeros()
    if b.lo_if is not None:
        out_if += lo * _db(b.lo_if)
        out_lo += iff * _db(b.lo_if)
    if b.lo_rf is not None:
        out_rf += lo * _db(b.lo_rf)
    if b.rf_lo is not None:
        out_lo += rf * _db(b.rf_lo)
    if b.rf_if is not None:
        out_if += rf * _db(b.rf_if)
        out_rf += iff * _db(b.rf_if)
    if not lo.any():
        if rf.any() or iff.any():
            ctx.warn("no LO drive at the LO port; only leakage is produced")
        return {"rf": out_rf, "lo": out_lo, "if": out_if}

    core = ctx.cache.get("core")
    if core is None:   # the conversion terms alone: isolation is added once, above
        core = ctx.cache["core"] = Mixer(b.cl, None, None, None, None, b.iip3, b.iip2, b.table, b.lo_drive, b.lo_tol,
                                         b.lo_waveform, b.max_order)
    key = ("hq", ctx.n, ctx.fs)
    h = ctx.cache.get(key)
    if h is None:
        lo90 = np.full(len(ctx.f), -1j, dtype=complex)                  # cos -> sin: the Q branch's LO
        side = -90.0 if b.sideband == "upper" else 90.0
        hq = np.full(len(ctx.f), np.exp(1j * math.radians(side + b.phase_imb)), dtype=complex)
        h = ctx.cache[key] = (lo90, hq * 10 ** (b.gain_imb / 20))
    lo_q = ctx.filt(lo, h[0])
    i_ctx = dataclasses.replace(ctx, cache=ctx.cache.setdefault("i", {}))
    q_ctx = dataclasses.replace(ctx, cache=ctx.cache.setdefault("q", {}))
    from_i = _mixer(core, {"rf": rf, "lo": lo, "if": iff}, i_ctx)
    from_q = _mixer(core, {"rf": rf, "lo": lo_q, "if": iff}, q_ctx)
    out_if += (from_i["if"] + ctx.filt(from_q["if"], h[1])) / 2
    out_rf += (from_i["rf"] + ctx.filt(from_q["rf"], h[1])) / 2
    return {"rf": out_rf, "lo": out_lo, "if": out_if}


# ---------------------------------------------------------------- ADC: real samples
@td_model(ADC)
def _adc(b: ADC, inc, ctx):
    x = inc["in"]
    if not x.any():
        return {"out": ctx.zeros()}
    hit = ctx.cache.get("adc")
    if hit is not None and hit[0] is x:
        return {"out": hit[1]}
    from scipy import fft as sfft
    ratio = b.fs / ctx.df
    m = int(round(ratio))
    if abs(ratio - m) > 1e-6:
        ctx.warn(f"sample rate {b.fs:g} Hz is not on the {ctx.df:g} Hz grid; using {m * ctx.df:g} Hz")
    # sampling folds the spectrum: the samples' M-point DFT is the record's N-point DFT, wrapped
    folded = np.zeros(m, complex)
    np.add.at(folded, np.round(sfft.fftfreq(ctx.n, 1 / ctx.n)).astype(int) % m, sfft.fft(x))   # by signed frequency
    s = sfft.ifft(folded * (m / ctx.n)).real
    a_fs = ctx.amp(b.fsdbm)
    if np.max(np.abs(s)) > a_fs:
        ctx.warn(f"samples clip at full scale ({b.fsdbm:.1f} dBm)")
    # polynomial from the harmonics of a full-scale sine: amplitudes k2*A/2 and k3*A/4
    if b.hd2 is not None:
        s = s + 2 * 10 ** (b.hd2 / 20) * s * s / a_fs
    if b.hd3 is not None:
        s = s + 4 * 10 ** (b.hd3 / 20) * s * s * s / a_fs ** 2
    lsb = 2 * a_fs / 2 ** b.bits
    extra = (a_fs ** 2 / 2) * 10 ** (-b.snr / 10) - lsb ** 2 / 12       # noise beyond quantisation, to reach snr_db
    if extra > 0:
        rng = np.random.default_rng([ctx.seed, zlib.crc32((b.name or "adc").encode())])
        s = s + rng.normal(0.0, math.sqrt(extra), m)
    s = np.clip(s, -a_fs, a_fs - lsb)
    s = np.round(s / lsb) * lsb
    # back on the solver's grid as the first Nyquist zone, the samples' own content
    y = np.zeros(ctx.n // 2 + 1, complex)
    spec = sfft.rfft(s) * (ctx.n / m)
    keep = min(len(y), len(spec))
    y[:keep] = spec[:keep]
    if m % 2 == 0 and keep == len(spec) and ctx.n > m:
        y[keep - 1] /= 2                                                # the Nyquist bin splits when re-expanded
    y[0] = 0
    out = sfft.irfft(y, ctx.n)
    ctx.cache["adc"] = (x, out)
    return {"out": out}
