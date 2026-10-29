"""Unit parsing and dB helpers.

Frequencies are plain floats in Hz internally; powers are dBm. Anything that
comes from a netlist may be a string such as "10.7MHz", "-30dBm" or "6 dB".
"""
from __future__ import annotations

import math
import re

_FREQ_RE = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([kKMGT]?)\s*(?:[hH][zZ])?\s*$")
_SCALE = {"": 1.0, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
_DB_RE = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(?:dBm|dBc|dB)?\s*$", re.IGNORECASE)


def parse_freq(x) -> float:
    """Parse a frequency: number (Hz) or string like '10.7MHz', '455k', '2.4 GHz'."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = _FREQ_RE.match(str(x))
    if not m:
        raise ValueError(f"cannot parse frequency {x!r}")
    return float(m.group(1)) * _SCALE[m.group(2)]


def parse_db(x) -> float:
    """Parse a dB/dBm/dBc value: number or string like '-30dBm', '6 dB'."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = _DB_RE.match(str(x))
    if not m:
        raise ValueError(f"cannot parse dB value {x!r}")
    return float(m.group(1))


def dbm_to_mw(p_dbm: float) -> float:
    return 10.0 ** (p_dbm / 10.0)


def mw_to_dbm(p_mw: float) -> float:
    return 10.0 * math.log10(p_mw) if p_mw > 0 else -math.inf


def db_sum(*p_dbm: float) -> float:
    """Incoherent (power) sum of several levels in dBm."""
    return mw_to_dbm(sum(dbm_to_mw(p) for p in p_dbm if p > -math.inf))


def fmt_freq(f: float, digits: int = 6) -> str:
    """Engineering-format a frequency, e.g. 10700000 -> '10.7 MHz'."""
    a = abs(f)
    for scale, unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if a >= scale:
            return f"{f / scale:.{digits}g} {unit}"
    return f"{f:.{digits}g} Hz"
