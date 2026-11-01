"""Turn a netlist into a list of adjustable parameters, apply edits, and run a simulation.

Kept separate from the HTTP server so it can be tested (and reused by a future GUI)
without a browser.
"""
from __future__ import annotations

import copy
import inspect
import json
import math
import re
import threading
import time
from pathlib import Path

import numpy as np

from .. import analysis
from ..analysis import band_sweep, output_grid_fractions, phase_noise_skirts
from ..spectrum import vec_coeff
from ..blocks import REGISTRY, SPUR_PRESETS
from ..blocks.sources import MODES, SPACINGS, SWEEP_POINTS, WIDEBAND_POINTS, ToneSpec, parse_band
from ..circuit import Settings
from ..netlist import from_dict, to_dict
from ..units import fmt_freq, parse_db, parse_freq

FREQ_KEYS = {"fc", "f0", "bw", "f1", "f2", "fmin", "fmax", "sample_rate", "analog_bw"}
CHOICES = {
    "response": ["lowpass", "highpass", "bandpass", "bandstop"],
    "family": ["butter", "cheby1", "cheby2", "ellip", "bessel"],
    "lo_waveform": ["sine", "square"],
    "spur_table": sorted(SPUR_PRESETS),
    "detect": ["peak", "total"],
    "scale": ["log", "linear"],
    "sideband": ["upper", "lower"],
}
BOOLS = {"harmonics"}
SKIP = {"file", "tones", "freq", "power", "name", "phase", "harmonics_dbc"}

# what File > New starts from: the smallest circuit the model accepts, to build on
NEW_CIRCUIT = """\
# New circuit: add blocks with "+ Add block" and wire them port to port.
name: new circuit
blocks:
  src1: {type: source, tones: [{name: SRC1, freq: 10MHz, power: -30}]}
  load1: {type: load}
wires:
  - [src1.out, load1.in]
"""

# offered when a tone's phase noise is first switched on in the viewer: an example shape to edit
PN_EXAMPLE = [[1e3, -80.0], [10e3, -95.0], [100e3, -110.0], [1e6, -130.0], [10e6, -145.0]]

# name -> (min, max, step, value to offer when an optional parameter is switched on)
RANGES = {
    "gain_db": (-20, 60, 0.5, 10),
    "oip3_dbm": (-30, 60, 0.5, 20), "iip3_dbm": (-30, 60, 0.5, 10),
    "oip2_dbm": (-10, 90, 0.5, 40), "iip2_dbm": (-10, 90, 0.5, 40),
    "op1db_dbm": (-20, 40, 0.5, 15),
    "lo_drive_dbm": (-10, 27, 0.5, 7),
    "lo_drive_tolerance_db": (0.5, 10, 0.5, 3),
    "conversion_loss_db": (0, 20, 0.1, 6),
    "insertion_loss_db": (0, 30, 0.1, 1), "loss_db": (0, 60, 0.1, 3),
    "extra_loss_db": (0, 30, 0.1, 0), "excess_loss_db": (0, 30, 0.1, 0),
    "max_rejection_db": (20, 200, 1, 100), "rejection_db": (20, 200, 1, 100),
    "ripple_db": (0.01, 3, 0.01, 0.5), "stop_db": (10, 120, 1, 60),
    "reverse_isolation_db": (0, 100, 0.5, 30), "isolation_db": (0, 100, 0.5, 25),
    "order": (1, 12, 1, 3),
    "max_order": (2, 20, 1, 4),
    "nf_db": (0, 30, 0.1, 3), "vswr": (1, 10, 0.05, 1.5),
    "gain_imbalance_db": (-3, 3, 0.05, 0.5), "phase_imbalance_deg": (-20, 20, 0.1, 2),
    "bits": (4, 24, 1, 14), "full_scale_dbm": (-20, 30, 0.5, 10), "snr_db": (20, 120, 0.5, 70),
    "hd2_dbc": (-120, -30, 1, -80), "hd3_dbc": (-120, -30, 1, -80),
}
ISO = (0, 100, 0.5, 40)
SWEEP_BUDGET_S = 0.5   # time the viewer spends on a band sweep per update (automatic steps)
SWEEP_MAX_S = 10.0     # ...and at most, when the steps are set by hand
GENERIC = (-100, 100, 0.1, 0)


class NetlistTextError(ValueError):
    """A netlist edited in the browser that doesn't parse or build; `line` is 1-based when known."""

    def __init__(self, msg: str, line: int | None = None):
        super().__init__(msg)
        self.line = line


def parse_text(text: str) -> dict:
    """Netlist dict from YAML or JSON text (YAML is a superset of JSON, so either works)."""
    try:
        import yaml
    except ImportError:
        yaml = None
    if yaml is not None:
        try:
            d = yaml.safe_load(text)
        except yaml.MarkedYAMLError as e:
            mark = e.problem_mark or e.context_mark
            raise NetlistTextError(f"YAML: {e.problem or e}", mark.line + 1 if mark else None) from None
    else:
        try:
            d = json.loads(text)
        except json.JSONDecodeError as e:
            raise NetlistTextError(f"JSON: {e.msg}", e.lineno) from None
    if not isinstance(d, dict) or not isinstance(d.get("blocks"), dict) or not d["blocks"]:
        raise NetlistTextError("the netlist needs a 'blocks' mapping with at least one block")
    return d


def dump_text(d: dict, fmt: str = "yaml") -> str:
    if fmt == "yaml":
        try:
            import yaml
            return yaml.safe_dump(d, sort_keys=False)
        except ImportError:
            pass
    return json.dumps(d, indent=2) + "\n"


def _block_line(text: str, name: str) -> int | None:
    """Line where block `name` is defined, searching after the 'blocks' key."""
    start = re.search(r"^[ \t]*[\"']?blocks[\"']?[ \t]*:", text, re.M)
    pos = start.end() if start else 0
    m = re.compile(rf"^[ \t]*[\"']?{re.escape(name)}[\"']?[ \t]*:", re.M).search(text, pos)
    return text.count("\n", 0, m.start()) + 1 if m else None


# schematic editor: parameters a newly added block starts with, and its name prefix
ADD_DEFAULTS = {
    "source": {"freq": "10MHz", "power": -30},
    "amplifier": {"gain_db": 10},
    "mixer": {"conversion_loss_db": 6},
    "iq_mixer": {"conversion_loss_db": 6, "sideband": "lower"},
    "adc": {"sample_rate": "100MHz", "bits": 14},
    "filter": {"response": "lowpass", "family": "butter", "order": 3, "fc": "100MHz"},
    "ideal_filter": {"fmin": "1MHz", "fmax": "100MHz"},
    "attenuator": {"loss_db": 3},
    "splitter": {},
    "load": {},
    "detector": {"detect": "peak"},
    "xyplot": {},
}
_PREFIX = {"source": "src", "amplifier": "amp", "mixer": "mix", "iq_mixer": "iq", "adc": "adc", "filter": "flt", "ideal_filter": "flt",
           "attenuator": "att", "splitter": "split", "load": "load", "detector": "det", "xyplot": "xy"}
_NAME = re.compile(r"[A-Za-z_][\w-]*$")
_KEY = re.compile(r"""["']?([^"'\s:#\-\[{][^"':]*?)["']?[ \t]*:(?:[ \t]|$)""")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _sections(lines: list[str], indent: int = 0, start: int = 0, stop: int | None = None) -> dict:
    """key -> [first, end) line range of each mapping key at `indent` in lines[start:stop].
    Blank lines and shallower comments just before the next key belong to that key."""
    stop = len(lines) if stop is None else stop
    keys = []
    for i in range(start, stop):
        ln = lines[i]
        if ln.strip() and _indent(ln) == indent and (m := _KEY.match(ln[indent:])):
            keys.append((m.group(1), i))
    out = {}
    for j, (k, a) in enumerate(keys):
        e = keys[j + 1][1] if j + 1 < len(keys) else stop
        while e > a + 1 and (not lines[e - 1].strip()
                             or (lines[e - 1].lstrip().startswith("#") and _indent(lines[e - 1]) <= indent)):
            e -= 1
        out[k] = [a, e]
    return out


def _yaml_entry(key, value, indent: int = 0) -> str:
    import yaml

    class Dumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, indentless=False):   # "  - x", not "- x", under a key
            return super().increase_indent(flow, False)

    text = yaml.dump({key: value}, Dumper=Dumper, sort_keys=False, default_flow_style=None, width=1000)
    return "".join(" " * indent + ln for ln in text.splitlines(keepends=True))


def patch_text(text: str, old: dict, new: dict, renames: dict | None = None) -> str | None:
    """Rewrite only the parts of YAML `text` (parsed as `old`) that differ in `new`, keeping
    comments elsewhere. Returns None when it can't do that safely; the caller then regenerates."""
    try:
        import yaml  # noqa: F401
    except ImportError:
        return None
    renames = renames or {}
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    top = _sections(lines)
    if "blocks" not in top or set(top) != {str(k) for k in old}:
        return None
    edits = []   # (first, end, replacement), non-overlapping
    for key in list(old) + [k for k in new if k not in old]:
        if key == "blocks" or old.get(key) == new.get(key):
            continue
        a, e = top.get(key, (len(lines), len(lines)))
        edits.append((a, e, _yaml_entry(key, new[key]) if key in new else ""))
    ob, nb = old["blocks"], new["blocks"]
    if ob != nb or renames:
        a, e = top["blocks"]
        if lines[a].split(":", 1)[1].split("#")[0].strip():
            return None   # flow-style blocks: {...}
        body = [ln for ln in lines[a + 1:e] if ln.strip() and not ln.lstrip().startswith("#")]
        if not body:
            return None
        ind = _indent(body[0])
        sub = _sections(lines, ind, a + 1, e)
        back = {v: k for k, v in renames.items()}
        for name, (ba, be) in sub.items():
            was = name
            now = renames.get(name, name)
            if now not in nb:
                edits.append((ba, be, ""))
            elif ob.get(was) != nb[now]:
                edits.append((ba, be, _yaml_entry(now, nb[now], ind)))
            elif now != was:
                m = _KEY.match(lines[ba][ind:])
                edits.append((ba, ba + 1, lines[ba][:ind] + now + lines[ba][ind + m.end(1):].lstrip("\"'")))
        added = "".join(_yaml_entry(n, spec, ind) for n, spec in nb.items()
                        if n not in sub and back.get(n) not in sub)
        if added:
            end = max((be for _, be in sub.values()), default=a + 1)
            edits.append((end, end, added))
    for a, e, rep_ in sorted(edits, key=lambda t: (t[0], t[1]), reverse=True):
        lines[a:e] = [rep_] if rep_ else []
    out = "".join(lines)
    try:
        return out if parse_text(out) == new else None
    except NetlistTextError:
        return None


def _set_wiring(d: dict, c) -> None:
    """Store circuit `c`'s wires in netlist dict `d`. A 'chain' is kept while all its links survive."""
    chain = d.get("chain") or []
    links = set()
    for a, b in zip(chain, chain[1:]):
        try:
            pa, pb = c.port(a, "out"), c.port(b, "in")
        except (KeyError, ValueError):
            links = None
            break
        if c.wires.get(pa) != pb:
            links = None
            break
        links |= {(pa, pb), (pb, pa)}
    if links is None:
        d.pop("chain", None)
        links = set()
    wires = [[a, b] for a, b in c.wire_list()
             if (tuple(a.split(".", 1)), tuple(b.split(".", 1))) not in links]
    as_set = lambda ws: {frozenset(map(str, w)) for w in ws}  # noqa: E731
    if as_set(wires) == as_set(d.get("wires") or []):
        return   # unchanged: keep the user's spelling
    if wires:
        d["wires"] = wires
    else:
        d.pop("wires", None)


def _gcd_hz(freqs):
    g = 0
    for f in freqs:
        g = math.gcd(g, int(round(f)))
    return float(g or 1)


def source_freqs(d):
    """Frequencies the sources emit (a wideband tone's whole comb), for the default FFT grid."""
    out = []
    for spec in d["blocks"].values():
        if spec.get("type") == "source":
            for t in spec.get("tones", []):
                ts = ToneSpec.from_dict(t)
                if not ts.mute:
                    out += ts.grid() if ts.mode == "wideband" else [ts.freq]
        if spec.get("type") == "adc":   # its samples must fall on the FFT grid too
            out.append(parse_freq(spec["sample_rate"]))
    return out


def _is_freq_path(path: str) -> bool:
    last = path.split(".")[-1]
    return last in FREQ_KEYS or last == "freq" or ".band." in path


def _limit_values(path: str, rng) -> tuple[float, float]:
    conv = parse_freq if _is_freq_path(path) else parse_db
    return conv(rng[0]), conv(rng[1])


def _fmt_limit(v: float, path: str):
    if _is_freq_path(path):
        for scale, unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
            if abs(v) >= scale:
                return f"{v / scale:.10g}{unit}"
        return f"{v:.10g}Hz"
    return int(v) if float(v).is_integer() else v


def _apply_limits(params: list[dict], limits: dict) -> None:
    """Sliders with a netlist limit keep to it."""
    for p in params:
        rng = limits.get(p["path"])
        if rng is None or p["kind"] not in ("freq", "number", "int", "level"):
            continue
        lo, hi = _limit_values(p["path"], rng)
        p.update(min=lo, max=hi, limits=[lo, hi])


def _filter_guess(pname, spec):
    """A sensible frequency for a filter parameter that isn't set (e.g. fc of a bandpass)."""
    f = {k: parse_freq(spec[k]) for k in FREQ_KEYS if spec.get(k) is not None}
    if "f1" in f and "f2" in f:
        f0, bw = math.sqrt(f["f1"] * f["f2"]), f["f2"] - f["f1"]
    else:
        f0, bw = f.get("f0", f.get("fc", 1e6)), f.get("bw", f.get("fc", 1e6) / 10)
    return {"fc": f0, "f0": f0, "bw": bw, "f1": f0 - bw / 2, "f2": f0 + bw / 2}.get(pname, f0)


def describe(d: dict) -> list[dict]:
    """Adjustable parameters per block, with ranges and current values."""
    groups = []
    for name, spec in d["blocks"].items():
        kind = spec["type"]
        params = []
        limits = spec.get("limits") or {}
        if kind == "source":
            for i, t in enumerate(spec.get("tones", [])):
                ts = ToneSpec.from_dict(t)
                f = ts.freq
                tag = t.get("name") or f"{name}_{i}"
                pre = f"tones.{i}."
                base = {"optional": False, "enabled": True}
                # band behaviour: cw has no band; the others show the band edges
                params.append({**base, "path": pre + "mute", "label": f"{tag} mute", "kind": "bool",
                               "value": ts.mute, "mutes": pre})
                params.append({**base, "path": pre + "mode", "label": f"{tag} mode", "kind": "choice",
                               "choices": list(MODES), "value": ts.mode})
                params.append({**base, "path": pre + "freq", "label": f"{tag} freq", "kind": "freq", "snap": True,
                               "value": f, "scale": "lin", "min": max(f * 0.75, 0), "max": f * 1.25,
                               "when": {pre + "mode": ["cw", "tune", "sweep"]},
                               "band": [pre + "band.0", pre + "band.1", pre + "mode"]})
                edges = ts.band or (f * 0.9, f * 1.1)
                for j, edge in enumerate(("low", "high")):
                    params.append({**base, "path": f"{pre}band.{j}", "label": f"{tag} band {edge}", "kind": "freq",
                                   "snap": False, "scale": "lin", "value": edges[j], "min": edges[j] * 0.5,
                                   "max": edges[j] * 1.5, "when": {pre + "mode": ["tune", "sweep", "wideband"]}})
                params.append({**base, "path": pre + "spacing", "label": f"{tag} spacing", "kind": "choice",
                               "choices": list(SPACINGS), "value": ts.spacing,
                               "when": {pre + "mode": ["sweep", "wideband"]}})
                if ts.step is None:   # a comb given by `step` keeps it; points would be ignored
                    pts = ts.points or (SWEEP_POINTS if ts.mode == "sweep" else WIDEBAND_POINTS)
                    params.append({**base, "path": pre + "points", "label": f"{tag} points", "kind": "int",
                                   "value": pts, "min": 2, "max": 201, "step": 1,
                                   "when": {pre + "mode": ["sweep", "wideband"]}})
                params.append({"path": f"tones.{i}.power", "label": f"{tag} power", "kind": "level",
                               "value": parse_db(t.get("power", 0)), "min": -160, "max": 30, "step": 0.5,
                               "unit": "dBm", "optional": False, "enabled": True})
                params.append({"path": pre + "phase_noise", "label": f"{tag} phase noise", "kind": "pn",
                               "optional": True, "enabled": bool(ts.phase_noise),
                               "value": [list(x) for x in ts.phase_noise] if ts.phase_noise else PN_EXAMPLE,
                               "when": {pre + "mode": ["cw", "tune", "sweep"]}})
                ph = t.get("phase")
                params.append({"path": f"tones.{i}.phase", "label": f"{tag} phase", "kind": "number",
                               "value": ph if ph is not None else 0.0, "min": 0, "max": 360, "step": 1,
                               "unit": "deg", "optional": True, "enabled": ph is not None})
            vs = spec.get("vswr")
            if not isinstance(vs, (dict, list, tuple)):
                params.append({"path": "vswr", "label": "vswr", "kind": "number", "optional": True,
                               "enabled": vs is not None, "value": parse_db(vs) if vs is not None else 1.5,
                               "min": 1, "max": 10, "step": 0.05, "unit": ":1"})
            _apply_limits(params, limits)
            groups.append({"block": name, "type": kind, "params": params})
            continue
        cls = REGISTRY[kind]
        for pname, p in inspect.signature(cls.__init__).parameters.items():
            if pname == "self" or pname in SKIP or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            raw = spec.get(pname, p.default)
            if pname == "vswr" and isinstance(raw, (dict, list, tuple)) or raw == "file":
                continue  # per-port, a table, or from a file: edit it in the netlist
            optional = p.default is None
            enabled = raw is not None
            entry = {"path": pname, "label": pname, "optional": optional, "enabled": enabled}
            if pname in CHOICES:
                if isinstance(raw, dict) or (pname == "spur_table" and isinstance(raw, str)
                                             and raw not in SPUR_PRESETS):
                    continue  # a table loaded from a file / inline dict: not editable here
                choices = list(CHOICES[pname])
                entry.update(kind="choice", choices=choices, value=raw if raw is not None else choices[0])
            elif pname in BOOLS:
                entry.update(kind="bool", value=bool(raw))
            elif pname in FREQ_KEYS:
                v = parse_freq(raw) if raw is not None else None
                if v is None:
                    v = _filter_guess(pname, spec)
                entry.update(kind="freq", value=v, scale="log", min=v / 10, max=v * 10, snap=False)
                if kind == "filter":
                    # show the description the filter actually uses; hide the rest
                    edges = spec.get("f1") is not None and spec.get("f2") is not None
                    if pname == "fc":
                        entry.update(optional=False, enabled=True, when={"response": ["lowpass", "highpass"]})
                    elif pname in ("f1", "f2"):
                        if not edges:
                            continue
                        entry.update(optional=False, enabled=True, when={"response": ["bandpass", "bandstop"]})
                    elif pname in ("f0", "bw"):
                        if edges:
                            continue
                        entry.update(optional=False, enabled=True, when={"response": ["bandpass", "bandstop"]})
            else:
                lo, hi, step, fallback = RANGES.get(pname, ISO if pname.endswith("isolation_db") else GENERIC)
                v = parse_db(raw) if raw is not None else fallback
                if pname in ("order", "max_order", "bits"):
                    entry.update(kind="int", value=int(v), min=lo, max=hi, step=1)
                else:
                    unit = (":1" if pname == "vswr" else "deg" if pname.endswith("_deg") else "dBc" if pname.endswith("_dbc")
                            else "dBm" if pname.endswith("_dbm") else "dB")
                    entry.update(kind="number", value=v, min=min(lo, v), max=max(hi, v), step=step, unit=unit)
                if kind == "filter" and pname == "ripple_db":
                    entry["when"] = {"family": ["cheby1", "ellip"]}
                if kind == "filter" and pname == "stop_db":
                    entry["when"] = {"family": ["cheby2", "ellip"]}
            params.append(entry)
        _apply_limits(params, limits)
        groups.append({"block": name, "type": kind, "params": params})
    return groups


def _set_param(spec: dict, path: str, value) -> None:
    """Set one parameter path ('gain_db', 'tones.0.freq', 'tones.0.band.1'); None removes it."""
    rng = (spec.get("limits") or {}).get(path)
    if rng is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
        lo, hi = _limit_values(path, rng)
        value = min(max(value, lo), hi)
    keys = path.split(".")
    if keys[0] != "tones":
        if value is None:
            spec.pop(path, None)
        else:
            spec[path] = value
        return
    tone = spec["tones"][int(keys[1])]
    if len(keys) == 3:
        if value is None:
            tone.pop(keys[2], None)
        else:
            tone[keys[2]] = value
    elif len(keys) == 4 and keys[2] == "band":
        if value is None:
            return   # the band goes away with mode: cw
        band = tone.get("band")
        band = list(band.split(":") if isinstance(band, str) else band or [None, None])
        band[int(keys[3])] = value
        tone["band"] = band
    else:
        raise KeyError(f"no parameter {path!r}")


def _tidy_tone(t: dict) -> None:
    """After slider edits: mode cw drops the band; a tuned or swept tone stays inside its band."""
    if not t.get("mute"):
        t.pop("mute", None)
    if t.get("spacing") == "lin" or t.get("mode") not in ("sweep", "wideband"):
        t.pop("spacing", None)    # the default, or meaningless for this mode
    if t.get("spacing") == "log":
        t.pop("step", None)       # a log comb is given by points
    if t.get("mode") == "cw":
        for k in ("band", "mode", "points", "step"):
            t.pop(k, None)
        return
    band = t.get("band")
    if band is None or None in band:
        return
    band = parse_band(band)
    if (t.get("mode") or "tune") in ("tune", "sweep") and t.get("freq") is not None:
        t["freq"] = min(max(parse_freq(t["freq"]), band[0]), band[1])


def _snap_tone(t: dict, res: float) -> dict:
    """Put a tone on the FFT grid; returns the snapped parameter values (path suffix -> Hz)."""
    band = parse_band(t.get("band")) if t.get("band") else None
    mode = t.get("mode") or ("tune" if band else "cw")
    if mode == "wideband":   # edges and comb spacing on the grid, so every comb tone sits on a bin
        lo = max(round(band[0] / res), 1) * res
        hi = max(round(band[1] / res) * res, lo + res)
        t["band"] = [lo, hi]
        if t.get("spacing") == "log":   # geometric tones can't share a grid; the TD solver moves each to its bin
            return {"band.0": lo, "band.1": hi}
        if not t.get("step"):
            n = t.get("points") or WIDEBAND_POINTS
            t["step"] = max(round((hi - lo) / (n - 1) / res), 1) * res
        return {"band.0": lo, "band.1": hi}
    f = parse_freq(t["freq"])
    fs = max(round(f / res), 1) * res
    if band and mode in ("tune", "sweep") and band[1] - band[0] >= res:
        while fs < band[0]:
            fs += res
        while fs > band[1]:
            fs -= res
    t["freq"] = fs
    return {"freq": fs}


def apply_edits(base: dict, edits: dict, resolution: float | None,
                tone_freqs: dict | None = None) -> tuple[dict, dict]:
    """Copy of `base` with edits applied. Returns (netlist dict, snapped source frequencies).
    `tone_freqs` ({tone name: Hz}) puts named tones at given frequencies (an animated sweep's frame)."""
    d = copy.deepcopy(base)
    snapped = {}
    for block, kv in (edits or {}).items():
        spec = d["blocks"].get(block)
        if spec is None:
            raise KeyError(f"no block {block!r}")
        for path, value in kv.items():
            _set_param(spec, path, value)
    for spec in d["blocks"].values():
        for t in spec.get("tones", []) if tone_freqs else []:
            if t.get("name") in tone_freqs:
                t["freq"] = float(tone_freqs[t["name"]])
    for block, spec in d["blocks"].items():
        if spec.get("type") != "source":
            continue
        for i, t in enumerate(spec.get("tones", [])):
            _tidy_tone(t)
            if resolution:
                for key, f in _snap_tone(t, resolution).items():
                    snapped.setdefault(block, {})[f"tones.{i}.{key}"] = f
    return d, snapped


class ViewerModel:
    def __init__(self, base: dict, source_path: Path | None = None, base_dir: Path | None = None,
                 text: str | None = None):
        self.base = base
        self.sweep_workers: int | None = None   # band-sweep processes; None: one per physical core
        self.source_path = source_path
        self.base_dir = base_dir
        self.format = "json" if source_path and source_path.suffix.lower() == ".json" else "yaml"
        # netlist source shown in the browser editor; keeps the file's comments and layout
        self.text = text if text is not None else dump_text(base, self.format)
        self.disk_text = text   # what the file held when last read or written
        self.default_resolution = _gcd_hz(source_freqs(base))
        self.lock = threading.Lock()
        self._blocks: dict = {}   # name -> (spec json, block); unchanged blocks keep their TD caches

    def build(self, d: dict):
        """Circuit for netlist dict `d`, reusing block objects whose parameters haven't changed."""
        import json as _json
        from ..circuit import Circuit
        c = Circuit(d.get("name", ""))
        keep = {}
        for name, spec in d["blocks"].items():
            key = _json.dumps(spec, sort_keys=True, default=str)
            hit = self._blocks.get(name)
            if hit is not None and hit[0] == key:
                blk = hit[1]
            else:
                blk = from_dict({"blocks": {name: spec}}, self.base_dir)[0].blocks[name]
            keep[name] = (key, blk)
            c.add(name, blk)
        self._blocks = keep
        if d.get("chain"):
            c.chain(*d["chain"])
        for a, b in d.get("wires", []):
            c.connect(a, b)
        return c, Settings(**d.get("settings", {}))

    @classmethod
    def from_file(cls, path):
        from ..netlist import load
        c, st = load(path)
        return cls(to_dict(c, st), Path(path), Path(path).parent, Path(path).read_text())

    @classmethod
    def from_circuit(cls, circuit, settings=None):
        return cls(to_dict(circuit, settings))

    @classmethod
    def from_text(cls, text: str, base_dir: Path | None = None) -> "ViewerModel":
        """A model for netlist text not tied to a file (Save As gives it one)."""
        c, st = from_dict(parse_text(text), base_dir)
        return cls(to_dict(c, st), None, base_dir, text)

    @classmethod
    def new(cls) -> "ViewerModel":
        """A starting circuit for building from scratch: one source wired to a load."""
        return cls.from_text(NEW_CIRCUIT)

    def info(self) -> dict:
        c, _ = from_dict(self.base, self.base_dir)
        ports = [f"{n}.{p}" for n, b in c.blocks.items() for p in b.ports if p not in b.controls]
        # default probe: the last block's default output in the main chain
        probe = None
        for n, b in reversed(list(c.blocks.items())):
            if b.default_out and b.kind != "source":
                probe = f"{n}.{b.default_out}"
                break
        schematic = {
            "blocks": [{"name": n, "type": b.kind, "ports": list(b.ports), "controls": list(b.controls)}
                       for n, b in c.blocks.items()],
            "wires": [list(w) for w in c.wire_list()],
            "layout": self.base.get("layout") or {},
            "addable": list(ADD_DEFAULTS),
        }
        return {"name": self.base.get("name", ""), "groups": describe(self.base), "ports": ports,
                "probe": probe or ports[0], "resolution": self.default_resolution,
                "file": str(self.source_path) if self.source_path else None,
                "text": self.text, "format": self.format, "can_save": self.source_path is not None,
                "schematic": schematic}

    def edit(self, op: dict) -> dict:
        """Apply one schematic operation to the netlist text and rebuild.

        op: {"op": "move" | "add" | "delete" | "rename" | "connect" | "disconnect", ...}, plus an
        optional "layout" {block: {x, y}} merged first, so blocks placed automatically stay put.
        """
        old = parse_text(self.text)
        new = copy.deepcopy(old)
        blocks = new["blocks"]
        kind = op.get("op")
        if op.get("layout"):
            lay = dict(new.get("layout") or {})
            for name, xy in op["layout"].items():
                if name in blocks:
                    lay[name] = {**(lay.get(name) or {}), "x": round(xy["x"]), "y": round(xy["y"])}
            new["layout"] = lay
        renames, added = {}, None
        if kind == "move":
            pass
        elif kind == "add":
            t = op.get("type")
            if t not in ADD_DEFAULTS:
                raise NetlistTextError(f"can't add a block of type {t!r} here; choose one of {list(ADD_DEFAULTS)}")
            added = op.get("name") or next(f"{_PREFIX[t]}{i}" for i in range(1, 10000)
                                           if f"{_PREFIX[t]}{i}" not in blocks)
            self._check_name(added, blocks)
            spec = {"type": t, **ADD_DEFAULTS[t]}
            if t == "source":
                used = {str(tn.get("name")) for b in blocks.values() if isinstance(b, dict)
                        for tn in b.get("tones") or [] if isinstance(tn, dict)}
                used |= {str(b.get("name")) for b in blocks.values() if isinstance(b, dict)}
                spec["name"] = next(f"{added.upper()}{'' if i == 1 else i}" for i in range(1, 10000)
                                    if f"{added.upper()}{'' if i == 1 else i}" not in used)
            blocks[added] = spec
            if "x" in op and "y" in op:
                new["layout"] = {**(new.get("layout") or {}), added: {"x": round(op["x"]), "y": round(op["y"])}}
        elif kind == "rename":
            was, now = op.get("block"), op.get("to")
            if was not in blocks:
                raise NetlistTextError(f"no block {was!r}")
            self._check_name(now, blocks)
            new["blocks"] = {(now if k == was else k): v for k, v in blocks.items()}
            ren = lambda r: now + r[len(was):] if r == was or str(r).startswith(was + ".") else r  # noqa: E731
            if new.get("chain"):
                new["chain"] = [ren(r) for r in new["chain"]]
            if new.get("wires"):
                new["wires"] = [[ren(a), ren(b)] for a, b in new["wires"]]
            if new.get("layout") and was in new["layout"]:
                new["layout"] = {(now if k == was else k): v for k, v in new["layout"].items()}
            renames = {was: now}
        elif kind == "limits":
            name, path, rng = op.get("block"), op.get("path"), op.get("range")
            if name not in blocks:
                raise NetlistTextError(f"no block {name!r}")
            spec = blocks[name]
            lim = dict(spec.get("limits") or {})
            if rng is None:
                lim.pop(path, None)
            else:
                lo, hi = (float(v) for v in rng)
                if not lo < hi:
                    raise NetlistTextError("the lower limit must be below the upper one")
                lim[path] = [_fmt_limit(lo, path), _fmt_limit(hi, path)]
            if lim:
                spec["limits"] = lim
            else:
                spec.pop("limits", None)
        elif kind in ("delete", "connect", "disconnect"):
            c, _ = from_dict(old, self.base_dir)
            try:
                if kind == "delete":
                    name = op.get("block")
                    c.remove(name)
                    del blocks[name]
                    if new.get("layout"):
                        new["layout"].pop(name, None)
                elif kind == "connect":
                    pa, pb = c.port(op["a"]), c.port(op["b"])
                    if pa[0] == pb[0]:
                        raise ValueError("can't wire a block to itself")
                    c.disconnect(pa)    # a port takes one wire: the new one replaces the old
                    c.disconnect(pb)
                    c.connect(pa, pb)
                else:
                    c.disconnect(c.port(op["a"]))
            except (KeyError, ValueError) as e:
                raise NetlistTextError(str(e).strip("'\"")) from None
            _set_wiring(new, c)
        else:
            raise NetlistTextError(f"unknown schematic operation {kind!r}")
        if not new.get("layout"):
            new.pop("layout", None)
        text = patch_text(self.text, old, new, renames) if self.format == "yaml" else None
        info = self.set_text(text if text is not None else dump_text(new, self.format))
        if added:
            info["added"] = added
        return info

    @staticmethod
    def _check_name(name, blocks):
        if not isinstance(name, str) or not _NAME.match(name):
            raise NetlistTextError(f"block names are letters, digits, '_' and '-', not {name!r}")
        if name in blocks:
            raise NetlistTextError(f"there is already a block called {name!r}")

    def set_text(self, text: str) -> dict:
        """Replace the netlist with edited source text. Raises NetlistTextError and keeps the old one."""
        d = parse_text(text)
        for name, spec in d["blocks"].items():  # build blocks one at a time to point at the bad one
            line = _block_line(text, str(name))
            if not isinstance(spec, dict) or "type" not in spec:
                raise NetlistTextError(f"block {name!r}: needs a 'type'", line)
            try:
                from_dict({"blocks": {name: spec}}, self.base_dir)
            except Exception as e:  # noqa: BLE001
                raise NetlistTextError(f"block {name!r}: {type(e).__name__}: {e}", line) from None
        try:
            c, st = from_dict(d, self.base_dir)
        except Exception as e:  # noqa: BLE001  - wiring, chain or settings
            raise NetlistTextError(f"{type(e).__name__}: {e}") from None
        with self.lock:
            self.base = to_dict(c, st)
            self.text = text
            self.default_resolution = _gcd_hz(source_freqs(self.base))
        return self.info()

    def save(self, force: bool = False) -> None:
        """Write the current source text back to the netlist file."""
        if self.source_path is None:
            raise ValueError("this circuit didn't come from a file; use Export instead")
        if not force and self.source_path.exists() and self.source_path.read_text() != self.disk_text:
            raise FileChangedError(f"{self.source_path} changed on disk since it was loaded")
        self.source_path.write_text(self.text)
        self.disk_text = self.text

    def save_as(self, path) -> dict:
        """Write the netlist to a new file (YAML or JSON by its suffix) and make it this model's file.
        Relative file references (CSV spur tables, Touchstone) resolve from there on."""
        path = Path(path)
        fmt = "json" if path.suffix.lower() == ".json" else "yaml"
        text = self.text if fmt == self.format else dump_text(parse_text(self.text), fmt)
        path.write_text(text)
        self.source_path, self.base_dir, self.format = path, path.parent, fmt
        info = self.set_text(text)
        self.disk_text = text
        return info

    def netlist(self, edits, resolution=None) -> dict:
        d, _ = apply_edits(self.base, edits, resolution)
        plain, _ = apply_edits(self.base, edits, None)
        for name, spec in d["blocks"].items():   # a comb step derived for the FFT grid is not the user's
            for t, u in zip(spec.get("tones", []), plain["blocks"][name].get("tones", [])):
                if "step" in t and "step" not in u:
                    del t["step"]
        return d

    def simulate(self, req: dict) -> dict:
        t0 = time.perf_counter()
        res = float(req.get("resolution") or self.default_resolution)
        solver = req.get("solver", "td")
        d, snapped = apply_edits(self.base, req.get("params", {}), res if solver != "tone" else None,
                                 req.get("tone_freqs"))
        frame = bool(req.get("sweep_frame"))   # an animation frame: the sweep and X/Y plots are kept as they were
        has_pn = any(t.get("phase_noise") for b in d["blocks"].values() for t in b.get("tones") or [])
        if not req.get("phase_noise", True):   # the header's switch: drop it for this update
            for b in d["blocks"].values():
                for t in b.get("tones") or []:
                    t.pop("phase_noise", None)
        rbw = float(req["rbw"]) if req.get("rbw") else None
        min_dbm = float(req.get("min_dbm", -200))
        probe = req["probe"]
        incident = bool(req.get("incident"))
        with self.lock:
            c, st = self.build(d)
            out = {"snapped": snapped, "has_pn": has_pn}
            t_tone = time.perf_counter()
            tone_result = c.simulate(st)
            t_tone = time.perf_counter() - t_tone
            tone_spec = tone_result.incident(probe) if incident else tone_result[probe]
            fund = fundamental_test(c)
            out["tone"] = [{"f": t.freq, "p": t.power_dbm, "label": t.label, "order": t.order, "fund": fund(t.vec)}
                           for t in tone_spec if t.power_dbm >= min_dbm]
            warn = tone_result.warning_text()
            dets = {n: b for n, b in c.blocks.items() if b.kind == "detector"}
            out["detectors"] = {n: _reading(b, tone_result.incident((n, "in"))) for n, b in dets.items()}
            swept = [t for b in c.blocks.values() for t in getattr(b, "sweep_specs", lambda: [])()]
            sweep_ports = [probe] + [f"{n}.in" for n in dets]   # detectors' inputs too, for the X/Y plots
            sw = xy_sw = None   # the tracks' sweep, and the X/Y plots' (the same unless on an output grid)
            if swept and not frame:
                par = analysis.POOL_SIZE if self.sweep_workers is None else max(1, self.sweep_workers)
                if analysis._POOL_FAILED:
                    par = 1
                per_step = max(t_tone, 1e-4) / min(par, analysis.POOL_SIZE)   # wall time per step
                asked = req.get("sweep_points")
                note = None
                if asked:   # set by hand: honoured up to a hard time cap
                    full = min(max(int(asked), 2), 2001)
                    n = min(full, max(2, int(SWEEP_MAX_S / per_step)))
                    if n < full:
                        note = (f"band sweep capped at {n} of the {full} steps asked for: each step takes "
                                f"{1000 * t_tone:.0f} ms over {par} process(es), and an update is limited "
                                f"to {SWEEP_MAX_S:g} s")
                else:       # automatic: the tones' points, as many as fit in the time budget
                    full = max(len(t.grid()) for t in swept)
                    n = min(full, max(5, int(SWEEP_BUDGET_S / per_step)))
                    if n < full:
                        note = (f"band sweep shown with {n} of {full} steps to keep updates quick "
                                f"({1000 * t_tone:.0f} ms per step over {par} process(es)); set Steps to choose")
                grid = req.get("sweep_out")
                if grid:   # steps placed so the anchor product lands on fixed output frequencies
                    names = {t.name for t in swept}
                    moving = [t for t in tone_spec if any(vec_coeff(t.vec, nm) for nm in names)]
                    anchor = grid.get("anchor")
                    if anchor and not any(t.label == anchor for t in moving):
                        note, anchor = f"anchor {anchor} isn't at this port; using the strongest swept product", None
                    else:
                        note = None
                    if not anchor and moving:
                        anchor = _default_anchor(c, moving, fund)
                    frac = []
                    if anchor:
                        pts = min(max(int(grid.get("points") or SWEEP_POINTS), 2), 2001)
                        frac = output_grid_fractions(c, anchor, np.linspace(float(grid["fmin"]), float(grid["fmax"]), pts))
                        cap = max(2, int(SWEEP_MAX_S / per_step))
                        if len(frac) > cap:
                            frac = frac[np.linspace(0, len(frac) - 1, cap).round().astype(int)]
                            note = f"output grid thinned to {cap} steps ({1000 * t_tone:.0f} ms per step)"
                    if len(frac):
                        sw = band_sweep(c, st, workers=self.sweep_workers, ports=sweep_ports, fractions=frac)
                        out["sweep"] = _sweep_json(sw, probe, incident, min_dbm, fund=fund)
                    else:
                        out["sweep"] = {"steps": [], "products": []}
                        note = note or (f"{anchor} doesn't reach {fmt_freq(grid['fmin'])} - {fmt_freq(grid['fmax'])} "
                                        "within the swept band" if anchor else "no swept product at this port")
                    out["sweep"]["anchor"] = anchor
                else:
                    sw = band_sweep(c, st, n, workers=self.sweep_workers, ports=sweep_ports)
                    out["sweep"] = _sweep_json(sw, probe, incident, min_dbm, fund=fund)
                if note:
                    out["sweep"]["note"] = note
            if not frame and any(b.kind == "xyplot" for b in c.blocks.values()):
                if swept and req.get("sweep_out"):
                    # an output grid only holds the steps that put its anchor in the window; an
                    # X/Y plot shows the whole swept band, evenly, with the same number of steps
                    xy_sw = band_sweep(c, st, n, workers=self.sweep_workers, ports=[f"{d}.in" for d in dets])
                out["xy"] = xy_plots(c, xy_sw or sw)
            if rbw and solver == "tone":   # phase noise for the RBW trace: the tone solver's skirts
                bn = 10 * math.log10(1.0645 * rbw)   # Gaussian RBW noise bandwidth
                sk = phase_noise_skirts(c, probe, st, incident, workers=self.sweep_workers, nominal=tone_result,
                                        min_dbm_hz=min_dbm - bn - 10)
                peak = {id(s): np.nanmax(s.density()) for s in sk}
                sk = [s for s in sk if peak[id(s)] + bn >= min_dbm - 3]
                sk.sort(key=lambda s: -peak[id(s)])
                out["skirts"] = [{"label": s.label, "src": s.source, "m": s.m, "f0": s.f0,
                                  "off": [round(float(o), 1) for o in s.offsets],
                                  "lvl": [round(float(v), 2) for v in s.level]} for s in sk[:200]]
                out["pn"] = {t.name: [list(x) for x in t.phase_noise] for b in c.blocks.values()
                             for t in getattr(b, "specs", []) if t.phase_noise}
            if solver in ("td", "both"):
                td = {"resolution": res}
                if req.get("fmax"):
                    td["fmax"] = parse_freq(req["fmax"])
                r = c.simulate_td(st, **td)
                sp = r.incident_spectrum(probe) if incident else r.spectrum(probe)
                idx = np.nonzero(sp.p_dbm >= min_dbm)[0]
                idx = idx[idx > 0]
                if len(idx) > 20000:
                    idx = idx[np.argsort(-sp.p_dbm[idx])[:20000]]
                    idx.sort()
                bins: dict[int, list] = {}
                for t in tone_spec:
                    bins.setdefault(int(round(t.freq / sp.df)), []).append(t)
                labels, funds = [], []
                for i in idx:
                    ts = sorted(bins.get(int(i), []), key=lambda t: -t.power_dbm)
                    labels.append([t.label for t in ts[:4]] + (["..."] if len(ts) > 4 else []))
                    funds.append(bool(ts) and fund(ts[0].vec))   # a bin takes its strongest product's class
                out["td"] = {"f": sp.freqs[idx].tolist(), "p": sp.p_dbm[idx].round(3).tolist(), "labels": labels,
                             "fund": funds}
                x = r.incident_waveform(probe) if incident else r.waveform(probe)
                out["waveform"] = _decimate(x, r.plan.fs)
                out["plan"] = str(r.plan)
                out["fs"], out["n"], out["resolution"] = r.plan.fs, r.plan.n, r.plan.resolution
                out["converged"] = r.converged
                warn = r.warning_text()
            out["warnings"] = warn
        out["elapsed_ms"] = round(1000 * (time.perf_counter() - t0), 1)
        return out


class FileChangedError(RuntimeError):
    pass


def lo_sources(c) -> set[str]:
    """Source blocks that drive a mixer's LO port, traced back through two-ports and splitters."""
    out = set()
    for name, b in c.blocks.items():
        if "lo" not in b.ports:
            continue
        port, seen = (name, "lo"), set()
        while port in c.wires and port not in seen:
            seen.add(port)
            blk, p = c.wires[port]
            other = c.blocks[blk]
            if other.kind == "source":
                out.add(blk)
                break
            if other.kind == "splitter" and p in ("a", "b"):
                port = (blk, "sum")
            elif len(other.ports) == 2:
                port = (blk, next(q for q in other.ports if q != p))
            else:
                break
    return out


def fundamental_test(c):
    """Predicate on a product's vector: a fundamental mixer product converts exactly one signal
    tone (coefficient +-1) through LOs (each +-1, each from a different source): 'LO - RF',
    'LO2 + RF - LO1', 'LO - B1'. Not LO x LO leakage ('LO2 - LO1') or intermods ('B2 - B1')."""
    owner = {t.name: n for n, b in c.blocks.items() for t in getattr(b, "tones", [])}
    los = lo_sources(c)

    def test(vec) -> bool:
        if len(vec) < 2 or any(abs(k) != 1 for _, k in vec):
            return False
        blocks = [owner.get(s) for s, _ in vec]
        if None in blocks or len(set(blocks)) != len(blocks):
            return False
        return sum(b not in los for b in blocks) == 1
    return test


def _finite_or_none(x: float):
    return x if math.isfinite(x) else None


def _reading(det, spectrum) -> dict:
    """A detector's reading: on its own scale (value, unit) and in dBm; None for -inf."""
    return {"value": _finite_or_none(det.output(spectrum)), "unit": det.unit,
            "dbm": _finite_or_none(det.read(spectrum))}


def xy_plots(c, sw) -> list[dict]:
    """Each X/Y plot block's trace over the band sweep `sw`: x the swept tone's frequency (from the
    source wired to `x`), y the reading of the detector wired to `y` at each step."""
    out = []
    for name, b in c.blocks.items():
        if b.kind != "xyplot":
            continue
        rec = {"name": name, "x": [], "y": [], "y_unit": "dBm", "x_label": "", "y_label": "", "note": None}
        xw, yw = c.wires.get((name, "x")), c.wires.get((name, "y"))
        src = c.blocks.get(xw[0]) if xw else None
        det = c.blocks.get(yw[0]) if yw else None
        if src is None or src.kind != "source" or xw[1] != "sweep":
            rec["note"] = f"{name}: wire x to a source's sweep port"
        elif det is None or det.kind != "detector" or yw[1] != "out":
            rec["note"] = f"{name}: wire y to a detector's out port"
        elif not src.sweep_specs():
            rec["note"] = f"{name}: {xw[0]} has no tone in sweep mode (give one a band and mode: sweep)"
        elif sw is None or not sw.steps:
            rec["note"] = f"{name}: no band sweep to plot"
        else:
            t = src.sweep_specs()[0]
            band = "" if det.fmin is None and det.fmax is None else \
                f", {fmt_freq(det.fmin or 0)} – {fmt_freq(det.fmax) if det.fmax else '…'}"
            rec.update(x=[s[t.name] for s in sw.steps], x_tone=t.name,
                       y=[_finite_or_none(det.output(r.incident((yw[0], "in")))) for r in sw.results],
                       y_unit=det.unit, x_label=f"{t.name} frequency ({xw[0]})", y_label=f"{yw[0]} ({det.unit})",
                       title=f"{name}: {yw[0]} {det.detect}, {det.scale}{band}, vs {t.name}")
        out.append(rec)
    return out


def _default_anchor(c, moving, fund) -> str:
    """The output grid's anchor when no track is selected: the conversion of the signal, i.e. the
    first non-LO source's tone (a comb's middle one), else the strongest fundamental, else the
    strongest product that moves with the sweep."""
    los = lo_sources(c)
    sig = next((b.tones for n, b in c.blocks.items() if b.kind == "source" and n not in los and b.tones), [])
    name = sig[len(sig) // 2].name if sig and sig[0].member is not None else sig[0].name if sig else None
    for pool in ([t for t in moving if fund(t.vec) and name and vec_coeff(t.vec, name)],
                 [t for t in moving if fund(t.vec)], moving):
        if pool:
            return max(pool, key=lambda t: t.power_dbm).label


def _sweep_json(sw, probe, incident, min_dbm, keep: int = 150, fund=None) -> dict:
    """Each product's track over the band sweep (None where absent), strongest `keep` only."""
    def arr(a, nd):
        return [None if v != v else round(float(v), nd) for v in a]
    prods = sw.products(probe, incident, min_dbm)[:keep]
    return {"steps": sw.steps,
            "products": [{"label": s.label, "f": arr(s.freqs, 1), "p": arr(s.powers, 2),
                          "fund": bool(fund and fund(s.vec))} for s in prods]}


def _decimate(x: np.ndarray, fs: float, buckets: int = 1500) -> dict:
    n = len(x)
    if n <= buckets * 2:
        return {"dt": 1 / fs, "min": x.tolist(), "max": x.tolist(), "span": n / fs}
    k = n // buckets
    m = x[: k * buckets].reshape(buckets, k)
    return {"dt": k / fs, "min": m.min(axis=1).tolist(), "max": m.max(axis=1).tolist(), "span": n / fs}


def describe_text(d: dict) -> str:
    """Plain listing of the adjustable parameters (handy for checking a netlist)."""
    lines = []
    for g in describe(d):
        lines.append(f"{g['block']} ({g['type']})")
        for p in g["params"]:
            v = fmt_freq(p["value"]) if p["kind"] == "freq" else p["value"]
            lines.append(f"    {p['label']:<24} {v}{'' if p['enabled'] else '  (off)'}")
    return "\n".join(lines)
