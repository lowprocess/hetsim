"""Netlist I/O (YAML or JSON) - the format a schematic GUI reads and writes.

    name: single-conversion receiver
    settings: {floor_dbm: -180}
    blocks:
      ant:  {type: source, tones: [{name: RF, freq: 100MHz, power: -60}]}
      pre:  {type: filter, response: bandpass, family: butter, order: 3, f1: 88MHz, f2: 108MHz}
      lna:  {type: amplifier, gain_db: 15, oip3_dbm: 30}
      lo:   {type: source, freq: 110.7MHz, power: 7, name: LO}
      mix:  {type: mixer, conversion_loss_db: 6, lo_rf_isolation_db: 35, spur_table: dbm_generic}
      iff:  {type: filter, response: bandpass, family: cheby1, order: 4, f0: 10.7MHz, bw: 200kHz}
    chain: [ant, pre, lna, mix, iff]          # optional shorthand, default ports
    wires:
      - [lo.out, mix.lo]
    layout: {ant: {x: 0, y: 0}}               # ignored by the engine, preserved on save

A block may carry `tolerance: {param: amount}`, e.g. `tolerance: {gain_db: 1.5, f0: 50kHz}`: the
+/- spread hetsim.tolerance gives that parameter. The solvers ignore it; it is preserved on save.

A block may carry `limits: {param: [min, max]}`, e.g. `limits: {fc: [10MHz, 50MHz]}` or
`limits: {tones.0.power: [-100, -20]}`: the range the viewer's slider for that parameter
keeps to. The engine ignores it; it is preserved on save.

Block 'type' is any key of hetsim.blocks.REGISTRY; the remaining keys are the
block's constructor arguments. Frequencies and levels may carry units.
"""
from __future__ import annotations

import json
from pathlib import Path

from .blocks import REGISTRY
from .circuit import Circuit, Settings


def from_dict(d: dict, base_dir: Path | None = None) -> tuple[Circuit, Settings]:
    c = Circuit(d.get("name", ""))
    for name, spec in d.get("blocks", {}).items():
        spec = dict(spec)
        kind = spec.pop("type")
        limits = spec.pop("limits", None)
        tolerance = spec.pop("tolerance", None)
        if tolerance is not None and not isinstance(tolerance, dict):
            raise ValueError(f"block {name!r}: tolerance must map parameters to amounts")
        if limits is not None and (not isinstance(limits, dict) or not all(
                isinstance(v, (list, tuple)) and len(v) == 2 for v in limits.values())):
            raise ValueError(f"block {name!r}: limits must map parameters to [min, max]")
        if kind not in REGISTRY:
            raise ValueError(f"block {name!r}: unknown type {kind!r}; known: {sorted(REGISTRY)}")
        if base_dir is not None:  # file parameters are relative to the netlist
            for key in ("file", "spur_table"):
                v = spec.get(key)
                if isinstance(v, str) and (v.endswith(".csv") or v.lower().endswith(".s2p")):
                    p = Path(v)
                    if not p.is_absolute():
                        spec[key] = str(base_dir / p)
        blk = c.add(name, REGISTRY[kind](**spec))
        if limits:
            blk.limits = {str(k): list(v) for k, v in limits.items()}
        if tolerance:
            blk.tolerance = {str(k): v for k, v in tolerance.items()}
    if d.get("chain"):
        c.chain(*d["chain"])
    for a, b in d.get("wires", []):
        c.connect(a, b)
    c.layout = d.get("layout", {})
    st = Settings(**d.get("settings", {}))
    return c, st


def to_dict(c: Circuit, settings: Settings | None = None) -> dict:
    d = {"name": c.name, "blocks": {}}
    if settings is not None:
        d["settings"] = {k: v for k, v in vars(settings).items() if v != getattr(Settings(), k)}
    for name, b in c.blocks.items():
        d["blocks"][name] = {"type": b.kind, **b.to_params()}
        if getattr(b, "limits", None):
            d["blocks"][name]["limits"] = b.limits
        if getattr(b, "tolerance", None):
            d["blocks"][name]["tolerance"] = b.tolerance
    d["wires"] = [list(w) for w in c.wire_list()]
    if c.layout:
        d["layout"] = c.layout
    return d


def load(path) -> tuple[Circuit, Settings]:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return from_dict(data, path.parent)


def save(c: Circuit, path, settings: Settings | None = None) -> None:
    path = Path(path)
    d = to_dict(c, settings)
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        path.write_text(yaml.safe_dump(d, sort_keys=False))
    else:
        path.write_text(json.dumps(d, indent=2))
