"""Circuit (schematic graph) and the wave solver.

Ports are joined point-to-point by wires; each wire carries a wave in each
direction. The solver repeatedly asks every block for its outgoing waves given
its incident waves, and feeds each outgoing wave to the port at the far end of
its wire, until nothing changes (to within tol_db). A port with no wire is a
matched termination: what leaves it is lost (but recorded). A port with a `vswr`
also sends part of what arrives straight back; bounces add in power here.

Branching needs an explicit Splitter, because a wire joins exactly two ports.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .blocks.base import Block, Context
from .spectrum import Spectrum

PortRef = tuple  # (block_name, port_name)


@dataclass
class Settings:
    floor_dbm: float = -200.0       # drop products below this level
    max_tones: int = 5000           # per port, strongest kept
    im_max_tones: int = 24          # tones considered for two-tone IM in each nonlinear block
    im_window_db: float = 100.0     # ...within this range of the strongest
    tol_db: float = 0.001           # convergence tolerance
    max_iter: int = 200
    td: dict | None = None          # time-domain solver options (see hetsim.td.TDSettings)


class Circuit:
    def __init__(self, name: str = ""):
        self.name = name
        self.blocks: dict[str, Block] = {}
        self.wires: dict[PortRef, PortRef] = {}
        self.layout: dict = {}   # free-form GUI data (positions...), carried through netlists

    # -- construction -----------------------------------------------------
    def add(self, name: str, block: Block) -> Block:
        if name in self.blocks:
            raise ValueError(f"duplicate block name {name!r}")
        if "." in name:
            raise ValueError("block names may not contain '.'")
        block.bind(name)
        self.blocks[name] = block
        return block

    def remove(self, name: str) -> Block:
        """Delete a block and its wires."""
        for p in [p for p in self.wires if p[0] == name]:
            self.disconnect(p)
        return self.blocks.pop(name)

    def replace(self, name: str, block: Block) -> Block:
        """Swap a block for another, keeping the wiring (ports are matched by name)."""
        kept = [(p[1], self.wires[p]) for p in self.wires if p[0] == name]
        missing = [p for p, _ in kept if p not in block.ports]
        if missing:
            raise ValueError(f"replacement for {name!r} lacks connected ports {missing}")
        self.remove(name)
        self.add(name, block)
        for p, peer in kept:
            self.connect((name, p), peer)
        return block

    def __getitem__(self, name: str) -> Block:
        return self.blocks[name]

    def port(self, ref, role: str = "out") -> PortRef:
        """Resolve 'blk.port' or 'blk' (default in/out port) to (blk, port)."""
        if isinstance(ref, tuple):
            b, p = ref
        elif "." in ref:
            b, p = ref.split(".", 1)
        else:
            b = ref
            blk = self.blocks[b]
            p = blk.default_out if role == "out" else blk.default_in
            if p is None:
                raise ValueError(f"block {b!r} has no default {role} port; use '{b}.<port>'")
        if b not in self.blocks:
            raise KeyError(f"no block {b!r}")
        if p not in self.blocks[b].ports:
            raise KeyError(f"block {b!r} ({self.blocks[b].kind}) has no port {p!r}; ports: {self.blocks[b].ports}")
        return (b, p)

    def connect(self, a, b) -> None:
        pa, pb = self.port(a, "out"), self.port(b, "in")
        for p in (pa, pb):
            if p in self.wires:
                raise ValueError(f"port {p[0]}.{p[1]} is already connected to "
                                 f"{self.wires[p][0]}.{self.wires[p][1]} (use a splitter to branch)")
        if pa == pb:
            raise ValueError("cannot connect a port to itself")
        ca, cb = (p[1] in self.blocks[p[0]].controls for p in (pa, pb))
        if ca != cb:
            ctl, rf = (pa, pb) if ca else (pb, pa)
            raise ValueError(f"{ctl[0]}.{ctl[1]} is a control port (for an X/Y plot); it can't take the RF port "
                             f"{rf[0]}.{rf[1]}")
        self.wires[pa] = pb
        self.wires[pb] = pa

    def disconnect(self, a) -> None:
        pa = self.port(a, "out")
        pb = self.wires.pop(pa, None)
        if pb:
            self.wires.pop(pb, None)

    def chain(self, *refs) -> None:
        """Connect blocks in series using default ports: chain('ant', 'lna', 'mix', 'iff')."""
        for a, b in zip(refs, refs[1:]):
            self.connect(self.port(a, "out"), self.port(b, "in"))

    def wire_list(self) -> list[tuple[str, str]]:
        seen, out = set(), []
        for a, b in self.wires.items():
            if (b, a) in seen:
                continue
            seen.add((a, b))
            out.append((f"{a[0]}.{a[1]}", f"{b[0]}.{b[1]}"))
        return out

    def symbols(self) -> dict[str, float]:
        """Fundamental tone symbols and their current frequencies."""
        out = {}
        for b in self.blocks.values():
            for t in getattr(b, "tones", []):
                if t.name in out:
                    raise ValueError(f"tone symbol {t.name!r} used twice")
                out[t.name] = t.freq
        return out

    # -- simulation -------------------------------------------------------
    def simulate_td(self, settings: "Settings | None" = None, td=None, **td_kw):
        """Time-domain / FFT solver; see hetsim.td. Returns a TDResult."""
        from .td import simulate_td
        return simulate_td(self, settings, td, **td_kw)

    def simulate_sweep(self, settings: "Settings | None" = None, points: int | None = None,
                       workers: int | None = 1, ports=None, fractions=None):
        """Step the source tones that have mode 'sweep' across their bands; see analysis.band_sweep
        (workers=None spreads the steps over one process per physical core; fractions places the
        steps explicitly, e.g. from analysis.output_grid_fractions)."""
        from .analysis import band_sweep
        return band_sweep(self, settings, points, workers, ports, fractions)

    def simulate(self, settings: Settings | None = None, **kw) -> "Result":
        st = settings or Settings()
        for k, v in kw.items():
            setattr(st, k, v)
        self.symbols()  # validates uniqueness

        warnings: dict[str, list[str]] = {}

        def make_ctx(name):
            def warn(msg, _n=name):
                lst = warnings.setdefault(_n, [])
                if msg not in lst:
                    lst.append(msg)
            return Context(st.floor_dbm, st.im_max_tones, st.im_window_db, warn)

        ctxs = {n: make_ctx(n) for n in self.blocks}
        all_ports = [(n, p) for n, b in self.blocks.items() for p in b.ports]
        incident = {p: Spectrum() for p in all_ports}
        outgoing: dict[PortRef, Spectrum] = {}
        converged = False
        it = 0
        for it in range(1, st.max_iter + 1):
            outgoing = {}
            for n, b in self.blocks.items():
                res = b.respond({p: incident[(n, p)] for p in b.ports}, ctxs[n])
                for p in b.ports:
                    s = res.get(p) or Spectrum()
                    if p in b.gamma2:
                        s.add_spectrum(b.reflected(p, incident[(n, p)]))
                    outgoing[(n, p)] = s.prune(st.floor_dbm, st.max_tones)
            new_inc = {}
            for p in all_ports:
                peer = self.wires.get(p)
                new_inc[p] = outgoing[peer] if peer else Spectrum()
            if all(new_inc[p].approx_equal(incident[p], st.tol_db, st.floor_dbm + 3) for p in all_ports):
                converged = True
                incident = new_inc
                break
            incident = new_inc
        if not converged:
            warnings.setdefault("<solver>", []).append(
                f"did not converge in {st.max_iter} iterations (feedback loop with gain?)")
        return Result(self, incident, outgoing, it, converged, warnings, st)


class Result:
    """Simulation result. Index by port: result['iff.out'] is the wave leaving that port.

    For a wire a.out -> b.in, result['a.out'] == result.incident('b.in'): the
    forward signal on that wire. result.incident('ant.out') is what arrives back
    at the antenna (e.g. LO radiation).
    """

    def __init__(self, circuit, incident, outgoing, iterations, converged, warnings, settings):
        self.circuit = circuit
        self._inc = incident
        self._out = outgoing
        self.iterations = iterations
        self.converged = converged
        self.warnings = warnings
        self.settings = settings

    def __getitem__(self, ref) -> Spectrum:
        return self._out[self.circuit.port(ref, "out")]

    def outgoing(self, ref) -> Spectrum:
        return self[ref]

    def incident(self, ref) -> Spectrum:
        return self._inc[self.circuit.port(ref, "in")]

    def warning_text(self) -> str:
        return "\n".join(f"[{b}] {m}" for b, ms in self.warnings.items() for m in ms)

    def summary(self, port, **table_kw) -> str:
        head = f"{self.circuit.name or 'circuit'}: wave leaving {port}  " \
               f"({'converged' if self.converged else 'NOT converged'} in {self.iterations} it.)"
        txt = head + "\n" + self[port].table(**table_kw)
        w = self.warning_text()
        return txt + ("\nwarnings:\n" + w if w else "")

    def __repr__(self):
        return f"Result({self.circuit.name!r}, iterations={self.iterations}, converged={self.converged})"
