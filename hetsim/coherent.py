"""The coherent tone solver: tones keep their phase, and same-product paths interfere.

The power solver iterates everything, and reflections between two ports slow it down as they near
total. Here the linear network (wires, filters, lines, amplifier gain, leakage, reflections) is solved
directly, one matrix per tone, so a stopband reflecting everything costs no more than a matched port:

    b = A(f) b + s(f)        b: wave leaving each port; s: what the blocks create there

A[p, peer(q)] is the block's gain from its port q to its port p, since what arrives at q left the far
end of q's wire. Only the nonlinear parts iterate: from the waves arriving, each block generates
its products (Block.generate) and sets its operating point, the network is solved again with
those as sources, and this settles in about as many rounds as there are conversions in a row.

Products keep their mixing-vector identity, so the same product by two paths adds as phasors. Different
products on one frequency stay separate and aren't summed; the time-domain solver adds those too.
"""
from __future__ import annotations

import numpy as np

from .blocks.base import Context
from .spectrum import Spectrum, coherent_mode

BATCH_TONES = 2000     # tones per batch of matrices: bounds the memory used
RUNAWAY_DBM = 100.0    # no real wave gets here: the solve is growing without end, and would overflow


def _network(circuit, ops, freqs, index):
    """A(f) for each of freqs: an array (tones, ports, ports)."""
    p_all = len(index)
    a = np.zeros((len(freqs), p_all, p_all), complex)
    for (name, port), i in index.items():
        b = circuit.blocks[name]
        paths = list(b.linear_c(port, freqs, ops[name]))
        if port in b.gamma2:
            paths.append((port, b.gamma_c(port, freqs)))
        for q, g in paths:
            peer = circuit.wires.get((name, q))
            if peer is not None:
                a[:, i, index[peer]] += g
    return a


def _loop_gain(a: np.ndarray, samples: int = 128) -> float:
    """Largest loop gain over a sample of the tones. Above one there is no steady state to solve for."""
    idx = np.unique(np.linspace(0, len(a) - 1, min(samples, len(a))).astype(int))
    return float(np.abs(np.linalg.eigvals(a[idx])).max())


def simulate_coherent(circuit, st):
    from .circuit import Result
    warnings: dict[str, list[str]] = {}

    def make_ctx(name):
        def warn(msg, _n=name):
            lst = warnings.setdefault(_n, [])
            if msg not in lst:
                lst.append(msg)
        return Context(st.floor_dbm, st.im_max_tones, st.im_window_db, warn, (st.td or {}).get("seed", 1))

    ctxs = {n: make_ctx(n) for n in circuit.blocks}
    every = [(n, p) for n, b in circuit.blocks.items() for p in b.ports]
    rf = [(n, p) for n, b in circuit.blocks.items() for p in b.ports if p not in b.controls]
    index = {k: i for i, k in enumerate(rf)}
    incident = {k: Spectrum() for k in every}
    outgoing: dict = {k: Spectrum() for k in every}
    tol_phase = np.radians(st.tol_deg)
    lim = 10 ** (st.floor_dbm / 10)
    converged, runaway, loop = False, False, 0.0
    it = 0
    with coherent_mode():
        for it in range(1, st.max_iter + 1):
            # what each block creates from the waves now arriving, and its operating point
            ops, sources, freqs = {}, {}, {}
            for n, b in circuit.blocks.items():
                inc = {p: incident[(n, p)] for p in b.ports}
                ops[n] = b.operating_point(inc)
                for p, s in b.generate(inc, ops[n], ctxs[n]).items():
                    if (n, p) not in index:
                        continue
                    for t in s:
                        freqs[t.vec] = t.freq
                        sources.setdefault(t.vec, []).append((index[(n, p)], t.power_dbm, t.phase))
            # the network, per tone
            vecs = list(freqs)
            f_all = np.array([freqs[v] for v in vecs], float)
            new_out = {k: Spectrum() for k in every}
            loop = 0.0
            for start in range(0, len(vecs), BATCH_TONES):
                chunk = vecs[start:start + BATCH_TONES]
                f = f_all[start:start + BATCH_TONES]
                a = _network(circuit, ops, f, index)
                s = np.zeros((len(chunk), len(index)), complex)
                for k, v in enumerate(chunk):
                    for i, dbm, ph in sources[v]:
                        s[k, i] += 10 ** (dbm / 20) * np.exp(1j * ph)
                loop = max(loop, _loop_gain(a))
                x = np.linalg.solve(np.eye(len(index))[None] - a, s[..., None])[..., 0]
                power, phase = np.abs(x) ** 2, np.angle(x)
                for i, key in enumerate(rf):
                    t = new_out[key]._t
                    for k in np.nonzero(power[:, i] >= lim)[0]:
                        t[chunk[k]] = [float(f[k]), float(power[k, i]), float(phase[k, i])]
            for key in rf:
                new_out[key].prune(st.floor_dbm, st.max_tones)
            new_inc = {}
            for key in every:
                peer = circuit.wires.get(key)
                new_inc[key] = new_out[peer] if peer else Spectrum()
            still = all(new_inc[k].approx_equal(incident[k], st.tol_db, st.floor_dbm + 3, tol_phase) for k in every)
            incident, outgoing = new_inc, new_out
            if still:
                converged = True
                break
            if max(s.total_dbm() for s in new_out.values()) > RUNAWAY_DBM:
                runaway = True
                break
    if runaway:
        warnings.setdefault("<solver>", []).append(
            f"levels passed {RUNAWAY_DBM:.0f} dBm and kept growing, so the solve stopped and these levels mean nothing: "
            "products gain on each pass. A mixer with reflective filters on both ports does this: put a pad of "
            "about 6 dB on one side, or give one filter a plain vswr")
    elif loop >= 1.0:
        converged = False
        warnings.setdefault("<solver>", []).append(
            f"a feedback loop has gain {loop:.2f} at some frequency: no steady state, these levels mean nothing")
    elif not converged:
        warnings.setdefault("<solver>", []).append(
            f"did not converge in {st.max_iter} rounds. A loop through a nonlinear block does this, such as a mixer with "
            "reflective filters on both ports: put a pad of about 6 dB on one side, or give one filter a plain vswr")
    return Result(circuit, incident, outgoing, it, converged, warnings, st)
