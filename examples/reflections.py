"""What a cable and a reactive filter do to a port's match, which only a solver with phase can show.

1. examples/vhf_cable.yaml looking into the coax: two mismatches a cable apart beat as frequency
   changes. Adding reflections in power gives a flat average; the phased figure shows the real ripple.
2. examples/vhf_reactive.yaml looking into the IF filter: its return loss comes from its own response,
   high at the reflection zeros in band and near 0 dB out of band.

Then both receivers go through the coherent tone solver and the time-domain solver, which should agree.

Run:  python examples/reflections.py   (writes reflections.png next to this file)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hetsim import load, match_sweep
from hetsim.plot import C1, C2, INK, INK2, SURFACE, _ax

HERE = Path(__file__).parent
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)

# ---------------------------------------------------------------- 1. ripple along a cable
c, st = load(HERE / "vhf_cable.yaml")
cable = match_sweep(c, "coax.in", np.linspace(88e6, 108e6, 201), st)
_ax(left)
x = cable.freqs / 1e6
left.plot(x, cable.vswr_worst, color=C2, linewidth=2, label="All in phase (limit)")
left.plot(x, cable.vswr_phased, color=C1, linewidth=2, label="Phased (real)")
left.plot(x, cable.vswr, color=INK2, linewidth=2, linestyle="--", label="Added in power (average)")
left.set(xlabel="Frequency (MHz)", ylabel="VSWR (:1)", title="Looking into 10 m of coax")
left.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3, fontsize=8, labelcolor=INK)

# ---------------------------------------------------------------- 2. a filter's own reflection
c, st = load(HERE / "vhf_reactive.yaml")
iff = match_sweep(c, "iff.in", np.linspace(10.3e6, 11.1e6, 3201), st)
_ax(right)
right.plot(iff.freqs / 1e6, iff.return_loss_db, color=C1, linewidth=2)
right.invert_yaxis()
right.set(xlabel="Frequency (MHz)", ylabel="Return loss (dB)",
          title="Looking into a reactive 10.7 MHz filter")
fig.tight_layout()
fig.savefig(HERE / "reflections.png", dpi=150)
print(f"cable: phased VSWR {cable.vswr_phased.min():.2f} to {cable.vswr_phased.max():.2f}, "
      f"added in power {cable.vswr.mean():.2f}")
print(f"filter: return loss {iff.return_loss_db.min():.2f} dB out of band, "
      f"{iff.return_loss_db[np.abs(iff.freqs - 10.7e6).argmin()]:.1f} dB at 10.7 MHz")

# ---------------------------------------------------------------- 3. the two solvers agree
for name, port, f in (("vhf_cable.yaml", "iff.out", 10.7e6), ("vhf_reactive.yaml", "ifamp.out", 10.7e6)):
    c, st = load(HERE / name)
    tone = c.simulate(st)[port]
    td = c.simulate_td(st, resolution="100kHz", fmax="400MHz").spectrum(port)
    t = max((t for t in tone if abs(t.freq - f) < 1), key=lambda t: t.power_dbm)
    print(f"{name}: {t.label} at {port}: tone {t.power_dbm:.3f} dBm, time domain {td.power_at(f):.3f} dBm")
