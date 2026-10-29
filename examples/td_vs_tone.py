"""Where the time-domain solver and the tone solver differ.

1. Two-tone test of an LNA swept into compression. Small-signal the solvers
   agree; near P1dB the TD solver shows the IM3 slope bending and IM5 appearing
   (the tone model keeps IM3 on its 3:1 line and has no IM5).
2. The VHF receiver's IF output: the wanted signal, the image-frequency signal
   and the blockers' IM3 all fall in the 10.7 MHz bin. The tone solver adds them
   in power; the TD solver adds them with their phases.

Run:  python examples/td_vs_tone.py   (writes td_*.png next to this file)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hetsim import Amplifier, Circuit, Source, load
from hetsim.plot import C1, C2, C3, INK2, SURFACE, _ax, plot_td_spectrum

HERE = Path(__file__).parent

# ---------------------------------------------------------------- 1. two-tone sweep
c = Circuit("LNA two-tone")
c.add("s", Source(tones=[{"name": "A", "freq": "100MHz", "power": -40},
                         {"name": "B", "freq": "101MHz", "power": -40}]))
c.add("lna", Amplifier(gain_db=15, oip3_dbm=30, op1db_dbm=18))
c.chain("s", "lna")

pins = np.arange(-40, 6, 1.0)
rows = []
for p in pins:
    c["s"].set(0, power=p)
    c["s"].set(1, power=p)
    tone = c.simulate()["lna.out"]
    td = c.simulate_td(fmax="400MHz").spectrum("lna.out")
    rows.append((tone.power("A"), tone.power("2*A - B"), td.power_at(100e6), td.power_at(99e6), td.power_at(98e6)))
rows = np.array(rows)

ax = _ax(None, figsize=(9, 5))
ax.plot(pins, rows[:, 2], color=C1, lw=2, label="fundamental, time domain")
ax.plot(pins, rows[:, 0], color=C1, lw=2, ls="--", label="fundamental, tone")
ax.plot(pins, rows[:, 3], color=C2, lw=2, label="IM3 (2A-B), time domain")
ax.plot(pins, rows[:, 1], color=C2, lw=2, ls="--", label="IM3, tone")
ax.plot(pins, np.maximum(rows[:, 4], -200), color=C3, lw=2, label="IM5 (3A-2B), time domain only")
ax.set_ylim(-160, 30)
ax.set_xlabel("input power per tone (dBm)", color=INK2)
ax.set_ylabel("output power (dBm)", color=INK2)
ax.set_title("LNA, G = 15 dB, OIP3 = +30 dBm, OP1dB = +18 dBm", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.figure.tight_layout()
ax.figure.savefig(HERE / "td_two_tone.png", dpi=150)
for p, r in zip(pins, rows):
    if p in (-30, -20, -10, 0):
        print(f"Pin {p:5.0f} dBm: fund {r[2]:7.2f} (tone {r[0]:7.2f})   IM3 {r[3]:8.2f} (tone {r[1]:8.2f})"
              f"   IM5 {r[4]:8.2f}")

# ---------------------------------------------------------------- 2. VHF IF, coherent addition
rx, st = load(HERE / "vhf_single_conversion.yaml")
r = rx.simulate_td(st)
print()
print(r.summary("ifamp.out", min_dbm=-150, fmin=9e6, fmax=12.5e6))
ax = plot_td_spectrum(r.spectrum("ifamp.out"), fmin=9.5e6, fmax=11.9e6, min_dbm=-140, labels=5,
                      band=(10.6e6, 10.8e6), overlay_tone=True,
                      title="VHF IF output, time-domain FFT vs tone solver")
ax.figure.tight_layout()
ax.figure.savefig(HERE / "td_vhf_if.png", dpi=150)
print(f"\nwrote {HERE / 'td_two_tone.png'} and {HERE / 'td_vhf_if.png'}")
