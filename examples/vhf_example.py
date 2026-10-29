"""VHF single conversion from the YAML netlist: IF spectrum, and the effect of the preselector.

Run:  python examples/vhf_example.py   (writes vhf_*.png next to this file)
"""
from pathlib import Path

import matplotlib.pyplot as plt

from hetsim import Attenuator, load
from hetsim.analysis import find_responses
from hetsim.plot import C1, C2, plot_response, plot_spectrum

HERE = Path(__file__).parent
BAND = (10.6e6, 10.8e6)

c, st = load(HERE / "vhf_single_conversion.yaml")
res = c.simulate(st)
print(res.summary("ifamp.out", fmin=9e6, fmax=12.5e6, ref="LO - RF", sort="power"))

ax = plot_spectrum(res["ifamp.out"], fmin=9.5e6, fmax=11.9e6, min_dbm=-140, band=BAND, labels=6,
                   title="IF output: wanted, image signal, and IM3 of two blockers all on 10.7 MHz")
ax.figure.tight_layout()
ax.figure.savefig(HERE / "vhf_if_spectrum.png", dpi=150)

# same receiver with the preselector swapped for a 1.5 dB pad
fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#fcfcfb")
with_pre = find_responses(c, "ant", "ifamp.out", BAND, "1MHz", "500MHz", settings=st, mute=["ant"])
c2, _ = load(HERE / "vhf_single_conversion.yaml")
c2.replace("pre", Attenuator(1.5))
without = find_responses(c2, "ant", "ifamp.out", BAND, "1MHz", "500MHz", settings=st, mute=["ant"])
plot_response(None, without, ax=ax, color=C2, name="no preselector", labels=0, floor_db=-110)
plot_response(None, with_pre, ax=ax, color=C1, name="88-108 MHz preselector", labels=5, floor_db=-110,
              title="Spurious responses of the VHF receiver")
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(HERE / "vhf_responses.png", dpi=150)

print("\nwith preselector:\n" + with_pre.table(limit=6))
print("\nwithout:\n" + without.table(limit=6))
