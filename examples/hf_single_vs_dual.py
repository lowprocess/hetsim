"""HF receiver, two architectures, same mixers:

  A) single conversion straight to 455 kHz (LO = RF + 455 kHz)
     -> image only 910 kHz away; the preselector can't reject it.
  B) up-convert to a 45 MHz first IF (LO1 = RF + 45 MHz), 15 kHz roofing filter,
     then down to 455 kHz (LO2 = 45.455 MHz).
     -> first image at RF + 90 MHz, killed by the 30 MHz low-pass;
        second image at 45.91 MHz, killed by the roofing filter.

Run:  python examples/hf_single_vs_dual.py   (writes hf_responses.png next to this file)
"""
from pathlib import Path

import matplotlib.pyplot as plt
from hetsim import Amplifier, Circuit, Filter, Mixer, Source
from hetsim.analysis import find_responses
from hetsim.plot import C1, C2, plot_response

F_RF = 7.1e6
IF2 = 455e3
IF1 = 45e6
OUT = Path(__file__).parent


def mixer():
    return Mixer(conversion_loss_db=6.5, lo_rf_isolation_db=40, lo_if_isolation_db=35, rf_if_isolation_db=30,
                 iip3_dbm=15, spur_table="dbm_generic", lo_drive_dbm=7)


def front_end(c):
    c.add("ant", Source(tones=[{"name": "RF", "freq": F_RF, "power": -73},        # S9
                               {"name": "BLK", "freq": F_RF + 50e3, "power": -20}]))  # strong neighbour
    c.add("lpf", Filter("lowpass", "ellip", 7, fc="30MHz", ripple_db=0.1, stop_db=70, insertion_loss_db=0.5))
    c.add("lna", Amplifier(gain_db=10, oip3_dbm=35))


def single():
    c = Circuit("A: single conversion to 455 kHz")
    front_end(c)
    c.add("lo", Source(freq=F_RF + IF2, power=7, name="LO"))
    c.add("mix", mixer())
    c.add("iff", Filter("bandpass", "cheby1", 5, f0=IF2, bw="6kHz", ripple_db=0.5, max_rejection_db=80))
    c.add("ifamp", Amplifier(gain_db=30, oip3_dbm=25))
    c.chain("ant", "lpf", "lna", "mix", "iff", "ifamp")
    c.connect("lo", "mix.lo")
    return c


def dual():
    c = Circuit("B: up-convert to 45 MHz, then 455 kHz")
    front_end(c)
    c.add("lo1", Source(freq=F_RF + IF1, power=7, name="LO1"))
    c.add("mix1", mixer())
    c.add("roof", Filter("bandpass", "butter", 4, f0=IF1, bw="15kHz", insertion_loss_db=2, max_rejection_db=80))
    c.add("if1amp", Amplifier(gain_db=12, oip3_dbm=30))
    c.add("lo2", Source(freq=IF1 + IF2, power=7, name="LO2"))
    c.add("mix2", mixer())
    c.add("iff", Filter("bandpass", "cheby1", 5, f0=IF2, bw="6kHz", ripple_db=0.5, max_rejection_db=80))
    c.add("ifamp", Amplifier(gain_db=30, oip3_dbm=25))
    c.chain("ant", "lpf", "lna", "mix1", "roof", "if1amp", "mix2", "iff", "ifamp")
    c.connect("lo1", "mix1.lo")
    c.connect("lo2", "mix2.lo")
    return c


def main():
    band = (IF2 - 3e3, IF2 + 3e3)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, facecolor="#fcfcfb")
    for ax, c, col in zip(axes, (single(), dual()), (C1, C2)):
        res = c.simulate()
        print(f"\n### {c.name}")
        print(res["ifamp.out"].table(*band, sort="power", limit=6))
        tab = find_responses(c, "ant", "ifamp.out", band, "100kHz", "150MHz", mute=["ant"])
        print("\nspurious responses (stimulus = RF tone, blocker muted):")
        print(tab.table(limit=8))
        plot_response(None, tab, ax=ax, title=c.name, color=col, labels=6, floor_db=-120)
        lo_out = res.incident("ant.out").strongest(1)
        if lo_out:
            print(f"LO arriving back at the antenna: {lo_out[0]}")
        w = res.warning_text()
        if w:
            print("warnings:\n" + w)
    axes[0].set_xlabel("")
    fig.tight_layout()
    fig.savefig(OUT / "hf_responses.png", dpi=150)

    print(f"\nwrote {OUT / 'hf_responses.png'}")


if __name__ == "__main__":
    main()
