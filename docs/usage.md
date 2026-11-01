# Usage

## Python

```python
from hetsim import Circuit, Source, Filter, Amplifier, Mixer, noise_figure

c = Circuit("rx")
c.add("ant", Source(tones=[{"name": "RF", "freq": "100MHz", "power": -80}]))
c.add("pre", Filter("bandpass", "butter", 3, f1="88MHz", f2="108MHz", insertion_loss_db=1.5))
c.add("lna", Amplifier(gain_db=15, oip3_dbm=30, nf_db=1.5))
c.add("lo",  Source(freq="110.7MHz", power=7, name="LO"))
c.add("mix", Mixer(conversion_loss_db=6, lo_rf_isolation_db=35, lo_if_isolation_db=30,
                   rf_if_isolation_db=25, iip3_dbm=15, spur_table="dbm_generic"))
c.add("iff", Filter("bandpass", "cheby1", 4, f0="10.7MHz", bw="200kHz"))
c.chain("ant", "pre", "lna", "mix", "iff")     # default in/out ports
c.connect("lo", "mix.lo")                      # anything else, by name

r = c.simulate()
print(r["iff.out"].table(ref="LO - RF", sort="power"))
print(noise_figure(c, "ant", "iff.out", tone="RF").table())
```

## Netlists

The same circuit in YAML. Keep it in a file and pass it to the command line or the viewers.
Values can carry units.

```yaml
name: rx
blocks:
  ant: {type: source, tones: [{name: RF, freq: 100MHz, power: -80}]}
  pre: {type: filter, response: bandpass, family: butter, order: 3, f1: 88MHz, f2: 108MHz, insertion_loss_db: 1.5}
  lna: {type: amplifier, gain_db: 15, oip3_dbm: 30, nf_db: 1.5}
  lo:  {type: source, name: LO, freq: 110.7MHz, power: 7}
  mix: {type: mixer, conversion_loss_db: 6, spur_table: dbm_generic}
  iff: {type: filter, response: bandpass, family: cheby1, order: 4, f0: 10.7MHz, bw: 200kHz}
chain: [ant, pre, lna, mix, iff]
wires:
  - [lo.out, mix.lo]
```

`python -m hetsim blocks` lists every block type and its parameters.

## Command line

```bash
# the spectrum leaving a port, or arriving at one
python -m hetsim run examples/vhf_single_conversion.yaml -p ifamp.out --sort power --limit 15
python -m hetsim run examples/vhf_single_conversion.yaml -p ant.out --incident
python -m hetsim run examples/vhf_single_conversion.yaml -p ifamp.out --solver td --band 9MHz:12MHz

# every stimulus frequency that reaches the output band
python -m hetsim responses examples/vhf_single_conversion.yaml --source ant --port ifamp.out \
    --band 10.6MHz:10.8MHz --range 1MHz:500MHz

# noise figure and port match
python -m hetsim noise examples/vhf_noise_match.yaml --source ant --tone RF --port ifamp.out --band 10.6MHz:10.8MHz
python -m hetsim match examples/vhf_noise_match.yaml --tone RF
python -m hetsim match examples/vhf_noise_match.yaml --port lna.in --range 88MHz:108MHz --points 11

# the viewers
python -m hetsim app examples/vhf_single_conversion.yaml
python -m hetsim view examples/vhf_single_conversion.yaml
```
