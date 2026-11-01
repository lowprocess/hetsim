# Architecture

## Layers

| Module | Job |
|---|---|
| `hetsim/spectrum.py` | Tones and their mixing vectors; `Spectrum` |
| `hetsim/blocks/` | The block types, and what each does to incident waves |
| `hetsim/circuit.py` | The block graph and the tone solver |
| `hetsim/td/` | The time-domain solver and its block models |
| `hetsim/analysis.py` | Response search, band sweeps, phase noise |
| `hetsim/smallsignal.py` | Noise figure and port match |
| `hetsim/netlist.py` | YAML and JSON netlists |
| `hetsim/cli.py` | The command line |
| `hetsim/viewer/`, `hetsim/desktop/` | The browser viewer and desktop app, sharing `viewer/model.py` |

## Tone solver

A wire joins two ports and carries a wave each way. On each iteration every block turns its
incident waves into outgoing waves (`Block.respond`), and each outgoing wave becomes the incident
wave at the far end of its wire, until nothing moves by more than `tol_db`. A port with no wire is
a matched termination.

A tone's identity is its mixing vector, such as `LO - RF`, and its frequency follows from that.
The same product reached by two paths merges by power sum, while different products on one
frequency stay separate and labelled.

A port with a `vswr` also sends part of its incident wave straight back. The solver adds this for
every block, so `respond` doesn't deal with it.

## Time-domain solver

The same iteration, on sampled waveforms. Tones sit exactly on FFT bins, so the steady state is
periodic: linear blocks multiply the spectrum by their complex response, and nonlinear blocks act
sample by sample. Waveforms add in amplitude and phase, so same-bin products interfere, which the
tone solver can't show. Block models are in `td/models.py`.

## Small-signal analyses

`smallsignal.py` traces a small wave backwards from one port at one frequency, using a tone solve
as the operating point (LO frequencies and compressed gain). Each block lists what feeds each port
(`Block.transfers`). The (port, frequency) pairs reached form a linear system, solved directly, so
loops are exact and the result gives the gain from every point in the circuit to that port.

- **Noise figure:** each port emits `Block.noise_emission`, the block's `noise_target` less what
  it passes through. The default target makes a passive block a load at T0.
- **Port match:** the trace ends where it started, with the port's wire cut.

## Adding a block

1. Subclass `Block` or `TwoPort` in `hetsim/blocks/` and decorate it with `@register("kind")`.
2. Name the constructor's parameters and pass them to `super().__init__`, so that netlists round-trip
   and the viewers can find them. Include `vswr=None`.
3. Implement `respond` (a `TwoPort` only needs `s21_db`).
4. Add a model with `@td_model` in `td/models.py` for time-domain support.
5. Implement `transfers` for noise and match to see through it, and `noise_target` if it's active.
