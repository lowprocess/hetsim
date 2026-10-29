# hetsim

Tone-domain simulation of mixer / filter / amplifier chains, for frequency planning
and comparing superheterodyne architectures. Datecode `261029-r3`.

You draw the chain as a graph of blocks (in Python or a YAML/JSON netlist a GUI can
write). The engine pushes discrete tones through it and tells you every product that
reaches every port, with its level and its origin, e.g. `LO2 + RF - LO1` or
`3*LO - RF`.

```python
from hetsim import Circuit, Source, Filter, Amplifier, Mixer

c = Circuit("rx")
c.add("ant", Source(tones=[{"name": "RF",  "freq": "100MHz",   "power": -80},
                           {"name": "IMG", "freq": "121.4MHz", "power": -50}]))
c.add("pre", Filter("bandpass", "butter", 3, f1="88MHz", f2="108MHz", insertion_loss_db=1.5))
c.add("lna", Amplifier(gain_db=15, oip3_dbm=30))
c.add("lo",  Source(freq="110.7MHz", power=7, name="LO"))
c.add("mix", Mixer(conversion_loss_db=6, lo_rf_isolation_db=35, lo_if_isolation_db=30,
                   rf_if_isolation_db=25, iip3_dbm=15, spur_table="dbm_generic"))
c.add("iff", Filter("bandpass", "cheby1", 4, f0="10.7MHz", bw="200kHz"))
c.chain("ant", "pre", "lna", "mix", "iff")     # default in/out ports
c.connect("lo", "mix.lo")

r = c.simulate()
print(r["iff.out"].table(ref="LO - RF", sort="power"))   # wave leaving the IF filter
print(r.incident("ant.out").table())                      # LO radiated from the antenna
```

## Model

**Tones carry their mixing vector.** Each source tone is a symbol (`RF`, `LO1`...).
Every product is an integer combination of symbols, and its frequency follows from
that. Products landing on the same frequency stay separate and labelled. The same
product reached by two paths merges by power sum.

**Waves in both directions.** Each wire carries a forward and a reverse wave, and
blocks map incident waves to outgoing waves at all their ports. The solver iterates
to a fixed point. This is how LO leakage travels back through the LNA to the
antenna, and how a second LO's leakage comes back through the first IF. It is also
where VSWR will plug in, as a reflection term on each port.

**Blocks** (`python -m hetsim blocks` lists parameters):

| type | what it models |
|---|---|
| `source` | CW tones, optional harmonics (`harmonics_dbc`); an antenna, a generator or an LO |
| `mixer` | ports `rf`, `lo`, `if`. Ideal multiplier by default (LO ± RF, nothing else). Optional: conversion loss, LO→RF, LO→IF, RF→IF and RF→LO isolation, IIP3, IIP2, m×n spur table, nominal LO drive (warns if off), `max_order` (depth: only m×n spurs with m + n ≤ it; 2 = the 1×1 product only). Reciprocal, so it also up-converts. |
| `amplifier` | gain, OIP3/IIP3, OIP2/IIP2 (two-tone IM, HD2, HD3), output P1dB (Rapp compression from total power), reverse isolation |
| `filter` | Butterworth, Chebyshev I/II, elliptic or Bessel LP/HP/BP/BS. Insertion loss and a finite ultimate rejection (`max_rejection_db`, e.g. ~80 dB for a crystal filter) |
| `sparam` | Touchstone `.s2p` (\|S21\|, \|S12\|), e.g. a filter out of openEMS, mom25d or Qucs |
| `ideal_filter`, `attenuator`, `splitter`, `load` | as named |

**Bands and source modes.** A source tone can carry a `band` (`[88MHz, 108MHz]` or
`"88MHz:108MHz"`), and its `mode` sets what the band means:

| mode | behaviour |
|---|---|
| `cw` | a single tone at `freq`, with no band (the default without a band) |
| `tune` | a single tone at `freq` (default: the band centre). The band is its tuning range, and the viewer's slider stays inside it. This is the default with a band. |
| `sweep` | as `tune` for a single run. `c.simulate_sweep()` (or `hetsim run --sweep`) also steps it across the band in `points` steps (default 41) and follows every product: the frequency range it covers and its worst level. All sweep tones move together, so an LO given its own band tracks the RF and the IF stays put. |
| `wideband` | `points` tones (default 21), or one every `step`, spread over the band and sharing `power` equally. Use it for a modulated or noise-like signal. Both solvers see the whole comb, and the tones are named `RF~1`, `RF~2`, and so on. |

```yaml
ant:
  type: source
  tones:
    - {name: RF, band: [88MHz, 108MHz], mode: sweep, power: -80}
    - {name: BLK, freq: 100.4MHz, power: -30, mute: true}   # kept, but not emitted
lo: {type: source, tones: [{name: LO, band: [98.7MHz, 118.7MHz], mode: sweep, power: 7}]}
```

`mute: true` switches a tone off without deleting it.

**Parallel sweeps.** The tone solver is pure Python, so threads can't run sweep
steps at the same time (the GIL); worker processes can. `c.simulate_sweep(workers=None)`
uses one process per physical core, and `workers=1` (the default in the Python API)
keeps it in-process. The viewer and `hetsim run --sweep` use the workers by default;
pass `--workers N` to change that. On a 12-core machine, a 32-step sweep that took
240 ms in-process takes about 40 ms. A script that uses workers needs the usual
`if __name__ == "__main__":` guard. If the workers can't start, the sweep warns once
and runs in-process.

`spacing: log` places sweep steps or comb tones geometrically (equal ratios, not
equal Hz), which suits bands spanning decades: `{band: [10MHz, 3GHz], mode: sweep,
spacing: log, points: 61}`.
- **Swept tones together:** at each step, every swept tone sits at the same
  fraction of its own band. The steps follow the log grid of the first log-spaced
  tone, so an LO whose band is the RF band plus the IF still tracks exactly.
- **Wideband combs:** a log comb takes `points`, not `step`. Its tones can't all
  share one FFT grid, so the time-domain solver moves each to its nearest bin and
  says so.

```python
sw = c.simulate_sweep()
print(sw.table("iff.out", fmin="10.6MHz", fmax="10.8MHz"))   # what crosses the IF, and its worst level
```

**Phase noise.** A tone can carry `phase_noise: [[1kHz, -75], [10kHz, -95],
[100kHz, -118], [1MHz, -135], [10MHz, -145]]`: single-sideband dBc/Hz at each offset,
interpolated on a log-frequency scale, with none outside the listed offsets (add a
far-out point to extend the floor).

- **Time-domain solver:** the tone's phase is modulated by noise with that spectrum,
  one sinusoid per FFT bin, so it stays periodic on the grid. The mixer takes its LO
  phase from the actual waveform, so LO noise converts, is filtered and compresses
  for real. To see it, the resolution has to be finer than the offsets you care about.
- **Tone solver** (`phase_noise_skirts()`, and the viewer's RBW trace): the noisy tone
  is detuned by offsets d and the circuit re-solved. A product carrying m times the
  tone then lands m·d away, at whatever level the filters on its path give it, and
  its density there is that level + L(d) + 20·log m. Samples are added where a
  product's level jumps (filter edges); the profile itself is evaluated exactly.
- **Reciprocal mixing** comes out of both. A blocker converted to just outside the
  IF filter puts its LO-noise skirt on the wanted channel. In the VHF example with
  the LO given that profile, the two solvers agree within 1 dB across the IF
  passband edges. It costs about 150 ms per update on the worker pool.

**Limits.** Any block can carry `limits: {param: [min, max]}`, for example
`limits: {f1: [80MHz, 95MHz], tones.0.power: [-100, -20]}`. The engine ignores
them. The viewer's slider for that parameter keeps to the range and clamps typed
values.

**Spur tables.** Entry (m, n) is dBc below the 1×1 output at RF level
`ref_rf_dbm`. Products of RF order n move (n−1) dB per dB of RF drive, which is
the usual datasheet convention. Load your mixer's chart from CSV (format in
`hetsim/blocks/spurtable.py`). The `dbm_generic` preset holds illustrative values
for a diode DBM, not any real part.

## Time-domain / FFT solver

`c.simulate_td()` (or `--solver td` on the CLI) runs the same circuit or netlist
with sampled waveforms instead of tone lists.

```python
r = c.simulate_td(resolution="100kHz")      # every option is optional
print(r.plan)                                # fs, N, Nyquist, how high the products reach
print(r.summary("iff.out", min_dbm=-150))    # FFT bins, labelled with tone-solver products
x = r.waveform("iff.out")                    # volts into 50 ohms, period 1/resolution
```

**Coherent grid.** The record is exactly one period of the slowest common
frequency. The bin spacing defaults to the GCD of all source frequencies, so every
tone sits on a bin: no window, no leakage, and bin power is tone power.
Filters multiply each bin by their complex response, including phase. Amplifiers
and mixers act sample by sample. The same two-way wave iteration links the
blocks, so the result is the exact periodic steady state.

**Sampling plan.** The tone solver runs first and finds how high the significant
products reach (within `alias_db`, default 120 dB, of the strongest). The sample
rate is then set to cover them, and anything that would alias is reported. The
`td` options are `resolution`, `fmax`, `fs`, `oversample`, `alias_db`,
`max_samples`, `impedance` and `seed`. They can go under `settings: td:` in a
netlist.

**What the TD solver shows that the tone solver can't:**
- Products in the same bin add with their phases. In the VHF example, the wanted
  signal, the image-frequency signal and the blockers' IM3 share the 10.7 MHz bin,
  and come out about 2 dB under their power sum. Source phases can be set per tone
  (`phase` in degrees); otherwise they are fixed pseudo-random values from `seed`.
- Full nonlinear interaction: products of products, triple beats, IM5 and above,
  and compression driven by instantaneous peaks.
- Real waveforms, ready for later noise, phase-noise or demodulation work.

**The TD block models** use the same parameters as the tone models. They are
calibrated so the two solvers agree for small signals, within 0.05 dB in the tests.
- **Amplifier:** cubic polynomial from OIP3/OIP2, plus a sample-wise Rapp limiter
  set from OP1dB. Near compression the limiter adds 5th-order distortion, so IM3
  rises above the ideal 3:1 line (`examples/td_vs_tone.py`).
- **Mixer:** each spur-table entry (m, n) becomes cos(m·φ_LO)·xⁿ. With several
  tones present, those terms also make the intermods they physically imply.
  `lo_waveform: square` gives a hard-switched mixer in both solvers.

**Limit.** It samples at RF, so N ≈ 2.5 · fmax / resolution. Fine up to a few
million samples. Tones that share no coarse common grid need a coarser
`resolution`, with sources moved onto it.

## Desktop app

```
pip install 'hetsim[desktop]'                         # PySide6
python -m hetsim app examples/vhf_single_conversion.yaml
python -m hetsim app                                  # or hetsim-app: opens on the start page
```

Opened without a netlist, the app shows a start page with **New circuit**, **Open
netlist…**, recent files and (in a source checkout) the examples. You can also drop a
netlist file on the window. A new circuit starts as one source wired to a load; build
it with **+ Add block** and give it a file with **Save As**. The File menu has New
(Ctrl+N), Open (Ctrl+O), Open Recent, Save (Ctrl+S, which applies text waiting in the
netlist editor first), Save As (Ctrl+Shift+S), Export netlist (with the current slider
values), and Close (back to the start page). Saving as `.json` writes JSON.

A native window (PySide6) with everything the browser viewer below has: parameter
sliders, schematic editor, spectrum and waveform, band sweeps, RBW trace and phase
noise, the netlist text editor. It needs no server and no browser: the window drives
the same `ViewerModel` in-process and simulates on a worker thread, where the newest
request wins, so dragging a slider stays smooth. Band sweeps use the worker processes
as in the viewer; `--workers 1` keeps everything in one process. It follows the
system's light or dark palette.

The parameter panel shows one block at a time: the block selected on the schematic, or
chosen from the menu at the top of the panel. On the schematic, each block shows its
critical values under its name (an amplifier's gain, OIP3 and P1dB, a mixer's conversion
loss, LO drive and IIP3, a filter's band, family and order, a source's tones). They follow
the sliders.

**Animated sweep.** With a tone in sweep mode (on an RF source or an LO), the sweep toolbar
plays the sweep: ▶ steps the swept tones through it, a full simulation per frame, and the
spectrum, the waveform and the detector readings follow. The current step is marked on
every band-sweep track and X/Y trace. Drag the step slider to scrub. Set the speed in
frames per second and choose whether it loops; ■ returns to the tones' own frequencies.
Frames reuse the last sweep rather than recomputing it.

**Detector and X/Y plot** (new block types, available from **+ Add block**):
- `detector` terminates whatever arrives at `in` and reads its power, over an optional
  `fmin`–`fmax` band. `detect: peak` gives the strongest single product (a peak
  detector); `total` gives the power sum (a power meter). The reading shows under the
  block.
- `xyplot` draws one detector's reading against one swept tone's frequency, a point
  per band-sweep step, in its own tab under the spectrum.
- Wiring uses **control ports**, drawn as diamonds with dashed wires: wire a source's
  `sweep` port to the plot's `x`, and the detector's `out` to its `y`. Control ports
  carry no RF, connect only to each other, and aren't offered as probe points.

With the LO swept and a detector behind the IF filter, the X/Y plot is a swept
spectrum analyser's display: a plateau wherever an input (wanted, image, blocker or
spur) converts into the detector's band.

```yaml
  det1: {type: detector, detect: peak, fmin: 10.6MHz, fmax: 10.8MHz}
  xy1:  {type: xyplot}
chain: [ant, pre, lna, mix, iff, ifamp, det1]
wires:
  - [lo.out, mix.lo]
  - [lo.sweep, xy1.x]
  - [det1.out, xy1.y]
```

From Python: `from hetsim.desktop import run; run(circuit=c)`.

## Live viewer

```
python -m hetsim view examples/vhf_single_conversion.yaml        # opens http://127.0.0.1:8765/
python -m hetsim view examples/hf_dual_conversion.yaml --port 9000 --no-browser
```

The viewer opens a page in your browser. The left side lists a control for every
parameter of every block in the netlist. The right side shows the FFT at any port,
and it recomputes as you drag. Optional parameters, such as a mixer's IIP3 or an
amplifier's OP1dB, have a checkbox that switches them in or out.

- **Colours.** Fundamental mixer products are orange, both stems and sweep tracks.
  That means the 1×1 conversion of each signal tone through the LOs (`LO - RF`,
  `LO2 + RF - LO1`, `LO - B1`, `LO1 - RF~3`). LO-by-LO leakage (`LO2 - LO1`),
  intermods and m×n spurs stay blue (stems) or aqua (tracks). LOs are found by
  tracing back from each mixer's LO port. A time-domain bin takes the colour of its
  strongest product. Tone-solver rings are grey.
- **Phase noise editor.** Each cw, tune or sweep tone has an optional "phase noise"
  row. Tick it to get a table of offset / dBc/Hz points to edit, add (a decade out,
  10 dB lower) or remove, plus ±5 dB buttons that shift the whole profile. A new
  profile starts from an example shape for you to edit; it is not a model of any
  real oscillator.
- **RBW trace.** RBW (1 kHz to 3 MHz) draws the spectrum as a swept analyser
  would. Each line goes through a Gaussian RBW filter (−3 dB at ±RBW/2), and a peak
  detector per display point keeps narrow lines from being missed. Phase noise is
  added as density × noise bandwidth. The stems fade underneath and stay
  hoverable. The **Phase noise** switch (shown when a source has phase noise) turns
  it off for comparison.
- **Mixer depth.** `max_order` on each mixer (a slider with an on/off box) limits
  the spur-table products it generates. It's also a speed control: on a two-mixer
  analyser chain, max order 4 took an update from 62 ms to 27 ms.
- **Source modes.** Each source tone has a mute checkbox and a mode menu (`cw`,
  `tune`, `sweep`, `wideband`). The band controls appear when the mode needs them.
  With sweep tones present, the FFT also shows each product's track across the
  sweep in green, with a □ at its worst level. Hover a point to see the step and
  the tone frequencies there. **Band sweep** in the header hides the tracks. The
  tracks come from the tone solver.
  - **Steps** sets how many points the sweep computes, independently of the
    tones' `points`. Left empty, it uses the tones' points, reduced if needed to
    keep each update to about 0.5 s; a value you type is used up to a 10 s cap.
    The page says when it uses fewer steps.
  - **Interpolation** (linear or smooth, ×4 to ×16) fills in the tracks between
    computed steps, in the browser. Frequencies come out exact, because each
    product's frequency is linear in the swept frequencies. Levels are estimated:
    the smooth option is a monotone cubic, so it never overshoots the computed
    points. The □ worst case is always a computed step, and hovering an
    interpolated point says it's interpolated. Interpolation can't reveal a
    response narrower than the step spacing, such as a product crossing a narrow
    IF filter between two steps. Add steps for that.
  - **Steps on output grid.** Instead of spreading the steps evenly over the swept
    band, this places them so that one product, the *anchor*, lands on a fixed grid
    of output frequencies (from–to, with Steps points). The "view" button fills in
    the plot's current range. The anchor is the selected track; with none selected,
    it's the conversion of the signal (the first non-LO source's tone, the middle
    one of a comb). A product's frequency is linear in the sweep position, so each
    output frequency gives its step directly, using the other tones' current
    frequencies. The samples therefore stay on the same output frequencies when
    you move the RF. A product that crosses the window twice gets steps on both
    passes. From Python: `analysis.output_grid_fractions()` and
    `simulate_sweep(fractions=...)`.
  - **One track.** Click a green track to show only that product; the others dim
    and stay clickable. Click it again, or press Esc, to show all. The list beside
    the sweep controls offers every product, strongest worst case first.
- **Limits.** Click a slider's label to set its min and max. **Save limits** stores
  them in the netlist (`limits:`), and the slider then keeps to them.
- **Port selection.** Pick any port in the circuit. You can view the wave
  *leaving* the port or the wave *arriving* at it (for example, LO radiated back
  out of the antenna).
- **Solver.** Choose time domain (FFT bins), TD + tone overlay (the tone solver's
  power sum per bin shown as rings), or tone only.
- **Hover.** Shows a bin's frequency, its level and which products the tone
  solver puts there.
- **Zoom.** Drag or use the wheel to zoom; double-click to reset.
- **Waveform strip.** Shows one record period of the time waveform below the FFT.
- **Resolution.** Sets the FFT grid. Source frequency sliders snap to it, so tones
  always stay on bins. A coarser grid gives a smaller record and faster updates.
- **Schematic.** The panel above the FFT draws the circuit with RF symbols. Each
  block shows its key values, which follow the sliders, and the probed port is
  marked. You can edit the circuit from the schematic:
  - Drag a block to move it. Positions go in the netlist's `layout`. Blocks with no
    saved position are placed automatically, with each LO branch above its mixer.
  - Drag from one port to another to connect them. Each port takes one wire, so the
    new wire replaces any old one.
  - Click a port to probe it. Click a block to open its parameters.
  - Use **+ Add block** to add a block. Select a block or wire and press Del to
    delete it. Double-click a block to rename it.
  - **Undo** (Ctrl+Z) steps back through netlist changes, including those applied
    from the text editor.

  Wheel zooms, dragging the background pans, and F or a background double-click fits
  the view. Drag the panel's bottom edge to resize it.

  Edits change only the affected parts of the netlist text, so your comments stay.
  A `chain:` is kept until one of its links is rewired; then it's written out as
  explicit `wires`. The sliders keep their values through schematic edits.
- **Netlist editor.** The **Netlist** button swaps the parameter list for the netlist
  source, comments included. **Apply** (Ctrl+Enter) rebuilds the circuit from the text,
  so you can add, remove, rename or rewire blocks. **Save** (Ctrl+S) applies the text
  and then writes it back to the netlist file. If the text doesn't parse or a block
  won't build, the page shows the error, marks the line, and keeps the old circuit.
  Save won't overwrite a file that changed on disk unless you confirm. Slider changes
  aren't in the text, so applying the text resets them. **Pull slider values into
  text** rewrites the text from the current settings instead, which drops the
  comments (Ctrl+Z undoes it).
- **Export netlist.** Downloads the circuit with the current settings as YAML.

It uses only the Python standard library and a single self-contained page, with
no internet access needed. The server listens on 127.0.0.1 only. Blocks whose
parameters didn't change keep their cached filter responses and LO tables
between updates. Typical update times:

| Netlist | Record length | Update time |
|---|---|---|
| VHF example | N = 21 504 | about 25 ms |
| HF dual-conversion example | N = 141 120 | about 250 ms |

The page shows the compute time for each update.

From Python: `from hetsim.viewer import serve; serve(circuit=c)`.

## Analysis

`find_responses(circuit, source, port, band, fmin, fmax)` finds every stimulus
frequency that gets into the output band: wanted, image, IF feedthrough, half-IF,
m×n and multi-conversion spurs. It reports gain and rejection for each, exactly,
however narrow the IF is.

It works by running one probe simulation. Each output product c·f + K predicts
where it would land in the band, and each of those frequencies is then simulated.
`response_sweep()` gives the swept curve, but a coarse sweep will step over
responses as narrow as the IF bandwidth.

The CLI wraps the same calls:

```
python -m hetsim run examples/vhf_single_conversion.yaml -p ifamp.out --sort power --ref "LO - RF"
python -m hetsim run examples/vhf_single_conversion.yaml -p ant.out --incident
python -m hetsim responses examples/vhf_single_conversion.yaml --source ant --port ifamp.out \
    --band 10.6MHz:10.8MHz --range 1MHz:500MHz --mute ant --plot responses.png
```

`--mute` silences the other tones in a source, such as blockers, during a response
search.

## Examples

- `examples/vhf_single_conversion.yaml` plus `vhf_example.py`: wanted signal,
  image-frequency signal and the IM3 of two blockers all landing on 10.7 MHz. The
  script also compares spurious responses with and without the preselector.
- `examples/td_vs_tone.py`: a two-tone LNA test swept into compression with both
  solvers, plus the VHF IF spectrum from the TD solver overlaid on the tone result.
- `examples/hf_single_vs_dual.py`: an HF receiver taken straight to 455 kHz, with
  0 dB image rejection, against up-conversion to 45 MHz with a roofing filter and a
  second conversion to 455 kHz, with about 80 dB rejection.

## Limitations

These are things that are not modelled yet, roughly in the order they would matter:

- **Port mismatch / VSWR.** Interfaces are ideal 50 Ω. The wave solver is built
  for it, so each block would gain a per-port Γ(f).
- **Noise.** There is no NF, sensitivity or image-noise folding yet. A noise
  density riding along with the tones is the natural next layer.
- **Phase and coherence.** In the tone solver, same-frequency products add in
  power. The TD solver handles phase, but there is no quadrature (IQ) mixer block
  yet for image-reject architectures.
- **LO model.** The strongest tone at the LO port is the LO. Other tones there only
  leak (phase noise on the LO itself is modelled; see Phase noise).
- **Two-tone nonlinearity.** It uses the strongest `im_max_tones` tones (default 24)
  and does not include triple beats.

## Tests

`pytest tests` runs 58 checks. They compare against hand-calculated values:
mixer products, isolation, spur scaling, IM3/IM2 and harmonics, P1dB, filter
edges and phase, reverse leakage, image rejection and netlist round trip. They
also check that the TD and tone solvers agree, and that same-bin products add
coherently in the TD solver. Viewer back-end tests cover parameter discovery,
edits, frequency snapping and the HTTP API.

Licence: GPL-3.0-or-later.
