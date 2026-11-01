# Roadmap

What hetsim doesn't do yet, and what comes next. Each entry says what happens today, so you can judge whether it matters to your design.

## Known bug

**Low-order Chebyshev II filters can show gain far out of band.** A filter normalises its peak to 0 dB by searching a limited frequency range. If the real peak lies outside that range, the response goes above 0 dB out there. Measured with `stop_db: 60`:

| Filter | Peak gain | At |
|---|---|---|
| Bandstop 90–110 MHz, order 1 | +25.3 dB | 1 kHz |
| Bandstop 90–110 MHz, order 2 | +0.12 dB | 1.25 kHz |
| Lowpass 1 GHz, order 1 | +3.0 dB | 1 kHz |

Butterworth and elliptic filters of orders 1, 2, 3 and 5 stayed within 0.01 dB of 0 dB in all four response types, as did Chebyshev II at orders 3 and 5. The fix is to find the peak on the normalised prototype, where its location is known.

## Next up

In priority order:

1. **A passive mixer model.** A mixer with reactive filters on both ports doesn't settle, and that is the most common receiver layout. The spur table gives no phases, so all conversion paths add in phase and a pass has amplitude gain of about 1. Today the solvers stop and warn, and a pad of about 6 dB is needed. The fix is to give the LO harmonics the phases of a real switching waveform, so that conversion stays passive when waves come back.
2. **Find the slow time-domain case.** Phase noise filling the band, with reflections, takes 24 s against 0.44 s without reflections on a 1.25M-point record, over 18 rounds. A sparse factorisation only took it from 30 s to 24 s, so the time goes elsewhere. Repeated FFTs are a guess, and it needs profiling.
3. **Line impedance.** A `line` is matched, with mismatch only from the `vswr` of what it joins, at zero phase there. A real Z0 would make a line transform impedance, and frequency-dependent loss would make long cables realistic.
4. **Choices for reactive filters.** Add a setting for series-first or shunt-first ladders, which have opposite reflection signs, and make the 0.1 dB best stopband return loss a parameter.
5. **Reflection settings in the viewers.** `vswr: reactive`, `vswr: file` and `settings: coherent` can only be changed in the netlist.

## Limitations

### Reflections and phase

- **Mixer spur phases.** See item 1. Spurs have levels from the table but no phases of their own, so interference between them is only indicative.
- **`vswr` numbers and tables have no phase.** A port reflects at zero phase, which is the worst case between zero-length neighbours. Only `sparam` with `vswr: file` and reactive filters have real phase.
- **Reactive filter phase is one realisation.** It assumes a series-first ladder, and Bessel filters take the minimum-phase choice. The magnitude doesn't depend on this.
- **Reactive filter orders.** Above about order 12 for elliptic and 16 for the others, the polynomials lose too much precision. These orders give an error.
- **`ideal_filter` can't be reactive.** A brick wall has no phase to derive. `splitter`, `attenuator` and other blocks take only a plain `vswr`.
- **`sparam` is 2-port only.** It reads Touchstone v1 `.s2p` data. Outside the file's range it holds the edge value and warns.

### Tone solvers

- **Different products on one frequency aren't summed.** They stay separate and labelled, even in the coherent solver. Use the time-domain solver for image against wanted in one bin.
- **The power solver adds in power.** That gives the average of any ripple, and reads optimistic on IM from several stages. The coherent solver and the budget's cascade don't have this problem.
- **The two prune differently.** The power solver drops products below the floor as it goes, so the coherent solver can list more tones.
- **Two-tone nonlinearity.** It uses the strongest `im_max_tones` tones (default 24) and has no triple beats. The time-domain solver has both.
- **IM3 stays on its 3:1 line.** The tone model has no IM5 or compression bend. The time-domain solver shows them.
- **LO model.** The strongest tone at a mixer's LO port is the LO, and other tones there only leak.

### Time-domain solver

- **Size.** It samples at RF, so N ≈ 2.5 · fmax / resolution. It is fine up to a few million samples.
- **Common grid.** Tones with no coarse common grid need a coarser `resolution`, with sources moved onto it. Log-spaced combs move each tone to its nearest bin, and it says so.

### Noise

- **Noise figure adds in power.** Correlated noise across reflections is approximate, and a line's ripple doesn't appear in the NF.
- **LO noise isn't in the noise figure.** LO wideband noise isn't converted by the mixer there, and phase noise has its own analysis.
- **Sources of noise.** It is T0 thermal plus `nf_db`, flat with frequency, and the source is always at T0. There is no flicker noise.
- **No noise on a spectrum.** Neither solver draws a noise floor.

### Receiver blocks

- **`iq_mixer` is one block.** There are no separate I and Q ports, so no filters inside the branches and only one `adc` for a complex baseband. A product exactly at DC is dropped, and DC offset isn't modelled.
- **`adc` clipping.** Only the time-domain solver clips. The tone solver warns instead.
- **Digital filtering.** Filters after an `adc` count as analogue for noise, so they add a little kT0 that a digital filter wouldn't.
- **Spur tables.** The `dbm_generic` preset holds illustrative values, not any real part. Load a chart from CSV for real figures.

### Tolerances

- **Draws are independent.** Parameters that move together on a real part, such as gain and NF, can't be correlated.
- **Which tolerance decides is approximate.** It uses squared correlation, which is rough where a response is strongly nonlinear, such as at a filter edge.
