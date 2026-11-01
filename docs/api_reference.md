# API reference

Everything here imports from `hetsim`. The docstrings have the details.

## Circuit

| Name | What it does |
|---|---|
| `Circuit(name)` | The block graph |
| `add(name, block)`, `remove(name)`, `replace(name, block)` | Change the blocks; `replace` keeps the wiring |
| `connect(a, b)`, `disconnect(a)`, `chain(*refs)` | Wire ports, written `"blk.port"` or `"blk"` for the default port |
| `simulate(settings)` | Tone solver; returns a `Result` |
| `simulate_td(settings, td)` | Time-domain solver; returns a `TDResult` |
| `simulate_sweep(settings, points, workers)` | Steps the `mode: sweep` tones; returns a `BandSweep` |
| `Settings` | Solver options: `floor_dbm`, `max_tones`, `im_max_tones`, `tol_db`, `max_iter`, `td` |

## Blocks

| Class | Netlist type | Models |
|---|---|---|
| `Source` | `source` | CW, tuned, swept and wideband tones, harmonics, phase noise |
| `Mixer` | `mixer` | Conversion, isolation, spur tables, IM, `nf_db` |
| `Amplifier` | `amplifier` | Gain, intercepts, Rapp compression, reverse isolation, `nf_db` |
| `Filter` | `filter` | Analytic Butterworth, Chebyshev, elliptic and Bessel responses |
| `IdealFilter` | `ideal_filter` | Brick wall |
| `SParamFilter` | `sparam` | A Touchstone `.s2p` file |
| `Attenuator` | `attenuator` | Flat loss |
| `Splitter` | `splitter` | 2-way splitter or combiner |
| `Load` | `load` | Termination |
| `Detector`, `XYPlot` | `detector`, `xyplot` | Power reading and X/Y display, for the desktop app (import from `hetsim.blocks`) |

All RF blocks take `vswr`. `register(kind)` adds a block type, and `REGISTRY` holds them.

## Results

| Name | What it does |
|---|---|
| `Result[ref]`, `Result.incident(ref)` | The `Spectrum` leaving or arriving at a port |
| `Result.summary(port)`, `Result.warning_text()` | Text output |
| `TDResult.waveform(ref)`, `incident_waveform(ref)` | Sampled waveforms |
| `TDResult.spectrum(ref)`, `incident_spectrum(ref)` | `TDSpectrum` from the FFT |
| `Spectrum.band(fmin, fmax)`, `total_dbm(fmin, fmax)`, `power(label)`, `strongest(n)` | Queries |
| `Spectrum.table(...)` | A text table |

## Analyses

| Function | Returns |
|---|---|
| `find_responses(circuit, source, port, band, fmin, fmax)` | `ResponseTable`: every stimulus frequency reaching the band |
| `response_sweep(circuit, source, port, band, freqs)` | `Sweep`: a swept response curve |
| `band_sweep(circuit, ...)` | `BandSweep`: each product followed across a sweep |
| `phase_noise_skirts(circuit, ref)` | A list of `Skirt`: noise density around each product |
| `noise_figure(circuit, source, port, band, tone)` | `NoiseFigure`: NF, gain, noise, budget, responses |
| `noise_sweep(circuit, source, port, band, tone)` | `NoiseSweep`: NF at each sweep step |
| `port_match(circuit, port, freq)` | `PortMatch`: VSWR, return loss, mismatch loss, worst case |
| `match_sweep(circuit, port, freqs)` | `MatchSweep`: the same across frequency |
| `mismatch_report(circuit, tone=None)` | `MismatchReport`: both ends of every wire, with ripple |

## Netlists

| Function | What it does |
|---|---|
| `load(path)` | Returns `(Circuit, Settings)` from YAML or JSON |
| `save(circuit, path, settings)` | Writes a netlist |
| `from_dict(d)`, `to_dict(circuit, settings)` | The same, on dictionaries |
