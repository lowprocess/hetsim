# hetsim documentation

`hetsim` simulates mixer, filter and amplifier chains in the tone domain, for frequency
planning and for comparing superheterodyne architectures. The [project README](../README.md)
is the full guide to the model and the viewers. These pages are the short route in.

| Page | Read it for |
|---|---|
| [usage.md](usage.md) | Building a circuit in Python or YAML, and running each analysis from the command line |
| [architecture.md](architecture.md) | How the solvers, blocks and analyses fit together, and how to add a block |
| [api_reference.md](api_reference.md) | The public classes and functions |

## Installation

Python 3.10 or newer. From the repository root:

```bash
pip install -e .              # the engine: numpy and scipy
pip install -e '.[yaml]'      # YAML netlists
pip install -e '.[plot]'      # saving plots from the CLI
pip install -e '.[desktop]'   # the desktop app (PySide6, includes YAML)
pip install -e '.[dev]'       # everything the tests need, apart from PySide6
```

## Development

```bash
python -m pytest tests
```

The desktop tests run offscreen and are skipped if PySide6 isn't installed. Tests compare
against hand-calculated values wherever one exists, so prefer that to snapshotting output.
Netlists without a new feature's parameters should simulate exactly as they did before.
