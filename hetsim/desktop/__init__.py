"""Native desktop viewer (PySide6): python -m hetsim app netlist.yaml

The browser viewer's features in a Qt window, driving ViewerModel in-process: no server, no
browser. Needs the optional dependency PySide6 (pip install hetsim[desktop]).
"""


def run(*args, **kw):
    from .app import run as _run
    return _run(*args, **kw)


__all__ = ["run"]
