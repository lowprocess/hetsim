"""Colours for the desktop viewer: the browser viewer's tokens, light and dark."""
from __future__ import annotations

from PySide6.QtGui import QColor, QGuiApplication, QPalette

LIGHT = {
    "surface-0": "#f4f3f0", "surface-1": "#fcfcfb", "surface-2": "#ffffff",
    "border": "#dddcd7", "grid": "#e7e6e2",
    "text-primary": "#0b0b0b", "text-secondary": "#52514e", "text-muted": "#85847f",
    "series-1": "#2a78d6", "series-2": "#eb6834", "series-3": "#1baf7a",
    "accent": "#2a78d6", "warn": "#b97700", "error": "#d03b3b", "select": "#1f2a78d6",
}
DARK = {
    "surface-0": "#121211", "surface-1": "#1a1a19", "surface-2": "#232321",
    "border": "#383835", "grid": "#2c2c2a",
    "text-primary": "#f0efec", "text-secondary": "#c3c2b7", "text-muted": "#8c8b84",
    "series-1": "#3987e5", "series-2": "#d95926", "series-3": "#199e70",
    "accent": "#3987e5", "warn": "#e0a320", "error": "#e66767", "select": "#2e3987e5",
}


def is_dark() -> bool:
    app = QGuiApplication.instance()
    if app is None:
        return False
    return app.palette().color(QPalette.ColorRole.Window).lightness() < 128


def c(name: str) -> QColor:
    """A colour token for the current light/dark scheme ('select' carries alpha as #aarrggbb)."""
    return QColor((DARK if is_dark() else LIGHT)[name])
