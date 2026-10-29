"""The start page shown when the app opens without a netlist: new, open, recent, examples."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget


def examples_dir() -> Path | None:
    """The examples folder of a source checkout (not installed with the package)."""
    d = Path(__file__).resolve().parent.parent.parent / "examples"
    return d if d.is_dir() else None


class StartPage(QWidget):
    newRequested = Signal()
    openRequested = Signal()
    openPath = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QHBoxLayout(self)
        outer.addStretch(1)
        col = QWidget()
        col.setMaximumWidth(560)
        v = QVBoxLayout(col)
        v.addStretch(1)
        title = QLabel("hetsim")
        title.setStyleSheet("font-size: 26px; font-weight: 600")
        sub = QLabel("Tone and time-domain simulation of mixer, filter and amplifier chains")
        sub.setStyleSheet("color: gray")
        v.addWidget(title)
        v.addWidget(sub)
        v.addSpacing(18)
        row = QHBoxLayout()
        new = QPushButton("New circuit")
        new.setToolTip("Start from one source wired to a load, and build with + Add block (Ctrl+N)")
        opn = QPushButton("Open netlist…")
        opn.setToolTip("Open a YAML or JSON netlist (Ctrl+O)")
        opn.setDefault(True)
        for b in (new, opn):
            b.setMinimumHeight(34)
            row.addWidget(b)
        new.clicked.connect(self.newRequested)
        opn.clicked.connect(self.openRequested)
        v.addLayout(row)
        v.addSpacing(14)
        self.recent_label = QLabel("Recent")
        self.recent_label.setStyleSheet("font-weight: 600")
        self.recent = self._list()
        v.addWidget(self.recent_label)
        v.addWidget(self.recent)
        ex = examples_dir()
        self.examples = None
        if ex is not None:
            files = sorted(p for p in ex.iterdir() if p.suffix.lower() in (".yaml", ".yml", ".json"))
            if files:
                lab = QLabel("Examples")
                lab.setStyleSheet("font-weight: 600")
                self.examples = self._list()
                for p in files:
                    self._add(self.examples, p)
                v.addSpacing(8)
                v.addWidget(lab)
                v.addWidget(self.examples)
        hint = QLabel("or drop a netlist file on this window")
        hint.setStyleSheet("color: gray")
        v.addSpacing(8)
        v.addWidget(hint)
        v.addStretch(2)
        outer.addWidget(col, 3)
        outer.addStretch(1)

    def _list(self) -> QListWidget:
        lw = QListWidget()
        lw.setMaximumHeight(150)
        # double-click or Enter (or one click, where the desktop is set to open on single click)
        lw.itemActivated.connect(lambda it: self.openPath.emit(it.data(Qt.ItemDataRole.UserRole)))
        return lw

    @staticmethod
    def _add(lw: QListWidget, path: Path):
        it = QListWidgetItem(f"{path.name}    {path.parent}")
        it.setData(Qt.ItemDataRole.UserRole, str(path))
        it.setToolTip(str(path))
        lw.addItem(it)

    def set_recent(self, paths: list[str]):
        self.recent.clear()
        for p in paths:
            self._add(self.recent, Path(p))
        self.recent_label.setVisible(bool(paths))
        self.recent.setVisible(bool(paths))
