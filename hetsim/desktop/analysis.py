"""Noise figure and VSWR for the circuit as the sliders have it now, shown as text tables."""
from __future__ import annotations

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QComboBox, QDialog, QFormLayout, QHBoxLayout, QLineEdit, QPlainTextEdit, QPushButton,
                               QVBoxLayout)

from ..netlist import from_dict
from ..smallsignal import mismatch_report, noise_figure


class AnalysisDialog(QDialog):
    def __init__(self, model, edits: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Noise figure and VSWR")
        self.resize(760, 560)
        self.circuit, self.settings = from_dict(model.netlist(edits), model.base_dir)
        self.source = QComboBox()
        self.source.addItems([n for n, b in self.circuit.blocks.items() if b.kind == "source"])
        self.tone = QComboBox()
        self.source.currentTextChanged.connect(self._tones)
        self.port = QComboBox()
        self.port.addItems([f"{n}.{p}" for n, b in self.circuit.blocks.items() for p in b.ports
                            if p not in b.controls and b.kind != "source"])
        self.port.setCurrentIndex(self.port.count() - 1)
        self.band = QLineEdit()
        self.band.setPlaceholderText("output band, e.g. 10.6MHz:10.8MHz (empty: spot figure)")
        form = QFormLayout()
        form.addRow("Source", self.source)
        form.addRow("Signal tone", self.tone)
        form.addRow("Output port", self.port)
        form.addRow("Band", self.band)
        run = QPushButton("Noise figure")
        run.clicked.connect(self.run_noise)
        match = QPushButton("VSWR at every wire")
        match.clicked.connect(self.run_match)
        buttons = QHBoxLayout()
        buttons.addWidget(run)
        buttons.addWidget(match)
        buttons.addStretch(1)
        self.out = QPlainTextEdit(readOnly=True)
        self.out.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addLayout(buttons)
        lay.addWidget(self.out, 1)
        self._tones(self.source.currentText())

    def _tones(self, name: str):
        self.tone.clear()
        if name:
            self.tone.addItems([t.name for t in self.circuit[name].specs if t.mode != "wideband"])

    def run_noise(self):
        band = self.band.text().strip()
        try:
            parts = band.split(":") if band else None
            if parts and len(parts) != 2:
                raise ValueError("the band is written low:high, e.g. 10.6MHz:10.8MHz")
            r = noise_figure(self.circuit, self.source.currentText(), self.port.currentText(), parts,
                             self.tone.currentText(), self.settings)
        except (ValueError, KeyError) as e:
            self.out.setPlainText(str(e))
            return
        self.out.setPlainText(r.table())

    def run_match(self):
        tone = self.tone.currentText() or None
        self.out.setPlainText(mismatch_report(self.circuit, self.settings, tone).table())
