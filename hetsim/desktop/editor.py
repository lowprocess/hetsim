"""Netlist text editor: the netlist source with line numbers, error marking, Apply / Save / Revert."""
from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QKeySequence, QPainter, QShortcut, QTextCursor, QTextFormat
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QTextEdit, QVBoxLayout, QWidget

from . import theme


class _Gutter(QWidget):
    def __init__(self, ed: "CodeEdit"):
        super().__init__(ed)
        self.ed = ed

    def sizeHint(self):
        return QSize(self.ed.gutter_width(), 0)

    def paintEvent(self, e):
        self.ed.paint_gutter(e)


class CodeEdit(QPlainTextEdit):
    """Plain-text editor with a line-number gutter; Tab inserts two spaces, Enter keeps the indent."""

    def __init__(self):
        super().__init__()
        f = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        f.setPointSize(10)
        self.setFont(f)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(" ") * 2)
        self.gutter = _Gutter(self)
        self.err_line = None
        self.blockCountChanged.connect(lambda _: self.setViewportMargins(self.gutter_width(), 0, 0, 0))
        self.updateRequest.connect(self._scroll_gutter)
        self.setViewportMargins(self.gutter_width(), 0, 0, 0)

    def gutter_width(self):
        return 14 + self.fontMetrics().horizontalAdvance("9") * max(3, len(str(self.blockCount())))

    def _scroll_gutter(self, rect, dy):
        if dy:
            self.gutter.scroll(0, dy)
        else:
            self.gutter.update(0, rect.y(), self.gutter.width(), rect.height())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        cr = self.contentsRect()
        self.gutter.setGeometry(QRect(cr.left(), cr.top(), self.gutter_width(), cr.height()))

    def paint_gutter(self, e):
        qp = QPainter(self.gutter)
        qp.fillRect(e.rect(), theme.c("surface-1"))
        block = self.firstVisibleBlock()
        n = block.blockNumber()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + round(self.blockBoundingRect(block).height())
        while block.isValid() and top <= e.rect().bottom():
            if block.isVisible() and bottom >= e.rect().top():
                bad = self.err_line == n + 1
                f = QFont(self.font())
                f.setBold(bad)
                qp.setFont(f)
                qp.setPen(theme.c("error") if bad else theme.c("text-muted"))
                qp.drawText(0, top, self.gutter.width() - 6, self.fontMetrics().height(),
                            Qt.AlignmentFlag.AlignRight, str(n + 1))
            block = block.next()
            top, bottom = bottom, bottom + round(self.blockBoundingRect(block).height())
            n += 1

    def mark_error(self, line: int | None):
        self.err_line = line
        sels = []
        if line:
            sel = QTextEdit.ExtraSelection()
            col = QColor(theme.c("error"))
            col.setAlpha(40)
            sel.format.setBackground(col)
            sel.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
            sel.cursor = QTextCursor(self.document().findBlockByNumber(line - 1))
            sels.append(sel)
            self.setTextCursor(sel.cursor)
            self.centerCursor()
        self.setExtraSelections(sels)
        self.gutter.update()

    def keyPressEvent(self, e):
        mods = e.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
                                | Qt.KeyboardModifier.MetaModifier)
        if not mods and e.key() == Qt.Key.Key_Tab:
            self.insertPlainText("  ")
            return
        if not mods and e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            line = self.textCursor().block().text()[: self.textCursor().positionInBlock()]
            ind = line[: len(line) - len(line.lstrip(" "))]
            if line.rstrip().endswith(":"):
                ind += "  "
            self.insertPlainText("\n" + ind)
            return
        super().keyPressEvent(e)


class NetlistEditor(QWidget):
    """The netlist source, edited as text. Emits apply(text, save) for the window to act on."""
    applyRequested = Signal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        bar.setContentsMargins(8, 6, 8, 6)
        self.file = QLabel()
        self.state = QLabel()
        self.state.setStyleSheet(f"color: {theme.c('warn').name()}")
        self.revert = QPushButton("Revert")
        self.revert.setToolTip("Discard edits since the last apply")
        self.apply = QPushButton("Apply")
        self.apply.setToolTip("Rebuild the circuit from this text (Ctrl+Enter)")
        self.apply.setDefault(True)
        self.save = QPushButton("Save")
        self.save.setToolTip("Apply, then write this text to the netlist file (Ctrl+S)")
        for w in (self.file, self.state):
            bar.addWidget(w)
        bar.addStretch(1)
        for w in (self.revert, self.apply, self.save):
            bar.addWidget(w)
        v.addLayout(bar)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setContentsMargins(8, 0, 8, 4)
        self.note.hide()
        v.addWidget(self.note)
        self.code = CodeEdit()
        v.addWidget(self.code, 1)
        self.text = ""
        self.can_save = False
        self.code.textChanged.connect(self.refresh)
        self.apply.clicked.connect(lambda: self.applyRequested.emit(self.code.toPlainText(), False))
        self.save.clicked.connect(lambda: self.applyRequested.emit(self.code.toPlainText(), self.can_save))
        self.revert.clicked.connect(self._revert)
        QShortcut(QKeySequence("Ctrl+Return"), self.code,
                  activated=lambda: self.applyRequested.emit(self.code.toPlainText(), False))
        QShortcut(QKeySequence.StandardKey.Save, self, context=Qt.ShortcutContext.WidgetWithChildrenShortcut,
                  activated=lambda: self.applyRequested.emit(self.code.toPlainText(), self.can_save))

    def modified(self) -> bool:
        return self.code.toPlainText() != self.text

    def load(self, text: str, file: str | None, can_save: bool):
        self.text, self.can_save = text, can_save
        if self.code.toPlainText() != text:
            self.code.setPlainText(text)
        self.file.setText(file.rsplit("/", 1)[-1] if file else "in-memory circuit")
        self.file.setToolTip(file or "not loaded from a file: use Export netlist to keep it")
        self.save.setVisible(can_save)
        self.refresh()

    def refresh(self):
        mod = self.modified()
        self.apply.setEnabled(mod)
        self.revert.setEnabled(mod)
        self.state.setText("● modified" if mod else "")
        if self.code.err_line:
            self.code.mark_error(None)

    def show_note(self, text: str | None, kind: str = ""):
        self.note.setVisible(bool(text))
        self.note.setText(text or "")
        col = {"err": theme.c("error"), "warn": theme.c("warn")}.get(kind, theme.c("text-secondary"))
        self.note.setStyleSheet(f"color: {col.name()}")

    def _revert(self):
        self.code.setPlainText(self.text)
        self.show_note(None)
