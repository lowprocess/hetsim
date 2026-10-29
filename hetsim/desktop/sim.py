"""Runs simulations off the GUI thread: the newest request wins, older pending ones are dropped."""
from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, Signal


class SimRunner(QObject):
    """Calls model.simulate(req) on a worker thread and emits `done(result, req)` on the GUI thread.

    While a simulation runs, further submissions replace each other, so dragging a slider
    simulates the latest position, not every intermediate one.
    """
    done = Signal(object, object)
    busy = Signal(bool)

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self._pending = None
        self._cv = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="hetsim-sim", daemon=True)
        self._thread.start()

    def submit(self, req: dict) -> None:
        with self._cv:
            self._pending = req
            self._cv.notify()

    def run_now(self, req: dict) -> dict:
        """Simulate on the calling thread (tests, first paint)."""
        return self._simulate(req)

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()

    def _loop(self):
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                req, self._pending = self._pending, None
            self.busy.emit(True)
            res = self._simulate(req)
            with self._cv:
                more = self._pending is not None
            self.busy.emit(more)
            self.done.emit(res, req)

    def _simulate(self, req):
        try:
            return self.model.simulate(req)
        except Exception as e:  # noqa: BLE001  - shown in the window, the app keeps running
            return {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=3)}
