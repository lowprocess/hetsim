"""Local web viewer: python -m hetsim view netlist.yaml

Serves one page on 127.0.0.1 and a small JSON API. Only the standard library is used
for the server; the page draws its own plots, so nothing is fetched from the internet.
"""
from __future__ import annotations

import json
import signal
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

from .model import FileChangedError, NetlistTextError, ViewerModel, dump_text


def _page() -> bytes:
    return resources.files("hetsim.viewer").joinpath("static/index.html").read_bytes()


class _Handler(BaseHTTPRequestHandler):
    model: ViewerModel = None
    quiet = True

    def log_message(self, fmt, *args):
        if not self.quiet:
            super().log_message(fmt, *args)

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, allow_nan=False, default=_clean).encode())

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, _page(), "text/html; charset=utf-8")
        elif self.path == "/api/model":
            try:
                self._json(self.model.info())
            except Exception as e:  # noqa: BLE001
                self._json({"error": str(e)}, 500)
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/sim":
                self._json(_finite(self.model.simulate(req)))
            elif self.path == "/api/analysis":
                from .analysis import analyse, options
                self._json(options(self.model) if req.get("kind") == "options" else analyse(self.model, req))
            elif self.path == "/api/export":
                d = self.model.netlist(req.get("params", {}), req.get("resolution"))
                self._send(200, dump_text(d, req.get("format", "yaml")).encode(), "text/plain; charset=utf-8")
            elif self.path == "/api/netlist":
                self._netlist(req)
            elif self.path == "/api/schematic":
                try:
                    self._json(self.model.edit(req))
                except NetlistTextError as e:
                    self._json({"error": str(e), "line": e.line})
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as e:  # noqa: BLE001  - report to the page, keep serving
            self._json({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=3)}, 200)

    def _netlist(self, req):
        """Apply edited netlist text, and optionally write it back to the file."""
        try:
            info = self.model.set_text(str(req["text"]))
        except NetlistTextError as e:
            return self._json({"error": str(e), "line": e.line})
        if req.get("save"):
            try:
                self.model.save(force=bool(req.get("force")))
            except FileChangedError as e:
                return self._json({**info, "save_error": str(e), "changed_on_disk": True})
            except OSError as e:
                return self._json({**info, "save_error": f"{type(e).__name__}: {e}"})
            info["saved"] = True
        self._json(info)


def _clean(o):
    return str(o)


def _finite(o):
    """Replace inf/nan (e.g. -inf dBm) so the JSON is valid."""
    if isinstance(o, float):
        return o if o == o and abs(o) != float("inf") else (-400.0 if o < 0 else 400.0)
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    return o


def make_server(model: ViewerModel, port: int = 8765, host: str = "127.0.0.1"):
    handler = type("Handler", (_Handler,), {"model": model})
    return ThreadingHTTPServer((host, port), handler)


def serve(netlist=None, circuit=None, settings=None, port: int = 8765, open_browser: bool = True,
          host: str = "127.0.0.1", workers: int | None = None):
    """Start the viewer for a netlist file or an in-memory Circuit. Blocks until Ctrl-C.

    workers: processes for band sweeps (None: one per physical core, 1: in-process).
    """
    model = ViewerModel.from_file(netlist) if netlist else ViewerModel.from_circuit(circuit, settings)
    model.sweep_workers = workers
    if workers is None or workers > 1:   # start the sweep workers while the page loads
        from ..analysis import warm_pool
        threading.Thread(target=warm_pool, daemon=True).start()
    srv = make_server(model, port, host)
    url = f"http://{host}:{srv.server_address[1]}/"
    print(f"hetsim viewer on {url}  (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    if threading.current_thread() is threading.main_thread():   # kill = Ctrl-C: tidy up the sweep workers too
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        from ..analysis import shutdown_pool
        shutdown_pool()
