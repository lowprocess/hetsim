"""Mixer spur tables (m x n charts).

Entry (m, n) gives the level, in dB below the desired 1x1 IF output, of the
product |m*fLO +/- n*fRF| when the RF input is at `ref_rf_dbm`. As in the usual
manufacturer charts, an n-th order RF product moves (n-1) dB per dB of RF
drive, so at RF level P the product is
    P_out(m, n) = P - CL - dBc(m, n) + (n - 1) * (P - ref_rf_dbm)

CSV format (first data column is the RF harmonic n, header row is the LO harmonic m):

    # ref_rf_dbm = -10
    n\\m, 0,  1,  2,  3
    1,   ,   0, 35, 15
    2, 55,  60, 60, 60
    3, 75,  70, 75, 65

Blank cells mean "not modelled". (n=1, m=0) is RF->IF feedthrough and comes
from the mixer's rf_if_isolation_db, not the table.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path


class SpurTable:
    def __init__(self, entries: dict[tuple[int, int], float], ref_rf_dbm: float = -10.0, name: str = ""):
        self.entries = {(int(m), int(n)): float(v) for (m, n), v in entries.items()}
        self.entries[(1, 1)] = 0.0
        self.entries.pop((0, 1), None)
        self.ref_rf_dbm = float(ref_rf_dbm)
        self.name = name

    def level(self, m: int, n: int, p_in_dbm: float, conversion_loss_db: float) -> float | None:
        dbc = self.entries.get((m, n))
        if dbc is None:
            return None
        return p_in_dbm - conversion_loss_db - dbc + (n - 1) * (p_in_dbm - self.ref_rf_dbm)

    @property
    def max_m(self) -> int:
        return max(m for m, _ in self.entries)

    @property
    def max_n(self) -> int:
        return max(n for _, n in self.entries)

    @classmethod
    def from_csv(cls, path_or_text, ref_rf_dbm: float | None = None, name: str = "") -> "SpurTable":
        s = str(path_or_text)
        text = s if "\n" in s else Path(s).read_text()
        ref = -10.0
        rows = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                if "ref_rf_dbm" in s:
                    ref = float(s.split("=", 1)[1])
                continue
            rows.append(s)
        reader = list(csv.reader(io.StringIO("\n".join(rows))))
        ms = [int(x) for x in reader[0][1:]]
        entries = {}
        for r in reader[1:]:
            n = int(r[0])
            for m, cell in zip(ms, r[1:]):
                cell = cell.strip()
                if cell:
                    entries[(m, n)] = float(cell)
        return cls(entries, ref_rf_dbm if ref_rf_dbm is not None else ref, name or str(path_or_text))

    def to_csv(self) -> str:
        ms = range(0, self.max_m + 1)
        lines = [f"# ref_rf_dbm = {self.ref_rf_dbm}", "n\\m," + ",".join(str(m) for m in ms)]
        for n in range(1, self.max_n + 1):
            cells = []
            for m in ms:
                v = self.entries.get((m, n))
                cells.append("" if v is None else f"{v:g}")
            lines.append(f"{n}," + ",".join(cells))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"SpurTable({self.name or 'custom'}, {len(self.entries)} entries, ref {self.ref_rf_dbm} dBm)"


# Illustrative numbers in the range typical of a diode double-balanced mixer at
# +7 dBm LO and -10 dBm RF. Not any particular part: load the real chart for your mixer.
_GENERIC_DBM = """
# ref_rf_dbm = -10
n\\m, 0,  1,  2,  3,  4,  5
1,   ,   0, 35, 15, 40, 22
2, 60,  62, 65, 62, 70, 68
3, 80,  72, 80, 68, 80, 72
4, 95,  95, 95, 95, 95, 95
5, 100, 98, 100, 98, 100, 98
"""

PRESETS = {
    "ideal": lambda: SpurTable({(1, 1): 0.0}, name="ideal"),
    "dbm_generic": lambda: SpurTable.from_csv(_GENERIC_DBM, name="dbm_generic"),
}


def get_spur_table(spec) -> SpurTable:
    if spec is None:
        return PRESETS["ideal"]()
    if isinstance(spec, SpurTable):
        return spec
    if isinstance(spec, dict):  # {"ref_rf_dbm": -10, "entries": {"1,1": 0, "3,1": 15}}
        ent = {tuple(int(x) for x in k.split(",")): v for k, v in spec["entries"].items()}
        return SpurTable(ent, spec.get("ref_rf_dbm", -10.0), spec.get("name", ""))
    if spec in PRESETS:
        return PRESETS[spec]()
    return SpurTable.from_csv(spec)
