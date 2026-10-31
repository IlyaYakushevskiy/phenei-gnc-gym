"""Single source of truth for every physical number in the simulation.

Project rule (see CLAUDE.md): no magic physical constants in code. Everything
comes from ``design_docs/params.csv`` and is referred to by its row ID, so that
the provenance of any number in a result is recoverable.

Usage::

    from phenei_gym.params import PARAMS
    g = PARAMS.value("SIM-E01")              # 1.6242 m/s^2
    t_lo, t_hi = PARAMS.range("SIM-P01")     # (3000.0, 6000.0) N
    print(PARAMS.audit())                    # provenance of what this run used
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent.parent / "design_docs" / "params.csv"

#: Statuses that mean "a human has not signed off on this number yet".
UNREVIEWED = frozenset({"Assumption (LLM)", "TBD"})

_RANGE_RE = re.compile(
    r"^\s*(-?[\d.]+(?:e-?\d+)?)\s*(?:to|-)\s*(-?[\d.]+(?:e-?\d+)?)\s*(?:\(.*\))?\s*$",
    re.IGNORECASE,
)


class ParamError(KeyError):
    """Raised when a parameter is missing or cannot be used as requested."""


@dataclass(frozen=True)
class Param:
    id: str
    name: str
    raw: str
    unit: str
    status: str
    source: str
    notes: str

    @property
    def unreviewed(self) -> bool:
        return self.status in UNREVIEWED

    def as_float(self) -> float:
        try:
            return float(self.raw.replace(",", ""))
        except ValueError as exc:
            raise ParamError(
                f"{self.id} ({self.name!r}) is not a scalar: {self.raw!r} [{self.unit}]. "
                f"Use PARAMS.range({self.id!r}) if it is a range, or read it as text."
            ) from exc

    def as_range(self) -> tuple[float, float]:
        m = _RANGE_RE.match(self.raw)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            return (lo, hi) if lo <= hi else (hi, lo)
        try:  # a scalar is a degenerate range
            v = self.as_float()
        except ParamError:
            raise ParamError(
                f"{self.id} ({self.name!r}) is not a range: {self.raw!r} [{self.unit}]"
            ) from None
        return (v, v)

    def __str__(self) -> str:
        return f"{self.id} {self.name} = {self.raw} {self.unit} [{self.status}]"


class Params:
    """Loaded parameter table. Records which IDs a run actually touched."""

    def __init__(self, path: Path = CSV_PATH) -> None:
        self.path = path
        self._rows: dict[str, Param] = {}
        self._used: set[str] = set()
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                pid = (row.get("ID") or "").strip()
                if not pid or pid.startswith("SIM-00"):
                    continue  # blank separators and the section banner
                self._rows[pid] = Param(
                    id=pid,
                    name=(row.get("Parameter") or "").strip(),
                    raw=(row.get("Value (with Margin)") or "").strip(),
                    unit=(row.get("Unit") or "").strip(),
                    status=(row.get("Status") or "").strip(),
                    source=(row.get("Source / requirement") or "").strip(),
                    notes=(row.get("Notes") or "").strip(),
                )
        if not self._rows:
            raise ParamError(f"no parameters parsed from {path}")

    # -- access ----------------------------------------------------------
    def __getitem__(self, pid: str) -> Param:
        try:
            p = self._rows[pid]
        except KeyError:
            near = get_close_matches(pid, self._rows, n=3)
            hint = f" Did you mean: {', '.join(near)}?" if near else ""
            raise ParamError(f"unknown parameter {pid!r} in {self.path.name}.{hint}") from None
        self._used.add(pid)
        return p

    def __contains__(self, pid: object) -> bool:
        return pid in self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def value(self, pid: str) -> float:
        return self[pid].as_float()

    def range(self, pid: str) -> tuple[float, float]:
        return self[pid].as_range()

    def text(self, pid: str) -> str:
        return self[pid].raw

    def select(self, prefix: str) -> list[Param]:
        return [p for pid, p in sorted(self._rows.items()) if pid.startswith(prefix)]

    # -- provenance ------------------------------------------------------
    def provenance(self, prefix: str = "") -> dict[str, int]:
        counts: dict[str, int] = {}
        for pid, p in self._rows.items():
            if pid.startswith(prefix):
                counts[p.status or "(blank)"] = counts.get(p.status or "(blank)", 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def unreviewed(self, prefix: str = "") -> list[Param]:
        return [p for p in self.select(prefix) if p.unreviewed]

    def audit(self) -> str:
        """Provenance of the parameters this run actually read."""
        used = [self._rows[p] for p in sorted(self._used)]
        if not used:
            return "no parameters read yet"
        bad = [p for p in used if p.unreviewed]
        lines = [f"{len(used)} parameters read, {len(bad)} not yet human-reviewed:"]
        lines += [f"    {p.id:9s} {p.status:17s} {p.name}" for p in bad]
        return "\n".join(lines)


PARAMS = Params()
