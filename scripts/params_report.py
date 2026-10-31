#!/usr/bin/env python3
"""Provenance report for design_docs/params.csv. Run before trusting any result."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phenei_gym.params import PARAMS  # noqa: E402

print(f"{len(PARAMS)} parameters in {PARAMS.path.name}\n")
for label, prefix in (("team SE parameters (P-xx)", "P-"), ("simulation parameters (SIM-xx)", "SIM-")):
    prov = PARAMS.provenance(prefix)
    print(f"{label}: {sum(prov.values())} rows")
    for status, n in prov.items():
        print(f"    {n:4d}  {status}")
    print()

bad = PARAMS.unreviewed("SIM-")
print(f"--- {len(bad)} simulation parameters awaiting human review ---")
for p in bad:
    print(f"  {p.id:9s} {p.status:17s} {p.name}  =  {p.raw} {p.unit}")
