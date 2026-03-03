#!/usr/bin/env python3
"""
Generate settlement_points.txt from ERCOT RT SPP (NP6-905).

It:
1) Pulls RT SPP for a short date range
2) Extracts settlementPoint values from the raw response
3) Writes settlement_points.txt with:
   - required hubs + load zones first (if present)
   - then fills up to N points using ONLY Resource Nodes (settlementPointType == "RN")
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import random

from ercot_bess_siting_rt_pipeline import client_from_env, fetch_rt_spp

DEFAULT_CORE_POINTS = [
    # Hubs
    "HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HOUSTON",
    # Load zones
    "LZ_NORTH", "LZ_SOUTH", "LZ_WEST", "LZ_HOUSTON",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--n", type=int, default=100, help="How many settlement points to output")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="settlement_points.txt")
    args = ap.parse_args()

    random.seed(args.seed)

    api = client_from_env()
    raw = fetch_rt_spp(api, args.start, args.end, settlement_points=None)

    if raw.empty:
        raise RuntimeError("No rows returned. Try a different date range (and ensure credentials work).")

    # Must have settlementPoint
    if "settlementPoint" not in raw.columns:
        raise RuntimeError(f"Expected column 'settlementPoint' not found. Columns={list(raw.columns)}")

    all_points = sorted(set(raw["settlementPoint"].dropna().astype(str)))

    core = [p for p in DEFAULT_CORE_POINTS if p in all_points]

    if "settlementPointType" in raw.columns:
        rn_points = sorted(set(raw.loc[raw["settlementPointType"] == "RN", "settlementPoint"].dropna().astype(str)))
    else:
        rn_points = []
        print("Warning: settlementPointType column not found; falling back to sampling from all settlement points.")

    if rn_points:
        rest = [p for p in rn_points if p not in set(core)]
    else:
        rest = [p for p in all_points if p not in set(core)]

    needed = max(0, args.n - len(core))
    needed = min(needed, len(rest))

    sample = random.sample(rest, needed)
    final = core + sample

    with open(args.out, "w", encoding="utf-8") as f:
        for p in final:
            f.write(p + "\n")

    print(f"Wrote {len(final)} settlement points to {args.out}")
    print(f"Included core points: {core}")
    print(f"Total available settlementPoint values in pull: {len(all_points)}")
    if rn_points:
        print(f"Total RN settlement points available in pull: {len(rn_points)}")
        print(f"RN points written (excluding core): {len(sample)}")

if __name__ == "__main__":
    main()