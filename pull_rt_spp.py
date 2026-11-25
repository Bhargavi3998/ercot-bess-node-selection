#!/usr/bin/env python3
"""
Pull a long history of ERCOT RT Settlement Point Prices (NP6-905), resumably.

Why this exists: the siting pipeline only had 3 days of data. A battery study needs
12-24 months so that monthly CVaR, spike features and a train/test split mean something.

Design:
  - One request per (settlement point, calendar month); each result is saved to
      data/rt_spp/<NODE>/<YYYY-MM>.csv.gz
    A re-run skips files that already exist, so you can stop and resume freely.
  - Paginates if the API reports more than one page (verify on your first run, see NOTE).
  - Sleeps between requests to stay under ERCOT's rate limit.
  - `--combine` merges everything into one tidy file for the next steps.

Run from the repo folder (needs .env with ERCOT credentials, same as the pipeline):

  python pull_rt_spp.py --start 2025-01-01 --end 2025-12-31 \
      --settlement-points-file settlement_points_40.txt --hub HB_NORTH
  python pull_rt_spp.py --combine

Start with a small test (one node, one month) before the full pull:

  python pull_rt_spp.py --start 2025-06-01 --end 2025-06-30 --nodes HB_NORTH
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import os
import sys
import time
from typing import Iterator, List, Optional, Tuple

import pandas as pd

# Reuse the existing, working client and parser from the siting pipeline.
from ercot_bess_siting_rt_pipeline import (
    RT_SPP_ENDPOINT,
    ErcotClient,
    client_from_env,
    load_settlement_points,
    response_to_df,
)

DATA_DIR = os.path.join("data", "rt_spp")
COMBINED_PATH = os.path.join("data", "rt_spp_all.csv.gz")


def month_windows(start: dt.date, end: dt.date) -> Iterator[Tuple[dt.date, dt.date]]:
    """Yield (first_day, last_day) for each calendar month overlapping [start, end]."""
    cur = dt.date(start.year, start.month, 1)
    while cur <= end:
        last = dt.date(cur.year, cur.month, calendar.monthrange(cur.year, cur.month)[1])
        yield max(cur, start), min(last, end)
        cur = last + dt.timedelta(days=1)


def month_path(node: str, first_day: dt.date) -> str:
    return os.path.join(DATA_DIR, node, f"{first_day:%Y-%m}.csv.gz")


def fetch_node_window(api: ErcotClient, node: str, d0: dt.date, d1: dt.date,
                      page_size: int, sleep_s: float) -> pd.DataFrame:
    """Fetch one node over one date window, following pages if the API reports them.

    NOTE: ERCOT's public API normally returns `_meta` with `totalPages` / `currentPage`
    and accepts `page` / `size` query params. If your first test run shows fewer rows
    than expected (about 96 per day per node), inspect one raw response and adjust here.
    """
    params = {
        "deliveryDateFrom": d0.isoformat(),
        "deliveryDateTo": d1.isoformat(),
        "settlementPoint": node,
        "size": page_size,
        "page": 1,
    }
    frames: List[pd.DataFrame] = []
    total_pages = 1
    while params["page"] <= total_pages:
        resp = api.get(RT_SPP_ENDPOINT, params=dict(params))
        try:
            frames.append(response_to_df(resp))
        except RuntimeError:
            # An empty page (no data for this node/window) cannot be parsed; treat as empty.
            break
        meta = resp.json().get("_meta", {}) if isinstance(resp.json(), dict) else {}
        total_pages = int(meta.get("totalPages", 1) or 1)
        params["page"] += 1
        time.sleep(sleep_s)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def pull(args: argparse.Namespace) -> None:
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    if end < start:
        sys.exit("--end must be on or after --start")

    nodes: List[str] = []
    if args.nodes:
        nodes += [n.strip() for n in args.nodes.split(",") if n.strip()]
    if args.settlement_points_file:
        nodes += load_settlement_points(args.settlement_points_file)
    if args.hub and args.hub not in nodes:
        nodes.append(args.hub)  # the siting basis features need the hub series
    nodes = list(dict.fromkeys(nodes))  # de-duplicate, keep order
    if not nodes:
        sys.exit("No nodes given. Use --nodes and/or --settlement-points-file.")

    windows = list(month_windows(start, end))
    jobs = [(n, w) for n in nodes for w in windows]
    todo = [(n, w) for n, w in jobs if not os.path.exists(month_path(n, w[0]))]
    print(f"{len(nodes)} nodes x {len(windows)} months = {len(jobs)} files; "
          f"{len(jobs) - len(todo)} already done, {len(todo)} to pull.")
    if not todo:
        return
    est_min = len(todo) * (args.sleep + 0.5) / 60
    print(f"Rough time at --sleep {args.sleep}s: ~{est_min:.0f} min (more if multiple pages).")

    api = client_from_env()
    failures: List[str] = []
    for i, (node, (d0, d1)) in enumerate(todo, 1):
        path = month_path(node, d0)
        try:
            df = fetch_node_window(api, node, d0, d1, args.page_size, args.sleep)
        except Exception as exc:  # keep going; failed items are retried on the next run
            failures.append(f"{node} {d0:%Y-%m}: {exc}")
            print(f"[{i}/{len(todo)}] FAILED {node} {d0:%Y-%m}: {exc}")
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Write even when empty so a genuinely empty month is not re-requested forever.
        df.to_csv(path, index=False, compression="gzip")
        print(f"[{i}/{len(todo)}] {node} {d0:%Y-%m}: {len(df)} rows")

    if failures:
        print(f"\n{len(failures)} failed. Re-run the same command to retry only those:")
        for f in failures:
            print("  -", f)


def combine() -> None:
    """Merge every saved month into one file with clean column names."""
    paths = []
    for root, _dirs, files in os.walk(DATA_DIR):
        paths += [os.path.join(root, f) for f in files if f.endswith(".csv.gz")]
    if not paths:
        sys.exit(f"No files under {DATA_DIR}. Run a pull first.")

    frames = []
    for p in sorted(paths):
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:  # a month with no data was saved as an empty file
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        sys.exit("All saved files are empty.")
    raw = pd.concat(frames, ignore_index=True)

    ts = (pd.to_datetime(raw["deliveryDate"])
          + pd.to_timedelta(raw["deliveryHour"] - 1, unit="h")
          + pd.to_timedelta((raw["deliveryInterval"] - 1) * 15, unit="m"))
    out = pd.DataFrame({
        "ts": ts,
        "settlement_point": raw["settlementPoint"],
        "price": pd.to_numeric(raw["settlementPointPrice"], errors="coerce"),
    })
    # DST fall-back creates duplicate local timestamps; the DSTFlag column distinguishes them.
    if "DSTFlag" in raw.columns:
        out["dst_flag"] = raw["DSTFlag"].astype(str).str.lower().isin(["true", "1", "y"])
    out = (out.dropna(subset=["price"])
              .drop_duplicates(subset=["ts", "settlement_point"] + (["dst_flag"] if "dst_flag" in out else []))
              .sort_values(["settlement_point", "ts"]))
    os.makedirs("data", exist_ok=True)
    out.to_csv(COMBINED_PATH, index=False, compression="gzip")

    per_node = out.groupby("settlement_point")["ts"].agg(["min", "max", "count"])
    print(f"Wrote {COMBINED_PATH}: {len(out):,} rows, {out['settlement_point'].nunique()} nodes")
    print(f"Date range: {out['ts'].min()} -> {out['ts'].max()}")
    sparse = per_node[per_node["count"] < per_node["count"].median() * 0.9]
    if not sparse.empty:
        print(f"Warning: {len(sparse)} node(s) have <90% of the median row count "
              f"(possible gaps): {', '.join(sparse.index[:10])}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--end", help="YYYY-MM-DD")
    ap.add_argument("--nodes", help="comma-separated settlement points")
    ap.add_argument("--settlement-points-file", default=None)
    ap.add_argument("--hub", default="HB_NORTH", help="always pulled so basis features work")
    ap.add_argument("--page-size", type=int, default=10000)
    ap.add_argument("--sleep", type=float, default=2.1, help="seconds between requests")
    ap.add_argument("--combine", action="store_true", help="merge saved files into one CSV and exit")
    args = ap.parse_args()

    if args.combine:
        combine()
        return
    if not (args.start and args.end):
        ap.error("--start and --end are required (unless using --combine)")
    pull(args)


if __name__ == "__main__":
    main()
