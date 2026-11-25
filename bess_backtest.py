#!/usr/bin/env python3
"""
Backtest a 3-hour battery on ERCOT RT prices and rank nodes by what they would really earn.

For every node and every day it computes:
  rev_pf       perfect foresight (the ceiling: optimize on the day's actual prices)
  rev_median   a realistic policy: optimize on the trailing-median price shape of the previous
               `lookback` days, then settle that schedule at the actual prices
  rev_persist  a naive policy: optimize on yesterday's prices, settle at actuals

Policies only ever see data from BEFORE the day they trade (no lookahead; tested).
"revenue capture" = what the realistic policy earned / what perfect foresight earned.

Then a walk-forward test: rank nodes on days before --split, pick a diversified top-k using only
that period, and check how the picks did on days from --split onward, against the average node
and against the best-in-hindsight picks.

Usage:
  python bess_backtest.py --prices data/rt_spp_all.csv.gz --split 2025-07-01
  python bess_backtest.py --synthetic            # code check on fake data, not a result
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bess_dispatch import Battery, optimize_day, settle

SLOTS = 96
DT_H = 0.25


# ----------------------------------------------------------------------------- data

def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    missing = {"ts", "settlement_point", "price"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return df[["ts", "settlement_point", "price"]].dropna()


def restrict_dates(df: pd.DataFrame, start: str = None, end: str = None) -> pd.DataFrame:
    """Keep rows from `start` through the END of day `end` (both inclusive, YYYY-MM-DD)."""
    if start:
        df = df[df["ts"] >= pd.Timestamp(start)]
    if end:
        df = df[df["ts"] < pd.Timestamp(end) + pd.Timedelta(days=1)]
    return df


def to_day_matrices(df: pd.DataFrame) -> Tuple[pd.DatetimeIndex, Dict[str, np.ndarray], Dict[str, int]]:
    """Per node: array [n_days, 96] over the full calendar range. Days that are not exactly 96
    clean 15-minute intervals (DST days, gaps, duplicates) become NaN rows and are skipped."""
    days = pd.date_range(df["ts"].min().normalize(), df["ts"].max().normalize(), freq="D")
    index = {d: i for i, d in enumerate(days)}
    mats: Dict[str, np.ndarray] = {}
    dropped: Dict[str, int] = {}
    for node, g in df.groupby("settlement_point"):
        g = g.copy()
        g["day"] = g["ts"].dt.normalize()
        g["slot"] = ((g["ts"] - g["day"]) / pd.Timedelta(minutes=15)).round().astype(int)
        counts = g.groupby("day")["slot"].agg(["size", "nunique"])
        good = counts[(counts["size"] == SLOTS) & (counts["nunique"] == SLOTS)].index
        dropped[node] = int(len(counts) - len(good))
        g = g[g["day"].isin(good)]
        M = np.full((len(days), SLOTS), np.nan)
        M[g["day"].map(index).values, g["slot"].values] = g["price"].values
        mats[node] = M
    return days, mats, dropped


# ------------------------------------------------------------------------- forecasts

def trailing_median_forecast(M: np.ndarray, lookback: int, min_days: int) -> np.ndarray:
    """Forecast for row i = median over rows [i-lookback, i). Never touches row i or later."""
    F = np.full_like(M, np.nan)
    for i in range(lookback, M.shape[0]):
        w = M[i - lookback:i]
        ok = ~np.isnan(w).any(axis=1)
        if ok.sum() >= min_days:
            F[i] = np.median(w[ok], axis=0)
    return F


def persistence_forecast(M: np.ndarray) -> np.ndarray:
    F = np.full_like(M, np.nan)
    F[1:] = M[:-1]
    return F


# --------------------------------------------------------------------------- backtest

def evaluate_node(args) -> pd.DataFrame:
    node, M, days, battery, lookback = args
    Fm = trailing_median_forecast(M, lookback, max(3, lookback // 2))
    Fp = persistence_forecast(M)
    rows = []
    for i in range(M.shape[0]):
        a = M[i]
        if np.isnan(a).any() or np.isnan(Fm[i]).any() or np.isnan(Fp[i]).any():
            continue
        pf = optimize_day(a, battery, DT_H)
        sm = optimize_day(Fm[i], battery, DT_H)
        sp = optimize_day(Fp[i], battery, DT_H)
        rows.append((node, days[i], pf.revenue,
                     settle(a, sm.charge, sm.discharge, battery, DT_H),
                     settle(a, sp.charge, sp.discharge, battery, DT_H)))
    return pd.DataFrame(rows, columns=["node", "date", "rev_pf", "rev_median", "rev_persist"])


def run_backtest(df: pd.DataFrame, battery: Battery, lookback: int, jobs: int):
    days, mats, dropped = to_day_matrices(df)
    work = [(n, M, days, battery, lookback) for n, M in mats.items()]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            parts = list(ex.map(evaluate_node, work))
    else:
        parts = [evaluate_node(w) for w in work]
    return pd.concat(parts, ignore_index=True), dropped


# --------------------------------------------------------------------------- summaries

def to_kw_year(daily_mean_usd: float, battery: Battery) -> float:
    """Mean $/day for the battery -> $/kW-year of power capacity."""
    return daily_mean_usd * 365.0 / (battery.power_mw * 1000.0)


def window_summary(daily: pd.DataFrame, battery: Battery) -> pd.DataFrame:
    g = daily.groupby("node")
    out = pd.DataFrame({
        "days": g.size(),
        "pf_kw_yr": g["rev_pf"].mean().map(lambda v: to_kw_year(v, battery)),
        "median_kw_yr": g["rev_median"].mean().map(lambda v: to_kw_year(v, battery)),
        "persist_kw_yr": g["rev_persist"].mean().map(lambda v: to_kw_year(v, battery)),
    })
    out["capture_median"] = g["rev_median"].sum() / g["rev_pf"].sum()
    out["capture_persist"] = g["rev_persist"].sum() / g["rev_pf"].sum()
    return out


def downside(series: pd.Series, tail: float = 0.05) -> float:
    """Expected shortfall of daily revenue: mean minus the mean of the worst `tail` of days."""
    k = max(1, int(np.ceil(tail * len(series))))
    return float(series.mean() - np.sort(series.values)[:k].mean())


def select_portfolio(score: pd.Series, corr: pd.DataFrame, k: int, lam: float, top_m: int) -> List[str]:
    """Best k nodes maximizing  sum(score) - lam * mean(score) * sum(pairwise correlations)."""
    cand = score.sort_values(ascending=False).head(top_m).index.tolist()
    if len(cand) < k:
        raise ValueError(f"need at least {k} candidate nodes, have {len(cand)}")
    C = corr.reindex(index=cand, columns=cand).fillna(0.0).values
    s = score.loc[cand].values
    scale = lam * float(np.mean(s))
    best, best_val = None, -np.inf
    for combo in itertools.combinations(range(len(cand)), k):
        pen = sum(C[i, j] for i, j in itertools.combinations(combo, 2))
        val = float(s[list(combo)].sum() - scale * pen)
        if val > best_val:
            best, best_val = combo, val
    return [cand[i] for i in best]


def walk_forward(daily: pd.DataFrame, battery: Battery, split: str, k: int, alpha: float,
                 lam: float, top_m: int, end: str = None) -> dict:
    """Pick nodes on days before `split`, judge them on days from `split` up to (not incl.) `end`."""
    split_ts = pd.Timestamp(split)
    train = daily[daily["date"] < split_ts]
    test = daily[daily["date"] >= split_ts]
    if end:
        test = test[test["date"] < pd.Timestamp(end)]
    if train.empty or test.empty:
        raise ValueError(f"--split {split} leaves an empty train or test window")
    tr, te = window_summary(train, battery), window_summary(test, battery)
    common = tr.index.intersection(te.index)
    tr, te = tr.loc[common], te.loc[common]

    risk = train.groupby("node")["rev_median"].apply(downside).map(lambda v: to_kw_year(v, battery))
    score = (tr["median_kw_yr"] - alpha * risk.loc[common]).rename("score")
    corr = train.pivot(index="date", columns="node", values="rev_median")[common].corr()

    chosen = select_portfolio(score, corr, k, lam, top_m)
    naive = score.sort_values(ascending=False).head(k).index.tolist()
    oracle = te["median_kw_yr"].sort_values(ascending=False).head(k).index.tolist()

    def test_mean(nodes): return float(te.loc[nodes, "median_kw_yr"].mean())

    rho_pol = spearmanr(tr["median_kw_yr"], te["median_kw_yr"])
    rho_pf = spearmanr(tr["pf_kw_yr"], te["pf_kw_yr"])
    port = test.pivot(index="date", columns="node", values="rev_median")[chosen].mean(axis=1)
    return {
        "split": split, "k": k, "alpha": alpha, "lambda": lam, "nodes": int(len(common)),
        "train_days": int(train["date"].nunique()), "test_days": int(test["date"].nunique()),
        "chosen": chosen, "naive_top_k": naive, "oracle_top_k": oracle,
        "test_kw_yr": {
            "chosen": test_mean(chosen), "naive_top_k": test_mean(naive),
            "all_nodes_average": float(te["median_kw_yr"].mean()), "oracle_top_k": test_mean(oracle),
        },
        "rank_corr_train_vs_test": {"policy": float(rho_pol.statistic), "perfect_foresight": float(rho_pf.statistic)},
        "test_capture_median_avg": float(te["capture_median"].mean()),
        "chosen_portfolio_test_downside_kw_yr": to_kw_year(downside(port), battery),
        "chosen_test_detail": te.loc[chosen, ["pf_kw_yr", "median_kw_yr", "capture_median"]].round(3).to_dict("index"),
    }


def rolling_walk_forward(daily: pd.DataFrame, battery: Battery, splits: List[str], k: int,
                         alpha: float, lam: float, top_m: int) -> dict:
    """Several train/test rounds. Round i trains on everything before splits[i] and is tested
    on splits[i] up to splits[i+1] (the last round runs to the end of the data). One split is
    fragile when prices change with the season; several rounds show how stable the result is."""
    bounds = list(splits) + [None]
    folds = [walk_forward(daily, battery, s, k, alpha, lam, top_m, end=bounds[i + 1])
             for i, s in enumerate(splits)]
    gain = [f["test_kw_yr"]["chosen"] - f["test_kw_yr"]["all_nodes_average"] for f in folds]
    return {
        "folds": folds,
        "avg_gain_vs_average_kw_yr": float(np.mean(gain)),
        "rounds_beating_average": int(sum(g > 0 for g in gain)),
        "rounds": len(folds),
        "avg_rank_corr_policy": float(np.mean([f["rank_corr_train_vs_test"]["policy"] for f in folds])),
    }


def drop_duplicate_series(df: pd.DataFrame):
    """Nodes sharing one pricing point have identical price series; keep the first of each group.
    Returns (filtered df, list of (dropped, kept) pairs)."""
    P = (df.drop_duplicates(["ts", "settlement_point"])
           .pivot(index="ts", columns="settlement_point", values="price"))
    h = pd.util.hash_pandas_object(P.T, index=False)
    h.index = P.columns
    dropped = []
    for _, names in h.groupby(h).groups.items():
        names = sorted(names)
        if len(names) > 1:
            dropped += [(n, names[0]) for n in names[1:]]
    return df[~df["settlement_point"].isin([d for d, _ in dropped])], dropped


def site_nodes(daily: pd.DataFrame, exclude_prefixes: Tuple[str, ...]) -> pd.DataFrame:
    """Hubs (HB_) and load zones (LZ_) are price benchmarks, not places to put a battery."""
    if not exclude_prefixes:
        return daily
    return daily[~daily["node"].str.startswith(exclude_prefixes)]


# -------------------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", default=os.path.join("data", "rt_spp_all.csv.gz"))
    ap.add_argument("--synthetic", action="store_true", help="use fake prices (code check only)")
    ap.add_argument("--from", dest="date_from", default=None, help="first day to use, YYYY-MM-DD")
    ap.add_argument("--to", dest="date_to", default=None, help="last day to use, YYYY-MM-DD")
    ap.add_argument("--split", default=None, help="YYYY-MM-DD; default = midpoint of the data")
    ap.add_argument("--splits", default=None,
                    help="comma-separated YYYY-MM-DD; several rolling train/test rounds (overrides --split)")
    ap.add_argument("--exclude-prefix", default="HB_,LZ_",
                    help="node-name prefixes treated as benchmarks, not sites ('' to keep all)")
    ap.add_argument("--lookback", type=int, default=14)
    ap.add_argument("--power-mw", type=float, default=1.0)
    ap.add_argument("--duration-h", type=float, default=3.0)
    ap.add_argument("--rte", type=float, default=0.85, help="round-trip efficiency")
    ap.add_argument("--deg", type=float, default=3.0, help="degradation cost, $/MWh discharged")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--top-m", type=int, default=25)
    ap.add_argument("--alpha", type=float, default=0.5, help="downside penalty weight")
    ap.add_argument("--lam", type=float, default=0.25, help="correlation penalty, relative to mean score")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join("outputs", "bess"))
    args = ap.parse_args()

    if args.synthetic:
        from synthetic_prices import make_synthetic
        df = make_synthetic()
        print("*** SYNTHETIC DATA: this checks the code; it is NOT a result about ERCOT. ***\n")
    else:
        if not os.path.exists(args.prices):
            raise SystemExit(f"{args.prices} not found. Run pull_rt_spp.py (then --combine) first, "
                             f"or try --synthetic.")
        df = load_prices(args.prices)

    df = restrict_dates(df, args.date_from, args.date_to)
    if df.empty:
        raise SystemExit("No price rows left after --from/--to.")
    df, same = drop_duplicate_series(df)
    for gone, kept in same:
        print(f"Note: {gone} has exactly the same prices as {kept} (same pricing point); counting it once.")
    print(f"Using {len(df):,} price rows, {df['settlement_point'].nunique()} nodes, "
          f"{df['ts'].min():%Y-%m-%d} to {df['ts'].max():%Y-%m-%d}.")
    battery = Battery(args.power_mw, args.duration_h, args.rte, args.deg)
    daily, dropped = run_backtest(df, battery, args.lookback, args.jobs)
    if daily.empty:
        raise SystemExit("No tradable days: need at least lookback+1 consecutive clean days per node.")
    n_drop = sum(dropped.values())
    if n_drop:
        print(f"Skipped {n_drop} node-days that were not exactly 96 clean intervals (DST days, gaps).")

    prefixes = tuple(p.strip() for p in args.exclude_prefix.split(",") if p.strip())
    sites = site_nodes(daily, prefixes)
    if sites["node"].nunique() < args.top_k:
        raise SystemExit("Fewer site nodes than --top-k after excluding benchmarks.")

    dates = np.sort(daily["date"].unique())
    if args.splits:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    else:
        splits = [args.split or str(pd.Timestamp(dates[len(dates) // 2]).date())]
    os.makedirs(args.out, exist_ok=True)
    daily.to_csv(os.path.join(args.out, "daily.csv"), index=False)
    full = window_summary(daily, battery)
    full.insert(0, "type", ["benchmark" if n.startswith(prefixes) and prefixes else "site" for n in full.index])
    full.round(4).to_csv(os.path.join(args.out, "node_summary.csv"))
    if len(splits) == 1:
        folds = [walk_forward(sites, battery, splits[0], args.top_k, args.alpha, args.lam, args.top_m)]
        result = {"folds": folds, "rounds": 1}
    else:
        result = rolling_walk_forward(sites, battery, splits, args.top_k, args.alpha, args.lam, args.top_m)
        folds = result["folds"]
    with open(os.path.join(args.out, "walk_forward.json"), "w") as f:
        json.dump(result, f, indent=2)

    pd.set_option("display.width", 130)
    print(f"Battery: {battery.power_mw:g} MW x {battery.duration_h:g} h, RTE {battery.round_trip_eff:.0%}, "
          f"wear ${battery.degradation_cost:g}/MWh. Energy-only, RT prices.\n")
    print("Per-node results over the whole period ($/kW-year); 'benchmark' rows are hubs/load zones, "
          "not sites:")
    print(full.sort_values("median_kw_yr", ascending=False).round(2).to_string())
    for f_ in folds:
        t = f_["test_kw_yr"]
        print(f"\nWalk-forward: pick on days < {f_['split']}, judge on the following "
              f"{f_['test_days']} days ({f_['train_days']} train days, {f_['nodes']} sites):")
        print(f"  chosen (diversified) : {f_['chosen']}  -> {t['chosen']:.1f} $/kW-yr")
        if f_["naive_top_k"] != f_["chosen"]:
            print(f"  naive top-{args.top_k}          : {f_['naive_top_k']}  -> {t['naive_top_k']:.1f}")
        print(f"  average site         : {t['all_nodes_average']:.1f}")
        print(f"  best in hindsight    : {f_['oracle_top_k']}  -> {t['oracle_top_k']:.1f}  (not achievable)")
        r = f_["rank_corr_train_vs_test"]
        print(f"  rank correlation train->test: policy {r['policy']:.2f}, perfect-foresight {r['perfect_foresight']:.2f}")
    if len(folds) > 1:
        print(f"\nAcross {result['rounds']} rounds: chosen beat the average site in "
              f"{result['rounds_beating_average']}; average gain {result['avg_gain_vs_average_kw_yr']:+.1f} $/kW-yr; "
              f"average rank correlation {result['avg_rank_corr_policy']:.2f}.")
    print(f"\nOutputs written to {args.out}/")


if __name__ == "__main__":
    main()
