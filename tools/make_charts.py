#!/usr/bin/env python3
"""
Draw the two README charts from the saved backtest results.

  python tools/make_charts.py            # reads results/*.csv, writes docs/*.png

Colors are the first three slots of a validated categorical palette (blue, orange, aqua), used
for the same entity in both charts: blue = realistic trader, orange = perfect foresight,
aqua = market benchmark. Orange and aqua are lighter than 3:1 on white, so every series is also
directly labeled and told apart by marker shape.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
INK, INK2, GRID, AXIS, SURFACE = "#191919", "#666666", "#e7e7e7", "#c2c2c2", "#ffffff"

# Fleet-average ERCOT battery revenue, $/kW-month, from Modo Energy's public benchmark reports.
# Some months are early estimates; it covers real batteries (mostly shorter duration) earning
# from several services, so it is a sanity check on scale and seasonality, not a like-for-like.
#   2025-12 and 2026-02: https://modoenergy.com/research/en/february-2026-ercot-bess-benchmark-rtcb-revenues-batteries-performance
#   2026-01:             https://modoenergy.com/research/ercot-march-2026-bess-benchmark-batteries-fern-revenues
#   2026-04:             https://modoenergy.com/research/en/ercot-battery-storage-2026-things-to-watch
MARKET = {"2025-12": 1.70, "2026-01": 3.94, "2026-02": 1.08, "2026-04": 3.12}

plt.rcParams.update({"font.family": "sans-serif", "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def site_value_chart(summary: pd.DataFrame, path: str) -> None:
    s = summary[summary["type"] == "site"].sort_values("median_kw_yr", ascending=True)
    fig, ax = plt.subplots(figsize=(8.6, 9.2))
    y = range(len(s))
    ax.hlines(list(y), s["median_kw_yr"], s["pf_kw_yr"], color=AXIS, linewidth=1.5, zorder=1)
    ax.scatter(s["pf_kw_yr"], list(y), s=70, marker="D", color=ORANGE, edgecolor=SURFACE, linewidth=1.5,
               zorder=3, label="Perfect foresight (ceiling)")
    ax.scatter(s["median_kw_yr"], list(y), s=70, marker="o", color=BLUE, edgecolor=SURFACE, linewidth=1.5,
               zorder=3, label="Realistic trader (plans from last 14 days)")
    ax.set_yticks(list(y))
    ax.set_yticklabels(s.index, fontsize=8.5)
    ax.set_xlabel("$ per kW of battery power, per year", fontsize=9.5)
    ax.set_xlim(0, 118)                      # room on the right for the one annotation
    ax.set_xticks(range(0, 91, 10))
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if "BATCAVE_RN" in s.index:
        i = list(s.index).index("BATCAVE_RN")
        cap = s.loc["BATCAVE_RN", "capture_median"]
        ax.annotate(f"highest ceiling,\nlowest capture ({cap:.0%})",
                    (s.loc["BATCAVE_RN", "pf_kw_yr"], i), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2, linespacing=1.3)
    fig.text(0.02, 0.975, "What a 3-hour battery would earn at each ERCOT site", fontsize=13.5,
             fontweight="bold", va="top")
    fig.text(0.02, 0.948, "Simulated, energy-only, real-time prices, Dec 19 2025 to Oct 31 2026. "
             "1 MW battery, 85% round-trip efficiency, $3/MWh wear.", fontsize=9, color=INK2, va="top")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.17, 1.005), ncol=2, frameon=False, fontsize=9,
              labelcolor=INK2, columnspacing=2.0)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.895, bottom=0.07)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def monthly_chart(daily: pd.DataFrame, summary: pd.DataFrame, path: str) -> None:
    sites = summary[summary["type"] == "site"].index
    d = daily[daily["node"].isin(sites)].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["m"] = d["date"].dt.to_period("M")
    g = d.groupby("m")[["rev_pf", "rev_median"]].mean()          # mean $/MW-day per node-day
    g["days"] = [p.days_in_month for p in g.index]
    for c in ("rev_pf", "rev_median"):
        g[c] = g[c] / 1000.0 * g["days"]                          # -> $/kW-month
    x = list(range(len(g)))
    labels = [p.strftime("%b") for p in g.index]
    first_days = d[d["m"] == g.index[0]]["date"].nunique()
    if first_days < g["days"].iloc[0]:           # a partial first month: say so on the axis
        labels[0] = f"{labels[0]}\n({first_days} days)"
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(x, g["rev_pf"], color=ORANGE, linewidth=2, marker="D", markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label="Perfect foresight (ceiling)", zorder=3)
    ax.plot(x, g["rev_median"], color=BLUE, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label="Realistic trader", zorder=3)
    mx = [list(g.index.astype(str)).index(k) for k in MARKET if k in set(g.index.astype(str))]
    mv = [MARKET[k] for k in MARKET if k in set(g.index.astype(str))]
    dodge = 0.2                                # nudge the market squares right so they don't hide the dots
    ax.scatter([m + dodge for m in mx], mv, s=90, marker="s", color=AQUA, edgecolor=SURFACE, linewidth=1.5,
               zorder=4, label="Market: fleet-average battery revenue (Modo Energy)")
    for xi, vi in zip(mx, mv):
        ax.annotate(f"{vi:.2f}", (xi + dodge, vi), xytext=(9, -3), textcoords="offset points", ha="left",
                    fontsize=8.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("$ per kW of battery power, per month", fontsize=9.5)
    ax.set_ylim(0, max(g["rev_pf"]) * 1.15)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK2)
    fig.text(0.02, 0.975, "Revenue swings with the season; the realistic trader lands in the market's ballpark",
             fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.02, 0.93, "Average of 31 simulated sites per month. The market series covers real batteries and "
             "all revenue streams, so compare scale and shape only.", fontsize=8.5, color=INK2, va="top")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.09)
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    summary = pd.read_csv(os.path.join("results", "node_summary.csv"), index_col=0)
    daily = pd.read_csv(os.path.join("results", "daily.csv"))
    os.makedirs("docs", exist_ok=True)
    site_value_chart(summary, os.path.join("docs", "site_value.png"))
    monthly_chart(daily, summary, os.path.join("docs", "monthly_vs_market.png"))
    print("Wrote docs/site_value.png and docs/monthly_vs_market.png")
