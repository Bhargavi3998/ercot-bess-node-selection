import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bess_backtest as bb  # noqa: E402
from bess_dispatch import Battery  # noqa: E402
from synthetic_prices import make_synthetic  # noqa: E402


def test_forecast_never_uses_the_day_it_predicts_or_later():
    rng = np.random.default_rng(0)
    M = rng.normal(40, 10, (40, 96))
    base_med = bb.trailing_median_forecast(M, 14, 7)
    base_per = bb.persistence_forecast(M)
    k = 25
    M2 = M.copy()
    M2[k:] += 1000.0                       # change day k and everything after it
    new_med = bb.trailing_median_forecast(M2, 14, 7)
    new_per = bb.persistence_forecast(M2)
    assert np.array_equal(base_med[:k], new_med[:k], equal_nan=True)   # earlier forecasts unchanged
    assert np.array_equal(base_per[:k + 0], new_per[:k + 0], equal_nan=True)
    assert not np.array_equal(base_med[k + 1], new_med[k + 1], equal_nan=True)  # but the future does react


def test_policies_never_beat_perfect_foresight_and_capture_is_sane():
    df = make_synthetic(n_nodes=3, end="2025-02-28", seed=3)
    daily, _ = bb.run_backtest(df, Battery(), lookback=14, jobs=1)
    assert len(daily) > 50
    assert (daily["rev_median"] <= daily["rev_pf"] + 1e-6).all()
    assert (daily["rev_persist"] <= daily["rev_pf"] + 1e-6).all()
    assert (daily["rev_pf"] >= -1e-6).all()          # idling is always available, so PF >= 0
    cap = daily["rev_median"].sum() / daily["rev_pf"].sum()
    assert 0.3 < cap < 1.0


def test_days_that_are_not_96_clean_intervals_are_dropped():
    df = make_synthetic(n_nodes=2, end="2025-01-20", seed=1)
    node = df["settlement_point"].unique()[1]
    bad_day = pd.Timestamp("2025-01-10")
    drop = (df["settlement_point"] == node) & (df["ts"].dt.normalize() == bad_day) & (df["ts"].dt.hour == 5)
    df = df[~drop]                                    # remove an hour -> 92 intervals that day
    days, mats, dropped = bb.to_day_matrices(df)
    i = list(days).index(bad_day)
    assert np.isnan(mats[node][i]).all()
    assert dropped[node] == 1
    assert not np.isnan(mats[node][i - 1]).any()


def test_unit_conversion_to_kw_year():
    # $1,000/day on a 1 MW battery = $365,000/yr = $365/kW-yr
    assert bb.to_kw_year(1000.0, Battery(power_mw=1.0)) == pytest.approx(365.0)
    # the same $1,000/day on 10 MW is 10x less per kW
    assert bb.to_kw_year(1000.0, Battery(power_mw=10.0)) == pytest.approx(36.5)


def test_portfolio_selection_avoids_redundant_nodes():
    score = pd.Series({"A": 100.0, "B": 99.0, "C": 90.0, "D": 89.0})
    corr = pd.DataFrame(0.0, index=score.index, columns=score.index)
    corr.loc["A", "B"] = corr.loc["B", "A"] = 1.0     # A and B move together
    no_penalty = bb.select_portfolio(score, corr, k=2, lam=0.0, top_m=4)
    penalized = bb.select_portfolio(score, corr, k=2, lam=0.5, top_m=4)
    assert set(no_penalty) == {"A", "B"}
    assert not {"A", "B"} <= set(penalized)


def test_walk_forward_runs_and_reports_consistently():
    df = make_synthetic(n_nodes=8, end="2025-06-30", seed=5)
    daily, _ = bb.run_backtest(df, Battery(), lookback=14, jobs=1)
    wf = bb.walk_forward(daily, Battery(), "2025-04-01", k=3, alpha=0.5, lam=0.25, top_m=8)
    assert len(wf["chosen"]) == 3 and len(set(wf["chosen"])) == 3
    t = wf["test_kw_yr"]
    assert t["oracle_top_k"] >= t["chosen"] - 1e-9     # hindsight can't be beaten
    assert t["oracle_top_k"] >= t["all_nodes_average"] - 1e-9
    assert -1.0 <= wf["rank_corr_train_vs_test"]["policy"] <= 1.0


def test_restrict_dates_includes_the_whole_last_day():
    df = make_synthetic(n_nodes=2, end="2025-01-10", seed=1)
    out = bb.restrict_dates(df, "2025-01-03", "2025-01-05")
    assert out["ts"].min() == pd.Timestamp("2025-01-03 00:00")
    assert out["ts"].max() == pd.Timestamp("2025-01-05 23:45")


def test_identical_price_series_are_counted_once():
    df = make_synthetic(n_nodes=3, end="2025-01-05", seed=2)
    nodes = sorted(df["settlement_point"].unique())
    twin = df[df["settlement_point"] == nodes[1]].copy()
    twin["settlement_point"] = "TWIN_OF_ONE"
    out, same = bb.drop_duplicate_series(pd.concat([df, twin], ignore_index=True))
    assert same == [("TWIN_OF_ONE", nodes[1])]
    assert "TWIN_OF_ONE" not in set(out["settlement_point"])
    assert out["settlement_point"].nunique() == 3          # the genuine nodes are all kept


def test_benchmarks_are_excluded_from_the_site_choice():
    daily = pd.DataFrame({"node": ["HB_NORTH", "LZ_WEST", "SITE_A", "SITE_B"], "x": [1, 2, 3, 4]})
    assert bb.site_nodes(daily, ("HB_", "LZ_"))["node"].tolist() == ["SITE_A", "SITE_B"]
    assert len(bb.site_nodes(daily, ())) == 4              # no prefixes -> keep everything


def test_rolling_rounds_do_not_overlap_and_training_grows():
    df = make_synthetic(n_nodes=6, end="2025-06-30", seed=4)
    daily, _ = bb.run_backtest(df, Battery(), lookback=14, jobs=1)
    out = bb.rolling_walk_forward(daily, Battery(), ["2025-03-01", "2025-05-01"], k=3, alpha=0.5, lam=0.25, top_m=6)
    f1, f2 = out["folds"]
    assert (f1["split"], f2["split"]) == ("2025-03-01", "2025-05-01")
    assert f1["train_days"] < f2["train_days"]
    assert f1["test_days"] + f2["test_days"] <= daily["date"].nunique()
    assert out["rounds"] == 2 and 0 <= out["rounds_beating_average"] <= 2
