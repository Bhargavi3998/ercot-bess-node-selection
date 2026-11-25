import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bess_dispatch import Battery, optimize_day, settle  # noqa: E402

DT = 0.25


def two_level(low, high, hours_each=4):
    n = int(hours_each / DT)
    return np.concatenate([np.full(n, float(low)), np.full(n, float(high))])


def test_flat_prices_do_nothing():
    r = optimize_day(np.full(96, 40.0), Battery(degradation_cost=0.0))
    assert r.revenue == pytest.approx(0.0, abs=1e-6)


def test_lossless_arbitrage_matches_hand_calculation():
    # Empty start, lossless: buy 3 MWh at $10, sell 3 MWh at $110 -> $300
    b = Battery(round_trip_eff=1.0, degradation_cost=0.0, soc_start_frac=0.0)
    r = optimize_day(two_level(10, 110), b)
    assert r.revenue == pytest.approx(300.0, abs=1e-4)


def test_efficiency_losses_match_hand_calculation():
    # eta = 0.9 each way. Store 3 MWh internally -> buy 3/0.9 MWh at $10, sell 3*0.9 MWh at $110
    b = Battery(round_trip_eff=0.81, degradation_cost=0.0, soc_start_frac=0.0)
    r = optimize_day(two_level(10, 110), b)
    expected = 2.7 * 110 - (3 / 0.9) * 10
    assert r.revenue == pytest.approx(expected, abs=1e-4)


def test_degradation_cost_blocks_thin_spreads():
    b = Battery(round_trip_eff=1.0, degradation_cost=15.0, soc_start_frac=0.0)
    assert optimize_day(two_level(50, 60), b).revenue == pytest.approx(0.0, abs=1e-6)  # $10 < $15 wear
    assert optimize_day(two_level(50, 150), b).revenue > 250                          # $100 spread trades


def test_power_limit_caps_charge_rate():
    b = Battery(power_mw=2.0, round_trip_eff=1.0, degradation_cost=0.0, soc_start_frac=0.0)
    r = optimize_day(two_level(10, 110), b)
    assert r.charge.max() <= 2.0 + 1e-9 and r.discharge.max() <= 2.0 + 1e-9
    assert r.revenue == pytest.approx(600.0, abs=1e-3)  # 6 MWh capacity (2 MW x 3 h) x $100


def test_schedule_is_physically_feasible():
    rng = np.random.default_rng(1)
    prices = 30 + 40 * rng.standard_normal(96)
    b = Battery()
    r = optimize_day(prices, b)
    assert r.soc.min() >= -1e-9 and r.soc.max() <= b.energy_mwh + 1e-9
    assert r.soc[-1] >= r.soc[0] - 1e-9                       # ends no emptier than it started
    soc_rebuilt = r.soc[0] + np.cumsum(b.eta * r.charge * DT - r.discharge * DT / b.eta)
    assert np.allclose(soc_rebuilt, r.soc[1:], atol=1e-7)    # energy balance holds
    assert settle(prices, r.charge, r.discharge, b) == pytest.approx(r.revenue)


def test_negative_prices_never_charge_and_discharge_together():
    # Deeply negative block with a full battery: a naive LP can "burn" energy to get paid.
    prices = np.concatenate([np.full(24, -80.0), np.full(24, 5.0), np.full(48, 60.0)])
    b = Battery(soc_start_frac=1.0, degradation_cost=0.0)
    r = optimize_day(prices, b)
    assert np.max(np.minimum(r.charge, r.discharge)) < 1e-6


def test_perfect_foresight_beats_any_other_feasible_schedule():
    rng = np.random.default_rng(7)
    b = Battery()
    for _ in range(20):
        actual = 30 + 35 * rng.standard_normal(96)
        forecast = actual + 25 * rng.standard_normal(96)
        best = optimize_day(actual, b).revenue
        other = optimize_day(forecast, b)
        assert settle(actual, other.charge, other.discharge, b) <= best + 1e-6


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        optimize_day(np.array([1.0, np.nan, 3.0]))
    with pytest.raises(ValueError):
        Battery(round_trip_eff=1.2)
