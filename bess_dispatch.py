"""
Battery dispatch engine: optimal charge/discharge schedule for one day of prices.

Model (per 1 MW of battery power, default 3-hour duration):
  variables  c_t  grid-side charging power  [0, P] MW
             d_t  grid-side discharge power [0, P] MW
             s_t  state of charge           [0, E] MWh, t = 0..T
  dynamics   s_{t+1} = s_t + eta * c_t * dt  -  d_t * dt / eta
             (eta = sqrt(round-trip efficiency): losses are split between charge and discharge)
  start/end  s_0 = soc_start_frac * E, and s_T >= s_0 (the battery may not borrow from tomorrow)
  objective  maximize  sum_t price_t * (d_t - c_t) * dt   -   degradation_cost * sum_t d_t * dt

Simultaneous charge and discharge is physically impossible. The plain LP can exploit it when
prices are negative (it "burns" energy to get paid for consuming it), so if the LP solution
overlaps, the day is re-solved as a MILP with a charge/discharge binary. That is rare, so the
common case stays fast.

Revenue here is energy arbitrage only. It excludes ancillary services, DA/RT settlement
differences, fees and the cost of imbalance, so treat results as an energy-only floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

OVERLAP_TOL = 1e-6  # MW of simultaneous charge+discharge that triggers the MILP re-solve


@dataclass(frozen=True)
class Battery:
    power_mw: float = 1.0
    duration_h: float = 3.0
    round_trip_eff: float = 0.85
    degradation_cost: float = 3.0  # $ per MWh discharged (wear cost; set 0 to ignore)
    soc_start_frac: float = 0.5

    def __post_init__(self) -> None:
        if self.power_mw <= 0 or self.duration_h <= 0:
            raise ValueError("power_mw and duration_h must be positive")
        if not 0 < self.round_trip_eff <= 1:
            raise ValueError("round_trip_eff must be in (0, 1]")
        if self.degradation_cost < 0:
            raise ValueError("degradation_cost must be >= 0")
        if not 0 <= self.soc_start_frac <= 1:
            raise ValueError("soc_start_frac must be in [0, 1]")

    @property
    def energy_mwh(self) -> float:
        return self.power_mw * self.duration_h

    @property
    def eta(self) -> float:
        return float(np.sqrt(self.round_trip_eff))


@dataclass
class Dispatch:
    charge: np.ndarray     # MW, length T
    discharge: np.ndarray  # MW, length T
    soc: np.ndarray        # MWh, length T + 1
    revenue: float         # $ at the prices the schedule was optimized for
    used_milp: bool


def settle(prices: np.ndarray, charge: np.ndarray, discharge: np.ndarray,
           battery: Battery, dt_h: float = 0.25) -> float:
    """Dollar result of running a fixed schedule against (possibly different) actual prices."""
    prices = np.asarray(prices, float)
    energy_value = np.sum(prices * (discharge - charge)) * dt_h
    wear = battery.degradation_cost * np.sum(discharge) * dt_h
    return float(energy_value - wear)


def _build_lp(prices: np.ndarray, b: Battery, dt: float):
    T = len(prices)
    eta = b.eta
    ident = sparse.identity(T, format="csr")
    # s_{t+1} - s_t  column pattern over the T+1 SOC variables
    soc_diff = sparse.diags([-np.ones(T), np.ones(T)], [0, 1], shape=(T, T + 1), format="csr")
    # row t:  -eta*dt*c_t + (dt/eta)*d_t + (s_{t+1} - s_t) = 0
    a_eq = sparse.hstack([-eta * dt * ident, (dt / eta) * ident, soc_diff], format="csr")
    cost = np.concatenate([
        prices * dt,                                   # charging pays the price
        -prices * dt + b.degradation_cost * dt,        # discharging earns it, minus wear
        np.zeros(T + 1),
    ])
    lb = np.zeros(3 * T + 1)
    ub = np.concatenate([np.full(2 * T, b.power_mw), np.full(T + 1, b.energy_mwh)])
    s0 = b.soc_start_frac * b.energy_mwh
    lb[2 * T] = ub[2 * T] = s0          # fixed starting SOC
    lb[3 * T] = max(lb[3 * T], s0)      # end the day no emptier than it started
    return cost, a_eq, lb, ub


def _solve_milp(cost, a_eq, lb, ub, T: int, b: Battery):
    n = 3 * T + 1
    ident = sparse.identity(T, format="csr")
    zeros_t = sparse.csr_matrix((T, T))
    zeros_s = sparse.csr_matrix((T, T + 1))
    p = b.power_mw
    a_eq_full = sparse.hstack([a_eq, sparse.csr_matrix((T, T))], format="csr")
    # c_t <= P*y_t   and   d_t <= P*(1 - y_t)
    a_c = sparse.hstack([ident, zeros_t, zeros_s, -p * ident], format="csr")
    a_d = sparse.hstack([zeros_t, ident, zeros_s, p * ident], format="csr")
    constraints = [
        LinearConstraint(a_eq_full, np.zeros(T), np.zeros(T)),
        LinearConstraint(a_c, -np.inf, np.zeros(T)),
        LinearConstraint(a_d, -np.inf, np.full(T, p)),
    ]
    integrality = np.concatenate([np.zeros(n), np.ones(T)])
    bounds = Bounds(np.concatenate([lb, np.zeros(T)]), np.concatenate([ub, np.ones(T)]))
    res = milp(np.concatenate([cost, np.zeros(T)]), constraints=constraints,
               integrality=integrality, bounds=bounds)
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.message}")
    return res.x[:n]


def optimize_day(prices, battery: Battery = Battery(), dt_h: float = 0.25) -> Dispatch:
    """Best schedule for one day given a price vector (actual prices = perfect foresight,
    forecast prices = a realistic policy)."""
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("prices must be a non-empty 1-D array")
    if not np.all(np.isfinite(p)):
        raise ValueError("prices contain NaN/inf")
    T = p.size
    cost, a_eq, lb, ub = _build_lp(p, battery, dt_h)

    res = linprog(cost, A_eq=a_eq, b_eq=np.zeros(T), bounds=np.column_stack([lb, ub]), method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    x = res.x
    used_milp = False
    if np.max(np.minimum(x[:T], x[T:2 * T])) > OVERLAP_TOL * battery.power_mw:
        x = _solve_milp(cost, a_eq, lb, ub, T, battery)
        used_milp = True

    charge = np.clip(x[:T], 0.0, battery.power_mw)
    discharge = np.clip(x[T:2 * T], 0.0, battery.power_mw)
    soc = np.clip(x[2 * T:], 0.0, battery.energy_mwh)
    return Dispatch(charge, discharge, soc, settle(p, charge, discharge, battery, dt_h), used_milp)
