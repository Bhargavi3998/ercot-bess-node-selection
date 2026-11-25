"""
Synthetic ERCOT-like 15-minute prices, in the same schema as data/rt_spp_all.csv.gz
(columns: ts, settlement_point, price).

FOR TESTING THE CODE ONLY. Results on this data say nothing about real ERCOT nodes.

What it mimics: a midday solar dip, an evening ramp, seasonal level, AR(1) noise, rare scarcity
spikes, and nodes that differ persistently in how much spread they offer (so a train/test
ranking has something real to find) plus occasional node-specific congestion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic(n_nodes: int = 12, start: str = "2025-01-01", end: str = "2025-12-31",
                   seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    n_days, slots = len(days), 96
    hour = np.arange(slots) * 0.25
    bump = lambda mu, sd: np.exp(-0.5 * ((hour - mu) / sd) ** 2)  # noqa: E731

    doy = days.dayofyear.values
    season = 1.0 + 0.5 * np.sin(2 * np.pi * (doy - 120) / 365)             # summer-peaking level
    day_scale = rng.lognormal(0.0, 0.35, n_days)                            # system tightness
    solar = rng.uniform(0.3, 1.0, n_days)                                   # cloudiness

    def ar1(shape, phi, sd):
        e = rng.standard_normal(shape) * sd * np.sqrt(1 - phi ** 2)
        x = np.zeros(shape)
        x[..., 0] = e[..., 0]
        for t in range(1, shape[-1]):
            x[..., t] = phi * x[..., t - 1] + e[..., t]
        return x

    shape_dev = (35.0 * bump(19.5, 2.0)[None, :] * day_scale[:, None]
                 - 18.0 * bump(13.5, 3.0)[None, :] * solar[:, None])
    hub = 22.0 * season[:, None] + shape_dev + ar1((n_days, slots), 0.9, 4.0)

    spike_day = rng.random(n_days) < 0.015
    for i in np.where(spike_day)[0]:
        s0 = int(rng.integers(60, 84))
        hub[i, s0:s0 + int(rng.integers(4, 9))] += rng.uniform(150, 900)

    amp = rng.lognormal(0.0, 0.25, n_nodes)           # persistent node quality (spread amplifier)
    frames = []
    for k in range(n_nodes):
        name = "HB_NORTH" if k == 0 else f"NODE_{k:02d}_RN"
        base = hub.mean(axis=1, keepdims=True)
        node = base + (hub - base) * (1.0 if k == 0 else amp[k]) + ar1((n_days, slots), 0.85, 2.5 if k else 0.0)
        if k:
            for i in np.where(rng.random(n_days) < 0.03)[0]:
                s0 = int(rng.integers(0, 80))
                node[i, s0:s0 + int(rng.integers(8, 20))] += rng.choice([-1, 1]) * rng.uniform(10, 60)
        node = np.clip(node, -250.0, 5000.0)
        ts = (days.values[:, None] + (np.arange(slots) * 15).astype("timedelta64[m]")[None, :]).ravel()
        frames.append(pd.DataFrame({"ts": ts, "settlement_point": name, "price": node.ravel()}))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = make_synthetic()
    print(df.groupby("settlement_point")["price"].describe().round(1))
