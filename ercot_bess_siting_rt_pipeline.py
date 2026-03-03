#!/usr/bin/env python3
"""
ERCOT BESS Siting Pilot (RT-only) - single-file pipeline

What it does:
1) Pull RT Settlement Point Prices (SPP) from ERCOT Public API (NP6-905)
2) Compute battery-relevant features per settlement point:
   - 3-hour spread tail (P90/P95)
   - cheap charging frequency (price < 0)
   - spike exposure + persistence (freq + run-length > $500/$1000)
   - basis tail vs hub (|node - hub| P95/P99)
   - basis stability (sign flip rate + extreme-episode persistence)
3) Learn feature weights w_k via ridge/elastic-net to predict y_i = E[R_{i,m}]
4) Compute downside D_i = CVaR95(max(0, H - R_{i,m}))
5) Select diversified top-3 using:
      max_{|S|=3} sum_{i in S}(V_i - alpha*D_i) - lambda * sum_{i<j} rho_ij

Requirements:
  pip install pandas numpy requests python-dotenv scikit-learn tqdm

Credentials:
  ERCOT_USERNAME
  ERCOT_PASSWORD
  ERCOT_SUBSCRIPTION_KEY

Usage:
  python ercot_bess_siting_rt_pipeline.py --start 2026-01-01 --end 2026-01-15 --hub HB_NORTH --top_m 150
"""

from __future__ import annotations

import argparse
import itertools
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

TOKEN_URL = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
SCOPE = "openid fec253ea-0d06-4272-a5e6-b478baeecd70 offline_access"
CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"
RESPONSE_TYPE = "id_token"

BASE_URL = "https://api.ercot.com/api/public-reports"
RT_SPP_ENDPOINT = "/np6-905-cd/spp_node_zone_hub"


@dataclass
class Token:
    id_token: str
    expires_at: float


def get_id_token(username: str, password: str) -> Token:
    data = {
        "username": username,
        "password": password,
        "grant_type": "password",
        "scope": SCOPE,
        "client_id": CLIENT_ID,
        "response_type": RESPONSE_TYPE,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(TOKEN_URL, data=data, headers=headers, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    token = payload.get("id_token")
    expires_in = int(payload.get("expires_in", 3600))
    if not token:
        raise RuntimeError(f"Missing id_token in response keys={list(payload.keys())}")
    return Token(id_token=token, expires_at=time.time() + expires_in - 30)


class ErcotClient:
    def __init__(self, username: str, password: str, subscription_key: str):
        self.username = username
        self.password = password
        self.subscription_key = subscription_key
        self._token: Optional[Token] = None

    def _ensure_token(self) -> str:
        if self._token is None or time.time() >= self._token.expires_at:
            self._token = get_id_token(self.username, self.password)
        return self._token.id_token

    def get(self, path: str, params: Dict[str, Any], timeout: int = 180) -> requests.Response:
        token = self._ensure_token()
        headers = {
            "Ocp-Apim-Subscription-Key": self.subscription_key,
            "Authorization": f"Bearer {token}",
        }
        url = BASE_URL + path

        max_tries = 12
        backoff = 1.0

        for attempt in range(max_tries):
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)

            if resp.status_code == 200:
                return resp

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after else backoff
                print(f"[429] Rate limited. Sleeping {sleep_s:.1f}s (attempt {attempt+1}/{max_tries})")
                time.sleep(sleep_s)
                backoff = min(backoff * 2, 60)
                continue

            if resp.status_code in (401, 403) and attempt < max_tries - 1:
                self._token = None
                token = self._ensure_token()
                headers["Authorization"] = f"Bearer {token}"
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            resp.raise_for_status()

        raise RuntimeError("Failed after repeated 429 rate limiting. Reduce nodes or increase sleep.")

def client_from_env() -> ErcotClient:
    load_dotenv()
    u = os.getenv("ERCOT_USERNAME", "*******").strip()
    p = os.getenv("ERCOT_PASSWORD", "******").strip()
    k = os.getenv("ERCOT_SUBSCRIPTION_KEY", "abcdefghi123456").strip()
    if not (u and p and k):
        raise RuntimeError("Missing ERCOT_USERNAME / ERCOT_PASSWORD / ERCOT_SUBSCRIPTION_KEY in env/.env")
    return ErcotClient(username=u, password=p, subscription_key=k)


def response_to_df(resp: requests.Response) -> pd.DataFrame:
    """
    ERCOT public-reports often returns JSON with keys like:
      _meta, report, fields, data
    This parser handles:
      - data as list[dict]
      - data as list[list] + fields providing column names
    """
    j = resp.json()

    if isinstance(j, dict):
        field_names = None
        if "fields" in j and isinstance(j["fields"], list) and j["fields"]:
            field_names = [f.get("name") for f in j["fields"] if isinstance(f, dict) and f.get("name")]

        data = j.get("data", None)
        if isinstance(data, list) and len(data) > 0:
            # Case 1: list of dicts
            if isinstance(data[0], dict):
                return pd.DataFrame(data)

            # Case 2: list of lists
            if isinstance(data[0], list):
                if field_names and len(field_names) == len(data[0]):
                    return pd.DataFrame(data, columns=field_names)
                return pd.DataFrame(data)

        if "_embedded" in j and isinstance(j["_embedded"], dict):
            for _, v in j["_embedded"].items():
                if isinstance(v, list) and v:
                    if isinstance(v[0], dict):
                        return pd.DataFrame(v)
                    if isinstance(v[0], list):
                        if field_names and len(field_names) == len(v[0]):
                            return pd.DataFrame(v, columns=field_names)
                        return pd.DataFrame(v)

    raise RuntimeError("Could not parse ERCOT response into a table. Inspect resp.text.")

def fetch_rt_spp(api: ErcotClient, start_date: str, end_date: str, settlement_points: Optional[List[str]] = None) -> pd.DataFrame:
    base_params = {"deliveryDateFrom": start_date, "deliveryDateTo": end_date}
    frames: List[pd.DataFrame] = []
    if settlement_points:
        for sp in tqdm(settlement_points, desc="Fetching RT SPP"):
            params = dict(base_params)
            params["settlementPoint"] = sp
            resp = api.get(RT_SPP_ENDPOINT, params=params)
            frames.append(response_to_df(resp))
    else:
        resp = api.get(RT_SPP_ENDPOINT, params=base_params)
        frames.append(response_to_df(resp))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def guess_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = {str(c).lower(): c for c in df.columns}
    ts = cols.get("deliverydatetime") or cols.get("delivery_datetime") or cols.get("timestamp") or ""
    sp = cols.get("settlementpoint") or cols.get("settlement_point") or cols.get("settlement_point_name") or ""
    price = cols.get("settlementpointprice") or cols.get("price") or cols.get("spp") or ""
    interval = cols.get("deliveryinterval") or cols.get("interval") or ""
    hour = cols.get("deliveryhour") or cols.get("hour") or ""
    date = cols.get("deliverydate") or cols.get("delivery_date") or cols.get("date") or ""
    return {"ts": ts, "sp": sp, "price": price, "interval": interval, "hour": hour, "date": date}


def prep_prices(raw: pd.DataFrame) -> pd.DataFrame:
    m = guess_columns(raw)
    if not (m["sp"] and m["price"]):
        raise ValueError(f"Could not infer settlement point / price columns. Columns={list(raw.columns)[:30]}")
    df = raw.copy().rename(columns={m["sp"]: "settlement_point", m["price"]: "price"})
    if m["ts"]:
        df["ts"] = pd.to_datetime(df[m["ts"]])
    else:
        if not m["date"]:
            raise ValueError("No timestamp or delivery date columns found.")
        df["delivery_date"] = pd.to_datetime(df[m["date"]]).dt.date
        hour = df[m["hour"]] if m["hour"] else 0
        interval = df[m["interval"]] if m["interval"] else 1
        df["ts"] = pd.to_datetime(df["delivery_date"].astype(str)) + pd.to_timedelta(hour, unit="h") + pd.to_timedelta((interval - 1) * 15, unit="m")
    df = df[["ts", "settlement_point", "price"]].dropna()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.dropna().sort_values(["settlement_point", "ts"])


def compute_daily_3h_spread(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df["date"] = df["ts"].dt.date
    deltas = df.groupby("settlement_point")["ts"].diff().dropna()
    minutes = int(np.round(deltas.dt.total_seconds().median() / 60)) if not deltas.empty else 15
    window = max(1, int(round(180 / minutes)))
    df["p_roll3h"] = df.groupby("settlement_point")["price"].transform(lambda s: s.rolling(window=window, min_periods=window).mean())
    daily = df.groupby(["settlement_point", "date"])["p_roll3h"].agg(["min", "max"]).dropna()
    daily["daily_3h_spread"] = daily["max"] - daily["min"]
    return daily.reset_index()[["settlement_point", "date", "daily_3h_spread"]]


def run_length_stats(series: pd.Series, threshold: float) -> Tuple[float, float]:
    x = series.values
    above = x > threshold
    if above.size == 0:
        return 0.0, 0.0
    freq = float(above.mean())
    runs: List[int] = []
    run = 0
    for b in above:
        if b:
            run += 1
        elif run > 0:
            runs.append(run)
            run = 0
    if run > 0:
        runs.append(run)
    avg_run = float(np.mean(runs)) if runs else 0.0
    return freq, avg_run


def compute_features(prices: pd.DataFrame, hub: str):
    daily = compute_daily_3h_spread(prices)

    spread_stats = daily.groupby("settlement_point")["daily_3h_spread"].agg(
        spread_p90=lambda s: np.nanpercentile(s, 90),
        spread_p95=lambda s: np.nanpercentile(s, 95),
        spread_mean="mean",
    ).reset_index()

    cheap = prices.groupby("settlement_point")["price"].apply(lambda s: float((s < 0).mean())).reset_index(name="cheap_freq_lt0")

    spike_rows = []
    for sp, g in prices.groupby("settlement_point"):
        freq500, run500 = run_length_stats(g["price"], 500)
        freq1000, run1000 = run_length_stats(g["price"], 1000)
        spike_rows.append((sp, freq500, run500, freq1000, run1000))
    spikes = pd.DataFrame(spike_rows, columns=["settlement_point", "spike_freq_gt500", "spike_runlen_gt500", "spike_freq_gt1000", "spike_runlen_gt1000"])

    hub_df = prices[prices["settlement_point"] == hub][["ts", "price"]].rename(columns={"price": "hub_price"})
    if hub_df.empty:
        raise ValueError(f"Hub '{hub}' not found in pulled data. Include it in your pull.")
    merged = prices.merge(hub_df, on="ts", how="inner")
    merged["basis"] = merged["price"] - merged["hub_price"]

    basis_stats = merged.groupby("settlement_point")["basis"].agg(
        basis_abs_p95=lambda s: np.nanpercentile(np.abs(s), 95),
        basis_abs_p99=lambda s: np.nanpercentile(np.abs(s), 99),
    ).reset_index()

    stab_rows = []
    for sp, g in merged.groupby("settlement_point"):
        b = g["basis"].values
        eps = 1e-6
        sign = np.sign(b)
        sign[np.abs(b) < eps] = 0
        s = sign[sign != 0]
        flips = np.sum(s[1:] != s[:-1]) if s.size > 1 else 0
        flip_rate = float(flips / max(1, (s.size - 1)))

        thr = np.nanpercentile(np.abs(b), 90)
        extreme = np.abs(b) > thr
        runs: List[int] = []
        run = 0
        for e in extreme:
            if e:
                run += 1
            elif run > 0:
                runs.append(run)
                run = 0
        if run > 0:
            runs.append(run)
        avg_episode = float(np.mean(runs)) if runs else 0.0
        stab_rows.append((sp, flip_rate, avg_episode))
    stability = pd.DataFrame(stab_rows, columns=["settlement_point", "basis_sign_flip_rate", "basis_extreme_avg_runlen"])

    feats = (spread_stats
             .merge(cheap, on="settlement_point", how="left")
             .merge(spikes, on="settlement_point", how="left")
             .merge(basis_stats, on="settlement_point", how="left")
             .merge(stability, on="settlement_point", how="left"))
    return feats, daily, merged


def compute_monthly_proxy(daily_3h_spread: pd.DataFrame) -> pd.DataFrame:
    df = daily_3h_spread.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    return df.groupby(["settlement_point", "month"])["daily_3h_spread"].mean().reset_index(name="R_i_m")


def fit_regularized_linear(X: pd.DataFrame, y: pd.Series, model: str, reg_alpha: float, l1_ratio: float):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    if model == "ridge":
        reg = Ridge(alpha=reg_alpha)
    else:
        reg = ElasticNet(alpha=reg_alpha, l1_ratio=l1_ratio, max_iter=10000)
    reg.fit(Xs, y.values)
    w = pd.Series(reg.coef_, index=X.columns, name="weight")
    V = pd.Series(Xs @ w.values, index=X.index, name="V_i")
    return w, V


def cvar95_shortfall(monthly: pd.DataFrame, hurdle: float) -> pd.Series:
    df = monthly.copy()
    df["shortfall"] = (hurdle - df["R_i_m"]).clip(lower=0)
    out = {}
    for sp, g in df.groupby("settlement_point"):
        s = g["shortfall"].values
        if s.size == 0:
            out[sp] = np.nan
            continue
        q = np.quantile(s, 0.95)
        tail = s[s >= q]
        out[sp] = float(tail.mean()) if tail.size else 0.0
    return pd.Series(out, name="D_i")


def corr_matrix_from_basis(basis_long: pd.DataFrame) -> pd.DataFrame:
    pivot = basis_long.pivot_table(index="ts", columns="settlement_point", values="basis")
    return pivot.corr()


def select_top3(node_df: pd.DataFrame, corr: pd.DataFrame, alpha: float, lam: float, top_m: int) -> Dict[str, Any]:
    df = node_df.copy()
    df["base"] = df["V_i"] - alpha * df["D_i"]
    cand = df.sort_values("base", ascending=False).head(top_m)
    if len(cand) < 3:
        raise ValueError(f"Need at least 3 candidate nodes after filtering; got {len(cand)}. "
                     f"Try a shorter date range or pass --settlement-points-file.")
    sps = list(cand.index)
    corr_sub = corr.loc[sps, sps].fillna(0.0)

    best_score = -1e18
    best = None
    for i, j, k in itertools.combinations(sps, 3):
        score = (cand.loc[i, "base"] + cand.loc[j, "base"] + cand.loc[k, "base"]) - lam * (
            corr_sub.loc[i, j] + corr_sub.loc[i, k] + corr_sub.loc[j, k]
        )
        if score > best_score:
            best_score = score
            best = (i, j, k)
    return {"top3": list(best), "objective": float(best_score)}


def load_settlement_points(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--hub", default="HB_NORTH")
    ap.add_argument("--settlement-points-file", default=None)
    ap.add_argument("--model", choices=["ridge", "elasticnet"], default="ridge")
    ap.add_argument("--reg-alpha", type=float, default=1.0)
    ap.add_argument("--l1-ratio", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.5)
    ap.add_argument("--top-m", type=int, default=150)
    ap.add_argument("--hurdle", type=float, default=None)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    api = client_from_env()

    sps = load_settlement_points(args.settlement_points_file) if args.settlement_points_file else None
    raw = fetch_rt_spp(api, args.start, args.end, settlement_points=sps)
    raw.to_csv(os.path.join(args.outdir, "raw_rt_spp.csv"), index=False)

    prices = prep_prices(raw)

    # Ensure hub present; if not, fetch hub and append
    if args.hub not in set(prices["settlement_point"].unique()):
        hub_raw = fetch_rt_spp(api, args.start, args.end, settlement_points=[args.hub])
        hub_prices = prep_prices(hub_raw)
        prices = pd.concat([prices, hub_prices], ignore_index=True).sort_values(["settlement_point", "ts"])

    feats, daily, merged = compute_features(prices, hub=args.hub)
    feats = feats.set_index("settlement_point")

    monthly = compute_monthly_proxy(daily)
    y = monthly.groupby("settlement_point")["R_i_m"].mean()

    common = feats.index.intersection(y.index)
    X = feats.loc[common]
    y = y.loc[common]

    w, V = fit_regularized_linear(X, y, model=args.model, reg_alpha=args.reg_alpha, l1_ratio=args.l1_ratio)

    hurdle = float(monthly["R_i_m"].quantile(0.25)) if args.hurdle is None else float(args.hurdle)
    D = cvar95_shortfall(monthly[monthly["settlement_point"].isin(common)], hurdle=hurdle)

    corr = corr_matrix_from_basis(merged[merged["settlement_point"].isin(common)][["ts", "settlement_point", "basis"]])
    corr.to_csv(os.path.join(args.outdir, "corr_matrix.csv"))

    node_out = pd.DataFrame({"V_i": V, "D_i": D}).dropna()
    node_out = node_out.join(feats, how="left")
    node_out.to_csv(os.path.join(args.outdir, "node_features.csv"))
    w.to_csv(os.path.join(args.outdir, "learned_weights.csv"))

    result = select_top3(node_out[["V_i", "D_i"]], corr, alpha=args.alpha, lam=args.lam, top_m=args.top_m)
    result.update({"hub": args.hub, "hurdle": hurdle, "alpha": args.alpha, "lambda": args.lam, "top_m": args.top_m, "model": args.model, "reg_alpha": args.reg_alpha})
    import json
    with open(os.path.join(args.outdir, "top3.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Done.")
    print(f"Top 3: {result['top3']}")
    print(f"Objective: {result['objective']}")
    print(f"Outputs in: {args.outdir}")


if __name__ == "__main__":
    main()
