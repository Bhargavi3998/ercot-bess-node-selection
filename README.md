# ERCOT BESS Siting Pilot (RT-only, Public Data)

This repository is a public-data pilot for selecting battery interconnection locations in ERCOT under uncertainty. It implements an end-to-end pipeline using ERCOT Real-Time (RT) Settlement Point Prices (SPP) to score and select three diversified settlement points that exhibit strong battery-monetizable price behavior.

## What this project solves

### Feature engineering (implemented)
Using public RT SPP at the native settlement interval, the code engineers settlement-point features aligned to a 3-hour battery:

- 3-hour spread tail (P90/P95) via rolling 3-hour averages
- Cheap charging frequency: fraction of intervals with price < 0
- Spike exposure and persistence: frequency and average run-length above $500 and $1000
- Basis vs hub tail: P95/P99 of |node − hub|
- Basis stability: sign-flip rate and persistence of extreme basis episodes

Relevant artifact:
- `outputs/node_features.csv` contains the engineered features per settlement point.

### Objective / selection (implemented)
The pipeline implements a portfolio-aware selection objective:

1) Learn a node attractiveness score \(V_i\) from features  
2) Penalize downside tail risk using CVaR shortfall \(D_i\)  
3) Penalize choosing highly correlated sites (diversification)

Selection objective:

\[
\max_{|S|=3}\ \sum_{i\in S}(V_i - \alpha D_i)\ -\ \lambda\sum_{i<j}\rho_{ij}
\]

where:
- \(D_i = \mathrm{CVaR}_{95}(\max(0,\ H - R_{i,m}))\)
- \(\rho_{ij}\) is correlation between node basis time series

Relevant artifacts:
- `outputs/top3.json` contains the selected top-3 and parameters
- `outputs/corr_matrix.csv` contains correlations \(\rho_{ij}\)

### Modeling approach (implemented)
The mapping from engineered features to node attractiveness is implemented as a simple, explainable model:

- Standardize features (z-scores)
- Fit ridge or elastic-net regression to learn weights \(w_k\)
- Compute \(V_i = \sum_k w_k z_{i,k}\)

Relevant artifacts:
- `outputs/learned_weights.csv` contains learned weights \(w_k\)
- `outputs/node_features.csv` contains \(V_i\) and \(D_i\)

## Data source (public)
- ERCOT Public API: RT Settlement Point Prices (NP6-905)

The script reads credentials from `.env` (not committed).

## Repository layout
- `ercot_bess_siting_rt_pipeline.py` — single-file end-to-end pipeline
- `tools/make_settlement_points.py` — generates a candidate list (optionally RN-only plus hubs/zones)
- `settlement_points*.txt` — candidate universes for the pilot
- `outputs/` — generated artifacts (ignored by git)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env