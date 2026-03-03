# ERCOT BESS Siting Pilot (RT-only, Public Data)

This repository is a public-data pilot for selecting battery interconnection locations in ERCOT under uncertainty. It implements an end-to-end pipeline using ERCOT Real-Time (RT) Settlement Point Prices (SPP) to score and select three diversified settlement points that exhibit strong battery-monetizable price behavior.

## What this project solves

### Feature engineering (implemented)
Using public RT SPP at the native settlement interval, the code engineers settlement-point features aligned to a 3-hour battery:

- 3-hour spread tail (P90/P95) via rolling 3-hour averages
- Cheap charging frequency: fraction of intervals with price < 0
- Spike exposure and persistence: frequency and average run-length above $500 and $1000
- Basis vs hub tail: P95/P99 of absolute (node price minus hub price)
- Basis stability: sign-flip rate and persistence of extreme basis episodes

Relevant artifact:
- `outputs/node_features.csv` contains the engineered features per settlement point.

### Objective / selection (implemented)
The pipeline selects three settlement points as a small portfolio, not three independent “top scores”.

Selection objective (plain English):

Pick three settlement points that:
- have high opportunity (high V)
- have low downside tail risk (low D)
- are not redundant (low correlation between sites)

Portfolio score = sum over chosen sites of (V - alpha * D) minus lambda * sum of pairwise correlations.

Where:
- V is the node attractiveness score learned from features.
- D is a downside risk penalty based on CVaR of monthly shortfall versus a hurdle (how bad the worst months are).
- Correlation is computed from each node’s basis time series (node RT price minus hub RT price).
- alpha controls how much you penalize downside risk.
- lambda controls how much you penalize selecting highly correlated sites.

Relevant artifacts:
- `outputs/top3.json` contains the selected top-3 and parameters
- `outputs/corr_matrix.csv` contains the correlation matrix used for diversification

### Modeling approach (implemented)
The mapping from engineered features to node attractiveness is implemented as a simple, explainable model:

- Standardize features (z-scores) so different units don’t distort the model
- Fit ridge or elastic-net regression to learn feature weights
- Compute V as a weighted sum of standardized features

Relevant artifacts:
- `outputs/learned_weights.csv` contains learned feature weights
- `outputs/node_features.csv` contains V and D for each settlement point

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