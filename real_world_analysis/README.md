# Real World Analysis

This folder contains a classical, data-driven VaR study using real securities.
We fit a double-Poisson jump model (separate positive and negative jump processes)
to daily log returns, then run Monte Carlo to estimate VaR and study sampling error.

Default tickers:
- Gold futures: `GC=F`
- S&P 500: `^GSPC`

## Quick start

From the repo root:

```powershell
.\.venv\Scripts\python.exe real_world_analysis\real_world_analysis.py
```

This writes outputs under `real_world_analysis/results/<ticker>/`:
- `fit_overlay.png` (empirical vs fitted distribution)
- `var_convergence.png`
- `error_scaling.png`
- `error_scaling.csv`
- `summary.json`

## Useful options

```powershell
.\.venv\Scripts\python.exe real_world_analysis\real_world_analysis.py `
  --tickers "GC=F,^GSPC" `
  --period 5y `
  --confidence 0.95 `
  --threshold-std 3.0 `
  --trials 200
```

## Quantum (IQAE) for double-Poisson

These scripts reuse the same IQAE workflow but replace the Gaussian state preparation
with a double-Poisson jump model fitted to real data. They require Classiq API access.

Run a single IQAE estimate:

```powershell
.\.venv\Scripts\python.exe real_world_analysis\iqae_real_world_run.py `
  --ticker GC=F `
  --alpha 0.05 `
  --num-qubits 7 `
  --num-shots 128
```

Epsilon scaling (queries vs 1/epsilon):

```powershell
.\.venv\Scripts\python.exe real_world_analysis\iqae_real_world_epsilon_scaling.py `
  --ticker GC=F `
  --num-qubits 7 `
  --num-shots 64
```

Shot-noise scaling:

```powershell
.\.venv\Scripts\python.exe real_world_analysis\iqae_real_world_shots_scaling.py `
  --ticker GC=F `
  --num-qubits 7
```

Notes:
- `alpha` is the tail probability (e.g., 0.05 for 95% VaR).
- The grid is built via Monte Carlo sampling of the fitted model.
- Results are saved under `real_world_analysis/results/` by default.

Notes:
- `threshold-std` controls how many standard deviations define a jump.
- The model uses two independent Poisson jump processes (up and down) to capture skew.
```
