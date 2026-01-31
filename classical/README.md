# Classical VaR (Monte Carlo)

This folder contains a classical Monte Carlo implementation for Value at Risk (VaR) using a Gaussian return model.

## What it does
- Defines a Gaussian return distribution
- Computes the **theoretical VaR** in closed form
- Estimates VaR via **Monte Carlo sampling**
- Measures convergence and plots **error vs sample size** to show O(1/sqrt(N)) scaling

## Quick start
From the repository root:

```powershell
python classical\run_classical.py
```

Outputs are written to `classical\results`:
- `error_scaling.csv`
- `summary.json`
- `var_convergence.png`
- `error_scaling.png`

## Common options
```powershell
python classical\run_classical.py --confidence 0.99 --mu 0.0003 --sigma 0.015 --portfolio 1.0
```

```powershell
python classical\run_classical.py --sample-sizes 100,200,500,1000,5000 --trials 300
```

## Notes
- VaR is computed as the confidence-quantile of **loss**: `L = -P&L`.
- For Gaussian returns `r ~ N(mu, sigma)`, the theoretical VaR is:
  `VaR = -(mu + sigma * z_{1-c}) * V` where `c` is confidence, `V` is portfolio value.
