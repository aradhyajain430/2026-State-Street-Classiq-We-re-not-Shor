# CVaR Extension

This folder contains a classical Monte Carlo pipeline for **Conditional Value at Risk (CVaR)**.
CVaR (Expected Shortfall) measures the average loss in the tail beyond VaR.

## Run

From the repo root:

```powershell
uv run python cvar\run_cvar.py
```

Or run the full CVaR pipeline (classical + IQAE + combined panel):

```powershell
uv run python cvar\run_all.py
```

Outputs (by default) are written to `writeup/`:
- `cvar_convergence.png`
- `cvar_error_scaling.png`
- `cvar_error_scaling.csv`
- `cvar_summary.json`

## Quantum IQAE (CVaR)

These scripts reuse the IQAE workflow to estimate tail probability and map it to CVaR
for a Gaussian return model.

```powershell
uv run python cvar\iqae_cvar_epsilon_scaling.py
uv run python cvar\iqae_cvar_shots_scaling.py
```

Outputs:
- `writeup/cvar_iqae_eps_scaling.png`
- `writeup/cvar_iqae_shots_scaling.png`
- `cvar/cvar_iqae_eps_scaling.csv`
- `cvar/cvar_iqae_shots_scaling.csv`

## Combined panel

```powershell
uv run python cvar\combine_cvar_panels.py
```

Output:
- `writeup/cvar_multi_panel.png`

## Options

```powershell
uv run python cvar\run_cvar.py --confidence 0.99 --trials 300
```
